"""Phase 3 single-sample Inspect task for DCI-Bench."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from inspect_ai import Task, task
from inspect_ai.agent import sandbox_agent_bridge
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver

from dci_bench.agents.pi_agent import run_pi_dci
from dci_bench.data.registry import TASKS
from dci_bench.protocol.contracts import MAX_AGENT_STEPS, SAMPLE_TIMEOUT_SECONDS
from dci_bench.scoring.retrieval import normalize_metric_ks, score_query, zero_metrics


RETRIEVAL_AGGREGATION_METRICS = {
    "recall_at_*": [mean()],
    "f1_at_*": [mean()],
    "ndcg_at_*": [mean()],
}


def _load_queries(task_name: str, metadata_root: Path) -> list[dict[str, Any]]:
    task_dir = metadata_root / task_name
    return [
        json.loads(line)
        for line in (task_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _query_ids(queries: Sequence[dict[str, Any]], *, task_name: str) -> list[str]:
    """Return query IDs after validating the host-side query manifest.

    ``queries.jsonl`` is part of the host-only metadata boundary.  Rejecting
    malformed or duplicate IDs here is important because silently replacing a
    duplicate in a dictionary would make an otherwise reproducible selection
    ambiguous.
    """

    ids: list[str] = []
    seen: set[str] = set()
    for index, query in enumerate(queries):
        query_id = query.get("_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(
                f"Invalid query _id at {task_name}/queries.jsonl line {index + 1}"
            )
        if query_id in seen:
            raise ValueError(f"Duplicate query id {task_name}/{query_id}")
        if not isinstance(query.get("text"), str):
            raise ValueError(f"Query {task_name}/{query_id} has no text")
        seen.add(query_id)
        ids.append(query_id)
    return ids


def _load_query_qrels(task_name: str, query_id: str, metadata_root: Path) -> dict[str, float]:
    qrels_path = metadata_root / task_name / "qrels.json"
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    query_qrels = qrels.get(query_id)
    if not isinstance(query_qrels, dict) or not query_qrels:
        raise ValueError(f"Missing qrels for {task_name}/{query_id} in {qrels_path}")
    return {str(doc_id): float(score) for doc_id, score in query_qrels.items()}


def _select_query_ids(
    task_name: str,
    *,
    metadata_root: Path,
    query_ids: Sequence[str] | None = None,
    all_queries: bool = False,
) -> list[str]:
    """Validate and resolve the explicit query selection.

    The Phase 4 entry point intentionally has no implicit ``first query``
    behavior.  ``all_queries`` follows the physical order in
    ``queries.jsonl``; explicit IDs preserve the caller's order.
    """

    if isinstance(query_ids, str):
        query_ids = [query_ids]
    if all_queries and query_ids is not None:
        raise ValueError("query_ids and all_queries are mutually exclusive")
    if not all_queries and query_ids is None:
        raise ValueError("one of query_ids or all_queries must be provided")

    queries = _load_queries(task_name, metadata_root)
    ordered_ids = _query_ids(queries, task_name=task_name)
    available = set(ordered_ids)
    if all_queries:
        selected = ordered_ids
        if not selected:
            raise ValueError(f"No queries found for task {task_name}")
    else:
        # Make a concrete list so generators cannot be consumed twice while
        # constructing a MemoryDataset and its metadata.
        selected = list(query_ids or [])
        if not selected:
            raise ValueError("query_ids must contain at least one query id")
        seen: set[str] = set()
        duplicates: list[str] = []
        for query_id in selected:
            if query_id in seen:
                duplicates.append(query_id)
            seen.add(query_id)
        if duplicates:
            raise ValueError(f"Duplicate query id(s): {', '.join(duplicates)}")
        unknown = [query_id for query_id in selected if query_id not in available]
        if unknown:
            raise ValueError(f"Unknown query id(s) for {task_name}: {', '.join(unknown)}")

    # qrels are deliberately read only on the host.  Do this validation for
    # every selected sample before an Inspect run can start.
    for query_id in selected:
        _load_query_qrels(task_name, query_id, metadata_root)
    return selected


def _make_samples(
    task_name: str,
    query_ids: Sequence[str] | None,
    metadata_root: Path,
    workspace_root: Path,
    *,
    all_queries: bool = False,
    bridge_port_pool: "BridgePortPool | None" = None,
) -> list[Sample]:
    """Build one host-validated Inspect sample for every selected query."""

    if isinstance(query_ids, str):
        query_ids = [query_ids]
    queries = _load_queries(task_name, metadata_root)
    selected_ids = _select_query_ids(
        task_name,
        metadata_root=metadata_root,
        query_ids=query_ids,
        all_queries=all_queries,
    )
    by_id = {query["_id"]: query for query in queries}
    samples: list[Sample] = []
    for sample_index, selected_id in enumerate(selected_ids):
        query = by_id[selected_id]
        metadata: dict[str, Any] = {
            "task": task_name,
            "query_id": selected_id,
            "query_index": sample_index,
            "workspace": str(workspace_root / task_name),
        }
        samples.append(
            Sample(
                id=selected_id,
                input=query["text"],
                metadata=metadata,
                sandbox="local",
            )
        )
    return samples


def _make_sample(task_name: str, query_id: str | None, metadata_root: Path, workspace_root: Path) -> Sample:
    """Backward-compatible single-sample constructor.

    Phase 3 callers historically omitted ``query_id`` and received the first
    query.  Keep that behavior only for this compatibility helper; Phase 4's
    public batch task requires explicit IDs or ``all_queries=True``.
    """

    queries = _load_queries(task_name, metadata_root)
    if query_id is None:
        if not queries:
            raise ValueError(f"No queries found for task {task_name}")
        query_id = str(queries[0]["_id"])
    samples = _make_samples(
        task_name,
        [query_id],
        metadata_root,
        workspace_root,
    )
    return samples[0]


@dataclass
class BridgePortPool:
    """Async lease/release pool for per-sample Inspect bridge ports.

    A static ``sample_index % concurrency`` mapping is unsafe: a later sample
    can start while an earlier sample is still serving requests.  Leases are
    therefore acquired immediately before opening the bridge and released in a
    ``finally`` block after the Pi runner exits.
    """

    base_port: int = 13131
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.base_port, bool) or not isinstance(self.base_port, int):
            raise ValueError("bridge base_port must be an integer")
        if isinstance(self.max_concurrency, bool) or not isinstance(self.max_concurrency, int):
            raise ValueError("max_concurrency must be an integer")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if self.base_port < 1 or self.base_port + self.max_concurrency - 1 > 65535:
            raise ValueError("bridge port pool must fit in TCP port range 1..65535")
        self._semaphore = anyio.Semaphore(self.max_concurrency)
        self._lock = anyio.Lock()
        self._in_use: set[int] = set()

    @property
    def ports(self) -> tuple[int, ...]:
        return tuple(self.base_port + offset for offset in range(self.max_concurrency))

    @asynccontextmanager
    async def lease(self):
        await self._semaphore.acquire()
        try:
            async with self._lock:
                free_ports = [port for port in self.ports if port not in self._in_use]
                # The semaphore and set are updated under the same lock.  This
                # should be unreachable unless a caller corrupts pool state;
                # fail loudly rather than assigning a duplicate bridge port.
                if not free_ports:
                    raise RuntimeError("bridge port pool exhausted while holding a lease")
                port = free_ports[0]
                self._in_use.add(port)
            yield port
        finally:
            if "port" in locals():
                async with self._lock:
                    self._in_use.remove(port)
            self._semaphore.release()


@asynccontextmanager
async def _bridge_port_lease(pool: BridgePortPool | None, fallback_port: int):
    if pool is None:
        yield fallback_port
    else:
        async with pool.lease() as port:
            yield port


@solver
def pi_dci_solver(
    output_dir: str = "results/phase3",
    bridge_port: int = 13131,
    include_task_dir: bool = True,
    bridge_port_pool: BridgePortPool | None = None,
) -> Any:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        task_name = str(state.metadata["task"])
        query_id = str(state.metadata["query_id"])
        sample_dir = Path(output_dir)
        if include_task_dir:
            sample_dir /= task_name
        sample_dir /= query_id
        output_path = sample_dir / "final.json"
        trace_path = sample_dir / "trace.json"
        async with _bridge_port_lease(bridge_port_pool, bridge_port) as sample_bridge_port:
            async with sandbox_agent_bridge(
                model="inspect",
                sandbox="local",
                port=sample_bridge_port,
                web_search=False,
                code_execution=False,
                client_mcp_servers=False,
                forward_generation_config=False,
            ):
                result = await anyio.to_thread.run_sync(
                    lambda: run_pi_dci(
                        query=state.input_text,
                        workspace=Path(str(state.metadata["workspace"])),
                        output_path=output_path,
                        trace_path=trace_path,
                        openai_base_url=f"http://localhost:{sample_bridge_port}/v1",
                        openai_api_key="inspect",
                        model="inspect",
                        timeout_seconds=SAMPLE_TIMEOUT_SECONDS,
                        max_agent_steps=MAX_AGENT_STEPS,
                    )
                )
        state.output.completion = json.dumps(
            {
                "ranked_doc_ids": result.ranked_doc_ids,
                "valid_output": result.valid_output,
                "failure_reason": result.failure_reason,
                "failure": result.failure,
                "failure_kind": result.failure_kind,
                "contract_version": result.contract_version,
                "usage_summary": result.usage_summary,
                "agent_steps": result.agent_steps,
                "tool_calls": result.tool_calls,
                "final_output_adapter": result.final_output_adapter,
                "repair_attempts": result.repair_attempts,
                "repair_reasons": result.repair_reasons,
                "body_evidence": result.body_evidence,
                "inspect_sandbox": result.inspect_sandbox,
                "tool_sandbox": result.tool_sandbox,
                "output_path": str(result.output_path),
                "trace_path": str(result.trace_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return state

    return solve


@scorer(metrics=RETRIEVAL_AGGREGATION_METRICS)
def dci_retrieval_scorer(
    task_name: str,
    metadata_root: str = "data/metadata",
    metric_ks: Sequence[int] | None = None,
) -> Any:
    ks = normalize_metric_ks(metric_ks)
    host_metadata_root = Path(metadata_root)

    async def score(state: TaskState, _target: Target) -> Score:
        try:
            output = json.loads(state.output.completion)
            query_id = str(state.metadata["query_id"])
            qrels = _load_query_qrels(task_name, query_id, host_metadata_root)
            metrics = score_query(
                {"ranked_doc_ids": output.get("ranked_doc_ids", [])},
                qrels,
                metric_ks=ks,
            )
            value = {
                key: metric_value
                for key, metric_value in metrics.items()
                if key.startswith(("recall_at_", "f1_at_", "ndcg_at_"))
            }
            answer = json.dumps({**output, **metrics}, ensure_ascii=False, sort_keys=True)
            return Score(value=value, answer=answer, metadata=metrics)
        except Exception as exc:
            metrics = {
                "valid_output": False,
                "failure_reason": str(exc),
                **zero_metrics(ks),
            }
            return Score(
                value=zero_metrics(ks),
                answer=state.output.completion,
                metadata=metrics,
            )

    return score


@task
def mteb_llm_retrieval_single(
    task_name: str = "LLMPublicHealthQA",
    query_id: str | None = "Q25",
    workspace_root: str = "data/workspaces",
    metadata_root: str = "data/metadata",
    output_dir: str = "results/phase3",
    metric_ks: Sequence[int] | None = None,
) -> Task:
    ks = normalize_metric_ks(metric_ks)
    if task_name not in TASKS:
        raise ValueError(f"Unknown task {task_name!r}. Available: {sorted(TASKS)}")
    sample = _make_sample(task_name, query_id, Path(metadata_root), Path(workspace_root))
    return Task(
        dataset=MemoryDataset([sample], name=f"{task_name}-{sample.id}"),
        solver=pi_dci_solver(output_dir=output_dir, include_task_dir=True),
        scorer=dci_retrieval_scorer(
            task_name=task_name,
            metadata_root=metadata_root,
            metric_ks=ks,
        ),
        sandbox="local",
        time_limit=SAMPLE_TIMEOUT_SECONDS,
        metadata={
            "inspect_sandbox": "local",
            "tool_sandbox": "bubblewrap",
            "task_name": task_name,
            "metric_ks": list(ks),
        },
    )


@task
def mteb_llm_retrieval(
    task_name: str = "LLMPublicHealthQA",
    query_ids: Sequence[str] | None = None,
    all_queries: bool = False,
    workspace_root: str = "data/workspaces",
    metadata_root: str = "data/metadata",
    output_dir: str = "results/phase4",
    metric_ks: Sequence[int] | None = None,
    bridge_port: int = 13131,
    max_concurrency: int = 1,
) -> Task:
    """Build one Inspect task containing the complete selected query batch.

    Unlike :func:`mteb_llm_retrieval_single`, this Phase 4 constructor has no
    default query.  Callers must provide explicit ``query_ids`` or set
    ``all_queries=True``.  Selection and qrels validation happen before the
    returned Task can be passed to ``inspect_ai.eval``.
    """

    if task_name not in TASKS:
        raise ValueError(f"Unknown task {task_name!r}. Available: {sorted(TASKS)}")
    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
        raise ValueError("max_concurrency must be an integer")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    pool = BridgePortPool(base_port=bridge_port, max_concurrency=max_concurrency)
    selected_ids = _select_query_ids(
        task_name,
        metadata_root=Path(metadata_root),
        query_ids=query_ids,
        all_queries=all_queries,
    )
    samples = _make_samples(
        task_name,
        query_ids,
        Path(metadata_root),
        Path(workspace_root),
        all_queries=all_queries,
        bridge_port_pool=pool,
    )
    ks = normalize_metric_ks(metric_ks)
    return Task(
        dataset=MemoryDataset(samples, name=f"{task_name}-batch"),
        solver=pi_dci_solver(
            output_dir=output_dir,
            bridge_port=bridge_port,
            include_task_dir=False,
            bridge_port_pool=pool,
        ),
        scorer=dci_retrieval_scorer(
            task_name=task_name,
            metadata_root=metadata_root,
            metric_ks=ks,
        ),
        sandbox="local",
        time_limit=SAMPLE_TIMEOUT_SECONDS,
        metadata={
            "inspect_sandbox": "local",
            "tool_sandbox": "bubblewrap",
            "task_name": task_name,
            "query_ids": selected_ids,
            "all_queries": all_queries,
            "metric_ks": list(ks),
            "max_concurrency": max_concurrency,
            "bridge_ports": list(pool.ports),
        },
    )


# Explicit alias for callers that prefer the name used in the Phase 4 plan.
mteb_llm_retrieval_batch = mteb_llm_retrieval
