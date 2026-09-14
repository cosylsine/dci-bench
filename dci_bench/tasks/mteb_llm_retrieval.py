"""Phase 3 single-sample Inspect task for DCI-Bench."""

from __future__ import annotations

import json
from collections.abc import Sequence
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


def _load_query_qrels(task_name: str, query_id: str, metadata_root: Path) -> dict[str, float]:
    qrels_path = metadata_root / task_name / "qrels.json"
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    query_qrels = qrels.get(query_id)
    if not isinstance(query_qrels, dict) or not query_qrels:
        raise ValueError(f"Missing qrels for {task_name}/{query_id} in {qrels_path}")
    return {str(doc_id): float(score) for doc_id, score in query_qrels.items()}


def _make_sample(task_name: str, query_id: str | None, metadata_root: Path, workspace_root: Path) -> Sample:
    queries = _load_queries(task_name, metadata_root)
    by_id = {query["_id"]: query for query in queries}
    selected_id = query_id or queries[0]["_id"]
    query = by_id[selected_id]
    _load_query_qrels(task_name, selected_id, metadata_root)
    return Sample(
        id=selected_id,
        input=query["text"],
        metadata={
            "task": task_name,
            "query_id": selected_id,
            "workspace": str(workspace_root / task_name),
        },
        sandbox="local",
    )


@solver
def pi_dci_solver(
    output_dir: str = "results/phase3",
    bridge_port: int = 13131,
) -> Any:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        task_name = str(state.metadata["task"])
        query_id = str(state.metadata["query_id"])
        sample_dir = Path(output_dir) / task_name / query_id
        output_path = sample_dir / "final.json"
        trace_path = sample_dir / "trace.json"
        async with sandbox_agent_bridge(
            model="inspect",
            sandbox="local",
            port=bridge_port,
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
                    openai_base_url=f"http://localhost:{bridge_port}/v1",
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
        solver=pi_dci_solver(output_dir=output_dir),
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
