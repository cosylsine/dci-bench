"""Online code retrieval: Pi solver, then host-only scoring for each sample."""

from __future__ import annotations

from contextlib import AsyncExitStack
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import anyio
from inspect_ai import Task
from inspect_ai.agent import sandbox_agent_bridge
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Metric, SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import Generate, TaskState, solver

from dci_bench.agents.code_agent import run_pi_code
from dci_bench.data.code_online import CodeSample
from dci_bench.protocol.code import DEFAULT_CODE_MAX_FILES, DEFAULT_CODE_METRIC_KS, DEFAULT_CODE_TOKEN_BUDGETS
from dci_bench.protocol.code_runtime import code_runtime_contract
from dci_bench.scoring.code import load_json, score_query, snapshot_identity
from dci_bench.tasks.mteb_llm_retrieval import (
    BridgePortPool, _remaining_pi_timeout_seconds, _serialized_bridge_start,
)


def write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@solver
def code_pi_solver(samples: Sequence[CodeSample], output_dir: Path, pool: BridgePortPool,
                   context_window: int, max_files: int, mock_script: Path | None = None) -> Any:
    by_id = {sample.query_id: sample for sample in samples}

    async def solve(state: TaskState, _generate: Generate) -> TaskState:
        sample = by_id[str(state.sample_id)]
        sample_dir = output_dir / sample.directory_name
        identity = snapshot_identity(sample.reference)
        async with pool.lease() as port, AsyncExitStack() as stack:
            if mock_script is None:
                context = sandbox_agent_bridge(
                    model="inspect", sandbox="local", port=port, web_search=False,
                    code_execution=False, client_mcp_servers=False, forward_generation_config=False,
                )
                await stack.enter_async_context(_serialized_bridge_start(pool, context))
            # Inspect limits are tied to the async sample context, not a worker thread.
            timeout_seconds = _remaining_pi_timeout_seconds()
            result = await anyio.to_thread.run_sync(lambda: run_pi_code(
                query=state.input_text, workspace=sample.workspace,
                files=sample.reference["files"], max_files=max_files,
                output_path=sample_dir / "final.json", trace_path=sample_dir / "trace.json",
                mock_script=mock_script, openai_base_url=f"http://localhost:{port}/v1",
                model="inspect", context_window=context_window,
                task_instruction=f"Retrieval task type: {sample.reference['task']}",
                timeout_seconds=timeout_seconds,
            ))
        if result.valid_output and result.failure is None and result.code_output is not None:
            prediction = {**identity, "output": result.code_output}
        else:
            prediction = {**identity, "failure_reason": result.failure_reason or result.failure_kind or "invalid Pi output"}
        state.output.completion = json.dumps(prediction, ensure_ascii=False)
        return state

    return solve


@metric
def code_eligible_mean() -> Metric:
    """Inspect preview uses each metric's eligible samples, matching score_run.

    A generic wildcard mean either fails on missing dictionary keys or counts
    unavailable annotations as zero. Entirely unavailable metrics are omitted
    from Inspect's numeric preview; code-result.json retains explicit nulls.
    """
    def aggregate(scores: list[SampleScore]) -> dict[str, float]:
        totals, counts = Counter(), Counter()
        for sample_score in scores:
            values = sample_score.score.value
            if not isinstance(values, dict):
                continue  # Inspect can supply an unscored sentinel after an error.
            for key, value in values.items():
                if value is not None:
                    totals[key] += float(value)
                    counts[key] += 1
        return {key: totals[key] / count for key, count in counts.items()}

    return aggregate


@scorer(metrics=[code_eligible_mean()])
def code_retrieval_scorer(samples: Sequence[CodeSample], output_dir: Path,
                          metric_ks: Sequence[int], max_files: int,
                          token_budgets: Sequence[int]) -> Any:
    by_id = {sample.query_id: sample for sample in samples}

    async def score(state: TaskState, _target: Target) -> Score:
        sample = by_id[str(state.sample_id)]
        # Labels remain in this host closure, never Sample.target or metadata.
        # Inspect exposes terminal errors/limits on EvalSample after scoring,
        # not on TaskState. The publisher reconciles them with these raw scores.
        try:
            prediction = load_json(state.output.completion)
        except (TypeError, ValueError):
            prediction = None
        result = score_query(prediction, sample.reference, metric_ks=metric_ks,
                             max_files=max_files, token_budgets=token_budgets)
        sample_dir = output_dir / sample.directory_name
        write_new_json(sample_dir / "scorer-prediction.json", prediction)
        write_new_json(sample_dir / "scorer-score.json", result)
        # Inspect's default epoch reducer converts None into zero, so keep
        # unavailable values only in metadata and the canonical score artifact.
        return Score(value={key: value for key, value in result["metrics"].items() if value is not None},
                     answer=state.output.completion, metadata=result)

    return score


def code_retrieval(
    samples: Sequence[CodeSample], *, output_dir: str | Path,
    metric_ks: Sequence[int] = DEFAULT_CODE_METRIC_KS,
    max_files: int = DEFAULT_CODE_MAX_FILES,
    token_budgets: Sequence[int] = DEFAULT_CODE_TOKEN_BUDGETS,
    bridge_port: int = 13131, max_concurrency: int = 1,
    model_context_window: int = 65536, mock_script: Path | None = None,
) -> Task:
    if not samples or len({sample.query_id for sample in samples}) != len(samples):
        raise ValueError("code task requires nonempty unique selected samples")
    contract = code_runtime_contract(max_files)
    # Validate scoring configuration before constructing a runnable task.
    for sample in samples:
        score_query(None, sample.reference, metric_ks=metric_ks, max_files=max_files, token_budgets=token_budgets)
    if type(model_context_window) is not int or model_context_window <= contract["max_model_output_tokens"] + 4096:
        raise ValueError("context window is too small for output and safety reserve")
    pool = BridgePortPool(bridge_port, max_concurrency)
    dataset = MemoryDataset([
        Sample(id=sample.query_id, input=sample.query, metadata={
            "evaluation_track": "code", "task": sample.reference["task"],
            "query_id": sample.query_id, "workspace": str(sample.workspace),
        }) for sample in samples
    ], name="code-retrieval")
    return Task(
        name="code_retrieval", dataset=dataset,
        solver=code_pi_solver(samples, Path(output_dir), pool, model_context_window, max_files, mock_script),
        scorer=code_retrieval_scorer(samples, Path(output_dir), metric_ks, max_files, token_budgets),
        sandbox="local", time_limit=contract["sample_timeout_seconds"],
        token_limit=contract["max_total_model_tokens"],
        metadata={"evaluation_track": "code", "runtime_contract": contract,
                  "metric_ks": list(metric_ks), "max_files": max_files, "token_budgets": list(token_budgets)},
    )
