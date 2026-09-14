#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dci_bench.scoring.retrieval import (
    DEFAULT_METRIC_KS,
    normalize_metric_ks,
    score_query,
    zero_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="LLMPublicHealthQA")
    parser.add_argument("--query-id", default="Q25")
    parser.add_argument("--model-path", default="/mnt/afs/share/Qwen3-4B")
    parser.add_argument("--served-model-name", default="Qwen3-4B")
    parser.add_argument(
        "--openai-base-url",
        "--vllm-base-url",
        dest="openai_base_url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible /v1 endpoint. --vllm-base-url remains a compatibility alias.",
    )
    parser.add_argument(
        "--openai-service",
        default="vllm",
        help="Inspect service prefix used in openai-api/<service>/<model>.",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--log-dir", default="results/inspect-logs")
    parser.add_argument("--output-dir", default="results/phase3")
    parser.add_argument("--metadata-root", default="data/metadata")
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument(
        "--metric-ks",
        nargs="+",
        type=int,
        default=list(DEFAULT_METRIC_KS),
        metavar="K",
        help="Retrieval metric cutoffs (default: 1 3 5 10 20).",
    )
    parser.add_argument("--require-valid-output", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    return parser.parse_args()


def _load_qrels(task_name: str, query_id: str, metadata_root: Path) -> dict[str, float]:
    qrels_path = metadata_root / task_name / "qrels.json"
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    query_qrels = qrels.get(query_id)
    if not isinstance(query_qrels, dict) or not query_qrels:
        raise ValueError(f"Missing qrels for {task_name}/{query_id} in {qrels_path}")
    return {str(doc_id): float(score) for doc_id, score in query_qrels.items()}


def build_sample_summary(
    *,
    task_name: str,
    query_id: str,
    output_dir: Path,
    metadata_root: Path,
    model_path: str,
    served_model_name: str,
    openai_base_url: str,
    openai_service: str,
    inspect_log_paths: list[str],
    metric_ks: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    ks = normalize_metric_ks(metric_ks)
    sample_dir = output_dir / task_name / query_id
    final_path = sample_dir / "final.json"
    trace_path = sample_dir / "trace.json"
    summary: dict[str, Any] = {
        "task": task_name,
        "query_id": query_id,
        "model_path": model_path,
        "served_model_name": served_model_name,
        "openai_base_url": openai_base_url,
        "openai_service": openai_service,
        "inspect_sandbox": "local",
        "tool_sandbox": "bubblewrap",
        "metric_ks": list(ks),
        "final_path": str(final_path),
        "trace_path": str(trace_path),
        "inspect_log_paths": inspect_log_paths,
    }
    if not final_path.is_file():
        summary.update(
            {
                "valid_output": False,
                "failure_reason": "Phase 3 did not produce final.json",
                "agent_steps": None,
                "tool_calls": None,
                **zero_metrics(ks),
            }
        )
        return summary

    try:
        final_payload = json.loads(final_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        summary.update(
            {
                "valid_output": False,
                "failure_reason": f"final.json is not valid JSON: {exc}",
                "agent_steps": None,
                "tool_calls": None,
                **zero_metrics(ks),
            }
        )
        return summary
    if not isinstance(final_payload, dict):
        summary.update(
            {
                "valid_output": False,
                "failure_reason": "final.json must contain a JSON object",
                "agent_steps": None,
                "tool_calls": None,
                **zero_metrics(ks),
            }
        )
        return summary

    try:
        metrics = score_query(
            {"ranked_doc_ids": final_payload.get("ranked_doc_ids", [])},
            _load_qrels(task_name, query_id, metadata_root),
            metric_ks=ks,
        )
    except Exception as exc:
        metrics = {
            "valid_output": False,
            "failure_reason": f"unable to score final output: {exc}",
            **zero_metrics(ks),
        }

    pi_valid = bool(final_payload.get("valid_output", False))
    sandbox_valid = (
        final_payload.get("inspect_sandbox") == "local"
        and final_payload.get("tool_sandbox") == "bubblewrap"
    )
    valid_output = pi_valid and sandbox_valid and bool(metrics["valid_output"])
    sandbox_failure = None
    if not sandbox_valid:
        sandbox_failure = "final.json does not attest the required local/bubblewrap sandbox boundary"
    failure_reason = (
        final_payload.get("failure_reason")
        or sandbox_failure
        or metrics["failure_reason"]
    )
    if not valid_output and not failure_reason:
        failure_reason = "Pi runner marked the final output invalid"
    summary.update(
        {
            "valid_output": valid_output,
            "failure_reason": failure_reason,
            "agent_steps": final_payload.get("agent_steps"),
            "tool_calls": final_payload.get("tool_calls"),
            "repair_attempts": final_payload.get("repair_attempts"),
            **{
                key: value
                for key, value in metrics.items()
                if key.startswith(("recall_at_", "f1_at_", "ndcg_at_"))
            },
        }
    )
    return summary


def _write_summary(summary: dict[str, Any], summary_path: Path | None) -> None:
    if summary_path is None:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if not args.openai_service or "/" in args.openai_service:
        raise ValueError("--openai-service must be a non-empty service prefix without '/'")

    # Keep summary helpers importable for offline tests without importing the
    # full Inspect stack. These imports are only required for an actual eval.
    from inspect_ai import eval as inspect_eval

    from dci_bench.data.registry import get_task
    from dci_bench.data.workspace_builder import build_workspace
    from dci_bench.tasks.mteb_llm_retrieval import mteb_llm_retrieval_single

    if args.prepare:
        build_workspace(get_task(args.task), overwrite=True)
    task = mteb_llm_retrieval_single(
        task_name=args.task,
        query_id=args.query_id,
        output_dir=args.output_dir,
        metric_ks=args.metric_ks,
    )
    logs = inspect_eval(
        task,
        model=f"openai-api/{args.openai_service}/{args.served_model_name}",
        model_base_url=args.openai_base_url,
        model_args={"api_key": args.api_key, "responses_api": False},
        log_dir=args.log_dir,
        display="plain",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        max_samples=1,
        fail_on_error=False,
        score_on_error=True,
    )
    inspect_log_paths = [str(getattr(log, "location", "")) for log in logs]
    summary = build_sample_summary(
        task_name=args.task,
        query_id=args.query_id,
        output_dir=Path(args.output_dir),
        metadata_root=Path(args.metadata_root),
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        openai_base_url=args.openai_base_url,
        openai_service=args.openai_service,
        inspect_log_paths=inspect_log_paths,
        metric_ks=args.metric_ks,
    )
    _write_summary(summary, args.summary_path)
    print(logs)
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if args.require_valid_output and not summary["valid_output"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
