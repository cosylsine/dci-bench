#!/usr/bin/env python
"""Run one immutable, auditable DCI-Bench Phase 4 task evaluation."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dci_bench.data.registry import TASKS, get_task
from dci_bench.results.phase4_runner import execute_phase4
from dci_bench.scoring.retrieval import DEFAULT_METRIC_KS, normalize_metric_ks

try:
    # Keep ``--help`` and offline argument validation usable on hosts that do
    # not install Inspect.  An actual run imports these lazily below.
    from dci_bench.tasks.mteb_llm_retrieval import (
        _select_query_ids,
        mteb_llm_retrieval,
    )
except (ImportError, OSError):  # pragma: no cover - depends on environment
    _select_query_ids = None  # type: ignore[assignment]
    mteb_llm_retrieval = None  # type: ignore[assignment]


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 65535:
        raise argparse.ArgumentTypeError("must be in TCP port range 1..65535")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _safe_component(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{name} must contain only letters, digits, '.', '_' or '-' and start with a letter/digit"
        )
    return value


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default="MiniCPM5-2B")
    parser.add_argument("--task", default="LLMPublicHealthQA", choices=sorted(TASKS))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--query-ids",
        nargs="+",
        metavar="QUERY_ID",
        help="one or more query IDs, evaluated in the order supplied",
    )
    selection.add_argument(
        "--all-queries",
        action="store_true",
        help="evaluate every query in metadata/queries.jsonl order",
    )
    runs = parser.add_mutually_exclusive_group()
    runs.add_argument("--run-id")
    runs.add_argument("--resume-run", metavar="RUN_ID")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="root for model/task/runs/<run-id> artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="compatibility override for the run directory (advanced)",
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("data/workspaces"))
    parser.add_argument("--metadata-root", type=Path, default=Path("data/metadata"))
    parser.add_argument("--backend-manifest", type=Path)
    parser.add_argument(
        "--backend-artifact",
        action="append",
        type=Path,
        default=[],
        help="backend diagnostic file to copy into this run (repeatable)",
    )
    parser.add_argument("--model-path", default="/mnt/afs/share/Qwen3-4B")
    parser.add_argument("--served-model-name", default="MiniCPM5-2B")
    parser.add_argument(
        "--openai-base-url",
        "--vllm-base-url",
        dest="openai_base_url",
        default="http://127.0.0.1:30000/v1",
        help="OpenAI-compatible /v1 endpoint",
    )
    parser.add_argument(
        "--openai-service",
        default="sglang",
        help="Inspect service prefix used in openai-api/<service>/<model>",
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--bridge-port", type=_port, default=13131)
    parser.add_argument("--max-concurrency", type=_positive_int, default=1)
    parser.add_argument("--launcher-preflight-seconds", type=_nonnegative_float)
    parser.add_argument(
        "--metric-ks",
        nargs="+",
        type=int,
        default=list(DEFAULT_METRIC_KS),
        metavar="K",
        help="retrieval metric cutoffs (default: 1 3 5 10 20)",
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--require-valid-output", action="store_true")
    return parser.parse_args(argv)


def _load_backend_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"backend manifest must contain a JSON object: {path}")
    return payload


def _backend_value(manifest: dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        value = manifest.get(key)
        if value is not None:
            return value
    return default


def _backend_section(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("backend", {})
    return value if isinstance(value, dict) else {}


def _model_section(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("model", {})
    return value if isinstance(value, dict) else {}


def _supports_keyword(callable_obj: Callable[..., Any], keyword: str) -> bool:
    """Check optional Inspect kwargs while supporting test doubles/old Inspect."""

    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind is parameter.VAR_KEYWORD or parameter.name == keyword
        for parameter in parameters
    )


class BatchRunInvocation:
    __slots__ = (
        "task_name",
        "model_key",
        "run_id",
        "run_dir",
        "query_ids",
        "inspect_log_paths",
    )

    def __init__(
        self,
        *,
        task_name: str,
        model_key: str,
        run_id: str,
        run_dir: Path,
        query_ids: tuple[str, ...],
        inspect_log_paths: tuple[str, ...],
    ) -> None:
        self.task_name = task_name
        self.model_key = model_key
        self.run_id = run_id
        self.run_dir = run_dir
        self.query_ids = query_ids
        self.inspect_log_paths = inspect_log_paths


def _resolve_run_dir(args: argparse.Namespace) -> tuple[str, Path]:
    model_key = _safe_component(args.model_key, name="model-key")
    task_name = _safe_component(args.task, name="task")
    if args.output_dir is not None:
        run_id = args.resume_run or args.run_id or args.output_dir.name
        _safe_component(run_id, name="run-id")
        run_dir = args.output_dir
    else:
        run_id = args.resume_run or args.run_id or _default_run_id()
        _safe_component(run_id, name="run-id")
        run_dir = args.results_root / model_key / task_name / "runs" / run_id
    if args.resume_run:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume run does not exist: {run_dir}")
        if (run_dir / "_COMPLETE").exists():
            raise ValueError(f"cannot resume a completed run: {run_dir}")
    elif run_dir.exists():
        raise FileExistsError(
            f"run artifact directory already exists: {run_dir}; choose a new --run-id"
        )
    return run_id, run_dir


def _inspect_eval_kwargs(
    args: argparse.Namespace,
    *,
    inspect_eval_fn: Callable[..., Any],
    manifest: dict[str, Any],
    log_dir: Path,
    sample_count: int,
) -> dict[str, Any]:
    backend = _backend_section(manifest)
    model = _model_section(manifest)
    base_url = str(
        _backend_value(
            {**manifest, **backend},
            "openai_base_url",
            "base_url",
            "endpoint",
            default=args.openai_base_url,
        )
    ).rstrip("/")
    service = str(
        _backend_value(
            {**manifest, **backend},
            "openai_service",
            "service",
            default=args.openai_service,
        )
    )
    served_model_name = str(
        _backend_value(
            {**manifest, **model},
            "served_model_name",
            "served_model",
            "model",
            default=args.served_model_name,
        )
    )
    api_key = str(_backend_value({**manifest, **backend}, "api_key", default=args.api_key))
    kwargs: dict[str, Any] = {
        "model": f"openai-api/{service}/{served_model_name}",
        "model_base_url": base_url,
        "model_args": {"api_key": api_key, "responses_api": False},
        "log_dir": str(log_dir),
        "display": "plain",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        # ``max_samples`` is the Inspect worker/concurrency cap used by the
        # pinned runner contract.  The dataset itself still contains all
        # selected samples; ``max_connections`` below controls the same cap on
        # newer Inspect releases.
        "max_samples": args.max_concurrency,
        "fail_on_error": False,
        "score_on_error": True,
    }
    # Inspect currently calls this setting ``max_connections``.  The fallback
    # keeps the CLI usable with test doubles and older versions exposing the
    # clearer ``max_concurrency`` spelling.
    if _supports_keyword(inspect_eval_fn, "max_connections"):
        kwargs["max_connections"] = args.max_concurrency
    elif _supports_keyword(inspect_eval_fn, "max_concurrency"):
        kwargs["max_concurrency"] = args.max_concurrency
    return kwargs


def run_batch(
    args: argparse.Namespace,
    *,
    inspect_eval_fn: Callable[..., Any] | None = None,
) -> BatchRunInvocation:
    """Validate selection, construct one batch Task, and invoke Inspect once."""

    global _select_query_ids, mteb_llm_retrieval
    if _select_query_ids is None or mteb_llm_retrieval is None:
        from dci_bench.tasks.mteb_llm_retrieval import (
            _select_query_ids as select_query_ids,
            mteb_llm_retrieval as batch_task,
        )

        _select_query_ids = select_query_ids
        mteb_llm_retrieval = batch_task

    if args.prepare:
        from dci_bench.data.workspace_builder import build_workspace

        # Preparation is explicit and fail-safe.  Existing workspaces are not
        # overwritten by the Phase 4 runner; use the data builder directly
        # with its explicit ``overwrite=True`` option when a rebuild is truly
        # intended.
        build_workspace(
            get_task(args.task),
            output_root=args.workspace_root,
            metadata_root=args.metadata_root,
            overwrite=False,
        )

    # Keep this preflight before importing/starting an endpoint.  It rejects
    # duplicate, unknown, and qrels-less IDs without making a model request.
    if args.task not in TASKS:
        raise ValueError(f"Unknown task {args.task!r}. Available: {sorted(TASKS)}")
    selected_ids = _select_query_ids(
        args.task,
        metadata_root=args.metadata_root,
        query_ids=args.query_ids,
        all_queries=bool(args.all_queries),
    )
    metric_ks = normalize_metric_ks(args.metric_ks)
    manifest = _load_backend_manifest(args.backend_manifest)
    run_id, run_dir = _resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or run_dir / "artifacts" / "inspect"
    log_dir.mkdir(parents=True, exist_ok=True)

    task = mteb_llm_retrieval(
        task_name=args.task,
        query_ids=selected_ids,
        all_queries=False,
        workspace_root=str(args.workspace_root),
        metadata_root=str(args.metadata_root),
        output_dir=str(run_dir),
        metric_ks=metric_ks,
        bridge_port=args.bridge_port,
        max_concurrency=args.max_concurrency,
    )
    if inspect_eval_fn is None:
        from inspect_ai import eval as inspect_eval_fn  # type: ignore[assignment]

    kwargs = _inspect_eval_kwargs(
        args,
        inspect_eval_fn=inspect_eval_fn,
        manifest=manifest,
        log_dir=log_dir,
        sample_count=len(selected_ids),
    )
    logs = inspect_eval_fn(task, **kwargs)
    paths = tuple(str(getattr(log, "location", "")) for log in logs)
    return BatchRunInvocation(
        task_name=args.task,
        model_key=args.model_key,
        run_id=run_id,
        run_dir=run_dir,
        query_ids=tuple(selected_ids),
        inspect_log_paths=paths,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        global _select_query_ids
        if args.backend_manifest is None:
            raise ValueError("--backend-manifest is required for a Phase 4 run")
        if args.output_dir is not None:
            raise ValueError("--output-dir is only available through the legacy run_batch() API")
        if args.prepare:
            from dci_bench.data.workspace_builder import build_workspace

            build_workspace(
                get_task(args.task),
                output_root=args.workspace_root,
                metadata_root=args.metadata_root,
                overwrite=False,
            )
        if _select_query_ids is None:
            from dci_bench.tasks.mteb_llm_retrieval import _select_query_ids as selector

            _select_query_ids = selector
        selected_ids = _select_query_ids(
            args.task,
            metadata_root=args.metadata_root,
            query_ids=args.query_ids,
            all_queries=bool(args.all_queries),
        )
        execution = execute_phase4(args, selected_query_ids=selected_ids)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"[DCI] Phase 4 configuration/integrity failure: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "task": args.task,
                "model_key": args.model_key,
                "run_id": execution.run_id,
                "run_dir": str(execution.run_dir),
                "query_ids": list(execution.query_ids),
                "valid_samples": execution.valid_samples,
                "invalid_samples": execution.invalid_samples,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return execution.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
