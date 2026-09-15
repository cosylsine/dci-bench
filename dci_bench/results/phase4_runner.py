"""Phase 4 run lifecycle and Inspect-to-DCI result adaptation."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from dci_bench.backends.manifest import load_backend_manifest
from dci_bench.protocol.contracts import (
    BENCHMARK_CONTRACT_VERSION,
    MAX_TOTAL_MODEL_TOKENS,
    protocol_contract,
)
from dci_bench.results.protocol import (
    ResultIntegrityError,
    ResultValidationError,
    append_task_index_run,
    artifact_record,
    build_run_manifest,
    build_sample_result,
    build_task_result,
    compute_run_fingerprint,
    finalize_run,
    read_json,
    run_directory,
    sha256_bytes,
    sha256_file,
    sha256_json,
    update_task_index_run,
    validate_run_manifest,
    validate_sample_result,
    validate_trajectory,
    verify_artifact_record,
    write_json_atomic,
    write_run_manifest,
    write_task_result,
)
from dci_bench.scoring.retrieval import normalize_metric_ks


@dataclass(frozen=True)
class Phase4Execution:
    run_id: str
    run_dir: Path
    query_ids: tuple[str, ...]
    valid_samples: int
    invalid_samples: int
    exit_code: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_command(argv: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_audit(root: Path) -> dict[str, Any]:
    dot_git = root / ".git"
    if dot_git.is_file():
        marker = dot_git.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir:"):
            raise ValueError(f"invalid gitdir marker: {dot_git}")
        git_dir = Path(marker.removeprefix("gitdir:").strip())
        if not git_dir.is_absolute():
            git_dir = (root / git_dir).resolve()
    else:
        git_dir = dot_git
    git = ["git", f"--git-dir={git_dir}", f"--work-tree={root}"]
    revision = _run_command([*git, "rev-parse", "HEAD"], cwd=root)
    status = _run_command([*git, "status", "--porcelain=v1", "--untracked-files=all"], cwd=root)
    diff = _run_command([*git, "diff", "--binary", "HEAD", "--"], cwd=root)
    if revision.returncode != 0 or status.returncode != 0 or diff.returncode != 0:
        return {
            "revision": None,
            "dirty": None,
            "diff_fingerprint": None,
            "audit_unavailable": True,
        }
    untracked_hashes: list[dict[str, str]] = []
    for line in status.stdout.splitlines():
        if not line.startswith("?? "):
            continue
        relative = line[3:]
        path = root / relative
        if path.is_file() and not relative.startswith("results/"):
            untracked_hashes.append({"path": relative, "sha256": sha256_file(path)})
    diff_payload = {
        "status": status.stdout,
        "tracked_diff_sha256": sha256_bytes(diff.stdout.encode("utf-8")),
        "untracked": untracked_hashes,
    }
    return {
        "revision": revision.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
        "diff_fingerprint": sha256_json(diff_payload),
        "audit_unavailable": False,
    }


def code_audit(repo_root: Path) -> dict[str, Any]:
    audit = _git_audit(repo_root)
    pi_root = repo_root / "pi-dci"
    pi_audit = _git_audit(pi_root) if pi_root.is_dir() else {
        "revision": None,
        "dirty": None,
        "diff_fingerprint": None,
        "audit_unavailable": True,
    }
    audited_sources = [
        *sorted((repo_root / "dci_bench").rglob("*.py")),
        *sorted((repo_root / "schemas" / "results").glob("*.json")),
        repo_root / "scripts" / "run_task.py",
        repo_root / "scripts" / "run_phase3_with_sglang.sh",
        repo_root / "scripts" / "serve_sglang_minicpm5.sh",
        repo_root / "pi-dci" / "scripts" / "dci-run.ts",
    ]
    source_hashes = [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in audited_sources
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return {
        "dci_bench": audit,
        "pi": pi_audit,
        "inspect_version": _package_version("inspect-ai"),
        "source_tree_fingerprint": sha256_json(source_hashes),
    }


def data_audit(metadata_root: Path, task: str) -> dict[str, Any]:
    task_root = metadata_root / task
    manifest_path = task_root / "manifest.json"
    queries_path = task_root / "queries.jsonl"
    if not manifest_path.is_file() or not queries_path.is_file():
        raise FileNotFoundError(f"missing task metadata manifest or queries under {task_root}")
    source = read_json(manifest_path)
    if not isinstance(source, Mapping):
        raise ValueError(f"data manifest must be an object: {manifest_path}")
    return {
        "dataset": source.get("dataset"),
        "requested_revision": source.get("revision"),
        "resolved_revision": source.get("resolved_revision"),
        "source_manifest_path": manifest_path.relative_to(metadata_root.parent).as_posix(),
        "source_manifest_sha256": sha256_file(manifest_path),
        "queries_sha256": sha256_file(queries_path),
        "corpus_sha256": source.get("corpus_hash"),
        "source_parquet_sha256": source.get("source_parquet_hash"),
    }


def load_query_texts(metadata_root: Path, task: str) -> dict[str, str]:
    path = metadata_root / task / "queries.jsonl"
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        query_id = item.get("_id")
        text = item.get("text")
        if not isinstance(query_id, str) or not query_id or not isinstance(text, str):
            raise ValueError(f"invalid query at {path}:{line_number}")
        if query_id in result:
            raise ValueError(f"duplicate query id {query_id!r} in {path}")
        result[query_id] = text
    return result


def _backend_audit(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = manifest.get("model")
    backend = manifest.get("backend")
    if not isinstance(model, Mapping) or not isinstance(backend, Mapping):
        raise ValueError("backend manifest requires model and backend objects")
    backend_info = dict(backend)
    backend_info["capabilities"] = dict(manifest.get("capabilities", {}))
    return dict(model), backend_info


def _run_id(fingerprint: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{fingerprint[:12]}-{secrets.token_hex(3)}"


def preflight_bridge_ports(base_port: int, max_concurrency: int) -> tuple[int, ...]:
    """Fail before model work if any bridge port in the requested pool is busy."""

    if base_port < 1 or max_concurrency < 1 or base_port + max_concurrency - 1 > 65535:
        raise ValueError("bridge port pool must fit in TCP port range 1..65535")
    listeners: list[socket.socket] = []
    ports = tuple(base_port + offset for offset in range(max_concurrency))
    try:
        for port in ports:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                listener.bind(("127.0.0.1", port))
            except OSError as exc:
                listener.close()
                raise ValueError(f"bridge port {port} is unavailable: {exc}") from exc
            listeners.append(listener)
    finally:
        for listener in listeners:
            listener.close()
    return ports


def _artifact_path(location: str) -> Path | None:
    if not location:
        return None
    parsed = urlparse(location)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        return None
    return Path(location)


def _usage_value(value: Any, name: str) -> int | None:
    raw = getattr(value, name, None)
    return raw if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else None


def inspect_usage(sample: Any) -> dict[str, Any]:
    raw_usage = getattr(sample, "model_usage", None)
    if not isinstance(raw_usage, Mapping) or not raw_usage:
        return {
            "available": False,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "by_model": {},
        }
    by_model: dict[str, Any] = {}
    for model_name, raw in raw_usage.items():
        input_tokens = _usage_value(raw, "input_tokens")
        output_tokens = _usage_value(raw, "output_tokens")
        total_tokens = _usage_value(raw, "total_tokens")
        cache_read = _usage_value(raw, "input_tokens_cache_read")
        cache_write = _usage_value(raw, "input_tokens_cache_write")
        available = input_tokens is not None and output_tokens is not None and total_tokens is not None
        by_model[str(model_name)] = {
            "available": available,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
        }
    available = all(item["available"] for item in by_model.values())

    def sum_required(name: str) -> int | None:
        values = [item[name] for item in by_model.values()]
        return sum(values) if all(value is not None for value in values) else None

    return {
        "available": available,
        "input_tokens": sum_required("input_tokens"),
        "output_tokens": sum_required("output_tokens"),
        "total_tokens": sum_required("total_tokens"),
        "cache_read_tokens": sum_required("cache_read_tokens"),
        "cache_write_tokens": sum_required("cache_write_tokens"),
        "by_model": by_model,
    }


def _score_metrics(sample: Any) -> dict[str, float]:
    scores = getattr(sample, "scores", None)
    if not isinstance(scores, Mapping):
        return {}
    for score in scores.values():
        value = getattr(score, "value", None)
        if isinstance(value, Mapping):
            return {
                str(name): float(metric)
                for name, metric in value.items()
                if str(name).startswith(("recall_at_", "f1_at_", "ndcg_at_"))
                and isinstance(metric, (int, float))
                and not isinstance(metric, bool)
            }
    return {}


def _load_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "file was not produced"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "JSON root is not an object"
    return payload, None


_FAILURE_MAP = {
    "timeout": "timeout",
    "token_limit": "token_limit",
    "invalid_json": "invalid_json",
    "tool_failure": "tool_failure",
    "context_overflow": "context_overflow",
    "agent_error": "agent_error",
    "max_agent_steps": "agent_error",
    "infrastructure_error": "infrastructure_error",
    "usage_unavailable": "usage_unavailable",
}


def _classify_message(message: str) -> str:
    lowered = message.lower()
    if "timeout" in lowered or "time limit" in lowered:
        return "timeout"
    if "context" in lowered and ("overflow" in lowered or "length" in lowered):
        return "context_overflow"
    if "token" in lowered and ("limit" in lowered or "budget" in lowered):
        return "token_limit"
    if "tool" in lowered:
        return "tool_failure"
    if "json" in lowered:
        return "invalid_json"
    return "infrastructure_error"


def _redact(message: str, secrets_to_remove: Sequence[str] = ()) -> str:
    result = message
    for secret in secrets_to_remove:
        if secret:
            result = result.replace(secret, "[redacted]")
    result = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[redacted]", result)
    result = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+", r"\1[redacted]", result)
    return result[:4000]


def _sample_error(sample: Any) -> str | None:
    error = getattr(sample, "error", None)
    if error is None:
        return None
    message = getattr(error, "message", None)
    return str(message if message is not None else error)


def _normalize_trajectory(
    *,
    path: Path,
    run_id: str,
    task: str,
    query_id: str,
    sample: Any | None,
    failure: Mapping[str, str] | None,
) -> dict[str, Any]:
    payload, read_error = _load_object(path)
    if payload is None:
        payload = {"events": [], "messages": [], "trace_unavailable": read_error}
    payload["schema_version"] = "dci-trajectory-v1"
    payload["run_id"] = run_id
    payload["task"] = task
    payload["query_id"] = query_id
    payload["started_at"] = (
        getattr(sample, "started_at", None) if sample is not None else payload.get("started_at")
    ) or _utc_now()
    payload["ended_at"] = (
        getattr(sample, "completed_at", None)
        if sample is not None
        else payload.get("completed_at") or payload.get("ended_at")
    ) or _utc_now()
    payload.setdefault("events", [])
    payload.setdefault("messages", [])
    payload["failure"] = dict(failure) if failure is not None else None
    write_json_atomic(path, payload)
    validate_trajectory(payload)
    return payload


def _trace_total(trace: Mapping[str, Any]) -> int | None:
    usage = trace.get("usage_summary", trace.get("usage"))
    if not isinstance(usage, Mapping):
        return None
    value = usage.get("total_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _failure_for(
    *,
    final: Mapping[str, Any] | None,
    final_error: str | None,
    inspect_sample: Any | None,
    usage: Mapping[str, Any],
    secrets_to_remove: Sequence[str],
) -> dict[str, str] | None:
    if final is not None:
        failure = final.get("failure")
        if isinstance(failure, Mapping):
            kind = _FAILURE_MAP.get(str(failure.get("kind")), "agent_error")
            message = _redact(str(failure.get("message") or kind), secrets_to_remove)
            return {"kind": kind, "message": message}
        if not bool(final.get("valid_output", False)):
            message = str(final.get("failure_reason") or "Pi runner marked output invalid")
            return {"kind": _classify_message(message), "message": _redact(message, secrets_to_remove)}
    if final_error is not None:
        return {
            "kind": "invalid_json" if "JSON" in final_error else "infrastructure_error",
            "message": _redact(f"final.json unavailable: {final_error}", secrets_to_remove),
        }
    if inspect_sample is None:
        return {"kind": "infrastructure_error", "message": "Inspect sample record unavailable"}
    inspect_error = _sample_error(inspect_sample)
    if inspect_error:
        return {"kind": _classify_message(inspect_error), "message": _redact(inspect_error, secrets_to_remove)}
    if usage.get("available") is not True:
        return {
            "kind": "usage_unavailable",
            "message": "Inspect sample model_usage was missing or incomplete",
        }
    return None


def _placeholder_final(path: Path, failure: Mapping[str, str]) -> dict[str, Any]:
    payload = {
        "contract_version": BENCHMARK_CONTRACT_VERSION,
        "valid_output": False,
        "failure_reason": failure["message"],
        "failure": dict(failure),
        "ranked_doc_ids": [],
        "inspect_sandbox": "local",
        "tool_sandbox": "bubblewrap",
        "artifact_unavailable": True,
    }
    write_json_atomic(path, payload)
    return payload


def build_and_publish_sample(
    *,
    run_dir: Path,
    staging_root: Path,
    run_id: str,
    task: str,
    query_id: str,
    query_text: str,
    metric_ks: Sequence[int],
    inspect_sample: Any | None,
    inspect_artifact: Mapping[str, Any] | None,
    secrets_to_remove: Sequence[str] = (),
) -> dict[str, Any]:
    staged = staging_root / query_id
    staged.mkdir(parents=True, exist_ok=True)
    final_path = staged / "final.json"
    trace_path = staged / "trace.json"
    final, final_error = _load_object(final_path)
    if final is not None:
        # Normalize adapter-produced JSON as part of the staging transaction.
        # This keeps every published audit file human-readable even when a
        # future backend adapter emits compact JSON.
        write_json_atomic(final_path, final)
    usage = inspect_usage(inspect_sample) if inspect_sample is not None else inspect_usage(object())
    failure = _failure_for(
        final=final,
        final_error=final_error,
        inspect_sample=inspect_sample,
        usage=usage,
        secrets_to_remove=secrets_to_remove,
    )
    if final is None:
        final = _placeholder_final(
            final_path,
            failure or {"kind": "infrastructure_error", "message": "final.json unavailable"},
        )
    raw_valid = bool(final.get("valid_output", False))
    sandbox_valid = (
        final.get("inspect_sandbox") == "local" and final.get("tool_sandbox") == "bubblewrap"
    )
    timing = {
        "started_at": getattr(inspect_sample, "started_at", None) if inspect_sample is not None else None,
        "ended_at": getattr(inspect_sample, "completed_at", None) if inspect_sample is not None else None,
        "wall_time_seconds": getattr(inspect_sample, "total_time", None) if inspect_sample is not None else None,
        "working_time_seconds": getattr(inspect_sample, "working_time", None) if inspect_sample is not None else None,
    }
    timing_available = timing["wall_time_seconds"] is not None and timing["working_time_seconds"] is not None
    valid = raw_valid and failure is None and sandbox_valid and usage["available"] is True and timing_available
    if raw_valid and failure is None and not sandbox_valid:
        failure = {
            "kind": "infrastructure_error",
            "message": "Pi result did not attest local Inspect and Bubblewrap tool sandboxes",
        }
    if raw_valid and failure is None and usage["available"] is not True:
        failure = {
            "kind": "usage_unavailable",
            "message": "Inspect sample model_usage was missing or incomplete",
        }
    if raw_valid and failure is None and not timing_available:
        failure = {
            "kind": "infrastructure_error",
            "message": "Inspect sample timing was missing or incomplete",
        }
    if not valid and failure is None:
        failure = {"kind": "agent_error", "message": "sample did not reach a valid terminal output"}
    trajectory = _normalize_trajectory(
        path=trace_path,
        run_id=run_id,
        task=task,
        query_id=query_id,
        sample=inspect_sample,
        failure=failure,
    )
    trace_total = _trace_total(trajectory)
    inspect_total = usage.get("total_tokens")
    metrics = _score_metrics(inspect_sample) if inspect_sample is not None else {}
    turns = final.get("usage_summary", {}).get("turns") if isinstance(final.get("usage_summary"), Mapping) else None
    model_calls = turns if isinstance(turns, int) and turns >= 0 else len(trajectory.get("usage_by_turn", []))
    artifacts: dict[str, Any] = {
        "final": artifact_record(final_path, root=staging_root),
        "trace": artifact_record(trace_path, root=staging_root),
    }
    if inspect_artifact is not None:
        artifacts["inspect_log"] = dict(inspect_artifact)
    sample_result = build_sample_result(
        run_id=run_id,
        task=task,
        query_id=query_id,
        query_sha256=sha256_bytes(query_text.encode("utf-8")),
        valid_output=valid,
        ranked_doc_ids=final.get("ranked_doc_ids", []) if valid else [],
        metrics=metrics,
        metric_ks=metric_ks,
        status="success" if valid else failure["kind"],
        failure=None if valid else failure,
        usage=usage,
        timing=timing,
        execution={
            "agent_steps": final.get("agent_steps", 0),
            "tool_calls": final.get("tool_calls", 0),
            "model_calls": model_calls,
            "repair_attempts": final.get("repair_attempts", 0),
            "repair_reasons": final.get("repair_reasons", []),
        },
        security={"inspect_sandbox": "local", "tool_sandbox": "bubblewrap"},
        artifacts=artifacts,
        provenance={
            "metrics": "inspect.sample.scores",
            "usage": "inspect.sample.model_usage",
            "timing": "inspect.sample.total_time/working_time",
            "execution": "pi.final/trace",
            "trace": "pi.trace",
            "trace_total_tokens": trace_total,
            "usage_cross_check": (
                None if trace_total is None or inspect_total is None else trace_total == inspect_total
            ),
        },
    )
    write_json_atomic(staged / "result.json", sample_result)
    validate_sample_result(read_json(staged / "result.json"))
    for name, record in sample_result["artifacts"].items():
        if name == "inspect_log":
            continue
        verify_artifact_record(record, root=staging_root)
    destination = run_dir / query_id
    if destination.exists():
        raise ResultIntegrityError(f"refusing to replace existing sample directory: {destination}")
    os.replace(staged, destination)
    return sample_result


def load_completed_sample(run_dir: Path, query_id: str) -> dict[str, Any] | None:
    sample_dir = run_dir / query_id
    if not sample_dir.exists():
        return None
    result_path = sample_dir / "result.json"
    if not result_path.is_file():
        raise ResultIntegrityError(f"existing sample is incomplete: {sample_dir}")
    payload = read_json(result_path)
    if not isinstance(payload, dict):
        raise ResultIntegrityError(f"sample result is not an object: {result_path}")
    validate_sample_result(payload)
    if payload.get("query_id") != query_id:
        raise ResultIntegrityError(f"sample query identity mismatch: {result_path}")
    for record in payload["artifacts"].values():
        verify_artifact_record(record, root=run_dir)
    return payload


def _inspect_sample_map(logs: Sequence[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for log in logs:
        for sample in getattr(log, "samples", None) or []:
            query_id = str(getattr(sample, "id", ""))
            if not query_id:
                continue
            if query_id in result:
                raise ResultIntegrityError(f"duplicate Inspect sample record for {query_id}")
            result[query_id] = sample
    return result


def _inspect_artifacts(run_dir: Path, logs: Sequence[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for log in logs:
        path = _artifact_path(str(getattr(log, "location", "")))
        if path is None or not path.is_file():
            continue
        records.append(artifact_record(path, root=run_dir))
    return records


def _manifest_entry(manifest: Mapping[str, Any], status: str) -> dict[str, Any]:
    return {
        "run_id": manifest["run_id"],
        "status": status,
        "run_fingerprint": manifest["run_fingerprint"],
        "task_result_path": f"runs/{manifest['run_id']}/task-result.json",
    }


def execute_phase4(
    args: Any,
    *,
    selected_query_ids: Sequence[str],
    inspect_eval_fn: Callable[..., Any] | None = None,
    task_factory: Callable[..., Any] | None = None,
) -> Phase4Execution:
    """Execute or resume one immutable Phase 4 task run."""

    repo_root = Path(__file__).resolve().parents[2]
    metric_ks = normalize_metric_ks(args.metric_ks)
    preflight_bridge_ports(args.bridge_port, args.max_concurrency)
    backend_source = Path(args.backend_manifest)
    backend_manifest = load_backend_manifest(backend_source)
    if backend_manifest["model"]["model_key"] != args.model_key:
        raise ValueError("--model-key does not match backend manifest model.model_key")
    model_info, backend_info = _backend_audit(backend_manifest)
    data_info = data_audit(Path(args.metadata_root), args.task)
    code_info = code_audit(repo_root)
    contract = protocol_contract()
    fingerprint = compute_run_fingerprint(
        model=model_info,
        backend=backend_info,
        data=data_info,
        benchmark_contract=contract,
        metric_ks=metric_ks,
        query_ids=selected_query_ids,
        code=code_info,
    )
    run_id = args.resume_run or args.run_id or _run_id(fingerprint)
    run_dir = run_directory(
        Path(args.results_root), model_key=args.model_key, task=args.task, run_id=run_id
    )
    task_root = Path(args.results_root) / args.model_key / args.task
    index_path = task_root / "index.json"
    manifest = build_run_manifest(
        run_id=run_id,
        model_key=args.model_key,
        task=args.task,
        query_ids=selected_query_ids,
        model=model_info,
        backend=backend_info,
        data=data_info,
        benchmark_contract=contract,
        metric_ks=metric_ks,
        code=code_info,
        run_fingerprint=fingerprint,
        max_total_model_tokens=MAX_TOTAL_MODEL_TOKENS,
    )
    if args.resume_run:
        if not run_dir.is_dir() or (run_dir / "_COMPLETE").exists():
            raise ValueError(f"resume requires an existing incomplete run: {run_dir}")
        existing_manifest = read_json(run_dir / "run-manifest.json")
        if not isinstance(existing_manifest, dict):
            raise ResultIntegrityError("existing run manifest is not an object")
        validate_run_manifest(existing_manifest)
        if existing_manifest["run_fingerprint"] != fingerprint:
            raise ResultIntegrityError("resume run fingerprint does not match current configuration")
        manifest = existing_manifest
    else:
        if run_dir.exists():
            raise FileExistsError(f"run already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        write_run_manifest(run_dir, manifest)
        append_task_index_run(
            index_path,
            model_key=args.model_key,
            task=args.task,
            run=_manifest_entry(manifest, "running"),
        )

    backend_copy = run_dir / "artifacts" / "backend" / "backend-manifest.json"
    backend_copy.parent.mkdir(parents=True, exist_ok=True)
    if backend_copy.exists():
        if sha256_file(backend_copy) != sha256_file(backend_source):
            raise ResultIntegrityError("backend manifest artifact changed during resume")
    else:
        shutil.copyfile(backend_source, backend_copy)
    backend_diagnostics: list[dict[str, Any]] = []
    for source_value in getattr(args, "backend_artifact", []):
        source = Path(source_value)
        if not source.is_file():
            raise FileNotFoundError(f"backend diagnostic artifact not found: {source}")
        destination = backend_copy.parent / source.name
        if destination == backend_copy:
            raise ValueError("backend diagnostic name conflicts with backend-manifest.json")
        if destination.exists():
            # Readiness and tool-smoke payloads can contain service-instance
            # identifiers and generated text, so a restarted but identically
            # configured backend need not reproduce the same bytes. Preserve
            # the original run evidence; the canonical backend manifest above
            # remains hash-checked and participates in the run fingerprint.
            if not args.resume_run:
                raise FileExistsError(f"backend diagnostic already exists: {destination}")
        else:
            shutil.copyfile(source, destination)
        backend_diagnostics.append(artifact_record(destination, root=run_dir))

    existing: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for query_id in selected_query_ids:
        sample = load_completed_sample(run_dir, query_id)
        if sample is None:
            pending.append(query_id)
        else:
            if sample.get("run_id") != run_id or sample.get("task") != args.task:
                raise ResultIntegrityError(f"sample identity mismatch for {query_id}")
            existing[query_id] = sample
    if (run_dir / "task-result.json").exists() and pending:
        raise ResultIntegrityError("task-result.json exists while planned samples are incomplete")

    # Staging files are deliberately non-terminal and may be truncated when a
    # launcher or service is interrupted. A fingerprint-validated resume must
    # never promote or retain them as audit artifacts; only published Qxx
    # directories with verified hashes are eligible for skipping.
    stale_staging = run_dir / ".staging"
    if args.resume_run and stale_staging.exists():
        shutil.rmtree(stale_staging)

    query_texts = load_query_texts(Path(args.metadata_root), args.task)
    evaluation_started = time.monotonic()
    logs: list[Any] = []
    eval_error: Exception | None = None
    staging_root = run_dir / ".staging" / secrets.token_hex(8)
    if pending:
        staging_root.mkdir(parents=True)
        if task_factory is None:
            from dci_bench.tasks.mteb_llm_retrieval import mteb_llm_retrieval as task_factory
        task = task_factory(
            task_name=args.task,
            query_ids=pending,
            all_queries=False,
            workspace_root=str(args.workspace_root),
            metadata_root=str(args.metadata_root),
            output_dir=str(staging_root),
            metric_ks=metric_ks,
            bridge_port=args.bridge_port,
            max_concurrency=args.max_concurrency,
        )
        if inspect_eval_fn is None:
            from inspect_ai import eval as inspect_eval_fn
        backend = backend_manifest["backend"]
        model = backend_manifest["model"]
        log_dir = run_dir / "artifacts" / "inspect"
        log_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "model": f"openai-api/{backend['openai_service']}/{model['served_model_name']}",
            "model_base_url": backend["base_url"],
            "model_args": {"api_key": args.api_key, "responses_api": False},
            "log_dir": str(log_dir),
            "display": "plain",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "max_samples": args.max_concurrency,
            "fail_on_error": False,
            "score_on_error": True,
        }
        try:
            logs = list(inspect_eval_fn(task, **kwargs))
        except Exception as exc:  # retain per-sample infrastructure failures
            eval_error = exc
    evaluation_elapsed = time.monotonic() - evaluation_started

    inspect_by_id = _inspect_sample_map(logs)
    inspect_records = _inspect_artifacts(run_dir, logs)
    shared_inspect = inspect_records[0] if len(inspect_records) == 1 else None
    produced: dict[str, dict[str, Any]] = {}
    for query_id in pending:
        inspect_sample = inspect_by_id.get(query_id)
        if eval_error is not None and inspect_sample is None:
            failure = {
                "kind": "infrastructure_error",
                "message": _redact(str(eval_error), [args.api_key]),
            }
            sample_stage = staging_root / query_id
            sample_stage.mkdir(parents=True, exist_ok=True)
            _placeholder_final(sample_stage / "final.json", failure)
        produced[query_id] = build_and_publish_sample(
            run_dir=run_dir,
            staging_root=staging_root,
            run_id=run_id,
            task=args.task,
            query_id=query_id,
            query_text=query_texts[query_id],
            metric_ks=metric_ks,
            inspect_sample=inspect_sample,
            inspect_artifact=shared_inspect,
            secrets_to_remove=[args.api_key],
        )
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_parent = run_dir / ".staging"
    if staging_parent.is_dir():
        try:
            staging_parent.rmdir()
        except OSError:
            # Another abandoned staging generation is retained only during a
            # non-resume execution; a later fingerprint-validated resume
            # removes the entire staging tree before evaluation.
            pass

    ordered_samples = [existing.get(query_id) or produced[query_id] for query_id in selected_query_ids]
    if (run_dir / "task-result.json").exists():
        task_result = read_json(run_dir / "task-result.json")
        if not isinstance(task_result, dict):
            raise ResultIntegrityError("task-result.json is not an object")
    else:
        all_inspect_records = [
            artifact_record(path, root=run_dir)
            for path in sorted((run_dir / "artifacts" / "inspect").glob("*.eval"))
        ] if (run_dir / "artifacts" / "inspect").is_dir() else []
        task_result = build_task_result(
            run_id=run_id,
            task=args.task,
            samples=ordered_samples,
            query_ids=selected_query_ids,
            model=model_info,
            backend=backend_info,
            data=data_info,
            code=code_info,
            evaluation_elapsed_seconds=evaluation_elapsed if pending else None,
            launcher_preflight_seconds=getattr(args, "launcher_preflight_seconds", None),
            inspect_log={"artifacts": all_inspect_records, "eval_count": len(all_inspect_records)},
            integrity={
                "algorithm": "sha256",
                "backend_manifest": artifact_record(backend_copy, root=run_dir),
                "backend_diagnostics": backend_diagnostics,
            },
        )
        write_task_result(run_dir, task_result)
    finalize_run(run_dir, manifest=manifest, task_result=task_result)
    update_task_index_run(index_path, run=_manifest_entry(manifest, "complete"))
    valid = sum(bool(sample["valid_output"]) for sample in ordered_samples)
    invalid = len(ordered_samples) - valid
    return Phase4Execution(
        run_id=run_id,
        run_dir=run_dir,
        query_ids=tuple(selected_query_ids),
        valid_samples=valid,
        invalid_samples=invalid,
        exit_code=0 if invalid == 0 else 1,
    )


__all__ = [
    "Phase4Execution",
    "build_and_publish_sample",
    "code_audit",
    "data_audit",
    "execute_phase4",
    "inspect_usage",
    "load_completed_sample",
    "load_query_texts",
    "preflight_bridge_ports",
]
