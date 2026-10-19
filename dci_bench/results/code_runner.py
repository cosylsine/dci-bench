"""Independent online code run publication; never routes through document schemas."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time
from typing import Any, Callable, Sequence

from dci_bench.agents.code_agent import preflight_code_runtime
from dci_bench.backends.manifest import sanitize_base_url
from dci_bench.data.code_online import CodeSample, validate_workspace
from dci_bench.protocol.code import CODE_LABELS_VERSION, code_protocol_contract
from dci_bench.protocol.code_runtime import code_runtime_contract, runtime_contract_sha256
from dci_bench.results.phase4_runner import _classify_message, _inspect_limit_failure, _redact, _sample_error
from dci_bench.scoring.code import load_json, score_query, score_run, snapshot_identity


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def code_preflight(samples: Sequence[CodeSample], *, max_files: int, bridge_port: int,
                   max_concurrency: int) -> None:
    from dci_bench.tasks.mteb_llm_retrieval import BridgePortPool
    pool = BridgePortPool(bridge_port, max_concurrency)
    sockets = []
    try:
        for port in pool.ports:
            handle = socket.socket()
            sockets.append(handle)
            handle.bind(("127.0.0.1", port))
    finally:
        for handle in sockets:
            handle.close()
    seen = set()
    for sample in samples:
        key = (str(sample.workspace), snapshot_identity(sample.reference)["corpus_sha256"])
        if key not in seen:
            validate_workspace(sample.workspace, sample.reference["files"])
            preflight_code_runtime(sample.workspace, sample.reference["files"], max_files=max_files)
            seen.add(key)


def execute_code_run(
    samples: Sequence[CodeSample], *, run_dir: Path, model: str, base_url: str | None,
    api_key: str = "EMPTY", metric_ks: Sequence[int] = (5, 10, 20), max_files: int = 20,
    token_budgets: Sequence[int] = (4000, 8000, 16000, 32000), bridge_port: int = 13131,
    max_concurrency: int = 1, context_window: int = 65536,
    mock_script: Path | None = None, backend_manifest: dict[str, Any] | None = None,
    inspect_eval_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    from dci_bench.tasks.code_retrieval import code_retrieval, write_new_json
    run_dir = Path(run_dir).resolve()
    if base_url is not None:
        base_url = sanitize_base_url(base_url)
    for sample in samples:
        if run_dir.is_relative_to(sample.workspace.resolve()):
            raise ValueError("run output must be outside every candidate workspace")
        if sample.export_root is not None and run_dir.is_relative_to(sample.export_root.resolve()):
            raise ValueError("run output must be outside the immutable code export")
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists; choose a new path: {run_dir}")
    labels = {"schema_version": CODE_LABELS_VERSION, "queries": {s.query_id: s.reference for s in samples}}
    # Constructing validates selection/config but makes no model requests.
    task = code_retrieval(samples, output_dir=run_dir / "samples", metric_ks=metric_ks,
                          max_files=max_files, token_budgets=token_budgets, bridge_port=bridge_port,
                          max_concurrency=max_concurrency, model_context_window=context_window,
                          mock_script=mock_script)
    code_preflight(samples, max_files=max_files, bridge_port=bridge_port, max_concurrency=max_concurrency)
    runtime = code_runtime_contract(max_files)
    generation = runtime["generation_config"]
    scoring = code_protocol_contract()
    scoring.update(metric_ks=list(metric_ks), max_files=max_files, token_budgets=list(token_budgets))
    scoring["final_answer_schema"]["properties"]["ranked_files"]["maxItems"] = max_files
    run_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source_names = [
        "dci_bench/protocol/contract.json", "dci_bench/protocol/code.py", "dci_bench/protocol/code_runtime.py",
        "dci_bench/scoring/code.py", "dci_bench/data/code_online.py", "dci_bench/data/expansion_code.py",
        "dci_bench/agents/pi_agent.py", "dci_bench/agents/code_agent.py", "dci_bench/backends/manifest.py",
        "dci_bench/tasks/code_retrieval.py", "dci_bench/tasks/mteb_llm_retrieval.py",
        "dci_bench/results/code_runner.py", "dci_bench/results/phase4_runner.py",
        "scripts/run_code_task.py", "pi-dci/scripts/dci-run.ts",
    ]
    sources = {}
    for name in source_names:
        data = (root / name).read_bytes()
        destination = run_dir / "artifacts" / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(data)
        sources[name] = hashlib.sha256(data).hexdigest()
    manifest = {
        "schema_version": "dci-code-online-run-v1", "evaluation_track": "code",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model, "base_url": base_url, "backend_manifest": backend_manifest,
        "generation_config": generation, "context_window": context_window,
        "max_concurrency": max_concurrency, "bridge_port": bridge_port,
        "metric_ks": list(metric_ks), "max_files": max_files, "token_budgets": list(token_budgets),
        "runtime_contract": runtime, "runtime_contract_sha256": runtime_contract_sha256(runtime),
        "scoring_contract": scoring, "scoring_contract_sha256": _hash(scoring), "labels_sha256": _hash(labels),
        "source_sha256": sources, "mock": mock_script is not None,
        "mock_script_sha256": hashlib.sha256(mock_script.read_bytes()).hexdigest() if mock_script else None,
        "queries": [{"query_id": s.query_id, "query": s.query, "task": s.reference["task"],
                     "directory": s.directory_name, "workspace": str(s.workspace),
                     **snapshot_identity(s.reference)} for s in samples],
    }
    manifest["fingerprint"] = _hash({k: v for k, v in manifest.items() if k != "created_at"})
    write_new_json(run_dir / "run-manifest.json", manifest)
    if inspect_eval_fn is None:
        from inspect_ai import eval as inspect_eval_fn
    logs, evaluation_error = [], None
    start = time.monotonic()
    try:
        logs = list(inspect_eval_fn(
            task, model=model, model_base_url=base_url,
            model_args={"api_key": api_key, "responses_api": False} if mock_script is None else {},
            log_dir=str(run_dir / "artifacts" / "inspect"), display="plain",
            max_samples=max_concurrency, fail_on_error=False, score_on_error=True,
            temperature=generation["temperature"], max_tokens=generation["max_tokens"], seed=generation["seed"],
            extra_body={"chat_template_kwargs": {"enable_thinking": generation["enable_thinking"]}},
        ))
    except Exception as exc:
        evaluation_error = _redact(str(exc), (api_key,))
    elapsed = time.monotonic() - start
    if evaluation_error is None and (len(logs) != 1 or getattr(logs[0], "status", None) != "success"):
        evaluation_error = _redact("Inspect task did not complete: " + ", ".join(
            str(getattr(log, "error", None) or getattr(log, "status", "missing status")) for log in logs
        ), (api_key,))
    inspect_samples = {}
    for log in logs:
        for row in getattr(log, "samples", None) or []:
            if str(row.id) in inspect_samples:
                raise ValueError(f"duplicate Inspect sample: {row.id}")
            inspect_samples[str(row.id)] = row
    predictions = {}
    infrastructure_failures = []
    failures = {}
    for sample in samples:
        directory = run_dir / "samples" / sample.directory_name
        raw_prediction, raw_score = directory / "scorer-prediction.json", directory / "scorer-score.json"
        prediction = load_json(raw_prediction.read_text(encoding="utf-8")) if raw_prediction.exists() else None
        expected = score_query(prediction, sample.reference, metric_ks=metric_ks,
                               max_files=max_files, token_budgets=token_budgets)
        if raw_score.exists() and load_json(raw_score.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"online scorer/result mismatch: {sample.query_id}")

        # Terminal Inspect events are authoritative even when a scorer already
        # saved a valid answer. Preserve raw artifacts; publish a separate result.
        logged = inspect_samples.get(sample.query_id)
        failure = _inspect_limit_failure(logged, secrets_to_remove=(api_key,))
        error = _sample_error(logged)
        if failure is None and error is not None:
            failure = {"kind": _classify_message(error), "message": _redact(error, (api_key,))}
        if failure is None and (logged is None or not raw_prediction.exists() or not raw_score.exists()):
            failure = {"kind": "infrastructure_error", "message": evaluation_error or "Inspect produced no scored sample"}
        if failure is None and not expected["valid_output"]:
            final_path = directory / "final.json"
            final = load_json(final_path.read_text(encoding="utf-8")) if final_path.exists() else {}
            pi_failure = final.get("failure") if isinstance(final, dict) else None
            reason = expected["failure_reason"] or "invalid code prediction"
            failure = {"kind": str(pi_failure.get("kind") or "agent_error") if isinstance(pi_failure, dict) else "invalid_json",
                       "message": _redact(reason, (api_key,))}
        if failure is not None:
            failures[sample.query_id] = failure
            prediction = {**snapshot_identity(sample.reference),
                          "failure_reason": failure["kind"] + ": " + failure["message"]}
        if failure is not None and failure["kind"] == "infrastructure_error":
            infrastructure_failures.append(sample.query_id)
        predictions[sample.query_id] = prediction
        expected = score_query(prediction, sample.reference,
                               metric_ks=metric_ks, max_files=max_files, token_budgets=token_budgets)
        write_new_json(directory / "prediction.json", prediction)
        write_new_json(directory / "score.json", expected)
    report = score_run(predictions, labels, metric_ks=metric_ks, max_files=max_files, token_budgets=token_budgets)
    report.update(
        run_schema_version="dci-code-online-run-v1", evaluation_elapsed_seconds=elapsed,
        infrastructure_failures=infrastructure_failures, evaluation_error=evaluation_error,
        failures=failures, failure_counts=dict(Counter(f["kind"] for f in failures.values())),
        run_fingerprint=manifest["fingerprint"],
        status=("infrastructure_error" if evaluation_error or infrastructure_failures else
                "complete_with_sample_failures" if report["invalid_outputs"] else "complete"),
    )
    write_new_json(run_dir / "predictions.json", predictions)
    write_new_json(run_dir / "code-result.json", report)
    artifacts = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("run artifact must not be a symlink")
        if path.is_file():
            artifacts[path.relative_to(run_dir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_new_json(run_dir / "_COMPLETE.json", {
        "schema_version": "dci-code-online-complete-v1", "evaluation_track": "code",
        "run_fingerprint": manifest["fingerprint"], "status": report["status"],
        "num_queries": report["num_queries"], "valid_outputs": report["valid_outputs"],
        "invalid_outputs": report["invalid_outputs"], "artifact_sha256": artifacts,
    })
    return report
