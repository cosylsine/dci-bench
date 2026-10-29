"""Subprocess adapter for the modified Pi DCI runner."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dci_bench.protocol.contracts import (
    BENCHMARK_CONTRACT_VERSION,
    CONTRACT_SHA256,
    CONTEXT_POLICY,
    MAX_AGENT_STEPS,
    MAX_TOTAL_MODEL_TOKENS,
    PI_FINAL_RESULT_VERSION,
    PROTOCOL_PROFILE,
    SAMPLE_TIMEOUT_SECONDS,
    TRAJECTORY_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class PiRunResult:
    ranked_doc_ids: list[str]
    valid_output: bool
    failure_reason: str | None
    raw_final_text: str
    agent_steps: int
    tool_calls: int
    final_output_adapter: str | None
    repair_attempts: int
    repair_reasons: list[str]
    body_evidence: dict[str, Any]
    inspect_sandbox: str
    tool_sandbox: str
    output_path: Path
    trace_path: Path | None
    contract_version: str
    failure_kind: str | None
    failure: dict[str, Any] | None
    usage_summary: dict[str, Any]
    code_output: dict[str, Any] | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pi_dci_root() -> Path:
    return repo_root() / "pi-dci"


def run_pi_dci(
    *,
    query: str,
    workspace: Path,
    output_path: Path,
    trace_path: Path | None = None,
    mock_script: Path | None = None,
    openai_base_url: str = "http://localhost:13131/v1",
    openai_api_key: str = "inspect",
    model: str = "inspect",
    timeout_seconds: float = SAMPLE_TIMEOUT_SECONDS,
    max_agent_steps: int = MAX_AGENT_STEPS,
    context_window: int = int(CONTEXT_POLICY.get("context_window", MAX_TOTAL_MODEL_TOKENS)),
    task_instruction: str = "",
    extra_env: dict[str, str] | None = None,
    _code_runtime: dict[str, Any] | None = None,
    _code_files_path: Path | None = None,
) -> PiRunResult:
    # Code calls enter through code_agent.run_pi_code. Ordinary callers retain
    # the document contract, even if unrelated code-mode env vars are present.
    expected_version = BENCHMARK_CONTRACT_VERSION
    expected_hash = CONTRACT_SHA256
    expected_profile = PROTOCOL_PROFILE
    if _code_runtime is not None:
        from dci_bench.protocol.code_runtime import runtime_contract_sha256
        if _code_files_path is None:
            raise ValueError("code runtime requires a host candidate manifest")
        expected_version = _code_runtime["benchmark_contract_version"]
        expected_profile = _code_runtime["protocol_profile"]
        expected_hash = runtime_contract_sha256(_code_runtime)
    workspace = workspace.resolve()
    output_path = output_path.resolve()
    trace_path = trace_path.resolve() if trace_path is not None else None
    mock_script = mock_script.resolve() if mock_script is not None else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(query)
        query_file = Path(handle.name)
    instruction_file: Path | None = None
    if task_instruction.strip():
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", delete=False
        ) as handle:
            handle.write(task_instruction.strip())
            instruction_file = Path(handle.name)

    try:
        command = [
            "node",
            "--import",
            "tsx",
            "scripts/dci-run.ts",
            "--query-file",
            str(query_file),
            "--workspace",
            str(workspace),
            "--output",
            str(output_path),
            "--trace",
            str(trace_path or output_path.with_suffix(".trace.json")),
            "--model",
            model,
            "--openai-base-url",
            openai_base_url,
            "--openai-api-key",
            openai_api_key,
            "--max-agent-steps",
            str(max_agent_steps),
            "--context-window",
            str(context_window),
            "--timeout-seconds",
            str(timeout_seconds),
        ]
        if mock_script is not None:
            command.extend(["--mock-script", str(mock_script)])
        if instruction_file is not None:
            command.extend(["--task-instruction-file", str(instruction_file)])
        if _code_files_path is not None:
            command.extend(["--code-files", str(_code_files_path)])

        env = os.environ.copy()
        env.update(
            {
                "OPENAI_BASE_URL": openai_base_url,
                "OPENAI_API_KEY": openai_api_key,
                "PI_DCI_BENCH_MODE": "1",
            }
        )
        if extra_env:
            env.update(extra_env)
        if _code_runtime is None and env.get("DCI_PROTOCOL_PROFILE", "v3") != PROTOCOL_PROFILE:
            raise ValueError("Pi subprocess profile must match the parent protocol profile")
        env["DCI_EVALUATION_TRACK"] = "code" if _code_runtime is not None else "document"
        if _code_runtime is None:
            env.pop("DCI_CODE_RUNTIME_CONTRACT", None)
        env["DCI_PROTOCOL_PROFILE"] = expected_profile
        env["DCI_PROTOCOL_CONTRACT_SHA256"] = expected_hash

        # Pass the same deadline to Pi before Node/tsx startup. Import/preflight
        # time must not grant the agent a new timeout window.
        command.extend(["--deadline-at-ms", str(time.time() * 1000 + timeout_seconds * 1000)])
        process = subprocess.Popen(
            command,
            cwd=pi_dci_root(),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            # The sample owns this process group. Give Pi time to abort its
            # HTTP/tool request and publish the partial trace before escalation.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
        finally:
            # Also reap descendants that outlived the process group leader.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if timed_out:
            trace_file = trace_path or output_path.with_suffix(".trace.json")
            try:
                partial = json.loads(trace_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                partial = {"events": [], "messages": [], "trace_unavailable": "timeout before first snapshot"}
            failure = {"kind": "timeout", "message": f"sample wall deadline reached ({timeout_seconds}s)"}
            partial.update(
                schema_version=TRAJECTORY_SCHEMA_VERSION, failure=failure, partial=True,
                contract_version=expected_version,
                protocol_profile=expected_profile, base_contract_sha256=expected_hash,
            )
            temporary = trace_file.with_suffix(".timeout.tmp")
            temporary.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(trace_file)
            # Preserve any orderly final fields, but never accept an answer
            # that arrived during cancellation grace after the deadline.
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = {}
            payload.update(
                contract_version=expected_version,
                protocol_profile=expected_profile,
                base_contract_sha256=expected_hash,
                control_events=partial.get("control_events", payload.get("control_events", [])),
                pi_result_version=PI_FINAL_RESULT_VERSION,
                ranked_doc_ids=[], valid_output=False, failure=failure,
                failure_reason=failure["message"],
                usage_summary=partial.get("usage_summary", {"available": False}),
                body_evidence=partial.get("body_evidence", payload.get("body_evidence", {})),
                inspect_sandbox=partial.get("inspect_sandbox", "unavailable"),
                tool_sandbox=partial.get("tool_sandbox", "unavailable"),
            )
            if _code_runtime is not None:
                payload.update(evaluation_track="code", code_output=None)
            temporary = output_path.with_suffix(".timeout.tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(output_path)
        elif process.returncode not in (0, 2):
            raise RuntimeError(
                "pi-dci runner failed with exit code "
                f"{process.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )

        payload: dict[str, Any] = json.loads(output_path.read_text(encoding="utf-8"))
        if payload.get("base_contract_sha256") != expected_hash:
            raise RuntimeError("Pi runner returned a different resolved protocol contract digest")
        contract_version = payload.get("contract_version")
        if contract_version != expected_version:
            raise RuntimeError(
                "Pi runner returned an unsupported protocol contract: "
                f"expected {expected_version!r}, got {contract_version!r}"
            )
        pi_result_version = payload.get("pi_result_version")
        if pi_result_version != PI_FINAL_RESULT_VERSION:
            raise RuntimeError(
                "Pi runner returned an unsupported Pi final result version: "
                f"expected {PI_FINAL_RESULT_VERSION!r}, got {pi_result_version!r}"
            )
        inspect_sandbox = payload.get("inspect_sandbox")
        tool_sandbox = payload.get("tool_sandbox")
        if not timed_out and (inspect_sandbox != "local" or tool_sandbox != "bubblewrap"):
            raise RuntimeError(
                "Pi runner did not attest the required sandbox boundary: "
                f"inspect_sandbox={inspect_sandbox!r}, tool_sandbox={tool_sandbox!r}"
            )
        failure_payload = payload.get("failure")
        if failure_payload is not None and not isinstance(failure_payload, dict):
            raise RuntimeError("Pi runner failure field must be an object or null")
        usage_payload = payload.get("usage_summary", payload.get("usage", {}))
        if not isinstance(usage_payload, dict):
            raise RuntimeError("Pi runner usage_summary field must be an object")
        return PiRunResult(
            ranked_doc_ids=list(payload.get("ranked_doc_ids", [])),
            valid_output=bool(payload.get("valid_output", False)),
            failure_reason=payload.get("failure_reason"),
            raw_final_text=str(payload.get("raw_final_text", "")),
            agent_steps=int(payload.get("agent_steps", 0)),
            tool_calls=int(payload.get("tool_calls", 0)),
            final_output_adapter=payload.get("final_output_adapter"),
            repair_attempts=int(payload.get("repair_attempts", 0)),
            repair_reasons=list(payload.get("repair_reasons", [])),
            body_evidence=dict(payload.get("body_evidence", {})),
            inspect_sandbox=inspect_sandbox,
            tool_sandbox=tool_sandbox,
            output_path=output_path,
            trace_path=trace_path or output_path.with_suffix(".trace.json"),
            contract_version=str(contract_version),
            failure_kind=(
                str(failure_payload["kind"])
                if isinstance(failure_payload, dict) and failure_payload.get("kind") is not None
                else None
            ),
            failure=failure_payload,
            usage_summary=usage_payload,
            code_output=payload.get("code_output") if _code_runtime is not None else None,
        )
    finally:
        query_file.unlink(missing_ok=True)
        if instruction_file is not None:
            instruction_file.unlink(missing_ok=True)
