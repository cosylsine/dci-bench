"""Subprocess adapter for the modified Pi DCI runner."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dci_bench.protocol.contracts import (
    BENCHMARK_CONTRACT_VERSION,
    MAX_AGENT_STEPS,
    PI_FINAL_RESULT_VERSION,
    SAMPLE_TIMEOUT_SECONDS,
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
    timeout_seconds: int = SAMPLE_TIMEOUT_SECONDS,
    max_agent_steps: int = MAX_AGENT_STEPS,
    extra_env: dict[str, str] | None = None,
) -> PiRunResult:
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

    try:
        command = [
            "npm",
            "--prefix",
            str(pi_dci_root()),
            "exec",
            "--",
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
        ]
        if mock_script is not None:
            command.extend(["--mock-script", str(mock_script)])

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

        completed = subprocess.run(
            command,
            cwd=pi_dci_root(),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode not in (0, 2):
            raise RuntimeError(
                "pi-dci runner failed with exit code "
                f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        payload: dict[str, Any] = json.loads(output_path.read_text(encoding="utf-8"))
        contract_version = payload.get("contract_version")
        if contract_version != BENCHMARK_CONTRACT_VERSION:
            raise RuntimeError(
                "Pi runner returned an unsupported protocol contract: "
                f"expected {BENCHMARK_CONTRACT_VERSION!r}, got {contract_version!r}"
            )
        pi_result_version = payload.get("pi_result_version")
        if pi_result_version != PI_FINAL_RESULT_VERSION:
            raise RuntimeError(
                "Pi runner returned an unsupported Pi final result version: "
                f"expected {PI_FINAL_RESULT_VERSION!r}, got {pi_result_version!r}"
            )
        inspect_sandbox = payload.get("inspect_sandbox")
        tool_sandbox = payload.get("tool_sandbox")
        if inspect_sandbox != "local" or tool_sandbox != "bubblewrap":
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
        )
    finally:
        query_file.unlink(missing_ok=True)
