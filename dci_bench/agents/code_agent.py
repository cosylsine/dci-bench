"""Code-specific Pi adapter; only public candidate metadata crosses into Pi."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from dci_bench.agents.pi_agent import PiRunResult, pi_dci_root, run_pi_dci
from dci_bench.protocol.code import DEFAULT_CODE_MAX_FILES
from dci_bench.protocol.code_runtime import code_runtime_contract, runtime_contract_sha256


@contextmanager
def _configuration(files: dict[str, Any], max_files: int):
    contract = code_runtime_contract(max_files)
    with tempfile.TemporaryDirectory(prefix="dci-code-runtime-") as temporary:
        root = Path(temporary)
        contract_path, files_path = root / "contract.json", root / "files.json"
        contract_path.write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")
        files_path.write_text(json.dumps(files, ensure_ascii=False), encoding="utf-8")
        env = {
            "DCI_EVALUATION_TRACK": "code",
            "DCI_PROTOCOL_PROFILE": "code-v1",
            "DCI_CODE_RUNTIME_CONTRACT": str(contract_path),
            "DCI_PROTOCOL_CONTRACT_SHA256": runtime_contract_sha256(contract),
        }
        yield contract, files_path, env


def preflight_code_runtime(workspace: Path, files: dict[str, Any], *, max_files: int = DEFAULT_CODE_MAX_FILES) -> None:
    """Validate the actual Pi sandbox and candidate bytes before model startup."""
    with _configuration(files, max_files) as (_, files_path, env):
        completed = subprocess.run(
            ["node", "--import", "tsx", "scripts/dci-run.ts", "--preflight-only",
             "--workspace", str(workspace.resolve()), "--code-files", str(files_path)],
            cwd=pi_dci_root(), env={**os.environ, **env}, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Code Pi preflight failed: {completed.stderr or completed.stdout}")


def run_pi_code(*, files: dict[str, Any], max_files: int = DEFAULT_CODE_MAX_FILES, **kwargs: Any) -> PiRunResult:
    """Run the shared Pi lifecycle with the independent code prompt/output/tools."""
    if "extra_env" in kwargs or any(key.startswith("_code_") for key in kwargs):
        raise ValueError("code runtime configuration is controlled by the host adapter")
    with _configuration(files, max_files) as (contract, files_path, env):
        kwargs.setdefault("max_agent_steps", contract["max_agent_steps"])
        kwargs.setdefault("timeout_seconds", contract["sample_timeout_seconds"])
        return run_pi_dci(**kwargs, extra_env=env, _code_runtime=contract, _code_files_path=files_path)
