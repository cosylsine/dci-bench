"""Versioned DCI-Bench protocol contract shared with the Pi runner.

The JSON file next to this module is the single source of truth for benchmark
prompts, tool definitions, budgets, and output policy. Keep the Python names
below as a small compatibility surface for existing adapters and tasks; values
are loaded from the canonical contract rather than duplicated here.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


_CONTRACT_PATH = Path(__file__).with_name("contract.json")
with _CONTRACT_PATH.open(encoding="utf-8") as _handle:
    _CONTRACT: dict[str, Any] = json.load(_handle)

# A profile is selected once, before importing the task/Inspect stack. Each
# pilot arm runs in a fresh process and Pi inherits the same selection. The
# v3 remains the unbudgeted control arm; all arms share the current ID contract.
PROTOCOL_PROFILE = os.environ.get("DCI_PROTOCOL_PROFILE", "v3")
if PROTOCOL_PROFILE != "v3":
    with _CONTRACT_PATH.with_name("pilot_profiles.json").open(encoding="utf-8") as _handle:
        _profiles = json.load(_handle)
    if PROTOCOL_PROFILE.startswith("_") or PROTOCOL_PROFILE not in _profiles:
        raise ValueError(f"Unknown DCI_PROTOCOL_PROFILE: {PROTOCOL_PROFILE!r}")
    _profile = _profiles[PROTOCOL_PROFILE]
    _CONTRACT["benchmark_contract_version"] = _profile.get("benchmark_contract_version", "dci-mvp-v3.3-pilot")
    _CONTRACT["protocol_profile"] = PROTOCOL_PROFILE
    _CONTRACT["generation_config"].update(_profile.get("generation_config", {}))
    if _profile.get("budget_control"):
        _CONTRACT["budget_control"] = deepcopy(_profiles[_profile["budget_control"]]["policy"])
        _CONTRACT["budget_control"].update(deepcopy(_profile.get("budget_control_overrides", {})))
    if _profile.get("search_policy"):
        _CONTRACT["search_policy"] = deepcopy(_profiles[_profile["search_policy"]]["search_policy"])
        if _CONTRACT_PATH.with_name("stable-rg.conf").read_text(encoding="utf-8") != _CONTRACT["search_policy"]["ripgrep_config"]:
            raise ValueError("stable-rg.conf does not match the protocol search policy")

CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(_CONTRACT, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _tuple(value: list[Any]) -> tuple[Any, ...]:
    return tuple(value)


BENCHMARK_CONTRACT_VERSION = str(_CONTRACT["benchmark_contract_version"])
PI_FINAL_RESULT_VERSION = str(_CONTRACT["pi_final_result_version"])
TRAJECTORY_SCHEMA_VERSION = str(_CONTRACT["trajectory_schema_version"])
TOP_K = int(_CONTRACT["top_k"])
MAX_AGENT_STEPS = int(_CONTRACT["max_agent_steps"])
MAX_FINAL_REPAIR_TURNS = int(_CONTRACT["max_final_repair_turns"])
SAMPLE_TIMEOUT_SECONDS = int(_CONTRACT["sample_timeout_seconds"])
TOOL_RESULT_MAX_BYTES = int(_CONTRACT["tool_result_max_bytes"])
MAX_SCRATCH_BYTES = int(_CONTRACT["max_scratch_bytes"])
MAX_TOTAL_MODEL_TOKENS = int(_CONTRACT["max_total_model_tokens"])
GENERATION_CONFIG: dict[str, Any] = deepcopy(_CONTRACT["generation_config"])
MAX_MODEL_OUTPUT_TOKENS = int(_CONTRACT["max_model_output_tokens"])
MAX_FINAL_OUTPUT_TOKENS = int(_CONTRACT["max_final_output_tokens"])
MAX_CORPUS_DOCUMENT_BYTES = int(_CONTRACT["max_corpus_document_bytes"])

TOOL_ALLOWLIST = _tuple(_CONTRACT["tool_allowlist"])
TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = tuple(
    deepcopy(item) for item in _CONTRACT["tool_definitions"]
)
DISABLED_CAPABILITIES = _tuple(_CONTRACT["disabled_capabilities"])
FAILURE_KINDS = _tuple(_CONTRACT["failure_kinds"])
FINAL_ANSWER_SCHEMA: dict[str, Any] = deepcopy(_CONTRACT["final_answer_schema"])
CONTEXT_POLICY: dict[str, Any] = deepcopy(_CONTRACT["context_policy"])
TERMINATION_CONDITION: dict[str, Any] = deepcopy(_CONTRACT["termination_condition"])
INSPECT_OUTPUT_CONTRACT: dict[str, Any] = deepcopy(_CONTRACT["inspect_output_contract"])
SYSTEM_PROMPT = str(_CONTRACT["system_prompt"])
QUERY_PROMPT_TEMPLATE = str(_CONTRACT["query_prompt_template"])
TASK_INSTRUCTION_TEMPLATE = str(_CONTRACT["task_instruction_template"])
REPAIR_PROMPT_TEMPLATE = str(_CONTRACT["repair_prompt_template"])


def protocol_contract(
    *,
    task_name: str | None = None,
    task_instruction: str | None = None,
) -> dict[str, Any]:
    """Return a canonical protocol snapshot, optionally resolved for one task.

    The base contract remains task-neutral. A resolved run records the trusted
    registry instruction here so prompt behavior participates in the immutable
    run fingerprint instead of being hidden in launcher state.
    """

    snapshot = deepcopy(_CONTRACT)
    if task_name is None and task_instruction is None:
        return snapshot
    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError("task_name must be a non-empty string for a task-aware contract")
    if not isinstance(task_instruction, str):
        raise ValueError("task_instruction must be a string")
    snapshot["task_context"] = {
        "task_name": task_name,
        "instruction": task_instruction.strip(),
    }
    return snapshot
