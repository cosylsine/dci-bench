"""Versioned DCI-Bench protocol contract shared with the Pi runner.

The JSON file next to this module is the single source of truth for benchmark
prompts, tool definitions, budgets, and output policy. Keep the Python names
below as a small compatibility surface for existing adapters and tasks; values
are loaded from the canonical contract rather than duplicated here.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


_CONTRACT_PATH = Path(__file__).with_name("contract.json")
with _CONTRACT_PATH.open(encoding="utf-8") as _handle:
    _CONTRACT: dict[str, Any] = json.load(_handle)


def _tuple(value: list[Any]) -> tuple[Any, ...]:
    return tuple(value)


BENCHMARK_CONTRACT_VERSION = str(_CONTRACT["benchmark_contract_version"])
PI_FINAL_RESULT_VERSION = str(_CONTRACT["pi_final_result_version"])
TRAJECTORY_SCHEMA_VERSION = str(_CONTRACT["trajectory_schema_version"])
TOP_K = int(_CONTRACT["top_k"])
MAX_AGENT_STEPS = int(_CONTRACT["max_agent_steps"])
MAX_FINAL_REPAIR_TURNS = int(_CONTRACT["max_final_repair_turns"])
SAMPLE_TIMEOUT_SECONDS = int(_CONTRACT["sample_timeout_seconds"])
TOOL_RESULT_MAX_CHARS = int(_CONTRACT["tool_result_max_chars"])
MAX_SCRATCH_BYTES = int(_CONTRACT["max_scratch_bytes"])
MAX_TOTAL_MODEL_TOKENS = int(_CONTRACT["max_total_model_tokens"])
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
REPAIR_PROMPT_TEMPLATE = str(_CONTRACT["repair_prompt_template"])


def protocol_contract() -> dict[str, Any]:
    """Return a deep-copied, JSON-serializable canonical protocol snapshot."""

    return deepcopy(_CONTRACT)
