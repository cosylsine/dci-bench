"""Frozen Phase 0 contracts for the DCI-Bench MVP.

These values are intentionally boring and explicit. Later Pi and Inspect
integration should import this module instead of redefining benchmark policy.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

BENCHMARK_CONTRACT_VERSION = "dci-mvp-v0"

TOP_K = 10
MAX_AGENT_STEPS = 1200
SAMPLE_TIMEOUT_SECONDS = 600
TOOL_RESULT_MAX_CHARS = 20_000
MAX_TOTAL_MODEL_TOKENS = 128_000
MAX_FINAL_OUTPUT_TOKENS = 1_024

TOOL_ALLOWLIST = ("read", "bash")
TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "read",
        "description": "Read UTF-8 text from a file under the mounted workspace.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["path"],
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "default": TOOL_RESULT_MAX_CHARS},
            },
        },
        "result_max_chars": TOOL_RESULT_MAX_CHARS,
    },
    {
        "name": "bash",
        "description": "Run a local shell command without network access.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["command"],
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 60, "default": 10},
            },
        },
        "result_max_chars": TOOL_RESULT_MAX_CHARS,
    },
)
DISABLED_CAPABILITIES = (
    "web",
    "retriever",
    "mcp",
    "skills",
    "sub_agents",
    "project_memory",
    "user_memory",
    "interactive_human_clarification",
    "provider_specific_agent_enhancement",
)

FINAL_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ranked_doc_ids"],
    "properties": {
        "ranked_doc_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": TOP_K,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        }
    },
}

CONTEXT_POLICY: dict[str, Any] = {
    "compaction": "disabled",
    "max_total_model_tokens": MAX_TOTAL_MODEL_TOKENS,
    "max_final_output_tokens": MAX_FINAL_OUTPUT_TOKENS,
}

TERMINATION_CONDITION: dict[str, Any] = {
    "success": "first assistant message that parses as the final answer JSON schema",
    "failure": (
        "timeout, max_agent_steps, max_total_model_tokens, tool error exhaustion, "
        "or model stop without a valid final answer"
    ),
}

INSPECT_OUTPUT_CONTRACT: dict[str, Any] = {
    "required_per_sample_fields": (
        "task",
        "query_id",
        "ranked_doc_ids",
        "valid_output",
        "ndcg_at_10",
        "recall_at_10",
        "agent_steps",
        "tool_calls",
        "model_input_tokens",
        "model_output_tokens",
        "wall_time_seconds",
        "failure_reason",
        "trace_path",
    ),
    "host_only_fields": ("qrels", "gold_doc_ids"),
}

SYSTEM_PROMPT = """You are the DCI-Bench retrieval agent.

Your task is to search the local corpus for documents relevant to the user's query.
The corpus is available as markdown files under /workspace/corpus. You may inspect
files with the provided read and bash tools. Do not use network access, web search,
external retrievers, embeddings, hidden labels, memory, skills, MCP tools, or
sub-agents.

You must inspect corpus document content before giving the final answer. A list
of filenames or file ids is not evidence. Use full-text search commands such as
rg -n -i or grep -Rin over important query keywords and synonyms, then read the
most promising document files.

Return only JSON matching this schema:
{"ranked_doc_ids":["doc_id_1","doc_id_2","doc_id_3"]}

Rules:
- Return at most 10 document ids.
- Rank the most relevant document first.
- Use the original document id, recovered from the corpus filename by removing
  the .md suffix and URL percent-decoding it when needed.
- Do not include explanations, markdown fences, scores, or extra keys.
"""


def protocol_contract() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the fixed MVP protocol."""

    return {
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "final_answer_schema": deepcopy(FINAL_ANSWER_SCHEMA),
        "tool_allowlist": list(TOOL_ALLOWLIST),
        "tool_definitions": deepcopy(TOOL_DEFINITIONS),
        "disabled_capabilities": list(DISABLED_CAPABILITIES),
        "top_k": TOP_K,
        "max_agent_steps": MAX_AGENT_STEPS,
        "sample_timeout_seconds": SAMPLE_TIMEOUT_SECONDS,
        "tool_result_max_chars": TOOL_RESULT_MAX_CHARS,
        "max_total_model_tokens": MAX_TOTAL_MODEL_TOKENS,
        "max_final_output_tokens": MAX_FINAL_OUTPUT_TOKENS,
        "context_policy": deepcopy(CONTEXT_POLICY),
        "termination_condition": deepcopy(TERMINATION_CONDITION),
        "system_prompt": SYSTEM_PROMPT,
        "inspect_output_contract": deepcopy(INSPECT_OUTPUT_CONTRACT),
    }
