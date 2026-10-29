"""Dataset-neutral candidate-pool tiers used by document protocol profiles.

The JSON file is deliberately metadata-only.  It describes selector caps,
budget limits, ID examples, and the offline retriever plan; this module never
loads a model or performs retrieval.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any


DOCUMENT_POOLS_PATH = Path(__file__).with_name("document_pools.json")
DOCUMENT_POOLS_V2_PATH = Path(__file__).with_name("document_pools_v2.json")
_PROFILE_PREFIX = "document-pool-v1-"
_V2_PROFILE_PREFIX = "document-pool-v2-"
_V2_PROFILE_ALIASES = {"lite": "document-pool-v2-lite", "full": "document-pool-v2-full"}
_V2_BUDGET_VALUES = (32000, 64000, 96000)
_TIER_FIELDS = (
    "capacity",
    "max_total_model_tokens",
    "reserve_tokens",
    "context_window",
    "max_model_output_tokens",
    "max_final_output_tokens",
    "sample_timeout_seconds",
)
_V2_TIER_FIELDS = _TIER_FIELDS + (
    "max_search_turns",
    "max_tool_invocations",
    "max_final_repair_turns",
    "max_agent_steps",
    "tool_result_max_bytes",
)


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"document pool {field} must be a positive integer")
    return value


def load_document_pools(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the central candidate-pool metadata."""

    source = DOCUMENT_POOLS_PATH if path is None else Path(path)
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read document pool config: {source}") from exc
    if not isinstance(config, dict):
        raise ValueError("document pool config must be an object")
    if config.get("schema_version") != "document-pool-v1":
        raise ValueError("unsupported document pool schema_version")
    if config.get("scope") != "dataset-neutral":
        raise ValueError("document pool config must be dataset-neutral")
    id_policy = config.get("id_policy")
    if not isinstance(id_policy, dict):
        raise ValueError("document pool id_policy must be an object")
    if id_policy.get("version") != "short-document-id-v1":
        raise ValueError("document pool id_policy version must declare short document aliases")
    if id_policy.get("filename_to_id_example") != "d000001.md -> d000001":
        raise ValueError("document pool id_policy must declare d000001.md -> d000001")
    aliases = id_policy.get("aliases")
    if not isinstance(aliases, list) or "d000001" not in aliases:
        raise ValueError("document pool id_policy must declare alias d000001")
    declared_aliases = id_policy.get("declared_aliases")
    if not isinstance(declared_aliases, dict) or declared_aliases.get("d000001.md") != "d000001":
        raise ValueError("document pool id_policy must declare d000001.md alias")

    retriever = config.get("retriever")
    if not isinstance(retriever, dict):
        raise ValueError("document pool retriever metadata must be an object")
    required_retriever = {
        "model_path": "/mnt/afs/share/embedding_models/KaLM-embedding-multilingual-mini-instruct-v2_5",
        "trust_remote_code": True,
        "offline": True,
        "planning_only": True,
        "load_model": False,
    }
    for field, expected in required_retriever.items():
        if retriever.get(field) != expected:
            raise ValueError(f"document pool retriever metadata has invalid {field}")

    prompt_policy = config.get("prompt_policy")
    if not isinstance(prompt_policy, dict):
        raise ValueError("document pool prompt_policy must be an object")
    for field in ("initial_prompt_template", "final_prompt", "final_repair_prompt"):
        if not isinstance(prompt_policy.get(field), str) or not prompt_policy[field]:
            raise ValueError(f"document pool prompt_policy requires {field}")
    if "{max_total_model_tokens}" not in prompt_policy["initial_prompt_template"]:
        raise ValueError("document pool initial prompt must expose max_total_model_tokens")
    if "d000001.md -> d000001" not in prompt_policy["final_prompt"]:
        raise ValueError("document pool final prompt must declare the short ID example")
    if "d000001.md -> d000001" not in prompt_policy["final_repair_prompt"]:
        raise ValueError("document pool final repair prompt must declare the short ID example")

    profiles = config.get("profiles")
    tiers = config.get("tiers")
    if not isinstance(profiles, dict) or not isinstance(tiers, dict):
        raise ValueError("document pool config requires profiles and tiers objects")
    if set(profiles) != {
        "document-pool-v1-small",
        "document-pool-v1-medium",
        "document-pool-v1-large",
    }:
        raise ValueError("document pool profiles must declare small, medium, and large")
    if set(tiers) != {"small", "medium", "large"}:
        raise ValueError("document pool tiers must be exactly small, medium, and large")
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict) or profile.get("tier") not in tiers:
            raise ValueError(f"document pool profile {profile_name!r} has an invalid tier")
    for tier_name, tier in tiers.items():
        if not isinstance(tier, dict):
            raise ValueError(f"document pool tier {tier_name!r} must be an object")
        for field in _TIER_FIELDS:
            _positive_int(tier.get(field), f"{tier_name}.{field}")
        if tier["context_window"] != 65536:
            raise ValueError("all document pool tiers must use a 65536 context window")
        if tier["max_model_output_tokens"] != 1024 or tier["max_final_output_tokens"] != 1024:
            raise ValueError("all document pool tiers must use 1024 response limits")
        if tier["sample_timeout_seconds"] != 600:
            raise ValueError("all document pool tiers must use the 600 second timeout")
        if tier["reserve_tokens"] > tier["max_total_model_tokens"]:
            raise ValueError(f"document pool tier {tier_name!r} reserve exceeds total budget")
    return config


def _v2_budget_override() -> int:
    raw = os.environ.get("DCI_DOCUMENT_TOKEN_BUDGET")
    if raw is None:
        return 64000
    if raw not in {str(value) for value in _V2_BUDGET_VALUES}:
        raise ValueError("DCI_DOCUMENT_TOKEN_BUDGET must be one of 32000, 64000, or 96000")
    return int(raw)


def load_document_pools_v2(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the additive Lite200/Full1000 protocol metadata."""

    source = DOCUMENT_POOLS_V2_PATH if path is None else Path(path)
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read document pool v2 config: {source}") from exc
    if not isinstance(config, dict):
        raise ValueError("document pool v2 config must be an object")
    if config.get("schema_version") != "document-pool-v2":
        raise ValueError("unsupported document pool v2 schema_version")
    if config.get("scope") != "dataset-neutral":
        raise ValueError("document pool v2 config must be dataset-neutral")
    id_policy = config.get("id_policy")
    if not isinstance(id_policy, dict):
        raise ValueError("document pool v2 id_policy must be an object")
    if id_policy.get("version") != "short-document-id-v1":
        raise ValueError("document pool v2 id_policy version must declare short document aliases")
    if id_policy.get("filename_to_id_example") != "d000001.md -> d000001":
        raise ValueError("document pool v2 id_policy must declare d000001.md -> d000001")
    aliases = id_policy.get("aliases")
    if not isinstance(aliases, list) or "d000001" not in aliases:
        raise ValueError("document pool v2 id_policy must declare alias d000001")
    declared_aliases = id_policy.get("declared_aliases")
    if not isinstance(declared_aliases, dict) or declared_aliases.get("d000001.md") != "d000001":
        raise ValueError("document pool v2 id_policy must declare d000001.md alias")

    retriever = config.get("retriever")
    if not isinstance(retriever, dict):
        raise ValueError("document pool v2 retriever metadata must be an object")
    required_retriever = {
        "model_path": "/mnt/afs/share/embedding_models/KaLM-embedding-multilingual-mini-instruct-v2_5",
        "trust_remote_code": True,
        "offline": True,
        "planning_only": True,
        "load_model": False,
    }
    for field, expected in required_retriever.items():
        if retriever.get(field) != expected:
            raise ValueError(f"document pool v2 retriever metadata has invalid {field}")

    prompt_policy = config.get("prompt_policy")
    if not isinstance(prompt_policy, dict):
        raise ValueError("document pool v2 prompt_policy must be an object")
    for field in ("initial_prompt_template", "final_prompt", "final_repair_prompt"):
        if not isinstance(prompt_policy.get(field), str) or not prompt_policy[field]:
            raise ValueError(f"document pool v2 prompt_policy requires {field}")
    if "{max_total_model_tokens}" not in prompt_policy["initial_prompt_template"]:
        raise ValueError("document pool v2 initial prompt must expose max_total_model_tokens")
    if "d000001.md -> d000001" not in prompt_policy["final_prompt"]:
        raise ValueError("document pool v2 final prompt must declare the short ID example")
    if "d000001.md -> d000001" not in prompt_policy["final_repair_prompt"]:
        raise ValueError("document pool v2 final repair prompt must declare the short ID example")

    profiles = config.get("profiles")
    tiers = config.get("tiers")
    expected_profiles = {
        "document-pool-v2-lite", "document-pool-v2-full", "lite", "full",
    }
    if not isinstance(profiles, dict) or not isinstance(tiers, dict):
        raise ValueError("document pool v2 config requires profiles and tiers objects")
    if set(profiles) != expected_profiles or set(tiers) != {"lite", "full"}:
        raise ValueError("document pool v2 profiles/tiers must declare lite and full")
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict) or profile.get("tier") not in tiers:
            raise ValueError(f"document pool v2 profile {profile_name!r} has an invalid tier")
    for tier_name, tier in tiers.items():
        if not isinstance(tier, dict):
            raise ValueError(f"document pool v2 tier {tier_name!r} must be an object")
        for field in _V2_TIER_FIELDS:
            if field == "reserve_tokens":
                if type(tier.get(field)) is not int or tier[field] < 0:
                    raise ValueError(f"document pool v2 {tier_name}.{field} must be a non-negative integer")
            else:
                _positive_int(tier.get(field), f"{tier_name}.{field}")
        if tier["context_window"] != 65536:
            raise ValueError("all document pool v2 tiers must use a 65536 context window")
        if tier["max_model_output_tokens"] != 1024 or tier["max_final_output_tokens"] != 1024:
            raise ValueError("all document pool v2 tiers must use 1024 response limits")
        if tier["sample_timeout_seconds"] != 300:
            raise ValueError("all document pool v2 tiers must use the 300 second timeout")
        if tier["max_search_turns"] != 12 or tier["max_tool_invocations"] != 24:
            raise ValueError("all document pool v2 tiers must use 12 search turns and 24 tool invocations")
        if tier["max_final_repair_turns"] != 1 or tier["max_agent_steps"] != 14:
            raise ValueError("all document pool v2 tiers must allow one final repair and 14 agent steps")
        if tier["tool_result_max_bytes"] != 8192:
            raise ValueError("all document pool v2 tiers must use an 8192-byte tool output cap")
        if tier["reserve_tokens"] > tier["max_total_model_tokens"]:
            raise ValueError(f"document pool v2 tier {tier_name!r} reserve exceeds total budget")
    common_fields = set(_V2_TIER_FIELDS) - {"capacity"}
    for field in common_fields:
        if config["tiers"]["lite"][field] != config["tiers"]["full"][field]:
            raise ValueError(f"document pool v2 tiers differ in shared control {field}")
    if config["tiers"]["lite"]["capacity"] != 200 or config["tiers"]["full"]["capacity"] != 1000:
        raise ValueError("document pool v2 capacities must be 200 and 1000")
    return config


def _profile_and_tier(name: str, config: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(name, str) or not name:
        raise ValueError("document pool profile/tier must be a non-empty string")
    profiles = config["profiles"]
    tiers = config["tiers"]
    if name in profiles:
        return name, str(profiles[name]["tier"])
    if name in tiers:
        return f"{_PROFILE_PREFIX}{name}", name
    raise ValueError(f"Unknown document pool profile/tier: {name!r}")


def _v2_profile_and_tier(name: str, config: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(name, str) or not name:
        raise ValueError("document pool v2 profile/tier must be a non-empty string")
    profiles = config["profiles"]
    tiers = config["tiers"]
    canonical_name = _V2_PROFILE_ALIASES.get(name, name)
    if canonical_name in profiles:
        return canonical_name, str(profiles[canonical_name]["tier"])
    if name in tiers:
        return f"{_V2_PROFILE_PREFIX}{name}", name
    raise ValueError(f"Unknown document pool v2 profile/tier: {name!r}")


def get_tier(name: str) -> dict[str, Any]:
    """Return a validated, independent tier record for a tier or profile name."""

    if name in _V2_PROFILE_ALIASES or name in _V2_PROFILE_ALIASES.values() or name in {"lite", "full"}:
        config = load_document_pools_v2()
        profile_name, tier_name = _v2_profile_and_tier(name, config)
        tier = deepcopy(config["tiers"][tier_name])
        tier["max_total_model_tokens"] = _v2_budget_override()
        return {
            "schema_version": config["schema_version"],
            "profile": profile_name,
            "tier": tier_name,
            **tier,
            "id_policy": deepcopy(config["id_policy"]),
            "retriever": deepcopy(config["retriever"]),
        }
    config = load_document_pools()
    profile_name, tier_name = _profile_and_tier(name, config)
    tier = deepcopy(config["tiers"][tier_name])
    return {
        "schema_version": config["schema_version"],
        "profile": profile_name,
        "tier": tier_name,
        **tier,
        "id_policy": deepcopy(config["id_policy"]),
        "retriever": deepcopy(config["retriever"]),
    }


def pool_policy(name: str) -> dict[str, Any]:
    """Alias for :func:`get_tier` used by contract and selector callers."""

    return get_tier(name)


def budget_overrides(name: str) -> dict[str, Any]:
    """Resolve budget and prompt overrides for one candidate-pool profile."""

    config = load_document_pools_v2() if (
        name in _V2_PROFILE_ALIASES or name in _V2_PROFILE_ALIASES.values() or name in {"lite", "full"}
    ) else load_document_pools()
    tier = get_tier(name)
    prompt_policy = config["prompt_policy"]
    total = tier["max_total_model_tokens"]
    initial_prompt = prompt_policy["initial_prompt_template"].replace(
        "{max_total_model_tokens}", str(total)
    )
    return {
        "reserve_tokens": tier["reserve_tokens"],
        "initial_prompt": initial_prompt,
        "final_prompt": prompt_policy["final_prompt"],
        "final_repair_prompt": prompt_policy["final_repair_prompt"],
    }


def rewrite_document_id_prompt(prompt: str, name: str) -> str:
    """Replace legacy long-ID examples while retaining all validation wording."""

    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    example = get_tier(name)["id_policy"]["filename_to_id_example"]
    source = "l%27exemple.md"
    marker = "has original ID report.md."
    rewritten = prompt
    start = 0
    while True:
        source_index = rewritten.find(source, start)
        if source_index < 0:
            break
        marker_index = rewritten.find(marker, source_index)
        if marker_index < 0:
            raise ValueError("legacy ID prompt example is incomplete")
        end = marker_index + len(marker)
        rewritten = rewritten[:source_index] + f"{example}." + rewritten[end:]
        start = source_index + len(example) + 1
    for old, new in (
        ("original document IDs", "document IDs"),
        ("original document ID", "document ID"),
        ("original document ids", "document ids"),
        ("original document id", "document id"),
        ("original corpus IDs", "corpus document IDs"),
        ("original corpus ID", "corpus document ID"),
        ("original IDs", "document IDs"),
        ("original ID", "document ID"),
        ("original doc_id", "doc_id"),
    ):
        rewritten = rewritten.replace(old, new)
    return rewritten


__all__ = [
    "DOCUMENT_POOLS_PATH",
    "budget_overrides",
    "get_tier",
    "load_document_pools",
    "pool_policy",
    "rewrite_document_id_prompt",
    "DOCUMENT_POOLS_V2_PATH",
    "load_document_pools_v2",
]
