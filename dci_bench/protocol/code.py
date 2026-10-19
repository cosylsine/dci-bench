"""Independent host-side code evaluation contract, never a Pi profile override."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CODE_SCORING_VERSION = "dci-code-retrieval-scoring-v1"
CODE_LABELS_VERSION = "dci-code-labels-v1"
CODE_REPORT_VERSION = "dci-code-report-v1"
DEFAULT_CODE_METRIC_KS = (5, 10, 20)
DEFAULT_CODE_MAX_FILES = 20
DEFAULT_CODE_TOKEN_BUDGETS = (4000, 8000, 16000, 32000)
CODE_TOKENIZER = "regex_code_tokenizer_v1"

_SPAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["path", "start_line", "end_line"],
    "properties": {
        "path": {"type": "string", "minLength": 1},
        "start_line": {"type": "integer", "minimum": 1},
        "end_line": {"type": "integer", "minimum": 1},
    },
}

CODE_FINAL_ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ranked_files", "abstain"],
    "properties": {
        "ranked_files": {
            "type": "array", "uniqueItems": True,
            "maxItems": DEFAULT_CODE_MAX_FILES,
            "items": {"type": "string", "minLength": 1},
        },
        "abstain": {"type": "boolean"},
        "spans": {"type": "array", "items": _SPAN_SCHEMA},
    },
    "allOf": [{
        "if": {"properties": {"abstain": {"const": True}}},
        "then": {"properties": {"ranked_files": {"maxItems": 0}, "spans": {"maxItems": 0}}},
        "else": {"properties": {"ranked_files": {"minItems": 1}}},
    }],
}


def code_protocol_contract() -> dict[str, Any]:
    """Return a fresh scoring specification without changing document contracts.

    Online code runtime settings are in code_runtime.py. This specification
    remains identical for saved predictions and online host-side scoring.
    """
    return {
        "evaluation_track": "code",
        "scoring_version": CODE_SCORING_VERSION,
        "labels_version": CODE_LABELS_VERSION,
        "report_version": CODE_REPORT_VERSION,
        "default_metric_ks": list(DEFAULT_CODE_METRIC_KS),
        "default_max_files": DEFAULT_CODE_MAX_FILES,
        "default_token_budgets": list(DEFAULT_CODE_TOKEN_BUDGETS),
        "final_answer_schema": deepcopy(CODE_FINAL_ANSWER_SCHEMA),
        "snapshot_identity": ["repository", "base_commit", "corpus_sha256"],
        "path_policy": "exact case-sensitive repository-relative POSIX paths; no normalization",
        "span_policy": "1-based inclusive LF-delimited source lines; overlapping spans are unioned per file",
        "file_metrics": "binary gold; precision uses actual returned prefix; MRR uses submitted ranking",
        "line_coverage": "gold-line union intersect selected-line union divided by gold-line union",
        "function_recall": "fraction of annotated functions whose complete required spans are selected",
        "functionality_recall": "fraction of functionality groups with any complete alternative evidence set selected",
        "missing_annotations": "unavailable (null), with per-metric eligible denominators",
        "missing_predictions": "invalid output, never an abstention; eligible metrics score zero",
        "selective_success": "positive: any gold in top k; explicit no-gold: valid abstention",
        "bcy": {
            "tokenizer": CODE_TOKENIZER,
            "packing": "ranked whole canonical corpus files with ### path headers and one separator token for nonempty text; prefix truncate at budget",
            "gold_threshold_tokens": 1,
            "empty_content_gold": "never credited",
            "interpretation": "reconstructed context yield, not proof of actual tool reads",
        },
        "host_only": ["gold_files", "gold_spans", "gold_functions", "gold_functionalities", "no_gold_kind"],
        "runtime_execution": "not performed; retrieval coverage is not test execution coverage or patch correctness",
        "reference": "https://arxiv.org/abs/2607.24882",
    }
