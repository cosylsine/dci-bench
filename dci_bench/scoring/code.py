"""Repository context retrieval metrics, independent of document scoring.

Ground truth is host-only and tied to an immutable repository snapshot. File
retrieval does not imply that source lines or functions have been inspected.
Only explicitly selected spans contribute to the optional coverage metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from dci_bench.protocol.code import (
    CODE_LABELS_VERSION,
    CODE_REPORT_VERSION,
    CODE_SCORING_VERSION,
    CODE_TOKENIZER,
    DEFAULT_CODE_MAX_FILES,
    DEFAULT_CODE_METRIC_KS,
    DEFAULT_CODE_TOKEN_BUDGETS,
)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^A-Za-z0-9_\s]", re.UNICODE)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
SpanMap = dict[str, list[tuple[int, int]]]


class InvalidCodeReferenceError(ValueError):
    """Malformed host labels are an evaluation error, not a model zero score."""


class InvalidCodePredictionError(ValueError):
    """The submitted result violates the code-track contract."""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _positive_values(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(type(v) is not int or v <= 0 for v in result):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _path(value: Any) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or any(part in ("", ".", "..") for part in value.split("/"))
            or re.match(r"^[A-Za-z]:", value)):
        raise ValueError("paths must be exact repository-relative POSIX paths")
    return value


def _keys(value: Any, required: set[str], optional: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if not required <= value.keys() or set(value) - required - optional:
        raise ValueError(f"{name} requires {sorted(required)}; optional keys: {sorted(optional)}")


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _spans(values: Any, files: Mapping[str, Any], *, allowed: set[str] | None = None) -> SpanMap:
    if not isinstance(values, list):
        raise ValueError("spans must be a list")
    by_path: SpanMap = {}
    for span in values:
        _keys(span, {"path", "start_line", "end_line"}, set(), "span")
        path = _path(span["path"])
        if path not in files or (allowed is not None and path not in allowed):
            raise ValueError(f"span path {path!r} is outside the allowed files")
        start, end = span["start_line"], span["end_line"]
        if (type(start) is not int or type(end) is not int
                or not 1 <= start <= end <= files[path]["line_count"]):
            raise ValueError(f"invalid 1-based inclusive line range for {path!r}")
        by_path.setdefault(path, []).append((start, end))
    return {path: _merge(intervals) for path, intervals in by_path.items()}


def _length(spans: SpanMap) -> int:
    return sum(end - start + 1 for ranges in spans.values() for start, end in ranges)


def _overlap(left: SpanMap, right: SpanMap) -> int:
    total = 0
    for path, ranges in left.items():
        other = right.get(path, [])
        i = j = 0
        while i < len(ranges) and j < len(other):
            a, b = ranges[i]
            c, d = other[j]
            total += max(0, min(b, d) - max(a, c) + 1)
            if b <= d:
                i += 1
            else:
                j += 1
    return total


def file_metadata(text: str) -> dict[str, Any]:
    """Freeze canonical UTF-8 text identity and exact ARB regex token costs.

    Call on the actual candidate corpus text (which may differ from an upstream
    full checkout). Do not wrap, normalize newlines, or prepend gold metadata.
    """
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        # Source tools count LF boundaries; str.splitlines() also splits on
        # form feeds and Unicode separators that can occur inside source lines.
        "line_count": text.count("\n") + int(bool(text) and not text.endswith("\n")),
        "content_tokens": len(_TOKEN_RE.findall(text)),
    }


def validate_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Validate labels before looking at predictions; do not infer no-gold."""
    try:
        _keys(reference, {"repository", "base_commit", "task", "files", "gold_files"},
              {"no_gold_kind", "gold_spans", "gold_functions", "gold_functionalities", "tokenizer"},
              "code reference")
        _text(reference["repository"], "repository")
        _text(reference["task"], "task")
        commit = reference["base_commit"]
        if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
            raise ValueError("base_commit must be a full lowercase 40- or 64-digit Git commit hash")
        files = reference["files"]
        if not isinstance(files, Mapping) or not files:
            raise ValueError("files must be a nonempty candidate manifest")
        token_fields = []
        for path, metadata in files.items():
            _path(path)
            _keys(metadata, {"sha256", "line_count"}, {"content_tokens"}, "file metadata")
            digest = metadata["sha256"]
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise ValueError(f"invalid sha256 for {path!r}")
            lines = metadata["line_count"]
            if type(lines) is not int or lines < 0:
                raise ValueError("line_count must be a nonnegative integer")
            token_fields.append("content_tokens" in metadata)
            if "content_tokens" in metadata:
                tokens = metadata["content_tokens"]
                if type(tokens) is not int or tokens < 0 or (lines == 0 and tokens != 0):
                    raise ValueError("content_tokens must be nonnegative and zero for an empty file")
        if any(token_fields) or "tokenizer" in reference:
            if not all(token_fields) or reference.get("tokenizer") != CODE_TOKENIZER:
                raise ValueError(f"BCY requires all file token counts and tokenizer={CODE_TOKENIZER}")
        gold = reference["gold_files"]
        if not isinstance(gold, list) or any(not isinstance(p, str) or p not in files for p in gold):
            raise ValueError("gold_files must be a list of exact candidate paths")
        if len(set(gold)) != len(gold):
            raise ValueError("gold_files contains duplicate paths")
        if not gold:
            if reference.get("no_gold_kind") not in ("natural", "counterfactual"):
                raise ValueError("empty gold_files requires explicit natural/counterfactual no_gold_kind")
            if any(key in reference for key in ("gold_spans", "gold_functions", "gold_functionalities")):
                raise ValueError("no-gold references cannot contain positive fine-grained labels")
        elif "no_gold_kind" in reference:
            raise ValueError("positive references must not declare no_gold_kind")
        if "gold_spans" in reference:
            if not _spans(reference["gold_spans"], files, allowed=set(gold)):
                raise ValueError("gold_spans must be nonempty when supplied; omit unavailable labels")
        for field, id_key, alternatives in (
            ("gold_functions", "function_id", False),
            ("gold_functionalities", "functionality_id", True),
        ):
            if field not in reference:
                continue
            entries = reference[field]
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"{field} must be nonempty when supplied")
            seen = set()
            for entry in entries:
                evidence_key = "evidence_sets" if alternatives else "spans"
                _keys(entry, {id_key, evidence_key}, set(), field)
                _text(entry[id_key], id_key)
                if entry[id_key] in seen:
                    raise ValueError(f"duplicate {id_key}")
                seen.add(entry[id_key])
                evidence_sets = entry[evidence_key] if alternatives else [entry[evidence_key]]
                if not isinstance(evidence_sets, list) or not evidence_sets:
                    raise ValueError("evidence_sets must be nonempty")
                for evidence in evidence_sets:
                    if not _spans(evidence, files, allowed=set(gold)):
                        raise ValueError("each evidence set must contain at least one span")
        return deepcopy(dict(reference))
    except (TypeError, ValueError) as exc:
        raise InvalidCodeReferenceError(str(exc)) from exc


def snapshot_identity(reference: Mapping[str, Any]) -> dict[str, str]:
    """Identity attached by the host adapter, not invented by the model."""
    ref = validate_reference(reference)
    return _snapshot_identity(ref)


def _snapshot_identity(reference: Mapping[str, Any]) -> dict[str, str]:
    return {
        "repository": reference["repository"],
        "base_commit": reference["base_commit"],
        "corpus_sha256": _canonical_hash(reference["files"]),
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(text: str) -> Any:
    """Strict JSON, also used by the CLI to preserve exact query identity."""
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject_constant)


def validate_code_output(output: Any, reference: Mapping[str, Any], *, max_files: int = DEFAULT_CODE_MAX_FILES) -> dict[str, Any]:
    """Validate model JSON against the host's exact candidate manifest."""
    if type(max_files) is not int or max_files <= 0:
        raise ValueError("max_files must be a positive integer")
    try:
        if isinstance(output, str):
            output = load_json(output)
        _keys(output, {"ranked_files", "abstain"}, {"spans"}, "code output")
        ranked = output["ranked_files"]
        if not isinstance(ranked, list) or len(ranked) > max_files:
            raise ValueError(f"ranked_files must be a list of at most {max_files} paths")
        for path in ranked:
            _path(path)
            if path not in reference["files"]:
                raise ValueError(f"unknown candidate file: {path!r}")
        if len(set(ranked)) != len(ranked):
            raise ValueError("ranked_files must not contain duplicate paths")
        abstain = output["abstain"]
        if type(abstain) is not bool or abstain != (len(ranked) == 0):
            raise ValueError("abstain must be true exactly when ranked_files is empty")
        _spans(output.get("spans", []), reference["files"], allowed=set(ranked))
        return deepcopy(dict(output))
    except (TypeError, ValueError) as exc:
        raise InvalidCodePredictionError(str(exc)) from exc


def _bcy(ranked: list[str], reference: Mapping[str, Any], budget: int) -> tuple[float, dict[str, Any]]:
    """ARB canonical whole-file packing, tau=1, using frozen host token costs."""
    used = hits = 0
    packed_files = 0
    gold = set(reference["gold_files"])
    for path in ranked:
        if used >= budget:
            break
        metadata = reference["files"][path]
        header = len(_TOKEN_RE.findall(f"### {path}\n"))
        content = metadata["content_tokens"]
        separator = int(metadata["line_count"] > 0)
        remaining = budget - used
        total = header + content + separator
        selected = content if total <= remaining else min(content, max(0, remaining - header))
        used += min(total, remaining)
        packed_files += 1
        hits += int(path in gold and content > 0 and selected >= 1)
    return hits / len(gold), {
        "used_tokens": used,
        "packed_files": packed_files,
        "budget_saturated": used == budget,
        "ranking_exhausted_before_budget": used < budget,
    }


def _metric_template(ref: Mapping[str, Any], ks: tuple[int, ...], budgets: tuple[int, ...]) -> dict[str, float | None]:
    positive = bool(ref["gold_files"])
    metrics: dict[str, float | None] = {}
    for k in ks:
        for family in ("file_precision", "file_recall", "file_f1", "file_ndcg", "file_hit", "all_gold_files"):
            metrics[f"{family}_at_{k}"] = 0.0 if positive else None
        metrics[f"selective_success_at_{k}"] = 0.0
    metrics.update({
        "reciprocal_rank": 0.0 if positive else None,
        "abstention": 0.0,
        "accepted": 0.0,
        "no_gold_abstention": None if positive else 0.0,
        "line_coverage": 0.0 if "gold_spans" in ref else None,
        "line_precision": 0.0 if "gold_spans" in ref else None,
        "line_f1": 0.0 if "gold_spans" in ref else None,
        "function_recall": 0.0 if "gold_functions" in ref else None,
        "functionality_recall": 0.0 if "gold_functionalities" in ref else None,
    })
    for budget in budgets:
        metrics[f"bcy_at_{budget}_tokens"] = 0.0 if positive and "tokenizer" in ref else None
    return metrics


def score_query(
    prediction: Any,
    reference: Mapping[str, Any],
    *,
    metric_ks: Sequence[int] = DEFAULT_CODE_METRIC_KS,
    max_files: int = DEFAULT_CODE_MAX_FILES,
    token_budgets: Sequence[int] = DEFAULT_CODE_TOKEN_BUDGETS,
) -> dict[str, Any]:
    """Score one host-stamped prediction; malformed labels raise before scoring.

    Prediction envelope: snapshot_identity(reference) plus ``output`` (model
    JSON), or a nonempty ``failure_reason`` from the collection runner. Absent
    predictions and upstream failures remain in all-sample denominators.
    """
    ref = validate_reference(reference)
    ks = _positive_values(metric_ks, "metric_ks")
    budgets = _positive_values(token_budgets, "token_budgets")
    if type(max_files) is not int or max_files <= 0:
        raise ValueError("max_files must be a positive integer")
    identity = _snapshot_identity(ref)
    result: dict[str, Any] = {
        "evaluation_track": "code", "scoring_version": CODE_SCORING_VERSION,
        **identity, "task": ref["task"],
        "stratum": ref.get("no_gold_kind", "positive"),
        "valid_output": False, "failure_kind": None, "failure_reason": None,
        "metrics": _metric_template(ref, ks, budgets),
        "context_packing": {},
    }
    if prediction is None:
        result.update(failure_kind="missing_prediction", failure_reason="No prediction for planned query")
        return result
    try:
        if isinstance(prediction, str):
            prediction = load_json(prediction)
        _keys(prediction, set(identity), {"output", "failure_reason"}, "prediction envelope")
        if any(prediction[key] != value for key, value in identity.items()):
            raise ValueError("prediction repository/base_commit/corpus_sha256 does not match the frozen reference")
        if "failure_reason" in prediction:
            _text(prediction["failure_reason"], "failure_reason")
            result.update(failure_kind="upstream_failure", failure_reason=prediction["failure_reason"])
            return result
        if "output" not in prediction:
            raise ValueError("prediction must contain output or failure_reason")
        output = validate_code_output(prediction["output"], ref, max_files=max_files)
    except (TypeError, ValueError) as exc:
        result.update(failure_kind="invalid_output", failure_reason=str(exc))
        return result

    result.update(valid_output=True, output=output)
    metrics = result["metrics"]
    ranked, abstain = output["ranked_files"], output["abstain"]
    gold = set(ref["gold_files"])
    metrics["abstention"], metrics["accepted"] = float(abstain), float(not abstain)
    for k in ks:
        if not gold:
            metrics[f"selective_success_at_{k}"] = float(abstain)
            continue
        window = ranked[:k]
        hits = len(gold.intersection(window))
        precision = hits / len(window) if window else 0.0
        recall = hits / len(gold)
        dcg = sum(1 / math.log2(i + 2) for i, path in enumerate(window) if path in gold)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
        metrics.update({
            f"file_precision_at_{k}": precision,
            f"file_recall_at_{k}": recall,
            f"file_f1_at_{k}": 2 * precision * recall / (precision + recall) if hits else 0.0,
            f"file_ndcg_at_{k}": dcg / ideal,
            f"file_hit_at_{k}": float(hits > 0),
            f"all_gold_files_at_{k}": float(hits == len(gold)),
            f"selective_success_at_{k}": float(hits > 0),
        })
    if not gold:
        metrics["no_gold_abstention"] = float(abstain)
        return result
    metrics["reciprocal_rank"] = next((1 / i for i, path in enumerate(ranked, 1) if path in gold), 0.0)
    selected = _spans(output.get("spans", []), ref["files"])
    if "gold_spans" in ref:
        required = _spans(ref["gold_spans"], ref["files"])
        overlap = _overlap(selected, required)
        recall = overlap / _length(required)
        precision = overlap / _length(selected) if selected else 0.0
        metrics.update(line_coverage=recall, line_precision=precision,
                       line_f1=2 * precision * recall / (precision + recall) if overlap else 0.0)
    for field, metric, alternatives in (
        ("gold_functions", "function_recall", False),
        ("gold_functionalities", "functionality_recall", True),
    ):
        if field not in ref:
            continue
        hits = 0
        for entry in ref[field]:
            evidence_sets = entry["evidence_sets"] if alternatives else [entry["spans"]]
            for evidence in evidence_sets:
                required = _spans(evidence, ref["files"])
                if _overlap(selected, required) == _length(required):
                    hits += 1
                    break
        metrics[metric] = hits / len(ref[field])
    if "tokenizer" in ref:
        for budget in budgets:
            value, diagnostics = _bcy(ranked, ref, budget)
            metrics[f"bcy_at_{budget}_tokens"] = value
            result["context_packing"][str(budget)] = diagnostics
    return result


def _aggregate(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    metrics = {}
    for key in metric_names:
        eligible = [row for row in rows if row["metrics"][key] is not None]
        valid = [row for row in eligible if row["valid_output"]]
        metrics[key] = {
            "mean": sum(row["metrics"][key] for row in eligible) / len(eligible) if eligible else None,
            "eligible_queries": len(eligible),
            "valid_only_mean": sum(row["metrics"][key] for row in valid) / len(valid) if valid else None,
            "valid_queries": len(valid),
        }
    valid_count = sum(row["valid_output"] for row in rows)
    failures: dict[str, int] = {}
    for row in rows:
        if not row["valid_output"]:
            kind = row["failure_kind"]
            failures[kind] = failures.get(kind, 0) + 1
    return {
        "num_queries": len(rows), "valid_outputs": valid_count,
        "invalid_outputs": len(rows) - valid_count,
        "valid_output_rate": valid_count / len(rows) if rows else None,
        "failure_counts": failures, "metrics": metrics,
    }


def score_run(
    predictions: Mapping[str, Any],
    labels: Mapping[str, Any],
    *,
    metric_ks: Sequence[int] = DEFAULT_CODE_METRIC_KS,
    max_files: int = DEFAULT_CODE_MAX_FILES,
    token_budgets: Sequence[int] = DEFAULT_CODE_TOKEN_BUDGETS,
) -> dict[str, Any]:
    """Macro means over planned queries, with separate tasks and no-gold strata."""
    _keys(labels, {"schema_version", "queries"}, set(), "code labels")
    if labels["schema_version"] != CODE_LABELS_VERSION:
        raise InvalidCodeReferenceError(f"expected schema_version={CODE_LABELS_VERSION}")
    queries = labels["queries"]
    if not isinstance(queries, Mapping) or not queries:
        raise InvalidCodeReferenceError("queries must be a nonempty query-id mapping")
    if not isinstance(predictions, Mapping):
        raise ValueError("predictions must be a query-id mapping")
    for query_id in queries:
        _text(query_id, "query_id")
    if set(predictions) - set(queries):
        raise ValueError("predictions contain query IDs outside the planned labels")
    ks = _positive_values(metric_ks, "metric_ks")
    budgets = _positive_values(token_budgets, "token_budgets")
    per_query = {query_id: score_query(predictions.get(query_id), reference,
                 metric_ks=ks, max_files=max_files, token_budgets=budgets)
                 for query_id, reference in queries.items()}
    rows = list(per_query.values())
    names = list(rows[0]["metrics"])
    return {
        "schema_version": CODE_REPORT_VERSION,
        "evaluation_track": "code", "scoring_version": CODE_SCORING_VERSION,
        "metric_ks": list(ks), "max_files": max_files, "token_budgets": list(budgets),
        **_aggregate(rows, names),
        "by_task": {task: _aggregate([r for r in rows if r["task"] == task], names)
                    for task in sorted({r["task"] for r in rows})},
        "by_stratum": {stratum: _aggregate([r for r in rows if r["stratum"] == stratum], names)
                       for stratum in ("positive", "natural", "counterfactual")},
        "per_query": per_query,
    }
