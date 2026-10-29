"""Quality checks and preparation helpers for DCI data bundles.

The pipeline deliberately treats a bundle as an ordinary Python mapping.  The
functions in this module never mutate the mapping supplied by the caller and
keep source identifiers and source text unchanged unless a preparation policy
explicitly requests a normalization operation.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import numbers
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BUNDLE_SCHEMA_VERSION = "dci-data-bundle-v1"
REPORT_SCHEMA_VERSION = "dci-data-quality-v1"
INPUT_KINDS = frozenset({"retrieval", "evidence_qa", "corpus"})
# DCI's corpus builder uses a decimal one-million-byte compatibility ceiling.
# Keep render/tool-result overhead outside this data limit; the builder owns it.
DEFAULT_MAX_DOCUMENT_BYTES = 1_000_000
DEFAULT_MAX_QUERY_BYTES = 1_000_000
DEFAULT_MAX_POSITIVE_PER_QUERY = 10
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.85
DEFAULT_MAX_NEAR_DUPLICATE_DOCS = 2_000
DEFAULT_MAX_NEAR_DUPLICATE_PAIRS = 10_000
SHORT_DOCUMENT_CHARS = 256
LONG_DOCUMENT_CHARS = 8_192

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def _new_report(*, strict: bool) -> dict[str, Any]:
    return {
        "report_version": REPORT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "strict": bool(strict),
        "valid": True,
        "status": "pass",
        "errors": [],
        "warnings": [],
        "issues": [],
        "review_required": [],
        "exclusions": [],
        "counts": {
            "documents": 0,
            "queries": 0,
            "qrels_queries": 0,
            "positive_pairs": 0,
            "evidence": 0,
        },
        "limits": {
            "max_document_bytes": DEFAULT_MAX_DOCUMENT_BYTES,
            "max_query_bytes": DEFAULT_MAX_QUERY_BYTES,
            "max_positive_per_query": DEFAULT_MAX_POSITIVE_PER_QUERY,
            "short_document_chars": SHORT_DOCUMENT_CHARS,
            "long_document_chars": LONG_DOCUMENT_CHARS,
        },
    }


def _finish_status(report: dict[str, Any]) -> None:
    """Set the small status vocabulary consumed by pipeline callers."""

    report["valid"] = not bool(report["errors"])
    if report["errors"]:
        report["status"] = "invalid"
    elif report.get("needs_queries"):
        report["status"] = "needs_queries"
    elif report.get("review_required"):
        report["status"] = "review_required"
    else:
        report["status"] = "pass"


def _safe_value(value: Any) -> Any:
    """Return a JSON-friendly value for issue details."""

    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item) for key, item in value.items()}
    return repr(value)


def _issue(
    report: dict[str, Any],
    code: str,
    message: str,
    *,
    path: str | None = None,
    severity: str = "error",
    **details: Any,
) -> None:
    item: dict[str, Any] = {
        "code": code,
        "message": message,
        "severity": severity,
    }
    if path is not None:
        item["path"] = path
    if details:
        item["details"] = _safe_value(details)
    report["issues"].append(item)
    if severity == "warning":
        report["warnings"].append(item)
    else:
        report["errors"].append(item)


def _review(
    report: dict[str, Any],
    code: str,
    message: str,
    *,
    query_id: str | None = None,
    document_ids: Iterable[str] = (),
    **details: Any,
) -> None:
    item: dict[str, Any] = {
        "code": code,
        "message": message,
        "review_required": True,
    }
    if query_id is not None:
        item["query_id"] = query_id
    ids = list(document_ids)
    if ids:
        item["document_ids"] = ids
    if details:
        item["details"] = _safe_value(details)
    report["review_required"].append(item)


def _exclude(
    exclusions: list[dict[str, Any]],
    kind: str,
    identifier: Any,
    reason: str,
    *,
    path: str | None = None,
    **details: Any,
) -> None:
    item: dict[str, Any] = {"kind": kind, "id": _safe_value(identifier), "reason": reason}
    if path is not None:
        item["path"] = path
    if details:
        item["details"] = _safe_value(details)
    exclusions.append(item)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _byte_length(value: str) -> int:
    return len(value.encode("utf-8"))


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _normalized_query_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", text).strip()).casefold()


def _find_all(text: str, needle: str) -> list[int]:
    starts: list[int] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return starts
        starts.append(index)
        # Move by one code point so overlapping occurrences are also visible.
        start = index + 1


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _shingles(text: str, size: int = 3, limit: int = 512) -> set[str]:
    tokens = _tokens(text)
    if not tokens:
        return set()
    if len(tokens) < size:
        return {" ".join(tokens)}
    shingles = {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}
    if len(shingles) <= limit:
        return shingles
    # Keep a deterministic bounded sample.  This is a candidate detector, not
    # a replacement for an exact similarity calculation.
    return set(sorted(shingles)[:limit])


def _script_hint(text: str) -> str | None:
    counts: Counter[str] = Counter()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "LATIN" in name:
            counts["Latin"] += 1
        elif "CYRILLIC" in name:
            counts["Cyrillic"] += 1
        elif "GREEK" in name:
            counts["Greek"] += 1
        elif "ARABIC" in name:
            counts["Arabic"] += 1
        elif "HEBREW" in name:
            counts["Hebrew"] += 1
        elif "DEVANAGARI" in name:
            counts["Devanagari"] += 1
        elif "HIRAGANA" in name or "KATAKANA" in name:
            counts["Japanese"] += 1
        elif "HANGUL" in name:
            counts["Hangul"] += 1
        elif "CJK" in name or "IDEOGRAPH" in name:
            counts["Han"] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _metadata(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = record.get("metadata")
    return value if isinstance(value, Mapping) else None


def _declared_language(record: Mapping[str, Any]) -> str:
    language = record.get("language")
    if isinstance(language, str) and language.strip():
        return language
    return "unknown"


def _verified_language(record: Mapping[str, Any]) -> str:
    metadata = _metadata(record)
    if metadata is None:
        return "unknown"
    value = metadata.get("verified_language")
    if isinstance(value, str) and value.strip():
        return value
    value = metadata.get("language_verified")
    if isinstance(value, str) and value.strip():
        return value
    return "unknown"


def _translation_class(record: Mapping[str, Any]) -> str:
    metadata = _metadata(record)
    if metadata is None:
        return "unknown"
    translation_of = metadata.get("translation_of")
    if _valid_id(translation_of):
        return "translated"
    if metadata.get("is_translation") is True:
        return "translated"
    status = metadata.get("translation_status")
    if isinstance(status, str) and status.casefold() in {"translated", "translation"}:
        return "translated"
    if metadata.get("native") is True:
        return "native"
    if isinstance(status, str) and status.casefold() in {"native", "original", "source"}:
        return "native"
    return "unknown"


def _annotation_source(record: Mapping[str, Any], provenance: Mapping[str, Any] | None = None) -> str:
    metadata = _metadata(record)
    containers: list[Mapping[str, Any]] = []
    if metadata is not None:
        containers.append(metadata)
    containers.append(record)
    if provenance is not None:
        containers.append(provenance)
    for container in containers:
        for key in ("annotation_source", "label_source", "judgment_source", "origin"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                normalized = value.casefold().strip()
                if normalized in {"model", "human", "source"}:
                    return normalized
                return "unknown"
    return "unknown"


def _value_from_record_or_provenance(record: Mapping[str, Any], key: str, provenance: Mapping[str, Any]) -> str:
    metadata = _metadata(record)
    for container in (metadata, record, provenance):
        if isinstance(container, Mapping):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return "unknown"


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _collection_stats(
    records: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
    include_lengths: bool,
) -> dict[str, Any]:
    languages: Counter[str] = Counter()
    verified_languages: Counter[str] = Counter()
    translation: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    genres: Counter[str] = Counter()
    scripts: Counter[str] = Counter()
    short = 0
    long = 0
    lengths: list[int] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        languages[_declared_language(record)] += 1
        verified_languages[_verified_language(record)] += 1
        translation[_translation_class(record)] += 1
        origins[_annotation_source(record, provenance)] += 1
        domain = _value_from_record_or_provenance(record, "domain", provenance)
        genre = _value_from_record_or_provenance(record, "genre", provenance)
        domains[domain] += 1
        genres[genre] += 1
        text = record.get("text")
        if isinstance(text, str):
            script = _script_hint(text)
            if script:
                scripts[script] += 1
            if include_lengths:
                length = len(text)
                lengths.append(length)
                short += int(length <= SHORT_DOCUMENT_CHARS)
                long += int(length > LONG_DOCUMENT_CHARS)
    result: dict[str, Any] = {
        "language": _counter_dict(languages),
        "declared_language": _counter_dict(languages),
        "verified_language": _counter_dict(verified_languages),
        "translation_status": _counter_dict(translation),
        "translation": _counter_dict(translation),
        "native_translated_unknown": _counter_dict(translation),
        "annotation_source": _counter_dict(origins),
        "domain": _counter_dict(domains),
        "genre": _counter_dict(genres),
        "script_hint": _counter_dict(scripts),
    }
    if include_lengths:
        result["short_documents"] = short
        result["long_documents"] = long
        result["length_chars"] = {
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "mean": (sum(lengths) / len(lengths)) if lengths else None,
        }
    return result


def _record_lists(
    bundle: Mapping[str, Any], report: dict[str, Any]
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    values: list[list[Any]] = []
    for field in ("documents", "queries", "evidence", "judgments"):
        value = bundle.get(field)
        if value is None and field == "judgments":
            value = []
        if not isinstance(value, list):
            _issue(report, "field_type", f"{field} must be a list", path=field)
            value = []
        values.append(value)
    return values[0], values[1], values[2], values[3]


def _near_duplicate_report(
    documents: Sequence[Mapping[str, Any]],
    *,
    max_docs: int = DEFAULT_MAX_NEAR_DUPLICATE_DOCS,
    max_pairs: int = DEFAULT_MAX_NEAR_DUPLICATE_PAIRS,
    threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> dict[str, Any]:
    candidates = [
        (str(record.get("_id")), _shingles(record.get("text", "")))
        for record in documents
        if isinstance(record, Mapping)
        and _valid_id(record.get("_id"))
        and isinstance(record.get("text"), str)
    ]
    skipped_documents = max(0, len(candidates) - max_docs)
    candidates = candidates[:max_docs]
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    minhash_seeds = (17, 31, 47, 61, 73, 89, 101, 127, 149, 163, 181, 197)
    signatures: list[tuple[int, ...]] = []
    for index, (_, shingles) in enumerate(candidates):
        if not shingles:
            signature = tuple([0] * len(minhash_seeds))
        else:
            signature = tuple(
                min(
                    int.from_bytes(
                        hashlib.blake2b(
                            f"{seed}:{shingle}".encode("utf-8"), digest_size=8
                        ).digest(),
                        "big",
                    )
                    for shingle in shingles
                )
                for seed in minhash_seeds
            )
        signatures.append(signature)
        for band in range(3):
            start = band * 4
            buckets[(band, signature[start : start + 4])].append(index)
    pair_indices: set[tuple[int, int]] = set()
    truncated = False
    for bucket_indices in buckets.values():
        if len(bucket_indices) < 2:
            continue
        for left_offset, left in enumerate(bucket_indices[:-1]):
            for right in bucket_indices[left_offset + 1 :]:
                pair = (min(left, right), max(left, right))
                if pair in pair_indices:
                    continue
                if len(pair_indices) >= max_pairs:
                    truncated = True
                    break
                pair_indices.add(pair)
            if truncated:
                break
        if truncated:
            break
    near_pairs: list[dict[str, Any]] = []
    compared = 0
    for left, right in sorted(pair_indices):
        left_id, left_shingles = candidates[left]
        right_id, right_shingles = candidates[right]
        # Exact duplicates already have a stronger, lossless grouping below.
        if left_shingles == right_shingles:
            continue
        compared += 1
        union = left_shingles | right_shingles
        score = (len(left_shingles & right_shingles) / len(union)) if union else 0.0
        if score >= threshold:
            near_pairs.append(
                {
                    "document_ids": [left_id, right_id],
                    "jaccard": round(score, 6),
                    "method": "token-shingles-minhash-candidates",
                }
            )
    return {
        "method": "token-shingles-minhash-candidates",
        "threshold": threshold,
        "max_documents": max_docs,
        "max_pair_comparisons": max_pairs,
        "candidate_documents": len(candidates),
        "skipped_documents": skipped_documents,
        "candidate_pairs": len(pair_indices),
        "compared_pairs": compared,
        "truncated": truncated,
        "pairs": near_pairs,
    }


def _valid_offset(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _audit_evidence(
    evidence: Sequence[Any],
    *,
    document_map: Mapping[str, Mapping[str, Any]],
    query_ids: set[str],
    report: dict[str, Any],
) -> dict[str, Any]:
    valid = 0
    ambiguous = 0
    invalid = 0
    duplicate_keys: set[tuple[Any, ...]] = set()
    duplicate_count = 0
    for index, item in enumerate(evidence):
        path = f"evidence[{index}]"
        if not isinstance(item, Mapping):
            _issue(report, "evidence_type", "evidence entries must be objects", path=path)
            invalid += 1
            continue
        query_id = item.get("query_id")
        doc_id = item.get("doc_id")
        item_valid = True
        if not _valid_id(query_id) or query_id not in query_ids:
            _issue(report, "unknown_evidence_query", "evidence refers to an unknown query", path=f"{path}.query_id", query_id=query_id)
            item_valid = False
        if not _valid_id(doc_id) or doc_id not in document_map:
            _issue(report, "unknown_evidence_document", "evidence refers to an unknown document", path=f"{path}.doc_id", doc_id=doc_id)
            item_valid = False
        text = document_map.get(doc_id, {}).get("text") if isinstance(doc_id, str) else None
        quote = item.get("quote")
        has_start = "start" in item
        has_end = "end" in item
        start = item.get("start")
        end = item.get("end")
        offsets_present = has_start or has_end
        offsets_valid = False
        if offsets_present:
            if not (has_start and has_end and _valid_offset(start) and _valid_offset(end)):
                _issue(report, "evidence_offset_type", "evidence start and end must be integer offsets", path=path)
                item_valid = False
            elif not isinstance(text, str) or start < 0 or end <= start or end > len(text):
                _issue(report, "evidence_offset_bounds", "evidence offsets are outside the document", path=path, start=start, end=end)
                item_valid = False
            else:
                offsets_valid = True
        occurrences: list[int] = []
        if quote is not None:
            if not isinstance(quote, str) or not quote:
                _issue(report, "evidence_quote_type", "evidence quote must be a nonempty string", path=f"{path}.quote")
                item_valid = False
            elif isinstance(text, str):
                occurrences = _find_all(text, quote)
                if not occurrences:
                    _issue(report, "evidence_quote_not_found", "evidence quote does not occur in the document", path=f"{path}.quote")
                    item_valid = False
                elif len(occurrences) > 1 and not offsets_valid:
                    ambiguous += 1
                    _issue(
                        report,
                        "ambiguous_evidence_quote",
                        "evidence quote occurs more than once; supply matching offsets",
                        path=f"{path}.quote",
                        severity="error",
                        occurrences=len(occurrences),
                    )
                    _review(report, "ambiguous_evidence_quote", "quote needs an explicit occurrence offset", query_id=query_id, document_ids=[doc_id] if isinstance(doc_id, str) else [])
                    item_valid = False
                if offsets_valid and isinstance(start, int) and isinstance(end, int) and text[start:end] != quote:
                    _issue(report, "evidence_offset_mismatch", "evidence offsets do not enclose the quote", path=path)
                    item_valid = False
        elif not offsets_valid:
            _issue(
                report,
                "evidence_missing_locator",
                "evidence must contain a quote or validated start/end offsets",
                path=path,
            )
            if "answer" in item:
                _issue(
                    report,
                    "answer_is_not_evidence",
                    "an answer string cannot be promoted to evidence without a source locator",
                    path=path,
                    severity="warning",
                )
            item_valid = False
        quote_key = quote if isinstance(quote, (str, int, float, bool, type(None))) else repr(quote)
        key = (
            query_id if _valid_id(query_id) else repr(query_id),
            doc_id if _valid_id(doc_id) else repr(doc_id),
            quote_key,
            start if _valid_offset(start) else None,
            end if _valid_offset(end) else None,
        )
        if key in duplicate_keys:
            duplicate_count += 1
            _issue(report, "duplicate_evidence", "duplicate evidence entry", path=path, severity="warning")
        duplicate_keys.add(key)
        if item_valid:
            valid += 1
        else:
            invalid += 1
    return {
        "total": len(evidence),
        "valid": valid,
        "invalid": invalid,
        "ambiguous": ambiguous,
        "duplicate": duplicate_count,
    }


def audit_bundle(bundle: dict, *, strict: bool = True) -> dict:
    """Audit a DCI data bundle without modifying it.

    The returned object is JSON serializable.  ``strict`` records the requested
    policy and controls the status field; diagnostics are retained in either
    mode so callers can inspect malformed input instead of losing its context.
    A malformed top-level value is represented as an invalid report rather than
    raising an exception, which makes this function suitable for ingestion
    preflight and batch reports.
    """

    report = _new_report(strict=strict)
    if not isinstance(bundle, dict):
        _issue(report, "bundle_type", "bundle must be an ordinary dict", path="bundle")
        _finish_status(report)
        return report

    schema_version = bundle.get("schema_version")
    if schema_version != BUNDLE_SCHEMA_VERSION:
        _issue(
            report,
            "schema_version",
            f"schema_version must be {BUNDLE_SCHEMA_VERSION!r}",
            path="schema_version",
            value=schema_version,
        )
    report["schema_version"] = schema_version
    input_kind = bundle.get("input_kind")
    if input_kind not in INPUT_KINDS:
        _issue(report, "input_kind", "input_kind must be retrieval, evidence_qa, or corpus", path="input_kind", value=input_kind)
        input_kind = None
    report["input_kind"] = input_kind

    source = bundle.get("source")
    if not isinstance(source, Mapping):
        _issue(report, "source_type", "source must be an object", path="source")
        source = {}
    else:
        for key in ("dataset", "revision"):
            if not _valid_id(source.get(key)):
                _issue(report, "source_identity", f"source.{key} must be a nonempty string", path=f"source.{key}")
    provenance = bundle.get("provenance")
    if not isinstance(provenance, Mapping):
        _issue(report, "provenance_type", "provenance must be an object", path="provenance")
        provenance = {}
    report["source"] = _safe_value(source)
    report["provenance"] = _safe_value(provenance)

    documents, queries, evidence, judgments = _record_lists(bundle, report)
    report["counts"]["documents"] = len(documents)
    report["counts"]["queries"] = len(queries)
    report["counts"]["evidence"] = len(evidence)
    if not isinstance(bundle.get("qrels", {}), Mapping):
        _issue(report, "qrels_type", "qrels must be an object", path="qrels")
        qrels: Mapping[Any, Any] = {}
    else:
        qrels = bundle.get("qrels", {})
    if input_kind != "corpus" and "qrels" not in bundle:
        _issue(report, "missing_qrels", "retrieval and evidence_qa bundles require qrels", path="qrels")
    if input_kind == "corpus" and ("qrels" not in bundle or not qrels):
        report["needs_queries"] = True
    else:
        report["needs_queries"] = False

    document_map: dict[str, Mapping[str, Any]] = {}
    document_indices: defaultdict[str, list[int]] = defaultdict(list)
    document_hashes: defaultdict[str, list[str]] = defaultdict(list)
    for index, record in enumerate(documents):
        path = f"documents[{index}]"
        if not isinstance(record, Mapping):
            _issue(report, "document_type", "document entries must be objects", path=path)
            continue
        doc_id = record.get("_id")
        if not _valid_id(doc_id):
            _issue(report, "document_id", "document _id must be a nonempty string", path=f"{path}._id")
        else:
            document_indices[doc_id].append(index)
            if doc_id not in document_map:
                document_map[doc_id] = record
        text = record.get("text")
        if not _is_nonempty_text(text):
            _issue(report, "document_text", "document text must be nonempty", path=f"{path}.text")
        elif _byte_length(text) > DEFAULT_MAX_DOCUMENT_BYTES:
            _issue(
                report,
                "document_too_large",
                "document text exceeds the 1,000,000-byte compatibility limit",
                path=f"{path}.text",
                bytes=_byte_length(text),
                limit=DEFAULT_MAX_DOCUMENT_BYTES,
            )
        if "title" in record and record["title"] is not None and not isinstance(record["title"], str):
            _issue(report, "document_title_type", "document title must be a string when supplied", path=f"{path}.title")
        if "language" in record and record["language"] is not None and not _valid_id(record["language"]):
            _issue(report, "document_language_type", "document language must be a nonempty string when supplied", path=f"{path}.language")
        if "metadata" in record and record["metadata"] is not None and not isinstance(record["metadata"], Mapping):
            _issue(report, "document_metadata_type", "document metadata must be an object", path=f"{path}.metadata")
        if isinstance(text, str):
            document_hashes[_text_hash(text)].append(doc_id if isinstance(doc_id, str) else f"<index:{index}>")

    query_map: dict[str, Mapping[str, Any]] = {}
    query_indices: defaultdict[str, list[int]] = defaultdict(list)
    query_hashes: defaultdict[str, list[str]] = defaultdict(list)
    for index, record in enumerate(queries):
        path = f"queries[{index}]"
        if not isinstance(record, Mapping):
            _issue(report, "query_type", "query entries must be objects", path=path)
            continue
        query_id = record.get("_id")
        if not _valid_id(query_id):
            _issue(report, "query_id", "query _id must be a nonempty string", path=f"{path}._id")
        else:
            query_indices[query_id].append(index)
            if query_id not in query_map:
                query_map[query_id] = record
        text = record.get("text")
        if not _is_nonempty_text(text):
            _issue(report, "query_text", "query text must be nonempty", path=f"{path}.text")
        elif _byte_length(text) > DEFAULT_MAX_QUERY_BYTES:
            _issue(report, "query_too_large", "query text exceeds the 1,000,000-byte compatibility limit", path=f"{path}.text")
        if "language" in record and record["language"] is not None and not _valid_id(record["language"]):
            _issue(report, "query_language_type", "query language must be a nonempty string when supplied", path=f"{path}.language")
        if "metadata" in record and record["metadata"] is not None and not isinstance(record["metadata"], Mapping):
            _issue(report, "query_metadata_type", "query metadata must be an object", path=f"{path}.metadata")
        if isinstance(text, str):
            query_hashes[_normalized_query_text(text)].append(query_id if isinstance(query_id, str) else f"<index:{index}>")

    for identifier, indices in document_indices.items():
        if len(indices) > 1:
            first = documents[indices[0]]
            conflict = any(documents[index] != first for index in indices[1:])
            _issue(
                report,
                "duplicate_document_id",
                "document IDs must be unique",
                path=f"documents[{indices[0]}]._id",
                document_id=identifier,
                indices=indices,
                conflict=conflict,
            )
    for identifier, indices in query_indices.items():
        if len(indices) > 1:
            first = queries[indices[0]]
            conflict = any(queries[index] != first for index in indices[1:])
            _issue(
                report,
                "duplicate_query_id",
                "query IDs must be unique",
                path=f"queries[{indices[0]}]._id",
                query_id=identifier,
                indices=indices,
                conflict=conflict,
            )

    qrel_counts: dict[str, int] = {}
    unknown_qids: list[str] = []
    unknown_docids: defaultdict[str, list[str]] = defaultdict(list)
    if isinstance(qrels, Mapping):
        for raw_query_id, raw_doc_scores in qrels.items():
            query_id = raw_query_id if isinstance(raw_query_id, str) else str(raw_query_id)
            if query_id not in query_map:
                unknown_qids.append(query_id)
                _issue(report, "unknown_qrel_query", "qrels contains an unknown query ID", path=f"qrels.{query_id}", query_id=raw_query_id)
            if not isinstance(raw_doc_scores, Mapping):
                _issue(report, "qrel_row_type", "each qrels query value must be an object", path=f"qrels.{query_id}")
                continue
            count = 0
            for raw_doc_id, score in raw_doc_scores.items():
                doc_id = raw_doc_id if isinstance(raw_doc_id, str) else str(raw_doc_id)
                if doc_id not in document_map:
                    unknown_docids[query_id].append(doc_id)
                    _issue(report, "unknown_qrel_document", "qrels contains an unknown document ID", path=f"qrels.{query_id}.{doc_id}", doc_id=raw_doc_id)
                if isinstance(score, bool) or not isinstance(score, numbers.Real) or not math.isfinite(float(score)) or float(score) <= 0:
                    _issue(report, "invalid_qrel_score", "qrels scores must be finite positive numbers; booleans are rejected", path=f"qrels.{query_id}.{doc_id}", score_type=type(score).__name__)
                else:
                    count += 1
            qrel_counts[query_id] = count
    report["counts"]["qrels_queries"] = len(qrels)
    report["counts"]["positive_pairs"] = sum(qrel_counts.values())
    report["qrel_counts"] = {key: qrel_counts[key] for key in sorted(qrel_counts)}
    report["unknown_qrel_queries"] = sorted(unknown_qids)
    report["unknown_qrel_documents"] = {key: sorted(value) for key, value in sorted(unknown_docids.items())}

    if input_kind != "corpus":
        for query_id in query_map:
            if query_id not in qrels:
                _issue(report, "query_without_qrels", "retrieval query has no qrels row", path=f"queries.{query_id}", query_id=query_id)
            elif not qrel_counts.get(query_id):
                _issue(report, "query_without_positive", "query qrels row has no valid positive labels", path=f"qrels.{query_id}", query_id=query_id)
    positive_over_limit = sorted(
        query_id for query_id, count in qrel_counts.items() if count > DEFAULT_MAX_POSITIVE_PER_QUERY
    )
    report["statistics"] = {
        "documents": _collection_stats(documents, provenance=provenance, include_lengths=True),
        "queries": _collection_stats(queries, provenance=provenance, include_lengths=False),
        "positive_count_gt_10": positive_over_limit,
        "positive_count_gt_10_count": len(positive_over_limit),
        "positive_count_over_10": positive_over_limit,
        "query_duplicate_groups": [
            {"query_ids": sorted(ids), "count": len(ids)}
            for ids in sorted(query_hashes.values(), key=lambda group: sorted(group))
            if len(ids) > 1
        ],
    }
    report["language_stats"] = {
        "documents": report["statistics"]["documents"]["language"],
        "queries": report["statistics"]["queries"]["language"],
        "verified_documents": report["statistics"]["documents"]["verified_language"],
        "verified_queries": report["statistics"]["queries"]["verified_language"],
    }
    report["annotation_sources"] = {
        "documents": report["statistics"]["documents"]["annotation_source"],
        "queries": report["statistics"]["queries"]["annotation_source"],
    }

    report["full_duplicate_groups"] = [
        {"text_sha256": digest, "document_ids": sorted(ids), "count": len(ids)}
        for digest, ids in sorted(document_hashes.items())
        if len(ids) > 1
    ]
    report["near_duplicate"] = _near_duplicate_report(documents)

    text_to_ids: defaultdict[str, list[str]] = defaultdict(list)
    for doc_id, record in document_map.items():
        text = record.get("text")
        if isinstance(text, str):
            text_to_ids[_text_hash(text)].append(doc_id)
    equivalent_unlabelled: list[dict[str, Any]] = []
    for query_id, raw_doc_scores in qrels.items():
        if query_id not in query_map or not isinstance(raw_doc_scores, Mapping):
            continue
        positive_ids = {
            doc_id
            for doc_id, score in raw_doc_scores.items()
            if isinstance(doc_id, str)
            and isinstance(score, numbers.Real)
            and not isinstance(score, bool)
            and math.isfinite(float(score))
            and float(score) > 0
        }
        for doc_id in sorted(positive_ids):
            record = document_map.get(doc_id)
            if not record or not isinstance(record.get("text"), str):
                continue
            equivalents = [candidate for candidate in text_to_ids[_text_hash(record["text"])] if candidate not in positive_ids]
            if equivalents:
                item = {
                    "query_id": query_id,
                    "positive_document_id": doc_id,
                    "equivalent_unlabelled_document_ids": sorted(equivalents),
                }
                equivalent_unlabelled.append(item)
                _review(
                    report,
                    "equivalent_unlabelled_document",
                    "a positive document has exact-text equivalents outside its qrels row",
                    query_id=query_id,
                    document_ids=[doc_id, *sorted(equivalents)],
                )
    report["equivalent_unlabelled_documents"] = equivalent_unlabelled

    evidence_report = _audit_evidence(evidence, document_map=document_map, query_ids=set(query_map), report=report)
    report["evidence_report"] = evidence_report
    report["cross_language_families"] = _cross_language_families(documents, queries)
    report["judgments"] = {
        "count": len(judgments),
        "type": "list",
    }
    review_queue = bundle.get("review_queue", [])
    if not isinstance(review_queue, list):
        _issue(report, "invalid_review_queue", "review_queue must be a list", path="review_queue")
    else:
        for index, entry in enumerate(review_queue):
            if not isinstance(entry, Mapping):
                _issue(report, "invalid_review_entry", "review entries must be objects", path=f"review_queue[{index}]")
                continue
            _review(report, "pending_review", "A retained annotation still requires review",
                    query_id=entry.get("query_id"),
                    document_ids=[entry["doc_id"]] if entry.get("doc_id") else [],
                    reason=entry.get("reason"), path=f"review_queue[{index}]")
    _finish_status(report)
    return report


def _cross_language_families(
    documents: Sequence[Any], queries: Sequence[Any]
) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for kind, records in (("document", documents), ("query", queries)):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            metadata = _metadata(record)
            if metadata is None:
                continue
            family = metadata.get("family_id")
            translation_of = metadata.get("translation_of")
            family_key: str | None = None
            relation = "family_id"
            if _valid_id(family):
                family_key = f"family_id:{family}"
            elif _valid_id(translation_of):
                family_key = f"translation_of:{translation_of}"
                relation = "translation_of"
            if family_key is None:
                continue
            groups[family_key].append(
                {
                    "kind": kind,
                    "_id": record.get("_id"),
                    "language": _declared_language(record),
                    "relation": relation,
                }
            )
    result = []
    for family, members in sorted(groups.items()):
        languages = sorted({member["language"] for member in members})
        if len(languages) > 1:
            result.append({"family": family, "languages": languages, "members": members, "explicit_metadata": True})
    return result


def _normalize_text(value: str, *, unicode_form: str | None, whitespace: bool) -> str:
    if unicode_form:
        value = unicodedata.normalize(unicode_form, value)
    if whitespace:
        value = _WHITESPACE_RE.sub(" ", value).strip()
    return value


def _normalization_changes(bundle: dict[str, Any], *, unicode_form: str | None, whitespace: bool) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    if not unicode_form and not whitespace:
        return changes
    for field in ("documents", "queries"):
        records = bundle.get(field)
        if not isinstance(records, list):
            continue
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                continue
            old = record["text"]
            new = _normalize_text(old, unicode_form=unicode_form, whitespace=whitespace)
            if old == new:
                continue
            record["text"] = new
            changes.append(
                {
                    "field": field,
                    "index": index,
                    "_id": record.get("_id"),
                    "operations": {
                        "unicode_form": unicode_form,
                        "whitespace": bool(whitespace),
                    },
                    "before_sha256": _text_hash(old),
                    "after_sha256": _text_hash(new),
                    "before_bytes": _byte_length(old),
                    "after_bytes": _byte_length(new),
                }
            )
    return changes


def _policy_int(policy: Mapping[str, Any], key: str, default: int) -> int:
    value = policy.get(key, default)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) <= 0:
        raise ValueError(f"policy {key} must be a positive integer")
    return int(value)


def prepare_bundle(bundle: dict, policy: dict | None = None) -> tuple[dict, dict]:
    """Return a cleaned bundle and an audit trail of every exclusion.

    Invalid or oversized documents are removed as records.  Any query whose
    qrels depend on such a document is removed as a whole; its qrels are never
    silently shortened.  Near-duplicate records are retained by default.
    """

    if not isinstance(bundle, dict):
        raise TypeError("bundle must be an ordinary dict")
    if policy is None:
        policy_map: Mapping[str, Any] = {}
    elif isinstance(policy, dict):
        policy_map = policy
    else:
        raise TypeError("policy must be a dict or None")
    cleaned = copy.deepcopy(bundle)
    max_doc_bytes = _policy_int(policy_map, "max_document_bytes", DEFAULT_MAX_DOCUMENT_BYTES)
    max_query_bytes = _policy_int(policy_map, "max_query_bytes", DEFAULT_MAX_QUERY_BYTES)
    max_positive = policy_map.get("max_positive_per_query", DEFAULT_MAX_POSITIVE_PER_QUERY)
    if max_positive is not None:
        if isinstance(max_positive, bool) or not isinstance(max_positive, numbers.Integral) or int(max_positive) <= 0:
            raise ValueError("policy max_positive_per_query must be a positive integer or None")
        max_positive = int(max_positive)
    unicode_form = policy_map.get("unicode_normalization")
    if unicode_form is not None and unicode_form not in {"NFC", "NFKC", "NFD", "NFKD"}:
        raise ValueError("policy unicode_normalization must be one of NFC, NFKC, NFD, or NFKD")
    whitespace = bool(policy_map.get("normalize_whitespace", False))
    normalization_changes = _normalization_changes(cleaned, unicode_form=unicode_form, whitespace=whitespace)
    exclusions: list[dict[str, Any]] = []

    documents = cleaned.get("documents")
    if not isinstance(documents, list):
        documents = []
    document_groups: defaultdict[str, list[tuple[int, Any]]] = defaultdict(list)
    for index, record in enumerate(documents):
        if isinstance(record, Mapping) and _valid_id(record.get("_id")):
            document_groups[record["_id"]].append((index, record))
    retained_documents: list[Any] = []
    retained_document_map: dict[str, Mapping[str, Any]] = {}
    removed_document_ids: set[str] = set()
    conflicting_document_ids: set[str] = set()
    for index, record in enumerate(documents):
        path = f"documents[{index}]"
        if not isinstance(record, Mapping):
            _exclude(exclusions, "document", index, "invalid_record", path=path)
            continue
        doc_id = record.get("_id")
        if not _valid_id(doc_id):
            _exclude(exclusions, "document", index, "invalid_id", path=f"{path}._id")
            continue
        if len(document_groups[doc_id]) > 1:
            first_record = document_groups[doc_id][0][1]
            conflict = any(other != first_record for _, other in document_groups[doc_id][1:])
            if conflict:
                conflicting_document_ids.add(doc_id)
                removed_document_ids.add(doc_id)
                _exclude(exclusions, "document", doc_id, "duplicate_id_conflict", path=f"{path}._id")
                continue
            if doc_id in retained_document_map:
                _exclude(exclusions, "document", doc_id, "duplicate_id", path=f"{path}._id")
                continue
        text = record.get("text")
        if not _is_nonempty_text(text):
            removed_document_ids.add(doc_id)
            _exclude(exclusions, "document", doc_id, "empty_or_invalid_text", path=f"{path}.text")
            continue
        if _byte_length(text) > max_doc_bytes:
            removed_document_ids.add(doc_id)
            _exclude(exclusions, "document", doc_id, "document_too_large", path=f"{path}.text", bytes=_byte_length(text), limit=max_doc_bytes)
            continue
        retained_documents.append(record)
        retained_document_map[doc_id] = record

    queries = cleaned.get("queries")
    if not isinstance(queries, list):
        queries = []
    query_groups: defaultdict[str, list[tuple[int, Any]]] = defaultdict(list)
    for index, record in enumerate(queries):
        if isinstance(record, Mapping) and _valid_id(record.get("_id")):
            query_groups[record["_id"]].append((index, record))
    retained_queries: list[Any] = []
    retained_query_map: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(queries):
        path = f"queries[{index}]"
        if not isinstance(record, Mapping):
            _exclude(exclusions, "query", index, "invalid_record", path=path)
            continue
        query_id = record.get("_id")
        if not _valid_id(query_id):
            _exclude(exclusions, "query", index, "invalid_id", path=f"{path}._id")
            continue
        if query_id in retained_query_map:
            _exclude(exclusions, "query", query_id, "duplicate_id", path=f"{path}._id")
            continue
        if len(query_groups[query_id]) > 1:
            _exclude(exclusions, "query", query_id, "duplicate_id", path=f"{path}._id")
            # Keep the first row, as for exact duplicate documents.
            if query_groups[query_id][0][0] != index:
                continue
        text = record.get("text")
        if not _is_nonempty_text(text):
            _exclude(exclusions, "query", query_id, "empty_or_invalid_text", path=f"{path}.text")
            continue
        if _byte_length(text) > max_query_bytes:
            _exclude(exclusions, "query", query_id, "query_too_large", path=f"{path}.text", bytes=_byte_length(text), limit=max_query_bytes)
            continue
        retained_queries.append(record)
        retained_query_map[query_id] = record

    raw_qrels = cleaned.get("qrels", {})
    if not isinstance(raw_qrels, Mapping):
        raw_qrels = {}
    kind = cleaned.get("input_kind")
    cleaned_qrels: dict[str, dict[str, numbers.Real]] = {}
    removed_query_ids: set[str] = set()
    for raw_query_id, raw_scores in raw_qrels.items():
        query_id = raw_query_id if isinstance(raw_query_id, str) else str(raw_query_id)
        if query_id not in retained_query_map:
            _exclude(exclusions, "qrel", query_id, "unknown_or_removed_query", path=f"qrels.{query_id}")
            continue
        if not isinstance(raw_scores, Mapping):
            removed_query_ids.add(query_id)
            _exclude(exclusions, "query", query_id, "invalid_qrels_row", path=f"qrels.{query_id}")
            continue
        row: dict[str, numbers.Real] = {}
        row_bad = False
        for raw_doc_id, score in raw_scores.items():
            doc_id = raw_doc_id if isinstance(raw_doc_id, str) else str(raw_doc_id)
            if doc_id not in retained_document_map:
                row_bad = True
                _exclude(exclusions, "query", query_id, "dependent_on_invalid_document", path=f"qrels.{query_id}.{doc_id}", document_id=doc_id)
                continue
            if isinstance(score, bool) or not isinstance(score, numbers.Real) or not math.isfinite(float(score)) or float(score) <= 0:
                row_bad = True
                _exclude(exclusions, "query", query_id, "invalid_qrel_score", path=f"qrels.{query_id}.{doc_id}")
                continue
            row[doc_id] = score
        if row_bad:
            removed_query_ids.add(query_id)
            _exclude(exclusions, "query", query_id, "invalid_or_incomplete_qrels", path=f"qrels.{query_id}")
            continue
        if max_positive is not None and len(row) > max_positive:
            removed_query_ids.add(query_id)
            _exclude(exclusions, "query", query_id, "positive_count_over_limit", path=f"qrels.{query_id}", count=len(row), limit=max_positive)
            continue
        if kind != "corpus" and not row:
            removed_query_ids.add(query_id)
            _exclude(exclusions, "query", query_id, "no_positive_qrels", path=f"qrels.{query_id}")
            continue
        cleaned_qrels[query_id] = row

    for query_id in list(retained_query_map):
        if kind != "corpus" and query_id not in cleaned_qrels:
            removed_query_ids.add(query_id)
            _exclude(exclusions, "query", query_id, "missing_qrels", path=f"queries.{query_id}")

    if removed_query_ids:
        retained_queries = [record for record in retained_queries if record.get("_id") not in removed_query_ids]
        retained_query_map = {record["_id"]: record for record in retained_queries}
        for query_id in removed_query_ids:
            cleaned_qrels.pop(query_id, None)

    # Exact-text equivalents outside a positive qrels row are review risks.
    text_to_ids: defaultdict[str, list[str]] = defaultdict(list)
    for doc_id, record in retained_document_map.items():
        text = record.get("text")
        if isinstance(text, str):
            text_to_ids[_text_hash(text)].append(doc_id)
    equivalent_exclusions = bool(policy_map.get("exclude_review_required", True))
    for query_id, row in list(cleaned_qrels.items()):
        equivalents: set[str] = set()
        for doc_id in row:
            text = retained_document_map[doc_id].get("text")
            if isinstance(text, str):
                equivalents.update(candidate for candidate in text_to_ids[_text_hash(text)] if candidate not in row)
        if equivalents:
            if equivalent_exclusions:
                removed_query_ids.add(query_id)
                _exclude(exclusions, "query", query_id, "equivalent_unlabelled_document", path=f"qrels.{query_id}", document_ids=sorted(equivalents))
            else:
                _exclude(exclusions, "query", query_id, "review_required_equivalent_unlabelled_document", path=f"qrels.{query_id}", document_ids=sorted(equivalents))
    if removed_query_ids:
        retained_queries = [record for record in retained_queries if record.get("_id") not in removed_query_ids]
        retained_query_map = {record["_id"]: record for record in retained_queries}
        for query_id in removed_query_ids:
            cleaned_qrels.pop(query_id, None)

    # Optional near-duplicate deletion is explicit.  Dropping a positive
    # document also removes every dependent query, rather than trimming gold.
    if bool(policy_map.get("drop_near_duplicates", False)) and len(retained_documents) > 1:
        near = _near_duplicate_report(retained_documents)
        drop_ids: set[str] = set()
        for pair in near["pairs"]:
            drop_ids.add(pair["document_ids"][1])
        if drop_ids:
            for doc_id in sorted(drop_ids):
                _exclude(exclusions, "document", doc_id, "near_duplicate_policy", path=f"documents.{doc_id}")
            retained_documents = [record for record in retained_documents if record.get("_id") not in drop_ids]
            retained_document_map = {record["_id"]: record for record in retained_documents}
            for query_id, row in list(cleaned_qrels.items()):
                if any(doc_id in drop_ids for doc_id in row):
                    removed_query_ids.add(query_id)
                    _exclude(exclusions, "query", query_id, "dependent_on_near_duplicate_document", path=f"qrels.{query_id}")
                    cleaned_qrels.pop(query_id, None)
            if removed_query_ids:
                retained_queries = [record for record in retained_queries if record.get("_id") not in removed_query_ids]
                retained_query_map = {record["_id"]: record for record in retained_queries}

    raw_evidence = cleaned.get("evidence", [])
    if not isinstance(raw_evidence, list):
        raw_evidence = []
    retained_evidence: list[Any] = []
    evidence_keys: set[tuple[Any, ...]] = set()
    for index, item in enumerate(raw_evidence):
        path = f"evidence[{index}]"
        if not isinstance(item, Mapping):
            _exclude(exclusions, "evidence", index, "invalid_record", path=path)
            continue
        query_id = item.get("query_id")
        doc_id = item.get("doc_id")
        if not _valid_id(query_id) or query_id not in retained_query_map:
            _exclude(exclusions, "evidence", index, "unknown_or_removed_query", path=path)
            continue
        if not _valid_id(doc_id) or doc_id not in retained_document_map:
            _exclude(exclusions, "evidence", index, "unknown_or_removed_document", path=path)
            continue
        quote = item.get("quote")
        start = item.get("start")
        end = item.get("end")
        valid_locator = False
        text = retained_document_map[doc_id].get("text")
        if isinstance(quote, str) and quote:
            occurrences = _find_all(text, quote) if isinstance(text, str) else []
            offsets_valid = (
                _valid_offset(start)
                and _valid_offset(end)
                and isinstance(text, str)
                and 0 <= start < end <= len(text)
                and text[start:end] == quote
            )
            valid_locator = len(occurrences) == 1 or offsets_valid
        elif quote is None:
            valid_locator = (
                _valid_offset(start)
                and _valid_offset(end)
                and isinstance(text, str)
                and 0 <= start < end <= len(text)
            )
        if not valid_locator:
            _exclude(exclusions, "evidence", index, "invalid_quote_or_offset", path=path,
                     query_id=query_id, document_id=doc_id)
            # Removing a bad locator must not turn an unverified annotation
            # into a passing release. Keep the source claim for human review.
            cleaned.setdefault("review_queue", []).append({
                "query_id": query_id, "doc_id": doc_id,
                "reason": "invalid_quote_or_offset", "evidence": copy.deepcopy(dict(item)),
            })
            continue
        key = (query_id, doc_id, quote, start, end)
        if key in evidence_keys:
            _exclude(exclusions, "evidence", index, "duplicate_evidence", path=path)
            continue
        evidence_keys.add(key)
        retained_evidence.append(item)

    cleaned["documents"] = retained_documents
    cleaned["queries"] = retained_queries
    cleaned["qrels"] = cleaned_qrels
    cleaned["evidence"] = retained_evidence
    if not isinstance(cleaned.get("judgments"), list):
        cleaned["judgments"] = []
    audit = audit_bundle(cleaned, strict=False)
    audit["exclusions"] = exclusions
    audit["normalization_changes"] = normalization_changes
    audit["preparation"] = {
        "policy": _safe_value(dict(policy_map)),
        "input_counts": {
            "documents": len(documents),
            "queries": len(queries),
            "evidence": len(raw_evidence),
        },
        "output_counts": {
            "documents": len(retained_documents),
            "queries": len(retained_queries),
            "evidence": len(retained_evidence),
        },
        "excluded_query_ids": sorted(removed_query_ids),
        "excluded_document_ids": sorted(removed_document_ids | conflicting_document_ids),
        "text_rewritten": bool(normalization_changes),
        "near_duplicates_removed": bool(policy_map.get("drop_near_duplicates", False)),
    }
    if normalization_changes:
        audit["warnings"].append(
            {
                "code": "text_normalized",
                "message": "explicit text normalization may invalidate existing evidence offsets",
                "severity": "warning",
            }
        )
    if any("equivalent" in str(entry.get("reason", "")) for entry in exclusions):
        if audit.get("status") not in {"invalid", "needs_queries"}:
            audit["status"] = "review_required"
    return cleaned, audit


def coverage_report(bundles: list[dict]) -> dict:
    """Build aggregate coverage and provenance statistics for several bundles."""

    if not isinstance(bundles, list):
        raise TypeError("bundles must be a list")
    result: dict[str, Any] = {
        "report_version": REPORT_SCHEMA_VERSION,
        "bundle_count": len(bundles),
        "status": "pass",
        "bundles": [],
        "totals": {
            "documents": 0,
            "queries": 0,
            "qrels_queries": 0,
            "positive_pairs": 0,
            "evidence": 0,
        },
        "language_stats": {"documents": Counter(), "queries": Counter()},
        "domain_stats": Counter(),
        "genre_stats": Counter(),
        "annotation_source_stats": Counter(),
        "id_overlap": {"documents": {}, "queries": {}},
        "invalid_bundles": [],
    }
    document_occurrences: defaultdict[str, list[int]] = defaultdict(list)
    query_occurrences: defaultdict[str, list[int]] = defaultdict(list)
    for bundle_index, bundle in enumerate(bundles):
        if not isinstance(bundle, dict):
            result["invalid_bundles"].append(bundle_index)
            audit = audit_bundle(bundle, strict=False)
            result["bundles"].append({"index": bundle_index, "audit": audit})
            continue
        audit = audit_bundle(bundle, strict=False)
        documents = bundle.get("documents") if isinstance(bundle.get("documents"), list) else []
        queries = bundle.get("queries") if isinstance(bundle.get("queries"), list) else []
        evidence = bundle.get("evidence") if isinstance(bundle.get("evidence"), list) else []
        qrels = bundle.get("qrels") if isinstance(bundle.get("qrels"), Mapping) else {}
        positive_pairs = 0
        for row in qrels.values():
            if isinstance(row, Mapping):
                positive_pairs += sum(
                    int(isinstance(score, numbers.Real) and not isinstance(score, bool) and math.isfinite(float(score)) and float(score) > 0)
                    for score in row.values()
                )
        source = bundle.get("source") if isinstance(bundle.get("source"), Mapping) else {}
        provenance = bundle.get("provenance") if isinstance(bundle.get("provenance"), Mapping) else {}
        row = {
            "index": bundle_index,
            "source": _safe_value(source),
            "input_kind": bundle.get("input_kind"),
            "documents": len(documents),
            "queries": len(queries),
            "qrels_queries": len(qrels),
            "positive_pairs": positive_pairs,
            "evidence": len(evidence),
            "needs_queries": bool(audit.get("needs_queries", False)),
            "valid": bool(audit.get("valid", False)),
            "audit_status": audit.get("status"),
            "audit_errors": len(audit.get("errors", [])),
        }
        result["bundles"].append(row)
        for key in result["totals"]:
            result["totals"][key] += row[key]
        doc_stats = audit.get("statistics", {}).get("documents", {})
        query_stats = audit.get("statistics", {}).get("queries", {})
        for language, count in doc_stats.get("language", {}).items():
            result["language_stats"]["documents"][language] += count
        for language, count in query_stats.get("language", {}).items():
            result["language_stats"]["queries"][language] += count
        for record in [*documents, *queries]:
            if not isinstance(record, Mapping):
                continue
            metadata = _metadata(record)
            domain = _value_from_record_or_provenance(record, "domain", provenance)
            genre = _value_from_record_or_provenance(record, "genre", provenance)
            result["domain_stats"][domain] += 1
            result["genre_stats"][genre] += 1
            result["annotation_source_stats"][_annotation_source(record, provenance)] += 1
        for record in documents:
            if isinstance(record, Mapping) and _valid_id(record.get("_id")):
                document_occurrences[record["_id"]].append(bundle_index)
        for record in queries:
            if isinstance(record, Mapping) and _valid_id(record.get("_id")):
                query_occurrences[record["_id"]].append(bundle_index)
    result["language_stats"] = {
        key: _counter_dict(value) for key, value in result["language_stats"].items()
    }
    result["domain_stats"] = _counter_dict(result["domain_stats"])
    result["genre_stats"] = _counter_dict(result["genre_stats"])
    result["annotation_source_stats"] = _counter_dict(result["annotation_source_stats"])
    result["id_overlap"] = {
        "documents": {key: sorted(value) for key, value in sorted(document_occurrences.items()) if len(set(value)) > 1},
        "queries": {key: sorted(value) for key, value in sorted(query_occurrences.items()) if len(set(value)) > 1},
    }
    statuses = {row.get("audit_status") for row in result["bundles"] if isinstance(row, Mapping)}
    if "invalid" in statuses or result["invalid_bundles"]:
        result["status"] = "invalid"
    elif "needs_queries" in statuses:
        result["status"] = "needs_queries"
    elif "review_required" in statuses:
        result["status"] = "review_required"
    return result


def _explicit_group_keys(record: Mapping[str, Any], kind: str) -> list[str]:
    metadata = _metadata(record)
    if metadata is None:
        return []
    keys: list[str] = []
    for field in ("family_id", "group_id", "source_id", "source_group", "source"):
        value = metadata.get(field)
        if _valid_id(value):
            keys.append(f"{field}:{value}")
    translation_of = metadata.get("translation_of")
    if _valid_id(translation_of):
        keys.append(f"translation_of:{translation_of}")
    return keys


class _UnionFind:
    def __init__(self, nodes: Iterable[tuple[str, str]]) -> None:
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in self.parent}

    def find(self, node: tuple[str, str]) -> tuple[str, str]:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def assign_splits(
    bundle: dict,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict:
    """Assign connected query/document components to train/validation/test.

    Explicit qrels, evidence links, exact duplicate text, ``family_id``,
    ``translation_of``, and source group metadata form the connected
    components.  Components stay intact even when that makes the requested
    ratios imperfect; large shared components are reported in
    ``large_components``.
    """

    if not isinstance(bundle, dict):
        raise TypeError("bundle must be an ordinary dict")
    if not isinstance(ratios, (tuple, list)) or len(ratios) != 3:
        raise ValueError("ratios must contain train, validation, and test values")
    parsed_ratios: list[float] = []
    for ratio in ratios:
        if isinstance(ratio, bool) or not isinstance(ratio, numbers.Real) or not math.isfinite(float(ratio)) or float(ratio) < 0:
            raise ValueError("split ratios must be finite nonnegative numbers")
        parsed_ratios.append(float(ratio))
    ratio_sum = sum(parsed_ratios)
    if ratio_sum <= 0:
        raise ValueError("split ratios must have a positive sum")
    parsed_ratios = [ratio / ratio_sum for ratio in parsed_ratios]
    names = ("train", "validation", "test")

    documents = bundle.get("documents") if isinstance(bundle.get("documents"), list) else []
    queries = bundle.get("queries") if isinstance(bundle.get("queries"), list) else []
    document_ids = [record.get("_id") for record in documents if isinstance(record, Mapping) and _valid_id(record.get("_id"))]
    query_ids = [record.get("_id") for record in queries if isinstance(record, Mapping) and _valid_id(record.get("_id"))]
    nodes = [("document", identifier) for identifier in document_ids] + [("query", identifier) for identifier in query_ids]
    union_find = _UnionFind(nodes)
    node_set = set(nodes)

    group_members: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for kind, records in (("document", documents), ("query", queries)):
        for record in records:
            if not isinstance(record, Mapping) or not _valid_id(record.get("_id")):
                continue
            node = (kind, record["_id"])
            text = record.get("text")
            if isinstance(text, str):
                # Exact duplicate bodies/queries are a leakage link even when
                # the source omitted family_id.  Keep the original IDs; only
                # the split graph uses this content hash.
                duplicate_key = (
                    f"{kind}_text_sha256:{_text_hash(text)}"
                    if kind == "document"
                    else f"{kind}_text:{_normalized_query_text(text)}"
                )
                group_members[duplicate_key].append(node)
            for group_key in _explicit_group_keys(record, kind):
                group_members[group_key].append(node)
            metadata = _metadata(record)
            translation_of = metadata.get("translation_of") if metadata else None
            if _valid_id(translation_of):
                for candidate_kind in ("document", "query"):
                    candidate = (candidate_kind, translation_of)
                    if candidate in node_set:
                        union_find.union(node, candidate)
    for members in group_members.values():
        for member in members[1:]:
            union_find.union(members[0], member)

    qrels = bundle.get("qrels") if isinstance(bundle.get("qrels"), Mapping) else {}
    for query_id, row in qrels.items():
        query_node = ("query", query_id)
        if query_node not in node_set or not isinstance(row, Mapping):
            continue
        for doc_id in row:
            doc_node = ("document", doc_id)
            if doc_node in node_set:
                union_find.union(query_node, doc_node)
    evidence = bundle.get("evidence") if isinstance(bundle.get("evidence"), list) else []
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        query_node = ("query", item.get("query_id"))
        doc_node = ("document", item.get("doc_id"))
        if query_node in node_set and doc_node in node_set:
            union_find.union(query_node, doc_node)

    components: defaultdict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for node in nodes:
        components[union_find.find(node)].append(node)
    component_list = [sorted(members) for members in components.values()]
    component_list.sort(key=lambda members: (len(members), members))
    rng = random.Random(seed)
    rng.shuffle(component_list)
    component_list.sort(key=lambda members: -len(members))
    assigned_nodes = [0, 0, 0]
    total_nodes = len(nodes)
    assignments: dict[tuple[str, str], str] = {}
    component_rows: list[dict[str, Any]] = []
    for members in component_list:
        scores = []
        for index, ratio in enumerate(parsed_ratios):
            if ratio == 0:
                scores.append(float("inf"))
            else:
                target = total_nodes * ratio
                scores.append(assigned_nodes[index] / target if target else 0.0)
        best = min(range(3), key=lambda index: (scores[index], index))
        split = names[best]
        for node in members:
            assignments[node] = split
        assigned_nodes[best] += len(members)
        component_rows.append(
            {
                "component_id": len(component_rows),
                "split": split,
                "size": len(members),
                "queries": sorted(identifier for kind, identifier in members if kind == "query"),
                "documents": sorted(identifier for kind, identifier in members if kind == "document"),
            }
        )
    query_splits = {identifier: assignments[("query", identifier)] for identifier in sorted(query_ids)}
    document_splits = {identifier: assignments[("document", identifier)] for identifier in sorted(document_ids)}
    document_families = {
        record["_id"]: _explicit_group_keys(record, "document")
        for record in documents
        if isinstance(record, Mapping) and _valid_id(record.get("_id"))
    }
    query_families = {
        record["_id"]: _explicit_group_keys(record, "query")
        for record in queries
        if isinstance(record, Mapping) and _valid_id(record.get("_id"))
    }
    large_threshold = max(10, int(math.ceil(max(total_nodes, 1) * 0.25)))
    large_components = [
        row
        for row in component_rows
        if row["size"] >= large_threshold and row["queries"] and row["documents"]
    ]
    return {
        "split_version": "dci-data-split-v1",
        "seed": seed,
        "ratios": {name: ratio for name, ratio in zip(names, parsed_ratios)},
        "splits": {
            "queries": query_splits,
            "documents": document_splits,
        },
        "query_splits": query_splits,
        "document_splits": document_splits,
        "document_families": document_families,
        "query_families": query_families,
        "components": component_rows,
        "component_count": len(component_rows),
        "counts": {
            "nodes": total_nodes,
            "queries": len(query_ids),
            "documents": len(document_ids),
            "train": assigned_nodes[0],
            "validation": assigned_nodes[1],
            "test": assigned_nodes[2],
        },
        "large_components": large_components,
        "warnings": [
            {
                "code": "large_shared_component",
                "message": "a connected family/source component is large; ratios were not enforced by splitting it",
                "component": row,
            }
            for row in large_components
        ],
        "cross_split_components": [],
    }


def write_report(report: dict, path: Path) -> None:
    """Write a report as JSON, or a small self-contained HTML view."""

    if not isinstance(report, dict):
        raise TypeError("report must be a dict")
    if not isinstance(path, Path):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.casefold() == ".html":
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        document = "<!doctype html><meta charset='utf-8'><title>DCI quality report</title><pre>" + html.escape(payload) + "</pre>\n"
        path.write_text(document, encoding="utf-8")
    else:
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "assign_splits",
    "audit_bundle",
    "coverage_report",
    "prepare_bundle",
    "write_report",
]
