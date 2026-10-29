"""Bounded, immutable document candidate pools for document evaluation.

The selector in this module deliberately consumes saved rankings only.  It
does not import an embedding library or start a model service.  Rankings are
validated against the original document IDs before any pool is written.
Host-side pool metadata keeps selected IDs in deterministic score order (with
explicit tie rules); runtime code constructs the isolated per-query workspace
from those IDs and the task-level shared corpus.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any
from urllib.parse import quote, unquote
import uuid


from dci_bench.data.document_ids import DOCUMENT_ID_POLICY, validate_source_document_ids
from dci_bench.protocol.document_pools import load_document_pools

_POOL_CONFIG = load_document_pools()
POOL_SIZES: dict[str, int] = {name: tier["capacity"] for name, tier in _POOL_CONFIG["tiers"].items()}
"""Named candidate-pool capacities used by the document profiles."""

POOL_SCHEMA_VERSION = "dci-document-pools-20261021-v1"
V2_POOL_SCHEMA_VERSION = "dci-document-pools-20261029-v2"
SUPPORTED_POOL_SCHEMA_VERSIONS = {POOL_SCHEMA_VERSION, V2_POOL_SCHEMA_VERSION}
RANKINGS_SCHEMA_VERSION = "dci-saved-retrieval-v1"
EXPECTED_EMBEDDING_MODEL_PATH = _POOL_CONFIG["retriever"]["model_path"]
EXPECTED_TRUST_REMOTE_CODE = _POOL_CONFIG["retriever"]["trust_remote_code"]

_METADATA_FILES = (
    "candidate_pools.json",
    "doc_id_files.json",
    "file_inventory.json",
    "manifest.json",
    "original_qrels.json",
    "qrels.json",
    "queries.jsonl",
    "source_doc_ids.json",
    "source_identity.json",
)

_CONTENT_HASH_VERIFICATION_POLICIES = {
    True: "content-hashes-verified",
    False: "declared-source-digests",
}


@dataclass(frozen=True)
class DocumentPoolBuildResult:
    """Paths and counts returned by :func:`prepare_document_pools`."""

    suite_root: Path
    task: str
    metadata_dir: Path
    workspace_dir: Path
    candidate_pools_path: Path
    completion_path: Path
    num_queries: int
    corpus_size: int


def stable_json_sha256(value: Any) -> str:
    """Hash JSON values independently of whitespace and dictionary order."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_string_id(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _validate_id_sequence(values: Sequence[str] | None, *, name: str) -> list[str]:
    if values is None:
        raise ValueError(f"{name} is required")
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of document IDs")
    try:
        result = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of document IDs") from exc
    for value in result:
        if not _is_string_id(value):
            raise ValueError(f"{name} contains a non-string or empty ID: {value!r}")
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in result:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"duplicate {name} IDs: {duplicates[:5]}")
    return result


def _validate_qrels(
    qrels: Mapping[str, float] | None,
    corpus_set: set[str],
    *,
    query_id: str,
) -> dict[str, float]:
    if qrels is None:
        return {}
    if not isinstance(qrels, Mapping):
        raise ValueError(f"qrels for query {query_id!r} must be a mapping")
    clean: dict[str, float] = {}
    unknown: list[str] = []
    for doc_id, score in qrels.items():
        if not _is_string_id(doc_id):
            raise ValueError(f"qrels for query {query_id!r} has an invalid document ID")
        if doc_id not in corpus_set:
            unknown.append(doc_id)
            continue
        if isinstance(score, bool):
            raise ValueError(f"qrels score must be numeric for {query_id!r}/{doc_id!r}")
        try:
            numeric_score = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"qrels score must be numeric for {query_id!r}/{doc_id!r}") from exc
        if not math.isfinite(numeric_score):
            raise ValueError(f"qrels score must be finite for {query_id!r}/{doc_id!r}")
        clean[doc_id] = numeric_score
    if unknown:
        raise ValueError(
            f"qrels for query {query_id!r} reference unknown corpus IDs: {sorted(unknown)[:5]}"
        )
    return clean


def _validate_scores(
    scores: Mapping[str, float] | None,
    corpus_set: set[str],
    *,
    query_id: str,
) -> dict[str, float] | None:
    if scores is None:
        return None
    if not isinstance(scores, Mapping):
        raise ValueError(f"scores for query {query_id!r} must be a mapping")
    clean: dict[str, float] = {}
    unknown: list[str] = []
    for doc_id, score in scores.items():
        if not _is_string_id(doc_id):
            raise ValueError(f"scores for query {query_id!r} has an invalid document ID")
        if doc_id not in corpus_set:
            unknown.append(doc_id)
            continue
        if isinstance(score, bool):
            raise ValueError(f"retrieval score must be numeric for {query_id!r}/{doc_id!r}")
        try:
            numeric_score = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"retrieval score must be numeric for {query_id!r}/{doc_id!r}"
            ) from exc
        if not math.isfinite(numeric_score):
            raise ValueError(f"retrieval score must be finite for {query_id!r}/{doc_id!r}")
        clean[doc_id] = numeric_score
    if unknown:
        raise ValueError(
            f"scores for query {query_id!r} reference unknown corpus IDs: {sorted(unknown)[:5]}"
        )
    return clean


def _validate_scored_ranking(
    ranking: Sequence[str],
    scores: Mapping[str, float],
    positive_ids: Sequence[str],
    corpus_set: set[str],
    *,
    query_id: str,
    require_complete: bool = False,
) -> None:
    missing_ranked_scores = [doc_id for doc_id in ranking if doc_id not in scores]
    if missing_ranked_scores:
        raise ValueError(
            f"scores for query {query_id!r} are missing ranked IDs: "
            f"{missing_ranked_scores[:5]}"
        )
    missing_positive_scores = [doc_id for doc_id in positive_ids if doc_id not in scores]
    if missing_positive_scores:
        raise ValueError(
            f"scores for query {query_id!r} are missing positive IDs: "
            f"{missing_positive_scores[:5]}"
        )
    if require_complete and set(ranking) != corpus_set:
        missing = sorted(corpus_set - set(ranking))
        raise ValueError(
            f"scored ranking for query {query_id!r} must cover the corpus; "
            f"missing IDs: {missing[:5]}"
        )
    for previous, current in zip(ranking, ranking[1:]):
        if scores[previous] < scores[current]:
            raise ValueError(
                f"ranking scores for query {query_id!r} must be non-increasing"
            )


def _score_ordered_ids(ids: Sequence[str], scores: Mapping[str, float]) -> list[str]:
    """Sort scored IDs descending while retaining the input order for ties."""

    return sorted(ids, key=lambda doc_id: -scores[doc_id])


def _scoped_rng(seed: int, task: str, query_id: str) -> tuple[random.Random, int, str]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not _is_string_id(task) or not _is_string_id(query_id):
        raise ValueError("task and query_id must be non-empty strings")
    scope = json.dumps([seed, task, query_id], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()
    scoped_seed = int(digest, 16)
    return random.Random(scoped_seed), scoped_seed, digest


def select_candidate_ids(
    corpus_ids: Sequence[str],
    ranked_ids: Sequence[str] | None,
    qrels: Mapping[str, float] | None,
    *,
    scores: Mapping[str, float] | None = None,
    capacity: int,
    seed: int,
    task: str,
    query_id: str,
) -> tuple[list[str], dict[str, Any]]:
    """Select one bounded candidate pool for one query.

    This public entrypoint validates the corpus on every call.  The builder
    uses the private validated-corpus helper below so a large source corpus is
    checked once per suite instead of once per query.
    """

    corpus = _validate_id_sequence(corpus_ids, name="corpus")
    return _select_candidate_ids_validated(
        corpus,
        set(corpus),
        ranked_ids,
        qrels,
        scores=scores,
        capacity=capacity,
        seed=seed,
        task=task,
        query_id=query_id,
    )


def _select_candidate_ids_validated(
    corpus: Sequence[str],
    corpus_set: set[str],
    ranked_ids: Sequence[str] | None,
    qrels: Mapping[str, float] | None,
    *,
    scores: Mapping[str, float] | None = None,
    capacity: int,
    seed: int,
    task: str,
    query_id: str,
) -> tuple[list[str], dict[str, Any]]:
    """Select using a corpus sequence and set validated by the caller."""

    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    clean_qrels = _validate_qrels(qrels, corpus_set, query_id=query_id)
    clean_scores = _validate_scores(scores, corpus_set, query_id=query_id)
    positive_ids = sorted(doc_id for doc_id, score in clean_qrels.items() if score > 0)
    rng, scoped_seed, scope_hash = _scoped_rng(seed, task, query_id)

    if len(corpus) <= capacity:
        ranking_complete = False
        if ranked_ids is None:
            if clean_scores is not None:
                raise ValueError(
                    f"ranking is required for scored full-corpus query {query_id!r}"
                )
            selected = sorted(corpus_set)
            initial_top = list(selected)
            selected_scores: list[float] | None = None
            ordering = "unscored"
        else:
            ranking = _validate_ranking_ids(ranked_ids, corpus_set, query_id=query_id)
            ranking_complete = set(ranking) == corpus_set
            if clean_scores is None:
                selected = sorted(corpus_set)
                initial_top = list(selected)
                selected_scores = None
                ordering = "unscored"
            else:
                _validate_scored_ranking(
                    ranking,
                    clean_scores,
                    positive_ids,
                    corpus_set,
                    query_id=query_id,
                    require_complete=True,
                )
                selected = _score_ordered_ids(ranking, clean_scores)
                initial_top = list(ranking)
                selected_scores = [clean_scores[doc_id] for doc_id in selected]
                ordering = "scored"
        audit: dict[str, Any] = {
            "selection_version": POOL_SCHEMA_VERSION,
            "task": task,
            "query_id": query_id,
            "seed": seed,
            "rng_scope": [seed, task, query_id],
            "rng_scoped_seed_sha256": scope_hash,
            "rng_algorithm": "python-random-sample-v1",
            "capacity": capacity,
            "source_size": len(corpus),
            "initial_top_ids": initial_top,
            "injected_positive_ids": [],
            "removed_nonpositive_ids": [],
            "selected_ids": list(selected),
            "selected_scores": selected_scores,
            "ordering": ordering,
            "positive_count": len(positive_ids),
            "injection_count": 0,
            "eviction_count": 0,
            "ranking_required": False,
            "ranking_complete": ranking_complete,
        }
        return selected, audit

    if ranked_ids is None:
        raise ValueError(
            f"saved ranking is required for query {query_id!r} when corpus size "
            f"{len(corpus)} exceeds capacity {capacity}"
        )
    ranking = _validate_ranking_ids(
        ranked_ids,
        corpus_set,
        query_id=query_id,
        min_length=capacity,
    )
    if len(positive_ids) > capacity:
        raise ValueError(
            f"positive qrels exceed candidate capacity for query {query_id!r}: "
            f"{len(positive_ids)} positives > capacity {capacity}"
        )
    if clean_scores is None:
        raise ValueError(
            f"saved ranking scores are required for query {query_id!r} when corpus size "
            f"{len(corpus)} exceeds capacity {capacity}"
        )
    _validate_scored_ranking(
        ranking,
        clean_scores,
        positive_ids,
        corpus_set,
        query_id=query_id,
    )
    initial_top = ranking[:capacity]
    initial_set = set(initial_top)
    ranking_set = set(ranking)
    injected = [doc_id for doc_id in positive_ids if doc_id not in initial_set]
    nonpositive = sorted(
        doc_id
        for doc_id in initial_set
        if clean_qrels.get(doc_id, 0.0) <= 0
    )
    if len(injected) > len(nonpositive):
        # This should be mathematically impossible when positive_count <= cap,
        # but keeping the guard makes the no-label-loss invariant explicit.
        raise ValueError(
            f"cannot inject all positive qrels for query {query_id!r}: "
            f"need {len(injected)} non-positive evictions, have {len(nonpositive)}"
        )
    removed = rng.sample(nonpositive, len(injected)) if injected else []
    removed_set = set(removed)
    selected_set = (initial_set - removed_set) | set(injected)
    if len(selected_set) != capacity:
        raise AssertionError("candidate selection changed pool cardinality")
    if not set(positive_ids).issubset(selected_set):
        raise AssertionError("candidate selection dropped a positive qrel")
    # Preserve backend rank order for positives that occur after the initial
    # top-capacity slice.  A positive absent from the saved ranking has no
    # backend order, so equal-score absent positives are appended by original
    # ID deterministically after all ranked IDs.
    unranked_injected = [doc_id for doc_id in injected if doc_id not in ranking_set]
    unranked_injected.sort(key=lambda doc_id: (-clean_scores[doc_id], doc_id))
    selected_before_score = [
        doc_id for doc_id in ranking if doc_id in selected_set
    ] + unranked_injected
    selected = _score_ordered_ids(selected_before_score, clean_scores)
    selected_scores = [clean_scores[doc_id] for doc_id in selected]
    audit = {
        "selection_version": POOL_SCHEMA_VERSION,
        "task": task,
        "query_id": query_id,
        "seed": seed,
        "rng_scope": [seed, task, query_id],
        "rng_scoped_seed_sha256": scope_hash,
        "rng_algorithm": "python-random-sample-v1",
        "capacity": capacity,
        "source_size": len(corpus),
        "initial_top_ids": initial_top,
        "injected_positive_ids": injected,
        "removed_nonpositive_ids": removed,
        "selected_ids": selected,
        "selected_scores": selected_scores,
        "ordering": "scored",
        "positive_count": len(positive_ids),
        "injection_count": len(injected),
        "eviction_count": len(removed),
        "ranking_required": True,
        "ranking_complete": len(ranking) == len(corpus),
    }
    return selected, audit


def _validate_ranking_ids(
    ranked_ids: Sequence[str],
    corpus_set: set[str],
    *,
    query_id: str,
    min_length: int | None = None,
) -> list[str]:
    ranking = _validate_id_sequence(ranked_ids, name=f"ranking for query {query_id!r}")
    unknown = sorted(set(ranking) - corpus_set)
    if unknown:
        raise ValueError(
            f"ranking for query {query_id!r} contains unknown corpus IDs: {unknown[:5]}"
        )
    if min_length is not None and len(ranking) < min_length:
        raise ValueError(
            f"incomplete ranking for query {query_id!r}; requires at least "
            f"{min_length} known IDs, got {len(ranking)}"
        )
    return ranking


def build_source_identity(
    *,
    task: str,
    corpus_ids: Sequence[str],
    query_ids: Sequence[str],
    source_doc_ids: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, float]],
    query_records: Sequence[Mapping[str, Any]],
    corpus_file_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Return the exact identity required by a saved retrieval export.

    The query hash covers complete query records, including text, rather than
    only their IDs.  The corpus hash covers a canonical alias-to-rendered-file
    digest map, and the mapping hash covers the complete alias-to-original-ID
    map supplied by the short-ID preparation step.  ``task`` and ``qrels``
    remain part of this helper's call contract for callers that already carry
    them, but are intentionally not encoded as alternate identity fields.
    """

    del task, qrels
    canonical_corpus_ids = _validate_id_sequence(corpus_ids, name="corpus")
    canonical_query_ids = _validate_id_sequence(query_ids, name="query")
    canonical_mapping = dict(sorted(source_doc_ids.items()))
    if not canonical_mapping or any(
        not _is_string_id(alias) or not _is_string_id(original)
        for alias, original in canonical_mapping.items()
    ):
        raise ValueError("source_doc_ids must map non-empty aliases to original IDs")
    if len(set(canonical_mapping.values())) != len(canonical_mapping):
        raise ValueError("source_doc_ids must map each original ID to one alias")
    if not set(canonical_corpus_ids).issubset(set(canonical_mapping.values())):
        unknown = sorted(set(canonical_corpus_ids) - set(canonical_mapping.values()))
        raise ValueError(f"corpus IDs are absent from source_doc_ids: {unknown[:5]}")

    canonical_records: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    for record in query_records:
        if not isinstance(record, Mapping):
            raise ValueError("query_records must contain JSON objects")
        row = dict(record)
        query_id = row.get("_id", row.get("query_id"))
        if not _is_string_id(query_id) or query_id in seen_query_ids:
            raise ValueError("query_records contain invalid or duplicate query IDs")
        seen_query_ids.add(query_id)
        canonical_records.append(row)
    if seen_query_ids != set(canonical_query_ids):
        missing = sorted(set(canonical_query_ids) - seen_query_ids)
        extra = sorted(seen_query_ids - set(canonical_query_ids))
        raise ValueError(
            f"query_records IDs differ from query_ids; missing={missing[:5]} extra={extra[:5]}"
        )
    canonical_records.sort(key=lambda row: row.get("_id", row.get("query_id")))

    if not isinstance(corpus_file_hashes, Mapping) or not corpus_file_hashes:
        raise ValueError("corpus_file_hashes must be a non-empty alias-to-SHA256 mapping")
    canonical_files = dict(sorted(corpus_file_hashes.items()))
    if any(
        not _is_string_id(alias) or not isinstance(file_hash, str) or not file_hash
        for alias, file_hash in canonical_files.items()
    ):
        raise ValueError("corpus_file_hashes must map non-empty aliases to non-empty hashes")
    if not set(canonical_files).issubset(set(canonical_mapping)):
        unknown = sorted(set(canonical_files) - set(canonical_mapping))
        raise ValueError(f"corpus_file_hashes contain unknown source aliases: {unknown[:5]}")

    # Keep the exact three keys stable so ranking exporters cannot silently
    # substitute an ID-only hash or an old synonym.
    return {
        "queries_hash": stable_json_sha256(canonical_records),
        "corpus_files_sha256": stable_json_sha256(canonical_files),
        "source_doc_ids_hash": stable_json_sha256(canonical_mapping),
    }


def load_saved_rankings(
    path: Path,
    *,
    task: str,
    source_identity: Mapping[str, Any],
    corpus_ids: Sequence[str],
    query_ids: Sequence[str],
    capacity: int,
    qrels: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load and validate a saved original-ID ranking export.

    The accepted JSON representation is intentionally canonical:
    ``{"rankings": {qid: [doc_id, ...]}, "scores": {qid: {doc_id: score}}}``.
    Rankings are original document IDs.  A large source corpus requires a
    score for every saved ranked ID and every positive qrel (a positive may be
    outside the truncated ranking and will be injected later).  For a corpus
    within the cap, scores are optional; if present they require a complete
    corpus ranking and are used to order the full pool.  The model and source
    identity fields are mandatory.
    """

    path = Path(path)
    try:
        if path.suffix == ".gz":
            import gzip
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"saved rankings are not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("saved rankings must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version != RANKINGS_SCHEMA_VERSION:
        raise ValueError(
            f"saved rankings schema_version must be {RANKINGS_SCHEMA_VERSION!r}"
        )
    saved_task = payload.get("task")
    if saved_task != task:
        raise ValueError(f"saved ranking task {saved_task!r} does not match {task!r}")
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("saved rankings must contain a model object")
    model_path = model.get("path")
    trust_remote_code = model.get("trust_remote_code")
    if model_path != EXPECTED_EMBEDDING_MODEL_PATH:
        raise ValueError(
            "saved rankings model path must be " + EXPECTED_EMBEDDING_MODEL_PATH
        )
    if trust_remote_code is not EXPECTED_TRUST_REMOTE_CODE:
        raise ValueError("saved rankings must record trust_remote_code=true")

    expected_identity = source_identity
    saved_identity = payload.get("source_identity")
    if not isinstance(saved_identity, Mapping):
        raise ValueError("saved rankings must contain a source_identity object")
    _validate_saved_identity(saved_identity, expected_identity)

    ranking_rows = payload.get("rankings")
    if not isinstance(ranking_rows, Mapping):
        raise ValueError("saved rankings must contain a rankings mapping")
    parsed: dict[str, list[str]] = {}
    for query_id, row in ranking_rows.items():
        if not _is_string_id(query_id) or query_id in parsed:
            raise ValueError("saved rankings contain an invalid or duplicate query ID")
        if not isinstance(row, list):
            raise ValueError(f"saved ranking for {query_id!r} must be a list of original IDs")
        parsed[query_id] = _validate_id_sequence(row, name=f"ranking for query {query_id!r}")
    expected_query_ids = set(query_ids)
    if set(parsed) != expected_query_ids:
        missing = sorted(expected_query_ids - set(parsed))
        extra = sorted(set(parsed) - expected_query_ids)
        raise ValueError(
            f"saved rankings query IDs incomplete or unexpected; missing={missing[:5]} extra={extra[:5]}"
        )
    corpus_set = set(corpus_ids)
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    large_corpus = len(corpus_set) > capacity
    scores_rows = payload.get("scores")
    if scores_rows is None:
        if large_corpus:
            raise ValueError(
                "saved rankings must contain scores when corpus size exceeds capacity"
            )
        parsed_scores: dict[str, dict[str, float] | None] = {
            query_id: None for query_id in parsed
        }
    else:
        if not isinstance(scores_rows, Mapping):
            raise ValueError("saved rankings scores must be a query-to-score mapping")
        if set(scores_rows) != set(parsed):
            missing = sorted(set(parsed) - set(scores_rows))
            extra = sorted(set(scores_rows) - set(parsed))
            raise ValueError(
                "saved rankings score query IDs differ from rankings; "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        parsed_scores = {}
        for query_id in sorted(parsed):
            score_row = scores_rows[query_id]
            if not isinstance(score_row, Mapping):
                raise ValueError(f"scores for query {query_id!r} must be a mapping")
            parsed_scores[query_id] = _validate_scores(
                score_row,
                corpus_set,
                query_id=query_id,
            )

    if qrels is None:
        qrels_by_query: Mapping[str, Mapping[str, float]] = {}
    else:
        if not isinstance(qrels, Mapping):
            raise ValueError("qrels must be a query-to-document mapping")
        unknown_qrels = sorted(set(qrels) - set(parsed))
        if unknown_qrels:
            raise ValueError(f"qrels contain unknown query IDs: {unknown_qrels[:5]}")
        qrels_by_query = qrels

    result: dict[str, dict[str, Any]] = {}
    for query_id in sorted(parsed):
        ranking = _validate_ranking_ids(
            parsed[query_id],
            corpus_set,
            query_id=query_id,
            min_length=capacity if large_corpus else None,
        )
        query_qrels = qrels_by_query.get(query_id)
        if query_qrels is None:
            positive_ids: list[str] = []
        else:
            clean_qrels = _validate_qrels(query_qrels, corpus_set, query_id=query_id)
            positive_ids = [doc_id for doc_id, score in clean_qrels.items() if score > 0]
        query_scores = parsed_scores[query_id]
        if large_corpus and query_scores is None:
            raise ValueError(
                f"saved rankings scores are required for query {query_id!r} when corpus size "
                f"{len(corpus_set)} exceeds capacity {capacity}"
            )
        if query_scores is not None:
            _validate_scored_ranking(
                ranking,
                query_scores,
                positive_ids,
                corpus_set,
                query_id=query_id,
                require_complete=not large_corpus,
            )
        result[query_id] = {
            "ranked_ids": ranking,
            "scores": query_scores,
        }
    return result


def _validate_saved_identity(saved: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    required = ("queries_hash", "corpus_files_sha256", "source_doc_ids_hash")
    for key in required:
        saved_value = saved.get(key)
        expected_value = expected.get(key)
        if not isinstance(saved_value, str) or not saved_value:
            raise ValueError(f"saved rankings source identity missing exact key {key!r}")
        if not isinstance(expected_value, str) or not expected_value:
            raise ValueError(f"local source identity missing exact key {key!r}")
        if saved_value != expected_value:
            raise ValueError(f"saved rankings {key} does not match source")
    optional = {"source_manifest_sha256", "source_inventory_sha256"}
    for key in sorted(set(saved) & optional):
        expected_value = expected.get(key)
        if not isinstance(expected_value, str) or not expected_value:
            raise ValueError(f"local source identity missing exact key {key!r}")
        if saved[key] != expected_value:
            raise ValueError(f"saved rankings {key} does not match source")
    unexpected = sorted(set(saved) - set(required) - optional)
    if unexpected:
        raise ValueError(
            "saved rankings source_identity contains unsupported hash keys: "
            + ", ".join(unexpected)
        )


def safe_query_dirname(query_id: str) -> str:
    if not _is_string_id(query_id):
        raise ValueError("query ID must be a non-empty string")
    dirname = quote(query_id, safe="-_.~")
    if dirname in {"", ".", ".."} or "/" in dirname or "\\" in dirname:
        raise ValueError(f"query ID cannot be represented as a safe directory: {query_id!r}")
    return dirname


def _safe_task_name(task: str) -> None:
    if not _is_string_id(task) or task in {".", ".."} or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in task
    ):
        raise ValueError(f"unsafe task name: {task!r}")


def compact_candidate_payload(payload: dict[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    """Losslessly omit derived lists and store audit IDs in the short namespace."""
    import copy
    result = copy.deepcopy(payload)
    if "storage_encoding" in result:
        raise ValueError("candidate payload is already encoded")
    inverse = {original: alias for alias, original in mapping.items()}
    result["storage_encoding"] = "short-id-derived-audit-v1"
    for record in result["pools"].values():
        originals = [mapping[alias] for alias in record["candidate_doc_ids"]]
        if record.pop("candidate_original_doc_ids") != originals:
            raise ValueError("candidate original IDs differ before compaction")
        audit = record["audit"]
        if audit.pop("selected_ids") != originals or audit.pop("selected_scores") != record["candidate_scores"]:
            raise ValueError("candidate audit differs before compaction")
        for key in ("initial_top_ids", "injected_positive_ids", "removed_nonpositive_ids"):
            audit[key] = [inverse[original] for original in audit[key]]
    return result


def _load_json(path: Path, *, label: str) -> Any:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"missing {label}: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {path}") from exc
    if path.name == "candidate_pools.json" and isinstance(payload, dict) and "storage_encoding" in payload:
        if payload.pop("storage_encoding") != "short-id-derived-audit-v1":
            raise ValueError("unknown candidate pool storage encoding")
        mapping = _load_json(path.parent / "source_doc_ids.json", label="document IDs")
        try:
            for record in payload["pools"].values():
                if "candidate_original_doc_ids" in record or "selected_ids" in record["audit"] or "selected_scores" in record["audit"]:
                    raise ValueError("compact candidate pool contains conflicting derived fields")
                originals = [mapping[alias] for alias in record["candidate_doc_ids"]]
                record["candidate_original_doc_ids"] = originals
                audit = record["audit"]
                audit["selected_ids"] = originals.copy()
                audit["selected_scores"] = record["candidate_scores"]
                for key in ("initial_top_ids", "injected_positive_ids", "removed_nonpositive_ids"):
                    audit[key] = [mapping[alias] for alias in audit[key]]
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid compact candidate pool") from exc
    return payload


def _locate_source_file(source_suite: Path, task: str, name: str) -> Path:
    candidates = (
        source_suite / "metadata" / task / name,
        source_suite / "metadata" / name,
        source_suite / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"missing {name}; checked: " + ", ".join(str(candidate) for candidate in candidates)
    )


def _load_source_qrels(path: Path) -> dict[str, dict[str, float]]:
    payload = _load_json(path, label="original qrels")
    if isinstance(payload, Mapping) and isinstance(payload.get("qrels"), Mapping):
        payload = payload["qrels"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("original qrels must be a non-empty query-to-document mapping")
    result: dict[str, dict[str, float]] = {}
    for query_id, docs in payload.items():
        if not _is_string_id(query_id) or not isinstance(docs, Mapping):
            raise ValueError("original qrels must map query IDs to document-score mappings")
        if query_id in result:
            raise ValueError(f"duplicate qrels query ID: {query_id}")
        clean: dict[str, float] = {}
        for doc_id, score in docs.items():
            if not _is_string_id(doc_id) or isinstance(score, bool):
                raise ValueError(f"invalid qrels row for {query_id!r}/{doc_id!r}")
            try:
                value = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid qrels score for {query_id!r}/{doc_id!r}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite qrels score for {query_id!r}/{doc_id!r}")
            clean[doc_id] = value
        result[query_id] = clean
    return result


def _load_raw_qrels_payload(path: Path) -> Mapping[str, Any]:
    payload = _load_json(path, label="qrels")
    if isinstance(payload, Mapping) and isinstance(payload.get("qrels"), Mapping):
        payload = payload["qrels"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("qrels must be a non-empty query-to-document mapping")
    return payload


def _load_source_mapping(path: Path) -> dict[str, str]:
    payload = _load_json(path, label="source document mapping")
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("source_doc_ids.json must be a non-empty alias-to-original mapping")
    return validate_source_document_ids(payload)


def _load_doc_id_files(path: Path, mapping: Mapping[str, str]) -> dict[str, str]:
    payload = _load_json(path, label="document filename mapping")
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("doc_id_files.json must be a non-empty alias-to-filename mapping")
    result: dict[str, str] = {}
    filenames: set[str] = set()
    for alias, filename in payload.items():
        if not _is_string_id(alias) or not isinstance(filename, str) or not filename:
            raise ValueError("doc_id_files.json must map non-empty aliases to filenames")
        if alias not in mapping:
            raise ValueError(f"doc_id_files.json references unknown alias: {alias!r}")
        if filename != f"{alias}.md":
            raise ValueError("short-ID corpus filenames must be <alias>.md")
        if alias in result or filename in filenames:
            raise ValueError("doc_id_files.json contains duplicate aliases or filenames")
        if Path(filename).name != filename or Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError(f"unsafe document filename in doc_id_files.json: {filename!r}")
        result[alias] = filename
        filenames.add(filename)
    return result


def _source_corpus_files(
    corpus_dir: Path,
    mapping: Mapping[str, str],
    *,
    doc_id_files: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"missing source corpus directory: {corpus_dir}")
    result: dict[str, Path] = {}
    for path in sorted(corpus_dir.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source corpus contains a non-regular file: {path}")
        raw_name = path.name[:-3] if path.name.endswith(".md") else path.name
        candidates = (raw_name, unquote(raw_name), path.name, unquote(path.name))
        if doc_id_files is not None:
            candidates = tuple(
                candidate
                for candidate in candidates
                if candidate in doc_id_files and doc_id_files[candidate] == path.name
            )
        aliases = [candidate for candidate in candidates if candidate in mapping]
        if len(set(aliases)) != 1:
            if not aliases:
                raise ValueError(f"source corpus file has no source_doc_ids alias: {path.name}")
            raise ValueError(f"source corpus file ambiguously maps to aliases: {path.name}")
        alias = aliases[0]
        if alias in result:
            raise ValueError(f"duplicate source corpus alias: {alias}")
        result[alias] = path
    if doc_id_files is None and set(result) != set(mapping):
        missing = sorted(set(mapping) - set(result))
        extra = sorted(set(result) - set(mapping))
        raise ValueError(f"source corpus/mapping mismatch: missing={missing[:5]} extra={extra[:5]}")
    if doc_id_files is not None and set(result) != set(doc_id_files):
        missing = sorted(set(doc_id_files) - set(result))
        extra = sorted(set(result) - set(doc_id_files))
        raise ValueError(f"source corpus/doc_id_files mismatch: missing={missing[:5]} extra={extra[:5]}")
    return result


def _validate_source_corpus_inventory(
    path: Path,
    files: Mapping[str, Path],
    mapping: Mapping[str, str],
    doc_id_files: Mapping[str, str],
    verify_content_hashes: bool = True,
) -> tuple[str, dict[str, str]]:
    """Validate an optional expansion ``corpus_files.jsonl`` inventory.

    ``verify_content_hashes=False`` trusts each inventory row's declared
    digest after validating its identity, filename, and byte count.  This is
    the bounded path used when a source release has already sealed the
    inventory; the default continues to verify every source document.
    """

    if not isinstance(verify_content_hashes, bool):
        raise ValueError("verify_content_hashes must be a boolean")

    original_to_alias = {original: alias for alias, original in mapping.items()}
    seen_aliases: set[str] = set()
    verified_hashes: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid source corpus inventory JSON at line {line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"source corpus inventory row {line_number} is not an object")
            original = row.get("doc_id")
            filename = row.get("filename")
            size = row.get("bytes")
            digest = row.get("sha256")
            if (
                not _is_string_id(original)
                or not isinstance(filename, str)
                or not filename
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or not isinstance(digest, str)
                or not digest
            ):
                raise ValueError(f"invalid source corpus inventory row {line_number}")
            if "source_doc_id" in row:
                # v3 inventory explicitly distinguishes the two namespaces.
                alias = original
                if alias not in mapping or row["source_doc_id"] != mapping[alias]:
                    raise ValueError("source corpus inventory alias/source ID mismatch")
            else:
                # Legacy inventories stored original IDs in doc_id. Never guess
                # the namespace from an ID that happens to look like an alias.
                alias = original_to_alias.get(original)
            if alias is None or alias not in files or alias in seen_aliases:
                raise ValueError(f"source corpus inventory has unknown or duplicate document: {original!r}")
            source_path = files[alias]
            if filename != doc_id_files[alias] or source_path.name != filename:
                raise ValueError(f"source corpus inventory filename mismatch: {original!r}")
            if source_path.is_symlink() or not source_path.is_file():
                raise ValueError(f"source corpus inventory references a non-regular file: {original!r}")
            if source_path.stat().st_size != size:
                if verify_content_hashes:
                    raise ValueError(f"source corpus inventory hash mismatch: {original!r}")
                raise ValueError(f"source corpus inventory size mismatch: {original!r}")
            if verify_content_hashes and sha256_file(source_path) != digest:
                raise ValueError(f"source corpus inventory hash mismatch: {original!r}")
            seen_aliases.add(alias)
            verified_hashes[alias] = digest
    if seen_aliases != set(files):
        missing = sorted(set(files) - seen_aliases)
        extra = sorted(seen_aliases - set(files))
        raise ValueError(
            f"source corpus inventory does not cover active corpus: missing={missing[:5]} extra={extra[:5]}"
        )
    return sha256_file(path), verified_hashes


def _load_queries(source_suite: Path, task: str, qrels: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = (
        source_suite / "metadata" / task / "queries.jsonl",
        source_suite / "metadata" / task / "queries.json",
        source_suite / "metadata" / "queries.jsonl",
        source_suite / "metadata" / "queries.json",
    )
    query_file = next((candidate for candidate in candidates if candidate.is_file()), None)
    if query_file is None:
        raise FileNotFoundError(
            f"missing prepared query inventory; checked: "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    if query_file.suffix == ".jsonl":
        rows = []
        with query_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows.append(row)
    else:
        rows = _load_json(query_file, label="queries")
        if isinstance(rows, Mapping) and "queries" in rows:
            rows = rows["queries"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ValueError(f"queries file must contain a list: {query_file}")
    ids: list[str] = []
    seen_ids: set[str] = set()
    qrel_ids = set(qrels)
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"queries file contains a non-object row: {query_file}")
        query_id = row.get("_id", row.get("query_id"))
        if not _is_string_id(query_id) or query_id in seen_ids:
            raise ValueError(f"queries file contains an invalid or duplicate query ID: {query_file}")
        ids.append(query_id)
        seen_ids.add(query_id)
    if not seen_ids.issubset(qrel_ids):
        unknown = sorted(seen_ids - qrel_ids)
        raise ValueError(f"query inventory references unknown qrels IDs: {unknown[:5]}")
    # Preserve the source JSONL order for byte-for-byte output.  Hashing and
    # identity comparison canonicalise records separately, so ordering cannot
    # make two otherwise identical query inventories differ.
    return [dict(row) for row in rows]


def _load_query_ids(source_suite: Path, task: str, qrels: Mapping[str, Any]) -> list[str]:
    return [
        row.get("_id", row.get("query_id"))
        for row in _load_queries(source_suite, task, qrels)
    ]


def _load_prepared_source_manifest(source_suite: Path, task: str) -> tuple[Path, dict[str, Any]]:
    path = source_suite / "metadata" / task / "manifest.json"
    payload = _load_json(path, label="prepared source manifest")
    if not isinstance(payload, dict):
        raise ValueError("prepared source manifest must be a JSON object")
    required = (
        "task",
        "dataset",
        "revision",
        "resolved_revision",
        "task_instruction",
        "document_id_policy",
        "qrels_hash",
        "queries_hash",
        "source_doc_ids_hash",
        "original_qrels_hash",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            "prepared source manifest is missing required identity fields: "
            + ", ".join(missing)
        )
    if payload["task"] != task:
        raise ValueError(f"prepared source manifest task {payload['task']!r} does not match {task!r}")
    if payload["document_id_policy"] != DOCUMENT_ID_POLICY:
        raise ValueError(
            f"prepared source manifest document_id_policy must be {DOCUMENT_ID_POLICY!r}"
        )
    for key in required:
        if key == "task":
            continue
        if not isinstance(payload[key], str):
            raise ValueError("prepared source manifest identity fields must be strings")
        if key != "task_instruction" and not payload[key]:
            raise ValueError("prepared source manifest identity fields must be non-empty strings")
    return path, payload


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _copy_bytes(source: Path, destination: Path, *, compute_hash: bool = True) -> str | None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"cannot copy non-regular source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256() if compute_hash else None
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        for chunk in iter(lambda: source_handle.read(4 * 1024 * 1024), b""):
            destination_handle.write(chunk)
            if digest is not None:
                digest.update(chunk)
    return digest.hexdigest() if digest is not None else None


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def prepare_document_pools(
    source_suite: Path,
    output_suite: Path,
    task: str,
    *,
    pool_size: str | None = None,
    capacity: int | None = None,
    seed: int = 0,
    rankings_path: Path | None = None,
    source_qrels: Path | None = None,
    source_doc_ids: Path | None = None,
    source_workspace: Path | None = None,
    verify_content_hashes: bool = True,
) -> DocumentPoolBuildResult:
    """Build a new candidate-pool suite from an existing prepared suite.

    Existing output directories are never overwritten.  The source workspace
    and all labels remain untouched; the selected document union is copied
    once into a task-level shared corpus.  Runtime code builds temporary
    per-query views from the candidate IDs.
    """

    _safe_task_name(task)
    if not isinstance(verify_content_hashes, bool):
        raise ValueError("verify_content_hashes must be a boolean")
    source_suite = Path(source_suite)
    output_suite = Path(output_suite)
    if os.path.lexists(output_suite):
        raise FileExistsError(f"refusing to overwrite existing output suite: {output_suite}")
    source_resolved = source_suite.resolve()
    output_resolved = output_suite.resolve()
    if output_resolved == source_resolved or source_resolved in output_resolved.parents:
        raise ValueError("output suite must be outside the source suite")
    if pool_size is not None:
        if pool_size not in POOL_SIZES:
            raise ValueError(f"unknown pool size {pool_size!r}; choose small, medium, or large")
        named_capacity = POOL_SIZES[pool_size]
        if capacity is not None and capacity != named_capacity:
            raise ValueError("pool_size and capacity disagree")
        capacity = named_capacity
    if capacity is None:
        raise ValueError("one of pool_size or capacity is required")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if source_workspace is None:
        source_workspace = source_suite / "workspaces" / task
    else:
        source_workspace = Path(source_workspace)
    corpus_dir = source_workspace / "corpus"
    if source_qrels is None:
        source_qrels = _locate_source_file(source_suite, task, "original_qrels.json")
    else:
        source_qrels = Path(source_qrels)
    if source_doc_ids is None:
        source_doc_ids = _locate_source_file(source_suite, task, "source_doc_ids.json")
    else:
        source_doc_ids = Path(source_doc_ids)
    source_manifest_path, source_manifest = _load_prepared_source_manifest(source_suite, task)
    source_doc_id_files_path = _locate_source_file(source_suite, task, "doc_id_files.json")
    source_query_path = _locate_source_file(source_suite, task, "queries.jsonl")

    # Parse and validate every source before creating the staging output.
    original_qrels = _load_source_qrels(source_qrels)
    raw_original_qrels = _load_raw_qrels_payload(source_qrels)
    mapping = _load_source_mapping(source_doc_ids)
    doc_id_files = _load_doc_id_files(source_doc_id_files_path, mapping)
    files = _source_corpus_files(corpus_dir, mapping, doc_id_files=doc_id_files)
    source_corpus_inventory_path = source_suite / "metadata" / task / "corpus_files.jsonl"
    source_inventory_sha256: str | None = None
    verified_source_file_hashes: dict[str, str] | None = None
    if source_corpus_inventory_path.is_file():
        source_inventory_sha256, verified_source_file_hashes = _validate_source_corpus_inventory(
            source_corpus_inventory_path,
            files,
            mapping,
            doc_id_files,
            verify_content_hashes=verify_content_hashes,
        )
    elif not verify_content_hashes:
        raise ValueError(
            "verify_content_hashes=False requires a source corpus inventory: "
            f"{source_corpus_inventory_path}"
        )
    original_to_alias = {original: alias for alias, original in mapping.items()}
    query_rows = _load_queries(source_suite, task, original_qrels)
    query_ids = [row.get("_id", row.get("query_id")) for row in query_rows]
    active_qrels: dict[str, dict[str, float]] = {}
    available_originals = {mapping[alias] for alias in files}
    for query_id in query_ids:
        if query_id not in original_qrels:
            raise ValueError(f"active query has no original qrels: {query_id!r}")
        unknown_positive_ids = sorted(
            document_id
            for document_id, score in original_qrels[query_id].items()
            if document_id not in available_originals and score > 0
        )
        if unknown_positive_ids:
            raise ValueError(
                f"active query {query_id!r} references unavailable corpus IDs: "
                f"{unknown_positive_ids[:5]}"
            )
        # Zero and negative labels for unavailable source documents remain in
        # the immutable raw and canonical host qrels; the selector receives
        # only labels whose source documents are available.
        active_qrels[query_id] = {
            document_id: score
            for document_id, score in original_qrels[query_id].items()
            if document_id in available_originals
        }
    corpus_ids = _validate_id_sequence(
        sorted(mapping[alias] for alias in files), name="corpus"
    )
    corpus_ids_set = set(corpus_ids)

    # Validate the hashes emitted by the short-ID/source preparation step.
    source_qrels_canonical_path = _locate_source_file(source_suite, task, "qrels.json")
    source_qrels_canonical = _load_json(source_qrels_canonical_path, label="source canonical qrels")
    if stable_json_sha256(source_qrels_canonical) != source_manifest["qrels_hash"]:
        raise ValueError("prepared source qrels hash does not match its manifest")
    if stable_json_sha256(mapping) != source_manifest["source_doc_ids_hash"]:
        raise ValueError("prepared source_doc_ids hash does not match its manifest")
    if stable_json_sha256(raw_original_qrels) != source_manifest["original_qrels_hash"]:
        raise ValueError("prepared original_qrels hash does not match its manifest")
    canonical_query_rows = sorted(
        query_rows, key=lambda row: row.get("_id", row.get("query_id"))
    )
    if stable_json_sha256(canonical_query_rows) != source_manifest["queries_hash"]:
        raise ValueError("prepared query text hash does not match its manifest")
    if set(source_qrels_canonical) != set(query_ids):
        raise ValueError("prepared canonical qrels query IDs differ from active queries")
    for query_id, docs in source_qrels_canonical.items():
        if not isinstance(docs, Mapping):
            raise ValueError(f"prepared canonical qrels row is invalid: {query_id}")
        for alias, score in docs.items():
            if alias not in mapping:
                raise ValueError(f"prepared canonical qrels contain an unknown alias: {query_id}")
            if alias not in doc_id_files and float(score) > 0:
                raise ValueError(
                    f"prepared canonical qrels contain an unavailable positive alias: {query_id}"
                )
    expected_source_canonical = {
        query_id: {
            original_to_alias[doc_id]: score
            for doc_id, score in sorted(original_qrels[query_id].items())
        }
        for query_id in query_ids
    }
    if source_qrels_canonical != expected_source_canonical:
        raise ValueError("prepared canonical qrels do not match active original labels")
    if verified_source_file_hashes is None:
        corpus_file_hashes = {
            alias: sha256_file(path) for alias, path in sorted(files.items())
        }
    else:
        corpus_file_hashes = {
            alias: verified_source_file_hashes[alias] for alias in sorted(files)
        }
    source_identity = build_source_identity(
        task=task,
        corpus_ids=corpus_ids,
        query_ids=query_ids,
        source_doc_ids=mapping,
        qrels=active_qrels,
        query_records=query_rows,
        corpus_file_hashes=corpus_file_hashes,
    )
    # Keep the saved-ranking identity focused on query/corpus/mapping hashes,
    # while recording the exact prepared release in host-side provenance.
    # Canonical ranking exports containing only the three required hashes are
    # still accepted against these source bytes.
    source_identity["source_manifest_sha256"] = sha256_file(source_manifest_path)
    if source_inventory_sha256 is not None:
        source_identity["source_inventory_sha256"] = source_inventory_sha256
    rankings: dict[str, dict[str, Any]] | None = None
    if rankings_path is not None:
        rankings = load_saved_rankings(
            Path(rankings_path),
            task=task,
            source_identity=source_identity,
            corpus_ids=corpus_ids,
            query_ids=query_ids,
            capacity=capacity,
            qrels=active_qrels,
        )
    elif len(corpus_ids) > capacity:
        raise ValueError(
            f"saved per-query rankings are required because source corpus size "
            f"{len(corpus_ids)} exceeds capacity {capacity}"
        )

    num_queries = len(query_ids)
    corpus_size = len(corpus_ids)
    stage = output_suite.parent / f".{output_suite.name}.staging-{uuid.uuid4().hex}"
    metadata_dir = stage / "metadata" / task
    workspace_dir = stage / "workspaces" / task
    try:
        metadata_dir.mkdir(parents=True, exist_ok=False)
        workspace_dir.mkdir(parents=True, exist_ok=False)
        pool_records: dict[str, dict[str, Any]] = {}
        inventory: dict[str, dict[str, Any]] = {}
        selected_aliases_union: set[str] = set()
        for query_id in query_ids:
            ranking_row = rankings[query_id] if rankings is not None else None
            ranked = ranking_row["ranked_ids"] if ranking_row is not None else None
            ranking_scores = ranking_row["scores"] if ranking_row is not None else None
            selected_original, audit = _select_candidate_ids_validated(
                corpus_ids,
                corpus_ids_set,
                ranked,
                active_qrels[query_id],
                scores=ranking_scores,
                capacity=capacity,
                seed=seed,
                task=task,
                query_id=query_id,
            )
            # Keep score order exactly aligned across original IDs, aliases and
            # scores.  Unscored pools are already deterministic original-ID
            # order from the selector.
            canonical_aliases = [original_to_alias[doc_id] for doc_id in selected_original]
            selected_aliases_union.update(canonical_aliases)
            pool_records[query_id] = {
                "query_id": query_id,
                "candidate_doc_ids": canonical_aliases,
                "candidate_original_doc_ids": list(selected_original),
                "candidate_scores": audit["selected_scores"],
                "ordering": audit["ordering"],
                "num_docs": len(canonical_aliases),
                "audit": audit,
            }

        # Materialise the union once at task scope.  Runtime code creates an
        # isolated per-query view from candidate_doc_ids, so query records do
        # not duplicate document bytes in the prepared suite.
        shared_corpus = workspace_dir / "corpus"
        shared_corpus.mkdir(parents=True, exist_ok=False)
        for alias in sorted(selected_aliases_union):
            source_path = files[alias]
            destination_path = shared_corpus / source_path.name
            expected_source_hash = corpus_file_hashes[alias]
            copied_hash = _copy_bytes(
                source_path,
                destination_path,
                compute_hash=verify_content_hashes,
            )
            if destination_path.stat().st_size != source_path.stat().st_size:
                raise ValueError(f"copied workspace size mismatch: {alias}")
            if verify_content_hashes and copied_hash != expected_source_hash:
                raise ValueError(f"copied workspace hash mismatch: {alias}")
            relative_suite = _relative_path(stage, destination_path)
            inventory[relative_suite] = {
                "alias": alias,
                "original_id": mapping[alias],
                "bytes": destination_path.stat().st_size,
                # In nohash mode this is the source release's declared digest;
                # it is intentionally not presented as a fresh copy hash.
                "sha256": copied_hash if verify_content_hashes else expected_source_hash,
            }

        # Preserve the source label and alias files byte-for-byte in host-only
        # metadata.  Runtime code may consume qrels.json (canonical aliases),
        # while original_qrels.json remains the immutable audit source.
        (metadata_dir / "original_qrels.json").write_bytes(source_qrels.read_bytes())
        (metadata_dir / "source_doc_ids.json").write_bytes(source_doc_ids.read_bytes())
        (metadata_dir / "doc_id_files.json").write_bytes(source_doc_id_files_path.read_bytes())
        (metadata_dir / "queries.jsonl").write_bytes(source_query_path.read_bytes())
        (metadata_dir / "qrels.json").write_bytes(source_qrels_canonical_path.read_bytes())
        _write_json(metadata_dir / "source_identity.json", source_identity)
        _write_json(metadata_dir / "file_inventory.json", inventory)
        candidate_payload = {
            "schema_version": POOL_SCHEMA_VERSION,
            "task": task,
            "pool_size": pool_size or f"capacity-{capacity}",
            "capacity": capacity,
            "seed": seed,
            "source_identity": source_identity,
            "source_release": {
                "source_manifest_sha256": sha256_file(source_manifest_path),
                "source_inventory_sha256": source_inventory_sha256,
            },
            "shared_corpus": "corpus",
            "num_queries": len(query_ids),
            "corpus_size": len(corpus_ids),
            "pools": pool_records,
        }
        _write_json(metadata_dir / "candidate_pools.json", candidate_payload)
        manifest = {
            "schema_version": POOL_SCHEMA_VERSION,
            "task": task,
            "dataset": source_manifest["dataset"],
            "revision": source_manifest["revision"],
            "resolved_revision": source_manifest["resolved_revision"],
            "task_instruction": source_manifest["task_instruction"],
            "document_id_policy": DOCUMENT_ID_POLICY,
            "pool_size": pool_size or f"capacity-{capacity}",
            "capacity": capacity,
            "seed": seed,
            "num_queries": len(query_ids),
            "corpus_size": len(corpus_ids),
            "qrels_hash": source_manifest["qrels_hash"],
            "queries_hash": source_manifest["queries_hash"],
            "source_doc_ids_hash": source_manifest["source_doc_ids_hash"],
            "original_qrels_hash": source_manifest["original_qrels_hash"],
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_inventory_sha256": source_inventory_sha256,
            "source_identity": source_identity,
            "metadata_dir": f"metadata/{task}",
            "workspace_dir": f"workspaces/{task}",
            "shared_corpus_dir": f"workspaces/{task}/corpus",
            "rankings_schema_version": RANKINGS_SCHEMA_VERSION if rankings_path else None,
            "rankings_path": str(rankings_path) if rankings_path else None,
            "verification_policy": _CONTENT_HASH_VERIFICATION_POLICIES[verify_content_hashes],
            "content_hashes_verified": verify_content_hashes,
        }
        _write_json(metadata_dir / "manifest.json", manifest)

        metadata_hashes = {
            name: sha256_file(metadata_dir / name) for name in _METADATA_FILES
        }
        workspace_hashes = {
            relative: row["sha256"] for relative, row in sorted(inventory.items())
        }
        completion = {
            "schema_version": POOL_SCHEMA_VERSION,
            "task": task,
            "metadata_hashes": metadata_hashes,
            "workspace_hashes": workspace_hashes,
            "file_inventory_sha256": sha256_file(metadata_dir / "file_inventory.json"),
            "verification_policy": _CONTENT_HASH_VERIFICATION_POLICIES[verify_content_hashes],
            "content_hashes_verified": verify_content_hashes,
        }
        _write_json(metadata_dir / "_COMPLETE.json", completion)
        # The on-disk JSON is now the source of truth for verification.  Drop
        # the large in-memory ranking/pool objects before loading that JSON
        # again, which matters for many queries and 1k-document pools.
        del (
            rankings,
            pool_records,
            candidate_payload,
            active_qrels,
            inventory,
            metadata_hashes,
            workspace_hashes,
            completion,
            files,
            mapping,
            doc_id_files,
            original_to_alias,
            query_rows,
            query_ids,
            corpus_ids,
            corpus_file_hashes,
            verified_source_file_hashes,
            selected_aliases_union,
            corpus_ids_set,
            available_originals,
            original_qrels,
            raw_original_qrels,
            source_qrels_canonical,
            canonical_query_rows,
            expected_source_canonical,
        )
        verify_document_pool_suite(
            stage,
            task=task,
            verify_content_hashes=verify_content_hashes,
        )
        output_suite.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(output_suite)
    except BaseException as exc:
        # Keep the partial suite for diagnosis and make the failure explicit.
        # A completion marker written before a late verification failure must
        # never be mistaken for a published output.
        if stage.exists():
            unpublished_complete = metadata_dir / "_COMPLETE.json"
            if unpublished_complete.is_file():
                unpublished_complete.rename(stage / "_UNPUBLISHED_COMPLETE.json")
            try:
                _write_json(
                    stage / "_FAILED.json",
                    {
                        "stage": "prepare-document-pools",
                        "exception": type(exc).__name__,
                    },
                )
            except BaseException:
                # Preserve the original build error even if the failure marker
                # itself cannot be written (for example, a full filesystem).
                pass
        raise
    return DocumentPoolBuildResult(
        suite_root=output_suite,
        task=task,
        metadata_dir=output_suite / "metadata" / task,
        workspace_dir=output_suite / "workspaces" / task,
        candidate_pools_path=output_suite / "metadata" / task / "candidate_pools.json",
        completion_path=output_suite / "metadata" / task / "_COMPLETE.json",
        num_queries=num_queries,
        corpus_size=corpus_size,
    )


def build_document_pools(*args: Any, **kwargs: Any) -> DocumentPoolBuildResult:
    """Compatibility alias for callers that use ``build_*`` terminology."""

    return prepare_document_pools(*args, **kwargs)


def _safe_relative(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or os.path.isabs(relative) or "\\" in relative:
        raise ValueError(f"invalid relative artifact path: {relative!r}")
    candidate = Path(relative)
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"invalid relative artifact path: {relative!r}")
    root = root.resolve()
    path = root.joinpath(*candidate.parts)
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path escapes suite root: {relative!r}") from exc
    current = root
    for part in candidate.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"artifact path traverses a symlink: {relative!r}")
    return path


def shared_pool_corpus(suite_root: Path, task: str) -> Path | None:
    """Resolve the explicit compact layout; never follow arbitrary external paths."""
    marker = suite_root / "shared_corpus.json"
    if not marker.exists():
        return None
    if marker.is_symlink():
        raise ValueError("shared corpus marker must be a regular file")
    value = _load_json(marker, label="shared corpus reference")
    expected = {"schema_version": "dci-shared-pool-corpus-v1", "task": task,
                "corpus": f"source/workspaces/{task}/corpus"}
    if value == expected and suite_root.parent.name == "pools":
        corpus = _safe_relative(suite_root.parent.parent, value["corpus"])
    elif (
        isinstance(value, Mapping)
        and value.get("schema_version") == "dci-shared-pool-corpus-v2"
        and value.get("task") == task
        and isinstance(value.get("corpus"), str)
        and value["corpus"]
    ):
        declared = Path(value["corpus"])
        if declared.is_absolute():
            corpus = declared.resolve(strict=False)
        else:
            corpus = _safe_relative(suite_root, value["corpus"])
    else:
        raise ValueError("invalid shared corpus reference/layout")
    if corpus.is_symlink() or not corpus.is_dir():
        raise ValueError("shared source corpus is missing or linked")
    return corpus


def verify_document_pool_suite(
    suite_root: Path,
    *,
    task: str | None = None,
    verify_content_hashes: bool = True,
) -> dict[str, Any]:
    """Verify completion, source closure, and copied candidate files.

    Metadata hashes are always checked for v2 suites.  With an external v2
    corpus and ``verify_content_hashes=False``, the bounded metadata-only path
    defers selected body regular-file, size, and digest checks to runtime;
    v1 suites retain their existing false-mode behavior.
    """

    if not isinstance(verify_content_hashes, bool):
        raise ValueError("verify_content_hashes must be a boolean")
    suite_root = Path(suite_root)
    if task is None:
        candidates = sorted(
            path.parent.name
            for path in (suite_root / "metadata").glob("*/candidate_pools.json")
        )
        if len(candidates) != 1:
            raise ValueError("task is required when suite does not contain exactly one pool task")
        task = candidates[0]
    _safe_task_name(task)
    metadata_dir = suite_root / "metadata" / task
    workspace_dir = suite_root / "workspaces" / task
    external_corpus = shared_pool_corpus(suite_root, task)
    def document_path(relative):
        if external_corpus is not None:
            # The virtual tier workspace is empty. Validate its exact lexical
            # shape; repeated realpath walks would add remote I/O per document.
            prefix = f"workspaces/{task}/corpus/"
            if not isinstance(relative, str) or not relative.startswith(prefix):
                raise ValueError("shared corpus inventory path is invalid")
            name = relative[len(prefix):]
            if not name or name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError("shared corpus inventory path is invalid")
            return external_corpus / name
        return _safe_relative(suite_root, str(relative))
    completion_path = metadata_dir / "_COMPLETE.json"
    completion = _load_json(completion_path, label="pool completion")
    completion_schema = completion.get("schema_version") if isinstance(completion, Mapping) else None
    if completion_schema not in SUPPORTED_POOL_SCHEMA_VERSIONS:
        raise ValueError("invalid candidate-pool completion schema")
    # Benchmark-v2 suites point at a shared source corpus outside the suite.
    # Their metadata and inventory are the pinned source release; checking
    # every body here would turn a metadata-only audit into an O(number of
    # documents) filesystem walk.  Runtime materialization verifies each
    # selected file before it reaches the evaluator.  Legacy v1 suites keep
    # their existing false-mode checks for compatibility.
    metadata_only_external = (
        external_corpus is not None
        and completion_schema == V2_POOL_SCHEMA_VERSION
        and not verify_content_hashes
    )
    if external_corpus is not None and completion.get("shared_corpus_sha256") != sha256_file(suite_root / "shared_corpus.json"):
        raise ValueError("shared corpus reference hash mismatch")
    completion_policy = completion.get("verification_policy")
    completion_verified = completion.get("content_hashes_verified")
    if completion_policy is not None or completion_verified is not None:
        if completion_policy not in _CONTENT_HASH_VERIFICATION_POLICIES.values():
            raise ValueError("invalid candidate-pool verification policy")
        if not isinstance(completion_verified, bool):
            raise ValueError("candidate-pool content_hashes_verified must be a boolean")
        if completion_verified != (
            completion_policy == _CONTENT_HASH_VERIFICATION_POLICIES[True]
        ):
            raise ValueError("candidate-pool verification policy is inconsistent")
    metadata_hashes = completion.get("metadata_hashes")
    if not isinstance(metadata_hashes, Mapping) or set(metadata_hashes) != set(_METADATA_FILES):
        raise ValueError("candidate-pool completion has incomplete metadata hashes")
    expected_metadata_names = set(_METADATA_FILES) | {"_COMPLETE.json"}
    if {path.name for path in metadata_dir.iterdir()} != expected_metadata_names:
        raise ValueError("unexpected files in candidate-pool metadata directory")
    verify_metadata_hashes = verify_content_hashes or completion_schema == V2_POOL_SCHEMA_VERSION
    for name, expected in metadata_hashes.items():
        path = metadata_dir / str(name)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"candidate-pool metadata hash mismatch: {name}")
        if verify_metadata_hashes and sha256_file(path) != expected:
            raise ValueError(f"candidate-pool metadata hash mismatch: {name}")
    inventory = _load_json(metadata_dir / "file_inventory.json", label="file inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("file inventory must be an object")
    workspace_hashes = completion.get("workspace_hashes")
    if not isinstance(workspace_hashes, Mapping) or set(workspace_hashes) != set(inventory):
        raise ValueError("candidate-pool completion has incomplete workspace hashes")
    if metadata_only_external:
        # The inventory digest is already sealed in _COMPLETE and its rows
        # carry the source byte hashes.  Keep those declarations for the
        # structural checks below without lstat/stat/hash calls on the source
        # corpus.  Runtime validates regular files, sizes, and copied bytes.
        verified_workspace_hashes = {
            str(relative): str(expected) for relative, expected in workspace_hashes.items()
        }
    else:
        verified_workspace_hashes = {}
        for relative, expected in workspace_hashes.items():
            path = document_path(relative)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"candidate-pool workspace file is missing or non-regular: {relative}")
            if path.stat().st_nlink != 1:
                raise ValueError(f"candidate-pool workspace file has unexpected hard links: {relative}")
            actual_hash = sha256_file(path) if verify_content_hashes else str(expected)
            verified_workspace_hashes[str(relative)] = actual_hash
            if verify_content_hashes and actual_hash != expected:
                raise ValueError(f"candidate-pool workspace hash mismatch: {relative}")
    if (verify_content_hashes or completion_schema == V2_POOL_SCHEMA_VERSION) and completion.get("file_inventory_sha256") != sha256_file(
        metadata_dir / "file_inventory.json"
    ):
        raise ValueError("candidate-pool file inventory hash mismatch")
    shared_corpus = external_corpus or workspace_dir / "corpus"
    if external_corpus is None:
        actual_workspace_files = {
            _relative_path(suite_root, path)
            for path in workspace_dir.rglob("*")
            if path.is_file()
        }
        for path in workspace_dir.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"candidate-pool workspace contains a symlink: {path}")
            if path.is_file() and path.stat().st_nlink != 1:
                raise ValueError(f"candidate-pool workspace file has unexpected hard links: {path}")
        if actual_workspace_files != set(inventory):
            raise ValueError("candidate-pool workspace file inventory differs from files on disk")
        if {path.name for path in workspace_dir.iterdir()} != {"corpus"}:
            raise ValueError("unexpected files in candidate-pool workspace root")
        shared_corpus = workspace_dir / "corpus"
        if any(
            path.is_dir() or path.parent != shared_corpus
            for path in shared_corpus.rglob("*")
        ):
            raise ValueError("candidate-pool shared corpus must contain only regular files")
    elif any(workspace_dir.iterdir()):
        raise ValueError("compact pool workspace must remain empty")

    source_identity = _load_json(metadata_dir / "source_identity.json", label="source identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("candidate-pool source identity must be an object")
    _validate_saved_identity(source_identity, source_identity)
    manifest = _load_json(metadata_dir / "manifest.json", label="pool manifest")
    if not isinstance(manifest, Mapping) or manifest.get("task") != task:
        raise ValueError("candidate-pool manifest task is invalid")
    manifest_policy = manifest.get("verification_policy")
    manifest_verified = manifest.get("content_hashes_verified")
    if manifest_policy is not None or manifest_verified is not None:
        if manifest_policy not in _CONTENT_HASH_VERIFICATION_POLICIES.values():
            raise ValueError("invalid candidate-pool manifest verification policy")
        if not isinstance(manifest_verified, bool):
            raise ValueError("candidate-pool manifest content_hashes_verified must be a boolean")
        if manifest_verified != (
            manifest_policy == _CONTENT_HASH_VERIFICATION_POLICIES[True]
        ):
            raise ValueError("candidate-pool manifest verification policy is inconsistent")
    if (
        completion_policy is not None
        and manifest_policy is not None
        and (completion_policy != manifest_policy or completion_verified != manifest_verified)
    ):
        raise ValueError("candidate-pool completion and manifest policies differ")
    mapping = _load_source_mapping(metadata_dir / "source_doc_ids.json")
    doc_id_files = _load_doc_id_files(metadata_dir / "doc_id_files.json", mapping)
    doc_id_file_aliases = set(doc_id_files)
    qrels = _load_source_qrels(metadata_dir / "original_qrels.json")
    raw_qrels = _load_raw_qrels_payload(metadata_dir / "original_qrels.json")
    canonical_qrels = _load_source_qrels(metadata_dir / "qrels.json")
    raw_canonical_qrels = _load_raw_qrels_payload(metadata_dir / "qrels.json")
    original_to_alias = {original: alias for alias, original in mapping.items()}
    original_ids = set(original_to_alias)
    query_rows = []
    with (metadata_dir / "queries.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError("prepared query inventory contains a non-object row")
                query_rows.append(dict(row))
    query_ids = [row.get("_id", row.get("query_id")) for row in query_rows]
    if len(query_ids) != len(set(query_ids)) or any(not _is_string_id(query_id) for query_id in query_ids):
        raise ValueError("prepared query inventory has invalid or duplicate IDs")
    if set(query_ids) != set(canonical_qrels):
        raise ValueError("canonical qrels query IDs differ from active queries")
    expected_canonical = {
        query_id: {
            original_to_alias[doc_id]: score for doc_id, score in sorted(docs.items())
        }
        for query_id, docs in sorted(qrels.items())
        if query_id in canonical_qrels
    }
    if canonical_qrels != expected_canonical:
        raise ValueError("canonical qrels do not preserve original labels")
    if manifest.get("qrels_hash") != stable_json_sha256(raw_canonical_qrels):
        raise ValueError("candidate-pool qrels hash does not match manifest")
    if manifest.get("queries_hash") != stable_json_sha256(sorted(query_rows, key=lambda row: row.get("_id", row.get("query_id")))):
        raise ValueError("candidate-pool query hash does not match manifest")
    if manifest.get("source_doc_ids_hash") != stable_json_sha256(mapping):
        raise ValueError("candidate-pool source mapping hash does not match manifest")
    if manifest.get("original_qrels_hash") != stable_json_sha256(raw_qrels):
        raise ValueError("candidate-pool original qrels hash does not match manifest")
    for required in ("dataset", "revision", "resolved_revision", "task_instruction"):
        if not isinstance(manifest.get(required), str) or (
            required != "task_instruction" and not manifest[required]
        ):
            raise ValueError(f"candidate-pool manifest missing source field: {required}")
    if manifest.get("document_id_policy") != DOCUMENT_ID_POLICY:
        raise ValueError("candidate-pool manifest document_id_policy is invalid")
    candidate_payload = _load_json(metadata_dir / "candidate_pools.json", label="candidate pools")
    if not isinstance(candidate_payload, Mapping):
        raise ValueError("candidate pools must be an object")
    if candidate_payload.get("schema_version") not in SUPPORTED_POOL_SCHEMA_VERSIONS or candidate_payload.get("task") != task:
        raise ValueError("candidate pool metadata schema/task mismatch")
    if candidate_payload.get("source_identity") != source_identity:
        raise ValueError("candidate pool/source identity mismatch")
    source_release = candidate_payload.get("source_release")
    if not isinstance(source_release, Mapping):
        raise ValueError("candidate pool source release is missing")
    if (
        source_release.get("source_manifest_sha256") != manifest.get("source_manifest_sha256")
        or source_release.get("source_inventory_sha256")
        != manifest.get("source_inventory_sha256")
    ):
        raise ValueError("candidate pool source release differs from manifest")
    capacity = candidate_payload.get("capacity")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("candidate pool capacity is invalid")
    if candidate_payload.get("pool_size") != manifest.get("pool_size"):
        raise ValueError("candidate pool tier differs from manifest")
    if candidate_payload.get("seed") != manifest.get("seed") or capacity != manifest.get("capacity"):
        raise ValueError("candidate pool selection settings differ from manifest")
    pools = candidate_payload.get("pools")
    if not isinstance(pools, Mapping) or set(pools) != set(canonical_qrels):
        raise ValueError("candidate pool query IDs differ from active qrels")
    if candidate_payload.get("shared_corpus") != "corpus":
        raise ValueError("candidate pool shared corpus path is invalid")
    if manifest.get("shared_corpus_dir") != f"workspaces/{task}/corpus":
        raise ValueError("candidate pool shared corpus manifest path is invalid")
    inventory_aliases: set[str] = set()
    for relative, row in inventory.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"file inventory row is invalid: {relative}")
        path = document_path(relative)
        if path.parent != shared_corpus:
            raise ValueError(f"file inventory path is outside shared corpus: {relative}")
        alias = row.get("alias")
        original_id = row.get("original_id")
        if (
            not _is_string_id(alias)
            or alias in inventory_aliases
            or alias not in doc_id_files
            or original_id != mapping[alias]
            or path.name != doc_id_files[alias]
            or not isinstance(row.get("bytes"), int)
            or row.get("bytes") < 0
            or not isinstance(row.get("sha256"), str)
            or not row.get("sha256")
        ):
            raise ValueError(f"file inventory row does not match shared corpus: {relative}")
        if not metadata_only_external and path.stat().st_size != row["bytes"]:
            raise ValueError(f"file inventory size mismatch: {relative}")
        # No-hash mode skips reading file bytes for a fresh digest, but the
        # completion declaration and file inventory must still agree.
        if verified_workspace_hashes.get(str(relative)) != row["sha256"]:
            raise ValueError(f"file inventory hash mismatch: {relative}")
        inventory_aliases.add(alias)
    del verified_workspace_hashes
    del completion, workspace_hashes
    if external_corpus is None:
        shared_files = {
            path.name for path in shared_corpus.iterdir() if path.is_file() and not path.is_symlink()
        }
        if shared_files != {doc_id_files[alias] for alias in inventory_aliases}:
            raise ValueError("shared corpus files differ from file inventory")
    expected_pool_size = min(capacity, len(doc_id_files))
    selected_aliases_union: set[str] = set()
    for query_id, record in pools.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"candidate pool record is not an object: {query_id}")
        expected_record_keys = {
            "query_id",
            "candidate_doc_ids",
            "candidate_original_doc_ids",
            "candidate_scores",
            "ordering",
            "num_docs",
            "audit",
        }
        if set(record) != expected_record_keys or record.get("query_id") != query_id:
            raise ValueError(f"candidate pool record schema is invalid: {query_id}")
        aliases = record.get("candidate_doc_ids")
        originals = record.get("candidate_original_doc_ids")
        ordering = record.get("ordering")
        candidate_scores = record.get("candidate_scores")
        if not isinstance(aliases, list) or not isinstance(originals, list):
            raise ValueError(f"candidate pool IDs are not canonical: {query_id}")
        alias_set = set(aliases) if all(_is_string_id(alias) for alias in aliases) else set()
        original_set = (
            set(originals) if all(_is_string_id(original) for original in originals) else set()
        )
        if (
            any(not _is_string_id(alias) for alias in aliases)
            or any(not _is_string_id(original) for original in originals)
            or len(aliases) != len(alias_set)
            or len(originals) != len(original_set)
            or ordering not in {"scored", "unscored"}
        ):
            raise ValueError(f"candidate pool IDs are not canonical: {query_id}")
        if len(aliases) != len(originals) or len(aliases) != expected_pool_size:
            raise ValueError(f"candidate pool size invalid: {query_id}")
        if not alias_set.issubset(doc_id_file_aliases):
            raise ValueError(f"candidate pool contains unknown aliases: {query_id}")
        if not original_set.issubset(original_ids):
            raise ValueError(f"candidate pool contains unknown original IDs: {query_id}")
        expected_aliases = [original_to_alias[original] for original in originals]
        if aliases != expected_aliases:
            raise ValueError(f"candidate pool alias/original IDs differ: {query_id}")
        if ordering == "unscored":
            if candidate_scores is not None or originals != sorted(originals):
                raise ValueError(f"unscored candidate pool order is invalid: {query_id}")
        else:
            if (
                not isinstance(candidate_scores, list)
                or len(candidate_scores) != len(aliases)
                or any(
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                    for score in candidate_scores
                )
                or any(
                    float(previous) < float(current)
                    for previous, current in zip(candidate_scores, candidate_scores[1:])
                )
            ):
                raise ValueError(f"scored candidate pool scores are invalid: {query_id}")
        if record.get("num_docs") != expected_pool_size:
            raise ValueError(f"candidate pool document count differs from capacity: {query_id}")
        selected_aliases_union.update(aliases)
        audit = record.get("audit")
        if not isinstance(audit, Mapping) or audit.get("query_id") != query_id:
            raise ValueError(f"candidate pool audit is missing or mismatched: {query_id}")
        if (
            audit.get("selected_ids") != originals
            or audit.get("selected_scores") != candidate_scores
            or audit.get("ordering") != ordering
            or audit.get("capacity") != capacity
            or audit.get("source_size") != len(doc_id_files)
        ):
            raise ValueError(f"candidate pool audit does not match canonical IDs: {query_id}")
        positives = {
            original_to_alias[doc_id]
            for doc_id, score in qrels[query_id].items()
            if score > 0
        }
        if not positives.issubset(alias_set):
            raise ValueError(f"candidate pool dropped a positive label: {query_id}")
        if not alias_set.issubset(inventory_aliases):
            raise ValueError(f"candidate pool references a missing shared document: {query_id}")
    if selected_aliases_union != inventory_aliases:
        raise ValueError("shared corpus is not exactly the candidate-pool union")
    # The shared corpus is a strict subset of the active source corpus when
    # query pools do not collectively select every source document. Excluded
    # source IDs remain in host mapping/qrels but never enter the workspace.
    # Only expected roots are allowed in a prepared suite.  This catches a
    # stray README or label file accidentally copied into an agent workspace.
    allowed_top = {"metadata", "workspaces"}
    if external_corpus is not None:
        allowed_top.add("shared_corpus.json")
    if {path.name for path in suite_root.iterdir()} != allowed_top:
        raise ValueError("unexpected files at candidate-pool suite root")
    return {
        "schema_version": completion_schema,
        "task": task,
        "capacity": capacity,
        "num_queries": len(pools),
        "num_workspace_files": len(inventory),
        "verification_policy": _CONTENT_HASH_VERIFICATION_POLICIES[verify_content_hashes],
        "content_hashes_verified": verify_content_hashes,
        "metadata_hashes_verified": verify_metadata_hashes,
        "content_verification_boundary": (
            "metadata-only; external shared corpus bodies are verified at runtime"
            if metadata_only_external
            else "full corpus content hashes verified"
            if verify_content_hashes
            else "declared workspace digests"
        ),
    }


def load_document_pool_manifest(
    suite_root: Path,
    *,
    task: str | None = None,
    verify_content_hashes: bool = True,
) -> dict[str, Any]:
    """Verify and load the host-side candidate-pool manifest.

    Runtime callers should use this entrypoint before copying the selected
    IDs from the task-level shared corpus into a temporary query workspace.
    Verification happens before JSON is returned, so a stale or tampered pool
    cannot silently become an evaluation input.
    """

    if not isinstance(verify_content_hashes, bool):
        raise ValueError("verify_content_hashes must be a boolean")
    suite_root = Path(suite_root)
    if task is None:
        candidates = sorted(
            path.parent.name
            for path in (suite_root / "metadata").glob("*/candidate_pools.json")
        )
        if len(candidates) != 1:
            raise ValueError("task is required when suite does not contain exactly one pool task")
        task = candidates[0]
    verify_document_pool_suite(
        suite_root,
        task=task,
        verify_content_hashes=verify_content_hashes,
    )
    payload = _load_json(
        suite_root / "metadata" / task / "candidate_pools.json",
        label="candidate pools",
    )
    if not isinstance(payload, dict):
        raise ValueError("candidate pools must be a JSON object")
    return payload


__all__ = [
    "DocumentPoolBuildResult",
    "DOCUMENT_ID_POLICY",
    "EXPECTED_EMBEDDING_MODEL_PATH",
    "POOL_SCHEMA_VERSION",
    "V2_POOL_SCHEMA_VERSION",
    "POOL_SIZES",
    "RANKINGS_SCHEMA_VERSION",
    "build_document_pools",
    "build_source_identity",
    "load_saved_rankings",
    "load_document_pool_manifest",
    "prepare_document_pools",
    "safe_query_dirname",
    "select_candidate_ids",
    "sha256_file",
    "stable_json_sha256",
    "verify_document_pool_suite",
]
