#!/usr/bin/env python3
"""Prepare the pinned native Chinese STARD retrieval source on CPU.

The raw snapshot is materialised by the existing connector boundary.  Parsing
is deliberately source-local: STARD stores the corpus as JSONL and the
question/qrels records as one JSON array, so the generic JSONL parser cannot
recover the original ``match_id`` links without changing their meaning.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dci_bench.data.benchmark_v2 import (  # noqa: E402
    _prepare_query_eligibility,
    sample_query_splits,
)
from dci_bench.data.expansion import verify_document_task  # noqa: E402
from dci_bench.data.pipeline.connectors import (  # noqa: E402
    _spec_fingerprint,
    fetch_source,
)
from dci_bench.data.pipeline.core import (  # noqa: E402
    export_dci,
    file_digest,
    load_bundle,
    publish_bundle,
    read_json,
    verify_release,
)


REVISION = "bae10e1f358685446c3af4b15e7b94724f83bc9e"
SOURCE_ID = "gapfill_stard_legal_zh"
DATASET = "oneal2000/STARD"
REPOSITORY_URL = "https://github.com/oneal2000/STARD"
RAW_BASE_URL = f"https://raw.githubusercontent.com/oneal2000/STARD/{REVISION}"
CORPUS_URL = f"{RAW_BASE_URL}/data/corpus.jsonl"
QUERIES_URL = f"{RAW_BASE_URL}/data/queries.json"
LICENSE_URL = f"{RAW_BASE_URL}/LICENSE"
DATA_STATEMENT_URL = f"{RAW_BASE_URL}/Data%20Statement.md"

FULL_CORPUS_COUNT = 55_348
FULL_QUERY_COUNT = 1_543
FULL_POSITIVE_PAIRS = 2_717
MAX_POSITIVES = 10
OVER_LIMIT_QUERY_ID = "651"
CORPUS_BYTES = 24_120_549
CORPUS_SHA256 = "15c77a1333d947ec88cf4cb4c99d70dafd946ec91d35f2bbc77824817c7f7fa4"
QUERIES_BYTES = 1_818_895
QUERIES_SHA256 = "e64dcdbda5d2b734a7242253620dcf5e8325a3d726f0ca0a86e9b3664b8ba644"

DEFAULT_ROOT = Path(
    "/mnt/afs2/202611/embedding_models/dci-bench/eval_data/"
    "benchmark_legal_pos10_20261116_v1"
)
RAW_RELATIVE_ROOT = Path("raw/stard")
RELEASE_RELATIVE_ROOT = Path("releases") / SOURCE_ID
EXPORT_RELATIVE_ROOT = Path("run/source")


def _resource(url: str, name: str, size: int, role: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"url": url, "name": name, "bytes": size}
    if role is not None:
        result["role"] = role
    return result


SPEC: dict[str, Any] = {
    "id": SOURCE_ID,
    "connector": "generic_jsonl",
    "input_kind": "retrieval",
    "domain": "legal",
    "language": "zh",
    "revision": REVISION,
    "license": "MIT (STARD repository; underlying statute texts retain their source status)",
    "options": {
        "dataset": DATASET,
        "files": [
            _resource(CORPUS_URL, "data/corpus.jsonl", CORPUS_BYTES, "documents"),
            _resource(QUERIES_URL, "data/queries.json", QUERIES_BYTES, "queries"),
        ],
        "max_bytes": 30_000_000,
        "timeout": 30,
        "retries": 2,
        "headers": {"User-Agent": "dci-bench/STARD-preparation"},
    },
    "metadata": {
        "official_repository": REPOSITORY_URL,
        "raw_files_are_commit_pinned": True,
        "native_language": "zh",
    },
}

POLICY = {
    "max_positive_per_query": MAX_POSITIVES,
    "exclude_review_required": True,
    "drop_near_duplicates": False,
}


class StardFormatError(ValueError):
    """The pinned STARD payload does not have the expected source schema."""


def _strict_json(value: str, *, where: str) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise StardFormatError(f"duplicate JSON key {key!r} in {where}")
            output[key] = item
        return output

    return json.loads(
        value,
        object_pairs_hook=unique_pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            StardFormatError(f"non-finite JSON constant {token!r} in {where}")
        ),
    )


def _read_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = _strict_json(line, where=f"{path}:{line_number}")
            if not isinstance(row, Mapping):
                raise StardFormatError(f"corpus row {path}:{line_number} is not an object")
            yield line_number, row


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StardFormatError(f"{where} must be a non-empty string")
    return value


def _source_id(value: Any, *, where: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise StardFormatError(f"{where} must be a string or integer ID")
    result = str(value)
    if not result:
        raise StardFormatError(f"{where} must not be empty")
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_corpus(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    documents: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    body_ids: dict[str, list[str]] = {}
    for line_number, row in _read_jsonl(path):
        document_id = _source_id(row.get("id"), where=f"{path}:{line_number}.id")
        if document_id in seen_ids:
            raise StardFormatError(f"duplicate corpus ID {document_id!r}")
        seen_ids.add(document_id)
        title = _text(row.get("name"), where=f"{path}:{line_number}.name")
        content = _text(row.get("content"), where=f"{path}:{line_number}.content")
        # The legal name and article number are part of the retrievable source
        # text.  Two distinct statute articles may have identical article
        # content; dropping the name would make the quality pass treat them as
        # equivalent documents and could erase a native positive.
        text = f"{title}\n\n{content}"
        digest = _sha256_text(text)
        body_ids.setdefault(digest, []).append(document_id)
        documents.append(
            {
                "_id": document_id,
                "title": title,
                "text": text,
                "language": "zh",
                "metadata": {
                    "original_id": document_id,
                    "original_name": title,
                    "original_content_sha256": _sha256_text(content),
                    "source_file": "data/corpus.jsonl",
                    "translation_status": "native",
                    "annotation_source": "source",
                    "domain": "legal",
                },
            }
        )
    if not documents:
        raise StardFormatError(f"corpus is empty: {path}")
    return documents, {digest: ids for digest, ids in body_ids.items() if len(ids) > 1}


def _load_queries(
    path: Path,
    document_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], list[dict[str, Any]], int, int]:
    raw = read_json(path)
    if not isinstance(raw, list):
        raise StardFormatError(f"queries file is not a JSON array: {path}")
    queries: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, int]] = {}
    exclusions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    raw_positive_pairs = 0
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise StardFormatError(f"query row {index} is not an object")
        query_id = _source_id(row.get("query_id"), where=f"{path}[{index}].query_id")
        if query_id in seen_ids:
            raise StardFormatError(f"duplicate query ID {query_id!r}")
        seen_ids.add(query_id)
        question = _text(row.get("问题"), where=f"{path}[{index}].问题")
        match_ids = row.get("match_id")
        match_names = row.get("match_name")
        related = row.get("相关法规")
        if not isinstance(match_ids, list) or not match_ids:
            raise StardFormatError(f"{path}[{index}].match_id must be a non-empty list")
        if not isinstance(match_names, list) or len(match_names) != len(match_ids):
            raise StardFormatError(f"{path}[{index}].match_name must align with match_id")
        if not isinstance(related, Mapping):
            raise StardFormatError(f"{path}[{index}].相关法规 must be an object")
        document_row: list[str] = []
        for match_index, value in enumerate(match_ids):
            document_id = _source_id(value, where=f"{path}[{index}].match_id[{match_index}]")
            if document_id not in document_ids:
                raise StardFormatError(
                    f"query {query_id!r} references unknown corpus ID {document_id!r}"
                )
            if document_id in document_row:
                raise StardFormatError(f"query {query_id!r} repeats positive ID {document_id!r}")
            document_row.append(document_id)
        raw_positive_pairs += len(document_row)
        if len(document_row) > MAX_POSITIVES:
            exclusions.append(
                {
                    "query_id": query_id,
                    "reason": "positive_count_over_limit",
                    "positive_count": len(document_row),
                    "limit": MAX_POSITIVES,
                    "scope": "whole_query",
                    "new_gold_created": False,
                }
            )
            continue
        metadata = {
            "group_id": f"stard-query:{query_id}",
            "raw_query_id": query_id,
            "source_file": "data/queries.json",
            "original_match_ids": document_row,
            "original_match_names": [str(value) for value in match_names],
            "related_law_names": [str(value) for value in related],
            "translation_status": "native",
            "annotation_source": "source",
            "domain": "legal",
        }
        queries.append(
            {
                "_id": query_id,
                "text": question,
                "language": "zh",
                "group_id": f"stard-query:{query_id}",
                "metadata": metadata,
            }
        )
        qrels[query_id] = {document_id: 1 for document_id in document_row}
    if not queries:
        raise StardFormatError("all STARD queries were excluded")
    if len(exclusions) != 1 or exclusions[0]["query_id"] != OVER_LIMIT_QUERY_ID or exclusions[0]["positive_count"] != 23:
        raise StardFormatError(
            "pinned STARD over-limit distribution changed; expected only query 651 with 23 positives"
        )
    return queries, qrels, exclusions, len(raw), raw_positive_pairs


def build_bundle(corpus_path: Path, queries_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse native STARD files without changing original IDs or positives."""
    documents, duplicate_groups = _load_corpus(Path(corpus_path))
    document_ids = {row["_id"] for row in documents}
    queries, qrels, exclusions, raw_query_count, raw_positive_pairs = _load_queries(
        Path(queries_path), document_ids
    )
    bundle = {
        "schema_version": "dci-data-bundle-v1",
        "input_kind": "retrieval",
        "source": {"dataset": DATASET, "revision": REVISION, "connector": "generic_jsonl"},
        "documents": documents,
        "queries": queries,
        "qrels": qrels,
        "evidence": [],
        "judgments": [],
        "provenance": {
            "source_id": SOURCE_ID,
            "source_family": "STARD",
            "domain": "legal",
            "language": "zh",
            "translation_status": "native",
            "annotation_source": "human",
            "license": "MIT for the STARD repository; underlying statute texts retain their source status",
            "official_repository": REPOSITORY_URL,
            "revision": REVISION,
            "task_instruction": "给定中文法律咨询问题，检索回答该问题所需的中国法律或司法解释条文。",
            "qrels_semantics": "Original STARD match_id links from real Chinese legal consultation questions to statute articles; every retained link is binary score 1.",
            "annotation_provenance": "Native Chinese queries from the 12348 legal consultation service, with article matching manually annotated by legal annotators; no generated rewrites or positives.",
            "query_filter_policy": "Exclude a whole original query when its original positive count exceeds 10; preserve every positive for every retained query.",
            "query_exclusions": exclusions,
            "corpus_body_policy": "Check exact body duplicates and let the existing quality policy exclude a whole query when a positive has an unlabelled exact-text equivalent; do not synthesize positives.",
            "body_duplicate_groups": len(duplicate_groups),
            "body_duplicate_occurrences_beyond_first": sum(len(ids) - 1 for ids in duplicate_groups.values()),
            "raw_counts": {
                "documents": len(documents),
                "queries": raw_query_count,
                "positive_pairs": raw_positive_pairs,
            },
            "pinned_urls": {
                "corpus": CORPUS_URL,
                "queries": QUERIES_URL,
                "license": LICENSE_URL,
                "data_statement": DATA_STATEMENT_URL,
            },
        },
    }
    if len(documents) != FULL_CORPUS_COUNT:
        raise StardFormatError(f"expected {FULL_CORPUS_COUNT} corpus rows, got {len(documents)}")
    if raw_query_count != FULL_QUERY_COUNT:
        raise StardFormatError(f"expected {FULL_QUERY_COUNT} raw queries, got {raw_query_count}")
    if raw_positive_pairs != FULL_POSITIVE_PAIRS:
        raise StardFormatError(f"expected {FULL_POSITIVE_PAIRS} raw positive pairs, got {raw_positive_pairs}")
    stats = {
        "raw_documents": len(documents),
        "raw_queries": raw_query_count,
        "raw_positive_pairs": raw_positive_pairs,
        "excluded_over_limit_queries": len(exclusions),
        "retained_queries_before_quality": len(queries),
        "retained_positive_pairs_before_quality": sum(len(row) for row in qrels.values()),
        "exact_duplicate_body_groups": len(duplicate_groups),
        "exact_duplicate_body_occurrences_beyond_first": sum(len(ids) - 1 for ids in duplicate_groups.values()),
    }
    return bundle, stats


def _verify_pinned_file(path: Path, *, size: int, sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ValueError(f"pinned file size mismatch for {path}: expected {size}, got {actual_size}")
    actual_sha256 = file_digest(path)
    if actual_sha256 != sha256:
        raise ValueError(f"pinned file sha256 mismatch for {path}: expected {sha256}, got {actual_sha256}")


def _raw_inputs(snapshot: Path) -> tuple[Path, Path]:
    direct = (snapshot / "corpus.jsonl", snapshot / "queries.json")
    nested = (snapshot / "data/corpus.jsonl", snapshot / "data/queries.json")
    if all(path.is_file() for path in direct):
        corpus, queries = direct
    elif all(path.is_file() for path in nested):
        corpus, queries = nested
    else:
        raise FileNotFoundError(f"STARD raw snapshot must contain corpus.jsonl and queries.json: {snapshot}")
    _verify_pinned_file(corpus, size=CORPUS_BYTES, sha256=CORPUS_SHA256)
    _verify_pinned_file(queries, size=QUERIES_BYTES, sha256=QUERIES_SHA256)
    return corpus, queries


def _direct_manifest(raw_root: Path) -> dict[str, Any]:
    """Write or verify the connector manifest for the two-file raw import."""
    corpus, queries = _raw_inputs(raw_root)
    expected_fingerprint = _spec_fingerprint(SPEC)
    manifest_path = raw_root / "source_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (
            manifest.get("requested_spec_fingerprint") != expected_fingerprint
            or manifest.get("spec_fingerprint") != expected_fingerprint
            or manifest.get("id") != SOURCE_ID
            or manifest.get("connector") != "generic_jsonl"
            or manifest.get("input_kind") != "retrieval"
            or manifest.get("resolved_revision") != REVISION
            or manifest.get("actual_urls") != [CORPUS_URL, QUERIES_URL]
        ):
            raise ValueError("existing STARD raw manifest has another source specification")
        if manifest.get("complete") is not True:
            raise ValueError("existing STARD raw manifest is incomplete")
        expected_files = {
            "corpus.jsonl": {"sha256": CORPUS_SHA256, "bytes": CORPUS_BYTES},
            "queries.json": {"sha256": QUERIES_SHA256, "bytes": QUERIES_BYTES},
        }
        recorded_files = {
            item.get("path"): {"sha256": item.get("sha256"), "bytes": item.get("bytes")}
            for item in manifest.get("files", [])
            if isinstance(item, Mapping)
        }
        if recorded_files != expected_files:
            raise ValueError("existing STARD raw manifest file inventory differs")
        return manifest
    manifest = {
        "schema_version": "dci-source-snapshot-v1",
        "complete": True,
        "id": SOURCE_ID,
        "connector": "generic_jsonl",
        "input_kind": "retrieval",
        "requested_spec_fingerprint": expected_fingerprint,
        "spec_fingerprint": expected_fingerprint,
        "requested_spec": SPEC,
        "actual_url": None,
        "actual_urls": [CORPUS_URL, QUERIES_URL],
        "resolved_revision": REVISION,
        "layout": {
            "documents": ["corpus.jsonl"],
            "queries": ["queries.json"],
        },
        "files": [
            {"path": "corpus.jsonl", "sha256": CORPUS_SHA256, "bytes": CORPUS_BYTES},
            {"path": "queries.json", "sha256": QUERIES_SHA256, "bytes": QUERIES_BYTES},
        ],
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return manifest


def _get_snapshot(data_root: Path, requested: Path | None, *, create_manifest: bool = True) -> Path:
    if requested is not None:
        snapshot = Path(requested)
        _raw_inputs(snapshot)
        if not (snapshot / "source_manifest.json").is_file():
            raise FileNotFoundError(f"raw snapshot has no source_manifest.json: {snapshot}")
        _direct_manifest(snapshot)
        return snapshot
    raw_root = data_root / RAW_RELATIVE_ROOT
    direct_present = any((raw_root / name).exists() for name in ("corpus.jsonl", "queries.json"))
    if direct_present:
        if not create_manifest and not (raw_root / "source_manifest.json").is_file():
            raise FileNotFoundError(f"raw snapshot has no source_manifest.json: {raw_root}")
        _direct_manifest(raw_root)
        return raw_root
    if not create_manifest:
        raise FileNotFoundError(f"missing STARD raw snapshot: {raw_root}")
    return fetch_source(SPEC, raw_root)


def _eligible(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qrels = bundle.get("qrels", {})
    rows = []
    for query in bundle.get("queries", []):
        query_id = query["_id"]
        positive_count = len(qrels[query_id])
        if not 1 <= positive_count <= MAX_POSITIVES:
            raise ValueError(f"{query_id}: positive count outside 1..{MAX_POSITIVES}")
        metadata = query.get("metadata") if isinstance(query.get("metadata"), Mapping) else {}
        rows.append({
            "query_id": query_id,
            "text": query["text"],
            "group_id": metadata.get("group_id", f"stard-query:{query_id}"),
        })
    eligible, exclusions, _ = _prepare_query_eligibility(rows, qrels)
    return eligible, exclusions


def _export_or_verify(release: Path, suite: Path, manifest: Mapping[str, Any], *, io_workers: int) -> dict[str, Any]:
    metadata = suite / "metadata" / SOURCE_ID
    if metadata.exists():
        source_manifest = metadata / "source_manifest.json"
        if not source_manifest.is_file():
            raise FileExistsError(f"existing export is incomplete: {metadata}")
        recorded = read_json(source_manifest)
        if recorded.get("source_fingerprint") != manifest["bundle_hash"]:
            raise ValueError("existing export has another source identity")
        return verify_document_task(suite, SOURCE_ID, io_workers=io_workers)
    for candidate in (
        suite / "workspaces" / SOURCE_ID,
        suite / "canonical" / SOURCE_ID,
    ):
        if candidate.exists():
            raise FileExistsError(f"existing export is incomplete: {candidate}")
    export_dci(release, suite, SOURCE_ID, io_workers=io_workers)
    return verify_document_task(suite, SOURCE_ID, io_workers=io_workers)


def _verify_export_only(suite: Path, manifest: Mapping[str, Any], *, io_workers: int) -> dict[str, Any]:
    metadata = suite / "metadata" / SOURCE_ID
    source_manifest = metadata / "source_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"missing STARD export: {metadata}")
    recorded = read_json(source_manifest)
    if recorded.get("source_fingerprint") != manifest["bundle_hash"]:
        raise ValueError("existing export has another source identity")
    return verify_document_task(suite, SOURCE_ID, io_workers=io_workers)


def _check_release(
    data_root: Path,
    *,
    raw_snapshot: Path | None,
    io_workers: int,
    check_export_only: bool,
) -> dict[str, Any]:
    data_root = Path(data_root)
    release = data_root / RELEASE_RELATIVE_ROOT
    suite = data_root / EXPORT_RELATIVE_ROOT
    if not release.is_dir():
        raise FileNotFoundError(f"missing STARD release: {release}")
    manifest = verify_release(release)
    bundle = load_bundle(release)
    if manifest["source"] != {"dataset": DATASET, "revision": REVISION, "connector": "generic_jsonl"}:
        raise ValueError("existing release has another STARD source identity")
    if manifest["state"] != "labels_ready" or manifest["automatic_labels"]:
        raise ValueError(f"STARD release is not human labels_ready: {manifest['state']}")
    if manifest["num_documents"] < 1_000:
        raise ValueError("STARD release has fewer than 1000 documents")
    eligible, eligibility_exclusions = _eligible(bundle)
    if len(eligible) < 60:
        raise ValueError(f"STARD release has fewer than 60 eligible queries: {len(eligible)}")
    sampling = sample_query_splits(
        eligible,
        dev_size=10,
        test_size=50,
        smoke_size=1,
        seed=42,
        task_id=SOURCE_ID,
        group_key="group_id",
    )
    export = (
        _verify_export_only(suite, manifest, io_workers=io_workers)
        if check_export_only
        else _export_or_verify(release, suite, manifest, io_workers=io_workers)
    )
    source_manifest = bundle.get("provenance", {})
    raw_counts = source_manifest.get("raw_counts", {}) if isinstance(source_manifest, Mapping) else {}
    query_exclusions = source_manifest.get("query_exclusions", []) if isinstance(source_manifest, Mapping) else []
    duplicate_groups = source_manifest.get("body_duplicate_groups", 0) if isinstance(source_manifest, Mapping) else 0
    duplicate_occurrences = source_manifest.get("body_duplicate_occurrences_beyond_first", 0) if isinstance(source_manifest, Mapping) else 0
    excluded_positive_pairs = sum(
        int(item.get("positive_count", 0))
        for item in query_exclusions
        if isinstance(item, Mapping)
    )
    result = {
        "source_id": SOURCE_ID,
        "dataset": DATASET,
        "revision": REVISION,
        "state": manifest["state"],
        "release": str(release.absolute()),
        "export": str(suite.absolute()),
        "release_completion_sha256": file_digest(release / "_COMPLETE.json"),
        "export_completion_sha256": file_digest(suite / "metadata" / SOURCE_ID / "_COMPLETE.json"),
        "documents": manifest["num_documents"],
        "queries": manifest["num_queries"],
        "positive_pairs": manifest["num_positive_pairs"],
        "eligible_queries": len(eligible),
        "benchmark_query_exclusions": eligibility_exclusions,
        "positive_min": min(len(row) for row in bundle["qrels"].values()),
        "positive_max": max(len(row) for row in bundle["qrels"].values()),
        "sampling_preview": sampling,
        "export_source_fingerprint": export["source_fingerprint"],
        "raw_counts": raw_counts,
        "release_bundle_hash": manifest["bundle_hash"],
        "parse_stats": {
            "raw_documents": raw_counts.get("documents"),
            "raw_queries": raw_counts.get("queries"),
            "raw_positive_pairs": raw_counts.get("positive_pairs"),
            "excluded_over_limit_queries": len(query_exclusions),
            "retained_queries_before_quality": (
                int(raw_counts.get("queries", 0)) - len(query_exclusions)
            ),
            "retained_positive_pairs_before_quality": (
                int(raw_counts.get("positive_pairs", 0)) - excluded_positive_pairs
            ),
            "exact_duplicate_body_groups": duplicate_groups,
            "exact_duplicate_body_occurrences_beyond_first": duplicate_occurrences,
        },
    }
    if raw_snapshot is not None:
        result["raw_snapshot"] = str(Path(raw_snapshot).absolute())
    return result


def _write_receipt_once(path: Path, result: Mapping[str, Any]) -> None:
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"existing STARD receipt differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _check_receipt(path: Path, result: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing STARD receipt: {path}")
    recorded = read_json(path)
    for key in (
        "source_id",
        "dataset",
        "revision",
        "state",
        "release",
        "export",
        "release_completion_sha256",
        "export_completion_sha256",
        "documents",
        "queries",
        "positive_pairs",
        "eligible_queries",
        "benchmark_query_exclusions",
        "export_source_fingerprint",
    ):
        if recorded.get(key) != result.get(key):
            raise ValueError(f"STARD receipt differs at {key}: {path}")


def prepare(data_root: Path, *, raw_snapshot: Path | None = None, io_workers: int = 8) -> dict[str, Any]:
    data_root = Path(data_root)
    release = data_root / RELEASE_RELATIVE_ROOT
    suite = data_root / EXPORT_RELATIVE_ROOT
    raw_snapshot = _get_snapshot(data_root, raw_snapshot)
    corpus, queries = _raw_inputs(raw_snapshot)
    if release.exists():
        result = _check_release(
            data_root,
            raw_snapshot=raw_snapshot,
            io_workers=io_workers,
            check_export_only=False,
        )
        result["raw_source_manifest_sha256"] = file_digest(raw_snapshot / "source_manifest.json")
        _write_receipt_once(data_root / "stard-pos10-receipt.json", result)
        return result

    bundle, _ = build_bundle(corpus, queries)
    manifest = publish_bundle(
        bundle,
        release,
        specification=SPEC,
        raw_snapshot=raw_snapshot,
        policy=POLICY,
    )
    result = _check_release(
        data_root,
        raw_snapshot=raw_snapshot,
        io_workers=io_workers,
        check_export_only=False,
    )
    result["raw_snapshot"] = str(raw_snapshot.absolute())
    result["raw_source_manifest_sha256"] = file_digest(raw_snapshot / "source_manifest.json")
    _write_receipt_once(data_root / "stard-pos10-receipt.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--raw-snapshot", type=Path, help="Reuse a verified connector snapshot")
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify pinned raw files, release, receipt identity, and export without downloading or writing",
    )
    args = parser.parse_args(argv)
    if args.check_only:
        snapshot = _get_snapshot(args.data_root, args.raw_snapshot, create_manifest=False)
        _raw_inputs(snapshot)
        result = _check_release(
            args.data_root,
            raw_snapshot=snapshot,
            io_workers=args.io_workers,
            check_export_only=True,
        )
        _check_receipt(Path(args.data_root) / "stard-pos10-receipt.json", result)
    else:
        result = prepare(args.data_root, raw_snapshot=args.raw_snapshot, io_workers=args.io_workers)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
