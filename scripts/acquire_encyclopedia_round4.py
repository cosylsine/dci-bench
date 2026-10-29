#!/usr/bin/env python3
"""Acquire and seal the round-four same-language encyclopedia cells.

The input is the pinned MTEB ``MIRACLRetrievalHardNegatives.v2`` development
set.  Its language-specific Parquet files are downloaded as immutable raw
inputs, checked against the Hugging Face LFS byte hashes, and converted into
the local pipeline's JSONL contract.  The original Parquet files and every
zero/relevance label remain in the raw snapshot; the cleaned qrels view keeps
the positive labels required by the DCI pipeline without treating score zero
as a positive judgment.

This script owns only ``r4_encyclopedia_*`` data directories and the round-four
encyclopedia research receipt.  It performs no model, GPU, cascade, or query
generation work.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping
import urllib.error
import urllib.request

try:
    import pyarrow.parquet as parquet
except ImportError as exc:  # pragma: no cover - environment preflight
    raise RuntimeError("pyarrow is required for MIRACL Parquet acquisition") from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dci_bench.data.pipeline.connectors import _spec_fingerprint  # noqa: E402
from dci_bench.data.pipeline.core import (  # noqa: E402
    build_source,
    load_bundle,
    read_json,
    verify_release,
)
from dci_bench.data.pipeline.download import file_digest  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/data_pipeline/encyclopedia_round4.json"
BASE = Path("/mnt/afs2/202609/embedding_models/dci-bench")
RAW_ROOT = BASE / "raw_data"
EVAL_ROOT = BASE / "eval_data"
RESEARCH_ROOT = EVAL_ROOT / "source_research_20261023_round4" / "encyclopedia"
DATASET = "mteb/MIRACLRetrievalHardNegatives"
DATASET_REVISION = "d7d94fa4b946cec4a27c84653aa0cf6b33f74a3c"
MTEB_TASK_REVISION = "2ad77827480c853ee9518eb207a244f13f8a0966"
HF_BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/"
LICENSE = "CC-BY-SA-4.0"
QUALITY_POLICY = {
    "max_positive_per_query": None,
    "drop_near_duplicates": False,
    "exclude_review_required": False,
}
LANGUAGES = ("bn", "fa", "fi", "fr", "id", "ko", "sw", "te", "yo")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_once(path: Path, payload: bytes) -> str:
    """Write an immutable file, preserving an existing conflicting artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"immutable destination is not a regular file: {path}")
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable destination differs; refusing overwrite: {path}")
        return digest
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"preserved partial exists; refusing overwrite: {partial}")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return digest


def _write_json_once(path: Path, value: Any) -> str:
    return _write_once(path, _json_bytes(value))


def _write_jsonl_once(path: Path, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    # Keep physical JSONL records newline-delimited even when source text
    # contains Unicode line/paragraph separators.  Escaping those two code
    # points preserves the decoded text while avoiding splitlines()-based
    # readers treating them as record boundaries.
    lines = []
    for row in rows:
        line = json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False)
        for separator in ("\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"):
            line = line.replace(separator, f"\\u{ord(separator):04x}")
        lines.append((line + "\n").encode("utf-8"))
    payload = b"".join(lines)
    digest = _write_once(path, payload)
    return {"path": str(path), "bytes": len(payload), "sha256": digest, "rows": len(rows)}


def _download_once(url: str, destination: Path, *, expected_bytes: int, expected_sha256: str, timeout: int = 180) -> dict[str, Any]:
    """Download one pinned file with a resumable-safe immutable destination."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        digest, size = file_digest(destination)
        if size != expected_bytes or digest != expected_sha256:
            raise RuntimeError(
                f"existing source file hash/size mismatch: {destination} "
                f"got {size}/{digest}, expected {expected_bytes}/{expected_sha256}"
            )
        return {"url": url, "path": str(destination), "bytes": size, "sha256": digest, "status": "reused_verified"}
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"preserved partial exists; refusing overwrite: {partial}")
    request = urllib.request.Request(url, headers={"User-Agent": "dci-bench-encyclopedia-round4"})
    result = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("xb") as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > expected_bytes:
                    raise RuntimeError(f"download exceeded pinned byte count: {url}")
                result.update(block)
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # The partial is deliberately retained for diagnosis.  It is never a
        # valid source input and a rerun refuses to overwrite it.
        raise
    digest = result.hexdigest()
    if total != expected_bytes or digest != expected_sha256:
        raise RuntimeError(
            f"download hash/size mismatch: {url} got {total}/{digest}, "
            f"expected {expected_bytes}/{expected_sha256}"
        )
    os.replace(partial, destination)
    return {"url": url, "path": str(destination), "bytes": total, "sha256": digest, "status": "fetched_verified"}


def _parquet_rows(path: Path, role: str) -> list[dict[str, Any]]:
    table = parquet.read_table(path)
    rows = [dict(row) for row in table.to_pylist()]
    if role == "corpus":
        required = {"id", "text", "title"}
    elif role == "queries":
        required = {"id", "text"}
    else:
        required = {"query-id", "corpus-id", "score"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"{role} Parquet missing columns {missing}: {path}")
    return rows


def _source_spec(source: Mapping[str, Any], raw_dir: Path) -> dict[str, Any]:
    language = str(source["language"])
    return {
        "id": str(source["id"]),
        "connector": "generic_jsonl",
        "input_kind": "retrieval",
        "domain": "encyclopedia",
        "language": language,
        "genre": "same-language Wikipedia passage retrieval",
        "license": LICENSE,
        "revision": DATASET_REVISION,
        "metadata": {
            "source_family": "MIRACLRetrievalHardNegatives.v2",
            "source_scope": "complete pinned MTEB dev corpus, queries, and qrels for this language",
            "source_dataset": DATASET,
            "mteb_task_revision": MTEB_TASK_REVISION,
            "mteb_dataset_revision": DATASET_REVISION,
            "split": "dev",
            "translation_status": "native",
            "corpus_language_policy": "same-language Wikipedia passages and same-language queries; no translation or padding",
            "hard_negative_policy": "the MTEB corpus is the complete released candidate pool; no candidates were added or removed",
            "raw_qrel_policy": "all source scores, including score zero, remain in source_inputs/qrels_raw.jsonl; canonical qrels retain positive scores only",
            "derived_view_policy": "exact title/text/language aliases are represented explicitly in derived_aliases.jsonl; raw corpus IDs and text remain intact, and near duplicates are never merged",
            "raw_snapshot": str(raw_dir),
        },
        "options": {
            "dataset": DATASET,
            "split": "dev",
            "files": {
                "documents": {"path": str(raw_dir / "canonical_input" / "documents.jsonl"), "name": "documents.jsonl"},
                "queries": {"path": str(raw_dir / "canonical_input" / "queries.jsonl"), "name": "queries.jsonl"},
                "qrels": {"path": str(raw_dir / "canonical_input" / "qrels.json"), "name": "qrels.json"},
            },
        },
        "selection": {"max_queries": None, "seed": 20261023},
    }


def _write_source_manifest(spec: Mapping[str, Any], raw_dir: Path, actual_urls: list[str]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.name in {"source_manifest.json", ".source_manifest.json.partial"}:
            continue
        # Failed/intermediate attempts are deliberately preserved beside the
        # active snapshot, but are not part of the active source inventory.
        # Otherwise merely preserving a prior artifact would mutate the
        # immutable manifest on the next verification pass.
        if any(token in part.lower() for part in path.relative_to(raw_dir).parts for token in ("failed", "prior", "review_required")):
            continue
        relative = path.relative_to(raw_dir).as_posix()
        digest, size = file_digest(path)
        files.append({"path": relative, "bytes": size, "sha256": digest})
    manifest = {
        "schema_version": "dci-source-snapshot-v1",
        "complete": True,
        "id": spec["id"],
        "connector": spec["connector"],
        "input_kind": spec["input_kind"],
        "requested_spec_fingerprint": _spec_fingerprint(spec),
        "spec_fingerprint": _spec_fingerprint(spec),
        "requested_spec": spec,
        "actual_url": actual_urls[0] if len(actual_urls) == 1 else None,
        "actual_urls": actual_urls,
        "resolved_revision": DATASET_REVISION,
        "layout": {
            "documents": ["canonical_input/documents.jsonl"],
            "queries": ["canonical_input/queries.jsonl"],
            "qrels": ["canonical_input/qrels.json"],
        },
        "files": files,
    }
    _write_json_once(raw_dir / "source_manifest.json", manifest)
    return manifest


def _materialize(source: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    source_id = str(source["id"])
    language = str(source["language"])
    raw_dir = RAW_ROOT / source_id
    source_inputs = raw_dir / "source_inputs"
    canonical_dir = raw_dir / "canonical_input"
    source_inputs.mkdir(parents=True, exist_ok=True)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Any] = {}
    actual_urls: list[str] = []
    parquet_paths: dict[str, Path] = {}
    for role in ("corpus", "queries", "qrels"):
        item = source["files"][role]
        relative = str(item["path"])
        destination = source_inputs / f"{role}.parquet"
        url = HF_BASE + relative
        downloaded[role] = _download_once(
            url,
            destination,
            expected_bytes=int(item["bytes"]),
            expected_sha256=str(item["sha256"]),
        )
        actual_urls.append(url)
        parquet_paths[role] = destination

    documents_raw = _parquet_rows(parquet_paths["corpus"], "corpus")
    queries_raw = _parquet_rows(parquet_paths["queries"], "queries")
    qrels_raw = _parquet_rows(parquet_paths["qrels"], "qrels")
    expected = source.get("expected_counts", {})
    if len(documents_raw) != int(expected["documents"]):
        raise ValueError(f"{source_id} corpus count mismatch: {len(documents_raw)} != {expected['documents']}")
    if len(queries_raw) != int(expected["queries"]):
        raise ValueError(f"{source_id} query count mismatch: {len(queries_raw)} != {expected['queries']}")

    document_ids: set[str] = set()
    documents: list[dict[str, Any]] = []
    for index, row in enumerate(documents_raw):
        document_id = str(row["id"])
        text = row["text"]
        title = row["title"]
        if not document_id or not isinstance(text, str) or not text.strip():
            raise ValueError(f"invalid corpus row {index} in {source_id}")
        if document_id in document_ids:
            raise ValueError(f"duplicate corpus ID {document_id} in {source_id}")
        document_ids.add(document_id)
        documents.append({
            "_id": document_id,
            "title": str(title or ""),
            "text": text,
            "language": language,
            "metadata": {
                "source_dataset": DATASET,
                "source_language": language,
                "source_document_id": document_id,
                "source_row_index": index,
                "translation_status": "native",
                "candidate_pool": "MTEB MIRACL hard-negative dev corpus",
            },
        })

    query_ids: set[str] = set()
    queries: list[dict[str, Any]] = []
    for index, row in enumerate(queries_raw):
        query_id = str(row["id"])
        text = row["text"]
        if not query_id or not isinstance(text, str) or not text.strip():
            raise ValueError(f"invalid query row {index} in {source_id}")
        if query_id in query_ids:
            raise ValueError(f"duplicate query ID {query_id} in {source_id}")
        query_ids.add(query_id)
        queries.append({
            "_id": query_id,
            "text": text,
            "language": language,
            "metadata": {
                "source_dataset": DATASET,
                "source_language": language,
                "source_query_id": query_id,
                "source_row_index": index,
                "translation_status": "native",
                "split": "dev",
            },
        })

    qrel_rows: list[dict[str, Any]] = []
    qrels: dict[str, dict[str, int]] = {}
    grade_counts: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    missing_queries: set[str] = set()
    missing_documents: set[str] = set()
    missing_positive_documents: set[str] = set()
    missing_zero_documents: set[str] = set()
    for index, row in enumerate(qrels_raw):
        query_id = str(row["query-id"])
        document_id = str(row["corpus-id"])
        score = row["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"non-finite qrel score at {source_id} row {index}: {score!r}")
        score = int(score) if float(score).is_integer() else float(score)
        pair = (query_id, document_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate qrel pair {pair} in {source_id}")
        seen_pairs.add(pair)
        if query_id not in query_ids:
            missing_queries.add(query_id)
        if document_id not in document_ids:
            missing_documents.add(document_id)
            if float(score) > 0:
                missing_positive_documents.add(document_id)
            else:
                missing_zero_documents.add(document_id)
        grade_counts[str(score)] += 1
        qrel_rows.append({"query-id": query_id, "corpus-id": document_id, "score": score})
        if float(score) > 0:
            qrels.setdefault(query_id, {})[document_id] = score
    # Hard-negative qrels carry score-zero pool members from the original
    # MIRACL judgment set.  Some of those IDs are outside the bounded pooled
    # corpus released by MTEB.  They remain byte-for-byte in qrels_raw.jsonl;
    # only an unresolved query or unresolved positive label blocks the
    # canonical release.  The two missing-ID sets are recorded explicitly so
    # this source-level boundary cannot be mistaken for a hidden exclusion.
    if missing_queries or missing_positive_documents:
        raise ValueError(
            f"qrels reference unknown IDs in {source_id}: "
            f"queries={sorted(missing_queries)[:3]}, positive_documents={sorted(missing_positive_documents)[:3]}"
        )
    if len(qrels) != len(query_ids):
        absent = sorted(query_ids - set(qrels))
        raise ValueError(f"queries without positive qrels in {source_id}: {absent[:5]}")
    positive_pairs = sum(len(row) for row in qrels.values())
    if positive_pairs != int(expected["positive_pairs"]):
        raise ValueError(f"{source_id} positive qrel count mismatch: {positive_pairs} != {expected['positive_pairs']}")

    # The pipeline keeps exact duplicate text outside a positive qrels row as
    # a review condition.  MIRACL's raw candidate pool contains a small number
    # of such aliases.  Build an explicit derived view that removes only
    # unlabelled *exact* aliases, keeps every positive-ID document, and records
    # each mapping.  The original corpus Parquet and IDs remain untouched.
    exact_groups: dict[tuple[str, str, str], list[str]] = {}
    for row in documents:
        exact_groups.setdefault((str(row["title"]), str(row["text"]), str(row["language"])), []).append(str(row["_id"]))
    positive_document_ids = {document_id for scores in qrels.values() for document_id in scores}
    derived_documents: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []
    keep_all: set[str] = set()
    for (title, text, language_key), group_ids in exact_groups.items():
        if len(group_ids) == 1:
            keep_ids = set(group_ids)
        else:
            labelled_ids = sorted(set(group_ids) & positive_document_ids)
            keep_ids = set(labelled_ids or [sorted(group_ids)[0]])
            canonical_id = sorted(keep_ids)[0]
            text_digest = hashlib.sha256((title + "\x00" + text + "\x00" + language_key).encode("utf-8")).hexdigest()
            for alias_id in sorted(set(group_ids) - keep_ids):
                alias_rows.append({
                    "alias_doc_id": alias_id,
                    "canonical_doc_id": canonical_id,
                    "reason": "exact_title_text_language_alias_unlabelled_in_source_qrels",
                    "text_sha256": text_digest,
                    "title": title,
                    "language": language_key,
                })
        keep_all.update(keep_ids)
    for row in documents:
        if row["_id"] in keep_all:
            derived_documents.append(row)
    alias_inventory = _write_jsonl_once(raw_dir / "derived_aliases.jsonl", alias_rows)

    # Keep the full source qrels, including zero grades, as a line-preserving
    # audit surface.  The pipeline's qrels contract intentionally accepts only
    # positive labels; no source row is silently discarded from raw inputs.
    _write_jsonl_once(source_inputs / "qrels_raw.jsonl", qrel_rows)
    _write_json_once(raw_dir / "source_counts.json", {
        "source_dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "language": language,
        "split": "dev",
        "documents": len(documents_raw),
        "derived_documents": len(derived_documents),
        "exact_duplicate_alias_rows": len(alias_rows),
        "exact_duplicate_alias_groups": len({row["text_sha256"] for row in alias_rows}),
        "queries": len(queries_raw),
        "raw_qrel_rows": len(qrel_rows),
        "positive_qrel_rows": positive_pairs,
        "zero_qrel_rows": sum(count for grade, count in grade_counts.items() if float(grade) == 0),
        "qrel_queries": len(qrels),
        "qrel_grade_counts": dict(sorted(grade_counts.items())),
        "missing_query_ids": sorted(missing_queries),
        "missing_document_ids": sorted(missing_documents),
        "missing_positive_document_ids": sorted(missing_positive_documents),
        "missing_zero_document_ids": sorted(missing_zero_documents),
        "positive_qrel_policy": "canonical qrels retain every source score > 0 with the original score; score-zero rows remain in qrels_raw.jsonl and source Parquet",
    })
    canonical_files = {
        "documents": canonical_dir / "documents.jsonl",
        "queries": canonical_dir / "queries.jsonl",
        "qrels": canonical_dir / "qrels.json",
    }
    canonical_inventory = {
        "documents": _write_jsonl_once(canonical_files["documents"], derived_documents),
        "queries": _write_jsonl_once(canonical_files["queries"], queries),
        "qrels": {"path": str(canonical_files["qrels"]), "bytes": len(_json_bytes(qrels)), "sha256": _write_json_once(canonical_files["qrels"], qrels)},
    }
    spec = _source_spec(source, raw_dir)
    manifest = _write_source_manifest(spec, raw_dir, actual_urls)
    return {
        "id": source_id,
        "language": language,
        "raw_dir": str(raw_dir),
        "canonical_dir": str(canonical_dir),
        "spec": spec,
        "manifest": manifest,
        "downloads": downloaded,
        "source_counts": read_json(raw_dir / "source_counts.json"),
        "canonical_inventory": canonical_inventory,
        "alias_inventory": alias_inventory,
        "raw_qrel_rows": qrel_rows,
        "derived_aliases": alias_rows,
        "derived_documents": derived_documents,
        "raw_parquet": {role: downloaded[role] for role in downloaded},
    }


def _reuse_existing(source: Mapping[str, Any]) -> dict[str, Any]:
    """Build a verification-only materialized record from sealed raw inputs.

    This path intentionally performs no rewrite and does not reread the large
    Parquet tables.  It checks the active source manifest, canonical file
    hashes/row counts, and the pinned Parquet byte hashes before the normal
    release verifier checks the sealed pipeline bundle.
    """

    source_id = str(source["id"])
    raw_dir = RAW_ROOT / source_id
    canonical_dir = raw_dir / "canonical_input"
    source_inputs = raw_dir / "source_inputs"
    source_manifest_path = raw_dir / "source_manifest.json"
    counts_path = raw_dir / "source_counts.json"
    if not raw_dir.is_dir() or not canonical_dir.is_dir() or not source_manifest_path.is_file() or not counts_path.is_file():
        raise RuntimeError(f"active raw snapshot incomplete: {raw_dir}")
    spec = _source_spec(source, raw_dir)
    manifest = read_json(source_manifest_path)
    if manifest.get("spec_fingerprint") != _spec_fingerprint(spec):
        raise RuntimeError(f"active source manifest specification differs: {raw_dir}")
    source_counts = read_json(counts_path)

    def _jsonl_inventory(path: Path) -> dict[str, Any]:
        digest, size = file_digest(path)
        rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        return {"path": str(path), "bytes": size, "sha256": digest, "rows": rows}

    canonical_inventory = {
        "documents": _jsonl_inventory(canonical_dir / "documents.jsonl"),
        "queries": _jsonl_inventory(canonical_dir / "queries.jsonl"),
    }
    qrels_path = canonical_dir / "qrels.json"
    qrels_digest, qrels_size = file_digest(qrels_path)
    qrels = read_json(qrels_path)
    canonical_inventory["qrels"] = {"path": str(qrels_path), "bytes": qrels_size, "sha256": qrels_digest, "rows": len(qrels)}
    alias_path = raw_dir / "derived_aliases.jsonl"
    alias_inventory = _jsonl_inventory(alias_path)
    raw_parquet: dict[str, Any] = {}
    for role in ("corpus", "queries", "qrels"):
        item = source["files"][role]
        path = source_inputs / f"{role}.parquet"
        digest, size = file_digest(path)
        if size != int(item["bytes"]) or digest != str(item["sha256"]):
            raise RuntimeError(f"pinned Parquet hash mismatch: {path}")
        raw_parquet[role] = {
            "url": HF_BASE + str(item["path"]),
            "path": str(path),
            "bytes": size,
            "sha256": digest,
            "status": "reused_verified",
        }
    return {
        "id": source_id,
        "language": str(source["language"]),
        "raw_dir": str(raw_dir),
        "canonical_dir": str(canonical_dir),
        "spec": spec,
        "manifest": manifest,
        "downloads": raw_parquet,
        "source_counts": source_counts,
        "canonical_inventory": canonical_inventory,
        "alias_inventory": alias_inventory,
        "derived_aliases": [],
        "derived_documents": [],
        "raw_qrel_rows": [],
        "raw_parquet": raw_parquet,
    }


def _publish(source: Mapping[str, Any], materialized: Mapping[str, Any]) -> dict[str, Any]:
    source_id = str(source["id"])
    release = EVAL_ROOT / "pipeline_v1" / source_id / "v1"
    spec = materialized["spec"]
    if release.exists():
        manifest = verify_release(release)
        if manifest.get("specification") != spec:
            raise RuntimeError(f"existing release specification differs: {release}")
        status = "reused_verified"
    else:
        manifest = build_source(
            spec,
            RAW_ROOT,
            release,
            snapshot=Path(materialized["raw_dir"]),
            policy=dict(QUALITY_POLICY),
        )
        manifest = verify_release(release)
        status = "fetched_verified"
    bundle = load_bundle(release)
    expected = source["expected_counts"]
    counts = {
        "documents": len(bundle["documents"]),
        "queries": len(bundle["queries"]),
        "qrel_queries": len(bundle["qrels"]),
        "positive_pairs": sum(len(row) for row in bundle["qrels"].values()),
    }
    if counts["documents"] != int(materialized["source_counts"]["derived_documents"]):
        raise RuntimeError(f"sealed derived document count mismatch {source_id}: {counts['documents']} != {materialized['source_counts']['derived_documents']}")
    for key in ("queries", "positive_pairs"):
        if counts[key] != int(expected[key]):
            raise RuntimeError(f"sealed release count mismatch {source_id} {key}: {counts[key]} != {expected[key]}")
    if counts["qrel_queries"] != counts["queries"]:
        raise RuntimeError(f"sealed release qrel query coverage mismatch: {source_id}")
    release_state = manifest.get("state")
    if release_state not in {"labels_ready", "review_required"}:
        raise RuntimeError(f"sealed release has an unexpected state: {source_id} state={release_state!r}")
    audit = read_json(release / "audit.json") if (release / "audit.json").is_file() else {}
    if not isinstance(audit, Mapping):
        audit = {}
    return {
        "id": source_id,
        "domain": "encyclopedia",
        "language": source["language"],
        "status": status,
        "state": release_state,
        "ready": release_state == "labels_ready",
        "release_audit": {
            "valid": audit.get("valid"),
            "status": audit.get("status"),
            "review_required": audit.get("review_required", []),
            "exclusions": audit.get("exclusions", []),
            "errors": audit.get("errors", []),
        },
        "path": {"raw": str(materialized["raw_dir"]), "release": str(release), "canonical_input": str(materialized["canonical_dir"])},
        "release_path": str(release),
        "raw_path": str(materialized["raw_dir"]),
        "manifest_sha256": file_digest(release / "manifest.json")[0],
        "raw_manifest_sha256": file_digest(Path(materialized["raw_dir"]) / "source_manifest.json")[0],
        "counts": {**materialized["source_counts"], **counts},
        "D": counts["documents"],
        "Q": counts["queries"],
        "pairs": counts["positive_pairs"],
        "source_counts": materialized["source_counts"],
        "source_parquet": materialized["raw_parquet"],
        "canonical_files": materialized["canonical_inventory"],
        "alias_file": materialized["alias_inventory"],
        "checks": [
            "three pinned Parquet files downloaded or reused with exact byte count and SHA-256",
            "Parquet schema, row counts, unique IDs, and qrel containment checked",
            "all source qrel rows retained in raw qrels_raw.jsonl; canonical qrels retain every positive grade",
            "exact duplicate aliases are explicit in derived_aliases.jsonl; no near-duplicate merge was performed",
            "build_source used max_positive_per_query=None, drop_near_duplicates=False, and exclude_review_required=False",
            "verify_release passed and sealed state is labels_ready or review_required under exclude_review_required=False",
        ],
    }


def _research_entry(source: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    language = str(source["language"])
    urls = [
        f"https://huggingface.co/datasets/{DATASET}/tree/{DATASET_REVISION}",
        f"https://github.com/embeddings-benchmark/mteb/blob/{MTEB_TASK_REVISION}/mteb/tasks/retrieval/multilingual/miracl_retrieval.py",
        "https://project-miracl.github.io/",
    ]
    preserved_versions: list[dict[str, Any]] = []
    if result["id"] == "r4_encyclopedia_yo":
        prior_release = EVAL_ROOT / "pipeline_v1" / "r4_encyclopedia_yo" / "v1_full_review_required"
        if prior_release.is_dir():
            prior_manifest = verify_release(prior_release)
            prior_audit = read_json(prior_release / "audit.json") if (prior_release / "audit.json").is_file() else {}
            prior_qrels = read_json(prior_release / "qrels.json")
            prior_documents = sum(1 for line in (prior_release / "documents.jsonl").open(encoding="utf-8") if line.strip())
            prior_queries = sum(1 for line in (prior_release / "queries.jsonl").open(encoding="utf-8") if line.strip())
            preserved_versions.append({
                "path": str(prior_release),
                "version": "v1_full_review_required",
                "scope": "full raw-corpus canonical view before explicit exact-alias derivation",
                "D": prior_documents,
                "Q": prior_queries,
                "pairs": sum(len(row) for row in prior_qrels.values()),
                "state": prior_manifest.get("state"),
                "audit_status": prior_audit.get("status"),
                "manifest_sha256": file_digest(prior_release / "manifest.json")[0],
                "preserved": True,
            })
    return {
        "id": result["id"],
        "domain": "encyclopedia",
        "language": language,
        "language_name": {"bn": "Bengali", "fa": "Persian", "fi": "Finnish", "fr": "French", "id": "Indonesian", "ko": "Korean", "sw": "Swahili", "te": "Telugu", "yo": "Yoruba"}[language],
        "D": result["D"],
        "Q": result["Q"],
        "pairs": result["pairs"],
        "counts": result["counts"],
        "state": result["state"],
        "ready": result["ready"],
        "status": result["status"],
        "path": result["path"],
        "url": urls,
        "revision": {
            "mteb_dataset": f"{DATASET}@{DATASET_REVISION}",
            "mteb_task_code": f"embeddings-benchmark/mteb@{MTEB_TASK_REVISION}",
            "split": "dev",
        },
        "license": LICENSE,
        "translation": "native",
        "translation_status": "native same-language queries and Wikipedia passages",
        "source_family": "MIRACLRetrievalHardNegatives.v2",
        "source_scope": "complete pinned language-specific MTEB dev candidate pool, queries, and qrels",
        "source_counts": result["source_counts"],
        "source_parquet": result["source_parquet"],
        "canonical_files": result["canonical_files"],
        "alias_file": result["alias_file"],
        "manifest_sha256": result["manifest_sha256"],
        "raw_manifest_sha256": result["raw_manifest_sha256"],
        "release_audit": result["release_audit"],
        "preserved_versions": preserved_versions,
        "checks": result["checks"],
        "limitations": [
            "MIRACL's MTEB hard-negative corpus is a bounded candidate pool formed by the benchmark's released pooling procedure; it is not the multi-million-passage original MIRACL corpus.",
            "Raw score-zero rows are preserved for audit but are not positive qrels in the DCI canonical contract.",
            "This receipt establishes data preparation and label integrity only; no embedding model or GPU benchmark was run.",
        ] + ([
            "The official source qrels contain score-zero IDs outside this bounded MTEB corpus; those rows remain in raw Parquet/qrels_raw.jsonl, while all positive qrel IDs resolve."
        ] if result["source_counts"].get("missing_zero_document_ids") else []),
    }


def _candidate_receipt() -> list[dict[str, Any]]:
    return [
        {
            "id": "MIRACLRetrieval",
            "domain": "encyclopedia",
            "languages": list(LANGUAGES),
            "status": "candidate_not_downloaded",
            "reason": "The full original MIRACL corpora are much larger; the pinned MTEB MIRACLRetrievalHardNegatives.v2 release is the bounded official MTEB task used for this round.",
            "url": "https://huggingface.co/datasets/mteb/MIRACLRetrieval",
            "revision": "9c09abc13478308c27598f350e31d8f06b9b5481",
        },
        {
            "id": "MIRACLRetrievalHardNegatives.v2_other_languages",
            "domain": "encyclopedia",
            "languages": ["ar", "de", "en", "es", "hi", "ja", "ru", "th", "zh"],
            "status": "already_ready_or_existing_matrix",
            "reason": "Those languages are outside this delegated new-language scope.",
            "url": f"https://huggingface.co/datasets/{DATASET}/tree/{DATASET_REVISION}",
            "revision": DATASET_REVISION,
        },
    ]


def run(
    config_path: Path,
    selected: list[str] | None = None,
    *,
    verify_only: bool = False,
    write_research: bool = True,
) -> dict[str, Any]:
    config = read_json(config_path)
    if not isinstance(config, Mapping) or not isinstance(config.get("sources"), list):
        raise ValueError("round-four config must contain a sources list")
    sources = {str(item["id"]): item for item in config["sources"]}
    selected_ids = selected or list(sources)
    unknown = sorted(set(selected_ids) - set(sources))
    if unknown:
        raise ValueError(f"unknown source IDs: {unknown}")
    RESEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    commands: list[str] = []
    for source_id in selected_ids:
        source = sources[source_id]
        try:
            raw_dir = RAW_ROOT / source_id
            release = EVAL_ROOT / "pipeline_v1" / source_id / "v1"
            if verify_only:
                if not raw_dir.exists() or not release.exists():
                    raise RuntimeError(f"verify-only artifact missing: raw={raw_dir.exists()} release={release.exists()}")
                materialized = _reuse_existing(source)
                result = _publish(source, materialized)
            else:
                materialized = _materialize(source, config)
                result = _publish(source, materialized)
            entries.append(_research_entry(source, result))
            commands.append(f"/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python {ROOT / 'scripts/acquire_encyclopedia_round4.py'} --source {source_id}")
        except Exception as exc:
            failure = {
                "id": source_id,
                "domain": "encyclopedia",
                "language": source.get("language"),
                "status": "failed_preserved",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "path": {"raw": str(RAW_ROOT / source_id), "release": str(EVAL_ROOT / "pipeline_v1" / source_id / "v1")},
                "url": f"https://huggingface.co/datasets/{DATASET}/tree/{DATASET_REVISION}",
                "revision": DATASET_REVISION,
            }
            failures.append(failure)

    source_by_id = {str(item["id"]): item for item in config["sources"]}
    has_entries = bool(entries)
    no_failures = not failures

    def _entry_source_counts(entry: Mapping[str, Any]) -> Mapping[str, Any]:
        counts = entry.get("source_counts", {})
        return counts if isinstance(counts, Mapping) else {}

    owned_files = [
        str(ROOT / "scripts/acquire_encyclopedia_round4.py"),
        str(config_path),
        str(ROOT / "docs/encyclopedia_round4.md"),
        str(RESEARCH_ROOT / "research.json"),
        *[entry["path"]["raw"] for entry in entries],
        *[entry["path"]["release"] for entry in entries],
    ]
    preserved_yo_release = EVAL_ROOT / "pipeline_v1" / "r4_encyclopedia_yo" / "v1_full_review_required"
    if preserved_yo_release.is_dir():
        owned_files.append(str(preserved_yo_release))

    research = {
        "schema_version": "dci-encyclopedia-round4-research-v1",
        "created_at": _now(),
        "status": "complete" if not failures and len(entries) == len(selected_ids) else "complete_with_failures",
        "domain": "encyclopedia",
        "scope": "new absent-language same-language cells only",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "mteb_task_revision": MTEB_TASK_REVISION,
        "license": LICENSE,
        "sources": entries,
        "acquired_verified": [entry["id"] for entry in entries if entry.get("state") == "labels_ready"],
        "failures": failures,
        "candidates": _candidate_receipt(),
        "quality_policy": QUALITY_POLICY,
        "raw_label_policy": "all Parquet qrel rows and score-zero labels are preserved in source_inputs/qrels.parquet and qrels_raw.jsonl; canonical qrels retain all source score>0 labels with their original grades; unresolved score-zero IDs are explicit source_counts fields, never hidden drops",
        "checks": {
            # Every aggregate is explicitly false for an incomplete run.  This
            # prevents a failure-only receipt from reporting vacuous success.
            "all_qrel_ids_resolve": bool(has_entries and no_failures and all(
                not _entry_source_counts(entry).get("missing_query_ids")
                and not _entry_source_counts(entry).get("missing_document_ids")
                for entry in entries
            )),
            "all_positive_qrel_ids_resolve": bool(has_entries and no_failures and all(
                not _entry_source_counts(entry).get("missing_query_ids")
                and not _entry_source_counts(entry).get("missing_positive_document_ids")
                for entry in entries
            )),
            "raw_zero_qrel_ids_resolve": bool(has_entries and no_failures and all(
                not _entry_source_counts(entry).get("missing_zero_document_ids")
                for entry in entries
            )),
            "full_source_counts_compare": bool(has_entries and no_failures and all(
                _entry_source_counts(entry).get("documents") == source_by_id[entry["id"]]["expected_counts"]["documents"]
                and _entry_source_counts(entry).get("queries") == source_by_id[entry["id"]]["expected_counts"]["queries"]
                and _entry_source_counts(entry).get("positive_qrel_rows") == source_by_id[entry["id"]]["expected_counts"]["positive_pairs"]
                and _entry_source_counts(entry).get("derived_documents") == entry["D"]
                and entry["Q"] == source_by_id[entry["id"]]["expected_counts"]["queries"]
                and entry["pairs"] == source_by_id[entry["id"]]["expected_counts"]["positive_pairs"]
                for entry in entries
            )),
            "all_releases_sealed": bool(has_entries and no_failures and all(
                entry.get("state") in {"labels_ready", "review_required"} for entry in entries
            )),
            "sealed_release_state_labels_ready": bool(has_entries and no_failures and all(
                entry.get("state") == "labels_ready" for entry in entries
            )),
            "all_exact_aliases_explicit": bool(has_entries and no_failures and all(
                int(_entry_source_counts(entry).get("exact_duplicate_alias_rows", -1))
                == int(entry.get("alias_file", {}).get("rows", -2))
                and Path(str(entry.get("alias_file", {}).get("path", ""))).is_file()
                for entry in entries
            )),
            "no_query_drops": bool(has_entries and no_failures and all(
                _entry_source_counts(entry).get("queries") == entry["Q"] for entry in entries
            )),
            "no_hidden_exclusions": bool(has_entries and no_failures and all(
                _entry_source_counts(entry).get("raw_qrel_rows") == _entry_source_counts(entry).get("positive_qrel_rows", 0) + _entry_source_counts(entry).get("zero_qrel_rows", 0)
                and _entry_source_counts(entry).get("raw_qrel_rows") == sum(_entry_source_counts(entry).get("qrel_grade_counts", {}).values())
                for entry in entries
            )),
        },
        "reproduction": {
            "python": "/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python",
            "command": f"/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python {ROOT / 'scripts/acquire_encyclopedia_round4.py'} --config {config_path}",
            "verify_commands": [
                f"/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python scripts/data_pipeline.py verify /mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/r4_encyclopedia_<lang>/v1",
                f"/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python {ROOT / 'scripts/acquire_encyclopedia_round4.py'} --config {config_path} --verify-only",
            ],
            "commands_by_source": commands,
            "gpu_or_cascade": "not run",
        },
        "owned_files": owned_files,
    }
    if write_research:
        _write_json_once(RESEARCH_ROOT / "research.json", research)
    return research


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", action="append", help="source ID; repeat for a bounded run")
    parser.add_argument("--verify-only", action="store_true", help="verify already sealed raw/release artifacts without changing them")
    parser.add_argument("--no-write-research", action="store_true", help="run acquisition and checks without writing the aggregate research receipt")
    args = parser.parse_args(argv)
    selected = args.source
    research = run(args.config, selected, verify_only=args.verify_only, write_research=not args.no_write_research)
    print(json.dumps({
        "status": research["status"],
        "acquired_verified": research["acquired_verified"],
        "failures": [{"id": item["id"], "error": item["error"]} for item in research["failures"]],
        "release_counts": [{"id": item["id"], "D": item["D"], "Q": item["Q"], "pairs": item["pairs"], "state": item["state"]} for item in research["sources"]],
    }, ensure_ascii=False, indent=2))
    return 0 if research["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
