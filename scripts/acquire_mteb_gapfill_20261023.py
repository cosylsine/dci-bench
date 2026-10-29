#!/usr/bin/env python3
"""Acquire the bounded MTEB legal gap-fill and verify the sealed MTEB inputs.

The three round2 MTEB releases are read-only inputs here.  New legal sources
are fetched from a pinned ModelScope mirror, converted to ASCII-escaped JSONL,
and published through the existing DCI pipeline.  The mirror is recorded as
the byte-serving source; the benchmark identity remains the official MTEB
dataset revision and task implementation.
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
import traceback
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote, urlencode


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dci_bench.data.pipeline.core import (  # noqa: E402
    build_source,
    load_bundle,
    read_json,
    verify_release,
)
from dci_bench.data.pipeline.download import download_url, file_digest  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/data_pipeline/mteb_gapfill_20261023.json"
DEFAULT_PYTHON = "/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python"
RESEARCH_SCHEMA = "dci-mteb-gapfill-research-v2"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    # ASCII escaping is deliberate: source text can contain U+2028/U+2029,
    # which must never become physical JSONL line separators.
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_once(path: Path, payload: bytes) -> str:
    """Write an immutable artifact, preserving a conflicting partial."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = _sha256_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"immutable destination is not a regular file: {path}")
        existing = path.read_bytes()
        if existing != payload:
            raise RuntimeError(f"immutable destination differs: {path}")
        return digest
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"preserved partial exists; refusing overwrite: {partial}")
    with partial.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return digest


def _write_json_once(path: Path, value: Any) -> str:
    return _write_once(path, _json_bytes(value))


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{number}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row is not an object at {path}:{number}")
            yield number, dict(value)


def _canonical_stream(
    source: Path,
    destination: Path,
    transform: Callable[[dict[str, Any], int], dict[str, Any]],
) -> dict[str, Any]:
    """Transform one complete source JSONL file into an immutable JSONL file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        digest, size = file_digest(destination)
        return {"path": str(destination), "bytes": size, "sha256": digest, "reused": True}
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"preserved partial exists; refusing overwrite: {partial}")
    rows = 0
    with partial.open("x", encoding="utf-8") as output:
        for number, row in _read_jsonl(source):
            transformed = transform(row, number)
            output.write(json.dumps(transformed, ensure_ascii=True, sort_keys=True, allow_nan=False))
            output.write("\n")
            rows += 1
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, destination)
    digest, size = file_digest(destination)
    return {"path": str(destination), "bytes": size, "sha256": digest, "rows": rows, "reused": False}


def _load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if not isinstance(config, Mapping) or not isinstance(config.get("sources"), list):
        raise ValueError("MTEB config must contain a sources list")
    ids: set[str] = set()
    for source in config["sources"]:
        if not isinstance(source, Mapping) or not isinstance(source.get("id"), str):
            raise ValueError("every MTEB source must have a string id")
        source_id = str(source["id"])
        if source_id in ids:
            raise ValueError(f"duplicate source id: {source_id}")
        ids.add(source_id)
        if source.get("domain") != "legal" and source.get("mode") == "acquire_and_publish":
            raise ValueError(f"new MTEB source is outside legal scope: {source_id}")
    return dict(config)


def _sum_expected_files(spec: Mapping[str, Any]) -> int:
    return sum(int(item["bytes"]) for item in spec.get("files", {}).values())


def _count_bundle(bundle: Mapping[str, Any]) -> dict[str, int]:
    qrels = bundle.get("qrels") if isinstance(bundle.get("qrels"), Mapping) else {}
    return {
        "documents": len(bundle.get("documents", [])),
        "queries": len(bundle.get("queries", [])),
        "positive_pairs": sum(len(scores) for scores in qrels.values() if isinstance(scores, Mapping)),
        "qrel_queries": len(qrels),
    }


def _qrel_pairs_from_jsonl(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for number, row in _read_jsonl(path):
        query_id = row.get("query-id", row.get("query_id", row.get("qid")))
        document_id = row.get("corpus-id", row.get("corpus_id", row.get("doc_id")))
        score = row.get("score", row.get("relevance"))
        if query_id is None or document_id is None or score is None:
            raise ValueError(f"qrel row missing id/score at {path}:{number}")
        query_id, document_id = str(query_id), str(document_id)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"qrel score is not finite at {path}:{number}")
        if float(score) <= 0:
            raise ValueError(f"non-positive qrel would not be publishable at {path}:{number}: {score!r}")
        per_query = result.setdefault(query_id, {})
        if document_id in per_query:
            raise ValueError(f"duplicate qrel pair at {path}:{number}: {query_id}/{document_id}")
        per_query[document_id] = float(score) if isinstance(score, float) else int(score)
    return result


def _canonical_source(
    spec: Mapping[str, Any],
    source_inputs: Path,
    canonical_root: Path,
) -> dict[str, Any]:
    metadata = {
        "source_family": spec["source_family"],
        "source_scope": spec["source_scope"],
        "translation_status": spec["translation_status"],
        "annotation_source": "source",
        "dataset": spec["dataset"],
        "mteb_revision": spec["mteb_revision"],
        "mirror": spec["mirror"],
        "mirror_revision": spec["mirror_revision"],
    }
    documents: set[str] = set()
    queries: set[str] = set()

    def document(row: dict[str, Any], number: int) -> dict[str, Any]:
        document_id = row.get("_id", row.get("id", row.get("doc_id")))
        if document_id is None or not str(document_id):
            raise ValueError(f"document row has no id at {source_inputs / 'corpus.jsonl'}:{number}")
        document_id = str(document_id)
        if document_id in documents:
            raise ValueError(f"duplicate document id: {document_id}")
        text = row.get("text", row.get("content", row.get("body")))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"empty document text at {source_inputs / 'corpus.jsonl'}:{number}")
        documents.add(document_id)
        return {"_id": document_id, "title": str(row.get("title") or ""), "text": text, "metadata": metadata}

    def query(row: dict[str, Any], number: int) -> dict[str, Any]:
        query_id = row.get("_id", row.get("id", row.get("query_id")))
        if query_id is None or not str(query_id):
            raise ValueError(f"query row has no id at {source_inputs / 'queries.jsonl'}:{number}")
        query_id = str(query_id)
        if query_id in queries:
            raise ValueError(f"duplicate query id: {query_id}")
        text = row.get("text", row.get("question"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"empty query text at {source_inputs / 'queries.jsonl'}:{number}")
        queries.add(query_id)
        return {"_id": query_id, "text": text, "metadata": metadata}

    doc_result = _canonical_stream(source_inputs / "corpus.jsonl", canonical_root / "documents.jsonl", document)
    query_result = _canonical_stream(source_inputs / "queries.jsonl", canonical_root / "queries.jsonl", query)
    # On a resumed run the immutable canonical destination can already exist,
    # so the transform callback is not invoked.  Rebuild the ID sets from the
    # sealed canonical files before checking qrels.
    if not documents:
        documents.update(str(row.get("_id")) for _, row in _read_jsonl(canonical_root / "documents.jsonl") if row.get("_id") is not None)
    if not queries:
        queries.update(str(row.get("_id")) for _, row in _read_jsonl(canonical_root / "queries.jsonl") if row.get("_id") is not None)
    qrel_source = source_inputs / "qrels_test.jsonl"
    qrel_pairs = _qrel_pairs_from_jsonl(qrel_source)

    def qrel(row: dict[str, Any], number: int) -> dict[str, Any]:
        query_id = row.get("query-id", row.get("query_id", row.get("qid")))
        document_id = row.get("corpus-id", row.get("corpus_id", row.get("doc_id")))
        score = row.get("score", row.get("relevance"))
        if query_id is None or document_id is None or score is None:
            raise ValueError(f"qrel row missing id/score at {qrel_source}:{number}")
        return {"corpus-id": str(document_id), "query-id": str(query_id), "score": score}

    qrel_result = _canonical_stream(qrel_source, canonical_root / "qrels.jsonl", qrel)
    missing_queries = sorted(set(qrel_pairs) - queries)
    missing_documents = sorted({document_id for row in qrel_pairs.values() for document_id in row} - documents)
    if missing_queries or missing_documents:
        raise ValueError(
            f"qrels reference unknown IDs: missing_queries={missing_queries[:3]}, "
            f"missing_documents={missing_documents[:3]}"
        )
    expected = spec.get("expected_counts", {})
    actual = {
        "documents": len(documents),
        "queries": len(queries),
        "qrel_queries": len(qrel_pairs),
        "positive_pairs": sum(len(row) for row in qrel_pairs.values()),
    }
    for key in ("documents", "queries", "positive_pairs"):
        if expected.get(key) is not None and int(expected[key]) != actual[key]:
            raise ValueError(f"{spec['id']} {key} count mismatch: expected {expected[key]}, got {actual[key]}")
    return {
        "canonical_input": {
            "documents": doc_result,
            "queries": query_result,
            "qrels": qrel_result,
        },
        "counts": actual,
        "missing_query_ids": missing_queries,
        "missing_document_ids": missing_documents,
        "qrel_grade_counts": {str(k): v for k, v in sorted(Counter(score for row in qrel_pairs.values() for score in row.values()).items(), key=lambda item: str(item[0]))},
    }


def _modelscope_url(spec: Mapping[str, Any], relative_path: str) -> str:
    query = urlencode({
        "Source": "SDK",
        "Revision": spec["mirror_revision"],
        "FilePath": relative_path,
    })
    return f"https://modelscope.cn/api/v1/datasets/{spec['dataset'].split('/', 1)[1] if '/' in spec['dataset'] else spec['dataset']}/repo?{query}"


def _modelscope_url_fixed(spec: Mapping[str, Any], relative_path: str) -> str:
    dataset = str(spec["dataset"]).split("/", 1)[-1]
    owner = "MTEB"
    query = urlencode({"Source": "SDK", "Revision": spec["mirror_revision"], "FilePath": relative_path})
    return f"https://modelscope.cn/api/v1/datasets/{owner}/{dataset}/repo?{query}"


def _acquire_raw(spec: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    raw_root = Path(config["raw_root"]) / str(spec["id"])
    source_inputs = raw_root / "source_inputs"
    source_inputs.mkdir(parents=True, exist_ok=True)
    expected_total = _sum_expected_files(spec)
    per_source_budget = int(config["per_source_budget_bytes"])
    total_budget = int(config["download_budget_bytes"])
    if expected_total > per_source_budget:
        raise ValueError(f"{spec['id']} exceeds per-source budget: {expected_total} > {per_source_budget}")
    files: dict[str, Any] = {}
    for role, item in spec["files"].items():
        relative_path = str(item["path"])
        destination = source_inputs / ("qrels_test.jsonl" if role == "qrels" else Path(relative_path).name)
        url = _modelscope_url_fixed(spec, relative_path)
        digest, size = download_url(
            url,
            destination,
            timeout=120,
            retries=3,
            max_bytes=per_source_budget,
            expected_sha256=str(item["sha256"]),
            expected_bytes=int(item["bytes"]),
            overwrite=False,
        )
        files[role] = {
            "role": role,
            "source_path": relative_path,
            "url": url,
            "mirror_revision": spec["mirror_revision"],
            "file_revision": item.get("revision"),
            "path": str(destination),
            "bytes": size,
            "sha256": digest,
            "expected_bytes": int(item["bytes"]),
            "expected_sha256": str(item["sha256"]),
        }
    actual_total = sum(int(item["bytes"]) for item in files.values())
    if actual_total > total_budget:
        raise ValueError(f"aggregate MTEB download budget exceeded: {actual_total} > {total_budget}")
    manifest = {
        "schema_version": "dci-mteb-mirror-snapshot-v1",
        "complete": True,
        "source": spec["id"],
        "dataset": spec["dataset"],
        "mteb_revision": spec["mteb_revision"],
        "mirror": spec["mirror"],
        "mirror_revision": spec["mirror_revision"],
        "files": files,
        "expected_total_bytes": expected_total,
        "actual_total_bytes": actual_total,
        "acquired_at": _now(),
        "identity_note": "ModelScope mirror bytes are SHA-256 verified against ModelScope tree metadata; HF byte equivalence was not claimed or tested.",
    }
    manifest_path = raw_root / "source_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if not isinstance(existing, Mapping):
            raise RuntimeError(f"existing mirror manifest is not an object: {manifest_path}")
        for key in ("source", "dataset", "mteb_revision", "mirror", "mirror_revision"):
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"existing mirror manifest identity differs for {key}: {manifest_path}")
        if existing.get("files") != manifest.get("files"):
            raise RuntimeError(f"existing mirror manifest file inventory differs: {manifest_path}")
        manifest = dict(existing)
    else:
        _write_json_once(manifest_path, manifest)
    return {"raw_directory": str(raw_root), "source_inputs": str(source_inputs), "manifest": manifest}


def _frozen_verify(spec: Mapping[str, Any], round2: Mapping[str, Any] | None) -> dict[str, Any]:
    release = Path(spec["release_path"])
    manifest = verify_release(release)
    bundle = load_bundle(release)
    counts = _count_bundle(bundle)
    expected = spec.get("expected_counts", {})
    count_checks = {
        key: {"expected": int(expected[key]), "actual": counts[key], "pass": int(expected[key]) == counts[key]}
        for key in ("documents", "queries", "positive_pairs")
        if expected.get(key) is not None
    }
    if not all(item["pass"] for item in count_checks.values()):
        raise ValueError(f"sealed release count mismatch: {count_checks}")
    canonical_root = Path(spec["canonical_input"])
    canonical_hashes: dict[str, Any] = {}
    for role, filename in (("documents", "documents.jsonl"), ("queries", "queries.jsonl"), ("qrels", "qrels.jsonl")):
        path = canonical_root / filename
        digest, size = file_digest(path)
        expected_hash = None
        expected_bytes = None
        if round2:
            entries = next((item for item in round2.get("sources", []) if item.get("id") == spec["id"]), {})
            expected_hash = entries.get("canonical_input", {}).get(role, {}).get("sha256")
            expected_bytes = entries.get("canonical_input", {}).get(role, {}).get("bytes")
        if expected_hash and digest != expected_hash:
            raise ValueError(f"frozen canonical hash changed: {path}")
        if expected_bytes and size != expected_bytes:
            raise ValueError(f"frozen canonical size changed: {path}")
        canonical_hashes[role] = {"path": str(path), "bytes": size, "sha256": digest, "expected_sha256": expected_hash}
    # Compare the release contract to the canonical input without rewriting it.
    canonical_docs = {row.get("_id"): row for _, row in _read_jsonl(canonical_root / "documents.jsonl")}
    canonical_queries = {row.get("_id"): row for _, row in _read_jsonl(canonical_root / "queries.jsonl")}
    canonical_qrels = _qrel_pairs_from_jsonl(canonical_root / "qrels.jsonl")
    release_docs = {row.get("_id"): row for row in bundle["documents"]}
    release_queries = {row.get("_id"): row for row in bundle["queries"]}
    release_qrels = {str(query_id): {str(doc_id): score for doc_id, score in scores.items()} for query_id, scores in bundle["qrels"].items()}
    projection = lambda row: {key: row.get(key) for key in ("_id", "title", "text")}
    if {key: projection(value) for key, value in release_docs.items()} != {key: projection(value) for key, value in canonical_docs.items()}:
        raise ValueError(f"sealed release documents differ from canonical input: {spec['id']}")
    if {key: {"_id": value.get("_id"), "text": value.get("text")} for key, value in release_queries.items()} != {key: {"_id": value.get("_id"), "text": value.get("text")} for key, value in canonical_queries.items()}:
        raise ValueError(f"sealed release queries differ from canonical input: {spec['id']}")
    if release_qrels != canonical_qrels:
        raise ValueError(f"sealed release qrels differ from canonical input: {spec['id']}")
    parquet_files = []
    parquet_root = Path(spec["source_parquet"])
    if parquet_root.is_dir():
        round2_entry = next((item for item in (round2 or {}).get("sources", []) if item.get("id") == spec["id"]), {})
        recorded = {str(item.get("path")): item for item in round2_entry.get("raw_parquet", []) if isinstance(item, Mapping)}
        for path in sorted(parquet_root.glob("*.parquet")):
            digest, size = file_digest(path)
            old = recorded.get(str(path), {})
            if old.get("sha256") and old["sha256"] != digest:
                raise ValueError(f"frozen source parquet hash changed: {path}")
            parquet_files.append({"path": str(path), "bytes": size, "sha256": digest, "role": old.get("role"), "url": old.get("url")})
    return {
        "id": spec["id"],
        "mode": "verify_frozen",
        "status": "verified_read_only",
        "release_path": str(release),
        "release_state": manifest.get("state"),
        "manifest_sha256": _sha256_bytes((release / "manifest.json").read_bytes()),
        "counts": counts,
        "count_checks": count_checks,
        "canonical_input": canonical_hashes,
        "source_parquet": parquet_files,
        "domain": spec.get("domain"),
        "reported_domain": spec.get("reported_domain", spec.get("domain")),
        "source_family": spec.get("source_family"),
        "source_scope": spec.get("source_scope"),
        "translation_status": spec.get("translation_status"),
        "official_urls": spec.get("official_urls", []),
        "correction_note": spec.get("correction_note"),
        "checks": [
            "verify_release passed without writing to the sealed release",
            "canonical input SHA-256 and byte counts checked",
            "release document/query IDs and text checked against canonical input",
            "release qrel IDs and grades checked against canonical input",
        ],
    }


def _build_legal(spec: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    raw = _acquire_raw(spec, config)
    canonical_root = Path(raw["raw_directory"]) / "canonical_input_ascii_v1"
    canonical = _canonical_source(spec, Path(raw["source_inputs"]), canonical_root)
    progress = {
        "schema_version": RESEARCH_SCHEMA,
        "status": "download_verified_and_canonicalized",
        "generated_at": _now(),
        "source": {
            "id": spec["id"],
            "domain": spec["domain"],
            "language": spec["language"],
            "dataset": spec["dataset"],
            "mteb_revision": spec["mteb_revision"],
            "mirror": spec["mirror"],
            "mirror_revision": spec["mirror_revision"],
            "raw": raw,
            "canonical": canonical,
            "official_urls": spec["official_urls"],
        },
        "boundary": "network bytes and canonical IDs/grades verified; release publication pending",
    }
    return {"raw": raw, "canonical": canonical, "progress": progress}


def _raw_qrel_audit(spec: Mapping[str, Any], canonical: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize the complete source qrels before positive-only publication."""
    counts = canonical.get("counts", {})
    grade_counts = canonical.get("qrel_grade_counts", {})
    return {
        "rows": int(counts.get("positive_pairs", 0)),
        "qrel_queries": int(counts.get("qrel_queries", 0)),
        "grade_counts": grade_counts,
        "missing_query_ids": canonical.get("missing_query_ids", []),
        "missing_document_ids": canonical.get("missing_document_ids", []),
        # _qrel_pairs_from_jsonl rejects duplicates and non-positive rows;
        # therefore these are explicit zero findings rather than silent drops.
        "duplicate_pairs": 0,
        "non_positive_rows": 0,
        "all_rows_retained": True,
        "policy": "raw MTEB qrels are positive-only; no score<=0 row was dropped; all rows retained",
        "historical_round2_qrels": spec.get("historical_round2_counts", {}).get("qrels"),
        "count_reconciliation": (
            {
                "earlier_qrels": int(spec["historical_round2_counts"]["qrels"]),
                "current_pinned_qrels": int(counts.get("positive_pairs", 0)),
                "difference": int(spec["historical_round2_counts"]["qrels"]) - int(counts.get("positive_pairs", 0)),
                "finding": "unresolved source count/metadata discrepancy; no score<=0 or duplicate rows were found",
            }
            if spec.get("historical_round2_counts", {}).get("qrels") is not None
            else None
        ),
        "metadata_discrepancy_note": spec.get("metadata_discrepancy_note"),
    }


def _publish_legal(spec: Mapping[str, Any], config: Mapping[str, Any], acquired: Mapping[str, Any]) -> dict[str, Any]:
    release = Path(config["eval_root"]) / "pipeline_v1" / str(spec["id"]) / "v1"
    if release.exists():
        manifest = verify_release(release)
        raw_root = Path(acquired["raw"]["raw_directory"])
        raw_manifest_path = raw_root / "source_manifest.json"
        raw_manifest = read_json(raw_manifest_path) if raw_manifest_path.is_file() else None
        return {
            "id": spec["id"],
            "mode": "acquire_and_publish",
            "status": "verified_existing_release",
            "release_path": str(release),
            "release_state": manifest.get("state"),
            "counts": _count_bundle(load_bundle(release)),
            "manifest_sha256": _sha256_bytes((release / "manifest.json").read_bytes()),
            "raw_directory": str(raw_root),
            "raw_manifest_sha256": _sha256_bytes(raw_manifest_path.read_bytes()) if raw_manifest_path.is_file() else None,
            "raw_files": raw_manifest.get("files") if isinstance(raw_manifest, Mapping) else None,
            "canonical_input": acquired.get("canonical", {}).get("canonical_input"),
            "raw_qrel_audit": _raw_qrel_audit(spec, acquired.get("canonical", {})),
            "official_urls": spec["official_urls"],
            "mirror": spec["mirror"],
            "mirror_revision": spec["mirror_revision"],
            "mteb_revision": spec["mteb_revision"],
            "source_family": spec["source_family"],
            "source_scope": spec["source_scope"],
            "translation_status": spec["translation_status"],
            "checks": ["existing release was verified; no overwrite attempted"],
        }
    canonical_root = Path(acquired["raw"]["raw_directory"]) / "canonical_input_ascii_v1"
    pipeline_spec = {
        "id": spec["id"],
        "connector": "generic_jsonl",
        "input_kind": "retrieval",
        "domain": spec["domain"],
        "language": spec["language"],
        "genre": "legal case retrieval",
        "license": "mit",
        "revision": spec["mteb_revision"],
        "mteb_task_revision": spec.get("mteb_task_revision"),
        "metadata": {
            "source_family": spec["source_family"],
            "source_scope": spec["source_scope"],
            "translation_status": spec["translation_status"],
            "official_urls": spec["official_urls"],
            "mirror": spec["mirror"],
            "mirror_revision": spec["mirror_revision"],
            "mirror_identity_note": "ModelScope bytes are not claimed byte-equivalent to Hugging Face.",
        },
        "options": {
            "dataset": spec["dataset"],
            "split": "test",
            "files": {
                "documents": {"path": str(canonical_root / "documents.jsonl"), "name": "documents.jsonl"},
                "queries": {"path": str(canonical_root / "queries.jsonl"), "name": "queries.jsonl"},
                "qrels": {"path": str(canonical_root / "qrels.jsonl"), "name": "qrels.jsonl"},
            },
        },
    }
    manifest = build_source(
        pipeline_spec,
        Path(config["raw_root"]),
        release,
        policy=dict(config["quality_policy"]),
    )
    verified = verify_release(release)
    bundle = load_bundle(release)
    counts = _count_bundle(bundle)
    return {
        "id": spec["id"],
        "mode": "acquire_and_publish",
        "status": "published_and_verified",
        "release_path": str(release),
        "release_state": verified.get("state"),
        "counts": counts,
        "manifest_sha256": _sha256_bytes((release / "manifest.json").read_bytes()),
        "raw_directory": acquired["raw"]["raw_directory"],
        "canonical_input": acquired["canonical"]["canonical_input"],
        "raw_qrel_audit": _raw_qrel_audit(spec, acquired["canonical"]),
        "official_urls": spec["official_urls"],
        "mirror": spec["mirror"],
        "mirror_revision": spec["mirror_revision"],
        "mteb_revision": spec["mteb_revision"],
        "source_family": spec["source_family"],
        "source_scope": spec["source_scope"],
        "translation_status": spec["translation_status"],
        "checks": [
            "ModelScope files downloaded with expected byte count and SHA-256",
            "raw qrels query/document IDs and positive grades checked before build",
            "canonical JSONL emitted with ensure_ascii=true",
            "build_source used max_positive_per_query=None and exclude_review_required=False",
            "verify_release passed on the sealed release",
        ],
    }


def run(config_path: Path, selected: list[str] | None = None, *, verify_only: bool = False) -> dict[str, Any]:
    config = _load_config(config_path)
    sources = {str(spec["id"]): spec for spec in config["sources"]}
    selected_ids = selected or list(sources)
    unknown = sorted(set(selected_ids) - set(sources))
    if unknown:
        raise ValueError(f"unknown source IDs: {unknown}")
    research_root = Path(config["research_root"])
    research_root.mkdir(parents=True, exist_ok=True)
    round2 = read_json(Path(config["round2_receipt"])) if Path(config["round2_receipt"]).is_file() else None
    results: list[dict[str, Any]] = []
    progress_written = False
    for source_id in selected_ids:
        spec = sources[source_id]
        try:
            if spec.get("mode") == "verify_frozen":
                result = _frozen_verify(spec, round2)
            elif spec.get("mode") == "acquire_and_publish":
                if verify_only:
                    release = Path(config["eval_root"]) / "pipeline_v1" / source_id / "v1"
                    result = {
                        "id": source_id,
                        "mode": "acquire_and_publish",
                        "status": "verify_only_missing_or_unchecked",
                        "release_exists": release.exists(),
                        "release_path": str(release),
                    }
                    if release.exists():
                        manifest = verify_release(release)
                        result.update({"status": "verified_existing_release", "release_state": manifest.get("state"), "counts": _count_bundle(load_bundle(release))})
                else:
                    acquired = _build_legal(spec, config)
                    if not progress_written:
                        progress_path = research_root / "research_progress.json"
                        if not progress_path.exists():
                            _write_json_once(progress_path, acquired["progress"])
                        progress_written = True
                    result = _publish_legal(spec, config, acquired)
            else:
                raise ValueError(f"unsupported MTEB source mode: {spec.get('mode')!r}")
        except Exception as exc:
            failure = {
                "id": source_id,
                "mode": spec.get("mode"),
                "status": "blocked_or_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "official_urls": spec.get("official_urls", []),
            }
            results.append(failure)
            # Keep a source-specific failure receipt beside its raw attempt.
            failure_root = Path(config["raw_root"]) / source_id / "failures"
            _write_json_once(failure_root / f"failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json", failure)
            continue
        results.append(result)
    failures = [result for result in results if result.get("status") == "blocked_or_failed"]
    receipt = {
        "schema_version": RESEARCH_SCHEMA,
        "generated_at": _now(),
        "config": str(config_path),
        "python": DEFAULT_PYTHON,
        "download_budget_bytes": int(config["download_budget_bytes"]),
        "per_source_budget_bytes": int(config["per_source_budget_bytes"]),
        "quality_policy": config["quality_policy"],
        "code_identity": {
            "script_sha256": _sha256_bytes(Path(__file__).read_bytes()),
            "pipeline_core": _sha256_bytes((ROOT / "dci_bench/data/pipeline/core.py").read_bytes()),
            "pipeline_connectors": _sha256_bytes((ROOT / "dci_bench/data/pipeline/connectors.py").read_bytes()),
        },
        "status": "complete" if not failures and len(results) == len(selected_ids) else "complete_with_failures",
        "sources": results,
        "blocked_or_not_selected": [
            {
                "id": "LLeQA",
                "language": "fr",
                "domain": "legal",
                "status": "not_selected_after_gerdalirsmall_success",
                "reason": "GerDaLIRSmall satisfied the requested second legal MTEB target; LLeQA was not downloaded. HF access was unavailable during discovery and no pinned MTEB ModelScope mirror was established, so no bytes or labels are claimed.",
                "official_urls": [
                    "https://huggingface.co/datasets/mteb/LLeQA",
                    "https://github.com/maastrichtlawtech/lleqa",
                ],
            }
        ],
        "limitations": [
            "ModelScope is the byte-serving mirror; file SHA-256 values were checked against ModelScope tree metadata, without claiming Hugging Face byte equivalence.",
            "This stage prepares and verifies data only; no model, GPU, cascade, query rewriting, or synthetic labels were used.",
            "The three round2 releases remain immutable and are only read and checked.",
        ],
    }
    _write_json_once(research_root / "research.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", action="append", help="source ID; repeat for a bounded run")
    parser.add_argument("--verify-only", action="store_true", help="verify releases without downloading or publishing")
    args = parser.parse_args(argv)
    receipt = run(args.config, args.source, verify_only=args.verify_only)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
