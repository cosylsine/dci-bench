#!/usr/bin/env python3
"""Acquire and seal the bounded multilingual MTEB mMARCO web views.

The MTEB task uses a bounded candidate pool rather than the full multilingual
mMARCO corpus.  This script downloads the exact pinned parquet files, checks
the raw IDs/text/grades before conversion, publishes one immutable DCI
retrieval release per language, and compares every released row back to the
canonical input.  It never translates or labels source data locally.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote
import urllib.request

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dci_bench.data.pipeline.core import build_source, load_bundle, publish_bundle, read_json, verify_release  # noqa: E402
from dci_bench.data.pipeline.download import download_url, file_digest  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/data_pipeline/web_round4.json"
DEFAULT_PYTHON = "/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python"
RESEARCH_SCHEMA = "dci-web-round4-research-v1"
CANONICAL_SCHEMA = "dci-web-round4-canonical-v1"
RAW_SCHEMA = "dci-web-round4-raw-v1"


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_once(path: Path, payload: bytes) -> str:
    """Write an immutable artifact and preserve conflicting partials."""

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


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row is not an object at {path}:{number}")
            yield dict(value)


def _iter_parquet(path: Path, columns: list[str]) -> Iterable[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"{path} is missing parquet columns: {missing}")
    for batch in parquet.iter_batches(batch_size=8192, columns=columns):
        arrays = [batch.column(index).to_pylist() for index in range(len(columns))]
        for values in zip(*arrays):
            yield dict(zip(columns, values))


def _require_string(value: Any, *, field: str, path: Path, row_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}:{row_number} has invalid {field}: {value!r}")
    return value


def _metadata(spec: Mapping[str, Any], identifier: str, kind: str) -> dict[str, Any]:
    return {
        "annotation_source": "source",
        "family_id": f"mmarco-v2:{kind}:{identifier}",
        "mteb_task": spec["mteb_task"],
        "source_family": spec["source_family"],
        "source_scope": spec["source_scope"],
        "source_split": spec["split"],
        "translation_of": f"MSMARCO:{kind}:{identifier}",
        "translation_status": "translated",
    }


def _document_row(raw: Mapping[str, Any], spec: Mapping[str, Any], row_number: int) -> dict[str, Any]:
    path = Path(str(spec["raw_parquet_paths"]["documents"]))
    identifier = _require_string(raw.get("id"), field="id", path=path, row_number=row_number)
    text = _require_string(raw.get("text"), field="text", path=path, row_number=row_number)
    title = raw.get("title")
    if title is None:
        title = ""
    if not isinstance(title, str):
        raise ValueError(f"{path}:{row_number} has invalid title: {title!r}")
    return {
        "_id": identifier,
        "title": title,
        "text": text,
        "language": spec["language"],
        "metadata": _metadata(spec, identifier, "document"),
    }


def _query_row(raw: Mapping[str, Any], spec: Mapping[str, Any], row_number: int) -> dict[str, Any]:
    path = Path(str(spec["raw_parquet_paths"]["queries"]))
    identifier = _require_string(raw.get("id"), field="id", path=path, row_number=row_number)
    text = _require_string(raw.get("text"), field="text", path=path, row_number=row_number)
    return {
        "_id": identifier,
        "text": text,
        "language": spec["language"],
        "metadata": _metadata(spec, identifier, "query"),
    }


def _qrel_row(raw: Mapping[str, Any], spec: Mapping[str, Any], row_number: int) -> dict[str, Any]:
    path = Path(str(spec["raw_parquet_paths"]["qrels"]))
    query_id = _require_string(raw.get("query-id"), field="query-id", path=path, row_number=row_number)
    document_id = _require_string(raw.get("corpus-id"), field="corpus-id", path=path, row_number=row_number)
    score = raw.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError(f"{path}:{row_number} has invalid score: {score!r}")
    if float(score) <= 0:
        raise ValueError(f"{path}:{row_number} has non-positive score: {score!r}")
    return {"query-id": query_id, "corpus-id": document_id, "score": int(score) if int(score) == score else float(score)}


def _stream_jsonl_once(destination: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        digest, size = file_digest(destination)
        return {"path": str(destination), "bytes": size, "sha256": digest, "reused": True}
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"preserved partial exists; refusing overwrite: {partial}")
    rows_count = 0
    with partial.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True, allow_nan=False))
            handle.write("\n")
            rows_count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    digest, size = file_digest(destination)
    return {"path": str(destination), "bytes": size, "sha256": digest, "rows": rows_count, "reused": False}


def _source_file_url(spec: Mapping[str, Any], relative_path: str) -> str:
    base = f"{spec.get('download_base_url', 'https://huggingface.co')}/datasets/{spec['dataset']}/resolve/{spec['dataset_revision']}/"
    return base + quote(relative_path, safe="/") + "?download=true"


def _download_files(spec: Mapping[str, Any], config: Mapping[str, Any], raw_dir: Path) -> dict[str, Any]:
    downloads = raw_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    per_file_budget = int(config["per_file_budget_bytes"])
    total_budget = int(config["download_budget_bytes"])
    actual_total = 0
    files: dict[str, Any] = {}
    for role, item in spec["files"].items():
        relative_path = str(item["path"])
        destination = downloads / relative_path
        url = _source_file_url({**spec, "download_base_url": config.get("download_base_url", "https://huggingface.co")}, relative_path)
        digest, size = download_url(
            url,
            destination,
            timeout=120,
            retries=3,
            max_bytes=per_file_budget,
            expected_sha256=str(item["sha256"]),
            expected_bytes=int(item["bytes"]),
            overwrite=False,
            opener=urllib.request.build_opener(),
        )
        actual_total += size
        files[role] = {
            "role": role,
            "source_path": relative_path,
            "url": url,
            "dataset_revision": spec["dataset_revision"],
            "file_revision": item.get("file_revision"),
            "path": str(destination),
            "bytes": size,
            "sha256": digest,
            "expected_bytes": int(item["bytes"]),
            "expected_sha256": str(item["sha256"]),
        }
    if actual_total > total_budget:
        raise ValueError(f"{spec['id']} download budget exceeded: {actual_total} > {total_budget}")
    return {"files": files, "actual_total_bytes": actual_total}


def _write_raw_manifest(spec: Mapping[str, Any], raw_dir: Path, download: Mapping[str, Any]) -> dict[str, Any]:
    manifest = {
        "schema_version": RAW_SCHEMA,
        "complete": True,
        "source": spec["id"],
        "dataset": spec["dataset"],
        "dataset_revision": spec["dataset_revision"],
        "mteb_task": spec["mteb_task"],
        "mteb_task_revision": spec["mteb_task_revision"],
        "split": spec["split"],
        "files": download["files"],
        "actual_total_bytes": download["actual_total_bytes"],
        "identity_note": "HF LFS sha256 and byte counts are pinned; this is the bounded MTEB candidate pool, not the full parent mMARCO corpus.",
    }
    path = raw_dir / "source_manifest.json"
    if path.exists():
        existing = read_json(path)
        if existing != manifest:
            raise RuntimeError(f"raw source manifest differs; refusing overwrite: {path}")
        return dict(existing)
    _write_json_once(path, manifest)
    return manifest


def _compare_stream(
    raw_path: Path,
    canonical_path: Path,
    columns: list[str],
    transform: Callable[[Mapping[str, Any], int], Mapping[str, Any]],
) -> dict[str, Any]:
    raw_rows = _iter_parquet(raw_path, columns)
    canonical_rows = _iter_jsonl(canonical_path)
    count = 0
    for count, raw in enumerate(raw_rows, 1):
        try:
            canonical = next(canonical_rows)
        except StopIteration as exc:
            raise ValueError(f"canonical file ended early at {canonical_path}, row {count}") from exc
        expected = dict(transform(raw, count))
        if canonical != expected:
            raise ValueError(f"raw/canonical row mismatch at {raw_path}:{count}")
    try:
        extra = next(canonical_rows)
    except StopIteration:
        extra = None
    if extra is not None:
        raise ValueError(f"canonical file has extra rows at {canonical_path}: {extra.get('_id', extra)}")
    return {"raw_rows": count, "canonical_rows": count, "all_rows_retained": True}


def _alias_groups(rows: Mapping[str, Mapping[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for identifier, row in rows.items():
        key = _sha256_bytes(
            json.dumps([row.get(field) for field in fields], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        groups[key].append(identifier)
    return [
        {"alias_sha256": key, "fields": list(fields), "ids": sorted(ids), "count": len(ids)}
        for key, ids in sorted(groups.items())
        if len(ids) > 1
    ]


def _load_canonical_audit(spec: Mapping[str, Any], canonical_root: Path) -> dict[str, Any]:
    documents: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(_iter_jsonl(canonical_root / "documents.jsonl"), 1):
        identifier = row.get("_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"invalid canonical document ID at row {row_number}")
        if identifier in documents:
            raise ValueError(f"duplicate canonical document ID: {identifier}")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"empty canonical document text at row {row_number}")
        documents[identifier] = row

    queries: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(_iter_jsonl(canonical_root / "queries.jsonl"), 1):
        identifier = row.get("_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"invalid canonical query ID at row {row_number}")
        if identifier in queries:
            raise ValueError(f"duplicate canonical query ID: {identifier}")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"empty canonical query text at row {row_number}")
        queries[identifier] = row

    qrels: dict[str, dict[str, float]] = {}
    raw_rows = 0
    grade_counts: Counter[str] = Counter()
    for row_number, row in enumerate(_iter_jsonl(canonical_root / "qrels.jsonl"), 1):
        raw_rows += 1
        query_id = row.get("query-id")
        document_id = row.get("corpus-id")
        score = row.get("score")
        if query_id not in queries:
            raise ValueError(f"qrel references unknown query at row {row_number}: {query_id}")
        if document_id not in documents:
            raise ValueError(f"qrel references unknown document at row {row_number}: {document_id}")
        if query_id in qrels and document_id in qrels[query_id]:
            raise ValueError(f"duplicate qrel pair at row {row_number}: {query_id}/{document_id}")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)) or float(score) <= 0:
            raise ValueError(f"invalid qrel grade at row {row_number}: {score!r}")
        qrels.setdefault(str(query_id), {})[str(document_id)] = float(score)
        grade_counts[str(score)] += 1

    expected = spec["expected_counts"]
    actual = {
        "documents": len(documents),
        "queries": len(queries),
        "qrel_queries": len(qrels),
        "positive_pairs": raw_rows,
    }
    for key in ("documents", "queries", "positive_pairs"):
        if actual[key] != int(expected[key]):
            raise ValueError(f"{spec['id']} {key} mismatch: expected {expected[key]}, got {actual[key]}")
    if actual["qrel_queries"] != actual["queries"]:
        raise ValueError(f"{spec['id']} qrels do not cover every query: {actual}")
    return {
        "counts": actual,
        "grade_counts": dict(sorted(grade_counts.items())),
        "duplicate_document_ids": 0,
        "duplicate_query_ids": 0,
        "duplicate_qrel_pairs": 0,
        "missing_qrel_queries": [],
        "missing_qrel_documents": [],
        "all_ids_text_grades_resolve": True,
        "document_exact_alias_groups": _alias_groups(documents, ("title", "text", "language")),
        "query_exact_alias_groups": _alias_groups(queries, ("text", "language")),
        "documents": documents,
        "queries": queries,
        "qrels": qrels,
    }


def _pipeline_spec(spec: Mapping[str, Any], canonical_root: Path) -> dict[str, Any]:
    return {
        "id": spec["id"],
        "connector": "generic_jsonl",
        "input_kind": "retrieval",
        "domain": "web",
        "language": spec["language"],
        "genre": "web search passage retrieval",
        "license": spec["license"],
        "revision": spec["dataset_revision"],
        "metadata": {
            "source_family": spec["source_family"],
            "source_scope": spec["source_scope"],
            "translation_status": spec["translation_status"],
            "official_urls": spec["official_urls"],
            "mteb_task": spec["mteb_task"],
            "mteb_task_revision": spec["mteb_task_revision"],
        },
        "options": {
            "dataset": spec["dataset"],
            "split": spec["split"],
            "files": {
                "documents": {"path": str(canonical_root / "documents.jsonl"), "name": "documents.jsonl"},
                "queries": {"path": str(canonical_root / "queries.jsonl"), "name": "queries.jsonl"},
                "qrels": {"path": str(canonical_root / "qrels.jsonl"), "name": "qrels.jsonl"},
            },
        },
    }


def _compare_release(spec: Mapping[str, Any], release: Path, canonical: Mapping[str, Any]) -> dict[str, Any]:
    manifest = verify_release(release)
    bundle = load_bundle(release)
    release_documents = {str(row.get("_id")): row for row in bundle["documents"]}
    release_queries = {str(row.get("_id")): row for row in bundle["queries"]}
    expected_documents = canonical["documents"]
    expected_queries = canonical["queries"]
    if release_documents != expected_documents:
        raise ValueError(f"released documents differ from canonical input: {spec['id']}")
    if release_queries != expected_queries:
        raise ValueError(f"released queries differ from canonical input: {spec['id']}")
    release_qrels = {
        str(query_id): {str(document_id): float(score) for document_id, score in scores.items()}
        for query_id, scores in bundle["qrels"].items()
    }
    if release_qrels != canonical["qrels"]:
        raise ValueError(f"released qrels differ from canonical input: {spec['id']}")
    release_files = {}
    for filename in ("documents.jsonl", "queries.jsonl", "qrels.json", "bundle_metadata.json", "manifest.json", "audit.json", "splits.json", "_COMPLETE.json"):
        digest, size = file_digest(release / filename)
        release_files[filename] = {"bytes": size, "sha256": digest}
    return {
        "status": "verified",
        "release_path": str(release),
        "release_state": manifest["state"],
        "manifest_sha256": _sha256_bytes((release / "manifest.json").read_bytes()),
        "documents_exact": True,
        "queries_exact": True,
        "qrels_exact": True,
        "release_counts": {
            "documents": len(bundle["documents"]),
            "queries": len(bundle["queries"]),
            "positive_pairs": sum(len(scores) for scores in bundle["qrels"].values()),
        },
        "release_files": release_files,
    }


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        for row in rows
    )


def _derive_exact_alias_release(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    parent_release: Path,
    raw_dir: Path,
) -> dict[str, Any]:
    """Publish a reversible view that aliases only exact document duplicates.

    The official bounded release remains immutable at ``parent_release``.  The
    derived view uses the exact ``(title, text, language)`` key, keeps one
    canonical document row per group, and remaps qrels through an explicit
    alias inventory.  Conflicting grades for one query after remapping fail
    closed; no near-duplicate or query filtering is attempted.
    """

    parent_release = Path(parent_release)
    parent_manifest = verify_release(parent_release)
    parent_bundle = load_bundle(parent_release)
    documents = list(parent_bundle["documents"])
    queries = list(parent_bundle["queries"])
    source_qrels = parent_bundle["qrels"]
    parent_by_id = {str(row["_id"]): row for row in documents}
    if len(parent_by_id) != len(documents):
        raise ValueError(f"parent release has duplicate document IDs: {parent_release}")

    groups: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in documents:
        key = (str(row.get("title", "")), str(row.get("text", "")), str(row.get("language", "")))
        groups[key].append(str(row["_id"]))

    aliases: dict[str, str] = {}
    alias_rows: list[dict[str, Any]] = []
    merged_groups: list[dict[str, Any]] = []
    for key, group_ids in groups.items():
        original_ids = sorted(group_ids)
        canonical_id = original_ids[0]
        for document_id in original_ids:
            aliases[document_id] = canonical_id
        if len(original_ids) <= 1:
            continue
        title, text, language = key
        key_digest = _sha256_bytes(json.dumps(list(key), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        merged_groups.append({
            "canonical_id": canonical_id,
            "original_document_ids": original_ids,
            "alias_key_fields": ["title", "text", "language"],
            "alias_key_sha256": key_digest,
        })
        for alias_id in original_ids:
            if alias_id == canonical_id:
                continue
            alias_rows.append({
                "alias_doc_id": alias_id,
                "canonical_doc_id": canonical_id,
                "original_document_ids": original_ids,
                "reason": "exact_title_text_language_alias",
                "title": title,
                "language": language,
                "text_sha256": _sha256_bytes(text.encode("utf-8")),
            })

    remapped_qrels: dict[str, dict[str, Any]] = {}
    same_grade_collisions: list[dict[str, Any]] = []
    for query_id, judgments in source_qrels.items():
        mapped: dict[str, Any] = {}
        for document_id, grade in judgments.items():
            document_id = str(document_id)
            if document_id not in aliases:
                raise ValueError(f"qrel references unknown parent document during aliasing: {document_id}")
            canonical_id = aliases[document_id]
            if canonical_id in mapped:
                if float(mapped[canonical_id]) != float(grade):
                    raise ValueError(
                        "conflicting grades after exact alias mapping: "
                        f"query={query_id} canonical={canonical_id}"
                    )
                same_grade_collisions.append({
                    "query_id": str(query_id),
                    "canonical_doc_id": canonical_id,
                    "grade": grade,
                    "original_doc_id": document_id,
                })
                continue
            mapped[canonical_id] = grade
        remapped_qrels[str(query_id)] = mapped

    kept_ids = {group["canonical_id"] for group in merged_groups}
    kept_ids.update(document_id for document_id, canonical_id in aliases.items() if document_id == canonical_id)
    derived_documents: list[dict[str, Any]] = []
    for row in documents:
        document_id = str(row["_id"])
        if document_id not in kept_ids:
            continue
        canonical_id = aliases[document_id]
        result = copy.deepcopy(row)
        group = next((item for item in merged_groups if item["canonical_id"] == canonical_id), None)
        if group is not None:
            metadata = result.get("metadata")
            if not isinstance(metadata, Mapping):
                metadata = {}
            result["metadata"] = {
                **dict(metadata),
                "exact_alias_canonical_id": canonical_id,
                "exact_document_aliases": list(group["original_document_ids"]),
            }
        for field in ("_id", "title", "text", "language"):
            if result.get(field) != parent_by_id[document_id].get(field):
                raise ValueError(f"derived alias changed source field {field}: {document_id}")
        derived_documents.append(result)

    parent_qrel_pairs = sum(len(scores) for scores in source_qrels.values())
    derived_qrel_pairs = sum(len(scores) for scores in remapped_qrels.values())
    if [row.get("_id") for row in queries] != [row.get("_id") for row in parent_bundle["queries"]]:
        raise ValueError("exact alias view changed query IDs")
    for query_id, judgments in source_qrels.items():
        for document_id, grade in judgments.items():
            canonical_id = aliases[str(document_id)]
            if float(remapped_qrels[str(query_id)][canonical_id]) != float(grade):
                raise ValueError(f"exact alias grade changed: {query_id}/{document_id}")

    base_id = str(spec["id"])
    derived_id = f"{base_id}_exact_alias"
    output = Path(config["eval_root"]) / "pipeline_v1" / derived_id / "v1"
    alias_path = Path(raw_dir) / "derived_aliases.jsonl"
    alias_file = {"path": str(alias_path), "rows": len(alias_rows), "sha256": _write_once(alias_path, _jsonl_bytes(alias_rows))}
    derived_spec = copy.deepcopy(dict(parent_manifest["specification"]))
    derived_spec["id"] = derived_id
    derived_spec.setdefault("metadata", {})
    derived_spec["metadata"] = {
        **dict(derived_spec["metadata"]),
        "source_scope": (
            "explicitly derived exact-alias view of the official bounded MTEB dev release; "
            "only exact title/text/language duplicate documents are aliased; "
            "the immutable parent release preserves every original ID/text/qrel grade"
        ),
        "parent_release": str(parent_release),
        "exact_alias_key": ["title", "text", "language"],
        "exact_alias_grade_policy": "equal grades required after mapping; conflicting grades fail closed",
    }
    derived_bundle = copy.deepcopy(parent_bundle)
    derived_bundle["documents"] = derived_documents
    derived_bundle["qrels"] = remapped_qrels
    provenance = derived_bundle.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    derived_bundle["provenance"] = {
        **dict(provenance),
        "source_id": derived_id,
        "parent_release": str(parent_release),
        "exact_alias_derivation": {
            "key_fields": ["title", "text", "language"],
            "grade_policy": "equal grades required after mapping; conflicting grades fail closed",
            "merged_groups": merged_groups,
            "alias_file": str(alias_path),
        },
    }
    if output.exists():
        derived_manifest = verify_release(output)
        if derived_manifest.get("specification") != derived_spec:
            raise RuntimeError(f"existing exact-alias specification differs: {output}")
    else:
        publish_bundle(
            derived_bundle,
            output,
            specification=derived_spec,
            parent_release=parent_release,
            policy=dict(config["quality_policy"]),
        )
    derived_manifest = verify_release(output)
    actual = load_bundle(output)
    if actual["queries"] != queries or actual["qrels"] != remapped_qrels or actual["documents"] != derived_documents:
        raise ValueError(f"exact-alias release differs from its derived input: {output}")
    if set(actual["qrels"]) != set(source_qrels):
        raise ValueError("exact-alias release dropped a query qrel row")
    derived_complete_sha256 = file_digest(output / "_COMPLETE.json")[0]
    audit = {
        "schema_version": "dci-web-round4-exact-alias-v1",
        "source_id": derived_id,
        "parent_release": str(parent_release),
        "parent_manifest_sha256": file_digest(parent_release / "manifest.json")[0],
        "parent_completion_sha256": file_digest(parent_release / "_COMPLETE.json")[0],
        "derived_release": str(output),
        "derived_completion_sha256": derived_complete_sha256,
        "key_fields": ["title", "text", "language"],
        "merged_groups": merged_groups,
        "alias_file": alias_file,
        "parent_counts": {
            "documents": len(documents),
            "queries": len(queries),
            "positive_pairs": parent_qrel_pairs,
        },
        "derived_counts": {
            "documents": len(derived_documents),
            "queries": len(actual["queries"]),
            "positive_pairs": sum(len(scores) for scores in actual["qrels"].values()),
        },
        "same_grade_qrel_collisions": same_grade_collisions,
        "raw_parent_release_preserved": True,
        "raw_qrel_grades_rechecked_under_aliases": True,
        "near_duplicates_merged": False,
        "conflicting_mapped_grades": [],
        "release_state": derived_manifest["state"],
        "checks": [
            "parent verify_release passed before derivation",
            "all exact aliases use title/text/language equality only",
            "all parent query IDs retained",
            "every parent qrel grade rechecked under its canonical alias",
            "conflicting mapped grades fail closed",
            "immutable parent v1 remains available for full source comparison",
            "derived release verify_release passed",
        ],
    }
    audit_path = Path(raw_dir) / "derived_alias_audit.json"
    _write_json_once(audit_path, audit)
    return {
        "id": derived_id,
        "status": "derived_verified",
        "parent_id": base_id,
        "parent_release": str(parent_release),
        "release_path": str(output),
        "release_state": derived_manifest["state"],
        "ready": derived_manifest["state"] == "labels_ready",
        "D": len(actual["documents"]),
        "Q": len(actual["queries"]),
        "pairs": sum(len(scores) for scores in actual["qrels"].values()),
        "merged_document_groups": len(merged_groups),
        "aliased_duplicate_documents": len(documents) - len(derived_documents),
        "alias_file": alias_file,
        "audit_path": str(audit_path),
        "audit_sha256": file_digest(audit_path)[0],
        "manifest_sha256": file_digest(output / "manifest.json")[0],
        "checks": audit["checks"],
    }


def _process_source(spec: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    raw_root = Path(config["raw_root"])
    eval_root = Path(config["eval_root"])
    raw_dir = raw_root / str(spec["id"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    downloaded = _download_files(spec, config, raw_dir)
    raw_manifest = _write_raw_manifest(spec, raw_dir, downloaded)

    raw_paths = {
        role: str(item["path"])
        for role, item in downloaded["files"].items()
    }
    source_for_conversion = dict(spec)
    source_for_conversion["raw_parquet_paths"] = raw_paths
    canonical_root = raw_dir / "canonical_input_ascii_v1"
    canonical_root.mkdir(parents=True, exist_ok=True)
    _stream_jsonl_once(
        canonical_root / "documents.jsonl",
        (_document_row(row, source_for_conversion, number) for number, row in enumerate(_iter_parquet(Path(raw_paths["documents"]), ["id", "title", "text"]), 1)),
    )
    _stream_jsonl_once(
        canonical_root / "queries.jsonl",
        (_query_row(row, source_for_conversion, number) for number, row in enumerate(_iter_parquet(Path(raw_paths["queries"]), ["id", "text"]), 1)),
    )
    _stream_jsonl_once(
        canonical_root / "qrels.jsonl",
        (_qrel_row(row, source_for_conversion, number) for number, row in enumerate(_iter_parquet(Path(raw_paths["qrels"]), ["query-id", "corpus-id", "score"]), 1)),
    )

    raw_canonical_checks = {
        "documents": _compare_stream(
            Path(raw_paths["documents"]), canonical_root / "documents.jsonl", ["id", "title", "text"],
            lambda row, number: _document_row(row, source_for_conversion, number),
        ),
        "queries": _compare_stream(
            Path(raw_paths["queries"]), canonical_root / "queries.jsonl", ["id", "text"],
            lambda row, number: _query_row(row, source_for_conversion, number),
        ),
        "qrels": _compare_stream(
            Path(raw_paths["qrels"]), canonical_root / "qrels.jsonl", ["query-id", "corpus-id", "score"],
            lambda row, number: _qrel_row(row, source_for_conversion, number),
        ),
    }
    canonical = _load_canonical_audit(spec, canonical_root)

    release = eval_root / "pipeline_v1" / str(spec["id"]) / "v1"
    pipeline_spec = _pipeline_spec(spec, canonical_root)
    if release.exists():
        verify_release(release)
    else:
        build_source(
            pipeline_spec,
            raw_root,
            release,
            policy=dict(config["quality_policy"]),
        )
    release_comparison = _compare_release(spec, release, canonical)
    conversion_audit = {
        "schema_version": CANONICAL_SCHEMA,
        "source": {
            "id": spec["id"],
            "dataset": spec["dataset"],
            "dataset_revision": spec["dataset_revision"],
            "mteb_task": spec["mteb_task"],
            "mteb_task_revision": spec["mteb_task_revision"],
            "split": spec["split"],
            "language": spec["language"],
            "domain": "web",
            "source_family": spec["source_family"],
            "source_scope": spec["source_scope"],
            "translation_status": spec["translation_status"],
        },
        "raw_files": downloaded["files"],
        "raw_canonical_comparison": raw_canonical_checks,
        "canonical_counts": canonical["counts"],
        "qrel_grade_counts": canonical["grade_counts"],
        "duplicate_scan": {
            "exact_key_fields_documents": ["title", "text", "language"],
            "exact_key_fields_queries": ["text", "language"],
            "document_exact_alias_groups": canonical["document_exact_alias_groups"],
            "query_exact_alias_groups": canonical["query_exact_alias_groups"],
            "near_duplicates_merged": False,
            "all_duplicate_ids_preserved": True,
            "all_qrel_rows_preserved": True,
        },
        "quality_policy": config["quality_policy"],
        "canonical_files": {
            name: {"path": str(canonical_root / name), "sha256": file_digest(canonical_root / name)[0], "bytes": file_digest(canonical_root / name)[1]}
            for name in ("documents.jsonl", "queries.jsonl", "qrels.jsonl")
        },
        "release_comparison": release_comparison,
        "validation": [
            "pinned parquet bytes and SHA-256 verified",
            "raw parquet rows compared one-for-one to canonical JSONL",
            "all source qrel IDs and positive grades resolved to actual canonical rows",
            "release verify_release passed",
            "released IDs/text/metadata and qrels compared to canonical input",
        ],
    }
    conversion_path = raw_dir / "conversion_audit.json"
    _write_json_once(conversion_path, conversion_audit)
    raw_manifest_path = raw_dir / "source_manifest.json"
    return {
        "id": spec["id"],
        "status": "acquired_verified",
        "domain": "web",
        "language": spec["language"],
        "D": canonical["counts"]["documents"],
        "Q": canonical["counts"]["queries"],
        "pairs": canonical["counts"]["positive_pairs"],
        "documents": canonical["counts"]["documents"],
        "queries": canonical["counts"]["queries"],
        "positive_pairs": canonical["counts"]["positive_pairs"],
        "release_state": release_comparison["release_state"],
        "path": str(release),
        "url": spec["official_urls"][0],
        "revision": spec["dataset_revision"],
        "translation": spec["translation_status"],
        "license": spec["license"],
        "source_family": spec["source_family"],
        "source_scope": spec["source_scope"],
        "raw_directory": str(raw_dir),
        "raw_manifest": str(raw_manifest_path),
        "raw_manifest_sha256": file_digest(raw_manifest_path)[0],
        "conversion_audit": str(conversion_path),
        "conversion_audit_sha256": file_digest(conversion_path)[0],
        "manifest_sha256": release_comparison["manifest_sha256"],
        "checks": {
            "raw_ids_text_grades_full_comparison": True,
            "release_ids_text_grades_full_comparison": True,
            "qrels_resolve_to_actual_documents": True,
            "all_rows_retained": True,
            "near_duplicates_dropped": False,
            "max_positive_per_query": None,
            "exclude_review_required": False,
        },
    }


def _load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if not isinstance(config, Mapping) or not isinstance(config.get("sources"), list):
        raise ValueError("web round4 config must contain a sources list")
    seen: set[str] = set()
    for source in config["sources"]:
        if not isinstance(source, Mapping) or not isinstance(source.get("id"), str):
            raise ValueError("every web source must have a string id")
        source_id = str(source["id"])
        if source_id in seen:
            raise ValueError(f"duplicate source id: {source_id}")
        seen.add(source_id)
        if source.get("domain") != "web":
            raise ValueError(f"source outside web scope: {source_id}")
        if source.get("language") in {"en", "zh"}:
            raise ValueError(f"source language is already reserved/excluded: {source_id}")
        for key in ("dataset_revision", "mteb_task_revision", "dataset", "files", "expected_counts"):
            if key not in source:
                raise ValueError(f"{source_id} missing {key}")
    return dict(config)


def _write_receipt(research_root: Path, receipt: Mapping[str, Any]) -> Path:
    research_root.mkdir(parents=True, exist_ok=True)
    path = research_root / "research.json"
    payload = _json_bytes(receipt)
    if not path.exists():
        _write_once(path, payload)
        return path
    existing = read_json(path)
    if existing == receipt:
        return path
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    retry = research_root / f"research_retry_{stamp}.json"
    _write_once(retry, payload)
    return retry


def run(config_path: Path, selected: list[str] | None = None, *, verify_only: bool = False) -> dict[str, Any]:
    config = _load_config(config_path)
    source_specs = {str(spec["id"]): dict(spec) for spec in config["sources"]}
    selected_ids = selected or list(source_specs)
    unknown = sorted(set(selected_ids) - set(source_specs))
    if unknown:
        raise ValueError(f"unknown source IDs: {unknown}")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if verify_only:
        for source_id in selected_ids:
            spec = source_specs[source_id]
            release = Path(config["eval_root"]) / "pipeline_v1" / source_id / "v1"
            if not release.exists():
                results.append({"id": source_id, "status": "missing_release", "path": str(release)})
                continue
            try:
                manifest = verify_release(release)
                results.append({"id": source_id, "status": "verified_existing_release", "release_state": manifest["state"], "path": str(release)})
            except Exception as exc:
                failures.append({"id": source_id, "status": "blocked_or_failed", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    else:
        for source_id in selected_ids:
            spec = source_specs[source_id]
            try:
                result = _process_source(spec, config)
                if result.get("release_state") == "review_required":
                    # Preserve the official v1 and expose a separate ready
                    # view only when exact alias remapping is grade-safe.
                    derived = _derive_exact_alias_release(
                        spec,
                        config,
                        Path(result["path"]),
                        Path(result["raw_directory"]),
                    )
                    result["derived_alias"] = derived
                    result["selected_release_state"] = derived["release_state"]
                    result["selected_path"] = derived["release_path"]
                    result["selected_ready"] = derived["ready"]
                results.append(result)
            except Exception as exc:
                failure = {
                    "id": source_id,
                    "status": "blocked_or_failed",
                    "domain": "web",
                    "language": spec.get("language"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "url": spec.get("official_urls", [None])[0],
                    "revision": spec.get("dataset_revision"),
                }
                failures.append(failure)
                failure_dir = Path(config["raw_root"]) / source_id / "failures"
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                _write_json_once(failure_dir / f"failure_{stamp}.json", failure)

    receipt = {
        "schema_version": RESEARCH_SCHEMA,
        "status": "complete" if not failures and len(results) == len(selected_ids) else "complete_with_failures",
        "acquired_verified": not failures and len(results) == len(selected_ids),
        "config": str(config_path),
        "python": DEFAULT_PYTHON,
        "dataset": "lopozz/MMarcoRetrievalMultilingual",
        "dataset_revision": "44480423f56a42b948edd899c707e640b8e46f19",
        "mteb_task": "MMarcoRetrievalMultilingual",
        "mteb_task_revision": "2ad77827480c853ee9518eb207a244f13f8a0966",
        "sources": results,
        "blocked": failures,
        "candidates": [
            {
                "id": "JaCWIRRetrieval",
                "language": "ja",
                "domain": "web",
                "status": "candidate_not_selected",
                "reason": "MMarcoRetrievalMultilingual already provides the requested bounded Japanese web cell; no second Japanese source was required for this round.",
                "url": "https://docs.mteb.org/overview/available_tasks/retrieval/",
            },
            {
                "id": "MIRACLRetrievalHardNegatives",
                "language": "multilingual",
                "domain": "encyclopedia",
                "status": "excluded_out_of_scope",
                "reason": "MIRACL is Wikipedia/encyclopedic and is owned by the separate encyclopedia worker; it is not claimed as web coverage here.",
                "url": "https://docs.mteb.org/overview/available_tasks/retrieval/",
            },
        ],
        "quality_policy": config["quality_policy"],
        "limitations": [
            "The official MTEB task describes a 100,000-candidate sample; the pinned dev parquet contains 106,813 corpus rows including its positive closure.",
            "This is an official bounded dev retrieval view, not the full 8M+ multilingual mMARCO parent corpus and not a full-parent retrieval claim.",
            "All non-English views are translated family members of MS MARCO; translation_status is recorded and no translation was performed locally.",
            "Data preparation and release verification do not imply model, GPU, cascade, or benchmark execution.",
        ],
    }
    research_root = Path(config["research_root"])
    receipt_path = _write_receipt(research_root, receipt)
    receipt = dict(receipt)
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def finalize_research(config_path: Path) -> dict[str, Any]:
    """Seal one stable all-language receipt from completed per-run receipts.

    Historical failure and partial receipts are left untouched.  The newest
    complete entry for each configured source is selected, so a later exact
    alias view can supersede its base review-required entry while the base
    release remains recorded in that source entry.
    """

    config = _load_config(config_path)
    research_root = Path(config["research_root"])
    source_specs = {str(item["id"]): item for item in config["sources"]}
    receipt_paths = sorted(
        path for path in research_root.glob("research*.json")
        if path.name != "research_final.json"
    )
    latest: dict[str, tuple[float, Path, dict[str, Any]]] = {}
    complete_receipts: list[Path] = []
    for path in receipt_paths:
        try:
            data = read_json(path)
        except Exception:
            continue
        if data.get("status") != "complete" or not isinstance(data.get("sources"), list):
            continue
        complete_receipts.append(path)
        for entry in data["sources"]:
            source_id = str(entry.get("id", ""))
            if source_id not in source_specs:
                continue
            marker = (path.stat().st_mtime, path, entry)
            if source_id not in latest or marker[0] >= latest[source_id][0]:
                latest[source_id] = marker
    missing = [source_id for source_id in source_specs if source_id not in latest]
    if missing:
        raise RuntimeError(f"cannot finalize research; missing complete source receipts: {missing}")

    # Use the most complete receipt for shared provenance/candidate metadata.
    base_receipt_path = max(
        complete_receipts,
        key=lambda path: (len(read_json(path).get("sources", [])), path.stat().st_mtime),
    )
    base_receipt = read_json(base_receipt_path)
    selected_sources: list[dict[str, Any]] = []
    source_receipts: dict[str, dict[str, Any]] = {}
    for source_id in source_specs:
        _, receipt_path, entry = latest[source_id]
        source_entry = copy.deepcopy(entry)
        source_receipts[source_id] = {
            "path": str(receipt_path),
            "sha256": file_digest(receipt_path)[0],
        }
        base_release = Path(str(source_entry["path"]))
        base_manifest = verify_release(base_release)
        selected = {
            "id": source_id,
            "path": str(base_release),
            "release_state": base_manifest["state"],
            "ready": base_manifest["state"] == "labels_ready",
            "D": int(source_entry["D"]),
            "Q": int(source_entry["Q"]),
            "pairs": int(source_entry["pairs"]),
            "view": "official_bounded_v1",
        }
        derived = source_entry.get("derived_alias")
        if isinstance(derived, Mapping):
            derived_release = Path(str(derived["release_path"]))
            derived_manifest = verify_release(derived_release)
            selected = {
                "id": str(derived["id"]),
                "parent_id": source_id,
                "parent_release": str(base_release),
                "path": str(derived_release),
                "release_state": derived_manifest["state"],
                "ready": derived_manifest["state"] == "labels_ready",
                "D": int(derived["D"]),
                "Q": int(derived["Q"]),
                "pairs": int(derived["pairs"]),
                "view": "exact_title_text_language_alias_derived",
                "alias_file": derived["alias_file"],
                "audit_path": derived["audit_path"],
                "audit_sha256": derived["audit_sha256"],
                "merged_document_groups": int(derived["merged_document_groups"]),
                "aliased_duplicate_documents": int(derived["aliased_duplicate_documents"]),
                "manifest_sha256": derived["manifest_sha256"],
                "checks": derived["checks"],
            }
        source_entry["selected_release"] = selected
        source_entry["source_receipt"] = source_receipts[source_id]
        selected_sources.append(source_entry)

    final = {
        "schema_version": RESEARCH_SCHEMA,
        "status": "complete",
        "acquired_verified": True,
        "finalized_from_receipts": [source_receipts[source_id] for source_id in source_specs],
        "base_receipt": {
            "path": str(base_receipt_path),
            "sha256": file_digest(base_receipt_path)[0],
        },
        "config": str(config_path),
        "python": DEFAULT_PYTHON,
        "dataset": base_receipt.get("dataset"),
        "dataset_revision": base_receipt.get("dataset_revision"),
        "mteb_task": base_receipt.get("mteb_task"),
        "mteb_task_revision": base_receipt.get("mteb_task_revision"),
        "quality_policy": config["quality_policy"],
        "sources": selected_sources,
        "selected_release_count": sum(bool(row["selected_release"]["ready"]) for row in selected_sources),
        "selected_release_languages": [row["language"] for row in selected_sources if row["selected_release"]["ready"]],
        "selected_release_policy": "one selected labels_ready view per source ID; exact-alias views are derived children and official v1 remains preserved",
        "blocked": [],
        "candidates": base_receipt.get("candidates", []),
        "limitations": base_receipt.get("limitations", []),
        "checks": {
            "all_configured_sources_present": len(selected_sources) == len(source_specs),
            "all_base_releases_verify_release": True,
            "all_selected_releases_verify_release": True,
            "all_selected_releases_labels_ready": all(row["selected_release"]["ready"] for row in selected_sources),
            "all_selected_qrels_resolve": all(bool(row["checks"]["qrels_resolve_to_actual_documents"]) for row in selected_sources),
            "all_raw_ids_text_grades_compared": all(bool(row["checks"]["raw_ids_text_grades_full_comparison"]) for row in selected_sources),
            "all_release_ids_text_grades_compared": all(bool(row["checks"]["release_ids_text_grades_full_comparison"]) for row in selected_sources),
            "near_duplicates_dropped": any(bool(row["checks"]["near_duplicates_dropped"]) for row in selected_sources),
            "base_review_required_views_preserved": all(
                row["release_state"] in {"labels_ready", "review_required"} for row in selected_sources
            ),
        },
    }
    final_path = research_root / "research_final.json"
    _write_json_once(final_path, final)
    final["receipt_path"] = str(final_path)
    return final


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", action="append", help="source ID; repeat for a bounded run")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--finalize", action="store_true", help="seal research_final.json from completed per-run receipts")
    args = parser.parse_args(argv)
    if args.finalize:
        if args.source or args.verify_only:
            parser.error("--finalize cannot be combined with --source or --verify-only")
        receipt = finalize_research(args.config)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "complete" else 2
    receipt = run(args.config, args.source, verify_only=args.verify_only)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
