#!/usr/bin/env python3
"""Acquire and seal round-4 scientific/medical/finance/argument/legal cells.

The benchmark identity is the pinned MTEB dataset revision and task source.
HF is used as the byte-serving source for the Dutch releases; pinned ModelScope
commits are used for the Polish parquet mirrors.  Source files are retained
under ``raw_data/r4_domains_*`` and are converted to the existing immutable
pipeline contract only after IDs, text, splits, and qrel grades are audited.

This script deliberately does not translate data, invent labels, rank pools, or
run a model.  It keeps non-positive raw qrel rows in the raw snapshot while
publishing only the positive qrel relation required by the pipeline contract,
with every such decision recorded in the per-source audit and research.json.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping
from urllib.parse import quote, urlencode
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dci_bench.data.pipeline.core import build_source, load_bundle, read_json, verify_release  # noqa: E402
from dci_bench.data.pipeline.download import download_url, file_digest  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/data_pipeline/domains_round4.json"
PYTHON = "/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python"
RESEARCH_SCHEMA = "dci-domains-round4-research-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_once(path: Path, value: Any | bytes) -> str:
    """Write an immutable artifact, accepting an identical existing file."""
    payload = value if isinstance(value, bytes) else _json_bytes(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
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


def _write_json_reuse(path: Path, value: Any) -> Any:
    """Create a JSON artifact once and return the preserved value on reruns."""
    if path.exists():
        existing = read_json(path)
        if not isinstance(existing, Mapping):
            raise RuntimeError(f"existing JSON artifact is not an object: {path}")
        return existing
    _write_once(path, value)
    return value


def _write_research_snapshot(path: Path, value: Any) -> None:
    """Publish the current run while retaining an earlier failed receipt."""
    payload = _json_bytes(value)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"research destination is not a regular file: {path}")
        if path.read_bytes() == payload:
            return
        previous_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        preserved = path.with_name(f"{path.name}.previous-{previous_digest[:16]}")
        if preserved.exists():
            if preserved.read_bytes() != path.read_bytes():
                raise RuntimeError(f"research preservation collision: {preserved}")
            suffix = 1
            while True:
                candidate = path.with_name(f"{path.name}.previous-{previous_digest[:16]}-{suffix}")
                if not candidate.exists():
                    preserved = candidate
                    break
                suffix += 1
        path.rename(preserved)
    _write_once(path, payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row is not an object: {path}:{number}")
            rows.append(dict(value))
    return rows


def _iter_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    # ``to_pylist`` converts Arrow scalars to ordinary JSON-friendly Python
    # values and retains every source column for the raw-to-release audit.
    return [dict(row) for row in table.to_pylist()]


def _sha256(path: Path) -> tuple[str, int]:
    return file_digest(path)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _first(row: Mapping[str, Any], keys: Iterable[str], *, where: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    raise ValueError(f"row has no expected field at {where}: {sorted(row)}")


def _text(row: Mapping[str, Any], *, where: str, query: bool = False) -> str:
    keys = ("text", "question", "query", "content", "body") if query else ("text", "content", "body", "document", "passage")
    value = _first(row, keys, where=where)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"empty text at {where}")
    return value


def _id(row: Mapping[str, Any], *, where: str, query: bool = False) -> str:
    if query:
        keys = ("_id", "id", "query_id", "query-id", "qid")
    else:
        keys = ("_id", "id", "doc_id", "document_id", "corpus_id", "corpus-id")
    value = _first(row, keys, where=where)
    identifier = str(value)
    if not identifier:
        raise ValueError(f"empty ID at {where}")
    return identifier


def _score(row: Mapping[str, Any], *, where: str) -> float:
    value = _first(row, ("score", "relevance", "label", "relevance_score"), where=where)
    if isinstance(value, bool):
        raise ValueError(f"boolean qrel score at {where}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric qrel score at {where}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite qrel score at {where}")
    return number


def _qrel_ids(row: Mapping[str, Any], *, where: str) -> tuple[str, str]:
    query = str(_first(row, ("query-id", "query_id", "qid", "query"), where=where))
    document = str(_first(row, ("corpus-id", "corpus_id", "doc_id", "document_id", "docid"), where=where))
    if not query or not document:
        raise ValueError(f"empty qrel ID at {where}")
    return query, document


def _safe_extra(row: Mapping[str, Any], reserved: set[str]) -> dict[str, Any]:
    """Keep source columns in metadata without allowing NaN into JSON."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in reserved:
            continue
        if hasattr(value, "as_py"):
            value = value.as_py()
        if isinstance(value, float) and not math.isfinite(value):
            value = str(value)
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            value = str(value)
        out[str(key)] = value
    return out


def _hf_tree(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    url = f"https://huggingface.co/api/datasets/{spec['dataset']}/tree/{quote(str(spec['revision']), safe='')}?recursive=true&limit=1000"
    request = urllib.request.Request(url, headers={"User-Agent": "dci-bench/domains-round4"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise RuntimeError(f"HF tree response is not a list: {spec['id']}")
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def _modelscope_tree(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    owner, dataset = str(spec["dataset"]).split("/", 1)
    url = f"https://modelscope.cn/api/v1/datasets/{owner}/{dataset}/repo/tree?Revision={quote(str(spec['mirror_revision']))}&Recursive=true"
    request = urllib.request.Request(url, headers={"User-Agent": "dci-bench/domains-round4"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    files = payload.get("Data", {}).get("Files") if isinstance(payload, Mapping) else None
    if not isinstance(files, list):
        raise RuntimeError(f"ModelScope tree response has no files: {spec['id']}")
    normalized: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        # ModelScope uses title-case keys while the HF tree uses lower-case.
        if "path" not in row and "Path" in row:
            row["path"] = row["Path"]
        if "size" not in row and "Size" in row:
            row["size"] = row["Size"]
        if "sha256" not in row and "Sha256" in row:
            row["sha256"] = row["Sha256"]
        normalized.append(row)
    return normalized


def _download_url(spec: Mapping[str, Any], relative: str) -> str:
    if spec["source_kind"].startswith("hf"):
        # Resolve is required for Git-LFS-backed files (raw would return the
        # small LFS pointer). The proxy opener below follows this redirect to
        # the pinned object while the revision remains explicit in manifests.
        return f"https://huggingface.co/datasets/{spec['dataset']}/resolve/{quote(str(spec['revision']), safe='')}/{quote(relative, safe='/')}"
    owner, dataset = str(spec["dataset"]).split("/", 1)
    query = urlencode({"Source": "SDK", "Revision": spec["mirror_revision"], "FilePath": relative})
    return f"https://modelscope.cn/api/v1/datasets/{owner}/{dataset}/repo?{query}"


def _tree_entry(tree: list[dict[str, Any]], relative: str) -> dict[str, Any] | None:
    return next((item for item in tree if str(item.get("path", "")) == relative), None)


def _acquire_source(spec: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    raw = Path(config["raw_root"]) / str(spec["id"])
    inputs = raw / "source_inputs"
    upstream = inputs / "upstream"
    inputs.mkdir(parents=True, exist_ok=True)
    tree = _hf_tree(spec) if spec["source_kind"].startswith("hf") else _modelscope_tree(spec)
    _write_json_reuse(raw / "mirror_tree.json", {"retrieved_at": _now(), "source": spec["id"], "files": tree})
    files: list[dict[str, Any]] = []
    for relative in spec["source_files"]:
        entry = _tree_entry(tree, relative)
        if entry is None:
            raise RuntimeError(f"{spec['id']} pinned tree has no requested file {relative}")
        expected_bytes = int(entry["size"]) if entry.get("size") is not None else None
        expected_sha = None
        if spec["source_kind"].startswith("modelscope"):
            expected_sha = str(entry.get("sha256") or "") or None
            expected = spec.get("mirror_file_hashes", {}).get(relative)
            if expected is None or int(expected[0]) != expected_bytes or str(expected[1]).lower() != expected_sha.lower():
                raise RuntimeError(f"{spec['id']} ModelScope tree hash differs from pinned config for {relative}")
        destination = upstream / relative
        partial = destination.with_name(destination.name + ".partial")
        # Preserve any failed attempt before retrying from byte zero.  In
        # particular, HF raw served an LFS pointer for several files; that
        # pointer must never be appended to the resolved object.
        if partial.exists() and expected_bytes is not None and partial.stat().st_size != expected_bytes:
            failed_digest, failed_size = _sha256(partial)
            preserved_partial = partial.with_name(
                f"{partial.name}.failed-{failed_size}-{failed_digest[:16]}"
            )
            if not preserved_partial.exists():
                partial.rename(preserved_partial)
            else:
                raise RuntimeError(f"preserved retry artifact already exists: {preserved_partial}")
        # The shared downloader defaults to a direct opener; HF is reachable
        # through the configured proxy in this environment.
        opener = urllib.request.build_opener() if spec["source_kind"].startswith("hf") else None
        digest, size = download_url(
            _download_url(spec, relative), destination,
            timeout=120, retries=3, max_bytes=int(config["per_source_budget_bytes"]),
            expected_bytes=expected_bytes, expected_sha256=expected_sha, overwrite=False,
            opener=opener,
        )
        files.append({
            "path": relative,
            "url": _download_url(spec, relative),
            "bytes": size,
            "sha256": digest,
            "tree_sha256": expected_sha,
            "tree_bytes": expected_bytes,
        })
    manifest = {
        "schema_version": "dci-r4-mirror-snapshot-v1",
        "complete": True,
        "source": spec["id"],
        "dataset": spec["dataset"],
        "official_revision": spec["revision"],
        "mirror": spec["mirror"],
        "mirror_revision": spec["mirror_revision"],
        "source_kind": spec["source_kind"],
        "files": files,
        "acquired_at": _now(),
        "identity_note": "Official MTEB revision and byte-serving mirror identity are recorded separately; HF and ModelScope bytes are not claimed equivalent.",
    }
    preserved = _write_json_reuse(raw / "source_manifest.json", manifest)
    manifest = dict(preserved)
    return {"raw": raw, "inputs": inputs, "upstream": upstream, "tree": tree, "files": files, "manifest": manifest}


def _canonical_from_rows(spec: Mapping[str, Any], acquired: Mapping[str, Any]) -> dict[str, Any]:
    inputs = Path(acquired["inputs"])
    upstream = Path(acquired["upstream"])
    canonical = inputs / "canonical_input"
    canonical.mkdir(parents=True, exist_ok=True)
    source_meta = {
        "source_family": spec["source_family"],
        "source_scope": spec["source_scope"],
        "translation_status": spec["translation_status"],
        "translation_of": spec.get("translation_of"),
        "source_dataset": spec["dataset"],
        "official_mteb_revision": spec["revision"],
        "mirror": spec["mirror"],
        "mirror_revision": spec["mirror_revision"],
        "annotation_source": "source",
        "domain": spec["domain"],
        "language": spec["language"],
    }

    def file_for(relative: str) -> Path:
        return upstream / relative

    is_parquet = str(spec["source_kind"]).endswith("parquet")
    if is_parquet:
        docs = _iter_parquet(file_for(spec["source_files"][0]))
        queries = _iter_parquet(file_for(spec["source_files"][1]))
        qrels = _iter_parquet(file_for(spec["source_files"][2]))
    else:
        docs = _read_jsonl(file_for("corpus.jsonl"))
        queries = _read_jsonl(file_for("queries.jsonl"))
        qrels = _read_jsonl(file_for("qrels/test.jsonl"))

    document_rows: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    text_by_document: dict[str, str] = {}
    empty_document_ids: list[str] = []
    for index, row in enumerate(docs, 1):
        identifier = _id(row, where=f"{spec['id']}:documents[{index}]")
        if identifier in document_ids:
            raise ValueError(f"duplicate document ID in {spec['id']}: {identifier}")
        try:
            text = _text(row, where=f"{spec['id']}:documents[{index}]")
        except ValueError:
            # Preserve source rows whose body is explicitly empty. The raw
            # snapshot remains authoritative; generic quality checks may
            # exclude an unjudged empty document, while a positive reference
            # is kept pending rather than silently dropped.
            body_value = next((row.get(key) for key in ("text", "content", "body", "document", "passage") if key in row), None)
            if body_value is None or (isinstance(body_value, str) and not body_value.strip()):
                text = "" if body_value is None else body_value
                empty_document_ids.append(identifier)
            else:
                raise
        title = row.get("title", "")
        if title is None:
            title = ""
        if not isinstance(title, str):
            title = str(title)
        metadata = dict(source_meta)
        metadata["source_document_id"] = identifier
        if identifier in empty_document_ids:
            metadata["empty_body_source_row"] = True
            metadata["empty_body_policy"] = "retain raw row; positive references remain pending review"
        metadata.update(_safe_extra(row, {"_id", "id", "doc_id", "document_id", "corpus_id", "corpus-id", "title", "text", "content", "body", "document", "passage", "language"}))
        document_rows.append({"_id": identifier, "title": title, "text": text, "language": spec["language"], "metadata": metadata})
        document_ids.add(identifier)
        text_by_document[identifier] = text

    raw_qrel_rows: list[dict[str, Any]] = []
    positive_qrel_rows: list[dict[str, Any]] = []
    qrel_pairs: set[tuple[str, str]] = set()
    qrel_scores: Counter[str] = Counter()
    qrel_query_ids: set[str] = set()
    for index, row in enumerate(qrels, 1):
        query_id, document_id = _qrel_ids(row, where=f"{spec['id']}:qrels[{index}]")
        score = _score(row, where=f"{spec['id']}:qrels[{index}]")
        if (query_id, document_id) in qrel_pairs:
            raise ValueError(f"duplicate qrel pair in {spec['id']}: {query_id}/{document_id}")
        qrel_pairs.add((query_id, document_id))
        qrel_scores[str(score)] += 1
        qrel_query_ids.add(query_id)
        raw_qrel_rows.append({"query-id": query_id, "corpus-id": document_id, "score": row.get("score", row.get("relevance", score))})
        if score > 0:
            positive_qrel_rows.append({"query-id": query_id, "corpus-id": document_id, "score": row.get("score", row.get("relevance", score))})
        if document_id not in document_ids:
            raise ValueError(f"qrel references unknown document in {spec['id']}: {document_id}")

    empty_positive_qrels = [
        row for row in positive_qrel_rows if str(row["corpus-id"]) in set(empty_document_ids)
    ]
    empty_audit = {
        "schema_version": "dci-r4-empty-document-audit-v1",
        "source": spec["id"],
        "policy": "empty source bodies are retained in raw input; unjudged rows may be quality-excluded, positive references stay pending",
        "empty_document_ids": sorted(empty_document_ids),
        "empty_document_count": len(empty_document_ids),
        "positive_qrel_references": empty_positive_qrels,
        "positive_reference_count": len(empty_positive_qrels),
    }
    empty_audit_path = inputs / "empty_document_audit.json"
    _write_once(empty_audit_path, empty_audit)
    if empty_positive_qrels:
        raise RuntimeError(
            f"{spec['id']} has positive qrels for empty source documents; pending review: "
            f"{empty_audit_path}"
        )

    positive_qrels_by_query: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positive_qrel_rows:
        positive_qrels_by_query[str(row["query-id"])].append(row)
    query_rows: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    raw_query_ids: set[str] = set()
    for index, row in enumerate(queries, 1):
        identifier = _id(row, where=f"{spec['id']}:queries[{index}]", query=True)
        raw_query_ids.add(identifier)
        if identifier not in positive_qrels_by_query:
            continue
        if identifier in query_ids:
            raise ValueError(f"duplicate selected query ID in {spec['id']}: {identifier}")
        text = _text(row, where=f"{spec['id']}:queries[{index}]", query=True)
        metadata = dict(source_meta)
        metadata["source_query_id"] = identifier
        metadata.update(_safe_extra(row, {"_id", "id", "query_id", "query-id", "qid", "text", "question", "query", "language"}))
        query_rows.append({"_id": identifier, "text": text, "language": spec["language"], "metadata": metadata})
        query_ids.add(identifier)
    missing_queries = sorted(set(positive_qrels_by_query) - query_ids)
    if missing_queries:
        raise ValueError(f"positive qrels reference queries absent from source in {spec['id']}: {missing_queries[:5]}")

    def write_jsonl(name: str, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        payload = b"".join((json.dumps(row, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n").encode("utf-8") for row in rows)
        path = canonical / name
        digest = _write_once(path, payload)
        return {"path": str(path), "bytes": len(payload), "sha256": digest, "rows": len(rows)}

    canonical_files = {
        "documents": write_jsonl("documents.jsonl", document_rows),
        "queries": write_jsonl("queries.jsonl", query_rows),
        "qrels": write_jsonl("qrels.jsonl", positive_qrel_rows),
    }

    raw_qrels_path = inputs / "raw_test_qrels.jsonl"
    raw_qrels_info = {
        "rows": len(raw_qrel_rows),
        "positive_rows": len(positive_qrel_rows),
        "non_positive_rows": len(raw_qrel_rows) - len(positive_qrel_rows),
        "scores": dict(qrel_scores),
    }
    raw_qrels_info["sha256"] = _write_once(raw_qrels_path, b"".join((json.dumps(row, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n").encode("utf-8") for row in raw_qrel_rows))
    raw_qrels_info["path"] = str(raw_qrels_path)

    text_groups: defaultdict[str, list[str]] = defaultdict(list)
    for identifier, text in text_by_document.items():
        text_groups[hashlib.sha256(text.encode("utf-8")).hexdigest()].append(identifier)
    full_duplicates = [
        {"text_sha256": digest, "document_ids": sorted(ids), "count": len(ids)}
        for digest, ids in sorted(text_groups.items()) if len(ids) > 1
    ]
    equivalent_unlabelled: list[dict[str, Any]] = []
    qrels_by_query: defaultdict[str, set[str]] = defaultdict(set)
    for row in positive_qrel_rows:
        qrels_by_query[str(row["query-id"])].add(str(row["corpus-id"]))
    for query_id, ids in sorted(qrels_by_query.items()):
        for document_id in sorted(ids):
            digest = hashlib.sha256(text_by_document[document_id].encode("utf-8")).hexdigest()
            aliases = sorted(candidate for group in full_duplicates if group["text_sha256"] == digest for candidate in group["document_ids"] if candidate not in ids)
            if aliases:
                equivalent_unlabelled.append({"query_id": query_id, "positive_document_id": document_id, "equivalent_unlabelled_document_ids": aliases})
    alias_audit = {
        "schema_version": "dci-r4-alias-audit-v1",
        "source": spec["id"],
        "policy": "drop_near_duplicates=false; exact duplicate bodies retained; equivalent positive aliases are explicitly surfaced",
        "full_duplicate_groups": full_duplicates,
        "equivalent_unlabelled_documents": equivalent_unlabelled,
        "quality_blocked": bool(equivalent_unlabelled),
        "document_count": len(document_rows),
        "query_count_raw": len(raw_query_ids),
        "query_count_release": len(query_rows),
        "raw_qrel_rows": raw_qrels_info,
    }
    alias_path = inputs / "alias_audit.json"
    _write_once(alias_path, alias_audit)

    compare = {
        "schema_version": "dci-r4-raw-release-comparison-v1",
        "source": spec["id"],
        "raw_source_files": [str(path) for path in sorted(upstream.rglob("*")) if path.is_file()],
        "release_input": canonical_files,
        "raw_counts": {
            "documents": len(document_rows),
            "queries": len(raw_query_ids),
            "test_qrel_rows": len(raw_qrel_rows),
            "positive_test_qrel_rows": len(positive_qrel_rows),
            "non_positive_test_qrel_rows": len(raw_qrel_rows) - len(positive_qrel_rows),
        },
        "release_input_counts": {
            "documents": len(document_rows),
            "queries": len(query_rows),
            "positive_qrel_rows": len(positive_qrel_rows),
        },
        "raw_ids": {
            "documents_sha256": _sha256_json(sorted(document_ids)),
            "queries_sha256": _sha256_json(sorted(raw_query_ids)),
            "test_qrel_rows_sha256": _sha256_json(raw_qrel_rows),
        },
        "release_ids": {
            "documents_sha256": _sha256_json(sorted(row["_id"] for row in document_rows)),
            "queries_sha256": _sha256_json(sorted(row["_id"] for row in query_rows)),
            "positive_qrel_rows_sha256": _sha256_json(positive_qrel_rows),
        },
        "exclusions": {
            "queries_without_positive_test_qrels": sorted(set(raw_query_ids) - query_ids),
            "non_positive_qrel_rows_retained_at": str(raw_qrels_path),
        },
        "text_digest": _sha256_json(sorted((row["_id"], row["title"], row["text"]) for row in document_rows)),
    }
    compare_path = inputs / "raw_release_compare.json"
    _write_once(compare_path, compare)
    return {
        "canonical": canonical,
        "canonical_files": canonical_files,
        "documents": document_rows,
        "queries": query_rows,
        "positive_qrels": positive_qrel_rows,
        "raw_qrels": raw_qrel_rows,
        "raw_qrels_info": raw_qrels_info,
        "alias_audit": alias_audit,
        "alias_path": alias_path,
        "empty_audit_path": empty_audit_path,
        "empty_document_ids": sorted(empty_document_ids),
        "compare": compare,
        "compare_path": compare_path,
    }


def _pipeline_spec(spec: Mapping[str, Any], canonical: Path) -> dict[str, Any]:
    return {
        "id": spec["id"],
        "connector": "generic_jsonl",
        "input_kind": "retrieval",
        "domain": spec["domain"],
        "language": spec["language"],
        "genre": spec["genre"],
        "revision": spec["revision"],
        "license": spec["license"],
        "options": {
            "dataset": spec["dataset"],
            "split": "test",
            "files": {
                "documents": {"path": str(canonical / "documents.jsonl"), "name": "documents.jsonl"},
                "queries": {"path": str(canonical / "queries.jsonl"), "name": "queries.jsonl"},
                "qrels": {"path": str(canonical / "qrels.jsonl"), "name": "qrels.jsonl"},
            },
        },
        "source_family": spec["source_family"],
        "source_scope": spec["source_scope"],
        "translation_status": spec["translation_status"],
        "translation_of": spec.get("translation_of"),
        "official_card": spec["official_card"],
        "official_task": spec["official_task"],
        "mirror": spec["mirror"],
        "mirror_revision": spec["mirror_revision"],
    }


def _build_release(spec: Mapping[str, Any], config: Mapping[str, Any], acquired: Mapping[str, Any], canonical: Mapping[str, Any]) -> dict[str, Any]:
    eval_root = Path(config["eval_root"])
    output = eval_root / "pipeline_v1" / str(spec["id"]) / "v1"
    pipeline_spec = _pipeline_spec(spec, Path(canonical["canonical"]))
    if output.exists():
        manifest = verify_release(output)
        bundle = load_bundle(output)
        return {"release": str(output), "manifest": manifest, "bundle": bundle, "reused": True, "specification": pipeline_spec}
    result = build_source(
        pipeline_spec,
        Path(config["raw_root"]),
        output,
        policy=config["quality_policy"],
    )
    manifest = verify_release(output)
    bundle = load_bundle(output)
    return {"release": str(output), "manifest": manifest, "bundle": bundle, "reused": False, "build": result, "specification": pipeline_spec}


def _release_counts(bundle: Mapping[str, Any]) -> dict[str, int]:
    qrels = bundle.get("qrels", {})
    return {
        "documents": len(bundle.get("documents", [])),
        "queries": len(bundle.get("queries", [])),
        "positive_pairs": sum(len(row) for row in qrels.values() if isinstance(row, Mapping)),
        "qrel_queries": len(qrels),
    }


def _research_entry(spec: Mapping[str, Any], canonical: Mapping[str, Any], release: Mapping[str, Any]) -> dict[str, Any]:
    manifest = release["manifest"]
    counts = _release_counts(release["bundle"])
    return {
        "id": spec["id"],
        "source_id": spec["id"],
        "domain": spec["domain"],
        "language": spec["language"],
        "D": counts["documents"],
        "Q": counts["queries"],
        "pairs": counts["positive_pairs"],
        "documents": counts["documents"],
        "queries": counts["queries"],
        "positive_pairs": counts["positive_pairs"],
        "state": manifest["state"],
        "release_state": manifest["state"],
        "path": str(release["release"]),
        "release_path": str(release["release"]),
        "sourceURL": spec["official_card"],
        "source_url": spec["official_card"],
        "official_task": spec["official_task"],
        "official_origin": spec["official_origin"],
        "revision": spec["revision"],
        "official_mteb_revision": spec["revision"],
        "license": spec["license"],
        "translation": {
            "status": spec["translation_status"],
            "translation_of": spec.get("translation_of"),
            "source_family": spec["source_family"],
        },
        "translation_status": spec["translation_status"],
        "source_family": spec["source_family"],
        "source_scope": spec["source_scope"],
        "mirror": spec["mirror"],
        "mirror_revision": spec["mirror_revision"],
        "raw_directory": str(canonical["canonical"].parents[1]),
        "raw_manifest": str(Path(canonical["canonical"]).parents[1] / "source_manifest.json"),
        "raw_release_comparison": str(canonical["compare_path"]),
        "alias_audit": str(canonical["alias_path"]),
        "empty_document_audit": str(canonical["empty_audit_path"]),
        "empty_document_ids": canonical["empty_document_ids"],
        "completion_sha256": file_digest(Path(release["release"]) / "_COMPLETE.json")[0],
        "bundle_hash": manifest["bundle_hash"],
        "quality": {
            "max_positive_per_query": None,
            "drop_near_duplicates": False,
            "exclude_review_required": False,
            "alias_quality_blocked": bool(canonical["alias_audit"]["quality_blocked"]),
            "raw_qrel_rows": canonical["raw_qrels_info"]["rows"],
            "raw_non_positive_qrel_rows": canonical["raw_qrels_info"]["non_positive_rows"],
        },
    }


def _load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if not isinstance(config, Mapping) or not isinstance(config.get("sources"), list):
        raise ValueError("config must contain sources")
    ids: set[str] = set()
    for source in config["sources"]:
        if not isinstance(source, Mapping) or not isinstance(source.get("id"), str):
            raise ValueError("every source needs an id")
        if source["id"] in ids:
            raise ValueError(f"duplicate source ID: {source['id']}")
        ids.add(source["id"])
        if not str(source["id"]).startswith("r4_domains_"):
            raise ValueError(f"source is outside the round-4 namespace: {source['id']}")
    return dict(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", nargs="*", help="explicit source IDs; default is all configured sources")
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    sources = {str(item["id"]): item for item in config["sources"]}
    selected = args.source or list(sources)
    unknown = sorted(set(selected) - set(sources))
    if unknown:
        raise ValueError(f"unknown source IDs: {unknown}")
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source_id in selected:
        spec = sources[source_id]
        try:
            acquired = _acquire_source(spec, config)
            canonical = _canonical_from_rows(spec, acquired)
            release = _build_release(spec, config, acquired, canonical)
            entry = _research_entry(spec, canonical, release)
            entries.append(entry)
            print(json.dumps({"source": source_id, "state": entry["state"], "D": entry["D"], "Q": entry["Q"], "pairs": entry["pairs"], "release": entry["release_path"]}, ensure_ascii=False, sort_keys=True), flush=True)
        except Exception as exc:
            failure = {"id": source_id, "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "time": _now()}
            failures.append(failure)
            print(json.dumps(failure, ensure_ascii=False, sort_keys=True), flush=True)

    research_root = Path(config["research_root"])
    research_root.mkdir(parents=True, exist_ok=True)
    # A targeted retry must update only the selected source IDs while
    # retaining receipts for earlier sources in the same research directory.
    previous_entries: list[dict[str, Any]] = []
    previous_failures: list[dict[str, Any]] = []
    previous_path = research_root / "research.json"
    if previous_path.exists():
        previous = read_json(previous_path)
        if isinstance(previous, Mapping):
            previous_entries = [dict(row) for row in previous.get("sources", []) if isinstance(row, Mapping)]
            previous_failures = [dict(row) for row in previous.get("failures", []) if isinstance(row, Mapping)]
    selected_ids = set(selected)
    merged_entries = [row for row in previous_entries if str(row.get("id")) not in selected_ids] + entries
    merged_failures = [row for row in previous_failures if str(row.get("id")) not in selected_ids] + failures
    payload = {
        "schema_version": RESEARCH_SCHEMA,
        "created_at": _now(),
        "config": str(args.config),
        "python": PYTHON,
        "quality_policy": config["quality_policy"],
        "acquired_verified": [entry["id"] for entry in merged_entries if entry["state"] in {"labels_ready", "review_required", "corpus_ready"}],
        "ready": [entry["id"] for entry in merged_entries if entry["state"] == "labels_ready"],
        "sources": merged_entries,
        "failures": merged_failures,
        "candidates_not_downloaded": [],
        "validation_boundary": "sealed data release and source audit; no GPU, ranking, candidate pools, or model evaluation",
    }
    _write_research_snapshot(research_root / "research.json", payload)
    return 0 if not failures else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"domains-round4: {exc}", file=sys.stderr)
        raise SystemExit(2)
