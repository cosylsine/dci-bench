"""Immutable document expansion artifacts using the existing DCI contract.

Raw downloads are never edited. Unsupported documents and every query depending
on one are listed explicitly, without pruning individual positive judgments.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any
from urllib.parse import unquote
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from dci_bench.data.loader import stable_json_sha256
from dci_bench.data.registry import TaskSpec, validate_task_manifest
from dci_bench.data.document_ids import (
    DOCUMENT_ID_POLICY,
    build_document_id_map,
    remap_qrels,
    restore_ranking,
    reverse_document_id_map,
    validate_source_document_ids,
)
from dci_bench.data.workspace_builder import (
    CORPUS_MAX_LINE_CHARS,
    WORKSPACE_FORMAT_VERSION,
    doc_id_to_filename,
    render_doc_markdown,
)
from dci_bench.protocol.contracts import (
    BENCHMARK_CONTRACT_VERSION, MAX_CORPUS_DOCUMENT_BYTES, TOP_K, protocol_contract,
)
from dci_bench.scoring.retrieval import score_query

EXPANSION_VERSION = "dci-dataset-expansion-20261014-v1"
LEGACY_METADATA_FILES = {"queries.jsonl", "qrels.json", "host_labels.json", "doc_id_files.json",
                         "audit.json", "source_manifest.json", "provenance.json", "manifest.json", "corpus_files.jsonl"}
METADATA_FILES = LEGACY_METADATA_FILES | {"source_doc_ids.json", "original_qrels.json"}
CANONICAL_FILES = {"corpus/corpus.parquet", "queries/queries.parquet", "data/test.parquet"}


def _validation_receipt(manifest: dict, *, verify_content_hashes: bool) -> dict:
    """Return a verification receipt without changing the frozen manifest.

    The receipt is intentionally attached only to the in-memory return value.
    Existing completion and source manifest hashes therefore remain valid while
    callers can distinguish structural validation from byte-level validation.
    """
    result = dict(manifest)
    if verify_content_hashes:
        # Preserve the strict API's manifest return contract.  In particular,
        # creating an artifact does not claim an additional byte readback.
        return result
    verified = bool(verify_content_hashes)
    result["content_hashes_verified"] = verified
    result["corpus_bytes_verified"] = verified
    boundary = (
        "artifact content hashes recomputed and structural relationships validated"
        if verified else
        "artifact structure, sizes, IDs, qrels and JSONL parsed; stored digests declared only"
    )
    result["verification_boundary"] = boundary
    result["validation_receipt"] = {"content_hashes_verified": verified, "boundary": boundary}
    result.setdefault("validation_boundary", boundary)
    return result


def _is_sha256_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _require_directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"Missing {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be an independent directory: {path}")


def _require_regular_file(
    path: Path,
    label: str,
    *,
    expected_size: int | None = None,
) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"Missing {label}: {path}") from exc
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1):
        raise ValueError(f"{label} must be an independent regular file: {path}")
    if expected_size is not None and info.st_size != expected_size:
        raise ValueError(f"{label} size mismatch: {path}")
    return info


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def verify_hfd_source(raw: Path, dataset: str, revision: str) -> dict:
    """Check every selected download against HF size and LFS/Git object digest."""
    metadata = json.loads((raw / ".hfd/repo_metadata.json").read_text())
    if metadata.get("id") != dataset or metadata.get("sha") != revision:
        raise ValueError(f"Raw source identity mismatch: {raw}")
    siblings = {row["rfilename"]: row for row in metadata["siblings"]}
    files = {}
    for line in (raw / ".hfd/manifest").read_text().splitlines():
        size, relative = line.split("\t", 1)
        path = raw / relative
        if relative not in siblings or not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing/unknown selected raw file: {path}")
        if path.with_name(path.name + ".aria2").exists():
            raise ValueError(f"Incomplete aria2 download: {path}")
        if path.stat().st_size != int(size):
            raise ValueError(f"Raw size mismatch: {path}")
        row = siblings[relative]
        digest = sha256_file(path)
        lfs = row.get("lfs", {})
        upstream_hash = lfs.get("sha256") or lfs.get("oid")
        if upstream_hash:
            if digest != upstream_hash.removeprefix("sha256:"):
                raise ValueError(f"Raw LFS SHA256 mismatch: {path}")
        elif row.get("blobId"):
            git_hash = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    git_hash.update(chunk)
            if git_hash.hexdigest() != row["blobId"]:
                raise ValueError(f"Raw Git blob mismatch: {path}")
        else:
            raise ValueError(f"No upstream integrity hash for {path}")
        files[relative] = {"bytes": int(size), "sha256": digest}
    if not files:
        raise ValueError("Empty selected source manifest")
    return {"dataset": dataset, "revision": revision, "raw_directory": str(raw),
            "files": files, "source_fingerprint": stable_json_sha256(files)}


def _write_doc(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)


def _parquet(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def build_document_task(task: dict, suite: Path, source: dict, *, io_workers: int = 16) -> dict:
    """Publish one full DCI workspace, metadata and portable parquet task.

The compatibility policy excludes only whole oversized/empty source documents.
Queries with ANY excluded positive are removed in their entirety. Original
document IDs and qrels remain in host-only source_doc_ids.json and
original_qrels.json alongside raw sources and host_labels.json.
"""
    name = task["name"]
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Unsafe task name")
    workspace = suite / "workspaces" / name
    metadata = suite / "metadata" / name
    canonical = suite / "canonical" / name
    input_task_hash = stable_json_sha256({k: v for k, v in task.items() if k != "corpus"})
    builder_hash = sha256_file(Path(__file__))
    if any(path.exists() for path in (workspace, metadata, canonical)):
        if (metadata / "_COMPLETE.json").is_file():
            result = verify_document_task(suite, name)
            if result["source_fingerprint"] != source["source_fingerprint"]:
                raise ValueError("Existing task raw fingerprint differs")
            if result["revision"] != task["revision"] or result["dataset"] != task["dataset"]:
                raise ValueError("Existing task source identity differs")
            if result.get("document_id_policy") != DOCUMENT_ID_POLICY:
                raise ValueError(
                    "Existing task uses a legacy document ID policy; use a fresh destination"
                )
            if result.get("input_task_hash") != input_task_hash:
                raise ValueError("Existing task adapter inputs or builder differ")
            if result.get("builder_sha256") != builder_hash:
                raise ValueError("Existing task adapter inputs or builder differ")
            return result
        raise FileExistsError(f"Refusing to overwrite existing/incomplete task: {name}")
    staging = suite / ".staging" / f"{name}-{uuid.uuid4().hex}"
    ws, md, canon = (staging / sub for sub in ("workspace", "metadata", "canonical"))
    (ws / "corpus").mkdir(parents=True)
    md.mkdir()
    (canon / "corpus").mkdir(parents=True)
    corpus_schema = pa.schema([(key, pa.string()) for key in ("_id", "title", "text")])
    seen: set[str] = set()
    raw_doc_count = 0
    # The source iterator may be one-shot and the corpus may be very large.
    # Stage complete rows on disk, retain only IDs in memory, then assign aliases
    # from the full inventory before rendering any model-visible files.
    spool = staging / "corpus_source.jsonl"
    corpus_hash = hashlib.sha256()
    with spool.open("x", encoding="utf-8") as source_rows:
        for row in task["corpus"]:
            raw_doc_count += 1
            if set(row) != {"_id", "title", "text"} or any(not isinstance(v, str) for v in row.values()):
                raise ValueError("Corpus rows must contain exactly string _id/title/text")
            doc_id = row["_id"]
            if not doc_id or doc_id in seen:
                raise ValueError(f"Empty/duplicate corpus ID: {doc_id!r}")
            seen.add(doc_id)
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            corpus_hash.update((encoded + "\n").encode("utf-8"))
            source_rows.write(encoded + "\n")
    if not seen:
        raise ValueError("No source corpus documents")

    original_to_alias = build_document_id_map(seen)
    source_doc_ids = reverse_document_id_map(original_to_alias)
    filenames, doc_id_files = set(), {}
    doc_exclusions, sizes, buffer = [], [], []
    writer = pq.ParquetWriter(canon / "corpus/corpus.parquet", corpus_schema, compression="zstd")
    pending = deque()
    try:
        with ThreadPoolExecutor(max_workers=io_workers) as pool, (md / "corpus_files.jsonl").open("x", encoding="utf-8") as hashes:
            with spool.open(encoding="utf-8") as source_rows:
                for line in source_rows:
                    row = json.loads(line)
                    original_id = row["_id"]
                    alias = original_to_alias[original_id]
                    filename = doc_id_to_filename(alias)
                    if filename in filenames:
                        raise ValueError(f"Filename collision: {filename}")
                    filenames.add(filename)
                    source_bytes = len(row["text"].encode("utf-8"))
                    if source_bytes > MAX_CORPUS_DOCUMENT_BYTES:
                        doc_exclusions.append({"document_id": original_id, "reason": "document_exceeds_1MB",
                                               "source_text_bytes": source_bytes})
                        continue
                    data = render_doc_markdown(row["title"], row["text"]).encode("utf-8")
                    reason = ("document_exceeds_1MB" if len(data) > MAX_CORPUS_DOCUMENT_BYTES else
                              "empty_document" if not data.strip() else None)
                    if reason:
                        doc_exclusions.append({"document_id": original_id, "reason": reason, "rendered_bytes": len(data)})
                        continue
                    doc_id_files[alias] = filename
                    sizes.append(len(data))
                    pending.append(pool.submit(_write_doc, ws / "corpus" / filename, data))
                    if len(pending) >= io_workers * 4:
                        pending.popleft().result()
                    hashes.write(json.dumps({"doc_id": alias, "source_doc_id": original_id, "filename": filename,
                                             "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
                                            ensure_ascii=False, sort_keys=True) + "\n")
                    buffer.append({"_id": alias, "title": row["title"], "text": row["text"]})
                    if len(buffer) >= 1024:
                        writer.write_table(pa.Table.from_pylist(buffer, schema=corpus_schema))
                        buffer.clear()
            for future in pending:
                future.result()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=corpus_schema))
    finally:
        writer.close()
    spool.unlink()
    if not doc_id_files:
        raise ValueError("No compatible corpus documents")
    queries, query_exclusions, query_ids = [], [], set()
    original_qrels: dict[str, dict[str, float]] = {}
    for query in task["queries"]:
        qid = query["_id"]
        if not isinstance(qid, str) or not qid or qid in query_ids:
            raise ValueError("Empty/duplicate query ID")
        query_ids.add(qid)
        if not isinstance(query.get("text"), str) or not query["text"].strip():
            raise ValueError(f"Invalid query text: {qid}")
        gold = task["qrels"].get(qid)
        if not isinstance(gold, dict) or not gold or any(not isinstance(d, str) or not math.isfinite(s) or s <= 0 for d, s in gold.items()):
            raise ValueError(f"Invalid gold labels for {qid}")
        unknown = sorted(set(gold) - seen)
        if unknown:
            raise ValueError(f"Unknown source gold IDs: {qid}: {unknown[:5]}")
        original_qrels[qid] = dict(sorted(gold.items()))
        excluded = sorted(d for d in gold if original_to_alias[d] not in doc_id_files)
        if excluded:
            query_exclusions.append({"query_id": qid, "reason": "positive_document_outside_compatible_corpus", "document_ids": excluded})
            continue
        queries.append(query)
    if set(task["qrels"]) != query_ids:
        raise ValueError("Query/qrels key sets differ")
    if not queries:
        raise ValueError("No compatible queries")
    # Validate the complete source-to-alias mapping once, then remap all qrels
    # in one pass.  Calling remap_qrels for each query repeats the O(D)
    # canonical-map validation and turns large corpora into O(Q*D) work.
    remapped_qrels = remap_qrels(original_qrels, original_to_alias)
    qrels = {
        query["_id"]: dict(sorted(remapped_qrels[query["_id"]].items()))
        for query in queries
    }
    queries.sort(key=lambda r: r["_id"])
    qrels = dict(sorted(qrels.items()))
    original_qrels = dict(sorted(original_qrels.items()))
    counts = [len(value) for value in qrels.values()]
    audit = {"raw_num_docs": raw_doc_count, "num_docs": len(doc_id_files), "num_queries": len(queries),
             "num_qrels": sum(counts), "positive_count_histogram": dict(sorted(Counter(counts).items())),
             "queries_over_top_k": sum(n > TOP_K for n in counts),
             "mean_recall_ceiling_at_10": sum(min(TOP_K, n) / n for n in counts) / len(counts),
             "mean_f1_ceiling_at_10": sum(2 * min(TOP_K, n) / (n + min(TOP_K, n)) for n in counts) / len(counts),
             "max_document_bytes": max(sizes), "corpus_bytes": sum(sizes),
             "document_exclusions": doc_exclusions, "query_exclusions": query_exclusions,
             "adapter_exclusions": task["provenance"].get("exclusions", []),
             "unknown_qrel_documents": 0,
             "compatibility_policy": "exclude whole incompatible documents and every query touching them; never prune gold labels"}
    # Scorer sanity checks use synthetic perfect rankings, never a model result.
    for qid, gold in qrels.items():
        ranking = sorted(gold, key=lambda d: (-gold[d], d))[:TOP_K]
        score = score_query(ranking, gold, metric_ks=[TOP_K])
        if not score["valid_output"] or abs(score["recall_at_10"] - min(TOP_K, len(gold)) / len(gold)) > 1e-12:
            raise ValueError(f"Scorer oracle mismatch for {qid}")
    audit["scorer_oracle_checked_queries"] = len(queries)
    write_jsonl(md / "queries.jsonl", queries)
    write_json(md / "qrels.json", qrels)
    write_json(md / "host_labels.json", task["host_labels"])
    write_json(md / "doc_id_files.json", dict(sorted(doc_id_files.items())))
    write_json(md / "source_doc_ids.json", dict(sorted(source_doc_ids.items())))
    write_json(md / "original_qrels.json", original_qrels)
    write_json(md / "audit.json", audit)
    write_json(md / "source_manifest.json", source)
    write_json(md / "provenance.json", task["provenance"])
    _parquet(canon / "queries/queries.parquet", queries, pa.schema([("_id", pa.string()), ("text", pa.string())]))
    _parquet(canon / "data/test.parquet", [{"query-id": qid, "corpus-id": d, "score": s} for qid, g in qrels.items() for d, s in g.items()],
             pa.schema([("query-id", pa.string()), ("corpus-id", pa.string()), ("score", pa.float64())]))
    instruction = task.get("task_instruction", "")
    manifest = {"expansion_version": EXPANSION_VERSION, "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
                "task": name, "dataset": task["dataset"], "revision": task["revision"],
                "resolved_revision": task.get("resolved_revision", task["revision"]),
                "evaluation_track": "document", "task_instruction": instruction, "top_k": TOP_K,
                "document_id_policy": DOCUMENT_ID_POLICY,
                **{k: audit[k] for k in ("num_docs", "num_queries", "num_qrels")},
                "source_num_docs": len(source_doc_ids),
                "corpus_hash": corpus_hash.hexdigest(), "corpus_hash_format": "source-order canonical JSONL _id/title/text",
                "queries_hash": stable_json_sha256(queries), "qrels_hash": stable_json_sha256(qrels),
                "source_doc_ids_hash": stable_json_sha256(source_doc_ids),
                "original_qrels_hash": stable_json_sha256(original_qrels),
                "source_fingerprint": source["source_fingerprint"], "source_dir": source["raw_directory"],
                "input_task_hash": input_task_hash, "builder_sha256": builder_hash,
                "workspace_dir": str(workspace), "metadata_dir": str(metadata),
                "workspace_format": {"version": WORKSPACE_FORMAT_VERSION, "max_line_chars": CORPUS_MAX_LINE_CHARS,
                                     "document_id_policy": DOCUMENT_ID_POLICY},
                "protocol": protocol_contract(task_name=name, task_instruction=instruction)}
    write_json(md / "manifest.json", manifest)
    (ws / "README.md").write_text(
        f"# DCI-Bench corpus\n\nTask: {name}\n\n"
        "Corpus files use deterministic task-local short document aliases plus `.md`.\n",
        encoding="utf-8",
    )
    completion = {"schema_version": EXPANSION_VERSION, "readme_sha256": sha256_file(ws / "README.md"),
                  "metadata_hashes": {p.name: sha256_file(p) for p in sorted(md.iterdir())},
                  "canonical_hashes": {str(p.relative_to(canon)): sha256_file(p) for p in sorted(canon.rglob("*.parquet"))}}
    write_json(md / "_COMPLETE.json", completion)
    for destination in (workspace, metadata, canonical):
        destination.parent.mkdir(parents=True, exist_ok=True)
    # A partial publish has no suite completion and cannot be silently reused.
    ws.rename(workspace)
    canon.rename(canonical)
    md.rename(metadata)
    staging.rmdir()
    return _validation_receipt(manifest, verify_content_hashes=True)


def _verify_corpus_file(path: Path, row: dict, *, verify_content_hashes: bool = True) -> None:
    info = _require_regular_file(path, "Corpus file", expected_size=row["bytes"])
    if info.st_size > MAX_CORPUS_DOCUMENT_BYTES:
        raise ValueError(f"Invalid corpus file: {path}")
    if verify_content_hashes and sha256_file(path) != row["sha256"]:
        raise ValueError(f"Changed corpus bytes: {path}")


def _validate_qrels(qrels: Any, query_ids: set[str], document_ids: set[str], *, label: str) -> None:
    if not isinstance(qrels, Mapping):
        raise ValueError(f"{label} must be an object")
    if any(not isinstance(query_id, str) or not query_id for query_id in qrels):
        raise ValueError(f"{label} contains an invalid query ID")
    for query_id, gold in qrels.items():
        if query_id not in query_ids:
            raise ValueError(f"{label} contains an unknown query ID")
        if not isinstance(gold, Mapping) or not gold:
            raise ValueError(f"{label} for {query_id} must be a nonempty object")
        for document_id, score in gold.items():
            if (not isinstance(document_id, str) or document_id not in document_ids
                    or isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(float(score)) or float(score) <= 0):
                raise ValueError(f"{label} contains an invalid judgment for {query_id}")


def _parse_json_lines(path: Path, label: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{label} rows must be objects")
            rows.append(dict(row))
    return rows


def verify_document_task(
    suite: Path,
    name: str,
    *,
    io_workers: int = 32,
    verify_content_hashes: bool = True,
) -> dict:
    if type(io_workers) is not int or not 1 <= io_workers <= 256:
        raise ValueError("io_workers must be an integer from 1 through 256")
    if type(verify_content_hashes) is not bool:
        raise ValueError("verify_content_hashes must be a boolean")
    suite = Path(suite)
    md, ws, canon = suite / "metadata" / name, suite / "workspaces" / name, suite / "canonical" / name
    _require_directory(suite, "Expansion suite")
    _require_directory(md, "Task metadata directory")
    _require_directory(ws, "Task workspace directory")
    _require_directory(canon, "Task canonical directory")
    complete_path = md / "_COMPLETE.json"
    _require_regular_file(complete_path, "Task completion marker")
    completion = json.loads(complete_path.read_text(encoding="utf-8"))
    if not isinstance(completion, Mapping):
        raise ValueError("Incomplete integrity manifest")
    metadata_hashes = completion.get("metadata_hashes")
    canonical_hashes = completion.get("canonical_hashes")
    if not isinstance(metadata_hashes, Mapping) or not isinstance(canonical_hashes, Mapping):
        raise ValueError("Incomplete integrity manifest")
    metadata_names = set(metadata_hashes)
    legacy = metadata_names == LEGACY_METADATA_FILES
    current = metadata_names == METADATA_FILES
    if (completion.get("schema_version") != EXPANSION_VERSION
            or not (legacy or current)
            or set(canonical_hashes) != CANONICAL_FILES
            or not _is_sha256_digest(completion.get("readme_sha256"))):
        raise ValueError("Incomplete integrity manifest")
    for filename, digest in [*metadata_hashes.items(), *canonical_hashes.items()]:
        if not isinstance(filename, str) or not _is_sha256_digest(digest):
            raise ValueError("Incomplete integrity manifest")

    readme = ws / "README.md"
    _require_regular_file(readme, "Workspace README")
    if verify_content_hashes and sha256_file(readme) != completion["readme_sha256"]:
        raise ValueError("Changed workspace README")

    actual_metadata_names = set()
    for path in md.iterdir():
        if path.name == "_COMPLETE.json":
            continue
        _require_regular_file(path, "Task metadata")
        actual_metadata_names.add(path.name)
    if actual_metadata_names != metadata_names:
        raise ValueError("Unexpected/missing metadata files")
    canonical_seen = set()
    for path in canon.rglob("*"):
        info = path.lstat()
        relative = str(path.relative_to(canon))
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Symlink not allowed in canonical data: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"Canonical data must be an independent regular file: {relative}")
        canonical_seen.add(relative)
    if canonical_seen != CANONICAL_FILES:
        raise ValueError("Unexpected/missing canonical files")
    if verify_content_hashes:
        for filename, expected_digest in metadata_hashes.items():
            if sha256_file(md / filename) != expected_digest:
                raise ValueError(f"Changed metadata: {name}/{filename}")
        for filename, expected_digest in canonical_hashes.items():
            if sha256_file(canon / filename) != expected_digest:
                raise ValueError(f"Changed canonical data: {name}/{filename}")

    # Parse all JSON sidecars in both modes.  No-hash verification still has a
    # physical parse boundary for metadata and JSONL, even when bytes are not
    # reread for digest comparison.
    for filename in metadata_names:
        if filename.endswith(".json"):
            json.loads((md / filename).read_text(encoding="utf-8"))
    manifest = json.loads((md / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("Task manifest must be an object")
    if (manifest.get("expansion_version") != EXPANSION_VERSION
            or manifest.get("evaluation_track") != "document"):
        raise ValueError("Unexpected document task manifest schema")
    for field in ("corpus_hash", "queries_hash", "qrels_hash", "input_task_hash", "builder_sha256"):
        if field in manifest and not _is_sha256_digest(manifest[field]):
            raise ValueError(f"Malformed task manifest digest: {field}")
    if not legacy:
        for field in ("source_doc_ids_hash", "original_qrels_hash"):
            if not _is_sha256_digest(manifest.get(field)):
                raise ValueError(f"Malformed task manifest digest: {field}")
    spec = TaskSpec(name, manifest["dataset"], manifest["revision"], name, manifest["task_instruction"])
    validate_task_manifest(spec, manifest)
    mapping = json.loads((md / "doc_id_files.json").read_text(encoding="utf-8"))
    qrels = json.loads((md / "qrels.json").read_text(encoding="utf-8"))
    if not isinstance(mapping, Mapping):
        raise ValueError("Document filename mapping must be an object")
    mapping = dict(mapping)
    if any(not isinstance(doc_id, str) or not doc_id or not isinstance(filename, str) or not filename
           for doc_id, filename in mapping.items()):
        raise ValueError("Document filename mapping contains invalid IDs")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Document filename mapping is not one-to-one")
    queries = _parse_json_lines(md / "queries.jsonl", "Query JSONL")
    query_ids = [query.get("_id") for query in queries]
    if (any(not isinstance(query_id, str) or not query_id for query_id in query_ids)
            or len(query_ids) != len(set(query_ids))):
        raise ValueError("Query IDs must be unique nonempty strings")
    if any(not isinstance(query.get("text"), str) or not query["text"].strip() for query in queries):
        raise ValueError("Query text must be nonempty")
    query_id_set = set(query_ids)
    _validate_qrels(qrels, query_id_set, set(mapping), label="Qrels")
    if (len(queries) != manifest["num_queries"] or len(mapping) != manifest["num_docs"]
            or sum(len(gold) for gold in qrels.values()) != manifest["num_qrels"]
            or set(qrels) != query_id_set):
        raise ValueError("Manifest counts or query/qrels coverage differ")
    if verify_content_hashes:
        if (stable_json_sha256(qrels) != manifest["qrels_hash"]
                or stable_json_sha256(queries) != manifest["queries_hash"]):
            raise ValueError("Query/qrels canonical hash mismatch")
    if legacy:
        # Legacy frozen artifacts use original IDs and percent-encoded original
        # filenames.  Their metadata set is intentionally accepted unchanged.
        if manifest.get("document_id_policy") is not None:
            raise ValueError("Legacy metadata unexpectedly declares short document IDs")
        source_doc_ids = None
    else:
        if manifest.get("document_id_policy") != DOCUMENT_ID_POLICY:
            raise ValueError("Current metadata must declare the short document ID policy")
        source_doc_ids = validate_source_document_ids(
            json.loads((md / "source_doc_ids.json").read_text(encoding="utf-8"))
        )
        if not set(mapping).issubset(source_doc_ids):
            raise ValueError("Document filename mapping contains an unknown short ID")
        original_qrels = json.loads((md / "original_qrels.json").read_text(encoding="utf-8"))
        if not isinstance(original_qrels, Mapping):
            raise ValueError("Original qrels must be an object")
        original_qrels = dict(original_qrels)
        source_to_alias = {original: alias for alias, original in source_doc_ids.items()}
        _validate_qrels(original_qrels, set(query_ids) | set(original_qrels), set(source_to_alias),
                        label="Original qrels")
        if verify_content_hashes:
            if stable_json_sha256(source_doc_ids) != manifest.get("source_doc_ids_hash"):
                raise ValueError("Source document mapping hash mismatch")
            if stable_json_sha256(original_qrels) != manifest.get("original_qrels_hash"):
                raise ValueError("Original qrels hash mismatch")
        if manifest.get("source_num_docs") != len(source_doc_ids):
            raise ValueError("Source document count mismatch")
        if not set(qrels).issubset(original_qrels):
            raise ValueError("Original qrels do not cover published qrels")
        remapped_original_qrels = remap_qrels(original_qrels, source_to_alias)
        for query_id, gold in qrels.items():
            # Every active query was admitted only when all of its positives
            # survived corpus compatibility filtering.  Compare the complete
            # remapped judgment set; silently dropping one positive would
            # change the benchmark contract.
            if remapped_original_qrels.get(query_id) != gold:
                raise ValueError("Published qrels do not match original qrels mapping")
        # Validate the complete alias inventory even if no qrel references a
        # particular source document.
        restore_ranking(list(source_doc_ids), source_doc_ids)
    checked = set()
    pending = deque()
    corpus_integrity = md / "corpus_files.jsonl"
    _require_regular_file(corpus_integrity, "Corpus integrity JSONL")
    with ThreadPoolExecutor(max_workers=io_workers) as pool, corpus_integrity.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("Corpus integrity rows must be objects")
            row = dict(row)
            filename = row.get("filename")
            bytes_count = row.get("bytes")
            if (not isinstance(filename, str) or not filename or Path(filename).name != filename
                    or isinstance(bytes_count, bool) or not isinstance(bytes_count, int)
                    or bytes_count < 0 or not _is_sha256_digest(row.get("sha256"))):
                raise ValueError("Malformed corpus integrity row")
            if legacy:
                document_id = row.get("doc_id")
                if (not isinstance(document_id, str) or not document_id
                        or mapping.get(document_id) != filename
                        or doc_id_to_filename(document_id) != filename
                        or unquote(filename[:-3]) != document_id):
                    raise ValueError("Non-reversible document filename")
            else:
                alias = row.get("doc_id")
                if (not isinstance(alias, str) or not alias
                        or mapping.get(alias) != filename
                        or alias not in source_doc_ids
                        or doc_id_to_filename(alias) != filename
                        or row.get("source_doc_id") != source_doc_ids[alias]):
                    raise ValueError("Non-reversible short document filename")
            if filename in checked:
                raise ValueError("Duplicate corpus integrity entry")
            checked.add(filename)
            pending.append(pool.submit(_verify_corpus_file, ws / "corpus" / filename, row,
                                       verify_content_hashes=verify_content_hashes))
            if len(pending) >= io_workers * 4:
                pending.popleft().result()
        for future in pending:
            future.result()
    corpus_dir = ws / "corpus"
    _require_directory(corpus_dir, "Workspace corpus directory")
    actual_workspace_files = {path.name for path in corpus_dir.iterdir()}
    if checked != set(mapping.values()) or checked != actual_workspace_files:
        raise ValueError("Unexpected/missing corpus files")
    if {p.name for p in ws.iterdir()} != {"corpus", "README.md"}:
        raise ValueError("Unexpected files in model-visible workspace")
    return _validation_receipt(manifest, verify_content_hashes=verify_content_hashes)
