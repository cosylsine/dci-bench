"""Immutable document expansion artifacts using the existing DCI contract.

Raw downloads are never edited. Unsupported documents and every query depending
on one are listed explicitly, without pruning individual positive judgments.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
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
from dci_bench.data.workspace_builder import doc_id_to_filename, render_doc_markdown
from dci_bench.protocol.contracts import (
    BENCHMARK_CONTRACT_VERSION, MAX_CORPUS_DOCUMENT_BYTES, TOP_K, protocol_contract,
)
from dci_bench.scoring.retrieval import score_query

EXPANSION_VERSION = "dci-dataset-expansion-20261014-v1"
METADATA_FILES = {"queries.jsonl", "qrels.json", "host_labels.json", "doc_id_files.json",
                  "audit.json", "source_manifest.json", "provenance.json", "manifest.json", "corpus_files.jsonl"}
CANONICAL_FILES = {"corpus/corpus.parquet", "queries/queries.parquet", "data/test.parquet"}


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

The compatibility policy excludes only whole oversized/unnamable/empty source
documents. Queries with ANY excluded positive are removed in their entirety.
All original data and annotations remain in raw sources and host_labels.json.
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
            if result.get("input_task_hash") != input_task_hash or result.get("builder_sha256") != builder_hash:
                raise ValueError("Existing task adapter inputs or builder differ")
            return result
        raise FileExistsError(f"Refusing to overwrite existing/incomplete task: {name}")
    staging = suite / ".staging" / f"{name}-{uuid.uuid4().hex}"
    ws, md, canon = (staging / sub for sub in ("workspace", "metadata", "canonical"))
    (ws / "corpus").mkdir(parents=True)
    md.mkdir()
    (canon / "corpus").mkdir(parents=True)
    corpus_schema = pa.schema([(key, pa.string()) for key in ("_id", "title", "text")])
    writer = pq.ParquetWriter(canon / "corpus/corpus.parquet", corpus_schema, compression="zstd")
    seen, filenames, doc_id_files = set(), set(), {}
    doc_exclusions, sizes, buffer = [], [], []
    raw_doc_count = 0
    # Hash the deterministic source-order JSONL stream without holding GBs of text.
    corpus_hash = hashlib.sha256()
    pending = deque()
    try:
        with ThreadPoolExecutor(max_workers=io_workers) as pool, (md / "corpus_files.jsonl").open("x", encoding="utf-8") as hashes:
            for row in task["corpus"]:
                raw_doc_count += 1
                if set(row) != {"_id", "title", "text"} or any(not isinstance(v, str) for v in row.values()):
                    raise ValueError("Corpus rows must contain exactly string _id/title/text")
                doc_id = row["_id"]
                if not doc_id or doc_id in seen:
                    raise ValueError(f"Empty/duplicate corpus ID: {doc_id!r}")
                seen.add(doc_id)
                filename = doc_id_to_filename(doc_id)
                if filename in filenames:
                    raise ValueError(f"Filename collision: {filename}")
                filenames.add(filename)
                source_bytes = len(row["text"].encode("utf-8"))
                if source_bytes > MAX_CORPUS_DOCUMENT_BYTES:
                    doc_exclusions.append({"document_id": doc_id, "reason": "document_exceeds_1MB",
                                           "source_text_bytes": source_bytes})
                    continue
                data = render_doc_markdown(row["title"], row["text"]).encode("utf-8")
                reason = ("filename_exceeds_255_bytes" if len(filename.encode()) > 255 else
                          "document_exceeds_1MB" if len(data) > MAX_CORPUS_DOCUMENT_BYTES else
                          "empty_document" if not data.strip() else None)
                if reason:
                    doc_exclusions.append({"document_id": doc_id, "reason": reason, "rendered_bytes": len(data)})
                    continue
                doc_id_files[doc_id] = filename
                sizes.append(len(data))
                pending.append(pool.submit(_write_doc, ws / "corpus" / filename, data))
                if len(pending) >= io_workers * 4:
                    pending.popleft().result()
                hashes.write(json.dumps({"doc_id": doc_id, "filename": filename,
                                         "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
                                        ensure_ascii=False, sort_keys=True) + "\n")
                corpus_hash.update((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
                buffer.append(row)
                if len(buffer) >= 1024:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=corpus_schema))
                    buffer.clear()
            for future in pending:
                future.result()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=corpus_schema))
    finally:
        writer.close()
    if not doc_id_files:
        raise ValueError("No compatible corpus documents")
    queries, qrels, query_exclusions, query_ids = [], {}, [], set()
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
        excluded = sorted(d for d in gold if d not in doc_id_files)
        if excluded:
            query_exclusions.append({"query_id": qid, "reason": "positive_document_outside_compatible_corpus", "document_ids": excluded})
            continue
        queries.append(query)
        qrels[qid] = dict(sorted(gold.items()))
    if set(task["qrels"]) != query_ids:
        raise ValueError("Query/qrels key sets differ")
    if not queries:
        raise ValueError("No compatible queries")
    queries.sort(key=lambda r: r["_id"])
    qrels = dict(sorted(qrels.items()))
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
                **{k: audit[k] for k in ("num_docs", "num_queries", "num_qrels")},
                "corpus_hash": corpus_hash.hexdigest(), "corpus_hash_format": "source-order canonical JSONL _id/title/text",
                "queries_hash": stable_json_sha256(queries), "qrels_hash": stable_json_sha256(qrels),
                "source_fingerprint": source["source_fingerprint"], "source_dir": source["raw_directory"],
                "input_task_hash": input_task_hash, "builder_sha256": builder_hash,
                "workspace_dir": str(workspace), "metadata_dir": str(metadata),
                "workspace_format": {"version": "dci-corpus-markdown-v2", "max_line_chars": 200},
                "protocol": protocol_contract(task_name=name, task_instruction=instruction)}
    write_json(md / "manifest.json", manifest)
    (ws / "README.md").write_text(f"# DCI-Bench corpus\n\nTask: {name}\n\nCorpus files are percent-encoded original document IDs plus `.md`.\n", encoding="utf-8")
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
    return manifest


def _verify_corpus_file(path: Path, row: dict) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != row["bytes"] or info.st_size > MAX_CORPUS_DOCUMENT_BYTES:
        raise ValueError(f"Invalid corpus file: {path}")
    if sha256_file(path) != row["sha256"]:
        raise ValueError(f"Changed corpus bytes: {path}")


def verify_document_task(suite: Path, name: str, *, io_workers: int = 32) -> dict:
    if type(io_workers) is not int or not 1 <= io_workers <= 256:
        raise ValueError("io_workers must be an integer from 1 through 256")
    md, ws, canon = suite / "metadata" / name, suite / "workspaces" / name, suite / "canonical" / name
    completion = json.loads((md / "_COMPLETE.json").read_text())
    if (completion.get("schema_version") != EXPANSION_VERSION
            or set(completion.get("metadata_hashes", {})) != METADATA_FILES
            or set(completion.get("canonical_hashes", {})) != CANONICAL_FILES):
        raise ValueError("Incomplete integrity manifest")
    if sha256_file(ws / "README.md") != completion.get("readme_sha256"):
        raise ValueError("Changed workspace README")
    readme_info = (ws / "README.md").lstat()
    if not stat.S_ISREG(readme_info.st_mode) or readme_info.st_nlink != 1:
        raise ValueError("Workspace README must be an independent regular file")
    if {str(p.relative_to(canon)) for p in canon.rglob("*") if p.is_file()} != CANONICAL_FILES:
        raise ValueError("Unexpected/missing canonical files")
    for filename, digest in completion["metadata_hashes"].items():
        if sha256_file(md / filename) != digest:
            raise ValueError(f"Changed metadata: {name}/{filename}")
    for filename, digest in completion["canonical_hashes"].items():
        if sha256_file(canon / filename) != digest:
            raise ValueError(f"Changed canonical data: {name}/{filename}")
    manifest = json.loads((md / "manifest.json").read_text())
    spec = TaskSpec(name, manifest["dataset"], manifest["revision"], name, manifest["task_instruction"])
    validate_task_manifest(spec, manifest)
    mapping = json.loads((md / "doc_id_files.json").read_text())
    qrels = json.loads((md / "qrels.json").read_text())
    queries = [json.loads(line) for line in (md / "queries.jsonl").read_text().splitlines()]
    if (len(queries) != manifest["num_queries"] or len(mapping) != manifest["num_docs"]
            or len({q["_id"] for q in queries}) != len(queries)
            or sum(len(g) for g in qrels.values()) != manifest["num_qrels"]):
        raise ValueError("Manifest counts or unique query IDs differ")
    if stable_json_sha256(qrels) != manifest["qrels_hash"] or stable_json_sha256(queries) != manifest["queries_hash"]:
        raise ValueError("Query/qrels canonical hash mismatch")
    if set(qrels) != {q["_id"] for q in queries} or any(not gold or any(d not in mapping for d in gold) for gold in qrels.values()):
        raise ValueError("Broken query/qrels/corpus closure")
    checked = set()
    pending = deque()
    with ThreadPoolExecutor(max_workers=io_workers) as pool, (md / "corpus_files.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            filename = row["filename"]
            if mapping.get(row["doc_id"]) != filename or doc_id_to_filename(row["doc_id"]) != filename or unquote(filename[:-3]) != row["doc_id"]:
                raise ValueError("Non-reversible document filename")
            if filename in checked:
                raise ValueError("Duplicate corpus integrity entry")
            checked.add(filename)
            pending.append(pool.submit(_verify_corpus_file, ws / "corpus" / filename, row))
            if len(pending) >= io_workers * 4:
                pending.popleft().result()
        for future in pending:
            future.result()
    if checked != set(mapping.values()) or checked != {p.name for p in (ws / "corpus").iterdir()}:
        raise ValueError("Unexpected/missing corpus files")
    if {p.name for p in ws.iterdir()} != {"corpus", "README.md"}:
        raise ValueError("Unexpected files in model-visible workspace")
    return manifest
