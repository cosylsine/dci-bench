"""Local parquet loader for pinned MTEB(LLM) retrieval datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dci_bench.data.registry import DEFAULT_DATA_ROOT, TaskSpec


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_files(paths: Iterable[Path]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        hasher.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def resolve_task_dir(task: TaskSpec, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    candidates = [
        data_root / task.local_dir_name,
        data_root / "mteb_llm_retrieval" / task.local_dir_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find local data for {task.name}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def parquet_files(task_dir: Path, subset: str) -> list[Path]:
    files = sorted((task_dir / subset).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {task_dir / subset}")
    return files


def read_parquet_rows(files: Iterable[Path], required_columns: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        table = pq.read_table(path)
        missing = required_columns.difference(table.column_names)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        rows.extend(table.to_pylist())
    return rows


def load_corpus(task_dir: Path) -> list[dict[str, str]]:
    rows = read_parquet_rows(parquet_files(task_dir, "corpus"), {"_id", "title", "text"})
    docs: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        doc_id = str(row["_id"])
        if doc_id in seen:
            raise ValueError(f"Duplicate corpus id: {doc_id}")
        seen.add(doc_id)
        docs.append(
            {
                "_id": doc_id,
                "title": "" if row.get("title") is None else str(row.get("title")),
                "text": "" if row.get("text") is None else str(row.get("text")),
            }
        )
    return docs


def load_queries(task_dir: Path) -> list[dict[str, str]]:
    rows = read_parquet_rows(parquet_files(task_dir, "queries"), {"_id", "text"})
    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        query_id = str(row["_id"])
        if query_id in seen:
            raise ValueError(f"Duplicate query id: {query_id}")
        seen.add(query_id)
        queries.append({"_id": query_id, "text": "" if row.get("text") is None else str(row.get("text"))})
    return queries


def load_qrels(task_dir: Path) -> dict[str, dict[str, float]]:
    rows = read_parquet_rows(parquet_files(task_dir, "data"), {"query-id", "corpus-id", "score"})
    qrels: dict[str, dict[str, float]] = {}
    for row in rows:
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])
        score = float(row["score"])
        query_qrels = qrels.setdefault(query_id, {})
        if corpus_id in query_qrels:
            raise ValueError(f"Duplicate qrel pair: query_id={query_id!r}, corpus_id={corpus_id!r}")
        query_qrels[corpus_id] = score
    return qrels


def assert_qrels_reference_known_ids(
    qrels: Mapping[str, Mapping[str, float]],
    query_ids: set[str],
    doc_ids: set[str],
) -> None:
    missing_queries = sorted(set(qrels) - query_ids)
    missing_docs = sorted({doc_id for docs in qrels.values() for doc_id in docs} - doc_ids)
    if missing_queries:
        raise ValueError(f"Qrels reference unknown query ids: {missing_queries[:5]}")
    if missing_docs:
        raise ValueError(f"Qrels reference unknown corpus ids: {missing_docs[:5]}")
