#!/usr/bin/env python3
"""Export verified source examples from a frozen benchmark-v2 suite.

The suite stores query metadata and candidate-pool metadata separately from the
shared source corpus.  This exporter follows those pointers for one deterministic
dev query per task, then verifies the selected positive body against the source
inventory before creating the output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


EXCERPT_CHARS = 700
MAX_WORKERS = 4
EXPECTED_TASK_COUNT = 20


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_jsonl_queries(path: Path) -> dict[str, dict[str, Any]]:
    queries: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            query = json.loads(line)
            query_id = query.get("_id")
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"{path}:{line_number}: query has no non-empty _id")
            if query_id in queries:
                raise ValueError(f"{path}:{line_number}: duplicate query id {query_id}")
            if not isinstance(query.get("text"), str):
                raise ValueError(f"{path}:{line_number}: query {query_id} has no text")
            queries[query_id] = query
    return queries


def metadata_paths(suite: Path, task: str, tier: str) -> tuple[Path, Path]:
    base = suite / "tasks" / task / tier
    metadata_dir = base / "metadata" / task
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"missing {tier} metadata directory for {task}: {metadata_dir}")
    return base, metadata_dir


def inventory_row(inventory: dict[str, Any], filename: str, task: str) -> tuple[str, dict[str, Any]]:
    matches = [(key, value) for key, value in inventory.items() if key.endswith("/" + filename)]
    if len(matches) != 1:
        raise ValueError(
            f"{task}: expected one inventory row for {filename}, found {len(matches)}"
        )
    key, value = matches[0]
    if not isinstance(value, dict):
        raise ValueError(f"{task}: inventory row {key} is not an object")
    return key, value


def pool_for_query(
    pool_metadata: dict[str, Any], query_id: str, tier: str, task: str
) -> dict[str, Any]:
    pools = pool_metadata.get("pools")
    if not isinstance(pools, dict) or query_id not in pools:
        raise ValueError(f"{task}: {tier} candidate pool missing query {query_id}")
    pool = pools[query_id]
    if not isinstance(pool, dict):
        raise ValueError(f"{task}: {tier} candidate pool for {query_id} is not an object")
    ids = pool.get("candidate_doc_ids")
    scores = pool.get("candidate_scores")
    originals = pool.get("candidate_original_doc_ids")
    if (
        not isinstance(ids, list)
        or not isinstance(scores, list)
        or not isinstance(originals, list)
        or len(ids) != len(scores)
        or len(ids) != len(originals)
    ):
        raise ValueError(f"{task}: malformed {tier} candidate pool for {query_id}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{task}: duplicate canonical ids in {tier} pool for {query_id}")
    return pool


def task_example(suite: Path, task: str, suite_manifest: dict[str, Any]) -> dict[str, Any]:
    full_base, full_metadata = metadata_paths(suite, task, "full")
    lite_base, lite_metadata = metadata_paths(suite, task, "lite")

    full_manifest_path = full_metadata / "manifest.json"
    lite_manifest_path = lite_metadata / "manifest.json"
    full_manifest = read_json(full_manifest_path)
    lite_manifest = read_json(lite_manifest_path)
    if full_manifest.get("task") != task or lite_manifest.get("task") != task:
        raise ValueError(f"{task}: task name mismatch in tier manifest")
    if full_manifest.get("pool_size") != "full" or lite_manifest.get("pool_size") != "lite":
        raise ValueError(f"{task}: tier manifest pool_size mismatch")
    if full_manifest.get("sampling", {}).get("splits") != lite_manifest.get("sampling", {}).get("splits"):
        raise ValueError(f"{task}: full/lite selected splits differ")

    dev_ids = full_manifest.get("sampling", {}).get("splits", {}).get("dev")
    if not isinstance(dev_ids, list) or not dev_ids:
        raise ValueError(f"{task}: missing selected dev ids")
    query_id = sorted(dev_ids)[0]

    full_qrels_path = full_metadata / "qrels.json"
    lite_qrels_path = lite_metadata / "qrels.json"
    full_qrels = read_json(full_qrels_path)
    lite_qrels = read_json(lite_qrels_path)
    original_qrels = read_json(full_metadata / "original_qrels.json")
    if full_qrels != lite_qrels:
        raise ValueError(f"{task}: full/lite qrels differ")
    qrels = full_qrels.get(query_id)
    if not isinstance(qrels, dict):
        raise ValueError(f"{task}: qrels missing selected dev query {query_id}")
    positives = [(doc_id, float(relevance)) for doc_id, relevance in qrels.items() if float(relevance) > 0]
    if not 1 <= len(positives) <= 10:
        raise ValueError(f"{task}: selected query {query_id} has {len(positives)} positives")

    full_pool_metadata = read_json(full_metadata / "candidate_pools.json")
    lite_pool_metadata = read_json(lite_metadata / "candidate_pools.json")
    full_pool = pool_for_query(full_pool_metadata, query_id, "full", task)
    lite_pool = pool_for_query(lite_pool_metadata, query_id, "lite", task)
    full_ids = set(full_pool["candidate_doc_ids"])
    lite_ids = set(lite_pool["candidate_doc_ids"])
    candidates = [row for row in positives if row[0] in full_ids and row[0] in lite_ids]
    if not candidates:
        raise ValueError(f"{task}: no qrel positive for {query_id} is present in both pools")
    canonical_id, relevance = sorted(candidates, key=lambda row: (-row[1], row[0]))[0]

    full_source_doc_ids = read_json(full_metadata / "source_doc_ids.json")
    lite_source_doc_ids = read_json(lite_metadata / "source_doc_ids.json")
    original_id = full_source_doc_ids.get(canonical_id)
    if not isinstance(original_id, str) or not original_id:
        raise ValueError(f"{task}: source_doc_ids has no original id for {canonical_id}")
    if lite_source_doc_ids.get(canonical_id) != original_id:
        raise ValueError(f"{task}: full/lite source_doc_ids differ for {canonical_id}")
    original_qrel = original_qrels.get(query_id)
    if not isinstance(original_qrel, dict):
        raise ValueError(f"{task}: original qrels missing selected dev query {query_id}")
    for positive_id, positive_relevance in positives:
        mapped_original_id = full_source_doc_ids.get(positive_id)
        if mapped_original_id is None or float(original_qrel.get(mapped_original_id, 0)) != positive_relevance:
            raise ValueError(f"{task}: original qrel mapping mismatch for {positive_id}")

    full_doc_files = read_json(full_metadata / "doc_id_files.json")
    lite_doc_files = read_json(lite_metadata / "doc_id_files.json")
    filename = full_doc_files.get(canonical_id)
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"{task}: doc_id_files has no filename for {canonical_id}")
    if lite_doc_files.get(canonical_id) != filename:
        raise ValueError(f"{task}: full/lite doc_id_files differ for {canonical_id}")

    full_inventory = read_json(full_metadata / "file_inventory.json")
    lite_inventory = read_json(lite_metadata / "file_inventory.json")
    inventory_key, inventory_entry = inventory_row(full_inventory, filename, task)
    _, lite_inventory_entry = inventory_row(lite_inventory, filename, task)
    if inventory_entry != lite_inventory_entry:
        raise ValueError(f"{task}: full/lite inventory rows differ for {filename}")
    if inventory_entry.get("alias") != canonical_id:
        raise ValueError(f"{task}: inventory alias mismatch for {filename}")
    if inventory_entry.get("original_id") != original_id:
        raise ValueError(f"{task}: inventory original id mismatch for {filename}")

    full_shared = read_json(full_base / "shared_corpus.json")
    lite_shared = read_json(lite_base / "shared_corpus.json")
    corpus_path = Path(full_shared.get("corpus", ""))
    if not corpus_path.is_dir():
        raise FileNotFoundError(f"{task}: shared corpus path is not a directory: {corpus_path}")
    if full_shared != lite_shared:
        raise ValueError(f"{task}: full/lite shared_corpus pointers differ")
    document_path = corpus_path / filename
    document_bytes = document_path.read_bytes()
    document_sha256 = sha256_bytes(document_bytes)
    expected_bytes = inventory_entry.get("bytes")
    expected_sha256 = inventory_entry.get("sha256")
    if len(document_bytes) != expected_bytes or document_sha256 != expected_sha256:
        raise ValueError(
            f"{task}: body hash/inventory mismatch for {document_path} "
            f"(got {len(document_bytes)}/{document_sha256}, expected {expected_bytes}/{expected_sha256})"
        )
    document_text = document_bytes.decode("utf-8")

    queries = read_jsonl_queries(full_metadata / "queries.jsonl")
    query = queries.get(query_id)
    if query is None:
        raise ValueError(f"{task}: queries missing selected dev query {query_id}")
    if query.get("_id") != query_id:
        raise ValueError(f"{task}: query id mismatch for {query_id}")
    if full_manifest.get("num_queries") != len(queries):
        raise ValueError(f"{task}: manifest/query count mismatch")

    full_rank = full_pool["candidate_doc_ids"].index(canonical_id) + 1
    lite_rank = lite_pool["candidate_doc_ids"].index(canonical_id) + 1
    if full_pool["candidate_original_doc_ids"][full_rank - 1] != original_id:
        raise ValueError(f"{task}: full candidate original id mismatch for {canonical_id}")
    if lite_pool["candidate_original_doc_ids"][lite_rank - 1] != original_id:
        raise ValueError(f"{task}: lite candidate original id mismatch for {canonical_id}")

    complete = read_json(full_metadata / "_COMPLETE.json")
    source_identity = read_json(full_metadata / "source_identity.json")
    benchmark_task = suite_manifest.get("tasks", {}).get(task, {})
    sampling = full_manifest.get("sampling", {})
    source_id = sampling.get("source_id")
    source_family = sampling.get("source_family")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(f"{task}: manifest has no source id")
    if not isinstance(source_family, str) or not source_family:
        raise ValueError(f"{task}: manifest has no source family")
    if source_id != lite_manifest.get("sampling", {}).get("source_id"):
        raise ValueError(f"{task}: full/lite source ids differ")

    excerpt = document_text[:EXCERPT_CHARS]
    return {
        "task": task,
        "source_id": source_id,
        "source_family": source_family,
        "domain": benchmark_task.get("domain"),
        "language": benchmark_task.get("language"),
        "selected_split": "dev",
        "query_id": query_id,
        "query_text": query["text"],
        "positive_count": len(positives),
        "positive_document_id": canonical_id,
        "original_document_id": original_id,
        "relevance": relevance,
        "document_excerpt": excerpt,
        "excerpt_truncated": len(document_text) > EXCERPT_CHARS,
        "document_bytes": len(document_bytes),
        "document_sha256": document_sha256,
        "document_path": str(document_path),
        "candidate_membership": {
            "lite": canonical_id in lite_ids,
            "full": canonical_id in full_ids,
            "lite_rank": lite_rank,
            "full_rank": full_rank,
        },
        "source_corpus_count": full_manifest.get("corpus_size"),
        "source_eligible_query_count": suite_manifest["tasks"][task].get("eligible_query_count"),
        "metadata": {
            "queries_hash": full_manifest.get("queries_hash"),
            "qrels_hash": full_manifest.get("qrels_hash"),
            "original_qrels_hash": full_manifest.get("original_qrels_hash"),
            "queries_file_sha256": sha256_file(full_metadata / "queries.jsonl"),
            "qrels_file_sha256": sha256_file(full_qrels_path),
            "original_qrels_file_sha256": sha256_file(full_metadata / "original_qrels.json"),
            "task_manifest_sha256": sha256_file(full_manifest_path),
            "complete_marker_sha256": sha256_file(full_metadata / "_COMPLETE.json"),
            "source_identity_sha256": sha256_file(full_metadata / "source_identity.json"),
            "source_identity": source_identity,
            "shared_corpus_sha256": complete.get("shared_corpus_sha256"),
            "shared_corpus_path": full_shared.get("corpus"),
            "inventory_key": inventory_key,
            "full_pool_capacity": full_pool_metadata.get("capacity"),
            "lite_pool_capacity": lite_pool_metadata.get("capacity"),
        },
    }


def build_export(suite: Path) -> dict[str, Any]:
    suite_manifest_path = suite / "benchmark-manifest.json"
    suite_manifest = read_json(suite_manifest_path)
    task_root = suite / "tasks"
    tasks = sorted(path.name for path in task_root.iterdir() if path.is_dir())
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise ValueError(f"expected {EXPECTED_TASK_COUNT} task directories, found {len(tasks)}")
    if set(tasks) != set(suite_manifest.get("tasks", {})):
        raise ValueError("task directories do not match benchmark-manifest tasks")

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as executor:
        examples = list(executor.map(lambda task: task_example(suite, task, suite_manifest), tasks))
    if len(examples) != EXPECTED_TASK_COUNT:
        raise ValueError(f"expected {EXPECTED_TASK_COUNT} examples, found {len(examples)}")
    if len({example["task"] for example in examples}) != EXPECTED_TASK_COUNT:
        raise ValueError("examples do not have distinct tasks")
    if len({example["source_id"] for example in examples}) != EXPECTED_TASK_COUNT:
        raise ValueError("examples do not have distinct source ids")
    for example in examples:
        if example["positive_count"] < 1 or example["positive_count"] > 10:
            raise ValueError(f"{example['task']}: positive count outside 1..10")
        if example["candidate_membership"] != {
            "lite": True,
            "full": True,
            "lite_rank": example["candidate_membership"]["lite_rank"],
            "full_rank": example["candidate_membership"]["full_rank"],
        }:
            raise ValueError(f"{example['task']}: selected positive is not in both pools")

    root_manifest_bytes = (suite / "manifest.json").read_bytes()
    return {
        "schema_version": "dci-benchmark-source-examples-v1",
        "suite": {
            "path": str(suite),
            "manifest_sha256": sha256_bytes(root_manifest_bytes),
            "benchmark_manifest_sha256": sha256_file(suite_manifest_path),
            "overall_sha256": suite_manifest.get("overall_sha256"),
            "config_identity": suite_manifest.get("config_identity"),
        },
        "examples": examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path, help="immutable benchmark suite directory")
    parser.add_argument("--output", required=True, type=Path, help="new JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = args.suite.resolve()
    output = args.output.resolve()
    if not suite.is_dir():
        raise FileNotFoundError(f"suite directory does not exist: {suite}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {output.parent}")
    payload = build_export(suite)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    print(f"wrote {output} ({len(payload['examples'])} examples)")


if __name__ == "__main__":
    main()
