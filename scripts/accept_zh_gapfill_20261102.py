#!/usr/bin/env python3
"""Validate two Chinese gap-fill releases and freeze a new benchmark-v2 suite."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dci_bench.data.benchmark_v2 import (  # noqa: E402
    FULL_CAPACITY,
    LITE_CAPACITY,
    _load_pool_metadata,
    _source_corpus_reference,
    audit_benchmark_v2,
    load_benchmark_config,
    prepare_benchmark_v2,
    sha256_file,
    stable_json_sha256,
)
from dci_bench.data.document_pools import shared_pool_corpus, verify_document_pool_suite  # noqa: E402
from dci_bench.data.expansion import verify_document_task  # noqa: E402
from dci_bench.data.pipeline.core import load_bundle, verify_release  # noqa: E402


SOURCES = {
    "software_zh": ("gapfill_t2ranking_software_zh", "T2Ranking"),
    "enterprise_zh": ("gapfill_cfqa_enterprise_zh", "CFQA"),
}


class AcceptanceError(ValueError):
    pass


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _resolve(value: str | Path, base: Path, data_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = data_root / path
    return candidate if candidate.exists() else base / path


def _source_row(data_root: Path, source_id: str) -> tuple[dict[str, Any], Path]:
    row_path = data_root / "run" / source_id / "catalog-row.json"
    row = _json(row_path)
    if isinstance(row, Mapping) and isinstance(row.get("catalog_row"), Mapping):
        row = dict(row["catalog_row"])
    if not isinstance(row, Mapping) or row.get("source") != source_id:
        raise AcceptanceError(f"catalog-row source identity mismatch: {row_path}")
    if row.get("status") is not None and row["status"] not in {"complete", "consolidated_verified", "ready"}:
        raise AcceptanceError(f"catalog-row is not complete: {row_path}")
    result = copy.deepcopy(dict(row))
    release = data_root / "releases" / source_id
    declared = row.get("release")
    declared_path = _resolve(declared, row_path.parent, data_root) if isinstance(declared, str) else None
    if declared_path and declared_path.resolve() != release.resolve():
        raise AcceptanceError(f"catalog-row release path differs from fixed release path: {source_id}")
    if row.get("content_hashes_verified") is False or row.get("source_files_present") is False:
        raise AcceptanceError(f"catalog-row content hashes are not verified: {source_id}")
    result["release"] = str(release)
    pools = result.get("pools")
    if not isinstance(pools, list):
        raise AcceptanceError(f"catalog-row has no pools: {source_id}")
    normalised = []
    for pool in pools:
        if not isinstance(pool, Mapping) or not isinstance(pool.get("capacity"), int) or not isinstance(pool.get("path"), str):
            raise AcceptanceError(f"invalid pool row: {source_id}")
        normalised.append({**pool, "path": str(_resolve(pool["path"], row_path.parent, data_root))})
    result["pools"] = normalised
    return result, row_path


def _pool(result: Mapping[str, Any], capacity: int) -> tuple[dict[str, Any], Path]:
    matches = [dict(row) for row in result["pools"] if row.get("capacity") == capacity]
    if len(matches) != 1:
        raise AcceptanceError(f"expected one {capacity} pool for {result.get('source')}")
    path = Path(matches[0]["path"])
    marker = path / "metadata" / str(result.get("task", result["source"])) / "_COMPLETE.json"
    if not marker.is_file() or marker.is_symlink():
        raise AcceptanceError(f"missing pool completion marker: {marker}")
    declared = matches[0].get("completion_sha256")
    actual = sha256_file(marker)
    if not isinstance(declared, str) or declared != actual:
        raise AcceptanceError(f"pool completion hash mismatch: {path}")
    matches[0]["path"] = str(path)
    return matches[0], path


def _audit_source(data_root: Path, source_id: str) -> dict[str, Any]:
    row, row_path = _source_row(data_root, source_id)
    release = Path(row["release"])
    if not (release / "_COMPLETE.json").is_file():
        raise AcceptanceError(f"missing release completion marker: {release}")
    release_manifest = verify_release(release)
    task = str(row.get("task", source_id))
    if task != source_id:
        raise AcceptanceError(f"source/task mapping mismatch: {source_id}/{task}")
    export = data_root / "run" / "source"
    declared_export = row.get("export")
    if not isinstance(declared_export, str) or _resolve(declared_export, row_path.parent, data_root).resolve() != export.resolve():
        raise AcceptanceError(f"catalog-row export path differs from fixed export path: {source_id}")
    export_metadata = export / "metadata" / task
    export_manifest = verify_document_task(export, task, verify_content_hashes=False)
    export_source_manifest = _json(export_metadata / "source_manifest.json")
    if export_source_manifest.get("source_fingerprint") != release_manifest.get("bundle_hash"):
        raise AcceptanceError(f"export/release bundle identity mismatch: {source_id}")
    release_completion_sha256 = sha256_file(release / "_COMPLETE.json")
    declared_release_hash = row.get("release_completion_sha256")
    if declared_release_hash is not None and declared_release_hash != release_completion_sha256:
        raise AcceptanceError(f"release completion hash mismatch: {source_id}")
    benchmark_task = next(
        task_id for task_id, (expected_source, _) in SOURCES.items()
        if expected_source == source_id
    )
    full_entry, full_root = _pool(row, FULL_CAPACITY)
    lite_entry, lite_root = _pool(row, LITE_CAPACITY)
    verify_document_pool_suite(full_root, task=task, verify_content_hashes=False)
    verify_document_pool_suite(lite_root, task=task, verify_content_hashes=False)
    full = _load_pool_metadata(full_root, task, include_candidate_pools=True)
    lite_meta = _load_pool_metadata(lite_root, task, include_candidate_pools=True)
    lite = {"manifest": lite_meta["manifest"], "source_identity": lite_meta["source_identity"]}
    export_manifest_sha256 = sha256_file(export_metadata / "manifest.json")
    if full["source_identity"].get("source_manifest_sha256") != export_manifest_sha256:
        raise AcceptanceError(f"full pool/export manifest identity mismatch: {source_id}")
    release_bundle = load_bundle(release)
    if full["original_qrels"] != release_bundle["qrels"]:
        raise AcceptanceError(f"full pool/release qrels mismatch: {source_id}")
    ranking_value = row.get("ranking")
    ranking_hash = row.get("ranking_sha256")
    if not isinstance(ranking_value, str) or not isinstance(ranking_hash, str):
        raise AcceptanceError(f"catalog-row ranking receipt is incomplete: {source_id}")
    ranking = _resolve(ranking_value, row_path.parent, data_root)
    if ranking.is_symlink() or not ranking.is_file() or sha256_file(ranking) != ranking_hash:
        raise AcceptanceError(f"catalog-row ranking hash mismatch: {source_id}")
    if full["manifest"].get("capacity") != FULL_CAPACITY or lite["manifest"].get("capacity") != LITE_CAPACITY:
        raise AcceptanceError(f"pool capacity marker mismatch: {source_id}")
    if full["source_identity"] != lite["source_identity"]:
        raise AcceptanceError(f"full/lite identity mismatch for {source_id}")
    if full["qrels"] != lite_meta["qrels"]:
        raise AcceptanceError(f"full/lite qrels differ: {source_id}")
    if row.get("queries") is not None and row["queries"] != len(full["queries"]):
        raise AcceptanceError(f"catalog query count differs from pool metadata: {source_id}")
    if row.get("documents") is not None and full["manifest"].get("corpus_size") is not None and row["documents"] != full["manifest"]["corpus_size"]:
        raise AcceptanceError(f"catalog document count differs from pool metadata: {source_id}")
    full_corpus = _source_corpus_reference(full_root, task)
    lite_corpus = _source_corpus_reference(lite_root, task)
    candidate_payload = full.get("candidate_pools")
    candidates = candidate_payload.get("pools") if isinstance(candidate_payload, Mapping) else None
    lite_payload = lite_meta.get("candidate_pools")
    lite_candidates = lite_payload.get("pools") if isinstance(lite_payload, Mapping) else None
    if not isinstance(candidates, Mapping) or not isinstance(lite_candidates, Mapping):
        raise AcceptanceError(f"full candidate pools missing: {source_id}")
    positive_pairs = 0
    positive_missing: list[dict[str, Any]] = []
    query_ids = [row.get("_id", row.get("query_id")) for row in full["queries"]]
    group_ids = [row.get("group_id") for row in full["queries"]]
    if any(not isinstance(value, str) or not value for value in query_ids) or len(query_ids) != len(set(query_ids)):
        raise AcceptanceError(f"source query IDs are missing or duplicated: {source_id}")
    if any(not isinstance(value, str) or not value for value in group_ids):
        raise AcceptanceError(f"source query groups are missing: {source_id}")
    if set(query_ids) != set(full["qrels"]) or set(candidates) != set(full["qrels"]):
        raise AcceptanceError(f"source query/qrels/candidate IDs differ: {source_id}")
    for query_id, labels in full["qrels"].items():
        record = candidates.get(query_id)
        if not isinstance(record, Mapping):
            raise AcceptanceError(f"full candidate pool missing query: {source_id}/{query_id}")
        ids = record.get("candidate_doc_ids")
        if not isinstance(ids, list) or len(ids) != FULL_CAPACITY or len(set(ids)) != FULL_CAPACITY:
            raise AcceptanceError(f"full pool is not exactly {FULL_CAPACITY}: {source_id}/{query_id}")
        lite_record = lite_candidates.get(query_id)
        lite_ids = lite_record.get("candidate_doc_ids") if isinstance(lite_record, Mapping) else None
        if (not isinstance(lite_ids, list) or len(lite_ids) != LITE_CAPACITY
                or len(set(lite_ids)) != LITE_CAPACITY):
            raise AcceptanceError(f"lite pool is not exactly {LITE_CAPACITY}: {source_id}/{query_id}")
        for doc_id, score in labels.items():
            if float(score) > 0:
                positive_pairs += 1
                if doc_id not in ids:
                    positive_missing.append({"query_id": query_id, "doc_id": doc_id, "tier": "full"})
                if doc_id not in lite_ids:
                    positive_missing.append({"query_id": query_id, "doc_id": doc_id, "tier": "lite"})
    if positive_missing:
        raise AcceptanceError(f"full pool dropped positives: {source_id}: {positive_missing[:3]}")
    catalog_row = copy.deepcopy(row)
    catalog_row.update(
        status="complete",
        release=str(release),
        release_completion_sha256=release_completion_sha256,
        source_files_present=True,
        content_hashes_verified=bool(catalog_row.get("content_hashes_verified", True)),
    )
    return {"source_id": source_id, "task": task, "benchmark_task": benchmark_task,
            "catalog_row": catalog_row, "catalog_row_path": str(row_path),
            "release": {**release_manifest, "path": str(release), "completion_sha256": release_completion_sha256,
                        "manifest_sha256": sha256_file(release / "manifest.json")},
            "export": {"path": str(export), "task": task,
                       "manifest": export_manifest,
                       "manifest_sha256": export_manifest_sha256,
                       "source_manifest_sha256": sha256_file(export_metadata / "source_manifest.json")},
            "ranking": {"path": str(ranking), "sha256": ranking_hash},
            "pools": {"full": full_entry, "lite": lite_entry},
            "corpus": {"path": str(full_corpus), "lite_path": str(lite_corpus)},
            "pool_identity": full["source_identity"], "queries": len(full["queries"]),
            "documents": row.get("documents", release_manifest.get("num_documents")),
            "positive_pairs": positive_pairs, "positive_missing": positive_missing,
            "all_positive_labels_retained": not positive_missing,
            "nested_audit": {"query_count": len(full["qrels"]), "source_pools_nested_checked": False,
                             "policy": "source tiers are independently selected; final benchmark_v2 is nested",
                             "lite_capacity": LITE_CAPACITY, "full_capacity": FULL_CAPACITY},
            "group_audit": {"query_ids_unique": len(set(query_ids)), "query_rows": len(full["queries"]),
                            "group_ids_unique": len(set(group_ids)),
                            "policy": "benchmark_v2 global equivalence groups"}}


def _append_sources(catalog: Mapping[str, Any], closures: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result = copy.deepcopy(dict(catalog))
    rows = list(result.get("sources", []))
    known = {row.get("source") for row in rows if isinstance(row, Mapping)}
    for source_id in (item[0] for item in SOURCES.values()):
        if source_id in known:
            raise AcceptanceError(f"new source already exists in base catalog: {source_id}")
        row = copy.deepcopy(dict(closures[source_id]["catalog_row"]))
        row["source"] = source_id
        row["release"] = closures[source_id]["release"]["path"]
        rows.append(row)
    result["sources"] = rows
    result["source_count"] = len(rows)
    result["pool_count"] = sum(len(row.get("pools", [])) for row in rows)
    result["queries"] = sum(int(row.get("queries", 0)) for row in rows)
    return result


def _append_config(config: Mapping[str, Any], closures: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    result = copy.deepcopy(dict(config))
    changes = {}
    for task_id, (source_id, family) in SOURCES.items():
        entry = result["sources"][task_id]
        old_candidates = copy.deepcopy(entry["candidates"])
        if source_id not in {candidate.get("source_id") for candidate in old_candidates}:
            preferred = {"evidence": closures[source_id]["catalog_row"].get("evidence", closures[source_id]["catalog_row_path"]),
                         "provenance": "native", "role": "preferred_native", "source_family": family, "source_id": source_id}
            entry["candidates"] = [preferred, *old_candidates]
        entry["status"] = "eligible"
        changes[task_id] = {"source_id": source_id, "old_status": config["sources"][task_id].get("status"),
                            "new_status": entry["status"], "candidate_count_before": len(old_candidates),
                            "candidate_count_after": len(entry["candidates"])}
    return result, changes


def _assert_old_selection(base_manifest: Mapping[str, Any], new_manifest: Mapping[str, Any]) -> dict[str, Any]:
    old = {task: task_row for task, task_row in base_manifest["tasks"].items() if task_row.get("status") == "ready"}
    new = new_manifest["tasks"]
    if len(old) != 17:
        raise AcceptanceError(f"base suite ready task count is {len(old)}, expected 17")
    changed = []
    for task_id, old_row in old.items():
        new_row = new.get(task_id, {})
        if new_row.get("source_id") != old_row.get("source_id") or new_row.get("query_ids") != old_row.get("query_ids"):
            changed.append(task_id)
    if changed:
        raise AcceptanceError(f"old task source/query selection changed: {changed}")
    ready = {task for task, row in new.items() if row.get("status") == "ready"}
    expected = set(old) | set(SOURCES)
    if ready != expected:
        raise AcceptanceError(f"ready task roster differs: expected {sorted(expected)}, got {sorted(ready)}")
    return {"old_ready_tasks": sorted(old), "new_ready_tasks": sorted(ready), "changed": changed}


def _selected_full_hashes(suite: Path, manifest: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    task = manifest["tasks"][task_id]
    root = suite / "tasks" / task_id / "full"
    metadata = _load_pool_metadata(root, task_id, include_candidate_pools=True)
    inventory: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    corpus = shared_pool_corpus(root, task_id)
    if corpus is None:
        raise AcceptanceError(f"final suite has no shared corpus reference: {root}")
    for relative, row in metadata["file_inventory"].items():
        if not isinstance(row, Mapping) or not isinstance(row.get("alias"), str):
            raise AcceptanceError(f"selected full file inventory row is invalid: {task_id}")
        alias = row["alias"]
        body = corpus / Path(str(relative)).name
        if alias in inventory and inventory[alias][0] != body:
            raise AcceptanceError(f"selected full alias has multiple body paths: {task_id}/{alias}")
        inventory[alias] = (body, row)
    mapping = metadata["source_doc_ids"]
    selected = sorted(set(task["query_ids"]["dev"]) | set(task["query_ids"]["test"]))
    records = metadata["candidate_pools"].get("pools") if isinstance(metadata["candidate_pools"], Mapping) else None
    if not isinstance(records, Mapping):
        raise AcceptanceError(f"selected full candidate pools missing: {task_id}")
    body_hashes: dict[str, tuple[str, int]] = {}
    rows = []
    for query_id in selected:
        if query_id not in records:
            raise AcceptanceError(f"selected full candidate pool missing: {task_id}/{query_id}")
        record = records[query_id]
        query_documents = []
        for alias in record["candidate_doc_ids"]:
            if alias not in mapping or alias not in inventory:
                raise AcceptanceError(f"selected full body inventory missing: {task_id}/{query_id}/{alias}")
            body, declared = inventory[alias]
            if alias not in body_hashes:
                if body.is_symlink() or not body.is_file():
                    raise AcceptanceError(f"selected full body is missing or linked: {task_id}/{alias}")
                actual = sha256_file(body)
                expected = declared.get("sha256")
                if not isinstance(expected, str) or actual != expected:
                    raise AcceptanceError(f"selected full body hash mismatch: {task_id}/{alias}")
                expected_bytes = declared.get("bytes")
                if expected_bytes is not None and body.stat().st_size != expected_bytes:
                    raise AcceptanceError(f"selected full body size mismatch: {task_id}/{alias}")
                body_hashes[alias] = (actual, body.stat().st_size)
            actual, byte_count = body_hashes[alias]
            query_documents.append({"alias": alias, "original_id": mapping[alias],
                                    "sha256": actual, "bytes": byte_count})
        rows.append({"query_id": query_id, "documents": query_documents})
    return {"task": task_id, "query_count": len(rows), "capacity": FULL_CAPACITY,
            "verified_unique_body_count": len(body_hashes),
            "selected_full_body_sha256": stable_json_sha256(rows), "query_rows": rows}


def accept(data_root: Path | str, output_suite: Path | str, evidence_dir: Path | str,
           config_path: Path | str, base_catalog: Path | str, base_suite: Path | str,
           *, check_only: bool = False) -> dict[str, Any]:
    data_root, output_suite, evidence_dir = map(Path, (data_root, output_suite, evidence_dir))
    if not check_only and (output_suite.exists() or output_suite.is_symlink()):
        raise FileExistsError(f"refusing to overwrite output suite: {output_suite}")
    base_manifest = _json(base_suite / "manifest.json")
    base_audit = audit_benchmark_v2(base_suite)
    if base_audit.get("ready_for_formal") is not False:
        raise AcceptanceError("base suite unexpectedly reports ready_for_formal")
    expected_base_blocked = {"enterprise_zh", "software_zh", "web_en"}
    if set(base_audit.get("ready_tasks", [])) | set(base_audit.get("blocked_tasks", [])) != set(base_manifest["tasks"]):
        raise AcceptanceError("base suite audit does not cover the manifest task roster")
    if len(base_audit.get("ready_tasks", [])) != 17 or set(base_audit.get("blocked_tasks", [])) != expected_base_blocked:
        raise AcceptanceError("base suite ready/blocked roster differs from the frozen 17-task baseline")
    closures = {source_id: _audit_source(data_root, source_id) for source_id, _ in SOURCES.values()}
    config = load_benchmark_config(Path(config_path))
    new_config, config_changes = _append_config(config, closures)
    catalog = _json(Path(base_catalog))
    new_catalog = _append_sources(catalog, closures)
    if check_only:
        receipt = {"status": "ready", "check_only": True, "base_audit": base_audit,
                   "source_ids": sorted(closures), "source_closure": closures}
        return receipt
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _write(evidence_dir / "config.json", new_config)
    _write(evidence_dir / "catalog.json", new_catalog)
    _write(evidence_dir / "source-closure.json", closures)
    _write(evidence_dir / "config-changes.json", config_changes)
    output_suite.parent.mkdir(parents=True, exist_ok=True)
    staging = output_suite.parent / f".{output_suite.name}.staging-{uuid.uuid4().hex}"
    try:
        prepared = prepare_benchmark_v2(output_dir=staging, config_path=evidence_dir / "config.json",
                                        catalog_path=evidence_dir / "catalog.json", seed=42)
        suite_audit = audit_benchmark_v2(staging)
        expected_ready = set(base_audit["ready_tasks"]) | set(SOURCES)
        if suite_audit.get("ready_for_formal") is not False:
            raise AcceptanceError("new suite unexpectedly reports ready_for_formal")
        if set(suite_audit.get("ready_tasks", [])) != expected_ready or set(suite_audit.get("blocked_tasks", [])) != {"web_en"}:
            raise AcceptanceError("new suite ready/blocked roster differs from expected 19-task closure")
        old_check = _assert_old_selection(base_manifest, prepared)
        selected_hashes = {task_id: _selected_full_hashes(staging, prepared, task_id) for task_id in SOURCES}
        os.replace(staging, output_suite)
    except Exception as exc:
        _write(evidence_dir / "status-failed.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                                      "staging": str(staging)})
        raise
    final_manifest = _json(output_suite / "manifest.json")
    audit = {"status": "ready", "task_count": suite_audit["task_count"],
             "ready_for_formal": suite_audit["ready_for_formal"],
             "ready_tasks": suite_audit["ready_tasks"], "blocked_tasks": suite_audit["blocked_tasks"],
             "base_audit": base_audit, "suite_audit": suite_audit,
             "old_selection": old_check, "selected_full_hashes": selected_hashes,
             "output_suite": str(output_suite), "manifest_sha256": sha256_file(output_suite / "manifest.json")}
    _write(evidence_dir / "prepared-manifest.json", final_manifest)
    _write(evidence_dir / "data-audit.json", audit)
    _write(evidence_dir / "receipt.json", audit)
    _write(evidence_dir / "status-ready.json", audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-suite", type=Path,
                        default=Path("/mnt/afs/250010100/zxp/dci-bench-zh-data/20261102_v4/suites/benchmark_v2_20261102_zh_v1"))
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-catalog", type=Path, required=True)
    parser.add_argument("--base-suite", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    result = accept(args.data_root, args.output_suite, args.evidence_dir, args.config,
                    args.base_catalog, args.base_suite, check_only=args.check_only)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
