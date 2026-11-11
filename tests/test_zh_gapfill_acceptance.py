"""CPU contract tests for the two-source Chinese gap-fill acceptance gate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dci_bench.data.document_pools import (
    EXPECTED_EMBEDDING_MODEL_PATH,
    build_source_identity,
    prepare_document_pools,
)
from dci_bench.data.pipeline.core import export_dci, publish_bundle
import scripts.accept_zh_gapfill_20261102 as acceptance
from scripts.accept_zh_gapfill_20261102 import (
    AcceptanceError,
    SOURCES,
    _append_config,
    _append_sources,
    _assert_old_selection,
    _selected_full_hashes,
    _pool,
    _audit_source,
    accept,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ChineseGapfillAcceptanceTests(unittest.TestCase):
    def test_pool_completion_hash_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            pool = Path(temp)
            marker = pool / "metadata" / "enterprise_zh" / "_COMPLETE.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("{}\n", encoding="utf-8")
            row = {
                "source": "gapfill_cfqa_enterprise_zh",
                "task": "enterprise_zh",
                "pools": [{
                    "capacity": 1000,
                    "path": str(pool),
                    "completion_sha256": "0" * 64,
                }],
            }
            with self.assertRaisesRegex(AcceptanceError, "hash mismatch"):
                _pool(row, 1000)

    def test_old_ready_selection_change_is_rejected(self):
        base_tasks = {
            f"old_{index}": {
                "status": "ready",
                "source_id": f"source_{index}",
                "query_ids": {"dev": [f"d{index}"], "test": [f"t{index}"]},
            }
            for index in range(17)
        }
        base = {"tasks": base_tasks}
        new_tasks = copy.deepcopy(base_tasks)
        new_tasks["old_0"]["query_ids"]["test"] = ["changed"]
        for task_id in SOURCES:
            new_tasks[task_id] = {"status": "ready", "source_id": f"gapfill_{task_id}", "query_ids": {}}
        with self.assertRaisesRegex(AcceptanceError, "selection changed"):
            _assert_old_selection(base, {"tasks": new_tasks})

    def test_accepted_append_prefers_native_sources_and_retains_old_candidates(self):
        old_candidates = {
            "enterprise_zh": [{"source_id": "enterprise_zh_missing", "provenance": "missing"}],
            "software_zh": [{"source_id": "gapfill_dqa_zh", "provenance": "unknown"}],
        }
        config = {"sources": {
            task: {"status": "blocked", "candidates": copy.deepcopy(candidates)}
            for task, candidates in old_candidates.items()
        }}
        closures = {}
        for task_id, (source_id, family) in SOURCES.items():
            closures[source_id] = {
                "catalog_row_path": f"run/{source_id}/catalog-row.json",
                "catalog_row": {
                    "source": source_id,
                    "task": task_id,
                    "status": "complete",
                    "documents": 1000,
                    "queries": 60,
                    "evidence": f"{family.lower()}-evidence",
                    "pools": [
                        {"capacity": 200, "path": f"releases/{source_id}/pools-200"},
                        {"capacity": 1000, "path": f"releases/{source_id}/pools-1000"},
                    ],
                },
                "release": {"path": f"/data/releases/{source_id}"},
            }
        updated_config, changes = _append_config(config, closures)
        updated_catalog = _append_sources(
            {"sources": [{"source": "old", "pools": [], "queries": 1}], "source_count": 1,
             "pool_count": 0, "queries": 1},
            closures,
        )

        self.assertEqual({task for task, value in updated_config["sources"].items() if value["status"] == "eligible"}, set(SOURCES))
        for task_id, (source_id, family) in SOURCES.items():
            candidates = updated_config["sources"][task_id]["candidates"]
            self.assertEqual(candidates[0]["source_id"], source_id)
            self.assertEqual(candidates[0]["provenance"], "native")
            self.assertEqual(candidates[0]["role"], "preferred_native")
            self.assertEqual(candidates[0]["source_family"], family)
            self.assertEqual(candidates[1:], old_candidates[task_id])
            self.assertEqual(changes[task_id]["candidate_count_after"], len(candidates))
        self.assertEqual(updated_catalog["source_count"], 3)
        self.assertEqual(updated_catalog["pool_count"], 4)
        self.assertEqual(
            {row["source"] for row in updated_catalog["sources"]},
            {"old", *(value[0] for value in SOURCES.values())},
        )

    def test_real_pipeline_accepts_launcher_source_row_and_independent_tiers(self):
        """Exercise publish, export, pool publication, and source closure on CPU."""
        from dci_bench.data.document_pools import load_document_pool_manifest

        source_id = "gapfill_cfqa_enterprise_zh"
        task = source_id
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_root = root / "data"
            release = data_root / "releases" / source_id
            run_root = data_root / "run"
            run = run_root / source_id
            run.mkdir(parents=True)
            positive = "original-001000"
            documents = [
                {"_id": f"original-{index:06d}", "title": f"Report {index}",
                 "text": f"Enterprise body {index}.", "language": "zh",
                 "metadata": {"translation_status": "native"}}
                for index in range(1001)
            ]
            queries = [
                {"_id": f"q/{index}", "text": f"企业问题 {index}", "language": "zh",
                 "group_id": f"report-{index}", "metadata": {}}
                for index in range(3)
            ]
            qrels = {row["_id"]: {positive: 1.0} for row in queries}
            bundle = {
                "schema_version": "dci-data-bundle-v1",
                "input_kind": "retrieval",
                "source": {"dataset": "test/cfqa", "revision": "fixture-r1"},
                "documents": documents,
                "queries": queries,
                "qrels": qrels,
                "evidence": [{"query_id": row["_id"], "doc_id": positive,
                              "quote": "Enterprise body 1000."} for row in queries],
                "judgments": [],
                "provenance": {"domain": "enterprise", "language": "zh", "genre": "reports"},
            }
            publish_bundle(bundle, release)
            exported = run_root / "source"
            export_dci(release, exported, task)
            metadata = exported / "metadata" / task
            mapping = json.loads((metadata / "source_doc_ids.json").read_text(encoding="utf-8"))
            original_qrels = json.loads((metadata / "original_qrels.json").read_text(encoding="utf-8"))
            query_rows = [json.loads(line) for line in (metadata / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
            corpus = exported / "workspaces" / task / "corpus"
            corpus_hashes = {
                alias: _sha256(corpus / f"{alias}.md") for alias in sorted(mapping)
            }
            identity = build_source_identity(
                task=task,
                corpus_ids=list(mapping.values()),
                query_ids=list(original_qrels),
                source_doc_ids=mapping,
                qrels=original_qrels,
                query_records=query_rows,
                corpus_file_hashes=corpus_hashes,
            )
            ranking_ids = list(mapping.values())
            ranking = run / "ranking.json"
            ranking.write_text(json.dumps({
                "schema_version": "dci-saved-retrieval-v1",
                "task": task,
                "model": {"path": EXPECTED_EMBEDDING_MODEL_PATH, "trust_remote_code": True},
                "source_identity": identity,
                "rankings": {query_id: ranking_ids for query_id in original_qrels},
                "scores": {query_id: {doc_id: float(len(ranking_ids) - index)
                                      for index, doc_id in enumerate(ranking_ids)}
                           for query_id in original_qrels},
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            lite_root = run / "pool-200"
            full_root = run / "pool-1000"
            # The two source tiers are independently selected.  This fixed
            # seed makes the independent positive replacement visible in the
            # fixture, so source acceptance cannot accidentally require
            # Lite to be a Full subset.
            prepare_document_pools(exported, lite_root, task, capacity=200, seed=15, rankings_path=ranking)
            prepare_document_pools(exported, full_root, task, capacity=1000, seed=15, rankings_path=ranking)
            lite_payload = load_document_pool_manifest(lite_root, task=task, verify_content_hashes=False)
            full_payload = load_document_pool_manifest(full_root, task=task, verify_content_hashes=False)
            self.assertEqual(len(lite_payload["pools"]["q/0"]["candidate_doc_ids"]), 200)
            self.assertEqual(len(full_payload["pools"]["q/0"]["candidate_doc_ids"]), 1000)
            self.assertNotEqual(
                set(lite_payload["pools"]["q/0"]["candidate_doc_ids"])
                - set(full_payload["pools"]["q/0"]["candidate_doc_ids"]),
                set(),
            )
            row = {
                "source": source_id,
                "task": source_id,
                "documents": 1001,
                "queries": 3,
                "content_hashes_verified": True,
                "export": str(exported),
                "ranking": str(ranking),
                "ranking_sha256": _sha256(ranking),
                "pools": [
                    {"capacity": 200, "path": str(lite_root),
                     "completion_sha256": _sha256(lite_root / "metadata" / task / "_COMPLETE.json")},
                    {"capacity": 1000, "path": str(full_root),
                     "completion_sha256": _sha256(full_root / "metadata" / task / "_COMPLETE.json")},
                ],
            }
            (run / "catalog-row.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            closure = _audit_source(data_root, source_id)
            self.assertEqual(closure["task"], source_id)
            self.assertEqual(closure["benchmark_task"], "enterprise_zh")
            self.assertFalse(closure["nested_audit"]["source_pools_nested_checked"])
            self.assertTrue(closure["all_positive_labels_retained"])

    def test_selected_full_hashes_read_each_body_once_and_reject_change(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp) / "suite"
            task_id = "enterprise_zh"
            root = suite / "tasks" / task_id / "full"
            metadata = root / "metadata" / task_id
            corpus = suite / "tasks" / "source" / "workspaces" / task_id / "corpus"
            metadata.mkdir(parents=True)
            corpus.mkdir(parents=True)
            mapping = {"a": "original-a", "b": "original-b"}
            for alias in mapping:
                (corpus / f"{alias}.md").write_text(f"body-{alias}\n", encoding="utf-8")
            inventory = {
                f"workspaces/{task_id}/corpus/{alias}.md": {
                    "alias": alias, "original_id": original,
                    "bytes": (corpus / f"{alias}.md").stat().st_size,
                    "sha256": _sha256(corpus / f"{alias}.md"),
                }
                for alias, original in mapping.items()
            }
            queries = [{"_id": "q0", "text": "q0"}, {"_id": "q1", "text": "q1"}]
            qrels = {"q0": {"a": 1}, "q1": {"b": 1}}
            candidate_pools = {
                "q0": {"candidate_doc_ids": ["a", "b"]},
                "q1": {"candidate_doc_ids": ["b", "a"]},
            }
            for name, value in {
                "manifest.json": {"capacity": 1000},
                "source_doc_ids.json": mapping,
                "doc_id_files.json": {alias: f"{alias}.md" for alias in mapping},
                "file_inventory.json": inventory,
                "original_qrels.json": qrels,
                "qrels.json": qrels,
                "source_identity.json": {},
                "candidate_pools.json": {"pools": candidate_pools},
            }.items():
                (metadata / name).write_text(json.dumps(value) + "\n", encoding="utf-8")
            (metadata / "queries.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in queries), encoding="utf-8"
            )
            (root / "shared_corpus.json").write_text(json.dumps({
                "schema_version": "dci-shared-pool-corpus-v2",
                "task": task_id,
                "corpus": str(corpus),
            }) + "\n", encoding="utf-8")
            manifest = {"tasks": {task_id: {"query_ids": {"dev": ["q0"], "test": ["q1"]}}}}
            with patch("scripts.accept_zh_gapfill_20261102.sha256_file",
                       wraps=acceptance.sha256_file) as digest:
                result = _selected_full_hashes(suite, manifest, task_id)
            self.assertEqual(result["verified_unique_body_count"], 2)
            self.assertEqual(result["query_count"], 2)
            body_calls = [call for call in digest.call_args_list if call.args and Path(call.args[0]).parent == corpus]
            self.assertEqual(len(body_calls), 2)
            (corpus / "b.md").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(AcceptanceError, "body hash mismatch"):
                _selected_full_hashes(suite, manifest, task_id)

    def test_check_only_does_not_create_evidence_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base_suite = root / "base"
            base_suite.mkdir()
            ready = {
                f"old_{index}": {"status": "ready", "source_id": f"source_{index}",
                                 "query_ids": {"dev": [], "test": []}}
                for index in range(17)
            }
            tasks = {**ready, **{
                task: {"status": "blocked", "source_id": None,
                       "query_ids": {"dev": [], "test": []}}
                for task in ("enterprise_zh", "software_zh", "web_en")
            }}
            (base_suite / "manifest.json").write_text(json.dumps({"tasks": tasks}) + "\n", encoding="utf-8")
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"sources": []}) + "\n", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({}) + "\n", encoding="utf-8")
            closures = {
                source_id: {"catalog_row": {"source": source_id, "task": source_id, "pools": []},
                            "release": {"path": f"/data/releases/{source_id}"},
                            "catalog_row_path": f"run/{source_id}/catalog-row.json"}
                for source_id, _ in SOURCES.values()
            }
            base_audit = {"ready_for_formal": False, "ready_tasks": sorted(ready),
                          "blocked_tasks": ["enterprise_zh", "software_zh", "web_en"]}
            evidence = root / "evidence"
            output = root / "output"
            with patch("scripts.accept_zh_gapfill_20261102.audit_benchmark_v2", return_value=base_audit), \
                 patch("scripts.accept_zh_gapfill_20261102._audit_source",
                       side_effect=lambda _data_root, source_id: closures[source_id]), \
                 patch("scripts.accept_zh_gapfill_20261102.load_benchmark_config", return_value={
                     "sources": {task: {"status": "blocked", "candidates": []} for task in SOURCES}
                 }):
                receipt = accept(root / "data", output, evidence, config_path, catalog, base_suite,
                                 check_only=True)
            self.assertTrue(receipt["check_only"])
            self.assertFalse(evidence.exists())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
