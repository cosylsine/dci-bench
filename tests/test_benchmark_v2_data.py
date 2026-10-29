from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dci_bench.data.benchmark_v2 import (
    DEFAULT_SEED,
    FULL_CAPACITY,
    LITE_CAPACITY,
    SamplingShortageError,
    audit_benchmark_v2,
    assert_global_group_leakage,
    assert_no_group_leakage,
    build_sampling_manifest,
    load_benchmark_config,
    prepare_benchmark_v2,
    sample_query_ids,
    sample_global_query_splits,
    sample_query_splits,
    select_nested_pools,
)
from dci_bench.data.document_pools import load_document_pool_manifest, verify_document_pool_suite


class BenchmarkV2SamplingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"query_id": f"q{i:03d}"} for i in range(100)]

    def test_reordered_input_is_deterministic_and_smoke_is_dev_subset(self):
        first = sample_query_splits(self.rows)
        second = sample_query_splits(list(reversed(self.rows)))
        self.assertEqual(first, second)
        self.assertEqual(len(first["dev"]), 10)
        self.assertEqual(len(first["test"]), 50)
        self.assertEqual(len(first["smoke"]), 1)
        self.assertTrue(set(first["smoke"]).issubset(first["dev"]))
        self.assertTrue(set(first["dev"]).isdisjoint(first["test"]))

    def test_seed_change_changes_sample(self):
        first = sample_query_ids(self.rows, count=10, seed=42)
        second = sample_query_ids(self.rows, count=10, seed=43)
        self.assertNotEqual(first, second)

    def test_no_replacement_and_shortage(self):
        self.assertEqual(len(sample_query_ids(self.rows, count=50)), 50)
        with self.assertRaises(SamplingShortageError):
            sample_query_ids(self.rows, count=101)

    def test_group_boundaries_and_global_leakage(self):
        rows = [
            {"query_id": f"q{i:02d}", "group_id": f"g{i // 2}"}
            for i in range(80)
        ]
        splits = sample_query_splits(rows, dev_size=10, test_size=20, group_key="group_id")
        self.assertTrue(assert_no_group_leakage(splits, rows, group_key="group_id") == {})
        task_splits = {"a": {"dev": ["a1"], "test": []}, "b": {"dev": [], "test": ["b1"]}}
        records = {
            "a": [{"query_id": "a1", "parallel_group": "same"}],
            "b": [{"query_id": "b1", "parallel_group": "same"}],
        }
        with self.assertRaises(ValueError):
            assert_global_group_leakage(task_splits, records, group_key="parallel_group")

    def test_global_custom_strategy_controls_dev_and_test_order(self):
        rows = {
            "task": [
                {"query_id": f"q{i:02d}", "text": f"question {i}"}
                for i in range(60)
            ]
        }

        class ReverseStrategy:
            name = "reverse"

            def __init__(self):
                self.calls = []

            def sample(self, query_ids, count, *, seed, task_id):
                self.calls.append(task_id)
                return sorted(query_ids, reverse=True)[:count]

        strategy = ReverseStrategy()
        default = sample_global_query_splits(rows)
        custom = sample_global_query_splits(rows, strategy=strategy)
        self.assertNotEqual(default["task"]["dev"], custom["task"]["dev"])
        self.assertEqual(custom["task"]["dev"], [f"q{i:02d}" for i in range(50, 60)])
        self.assertEqual(custom["task"]["test"], [f"q{i:02d}" for i in range(0, 50)])
        self.assertIn("task:dev", strategy.calls)
        self.assertIn("task:test", strategy.calls)

    def test_normalized_query_text_group_is_global(self):
        records = {
            task_id: [
                {"query_id": f"{task_id}-shared", "text": "  Same query  "},
                *[
                    {"query_id": f"{task_id}-q{i:02d}", "text": f"{task_id} question {i}"}
                    for i in range(60)
                ],
            ]
            for task_id in ("a", "b")
        }
        splits = sample_global_query_splits(records, dev_size=10, test_size=20)
        for split in ("dev", "test"):
            self.assertEqual(
                "a-shared" in splits["a"][split],
                "b-shared" in splits["b"][split],
            )


class BenchmarkV2PoolTests(unittest.TestCase):
    def test_nested_full_and_lite_retain_qrels(self):
        ranking = [f"d{i:04d}" for i in range(1200)]
        qrels = {"d0000": 1, "d0999": 2, "d1100": 1}
        result = select_nested_pools(ranking, qrels)
        self.assertEqual(len(result["full"]), FULL_CAPACITY)
        self.assertEqual(len(result["lite"]), LITE_CAPACITY)
        self.assertTrue(set(result["lite"]).issubset(result["full"]))
        self.assertTrue(set(qrels).issubset(result["full"]))
        self.assertTrue(set(qrels).issubset(result["lite"]))

    def test_positive_overflow_is_counted_and_lite_is_excluded(self):
        ranking = [f"d{i:04d}" for i in range(1000)]
        qrels = {doc_id: 1 for doc_id in ranking[:201]}
        result = select_nested_pools(ranking, qrels)
        self.assertIsNone(result["lite"])
        self.assertEqual(result["audit"]["positive_overflow_lite"], 1)
        self.assertEqual(result["audit"]["positive_count"], 201)
        self.assertEqual(len(result["full"]), 1000)

    def test_full_positive_overflow_is_explicit(self):
        ranking = [f"d{i:04d}" for i in range(1001)]
        result = select_nested_pools(ranking, ranking, full_capacity=1000)
        self.assertIsNone(result["full"])
        self.assertTrue(result["audit"]["full_excluded"])


class BenchmarkV2ManifestTests(unittest.TestCase):
    def test_manifest_records_identity(self):
        manifest = build_sampling_manifest(
            {"encyclopedia_en": {"status": "ready", "splits": {"dev": ["q1"], "test": [], "smoke": ["q1"]}}},
            input_identity={"rows": ["q1"]},
        )
        self.assertEqual(manifest["sampling"]["seed"], DEFAULT_SEED)
        self.assertTrue(manifest["sampling"]["input_digest"])

    def test_config_declares_all_twenty_tasks_and_scope(self):
        config = load_benchmark_config()
        self.assertEqual(len(config["sources"]), 20)
        self.assertEqual(config["tiers"]["lite"]["capacity"], LITE_CAPACITY)
        self.assertEqual(config["tiers"]["full"]["capacity"], FULL_CAPACITY)
        self.assertTrue(config["provenance_policy"]["unknown_requires_explicit_override_with_evidence"])


class BenchmarkV2PrepareAuditTests(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _build_source_fixture(self, root: Path) -> tuple[Path, Path]:
        task_root = root / "fixture-source"
        task = "fixture"
        corpus = task_root / "source" / "workspaces" / task / "corpus"
        corpus.mkdir(parents=True)
        mapping = {f"d{i:06d}": f"original-{i:06d}" for i in range(1, 1001)}
        doc_id_files = {alias: f"{alias}.md" for alias in mapping}
        digest = hashlib.sha256(b"fixture document\n").hexdigest()
        inventory = {
            f"workspaces/{task}/corpus/{alias}.md": {
                "alias": alias,
                "original_id": original,
                "bytes": len(b"fixture document\n"),
                "sha256": digest,
            }
            for alias, original in mapping.items()
        }
        for alias in mapping:
            (corpus / f"{alias}.md").write_bytes(b"fixture document\n")

        queries = [
            {"query_id": "q000", "text": "duplicate query"},
            {"query_id": "q001", "text": "  Duplicate   query "},
        ]
        queries.extend(
            {"query_id": f"q{i:03d}", "text": f"question {i}"}
            for i in range(2, 61)
        )
        queries.extend(
            [
                {"query_id": "q061", "text": "overflow query"},
                {"query_id": "q062", "text": "no positive query"},
                {"query_id": "q063"},
                {"query_id": "q064", "text": "conflicting query"},
                {"query_id": "q065", "text": " conflicting   query "},
            ]
        )
        qrels = {
            row["query_id"]: {"d000001": 1}
            for row in queries
        }
        qrels["q061"] = {f"d{i:06d}": 1 for i in range(1, 202)}
        qrels["q062"] = {}
        qrels["q065"] = {"d000002": 1}
        original_qrels = {
            query_id: {
                mapping[alias]: score
                for alias, score in labels.items()
            }
            for query_id, labels in qrels.items()
        }
        queries_jsonl = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in queries
        )
        source_identity = {
            "queries_hash": "q" * 64,
            "corpus_files_sha256": "c" * 64,
            "source_doc_ids_hash": "d" * 64,
        }
        marker = {
            "schema_version": "dci-shared-pool-corpus-v1",
            "task": task,
            "corpus": f"source/workspaces/{task}/corpus",
        }
        candidate_records = {}
        ranking = list(mapping)
        scores = [float(len(ranking) - i) for i in range(len(ranking))]
        for row in queries:
            query_id = row["query_id"]
            candidate_records[query_id] = {
                "query_id": query_id,
                "candidate_doc_ids": ranking,
                "candidate_original_doc_ids": [mapping[alias] for alias in ranking],
                "candidate_scores": scores,
                "ordering": "scored",
                "num_docs": len(ranking),
                "audit": {
                    "query_id": query_id,
                    "selected_ids": [mapping[alias] for alias in ranking],
                    "selected_scores": scores,
                    "ordering": "scored",
                    "capacity": FULL_CAPACITY,
                    "source_size": len(mapping),
                },
            }
        source_manifest_base = {
            "schema_version": "dci-document-pools-20261029-v2",
            "task": task,
            "dataset": "fixture",
            "revision": "r1",
            "resolved_revision": "r1",
            "task_instruction": "",
            "document_id_policy": "short-document-id-v1",
            "source_manifest_sha256": "a" * 64,
            "source_inventory_sha256": "b" * 64,
        }
        for capacity, pool_size in ((FULL_CAPACITY, "full"), (LITE_CAPACITY, "lite")):
            pool_root = task_root / "pools" / str(capacity)
            metadata = pool_root / "metadata" / task
            metadata.mkdir(parents=True)
            manifest = {
                **source_manifest_base,
                "capacity": capacity,
                "pool_size": pool_size,
            }
            self._write_json(metadata / "manifest.json", manifest)
            self._write_json(metadata / "source_identity.json", source_identity)
            self._write_json(metadata / "source_doc_ids.json", mapping)
            self._write_json(metadata / "doc_id_files.json", doc_id_files)
            self._write_json(metadata / "file_inventory.json", inventory)
            self._write_json(metadata / "qrels.json", qrels)
            self._write_json(metadata / "original_qrels.json", original_qrels)
            (metadata / "queries.jsonl").write_text(queries_jsonl, encoding="utf-8")
            if capacity == FULL_CAPACITY:
                self._write_json(
                    metadata / "candidate_pools.json",
                    {
                        "schema_version": "dci-document-pools-20261029-v2",
                        "task": task,
                        "pool_size": pool_size,
                        "capacity": capacity,
                        "seed": DEFAULT_SEED,
                        "source_identity": source_identity,
                        "source_release": {
                            "source_manifest_sha256": manifest["source_manifest_sha256"],
                            "source_inventory_sha256": manifest["source_inventory_sha256"],
                        },
                        "shared_corpus": "corpus",
                        "num_queries": len(queries),
                        "corpus_size": len(mapping),
                        "pools": candidate_records,
                    },
                )
            self._write_json(pool_root / "shared_corpus.json", marker)
        catalog = {
            "sources": [
                {
                    "source": "fixture_source",
                    "task": task,
                    "documents": len(mapping),
                    "queries": len(queries),
                    "pools": [
                        {"capacity": FULL_CAPACITY, "path": str(task_root / "pools" / "1000")},
                        {"capacity": LITE_CAPACITY, "path": str(task_root / "pools" / "200")},
                    ],
                }
            ]
        }
        catalog_path = root / "catalog.json"
        self._write_json(catalog_path, catalog)
        config = load_benchmark_config()
        config["sources"]["encyclopedia_en"] = {
            "status": "eligible",
            "candidates": [
                {
                    "source_id": "fixture_source",
                    "source_family": "fixture",
                    "provenance": "native",
                    "evidence": "test fixture",
                }
            ],
        }
        config_path = root / "config.json"
        self._write_json(config_path, config)
        return catalog_path, config_path

    def test_prepare_audit_filters_duplicates_overflow_and_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog_path, config_path = self._build_source_fixture(root)
            output = root / "benchmark-v2"
            manifest = prepare_benchmark_v2(
                output_dir=output,
                config_path=config_path,
                catalog_path=catalog_path,
            )
            task = manifest["tasks"]["encyclopedia_en"]
            self.assertEqual(manifest["status"], "partial_blocked")
            self.assertFalse(manifest["ready_for_formal"])
            self.assertEqual(task["status"], "ready")
            self.assertEqual(len(task["query_ids"]["dev"]), 10)
            self.assertEqual(len(task["query_ids"]["test"]), 50)
            self.assertEqual(len(task["query_ids"]["smoke"]), 1)
            self.assertEqual(task["overflow_excluded_count"], 1)
            self.assertGreaterEqual(task["dedup_excluded_count"], 1)
            exclusion_reasons = {row["reason"] for row in task["exclusions"]}
            self.assertTrue(
                {"duplicate_query_conflicting_qrels", "no_positive_labels", "missing_query_text"}
                <= exclusion_reasons
            )
            self.assertNotIn("q001", set(task["query_ids"]["dev"]) | set(task["query_ids"]["test"]))
            self.assertEqual(task["pools"]["lite"]["suite"], "tasks/encyclopedia_en/lite")
            self.assertEqual(task["pools"]["full"]["suite"], "tasks/encyclopedia_en/full")
            self.assertEqual(audit_benchmark_v2(output)["ready_for_formal"], False)

            full_metadata = output / "tasks" / "encyclopedia_en" / "full" / "metadata" / "encyclopedia_en"
            candidate_path = full_metadata / "candidate_pools.json"
            candidate_bytes = candidate_path.read_bytes()
            candidate_payload = json.loads(candidate_bytes)
            candidate_payload["num_queries"] += 1
            candidate_path.write_text(
                json.dumps(candidate_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "metadata hash mismatch"):
                load_document_pool_manifest(
                    full_metadata.parent.parent,
                    task="encyclopedia_en",
                    verify_content_hashes=False,
                )
            candidate_path.write_bytes(candidate_bytes)

            source_corpus = (
                root / "fixture-source" / "source" / "workspaces" / "fixture" / "corpus"
            ).resolve()
            original_stat = Path.stat

            def reject_document_stat(path, *args, **kwargs):
                if path != source_corpus and source_corpus in path.parents:
                    raise AssertionError(f"metadata-only audit statted source body: {path}")
                return original_stat(path, *args, **kwargs)

            with patch.object(Path, "stat", new=reject_document_stat):
                verification = verify_document_pool_suite(
                    full_metadata.parent.parent,
                    task="encyclopedia_en",
                    verify_content_hashes=False,
                )
            self.assertTrue(verification["metadata_hashes_verified"])
            self.assertFalse(verification["content_hashes_verified"])
            self.assertIn("verified at runtime", verification["content_verification_boundary"])

            manifest_path = output / "manifest.json"
            tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
            tampered["overall_sha256"] = "0" * 64
            manifest_path.write_text(
                json.dumps(tampered, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                audit_benchmark_v2(output)


if __name__ == "__main__":
    unittest.main()
