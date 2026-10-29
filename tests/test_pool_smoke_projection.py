"""Bounded tests for the single-query, three-tier pool projection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import tarfile
import unittest
import subprocess
import sys

from dci_bench.data.document_pools import (
    POOL_SCHEMA_VERSION,
    sha256_file,
    stable_json_sha256,
    verify_document_pool_suite,
)
from scripts.prepare_pool_smoke import CAPACITIES, prepare_pool_smoke, _verify_migrated_completion


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PoolSmokeProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog = self._make_fixture()

    def test_migration_authenticates_legacy_seal_and_rejects_changed_labels(self):
        task = 'fixture_pool_smoke'
        suite = self.root / 'tasks' / task / 'pools/200'
        metadata = suite / 'metadata' / task
        archive = suite.parent.parent / 'provenance/pool-200-controls.tar.gz'
        archive.parent.mkdir()
        legacy_sha = _sha(metadata / '_COMPLETE.json')
        with tarfile.open(archive, 'w:gz') as handle:
            for name in ['_COMPLETE.json', 'manifest.json']:
                handle.add(metadata / name, arcname=f'{task}/{name}')
        current = json.loads((metadata / '_COMPLETE.json').read_text())
        manifest = json.loads((metadata / 'manifest.json').read_text())
        manifest['rankings_path'] = str(suite.parent.parent / 'ranking.json.gz')
        _write_json(metadata / 'manifest.json', manifest)
        current['metadata_hashes']['manifest.json'] = _sha(metadata / 'manifest.json')
        proof = _verify_migrated_completion(suite, task, 200, legacy_sha, current)
        self.assertTrue(proof['migration_fields_verified'])
        current['metadata_hashes']['qrels.json'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'protected metadata'):
            _verify_migrated_completion(suite, task, 200, legacy_sha, current)

    def _make_fixture(self) -> Path:
        task = "fixture_pool_smoke"
        tasks = self.root / "tasks" / task
        export = tasks / "source"
        source_metadata = export / "metadata" / task
        source_corpus = export / "workspaces" / task / "corpus"
        source_corpus.mkdir(parents=True)
        mapping = {"d000001": "original-1", "d000002": "original-2", "d000003": "original-3"}
        doc_id_files = {alias: f"{alias}.md" for alias in mapping}
        bodies = {
            alias: f"Fixture document body for {original}.\n".encode()
            for alias, original in mapping.items()
        }
        for alias, body in bodies.items():
            (source_corpus / doc_id_files[alias]).write_bytes(body)
        query = {"_id": "q-fixture", "text": "Which fixture document is relevant?"}
        original_qrels = {"q-fixture": {"original-2": 1.0}}
        qrels = {"q-fixture": {"d000002": 1.0}}
        query_bytes = (json.dumps(query, ensure_ascii=False, sort_keys=True) + "\n").encode()
        source_metadata.mkdir(parents=True)
        _write_json(source_metadata / "doc_id_files.json", doc_id_files)
        _write_json(source_metadata / "source_doc_ids.json", mapping)
        _write_json(source_metadata / "original_qrels.json", original_qrels)
        _write_json(source_metadata / "qrels.json", qrels)
        (source_metadata / "queries.jsonl").write_bytes(query_bytes)
        inventory_rows = [
            {
                "doc_id": original,
                "filename": doc_id_files[alias],
                "bytes": len(bodies[alias]),
                "sha256": _sha(source_corpus / doc_id_files[alias]),
            }
            for alias, original in mapping.items()
        ]
        (source_metadata / "corpus_files.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in inventory_rows)
        )
        _write_json(source_metadata / "audit.json", {"document_exclusions": [], "query_exclusions": []})
        _write_json(source_metadata / "host_labels.json", {})
        _write_json(source_metadata / "provenance.json", {})
        _write_json(source_metadata / "source_manifest.json", {"source": "fixture"})
        source_manifest = {
            "task": task,
            "dataset": "fixture/dataset",
            "revision": "fixture-revision",
            "resolved_revision": "fixture-revision",
            "task_instruction": "",
            "document_id_policy": "short-document-id-v1",
            "num_docs": 3,
            "num_queries": 1,
            "num_qrels": 1,
            "qrels_hash": stable_json_sha256(qrels),
            "queries_hash": stable_json_sha256([query]),
            "source_doc_ids_hash": stable_json_sha256(mapping),
            "original_qrels_hash": stable_json_sha256(original_qrels),
        }
        _write_json(source_metadata / "manifest.json", source_manifest)
        source_metadata_hashes = {
            name: _sha(source_metadata / name)
            for name in (
                "audit.json",
                "corpus_files.jsonl",
                "doc_id_files.json",
                "host_labels.json",
                "manifest.json",
                "original_qrels.json",
                "provenance.json",
                "qrels.json",
                "queries.jsonl",
                "source_doc_ids.json",
                "source_manifest.json",
            )
        }
        _write_json(
            source_metadata / "_COMPLETE.json",
            {"schema_version": "fixture-source-v1", "metadata_hashes": source_metadata_hashes},
        )

        corpus_file_hashes = {
            alias: _sha(source_corpus / doc_id_files[alias]) for alias in mapping
        }
        source_identity = {
            "corpus_files_sha256": stable_json_sha256(corpus_file_hashes),
            "queries_hash": stable_json_sha256([query]),
            "source_doc_ids_hash": stable_json_sha256(mapping),
            "source_manifest_sha256": _sha(source_metadata / "manifest.json"),
            "source_inventory_sha256": _sha(source_metadata / "corpus_files.jsonl"),
        }

        pool_entries = []
        for capacity in CAPACITIES:
            pool_dir = tasks / "pools" / str(capacity)
            pool_metadata = pool_dir / "metadata" / task
            pool_metadata.mkdir(parents=True)
            marker = {
                "schema_version": "dci-shared-pool-corpus-v1",
                "task": task,
                "corpus": f"source/workspaces/{task}/corpus",
            }
            _write_json(pool_dir / "shared_corpus.json", marker)
            _write_json(pool_metadata / "source_doc_ids.json", mapping)
            _write_json(pool_metadata / "doc_id_files.json", doc_id_files)
            _write_json(pool_metadata / "source_identity.json", source_identity)
            (pool_metadata / "queries.jsonl").write_bytes(query_bytes)
            _write_json(pool_metadata / "original_qrels.json", original_qrels)
            _write_json(pool_metadata / "qrels.json", qrels)
            inventory = {
                f"workspaces/{task}/corpus/{doc_id_files[alias]}": {
                    "alias": alias,
                    "original_id": mapping[alias],
                    "bytes": len(bodies[alias]),
                    "sha256": corpus_file_hashes[alias],
                }
                for alias in mapping
            }
            _write_json(pool_metadata / "file_inventory.json", inventory)
            manifest = {
                "schema_version": POOL_SCHEMA_VERSION,
                "task": task,
                "dataset": source_manifest["dataset"],
                "revision": source_manifest["revision"],
                "resolved_revision": source_manifest["resolved_revision"],
                "task_instruction": "",
                "document_id_policy": "short-document-id-v1",
                "pool_size": {200: "small", 500: "medium", 1000: "large"}[capacity],
                "capacity": capacity,
                "seed": 17,
                "num_queries": 1,
                "corpus_size": 3,
                "qrels_hash": stable_json_sha256(qrels),
                "queries_hash": stable_json_sha256([query]),
                "source_doc_ids_hash": stable_json_sha256(mapping),
                "original_qrels_hash": stable_json_sha256(original_qrels),
                "source_manifest_sha256": _sha(source_metadata / "manifest.json"),
                "source_inventory_sha256": _sha(source_metadata / "corpus_files.jsonl"),
                "source_identity": source_identity,
                "metadata_dir": f"metadata/{task}",
                "workspace_dir": f"workspaces/{task}",
                "shared_corpus_dir": f"workspaces/{task}/corpus",
                "verification_policy": "declared-source-digests",
                "content_hashes_verified": False,
            }
            _write_json(pool_metadata / "manifest.json", manifest)
            record = {
                "query_id": "q-fixture",
                "candidate_doc_ids": ["d000001", "d000002", "d000003"],
                "candidate_scores": [0.9, 0.8, 0.7],
                "ordering": "scored",
                "num_docs": 3,
                "audit": {
                    "selection_version": POOL_SCHEMA_VERSION,
                    "task": task,
                    "query_id": "q-fixture",
                    "seed": 17,
                    "rng_scope": [17, task, "q-fixture"],
                    "rng_scoped_seed_sha256": "a" * 64,
                    "rng_algorithm": "python-random-sample-v1",
                    "capacity": capacity,
                    "source_size": 3,
                    "initial_top_ids": ["d000001", "d000003", "d000002"],
                    "injected_positive_ids": [],
                    "removed_nonpositive_ids": [],
                    "injection_count": 0,
                    "eviction_count": 0,
                    "ordering": "scored",
                    "positive_count": 1,
                    "ranking_required": True,
                    "ranking_complete": True,
                },
            }
            _write_json(
                pool_metadata / "candidate_pools.json",
                {
                    "capacity": capacity,
                    "corpus_size": 3,
                    "num_queries": 1,
                    "pool_size": manifest["pool_size"],
                    "pools": {"q-fixture": record},
                    "schema_version": POOL_SCHEMA_VERSION,
                    "seed": 17,
                    "shared_corpus": "corpus",
                    "source_identity": source_identity,
                    "source_release": {
                        "source_manifest_sha256": manifest["source_manifest_sha256"],
                        "source_inventory_sha256": manifest["source_inventory_sha256"],
                    },
                    "task": task,
                    "storage_encoding": "short-id-derived-audit-v1",
                },
            )
            pool_metadata_hashes = {
                name: _sha(pool_metadata / name)
                for name in (
                    "candidate_pools.json",
                    "doc_id_files.json",
                    "file_inventory.json",
                    "manifest.json",
                    "original_qrels.json",
                    "qrels.json",
                    "queries.jsonl",
                    "source_doc_ids.json",
                    "source_identity.json",
                )
            }
            workspace_hashes = {key: row["sha256"] for key, row in inventory.items()}
            _write_json(
                pool_metadata / "_COMPLETE.json",
                {
                    "schema_version": POOL_SCHEMA_VERSION,
                    "task": task,
                    "metadata_hashes": pool_metadata_hashes,
                    "workspace_hashes": workspace_hashes,
                    "file_inventory_sha256": _sha(pool_metadata / "file_inventory.json"),
                    "shared_corpus_sha256": _sha(pool_dir / "shared_corpus.json"),
                    "verification_policy": "declared-source-digests",
                    "content_hashes_verified": False,
                },
            )
            pool_entries.append(
                {
                    "capacity": capacity,
                    "completion_sha256": _sha(pool_metadata / "_COMPLETE.json"),
                    "path": str(pool_dir.resolve()),
                    "num_workspace_files": 3,
                }
            )

        catalog = {
            "schema_version": "dci-consolidated-pools-v1",
            "sources": [
                {
                    "source": task,
                    "task": task,
                    "documents": 3,
                    "queries": 1,
                    "export": str(export.resolve()),
                    "pools": pool_entries,
                    "source_files_present": 3,
                    "content_hashes_verified": False,
                }
            ],
            "source_count": 1,
            "pool_count": 3,
            "queries": 1,
        }
        catalog_path = self.root / "catalog.json"
        _write_json(catalog_path, catalog)
        _write_json(
            self.root / "_COMPLETE.json",
            {
                "catalog_sha256": _sha(catalog_path),
                "source_count": 1,
                "pool_count": 3,
                "content_hashes_verified": False,
            },
        )
        return catalog_path

    def _prepare(self) -> Path:
        output = self.root / "projection"
        prepare_pool_smoke(self.catalog, output, seed=20261026, expected_sources=1)
        return output

    def test_projection_is_strict_and_keeps_one_query_for_all_tiers(self) -> None:
        output = self._prepare()
        plan = json.loads((output / "plan.json").read_text())
        self.assertEqual((plan["sources"], plan["evaluations"], plan["unique_queries"]), (1, 3, 1))
        self.assertEqual({row["query_id"] for row in plan["rows"]}, {"q-fixture"})
        self.assertEqual({row["capacity"] for row in plan["rows"]}, set(CAPACITIES))
        for capacity in CAPACITIES:
            suite = output / "tasks" / "fixture_pool_smoke" / "pools" / str(capacity)
            result = verify_document_pool_suite(suite, task="fixture_pool_smoke", verify_content_hashes=True)
            self.assertTrue(result["content_hashes_verified"])
            self.assertEqual(result["num_queries"], 1)
        corpus = output / "tasks" / "fixture_pool_smoke" / "source" / "workspaces" / "fixture_pool_smoke" / "corpus"
        self.assertEqual(len(list(corpus.iterdir())), 3)

    def test_rejects_wrong_source_metadata_hash_and_leaves_failed_record(self) -> None:
        source_query = self.root / "tasks" / "fixture_pool_smoke" / "source" / "metadata" / "fixture_pool_smoke" / "queries.jsonl"
        source_query.write_text('{"_id":"q-fixture","text":"tampered"}\n')
        output = self.root / "rejected-metadata"
        with self.assertRaisesRegex(ValueError, "metadata SHA-256 mismatch"):
            prepare_pool_smoke(self.catalog, output, expected_sources=1)
        failed = json.loads((output / "_FAILED.json").read_text())
        self.assertEqual(failed["status"], "failed")

    def test_preflight_validates_publication_schema_for_each_tier(self) -> None:
        output = self._prepare()
        env = dict(os.environ)
        env.pop('DCI_PROTOCOL_PROFILE', None)
        runner = Path(__file__).resolve().parents[1] / 'scripts/run_document_pool.py'
        for row in json.loads((output / 'plan.json').read_text())['rows']:
            suite = Path(row['suite'])
            result = subprocess.run([
                sys.executable, '-B', str(runner), '--task', row['task'],
                '--metadata-root', str(suite / 'metadata'), '--workspace-root', str(suite / 'workspaces'),
                '--query-ids', row['query_id'], '--model-key', 'cpu-probe', '--preflight-only',
            ], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(result.stdout)
            self.assertTrue(record['run_manifest_data_validated'])
            self.assertEqual(record['model_calls'], 0)

    def test_rejects_modified_selected_body_using_inventory_hash(self) -> None:
        body = self.root / "tasks" / "fixture_pool_smoke" / "source" / "workspaces" / "fixture_pool_smoke" / "corpus" / "d000002.md"
        body.write_text("changed body\n")
        output = self.root / "rejected-body"
        with self.assertRaisesRegex(ValueError, "selected document copy SHA-256 mismatch"):
            prepare_pool_smoke(self.catalog, output, expected_sources=1)
        self.assertTrue((output / "_FAILED.json").is_file())

    def test_output_is_exclusive_and_source_candidate_bytes_stay_unchanged(self) -> None:
        source_candidate = self.root / "tasks" / "fixture_pool_smoke" / "pools" / "200" / "metadata" / "fixture_pool_smoke" / "candidate_pools.json"
        before_candidate = source_candidate.read_bytes()
        output = self._prepare()
        self.assertEqual(source_candidate.read_bytes(), before_candidate)
        with self.assertRaises(FileExistsError):
            prepare_pool_smoke(self.catalog, output, expected_sources=1)
        self.assertEqual(source_candidate.read_bytes(), before_candidate)


if __name__ == "__main__":
    unittest.main()
