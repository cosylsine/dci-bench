"""Synthetic integration: short IDs -> query pools -> task routing and scoring."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace

import anyio
from unittest.mock import patch

from dci_bench.data.expansion import build_document_task
from dci_bench.data.document_pools import prepare_document_pools
from dci_bench.data.document_runtime import audit_document_ids, load_pool_for_runtime, materialize_query_workspace
from dci_bench.data.document_ids import restore_ranking
from dci_bench.protocol.document_pools import get_tier
from dci_bench.scoring.retrieval import score_query

ROOT = Path(__file__).resolve().parents[1]


class DocumentPoolRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.output = self.root / "pools"
        self.task = "pool_fixture"
        self.originals = ["a" * 64, "folder/文档.md", "d000001"]
        self.labels = {"q/one": {self.originals[0]: 2.0},
                       "q/two": {self.originals[1]: 1.0, self.originals[2]: 3.0}}
        build_document_task({"name": self.task, "dataset": "fixture/document-pools",
            "revision": "b" * 40, "corpus": [
                {"_id": doc, "title": "", "text": f"Candidate body {i}."}
                for i, doc in enumerate(self.originals)],
            "queries": [{"_id": qid, "text": "Find useful text"} for qid in self.labels],
            "qrels": self.labels, "host_labels": {}, "provenance": {"exclusions": []}},
            self.source, {"source_fingerprint": "c" * 64, "raw_directory": "/synthetic"}, io_workers=1)
        prepare_document_pools(self.source, self.output, self.task, pool_size="small", seed=42)
        self.md = self.output / "metadata" / self.task

    def test_route_each_query_to_its_own_verified_corpus_without_host_labels(self):
        from dci_bench.tasks.mteb_llm_retrieval import _make_samples

        with patch("dci_bench.tasks.mteb_llm_retrieval.protocol_contract",
                   return_value={"candidate_pool": get_tier("small")}):
            samples = _make_samples(self.task, None, self.output / "metadata",
                                    self.output / "workspaces", all_queries=True)
        self.assertEqual(len(samples), 2)
        shared = self.output / "workspaces" / self.task
        pool = load_pool_for_runtime(self.output / "metadata", self.output / "workspaces",
                                     self.task, {"candidate_pool": get_tier("small")})
        self.assertEqual(len(list(shared.rglob("*.md"))), 3)
        self.assertFalse((shared / "queries").exists())
        for sample in samples:
            self.assertEqual(Path(sample.metadata["workspace"]), shared)
            with materialize_query_workspace(shared, sample.id, pool) as workspace:
                self.assertNotEqual(workspace, shared)
                self.assertEqual({p.name for p in workspace.iterdir()}, {"corpus"})
                self.assertEqual({p.name for p in (workspace / "corpus").iterdir()},
                                 {"d000001.md", "d000002.md", "d000003.md"})
                for document in (workspace / "corpus").iterdir():
                    self.assertEqual(document.stat().st_nlink, 1)
            self.assertFalse(workspace.exists())
            self.assertEqual(sample.target, "")
            self.assertEqual(set(sample.metadata), {"task", "query_id", "query_index", "workspace"})

    def test_runtime_cleans_temporary_workspace_after_failure(self):
        pool = load_pool_for_runtime(self.output / "metadata", self.output / "workspaces",
                                     self.task, {"candidate_pool": get_tier("small")})
        shared = self.output / "workspaces" / self.task
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with materialize_query_workspace(shared, "q/one", pool) as workspace:
                raise RuntimeError("synthetic failure")
        self.assertFalse(workspace.exists())
        self.assertEqual(len(list(shared.rglob("*.md"))), 3)
        # A mutation after preflight must fail before Pi is called as well.
        (shared / "corpus/d000001.md").write_text("Changed after verification")
        with self.assertRaisesRegex(ValueError, "changed since preflight"):
            with materialize_query_workspace(shared, "q/one", pool):
                self.fail("must not expose a changed document")

    def test_solver_passes_temporary_selected_corpus_to_pi_and_cleans_on_exception(self):
        from dci_bench.tasks.mteb_llm_retrieval import pi_dci_solver

        pool = load_pool_for_runtime(self.output / "metadata", self.output / "workspaces",
                                     self.task, {"candidate_pool": get_tier("small")})
        shared = self.output / "workspaces" / self.task
        # A smaller in-memory selection isolates the runtime boundary from
        # pool construction (the builder's capacity/positive tests are separate).
        selected = pool["pools"]["q/one"]["candidate_doc_ids"][:1]
        pool["pools"]["q/one"]["candidate_doc_ids"] = selected
        observed = {}

        @asynccontextmanager
        async def bridge():
            yield object()

        def fake_run(**kwargs):
            workspace = kwargs["workspace"]
            observed["workspace"] = workspace
            self.assertNotEqual(workspace, shared)
            self.assertEqual({p.name for p in workspace.iterdir()}, {"corpus"})
            self.assertEqual({p.stem for p in (workspace / "corpus").iterdir()}, set(selected))
            raise RuntimeError("synthetic Pi failure")

        async def exercise():
            state = SimpleNamespace(metadata={"task": self.task, "query_id": "q/one",
                "workspace": str(shared)}, input_text="query", output=SimpleNamespace())
            solve = pi_dci_solver(output_dir=str(self.root / "results"), candidate_pool=pool)
            with patch("dci_bench.tasks.mteb_llm_retrieval.sandbox_agent_bridge", return_value=bridge()), \
                 patch("dci_bench.tasks.mteb_llm_retrieval._remaining_pi_timeout_seconds", return_value=60), \
                 patch("dci_bench.tasks.mteb_llm_retrieval.run_pi_dci", side_effect=fake_run):
                with self.assertRaisesRegex(RuntimeError, "synthetic Pi failure"):
                    await solve(state, None)

        anyio.run(exercise)
        self.assertFalse(observed["workspace"].exists())
        self.assertEqual(len(list(shared.rglob("*.md"))), 3)

    def test_runtime_rejects_wrong_tier_and_mutated_document_before_model_call(self):
        roots = (self.output / "metadata", self.output / "workspaces", self.task)
        with self.assertRaisesRegex(ValueError, "profile"):
            load_pool_for_runtime(*roots, {})
        with self.assertRaisesRegex(ValueError, "tier/capacity"):
            load_pool_for_runtime(*roots, {"candidate_pool": get_tier("large")})
        document = next((self.output / "workspaces").rglob("*.md"))
        document.write_text("Changed body", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_pool_for_runtime(*roots, {"candidate_pool": get_tier("small")})

    def test_full_corpus_cannot_use_pool_budget_without_pool_preparation(self):
        with self.assertRaisesRegex(ValueError, "prepared per-query"):
            load_pool_for_runtime(self.source / "metadata", self.source / "workspaces",
                                  self.task, {"candidate_pool": get_tier("small")})

    def test_qrels_restore_preserves_graded_metrics_and_detects_mapping_tamper(self):
        manifest = json.loads((self.md / "manifest.json").read_text())
        audit_document_ids(self.md, manifest)
        aliases = json.loads((self.md / "source_doc_ids.json").read_text())
        qrels = json.loads((self.md / "qrels.json").read_text())
        ranking = list(reversed(sorted(aliases)))
        for qid in qrels:
            self.assertEqual(score_query(ranking, qrels[qid]),
                             score_query(restore_ranking(ranking, aliases), self.labels[qid]))
        aliases["d000001"] = "unrelated-source"
        (self.md / "source_doc_ids.json").write_text(json.dumps(aliases))
        with self.assertRaises(ValueError):
            audit_document_ids(self.md, manifest)

    def test_fresh_process_preflight_selects_profile_before_contract_import(self):
        env = os.environ.copy()
        env.pop("DCI_PROTOCOL_PROFILE", None)
        env.pop("DCI_PROTOCOL_CONTRACT_SHA256", None)
        process = subprocess.run([sys.executable, str(ROOT / "scripts/run_document_pool.py"),
            "--task", self.task, "--metadata-root", str(self.output / "metadata"),
            "--workspace-root", str(self.output / "workspaces"), "--all-queries", "--preflight-only"],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["profile"], "document-pool-v1-small")
        self.assertEqual(result["max_total_model_tokens"], 96000)
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse(result["sandbox_verified"])

    @unittest.skipUnless(shutil.which("node") and shutil.which("bwrap"), "Node and Bubblewrap required")
    def test_real_pi_mock_reads_short_id_and_scores_in_original_namespace(self):
        pool = load_pool_for_runtime(self.output / "metadata", self.output / "workspaces",
                                     self.task, {"candidate_pool": get_tier("small")})
        shared = self.output / "workspaces" / self.task
        qrels = json.loads((self.md / "qrels.json").read_text())
        alias = next(iter(qrels["q/one"]))
        mock = self.root / "mock.json"
        mock.write_text(json.dumps([
            {"type": "tool_call", "name": "read", "arguments": {"path": alias + ".md"},
             "id": "read_one", "usage": {"input": 100, "output": 10}},
            {"type": "final", "text": json.dumps({"ranked_doc_ids": [alias]}),
             "usage": {"input": 100, "output": 10}}]))
        env = dict(os.environ, DCI_PROTOCOL_PROFILE="document-pool-v1-small")
        env.pop("DCI_PROTOCOL_CONTRACT_SHA256", None)
        code = """import json,sys
from pathlib import Path
from dci_bench.agents.pi_agent import run_pi_dci
result = run_pi_dci(query='Find useful text', workspace=Path(sys.argv[1]),
    output_path=Path(sys.argv[2]), mock_script=Path(sys.argv[3]), context_window=65536)
print(json.dumps({'valid':result.valid_output, 'ids':result.ranked_doc_ids,
                  'failure':result.failure_reason}))
"""
        with materialize_query_workspace(shared, "q/one", pool) as workspace:
            process = subprocess.run([sys.executable, "-c", code, str(workspace),
                str(self.root / "final.json"), str(mock)], cwd=ROOT, env=env,
                capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["ids"], [alias])
        self.assertEqual(score_query(result["ids"], qrels["q/one"])["recall_at_10"], 1.0)


if __name__ == "__main__":
    unittest.main()

class SharedCorpusRuntimeTests(unittest.TestCase):
    """Run the existing isolation checks against the compact shared layout."""
    def setUp(self):
        DocumentPoolRuntimeTests.setUp(self)
        compact = self.root / "compact"
        compact.mkdir()
        shutil.move(str(self.source), compact / "source")
        self.source = compact / "source"
        destination = compact / "pools" / "200"
        destination.parent.mkdir()
        shutil.move(str(self.output), destination)
        self.output = destination
        self.md = self.output / "metadata" / self.task
        shutil.rmtree(self.output / "workspaces" / self.task / "corpus")
        (self.output / "shared_corpus.json").write_text(json.dumps({
            "schema_version": "dci-shared-pool-corpus-v1", "task": self.task,
            "corpus": f"source/workspaces/{self.task}/corpus"}))

        import hashlib
        from dci_bench.data.document_pools import compact_candidate_payload
        payload_path = self.md / "candidate_pools.json"
        payload_path.write_text(json.dumps(compact_candidate_payload(
            json.loads(payload_path.read_text()),
            json.loads((self.md / "source_doc_ids.json").read_text()))))
        seal_path = self.md / "_COMPLETE.json"
        seal = json.loads(seal_path.read_text())
        seal["metadata_hashes"]["candidate_pools.json"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        seal["shared_corpus_sha256"] = hashlib.sha256((self.output / "shared_corpus.json").read_bytes()).hexdigest()
        seal_path.write_text(json.dumps(seal))

    # Parent checks intentionally assert the old on-disk layout; replace those.
    def test_route_each_query_to_its_own_verified_corpus_without_host_labels(self):
        pool = load_pool_for_runtime(self.output / "metadata", self.output / "workspaces",
                                    self.task, {"candidate_pool": get_tier("small")})
        shared = self.output / "workspaces" / self.task
        self.assertEqual(list(shared.iterdir()), [])
        for qid in self.labels:
            with materialize_query_workspace(shared, qid, pool) as workspace:
                self.assertEqual(len(list((workspace / "corpus").iterdir())), 3)
                self.assertNotEqual(workspace, shared)
            self.assertFalse(workspace.exists())

    def test_runtime_cleans_temporary_workspace_after_failure(self):
        pool = load_pool_for_runtime(self.output / "metadata", self.output / "workspaces",
                                    self.task, {"candidate_pool": get_tier("small")})
        shared = self.output / "workspaces" / self.task
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with materialize_query_workspace(shared, "q/one", pool) as workspace:
                raise RuntimeError("synthetic")
        self.assertFalse(workspace.exists())
        (self.source / "workspaces" / self.task / "corpus/d000001.md").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed since preflight"):
            with materialize_query_workspace(shared, "q/one", pool):
                self.fail("changed corpus accepted")

    def test_pointer_escape_rejected(self):
        marker = self.output / "shared_corpus.json"
        payload = json.loads(marker.read_text()); payload["corpus"] = "../../outside"
        marker.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "reference/layout"):
            load_pool_for_runtime(self.output / "metadata", self.output / "workspaces",
                                  self.task, {"candidate_pool": get_tier("small")})
