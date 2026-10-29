import json
import hashlib
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import dci_bench.data.document_pools as document_pools
from dci_bench.data.document_pools import (
    EXPECTED_EMBEDDING_MODEL_PATH,
    build_source_identity,
    prepare_document_pools,
    select_candidate_ids,
    stable_json_sha256,
    verify_document_pool_suite,
)


class DocumentPoolSelectorTests(unittest.TestCase):
    def test_positive_injection_keeps_capacity_and_audits_original_ids(self):
        selected, audit = select_candidate_ids(
            ["D1", "D2", "D3", "D4", "D5"],
            ["D1", "D2", "D3", "D4", "D5"],
            {"D5": 3, "D2": 0},
            scores={"D1": 5, "D2": 4, "D3": 3, "D4": 2, "D5": 1},
            capacity=3,
            seed=7,
            task="toy",
            query_id="Q1",
        )
        self.assertEqual(len(selected), 3)
        self.assertIn("D5", selected)
        self.assertEqual(audit["injected_positive_ids"], ["D5"])
        self.assertEqual(audit["injection_count"], audit["eviction_count"])
        self.assertEqual(set(selected), set(audit["selected_ids"]))
        self.assertTrue(all(doc_id in {"D1", "D2", "D3", "D4", "D5"} for doc_id in selected))

    def test_deterministic_negative_eviction_is_order_independent(self):
        kwargs = {"capacity": 3, "seed": 11, "task": "toy", "query_id": "Q1"}
        scores = {"D1": 5, "D2": 4, "D3": 3, "D4": 2, "D5": 1}
        first = select_candidate_ids(
            ["D1", "D2", "D3", "D4", "D5"],
            ["D1", "D2", "D3", "D4", "D5"],
            {"D5": 1},
            scores=scores,
            **kwargs,
        )
        second = select_candidate_ids(
            ["D5", "D4", "D3", "D2", "D1"],
            ["D1", "D2", "D3", "D4", "D5"],
            {"D5": 1},
            scores=scores,
            **kwargs,
        )
        self.assertEqual(first, second)

    def test_all_positive_labels_survive_and_multiple_grades_are_supported(self):
        selected, audit = select_candidate_ids(
            ["D1", "D2", "D3", "D4", "D5", "D6"],
            ["D1", "D2", "D3", "D4", "D5", "D6"],
            {"D5": 1, "D6": 3, "D2": 0},
            scores={"D1": 6, "D2": 5, "D3": 4, "D4": 3, "D5": 2, "D6": 1},
            capacity=4,
            seed=0,
            task="toy",
            query_id="Q1",
        )
        self.assertTrue({"D5", "D6"}.issubset(selected))
        self.assertEqual(audit["positive_count"], 2)
        self.assertEqual(len(selected), 4)

    def test_small_corpus_keeps_whole_corpus_without_rankings(self):
        selected, audit = select_candidate_ids(
            ["D2", "D1"],
            None,
            {"D1": 1},
            capacity=200,
            seed=3,
            task="toy",
            query_id="Q1",
        )
        self.assertEqual(selected, ["D1", "D2"])
        self.assertFalse(audit["ranking_required"])
        selected_with_partial_ranking, _ = select_candidate_ids(
            ["D2", "D1"],
            ["D1"],
            {"D1": 1},
            capacity=200,
            seed=3,
            task="toy",
            query_id="Q1",
        )
        self.assertEqual(selected_with_partial_ranking, ["D1", "D2"])

    def test_duplicate_unknown_and_incomplete_rankings_fail(self):
        corpus = ["D1", "D2", "D3"]
        cases = [
            (["D1", "D1", "D2"], "duplicate"),
            (["D1", "D2", "DX"], "unknown"),
            (["D1"], "incomplete"),
        ]
        for ranking, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    select_candidate_ids(
                        corpus,
                        ranking,
                        {},
                        scores={"D1": 3, "D2": 2, "D3": 1},
                        capacity=2,
                        seed=0,
                        task="toy",
                        query_id="Q1",
                    )

    def test_truncated_topk_ranking_injects_positive_outside_top_cap(self):
        corpus = [f"D{i:03d}" for i in range(205)]
        # A saved top-k export can contain more than the selected cap without
        # being a full corpus permutation.  The positive is outside that
        # truncated ranking and must be injected while retaining cardinality.
        ranking = corpus[:201]
        scores = {doc_id: float(205 - index) for index, doc_id in enumerate(ranking)}
        scores["D204"] = 0.5
        selected, audit = select_candidate_ids(
            corpus,
            ranking,
            {"D204": 2},
            scores=scores,
            capacity=200,
            seed=17,
            task="toy",
            query_id="Q1",
        )
        self.assertEqual(len(selected), 200)
        self.assertIn("D204", selected)
        self.assertEqual(len(audit["initial_top_ids"]), 200)
        self.assertEqual(audit["injected_positive_ids"], ["D204"])
        self.assertEqual(audit["injection_count"], audit["eviction_count"])

    def test_scored_order_preserves_rank_ties_and_injected_tie_follows_ranked(self):
        selected, audit = select_candidate_ids(
            ["D1", "D2", "D3", "D4", "D5"],
            ["D1", "D2", "D3"],
            {"D3": 1, "D5": 1},
            scores={"D1": 5, "D2": 4, "D3": 3, "D5": 3},
            capacity=3,
            seed=17,
            task="toy",
            query_id="Q1",
        )
        self.assertEqual(audit["ordering"], "scored")
        self.assertEqual(len(selected), len(audit["selected_scores"]))
        self.assertEqual(audit["selected_scores"], sorted(audit["selected_scores"], reverse=True))
        self.assertIn("D3", selected)
        self.assertIn("D5", selected)
        # D3 is retained from the original ranking and D5 is injected at the
        # same score, so stable score ordering leaves D5 after D3.
        self.assertLess(selected.index("D3"), selected.index("D5"))

    def test_ranked_tail_tie_precedes_absent_positive_tie(self):
        selected, audit = select_candidate_ids(
            ["D1", "D2", "D3", "D4", "D5"],
            ["D1", "D2", "D3", "D4"],
            {"D3": 1, "D4": 1, "D5": 1},
            scores={"D1": 4, "D2": 3, "D3": 3, "D4": 3, "D5": 3},
            capacity=3,
            seed=4,
            task="toy",
            query_id="Q1",
        )
        self.assertEqual(selected, ["D3", "D4", "D5"])
        self.assertEqual(audit["injected_positive_ids"], ["D4", "D5"])

    def test_scored_ranking_rejects_missing_positive_and_nonmonotone_scores(self):
        common = {
            "capacity": 3,
            "seed": 0,
            "task": "toy",
            "query_id": "Q1",
        }
        with self.assertRaisesRegex(ValueError, "missing positive"):
            select_candidate_ids(
                ["D1", "D2", "D3", "D4"],
                ["D1", "D2", "D3"],
                {"D4": 1},
                scores={"D1": 3, "D2": 2, "D3": 1},
                **common,
            )
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            select_candidate_ids(
                ["D1", "D2", "D3", "D4"],
                ["D1", "D2", "D3"],
                {},
                scores={"D1": 1, "D2": 2, "D3": 0},
                **common,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            select_candidate_ids(
                ["D1", "D2", "D3", "D4"],
                ["D1", "D2", "D3"],
                {},
                scores={"D1": float("nan"), "D2": 2, "D3": 1},
                **common,
            )

    def test_full_corpus_scores_can_order_without_fabricated_scores(self):
        selected, audit = select_candidate_ids(
            ["D2", "D1"],
            ["D1", "D2"],
            {"D1": 1},
            scores={"D2": 0.5, "D1": 0.8},
            capacity=200,
            seed=3,
            task="toy",
            query_id="Q1",
        )
        self.assertEqual(selected, ["D1", "D2"])
        self.assertEqual(audit["selected_scores"], [0.8, 0.5])
        self.assertEqual(audit["ordering"], "scored")

        selected, audit = select_candidate_ids(
            ["D2", "D1"],
            None,
            {"D1": 1},
            capacity=200,
            seed=3,
            task="toy",
            query_id="Q1",
        )
        self.assertEqual(selected, ["D1", "D2"])
        self.assertIsNone(audit["selected_scores"])
        self.assertEqual(audit["ordering"], "unscored")

    def test_missing_ranking_and_positive_overflow_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "saved ranking is required"):
            select_candidate_ids(
                ["D1", "D2", "D3"],
                None,
                {},
                capacity=2,
                seed=0,
                task="toy",
                query_id="Q1",
            )

        with self.assertRaisesRegex(ValueError, "positive qrels exceed"):
            select_candidate_ids(
                ["D1", "D2", "D3"],
                ["D1", "D2", "D3"],
                {"D1": 1, "D2": 2, "D3": 3},
                scores={"D1": 3, "D2": 2, "D3": 1},
                capacity=2,
                seed=0,
                task="toy",
                query_id="Q1",
            )


class DocumentPoolBuilderTests(unittest.TestCase):
    task = "ToyTask"

    def _source(self, root: Path, *, n_docs: int = 4, n_queries: int = 2) -> tuple[Path, dict[str, str], dict[str, dict[str, float]]]:
        source = root / "source"
        corpus = source / "workspaces" / self.task / "corpus"
        metadata = source / "metadata" / self.task
        corpus.mkdir(parents=True)
        metadata.mkdir(parents=True)
        # Canonical short IDs are fixed-width and one-based.  The source
        # document IDs intentionally use a separate namespace so a malformed
        # corpus inventory cannot pass by coincidentally matching aliases.
        mapping = {f"d{i + 1:06d}": f"original-{i:06d}" for i in range(n_docs)}
        qrels = {
            f"Q{i}": {mapping[f"d{i % n_docs + 1:06d}"]: 1.0}
            for i in range(n_queries)
        }
        (metadata / "source_doc_ids.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        doc_id_files = {alias: f"{alias}.md" for alias in mapping}
        (metadata / "doc_id_files.json").write_text(
            json.dumps(doc_id_files, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (metadata / "original_qrels.json").write_text(
            json.dumps(qrels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        query_rows = [{"_id": qid, "text": "query"} for qid in qrels]
        (metadata / "queries.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in query_rows), encoding="utf-8"
        )
        for alias in mapping:
            (corpus / f"{alias}.md").write_bytes(f"document {alias}\n".encode())
        canonical_qrels = {
            query_id: {next(alias for alias, original in mapping.items() if original == doc_id): score for doc_id, score in docs.items()}
            for query_id, docs in qrels.items()
        }
        (metadata / "qrels.json").write_text(
            json.dumps(canonical_qrels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "task": self.task,
            "dataset": "toy/dataset",
            "revision": "rev1",
            "resolved_revision": "rev1-resolved",
            "task_instruction": "",
            "document_id_policy": "short-document-id-v1",
            "qrels_hash": stable_json_sha256(canonical_qrels),
            "queries_hash": stable_json_sha256(query_rows),
            "source_doc_ids_hash": stable_json_sha256(mapping),
            "original_qrels_hash": stable_json_sha256(qrels),
        }
        (metadata / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return source, mapping, qrels

    def _rankings(self, source: Path, mapping: dict[str, str], qrels: dict[str, dict[str, float]], *, task: str | None = None) -> Path:
        task = task or self.task
        existing_aliases = [
            alias
            for alias in mapping
            if (source / "workspaces" / task / "corpus" / f"{alias}.md").is_file()
        ]
        identity = build_source_identity(
            task=task,
            corpus_ids=[mapping[alias] for alias in existing_aliases],
            query_ids=list(qrels),
            source_doc_ids=mapping,
            qrels=qrels,
            query_records=[{"_id": query_id, "text": "query"} for query_id in sorted(qrels)],
            corpus_file_hashes={
                alias: hashlib.sha256(
                    (source / "workspaces" / task / "corpus" / f"{alias}.md").read_bytes()
                ).hexdigest()
                for alias in existing_aliases
            },
        )
        ranking = {
            "schema_version": "dci-saved-retrieval-v1",
            "task": task,
            "model": {"path": EXPECTED_EMBEDDING_MODEL_PATH, "trust_remote_code": True},
            "source_identity": identity,
            "rankings": {
                query_id: [mapping[alias] for alias in existing_aliases]
                for query_id in sorted(qrels)
            },
            "scores": {
                query_id: {
                    mapping[alias]: float(len(existing_aliases) - index)
                    for index, alias in enumerate(existing_aliases)
                }
                for query_id in sorted(qrels)
            },
        }
        path = source.parent / "rankings.json"
        path.write_text(json.dumps(ranking, indent=2), encoding="utf-8")
        return path

    def test_large_two_query_union_is_one_shared_corpus_with_injected_201st_doc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mapping, qrels = self._source(root, n_docs=201, n_queries=2)
            metadata = source / "metadata" / self.task
            qrels["Q1"] = {mapping["d000201"]: 1.0}
            (metadata / "original_qrels.json").write_text(
                json.dumps(qrels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            canonical_qrels = {
                query_id: {
                    next(alias for alias, original in mapping.items() if original == doc_id): score
                    for doc_id, score in docs.items()
                }
                for query_id, docs in qrels.items()
            }
            (metadata / "qrels.json").write_text(
                json.dumps(canonical_qrels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            manifest = json.loads((metadata / "manifest.json").read_text(encoding="utf-8"))
            manifest["qrels_hash"] = stable_json_sha256(canonical_qrels)
            manifest["original_qrels_hash"] = stable_json_sha256(qrels)
            (metadata / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            rankings = self._rankings(source, mapping, qrels)
            output = root / "prepared"
            prepare_document_pools(
                source,
                output,
                self.task,
                capacity=200,
                seed=19,
                rankings_path=rankings,
            )
            payload = json.loads(
                (output / "metadata" / self.task / "candidate_pools.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                payload["pools"]["Q1"]["candidate_original_doc_ids"][-1],
                "original-000200",
            )
            self.assertEqual(payload["pools"]["Q1"]["candidate_scores"][-1], 1.0)
            self.assertTrue(
                all(
                    previous >= current
                    for previous, current in zip(
                        payload["pools"]["Q1"]["candidate_scores"],
                        payload["pools"]["Q1"]["candidate_scores"][1:],
                    )
                )
            )
            inventory = json.loads(
                (output / "metadata" / self.task / "file_inventory.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(inventory), 201)
            self.assertLessEqual(len(inventory), 201)
            self.assertEqual(
                verify_document_pool_suite(output, task=self.task)["num_workspace_files"],
                201,
            )

    def test_builder_reuses_validated_corpus_for_scored_multi_query_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mapping, qrels = self._source(root, n_docs=201, n_queries=4)
            positive_original = mapping["d000201"]
            qrels = {
                query_id: {positive_original: 1.0}
                for query_id in qrels
            }
            metadata = source / "metadata" / self.task
            (metadata / "original_qrels.json").write_text(
                json.dumps(qrels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            canonical_qrels = {
                query_id: {"d000201": 1.0}
                for query_id in qrels
            }
            (metadata / "qrels.json").write_text(
                json.dumps(canonical_qrels, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest = json.loads((metadata / "manifest.json").read_text(encoding="utf-8"))
            manifest["qrels_hash"] = stable_json_sha256(canonical_qrels)
            manifest["original_qrels_hash"] = stable_json_sha256(qrels)
            (metadata / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            rankings = self._rankings(source, mapping, qrels)
            output = root / "prepared"
            with patch(
                "dci_bench.data.document_pools._validate_id_sequence",
                wraps=document_pools._validate_id_sequence,
            ) as validate_ids, patch(
                "dci_bench.data.document_pools.select_candidate_ids",
                side_effect=AssertionError("builder must use validated private selector"),
            ) as public_selector:
                result = prepare_document_pools(
                    source,
                    output,
                    self.task,
                    capacity=200,
                    seed=19,
                    rankings_path=rankings,
                )

            public_selector.assert_not_called()
            corpus_validation_calls = [
                call
                for call in validate_ids.call_args_list
                if call.kwargs.get("name") == "corpus"
            ]
            self.assertGreaterEqual(len(corpus_validation_calls), 1)
            self.assertLessEqual(
                len(corpus_validation_calls),
                2,
                "full-corpus validation must not grow with query count",
            )
            self.assertEqual(result.num_queries, 4)
            payload = json.loads(
                (output / "metadata" / self.task / "candidate_pools.json").read_text(
                    encoding="utf-8"
                )
            )
            for query_id, record in payload["pools"].items():
                self.assertEqual(record["audit"]["injection_count"], 1, query_id)
                self.assertIn(positive_original, record["candidate_original_doc_ids"])
                self.assertIsInstance(record["candidate_scores"], list)

    def test_builder_copies_bytes_and_keeps_host_labels_out_of_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mapping, qrels = self._source(root)
            # The source byte order is preserved even though identity hashing
            # canonicalises query records by ID.
            query_path = source / "metadata" / self.task / "queries.jsonl"
            query_path.write_bytes(b'{"_id": "Q1", "text": "query"}\n{"_id": "Q0", "text": "query"}\n')
            inventory_path = source / "metadata" / self.task / "corpus_files.jsonl"
            inventory_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "doc_id": alias,
                            "source_doc_id": original,
                            "filename": f"{alias}.md",
                            "bytes": (source / "workspaces" / self.task / "corpus" / f"{alias}.md").stat().st_size,
                            "sha256": hashlib.sha256(
                                (source / "workspaces" / self.task / "corpus" / f"{alias}.md").read_bytes()
                            ).hexdigest(),
                        }
                    )
                    + "\n"
                    for alias, original in mapping.items()
                ),
                encoding="utf-8",
            )
            rankings = self._rankings(source, mapping, qrels)
            output = root / "prepared"
            result = prepare_document_pools(
                source,
                output,
                self.task,
                capacity=2,
                seed=5,
                rankings_path=rankings,
            )
            self.assertTrue(result.candidate_pools_path.exists())
            self.assertEqual(
                (output / "metadata" / self.task / "original_qrels.json").read_bytes(),
                (source / "metadata" / self.task / "original_qrels.json").read_bytes(),
            )
            self.assertEqual(
                (output / "metadata" / self.task / "queries.jsonl").read_bytes(),
                query_path.read_bytes(),
            )
            self.assertFalse(list((output / "workspaces").rglob("*qrel*")))
            self.assertFalse(list((output / "workspaces").rglob("*candidate*")))
            source_doc = source / "workspaces" / self.task / "corpus" / "d000001.md"
            output_workspace = output / "workspaces" / self.task
            output_doc = output_workspace / "corpus" / "d000001.md"
            self.assertEqual(source_doc.read_bytes(), output_doc.read_bytes())
            self.assertNotEqual(source_doc.stat().st_ino, output_doc.stat().st_ino)
            self.assertEqual({path.name for path in output_workspace.iterdir()}, {"corpus"})
            self.assertFalse((output_workspace / "queries").exists())
            self.assertEqual(verify_document_pool_suite(output, task=self.task)["num_queries"], 2)
            candidate_payload = json.loads(
                (output / "metadata" / self.task / "candidate_pools.json").read_text(encoding="utf-8")
            )
            self.assertIsNotNone(candidate_payload["source_release"]["source_inventory_sha256"])
            self.assertEqual(candidate_payload["shared_corpus"], "corpus")
            for record in candidate_payload["pools"].values():
                self.assertNotIn("workspace", record)
                self.assertNotIn("corpus", record)
                self.assertNotIn("files", record)

    def test_output_exists_refuses_overwrite_and_source_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mapping, qrels = self._source(root)
            rankings = self._rankings(source, mapping, qrels)
            output = root / "prepared"
            output.mkdir()
            sentinel = output / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_document_pools(source, output, self.task, capacity=2, rankings_path=rankings)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual(
                (source / "metadata" / self.task / "source_doc_ids.json").read_text(encoding="utf-8"),
                json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
            )

    def test_failed_staging_is_retained_and_retry_uses_new_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mapping, qrels = self._source(root)
            rankings = self._rankings(source, mapping, qrels)
            output = root / "prepared"
            with patch(
                "dci_bench.data.document_pools._copy_bytes",
                side_effect=OSError("simulated copy failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated copy failure"):
                    prepare_document_pools(
                        source,
                        output,
                        self.task,
                        capacity=2,
                        rankings_path=rankings,
                    )

            failed_stages = sorted(root.glob(".prepared.staging-*"))
            self.assertEqual(len(failed_stages), 1)
            self.assertFalse(output.exists())
            failure = json.loads((failed_stages[0] / "_FAILED.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["stage"], "prepare-document-pools")
            self.assertEqual(failure["exception"], "OSError")

            result = prepare_document_pools(
                source,
                output,
                self.task,
                capacity=2,
                rankings_path=rankings,
            )
            self.assertTrue(result.completion_path.is_file())
            self.assertEqual(sorted(root.glob(".prepared.staging-*")), failed_stages)

    def test_final_verify_failure_preserves_unpublished_completion_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mapping, qrels = self._source(root)
            rankings = self._rankings(source, mapping, qrels)
            output = root / "prepared"
            with patch(
                "dci_bench.data.document_pools.verify_document_pool_suite",
                side_effect=RuntimeError("simulated final verification failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated final verification failure"):
                    prepare_document_pools(
                        source,
                        output,
                        self.task,
                        capacity=2,
                        rankings_path=rankings,
                    )

            failed_stages = sorted(root.glob(".prepared.staging-*"))
            self.assertEqual(len(failed_stages), 1)
            failed_stage = failed_stages[0]
            self.assertFalse(output.exists())
            self.assertTrue((failed_stage / "_UNPUBLISHED_COMPLETE.json").is_file())
            self.assertFalse(
                (failed_stage / "metadata" / self.task / "_COMPLETE.json").exists()
            )
            failure = json.loads((failed_stage / "_FAILED.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["exception"], "RuntimeError")

            result = prepare_document_pools(
                source,
                output,
                self.task,
                capacity=2,
                rankings_path=rankings,
            )
            self.assertTrue(result.completion_path.is_file())
            self.assertTrue(output.is_dir())
            self.assertEqual(sorted(root.glob(".prepared.staging-*")), failed_stages)
            self.assertEqual(
                json.loads((failed_stage / "_FAILED.json").read_text(encoding="utf-8"))["exception"],
                "RuntimeError",
            )

    def test_jsonl_query_text_with_unicode_line_separator_is_one_physical_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mapping, qrels = self._source(root)
            metadata = source / "metadata" / self.task
            query_rows = [
                {"_id": "Q0", "text": "first\u2028second"},
                {"_id": "Q1", "text": "query"},
            ]
            (metadata / "queries.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in query_rows),
                encoding="utf-8",
            )
            manifest = json.loads((metadata / "manifest.json").read_text(encoding="utf-8"))
            manifest["queries_hash"] = stable_json_sha256(query_rows)
            (metadata / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            output = root / "prepared"
            result = prepare_document_pools(source, output, self.task, capacity=200)
            self.assertEqual(result.num_queries, 2)
            self.assertEqual(
                verify_document_pool_suite(output, task=self.task)["num_queries"],
                2,
            )

    def test_full_host_mapping_and_zero_label_excluded_document_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mapping, qrels = self._source(root)
            metadata = source / "metadata" / self.task
            excluded_alias = "d000005"
            excluded_original = "original-000099"
            mapping[excluded_alias] = excluded_original
            qrels["Q0"][excluded_original] = 0.0
            qrels["Q-excluded"] = {excluded_original: 1.0}
            (metadata / "source_doc_ids.json").write_text(
                json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (metadata / "original_qrels.json").write_text(
                json.dumps(qrels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            canonical_qrels = {
                query_id: {
                    next(alias for alias, original in mapping.items() if original == doc_id): score
                    for doc_id, score in docs.items()
                }
                for query_id, docs in qrels.items()
                if query_id in {"Q0", "Q1"}
            }
            (metadata / "qrels.json").write_text(
                json.dumps(canonical_qrels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            manifest = json.loads((metadata / "manifest.json").read_text(encoding="utf-8"))
            manifest["source_doc_ids_hash"] = stable_json_sha256(mapping)
            manifest["original_qrels_hash"] = stable_json_sha256(qrels)
            manifest["qrels_hash"] = stable_json_sha256(canonical_qrels)
            (metadata / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            active_qrels = {query_id: qrels[query_id] for query_id in ("Q0", "Q1")}
            rankings = self._rankings(source, mapping, active_qrels)
            output = root / "prepared"
            prepare_document_pools(
                source, output, self.task, capacity=2, seed=5, rankings_path=rankings
            )
            self.assertEqual(
                (output / "metadata" / self.task / "original_qrels.json").read_bytes(),
                (metadata / "original_qrels.json").read_bytes(),
            )
            self.assertEqual(
                (output / "metadata" / self.task / "qrels.json").read_bytes(),
                (metadata / "qrels.json").read_bytes(),
            )
            self.assertFalse(
                (output / "workspaces" / self.task / "corpus" / f"{excluded_alias}.md").exists()
            )
            self.assertEqual(verify_document_pool_suite(output, task=self.task)["num_queries"], 2)

    def test_tampered_copied_file_fails_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mapping, qrels = self._source(root)
            rankings = self._rankings(source, mapping, qrels)
            output = root / "prepared"
            prepare_document_pools(source, output, self.task, capacity=2, rankings_path=rankings)
            document = output / "workspaces" / self.task / "corpus" / "d000001.md"
            document.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "workspace hash mismatch"):
                verify_document_pool_suite(output, task=self.task)

    def test_saved_ranking_source_identity_and_model_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mapping, qrels = self._source(root)
            rankings = self._rankings(source, mapping, qrels)
            payload = json.loads(rankings.read_text(encoding="utf-8"))
            payload["source_identity"]["source_doc_ids_hash"] = "wrong"
            rankings.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_doc_ids_hash"):
                prepare_document_pools(source, root / "prepared", self.task, capacity=2, rankings_path=rankings)

    def test_saved_ranking_missing_score_is_rejected_for_topk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mapping, qrels = self._source(root)
            rankings = self._rankings(source, mapping, qrels)
            payload = json.loads(rankings.read_text(encoding="utf-8"))
            payload["scores"]["Q0"].pop("original-000000")
            rankings.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing ranked IDs"):
                prepare_document_pools(
                    source,
                    root / "prepared",
                    self.task,
                    capacity=2,
                    rankings_path=rankings,
                )

    def test_saved_identity_binds_query_text_and_rendered_corpus_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mapping, qrels = self._source(root)
            rankings = self._rankings(source, mapping, qrels)

            metadata = source / "metadata" / self.task
            query_rows = [{"_id": "Q0", "text": "changed"}, {"_id": "Q1", "text": "query"}]
            (metadata / "queries.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in query_rows), encoding="utf-8"
            )
            manifest = json.loads((metadata / "manifest.json").read_text(encoding="utf-8"))
            manifest["queries_hash"] = stable_json_sha256(query_rows)
            (metadata / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "queries_hash"):
                prepare_document_pools(
                    source, root / "query-text-output", self.task, capacity=2, rankings_path=rankings
                )

            # Restore the source query inventory and change only rendered bytes;
            # IDs remain identical while corpus_files_sha256 must change.
            (metadata / "queries.jsonl").write_text(
                "".join(json.dumps({"_id": qid, "text": "query"}) + "\n" for qid in qrels),
                encoding="utf-8",
            )
            manifest["queries_hash"] = stable_json_sha256(
                [{"_id": qid, "text": "query"} for qid in qrels]
            )
            (metadata / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (source / "workspaces" / self.task / "corpus" / "d000001.md").write_text(
                "changed bytes", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "corpus_files_sha256"):
                prepare_document_pools(
                    source, root / "corpus-bytes-output", self.task, capacity=2, rankings_path=rankings
                )


if __name__ == "__main__":
    unittest.main()

class CompressedRankingTests(unittest.TestCase):
    def test_gzip_ranking_builds_identical_pool(self):
        import gzip
        fixture = DocumentPoolBuilderTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, mapping, qrels = fixture._source(root)
            rankings = fixture._rankings(source, mapping, qrels)
            compressed = rankings.with_suffix('.json.gz')
            with gzip.open(compressed, 'wb') as handle:
                handle.write(rankings.read_bytes())
            for path, name in [(rankings, 'plain'), (compressed, 'compressed')]:
                prepare_document_pools(source, root / name, fixture.task,
                                       capacity=2, seed=5, rankings_path=path)
            def records(name):
                return json.loads((root / name / 'metadata' / fixture.task /
                                   'candidate_pools.json').read_text())['pools']
            self.assertEqual(records('plain'), records('compressed'))
