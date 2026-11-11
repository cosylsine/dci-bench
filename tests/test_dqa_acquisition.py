import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/acquire_dqa_20261031.py"
SPEC = importlib.util.spec_from_file_location("acquire_dqa_20261031", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DQAConversionTest(unittest.TestCase):
    def test_product_rows_keep_source_doc_group_and_positive_qrels(self):
        documents, queries, qrels, audit = MODULE.convert_rows(
            [
                {
                    "question": "如何清理内存中的 Unique SQL？",
                    "answer": "调用 reset_unique_sql。",
                    "doc": "manual-63.json",
                    "retrieval": "reset_unique_sql(text,text,bigint) 用于清理 Unique SQL。",
                    "field": "rag",
                },
                {
                    "question": "哪个函数清理 Unique SQL？",
                    "answer": "调用 reset_unique_sql。",
                    "doc": "manual-63.json",
                    "retrieval": "reset_unique_sql(text,text,bigint) 用于清理 Unique SQL。",
                    "field": "rag",
                },
            ],
            category="specific_product",
            product_or_subtype="GaussDB",
            source_file="ZH/specific_product/GaussDB/test.json",
            source_split="test",
        )
        self.assertEqual(len(documents), 1)
        self.assertEqual(len(queries), 2)
        self.assertEqual(len(qrels), 2)
        self.assertEqual(set(qrels[queries[0]["_id"]].values()), {1})
        self.assertEqual(queries[0]["metadata"]["source_document_name"], "manual-63.json")
        self.assertEqual(queries[0]["metadata"]["source_group"], "dqa:zh:specific_product:GaussDB:manual-63.json")
        self.assertEqual(len(documents[next(iter(documents))]["metadata"]["source_aliases"]), 2)
        self.assertEqual(audit["counts"]["exact_duplicate_document_aliases"], 1)
        self.assertEqual(audit["native_status"], "unverified_translation")

    def test_forum_answer_mapping_retains_original_answer(self):
        documents, queries, qrels, _ = MODULE.convert_rows(
            [{"question": "如何配置数据库端口？", "answer": "检查防火墙和监听地址。", "field": "forum"}],
            category="forum_QA",
            product_or_subtype=None,
            source_file="ZH/general_knowledge/forum_QA/test.json",
            source_split="test",
        )
        self.assertEqual(len(documents), 1)
        query = queries[0]
        document_id = next(iter(qrels[query["_id"]]))
        self.assertEqual(qrels[query["_id"]][document_id], 1)
        self.assertEqual(query["metadata"]["source_answer"], "检查防火墙和监听地址。")
        self.assertEqual(documents[document_id]["text"], "检查防火墙和监听地址。")
        self.assertEqual(query["metadata"]["document_relation"], "published_answer_body")

    def test_instance_rows_are_explicitly_excluded(self):
        documents, queries, qrels, audit = MODULE.convert_rows(
            [{"question": "查询订单数", "answer": "Thought: ...", "tool": "sql_executor"}],
            category="specific_instance",
            product_or_subtype="tool-based",
            source_file="ZH/specific_instance/tool-based/test.json",
            source_split="test",
        )
        self.assertEqual(documents, {})
        self.assertEqual(queries, [])
        self.assertEqual(qrels, {})
        self.assertEqual(audit["counts"]["excluded_synthetic_instance_rows"], 1)
        self.assertEqual(audit["exclusions"][0]["reason"], "synthetic_instance_or_tool_trace_not_retrieval_corpus")

    def test_category_key_keeps_products_separate(self):
        row = {"question": "q", "answer": "a", "doc": "d", "retrieval": "正文"}
        converted = [
            ("specific_product", "GaussDB", "test", row),
            ("specific_product", "openGauss", "test", row),
        ]
        groups = {}
        for category, product, split, item in converted:
            groups.setdefault((category, product), []).append(
                MODULE.convert_rows(
                    [item],
                    category=category,
                    product_or_subtype=product,
                    source_file=f"ZH/specific_product/{product}/test.json",
                    source_split=split,
                )
            )
        merged = {
            category if product is None else f"{category}:{product}": MODULE._merge_category(category, product, items)
            for (category, product), items in groups.items()
        }
        self.assertEqual(set(merged), {"specific_product:GaussDB", "specific_product:openGauss"})

    def test_unverified_native_status_and_small_sample_block(self):
        converted = MODULE.convert_rows(
            [{"question": "q", "answer": "a"}],
            category="general_knowledge_sample",
            product_or_subtype=None,
            source_file="ZH/general_knowledge_subset_example/test.json",
            source_split="test",
        )
        group = MODULE._merge_category("general_knowledge_sample", None, [converted])
        ready, reasons = MODULE.eligibility(group)
        self.assertFalse(ready)
        self.assertTrue(any("native Chinese provenance" in reason for reason in reasons))
        self.assertTrue(any("independent source documents" in reason for reason in reasons))
        self.assertTrue(any("queries" in reason for reason in reasons))

    def test_strict_positive_qrels_reject_zero(self):
        group = {
            "documents": {"d": {"_id": "d"}},
            "queries": [{"_id": "q"}],
            "qrels": {"q": {"d": 0}},
        }
        with self.assertRaises(ValueError):
            MODULE._validate_positive_qrels(group)

    def test_retained_partial_is_not_retried(self):
        with tempfile.TemporaryDirectory(prefix="dqa-download-test-") as directory:
            root = Path(directory)
            partial = root / "public_samples" / "ZH" / "sample.json.partial"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"")
            with mock.patch.object(MODULE, "PUBLIC_SAMPLE_FILES", ("ZH/sample.json",)):
                manifest = MODULE.fetch_public_samples(root)
            self.assertFalse(manifest["complete"])
            self.assertIn("retained partial download refuses", manifest["failures"][0]["error"])

    def test_fixed_sample_materialization_is_hashed_and_immutable(self):
        with tempfile.TemporaryDirectory(prefix="dqa-materialize-test-") as directory:
            root = Path(directory)
            source = root / "source"
            source_file = source / "ZH" / "sample.json"
            source_file.parent.mkdir(parents=True)
            source_file.write_bytes(b'{"question":"q","answer":"a"}\n')
            with mock.patch.object(MODULE, "PUBLIC_SAMPLE_FILES", ("ZH/sample.json",)):
                manifest = MODULE.materialize_fixed_samples(source, root / "raw")
            destination = root / "raw" / "public_samples_reviewed_20261031" / "ZH" / "sample.json"
            self.assertTrue(manifest["complete"])
            self.assertEqual(destination.read_bytes(), source_file.read_bytes())
            source_file.write_bytes(b"changed")
            with mock.patch.object(MODULE, "PUBLIC_SAMPLE_FILES", ("ZH/sample.json",)):
                with self.assertRaises(ValueError):
                    MODULE.materialize_fixed_samples(source, root / "raw")

    def test_eligible_group_round_trips_generic_pipeline(self):
        converted = MODULE.convert_rows(
            [{"question": "q", "answer": "a"}],
            category="forum_QA",
            product_or_subtype=None,
            source_file="ZH/general_knowledge/forum_QA/test.json",
            source_split="test",
        )
        group = MODULE._merge_category("forum_QA", None, [converted])
        with tempfile.TemporaryDirectory(prefix="dqa-acquisition-test-") as directory:
            root = Path(directory)
            result = MODULE.build_verified_release(group, raw_root=root / "raw", release_root=root / "release")
            self.assertEqual(result["release_state"], "review_required")
            self.assertTrue((Path(result["release_path"]) / "_COMPLETE.json").is_file())
            manifest = (Path(result["release_path"]) / "manifest.json").read_text(encoding="utf-8")
            self.assertIn('"automatic_labels": true', manifest)


if __name__ == "__main__":
    unittest.main()
