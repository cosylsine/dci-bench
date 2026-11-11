from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import prepare_cfqa_zh_20261102 as cfqa


class CFQAPageContractTests(unittest.TestCase):
    def test_explicit_launcher_source_keeps_old_defaults(self):
        from scripts import run_gapfill_5090_4gpu as launcher
        spec = launcher.SOURCE_SPECS[cfqa.SOURCE]
        self.assertEqual(spec.batch_size, 8)
        self.assertFalse(spec.exclude_self)
        self.assertEqual(spec.receipt_name, "catalog-row.json")
        self.assertEqual((spec.ranking_name, spec.raw_ranking_name), ("ranking.json", "ranking.json"))
        self.assertEqual(spec.pool_names, ("pools-200", "pools-1000"))
        self.assertNotIn(cfqa.SOURCE, launcher.DEFAULT_SOURCES)

    def test_page_annotations_flatten_without_inventing_or_offsetting(self):
        self.assertEqual(cfqa.original_pages([[13], [14, 15], [13]]), [13, 14, 15])
        for invalid in ([], [[]], [[0]], [[True]], [["13"]], [13]):
            self.assertEqual(cfqa.original_pages(invalid), [])

    def test_narrative_scope_and_source_answer_required(self):
        row = {"问题": "伊之密2022年的采购模式是什么", "答案出自": [[14]],
               "答案": "伊之密公司采购由采购部门统一管理，不同原材料采用不同采购方式，通过采购计划管理和供应商质量控制保障生产。"}
        self.assertIsNone(cfqa.query_reason(row))
        self.assertEqual(cfqa.query_reason({**row, "答案": "无法在年报中找到相关信息"}), "empty_or_unanswerable")
        self.assertEqual(cfqa.query_reason({**row, "问题": "2022年研发投入是多少"}), "outside_narrative_enterprise_scope")
        self.assertEqual(cfqa.query_reason({**row, "答案": "2021年" + row["答案"]}), "answer_year_conflicts_with_question")
        self.assertEqual(cfqa.query_reason({**row, "答案出自": []}), "missing_or_invalid_original_physical_pages")
        self.assertIsNone(cfqa.query_reason({**row, "答案出自": [list(range(1, 201))]}))
        self.assertEqual(cfqa.query_reason({**row, "答案出自": [list(range(1, 202))]}),
                         "positive_count_overflow_whole_query")

    def test_financial_pages_do_not_become_enterprise_corpus(self):
        text = "企业通过产品研发和质量管理服务客户，通过采购管理保障生产，持续建设技术人才团队并加强客户服务。" * 5
        self.assertIsNone(cfqa.page_reason(text, in_financial_section=False))
        self.assertEqual(cfqa.page_reason(text, in_financial_section=True), "financial_statement_section")
        self.assertIsNotNone(cfqa.page_reason("资产负债表 1234567890 " * 100, in_financial_section=False))
        self.assertEqual(cfqa.page_body("公司2022年年度报告全文\n14\n企业采购管理。\n14 / 200"), "企业采购管理。")

    def test_exact_bodies_are_merged_and_every_original_positive_maps(self):
        docs = [{"_id": did, "title": did, "text": text, "metadata": {"physical_page_1based": page}}
                for did, text, page in (("b", "相同的企业正文", 2), ("a", "相同的企业正文", 1), ("c", "另一正文", 3))]
        kept, mapping, aliases = cfqa.collapse_pages(docs)
        self.assertEqual([doc["_id"] for doc in kept], ["a", "c"])
        self.assertEqual(mapping, {"a": "a", "b": "a", "c": "c"})
        self.assertEqual(kept[0]["metadata"]["original_page_ids"], ["a", "b"])
        self.assertEqual(len(kept[0]["metadata"]["source_aliases"]), 2)
        self.assertEqual(aliases, [{"canonical_id": "a", "original_page_ids": ["a", "b"]}])
        self.assertEqual({mapping[did] for did in ("b", "c")}, {"a", "c"})

    def test_company_year_template_and_exclusion_only_anchor(self):
        first = {"公司": "伊之密", "问题": "伊之密公司2022年的采购模式是什么？"}
        second = {"公司": "两面针", "问题": "两面针公司2021年的采购模式是什么？"}
        self.assertEqual(cfqa.query_template(first), cfqa.query_template(second))
        self.assertTrue(cfqa.anchor_present("公司采购部门通过供应商质量控制保障正常生产", "通过供应商质量控制保障正常生产"))
        self.assertFalse(cfqa.anchor_present("公司采购部门通过供应商质量控制保障正常生产", "这是完全不同内容的财务报表和审计意见"))

    def test_semantic_templates_connect_cross_company_paraphrases(self):
        pairs = [
            ("两面针2022年采购模式是什么", "伊之密2022年公司的采购模式", "procurement_mode"),
            ("两面针2022年生产模式是什么", "伊之密2022年公司的生产模式", "production_mode"),
            ("两面针2022年从事的业务情况如何", "伊之密2022年主要业务", "business_overview"),
            ("两面针2022年研发方向是什么", "伊之密2022年未来研发的重点方向", "research_direction"),
            ("两面针2022年技术优势是如何体现", "伊之密2022年技术研发创新优势", "technology_advantage"),
        ]
        for first, second, label in pairs:
            with self.subTest(label=label):
                self.assertEqual(cfqa.semantic_template({"公司": "两面针", "问题": first}), "semantic:" + label)
                self.assertEqual(cfqa.semantic_template({"公司": "伊之密", "问题": second}), "semantic:" + label)
        self.assertEqual(cfqa.semantic_template({"公司": "佳禾智能", "问题": "佳禾智能2022年涉及业务有哪些"}),
                         "semantic:business_overview")
        self.assertTrue(cfqa.semantic_template({"公司": "兆易创新", "问题": "兆易创新2022年知识产权方面有哪些优势"}).startswith("exact:"))

    def test_blocked_cli_returns_nonzero(self):
        with patch.object(sys, "argv", ["prepare_cfqa", "prepare", "--raw-root", "raw", "--acquisition", "acq",
                                       "--output", "out", "--evidence", "evidence"]), \
             patch.object(cfqa, "prepare", return_value={"status": "blocked"}), redirect_stdout(io.StringIO()):
            self.assertEqual(cfqa.main(), 2)

    def test_v4_review_gate_freezes_eight_whole_query_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(cfqa, "DEFAULT_CONTENT_REVIEW", Path(temporary) / "missing-review.json"):
            entries, path = cfqa.load_content_review()
        self.assertEqual(set(entries), set(cfqa.MANUAL_REVIEW_QUERY_IDS))
        self.assertEqual(len(entries), 8)
        self.assertIn("cfqa-train-7120", entries)
        self.assertIn("cfqa-test-2493", entries)
        self.assertTrue(path.endswith("missing-review.json"))
        self.assertTrue(all(item["disposition"].endswith("no new gold") for item in entries.values()))
        self.assertTrue(all(item["review_authority"] == "assistant_content_review"
                            and item["upstream_human_qrels"] == "unchanged"
                            and item["new_gold_created"] is False
                            and item["exclusion_scope"] == "whole_query"
                            for item in entries.values()))
        self.assertEqual(entries["cfqa-train-7120"]["category"], "question_answer_scope_gap")
        self.assertEqual(entries["cfqa-test-2493"]["category"], "outside_narrative_enterprise_scope")

    def test_download_preserves_partial_and_retries_into_a_new_temporary_file(self):
        row = {"股票代码": "000001", "公司": "测试企业", "问题": "测试企业2022年的采购模式是什么",
               "答案": "测试企业通过采购部门统一管理供应商和原材料，持续强化质量控制并保障正常生产经营。",
               "答案出自": [[1]], "id": 1}
        report = {"company": "测试企业", "identity_term": "测试企业", "year": 2022,
                  "announcement_id": "r1", "secName": "测试企业", "secCode": "000001",
                  "title": "2022年年度报告", "url": "https://example.test/report.PDF",
                  "declared_size_kb": 1}
        class Response:
            def __init__(self, state):
                self.state = state

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, _size):
                yield b"%PDF-part" if self.state == 1 else b"%PDF-final"
                if self.state == 1:
                    raise RuntimeError("simulated network interruption")

        state = {"calls": 0}
        def get(*_args, **_kwargs):
            state["calls"] += 1
            return Response(state["calls"])
        fake_requests = types.SimpleNamespace(get=get)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            files = []
            for split in ("dev", "test", "train"):
                name = f"dataset/split_by_company/split_by_company_{split}.json"
                path = raw / name
                cfqa.save(path, [row] if split == "test" else [])
                files.append({"path": name, "bytes": path.stat().st_size, "sha256": cfqa.sha(path)})
            cfqa.save(raw / "acquisition-receipt.json", {"repo": cfqa.DATASET, "revision": cfqa.REVISION,
                                                           "files": files})
            target = root / "acquisition"
            with patch.object(cfqa, "lookup", return_value={"company": "测试企业", "attempts": [],
                                                               "selected": report, "rejection": None}), \
                 patch.dict(sys.modules, {"requests": fake_requests}):
                with self.assertRaises(RuntimeError):
                    cfqa.download(raw, target, max_candidate_companies=1, max_bytes=10000)
                partial = target / "pdf" / "000001-2022-r1.PDF.part"
                self.assertTrue(partial.is_file())
                result = cfqa.download(raw, target, max_candidate_companies=1, max_bytes=10000)
            self.assertEqual(result["report_count"], 1)
            self.assertTrue(partial.is_file())
            self.assertTrue((target / "pdf" / "000001-2022-r1.PDF").is_file())

    def test_missing_any_positive_excludes_whole_query_and_small_slice_blocks(self):
        import pymupdf
        body = "测试企业产品研发和采购管理通过供应商质量控制保障正常生产，为客户提供专业技术服务，加强员工培训并建立质量管理体系。" * 4
        rows = [
            {"股票代码": "wrong", "公司": "测试企业", "问题": "测试企业2022年的采购模式是什么", "答案": body, "答案出自": [[1]], "id": 1},
            {"股票代码": "wrong", "公司": "测试企业", "问题": "测试企业2022年的研发技术是什么", "答案": body, "答案出自": [[1], [2]], "id": 2},
        ]
        class Page:
            def __init__(self, text):
                self.text = text
            def get_text(self):
                return self.text
        class PDF(list):
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, acq = root / "raw", root / "acquisition"
            acq.mkdir()
            files = []
            for split in ("dev", "test", "train"):
                name = f"dataset/split_by_company/split_by_company_{split}.json"
                path = raw / name
                cfqa.save(path, rows if split == "test" else [])
                files.append({"path": name, "bytes": path.stat().st_size, "sha256": cfqa.sha(path)})
            cfqa.save(raw / "acquisition-receipt.json", {"repo": cfqa.DATASET, "revision": cfqa.REVISION, "files": files})
            _, _, inputs = cfqa.load_queries(raw)
            cfqa.save(acq / "download-plan.json", {"inputs": inputs})
            pdf = acq / "report.PDF"
            pdf.write_bytes(b"%PDF-fixture")
            cfqa.save(acq / "download-complete.json", {
                "plan_sha256": cfqa.sha(acq / "download-plan.json"), "bytes": pdf.stat().st_size,
                "reports": [{"path": str(pdf), "sha256": cfqa.sha(pdf), "bytes": pdf.stat().st_size,
                             "company": "测试企业", "identity_term": "测试企业", "secCode": "123456",
                             "announcement_id": "report1", "url": "https://static.cninfo.com.cn/fixture.PDF"}]})
            parsed = PDF([Page("测试企业2022年年度报告\n1\n" + body), Page("第十节 财务报告\n" + body)])
            with patch.object(pymupdf, "open", return_value=parsed):
                receipt = cfqa.prepare(raw, acq, root / "release", root / "evidence")
            self.assertEqual(receipt["status"], "blocked")
            self.assertEqual(receipt["queries"], 1)
            self.assertFalse((root / "release").exists())
            excluded = json.loads((root / "evidence/query-exclusions.json").read_text())
            self.assertEqual(excluded[0]["reason"], "missing_or_filtered_positive_page_whole_query")
            retained = json.loads((root / "evidence/original-positive-retention.json").read_text())
            self.assertEqual(len(retained), 1)
            self.assertEqual(retained[0]["original_page_1based"], 1)
            self.assertEqual(retained[0]["query_id"], "cfqa-test-1")


if __name__ == "__main__":
    unittest.main()
