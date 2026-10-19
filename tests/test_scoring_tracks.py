import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from dci_bench.protocol.code import CODE_LABELS_VERSION, CODE_TOKENIZER, code_protocol_contract
from dci_bench.protocol.contracts import protocol_contract
from dci_bench.scoring import code, document, retrieval
from dci_bench.scoring.__main__ import main
import dci_bench.scoring as legacy_package


class DocumentCompatibilityTest(unittest.TestCase):
    def test_explicit_document_api_and_legacy_exports_are_identical(self):
        for name in legacy_package.__all__:
            self.assertIs(getattr(document, name), getattr(retrieval, name))
            self.assertIs(getattr(document, name), getattr(legacy_package, name))

    def test_document_results_and_defaults_unchanged(self):
        qrels = {"q1": {"d1": 3, "d2": 1}, "q2": {"d3": 1}}
        predictions = {"q1": {"ranked_doc_ids": ["d2", "d1"]}, "q2": {"ranked_doc_ids": []}}
        self.assertEqual(document.score_run(predictions, qrels), retrieval.score_run(predictions, qrels))
        self.assertEqual(document.DEFAULT_METRIC_KS, (1, 3, 5, 10, 20))
        self.assertFalse(document.score_query({"ranked_doc_ids": []}, {"d1": 1})["valid_output"])
        self.assertFalse(document.score_query(list(map(str, range(11))), {"0": 1})["valid_output"])
        self.assertEqual(document.score_query(["d1"], {"d1": 1})["f1_at_10"], 1)

    def test_code_contract_does_not_mutate_document_contract(self):
        before = protocol_contract()
        code_contract = code_protocol_contract()
        code_contract["final_answer_schema"]["properties"]["ranked_files"]["maxItems"] = 100
        self.assertEqual(before, protocol_contract())
        self.assertEqual(before["top_k"], 10)
        self.assertEqual(before["final_answer_schema"]["required"], ["ranked_doc_ids"])
        self.assertEqual(before["inspect_output_contract"]["retrieval_scoring_version"], "dci-retrieval-scoring-v2")


class OfflineEvaluationCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ref = {
            "repository": "demo/repo", "base_commit": "a" * 40, "task": "code2test",
            "files": {"a.py": code.file_metadata("def f():\n    return 1\n")},
            "gold_files": ["a.py"], "tokenizer": CODE_TOKENIZER,
        }

    def invoke(self, track, predictions, labels, *extra):
        self.pred_path = self.root / "predictions.json"
        self.labels_path = self.root / "labels.json"
        self.pred_path.write_text(json.dumps(predictions), encoding="utf-8")
        self.labels_path.write_text(json.dumps(labels), encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(["--track", track, "--predictions", str(self.pred_path),
                           "--labels", str(self.labels_path), *extra])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_code_cli_records_effective_settings_hashes_and_failure_denominators(self):
        labels = {"schema_version": CODE_LABELS_VERSION, "queries": {"q1": self.ref, "q2": self.ref}}
        predictions = {"q1": {**code.snapshot_identity(self.ref), "output": {"ranked_files": ["a.py"], "abstain": False}}}
        status, stdout, stderr = self.invoke("code", predictions, labels, "--metric-ks", "1", "--max-files", "30", "--token-budgets", "100")
        self.assertEqual((status, stderr), (0, ""))
        report = json.loads(stdout)
        result = report["result"]
        self.assertEqual(report["evaluation_track"], "code")
        self.assertEqual(result["invalid_outputs"], 1)
        self.assertEqual(result["metrics"]["file_recall_at_1"]["mean"], 0.5)
        self.assertEqual(report["evaluation_contract"]["max_files"], 30)
        self.assertEqual(report["evaluation_contract"]["final_answer_schema"]["properties"]["ranked_files"]["maxItems"], 30)
        self.assertEqual(report["evaluation_contract"]["token_budgets"], [100])
        self.assertEqual(len(report["input_sha256"]["labels"]), 64)
        self.assertEqual(len(report["evaluation_contract_sha256"]), 64)

    def test_document_cli_is_exact_legacy_scorer_with_separate_provenance(self):
        predictions, labels = {"q": ["a"]}, {"q": {"a": 1}}
        status, stdout, stderr = self.invoke("document", predictions, labels, "--metric-ks", "1", "10")
        self.assertEqual((status, stderr), (0, ""))
        report = json.loads(stdout)
        self.assertEqual(report["result"], retrieval.score_run(predictions, labels, metric_ks=[1, 10]))
        self.assertEqual(report["scoring_version"], "dci-retrieval-scoring-v2")

    def test_output_publication_never_overwrites(self):
        target = self.root / "report.json"
        args = ("--output", str(target))
        first = self.invoke("document", {"q": ["a"]}, {"q": {"a": 1}}, *args)
        self.assertEqual(first[0], 0)
        saved = target.read_bytes()
        second = self.invoke("document", {"q": ["b"]}, {"q": {"a": 1}}, *args)
        self.assertEqual(second[0], 2)
        self.assertEqual(saved, target.read_bytes())
        self.assertFalse(list(self.root.glob(".report.json.*")))

    def test_tracks_and_options_cannot_be_mixed(self):
        labels = {"schema_version": CODE_LABELS_VERSION, "queries": {"q": self.ref}}
        self.assertEqual(self.invoke("document", {}, labels)[0], 2)
        self.assertEqual(self.invoke("code", {}, {"q": {"a": 1}})[0], 2)
        self.assertEqual(self.invoke("document", {}, {"q": {"a": 1}}, "--max-files", "10")[0], 2)

    def test_bad_host_labels_do_not_publish_zero_score_report(self):
        self.ref["gold_files"] = []
        labels = {"schema_version": CODE_LABELS_VERSION, "queries": {"q": self.ref}}
        target = self.root / "report.json"
        status, _, stderr = self.invoke("code", {}, labels, "--output", str(target))
        self.assertEqual(status, 2)
        self.assertIn("no_gold_kind", stderr)
        self.assertFalse(target.exists())

    def test_duplicate_input_queries_are_rejected_before_scoring(self):
        path = self.root / "duplicate.json"
        path.write_text('{"q": [], "q": []}', encoding="utf-8")
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(["--track", "document", "--predictions", str(path), "--labels", str(path)])
        self.assertEqual(status, 2)
        self.assertIn("duplicate", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
