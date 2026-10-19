import copy
import unittest

from jsonschema import Draft202012Validator

from dci_bench.protocol.code import CODE_LABELS_VERSION, CODE_TOKENIZER, code_protocol_contract
from dci_bench.scoring.code import (
    InvalidCodeReferenceError,
    file_metadata,
    load_json,
    score_query,
    score_run,
    snapshot_identity,
)


def reference(*, fine=True, tokens=True):
    ref = {
        "repository": "org/repo",
        "base_commit": "a" * 40,
        "task": "trace2code",
        "files": {
            "src/a.py": file_metadata("a\nb\nc\nd\ne\nf\n"),
            "src/b.py": file_metadata("g\nh\ni\nj\n"),
            "tests/noise.py": file_metadata("x\ny\nz\n"),
        },
        "gold_files": ["src/a.py", "src/b.py"],
    }
    if tokens:
        ref["tokenizer"] = CODE_TOKENIZER
    else:
        for metadata in ref["files"].values():
            del metadata["content_tokens"]
    if fine:
        ref.update(
            gold_spans=[span("src/a.py", 2, 4), span("src/a.py", 3, 5), span("src/b.py", 1, 2)],
            gold_functions=[
                {"function_id": "a.func", "spans": [span("src/a.py", 2, 5)]},
                {"function_id": "b.func", "spans": [span("src/b.py", 1, 2)]},
            ],
            gold_functionalities=[
                {"functionality_id": "cross-file", "evidence_sets": [
                    [span("src/a.py", 2, 5), span("src/b.py", 1, 2)],
                ]},
                {"functionality_id": "alternatives", "evidence_sets": [
                    [span("src/a.py", 2, 5)], [span("src/b.py", 1, 2)],
                ]},
            ],
        )
    return ref


def span(path, start, end):
    return {"path": path, "start_line": start, "end_line": end}


def prediction(ref, ranked=None, spans=None, *, abstain=False):
    output = {"ranked_files": [] if abstain else (ranked or ["src/a.py"]), "abstain": abstain}
    if spans is not None:
        output["spans"] = spans
    return {**snapshot_identity(ref), "output": output}


def no_gold(kind="natural"):
    ref = reference(fine=False)
    ref["gold_files"] = []
    ref["no_gold_kind"] = kind
    return ref


class CodeScoringTest(unittest.TestCase):
    def test_file_metrics_use_actual_window_and_binary_relevance(self):
        ref = reference()
        result = score_query(prediction(ref, ["tests/noise.py", "src/a.py", "src/b.py"]), ref, metric_ks=[1, 2, 5])
        m = result["metrics"]
        self.assertTrue(result["valid_output"])
        self.assertEqual(m["file_recall_at_1"], 0)
        self.assertEqual(m["file_precision_at_2"], 0.5)
        self.assertEqual(m["file_recall_at_2"], 0.5)
        self.assertEqual(m["file_f1_at_5"], 0.8)
        self.assertEqual(m["file_recall_at_5"], 1)
        self.assertEqual(m["reciprocal_rank"], 0.5)
        self.assertEqual(m["all_gold_files_at_2"], 0)
        self.assertEqual(m["all_gold_files_at_5"], 1)
        self.assertGreater(m["file_ndcg_at_5"], 0)
        self.assertLess(m["file_ndcg_at_5"], 1)

    def test_line_union_and_strict_function_coverage(self):
        ref = reference()
        selected = [span("src/a.py", 2, 4), span("src/a.py", 3, 5), span("src/a.py", 2, 4)]
        m = score_query(prediction(ref, spans=selected), ref)["metrics"]
        self.assertAlmostEqual(m["line_coverage"], 4 / 6)
        self.assertEqual(m["line_precision"], 1)
        self.assertEqual(m["line_f1"], 0.8)
        self.assertEqual(m["function_recall"], 0.5)
        self.assertEqual(m["functionality_recall"], 0.5)
        m = score_query(prediction(ref, spans=[span("src/a.py", 2, 4)]), ref)["metrics"]
        self.assertEqual(m["function_recall"], 0)
        self.assertEqual(m["functionality_recall"], 0)

    def test_cross_file_evidence_and_alternative_sets(self):
        ref = reference()
        m = score_query(prediction(ref, ["src/b.py"], [span("src/b.py", 1, 2)]), ref)["metrics"]
        self.assertEqual(m["functionality_recall"], 0.5)
        full = prediction(ref, ["src/a.py", "src/b.py"], [span("src/a.py", 2, 5), span("src/b.py", 1, 2)])
        m = score_query(full, ref)["metrics"]
        self.assertEqual(m["function_recall"], 1)
        self.assertEqual(m["functionality_recall"], 1)
        self.assertEqual(m["line_f1"], 1)

    def test_noise_lines_reduce_precision_without_changing_coverage(self):
        ref = reference()
        selected = [span("src/a.py", 1, 6), span("src/b.py", 1, 4), span("tests/noise.py", 1, 3)]
        m = score_query(prediction(ref, list(ref["files"]), selected), ref)["metrics"]
        self.assertEqual(m["line_coverage"], 1)
        self.assertAlmostEqual(m["line_precision"], 6 / 13)

    def test_file_hit_does_not_imply_selected_source_lines(self):
        ref = reference()
        m = score_query(prediction(ref, ["src/a.py", "src/b.py"]), ref)["metrics"]
        self.assertEqual(m["file_recall_at_5"], 1)
        self.assertEqual(m["line_coverage"], 0)
        self.assertEqual(m["function_recall"], 0)
        self.assertEqual(m["functionality_recall"], 0)

    def test_missing_annotations_are_unavailable(self):
        ref = reference(fine=False, tokens=False)
        for pred in (None, prediction(ref)):
            m = score_query(pred, ref)["metrics"]
            for key in ("line_coverage", "function_recall", "functionality_recall", "bcy_at_8000_tokens"):
                self.assertIsNone(m[key])

    def test_no_gold_explicit_abstention_and_missing_are_distinct(self):
        for kind in ("natural", "counterfactual"):
            ref = no_gold(kind)
            result = score_query(prediction(ref, abstain=True), ref)
            self.assertTrue(result["valid_output"])
            self.assertEqual(result["metrics"]["no_gold_abstention"], 1)
            self.assertEqual(result["metrics"]["selective_success_at_20"], 1)
            self.assertIsNone(result["metrics"]["file_recall_at_20"])
            self.assertIsNone(result["metrics"]["bcy_at_8000_tokens"])
            missing = score_query(None, ref)
            self.assertFalse(missing["valid_output"])
            self.assertEqual(missing["failure_kind"], "missing_prediction")
            self.assertEqual(missing["metrics"]["no_gold_abstention"], 0)
            self.assertEqual(score_query(prediction(ref), ref)["metrics"]["no_gold_abstention"], 0)

    def test_positive_abstention_is_valid_but_scores_zero(self):
        ref = reference()
        result = score_query(prediction(ref, abstain=True), ref)
        self.assertTrue(result["valid_output"])
        self.assertEqual(result["metrics"]["file_recall_at_20"], 0)
        self.assertEqual(result["metrics"]["line_coverage"], 0)
        self.assertEqual(result["metrics"]["selective_success_at_20"], 0)

    def test_upstream_failure_cannot_become_successful_abstention(self):
        ref = no_gold()
        pred = prediction(ref, abstain=True)
        pred["failure_reason"] = "token_limit"
        result = score_query(pred, ref)
        self.assertFalse(result["valid_output"])
        self.assertEqual(result["failure_kind"], "upstream_failure")
        self.assertEqual(result["metrics"]["no_gold_abstention"], 0)

    def test_invalid_json_and_output_shapes_fail_as_samples(self):
        ref = reference()
        outputs = [None, True, [], 4, "null", "[]", "{", {},
                   {"ranked_doc_ids": ["src/a.py"]},
                   {"ranked_files": [], "abstain": False},
                   {"ranked_files": ["src/a.py"], "abstain": True},
                   {"ranked_files": ["src/a.py"], "abstain": 0},
                   '{"ranked_files": [], "ranked_files": ["src/a.py"], "abstain": false}',
                   {"ranked_files": ["src/a.py", "src/a.py"], "abstain": False},
                   {"ranked_files": ["missing.py"], "abstain": False},
                   {"ranked_files": [None], "abstain": False}]
        for output in outputs:
            with self.subTest(output=output):
                pred = {**snapshot_identity(ref), "output": output}
                result = score_query(pred, ref)
                self.assertFalse(result["valid_output"])
                self.assertEqual(result["metrics"]["file_recall_at_20"], 0)
                self.assertEqual(result["failure_kind"], "invalid_output")
        for pred in (True, [], 0, "null", "[]", "{", {}):
            self.assertFalse(score_query(pred, ref)["valid_output"])

    def test_snapshot_identity_is_checked_and_does_not_depend_on_gold(self):
        ref = reference()
        identity = snapshot_identity(ref)
        other = copy.deepcopy(ref)
        other["gold_files"] = ["src/a.py", "src/b.py", "tests/noise.py"]
        self.assertEqual(identity, snapshot_identity(other))
        for key in ("repository", "base_commit", "corpus_sha256"):
            pred = prediction(ref)
            pred[key] = "wrong"
            self.assertFalse(score_query(pred, ref)["valid_output"])
        other["files"]["src/a.py"]["sha256"] = "f" * 64
        self.assertNotEqual(identity["corpus_sha256"], snapshot_identity(other)["corpus_sha256"])

    def test_paths_and_spans_are_validated_without_normalization(self):
        ref = reference()
        for path in ("../src/a.py", "/src/a.py", "./src/a.py", "src//a.py", "src\\a.py", "C:/a.py", "src/A.py", "src/a.py\x00"):
            self.assertFalse(score_query(prediction(ref, [path]), ref)["valid_output"])
        for selected in ([span("src/a.py", 0, 2)], [span("src/a.py", 3, 2)],
                         [span("src/a.py", True, 2)], [span("src/a.py", 1, 7)],
                         [span("src/b.py", 1, 2)], None, [1]):
            pred = prediction(ref)
            pred["output"]["spans"] = selected
            self.assertFalse(score_query(pred, ref)["valid_output"])

    def test_more_than_ten_files_supported_and_cap_explicit(self):
        ref = reference(fine=False)
        ref["files"] = {f"file{i}.py": file_metadata("a\n") for i in range(21)}
        ref["gold_files"] = list(ref["files"])
        pred = prediction(ref, list(ref["files"])[:20])
        result = score_query(pred, ref)
        self.assertTrue(result["valid_output"])
        self.assertEqual(result["metrics"]["file_recall_at_20"], 20 / 21)
        pred = prediction(ref, list(ref["files"]))
        self.assertFalse(score_query(pred, ref)["valid_output"])
        self.assertTrue(score_query(pred, ref, max_files=21)["valid_output"])

    def test_bad_labels_fail_before_missing_predictions_are_scored(self):
        mutations = [
            lambda r: r.update(gold_files=[]),
            lambda r: r.update(no_gold_kind="natural"),
            lambda r: r.update(base_commit="main"),
            lambda r: r.update(gold_files=["unknown.py"]),
            lambda r: r.update(gold_files=["src/a.py", "src/a.py"]),
            lambda r: r.update(gold_spans=[]),
            lambda r: r.update(gold_functions=[]),
            lambda r: r.update(gold_functionalities=[]),
            lambda r: r["files"]["src/a.py"].update(sha256="bad"),
            lambda r: r["files"]["src/a.py"].update(line_count=True),
            lambda r: r["files"]["src/a.py"].update(content_tokens=-1),
            lambda r: r["files"]["src/a.py"].pop("content_tokens"),
            lambda r: r.update(tokenizer="some_other_tokenizer"),
            lambda r: r["gold_functions"].append(r["gold_functions"][0]),
            lambda r: r["gold_functionalities"][0].update(evidence_sets=[[]]),
            lambda r: r["gold_spans"].append(span("tests/noise.py", 1, 2)),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                ref = reference()
                mutate(ref)
                with self.assertRaises(InvalidCodeReferenceError):
                    score_query(None, ref)

    def test_macro_means_include_failures_but_not_unannotated_queries(self):
        fine, coarse, natural, counter = reference(), reference(fine=False), no_gold(), no_gold("counterfactual")
        labels = {"schema_version": CODE_LABELS_VERSION, "queries": {
            "fine": fine, "failed": fine, "coarse": coarse, "natural": natural, "counter": counter,
        }}
        predictions = {
            "fine": prediction(fine, ["src/a.py", "src/b.py"], [span("src/a.py", 2, 5), span("src/b.py", 1, 2)]),
            "coarse": prediction(coarse), "natural": prediction(natural, abstain=True),
        }
        result = score_run(predictions, labels)
        self.assertEqual(result["num_queries"], 5)
        self.assertEqual(result["valid_outputs"], 3)
        self.assertEqual(result["failure_counts"], {"missing_prediction": 2})
        self.assertEqual(result["metrics"]["line_coverage"], {
            "mean": 0.5, "eligible_queries": 2, "valid_only_mean": 1, "valid_queries": 1,
        })
        self.assertEqual(result["metrics"]["file_recall_at_20"]["mean"], 0.5)
        self.assertEqual(result["metrics"]["no_gold_abstention"]["mean"], 0.5)
        self.assertEqual(result["metrics"]["selective_success_at_20"]["mean"], 0.6)
        self.assertEqual(result["by_stratum"]["counterfactual"]["invalid_outputs"], 1)
        self.assertEqual(result["by_stratum"]["natural"]["valid_outputs"], 1)

    def test_unknown_query_ids_and_invalid_cutoffs_fail_closed(self):
        labels = {"schema_version": CODE_LABELS_VERSION, "queries": {"q": reference()}}
        with self.assertRaises(ValueError):
            score_run({"other": {}}, labels)
        for kwargs in ({"metric_ks": []}, {"metric_ks": [True]}, {"metric_ks": [1, 1]},
                       {"token_budgets": [0]}, {"max_files": 0}, {"max_files": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                score_query(None, reference(), **kwargs)

    def test_json_duplicate_keys_and_nonfinite_numbers_rejected(self):
        for text in ('{"q": {}, "q": {}}', '{"q": NaN}', '{"q": Infinity}'):
            with self.assertRaises(ValueError):
                load_json(text)

    def test_contract_copies_and_output_json_schema(self):
        contract = code_protocol_contract()
        schema = contract["final_answer_schema"]
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        self.assertTrue(validator.is_valid({"ranked_files": [], "abstain": True}))
        self.assertFalse(validator.is_valid({"ranked_files": [], "abstain": False}))
        self.assertFalse(validator.is_valid({"ranked_files": ["a.py"], "abstain": True}))
        schema["properties"]["ranked_files"]["maxItems"] = 1
        self.assertEqual(code_protocol_contract()["final_answer_schema"]["properties"]["ranked_files"]["maxItems"], 20)


class BudgetedContextYieldTest(unittest.TestCase):
    def test_header_only_boundary_gets_no_gold_credit(self):
        ref = reference(fine=False)
        ref["files"] = {"a.py": file_metadata("x y z"), "b.py": file_metadata("noise")}
        ref["gold_files"] = ["a.py"]
        # Header tokens: # # # a . py = 6. A content token first fits at 7.
        result = score_query(prediction(ref, ["a.py"]), ref, token_budgets=[5, 6, 7, 9, 10, 20])
        m = result["metrics"]
        self.assertEqual(m["bcy_at_5_tokens"], 0)
        self.assertEqual(m["bcy_at_6_tokens"], 0)
        self.assertEqual(m["bcy_at_7_tokens"], 1)
        self.assertEqual(result["context_packing"]["9"]["used_tokens"], 9)
        self.assertTrue(result["context_packing"]["10"]["budget_saturated"])
        self.assertTrue(result["context_packing"]["20"]["ranking_exhausted_before_budget"])
        self.assertEqual(result["context_packing"]["20"]["used_tokens"], 10)

    def test_noise_file_and_separator_consume_budget(self):
        ref = reference(fine=False)
        ref["files"] = {"a.py": file_metadata("x y z"), "b.py": file_metadata("noise")}
        ref["gold_files"] = ["a.py"]
        # b.py costs 6 header + 1 content + 1 separator; then a.py's header.
        m = score_query(prediction(ref, ["b.py", "a.py"]), ref, token_budgets=[14, 15])["metrics"]
        self.assertEqual(m["bcy_at_14_tokens"], 0)
        self.assertEqual(m["bcy_at_15_tokens"], 1)

    def test_empty_gold_file_never_receives_budget_credit(self):
        ref = reference(fine=False)
        ref["files"] = {"empty.py": file_metadata("")}
        ref["gold_files"] = ["empty.py"]
        m = score_query(prediction(ref, ["empty.py"]), ref)["metrics"]
        self.assertEqual(m["file_recall_at_20"], 1)
        self.assertEqual(m["bcy_at_8000_tokens"], 0)

    def test_regex_counts_unicode_punctuation_and_preserves_bytes(self):
        metadata = file_metadata("alpha12 + 345 中文\r\n")
        self.assertEqual(metadata["content_tokens"], 5)
        self.assertEqual(metadata["line_count"], 1)
        self.assertNotEqual(metadata["sha256"], file_metadata("alpha12 + 345 中文\n")["sha256"])

    def test_source_lines_use_lf_not_unicode_or_form_feed_separators(self):
        for text, expected in (("", 0), ("\n", 1), ("x\n", 1), ("x\ny", 2),
                               ("x\r\ny\r\n", 2), ("x = 1\f + 2\n", 1),
                               ("x\v\u0085\u2028\u2029y\n", 1)):
            with self.subTest(text=text):
                self.assertEqual(file_metadata(text)["line_count"], expected)


if __name__ == "__main__":
    unittest.main()
