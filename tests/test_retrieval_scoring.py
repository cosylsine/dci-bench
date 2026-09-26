import math
import unittest

from dci_bench.scoring.retrieval import (
    InvalidRankingError,
    f1_at_k,
    ndcg_at_k,
    normalize_metric_ks,
    precision_at_k,
    recall_at_k,
    score_query,
    score_run,
    validate_ranked_doc_ids,
)


class RetrievalScoringTest(unittest.TestCase):
    def test_validate_ranked_doc_ids_accepts_contract_json(self):
        self.assertEqual(validate_ranked_doc_ids('{"ranked_doc_ids":["d1","d2"]}'), ["d1", "d2"])

    def test_validate_ranked_doc_ids_rejects_extra_keys_and_duplicates(self):
        for payload in (
            {"ranked_doc_ids": ["d1"], "score": 1},
            {"ranked_doc_ids": ["d1", "d1"]},
            {"ranked_doc_ids": []},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(InvalidRankingError):
                    validate_ranked_doc_ids(payload)

    def test_ndcg_recall_and_f1_for_perfect_miss_and_ordered_multi_relevance(self):
        qrels = {"d1": 3.0, "d2": 2.0, "d3": 1.0}
        self.assertEqual(ndcg_at_k(["d1", "d2", "d3"], qrels, k=10), 1.0)
        self.assertEqual(recall_at_k(["d1", "d2", "d3"], qrels, k=10), 1.0)
        self.assertEqual(f1_at_k(["d1", "d2", "d3"], qrels, k=3), 1.0)
        self.assertEqual(precision_at_k(["d1", "d2", "d3"], qrels, k=10), 1.0)
        self.assertEqual(f1_at_k(["d1", "d2", "d3"], qrels, k=10), 1.0)
        self.assertEqual(ndcg_at_k(["x", "y"], qrels, k=10), 0.0)
        self.assertEqual(recall_at_k(["x", "y"], qrels, k=10), 0.0)
        self.assertEqual(precision_at_k(["x", "y"], qrels, k=10), 0.0)
        self.assertEqual(f1_at_k(["x", "y"], qrels, k=10), 0.0)

        reversed_score = ndcg_at_k(["d3", "d2", "d1"], qrels, k=10)
        self.assertGreater(reversed_score, 0.0)
        self.assertLess(reversed_score, 1.0)

    def test_f1_uses_actual_returned_window(self):
        qrels = {"d1": 1.0, "d2": 1.0, "d3": 1.0}
        ranked = ["d1", "x", "d2"]
        self.assertEqual(precision_at_k(ranked, qrels, k=2), 0.5)
        self.assertEqual(recall_at_k(ranked, qrels, k=2), 1 / 3)
        self.assertAlmostEqual(f1_at_k(ranked, qrels, k=2), 0.4)
        self.assertAlmostEqual(precision_at_k(ranked, qrels, k=10), 2 / 3)
        self.assertAlmostEqual(recall_at_k(ranked, qrels, k=10), 2 / 3)
        self.assertAlmostEqual(f1_at_k(ranked, qrels, k=10), 2 / 3)
        self.assertEqual(precision_at_k([], qrels, k=10), 0.0)
        self.assertEqual(f1_at_k([], qrels, k=10), 0.0)

    def test_score_query_reports_invalid_output_without_throwing(self):
        result = score_query('{"ranked_doc_ids":["d1","d1"]}', {"d1": 1.0})
        self.assertFalse(result["valid_output"])
        self.assertEqual(result["ndcg_at_10"], 0.0)
        self.assertEqual(result["recall_at_10"], 0.0)
        self.assertEqual(result["f1_at_10"], 0.0)
        self.assertIn("duplicate", result["failure_reason"])

    def test_unknown_doc_id_is_valid_but_scores_zero(self):
        result = score_query({"ranked_doc_ids": ["unknown"]}, {"d1": 1.0})
        self.assertTrue(result["valid_output"])
        self.assertEqual(result["ndcg_at_10"], 0.0)
        self.assertEqual(result["recall_at_10"], 0.0)
        self.assertEqual(result["f1_at_10"], 0.0)

    def test_score_query_defaults_to_requested_metric_cutoffs(self):
        result = score_query({"ranked_doc_ids": ["d1"]}, {"d1": 1.0})
        metric_keys = [
            key
            for key in result
            if key.startswith(("recall_at_", "f1_at_", "ndcg_at_"))
        ]
        self.assertEqual(
            metric_keys,
            [
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "recall_at_10",
                "recall_at_20",
                "f1_at_1",
                "f1_at_3",
                "f1_at_5",
                "f1_at_10",
                "f1_at_20",
                "ndcg_at_1",
                "ndcg_at_3",
                "ndcg_at_5",
                "ndcg_at_10",
                "ndcg_at_20",
            ],
        )

    def test_score_query_supports_custom_metric_cutoffs(self):
        result = score_query({"ranked_doc_ids": ["d1"]}, {"d1": 1.0}, metric_ks=[2, 7])
        self.assertIn("recall_at_2", result)
        self.assertIn("f1_at_7", result)
        self.assertIn("ndcg_at_7", result)
        self.assertNotIn("recall_at_10", result)

    def test_metric_cutoffs_must_be_unique_positive_integers(self):
        for metric_ks in ([], [0], [-1], [1, 1], [True]):
            with self.subTest(metric_ks=metric_ks):
                with self.assertRaises(ValueError):
                    normalize_metric_ks(metric_ks)

    def test_score_run_averages_queries_and_counts_failures(self):
        qrels = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
        predictions = {"q1": ["d1"], "q2": {"ranked_doc_ids": []}}
        result = score_run(predictions, qrels)
        self.assertEqual(result["num_queries"], 2)
        self.assertEqual(result["valid_outputs"], 1)
        self.assertEqual(result["invalid_outputs"], 1)
        self.assertTrue(math.isclose(result["mean_ndcg_at_10"], 0.5))
        self.assertTrue(math.isclose(result["mean_recall_at_10"], 0.5))
        self.assertTrue(math.isclose(result["mean_f1_at_10"], 0.5))
