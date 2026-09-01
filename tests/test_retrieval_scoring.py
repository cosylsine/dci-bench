import math
import unittest

from dci_bench.scoring.retrieval import (
    InvalidRankingError,
    ndcg_at_k,
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

    def test_ndcg_and_recall_at_10_for_perfect_miss_and_ordered_multi_relevance(self):
        qrels = {"d1": 3.0, "d2": 2.0, "d3": 1.0}
        self.assertEqual(ndcg_at_k(["d1", "d2", "d3"], qrels, k=10), 1.0)
        self.assertEqual(recall_at_k(["d1", "d2", "d3"], qrels, k=10), 1.0)
        self.assertEqual(ndcg_at_k(["x", "y"], qrels, k=10), 0.0)
        self.assertEqual(recall_at_k(["x", "y"], qrels, k=10), 0.0)

        reversed_score = ndcg_at_k(["d3", "d2", "d1"], qrels, k=10)
        self.assertGreater(reversed_score, 0.0)
        self.assertLess(reversed_score, 1.0)

    def test_score_query_reports_invalid_output_without_throwing(self):
        result = score_query('{"ranked_doc_ids":["d1","d1"]}', {"d1": 1.0})
        self.assertFalse(result["valid_output"])
        self.assertEqual(result["ndcg_at_10"], 0.0)
        self.assertEqual(result["recall_at_10"], 0.0)
        self.assertIn("duplicate", result["failure_reason"])

    def test_unknown_doc_id_is_valid_but_scores_zero(self):
        result = score_query({"ranked_doc_ids": ["unknown"]}, {"d1": 1.0})
        self.assertTrue(result["valid_output"])
        self.assertEqual(result["ndcg_at_10"], 0.0)
        self.assertEqual(result["recall_at_10"], 0.0)

    def test_score_run_averages_queries_and_counts_failures(self):
        qrels = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
        predictions = {"q1": ["d1"], "q2": {"ranked_doc_ids": []}}
        result = score_run(predictions, qrels)
        self.assertEqual(result["num_queries"], 2)
        self.assertEqual(result["valid_outputs"], 1)
        self.assertEqual(result["invalid_outputs"], 1)
        self.assertTrue(math.isclose(result["mean_ndcg_at_10"], 0.5))
        self.assertTrue(math.isclose(result["mean_recall_at_10"], 0.5))
