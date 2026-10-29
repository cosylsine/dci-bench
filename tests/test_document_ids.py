import unittest

from dci_bench.data.document_ids import (
    DOCUMENT_ID_POLICY,
    build_document_id_map,
    remap_qrels,
    restore_ranking,
    reverse_document_id_map,
    validate_source_document_ids,
)
from dci_bench.scoring.retrieval import score_query


class DocumentIdTests(unittest.TestCase):
    def test_policy_is_deterministic_and_uses_utf8_sorted_source_ids(self):
        original = ["z/9", "éclair", "d000001", "a/1", "短"]
        expected = {
            "a/1": "d000001",
            "d000001": "d000002",
            "z/9": "d000003",
            "éclair": "d000004",
            "短": "d000005",
        }
        self.assertEqual(DOCUMENT_ID_POLICY, "short-document-id-v1")
        self.assertEqual(build_document_id_map(reversed(original)), expected)
        self.assertEqual(build_document_id_map(original), expected)
        self.assertEqual(len(set(expected.values())), len(expected))

    def test_validation_rejects_empty_duplicates_and_tampered_aliases(self):
        with self.assertRaises(ValueError):
            build_document_id_map(["ok", ""])
        with self.assertRaises(ValueError):
            build_document_id_map(["ok", "ok"])
        with self.assertRaises(ValueError):
            validate_source_document_ids({"d000001": "a", "d000002": "a"})
        with self.assertRaises(ValueError):
            validate_source_document_ids({"d000001": "a", "d000003": "b"})

    def test_qrel_remap_and_ranking_restore_preserve_scoring(self):
        original_to_alias = build_document_id_map(["slash/id", "éclair", "short"])
        source_doc_ids = reverse_document_id_map(original_to_alias)
        original_qrels = {"q": {"slash/id": 3.0, "éclair": 1.0}}
        aliases = remap_qrels(original_qrels, original_to_alias)
        self.assertEqual(
            restore_ranking(list(aliases["q"]), source_doc_ids),
            ["slash/id", "éclair"],
        )
        original_score = score_query(["slash/id", "éclair"], original_qrels["q"], metric_ks=[2])
        alias_score = score_query(list(aliases["q"]), aliases["q"], metric_ks=[2])
        self.assertEqual(
            {key: alias_score[key] for key in ("recall_at_2", "f1_at_2", "ndcg_at_2")},
            {key: original_score[key] for key in ("recall_at_2", "f1_at_2", "ndcg_at_2")},
        )

    def test_remap_and_restore_reject_unknown_ids(self):
        mapping = build_document_id_map(["a"])
        with self.assertRaises(ValueError):
            remap_qrels({"q": {"missing": 1}}, mapping)
        with self.assertRaises(ValueError):
            restore_ranking(["d000002"], reverse_document_id_map(mapping))


if __name__ == "__main__":
    unittest.main()
