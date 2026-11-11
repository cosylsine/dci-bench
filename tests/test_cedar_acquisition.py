import importlib.util
import copy
from collections import Counter
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "acquire_cedar_20261031",
    Path(__file__).resolve().parents[1] / "scripts/acquire_cedar_20261031.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def span(text):
    return {"content": text, "start_offset": 0, "end_offset": len(text)}


def records(count=60):
    output = []
    for index in range(count):
        claim = f"论点{index}"
        counterclaim = f"反论点{index}"
        evidence = f"证据{index}"
        unrelated = f"未关联证据{index}"
        output.append({
            "competition": f"competition-{index}",
            "topic": f"topic-{index}",
            "topicType": "value",
            "result": "正方",
            "claims": [{**span(claim), "stance": ""}, {**span(counterclaim), "stance": ""}],
            "evidences": [
                {"Evidence_Type_Case": span(evidence)},
                {"Evidence_Type_Other": span(unrelated)},
            ],
            "relations": [
                {"from_text": evidence, "to_text": claim, "relation": "support"},
                {"from_text": counterclaim, "to_text": claim, "relation": "against"},
            ],
            "rhetoric": [],
            "modality": {},
            "debate": [],
        })
    return output


class CedarConversionTest(unittest.TestCase):
    def test_protocol_assignment_is_exactly_ten_dev_and_fifty_test_groups(self):
        assignment = MODULE.assign_protocol_groups([f"group-{index}" for index in range(60)])
        self.assertEqual(Counter(assignment.values()), Counter({"dev": 10, "test": 50}))

    def test_only_explicit_edges_create_qrels_and_topic_is_retained(self):
        documents, queries, qrels, evidence, audit = MODULE._parse_records(records())
        self.assertEqual(len(queries), 60)
        self.assertEqual(len(qrels), 60)
        self.assertEqual(len(documents), 240)
        self.assertEqual(len(evidence), 120)
        first = qrels["query-claim-0000-00000"]
        self.assertEqual(set(first), {"evidence-0000-00000", "claim-0000-00001"})
        self.assertNotIn("evidence-0000-00001", first)
        first_query = next(row for row in queries if row["_id"] == "query-claim-0000-00000")
        self.assertEqual(first_query["metadata"]["source_annotation_id"], "claim-0000-00000")
        self.assertEqual(
            {row["target_annotation_ids"][0] for row in evidence if row["query_id"] == first_query["_id"]},
            {"claim-0000-00000"},
        )
        self.assertEqual(audit["counts"].get("relations_role_unknown_or_collision", 0), 0)
        self.assertEqual(audit["counts"].get("conflicting_relation_pairs", 0), 0)
        self.assertEqual({query["metadata"]["group_id"] for query in queries}, {
            MODULE._topic_group(f"topic-{index}") for index in range(60)
        })
        self.assertEqual(Counter(query["metadata"]["protocol_group_split"] for query in queries),
                         Counter({"dev": 10, "test": 50}))

    def test_ambiguous_endpoint_is_excluded_without_fabricating_a_qrel(self):
        fixture = records(61)
        fixture[0]["evidences"].append({"Evidence_Type_Research": span("证据0")})
        documents, queries, qrels, evidence, audit = MODULE._parse_records(fixture)
        self.assertGreaterEqual(len(queries), 60)
        self.assertEqual(audit["counts"]["relations_ambiguous_endpoint_text"], 1)
        self.assertNotIn("evidence-0000-00000", qrels["query-claim-0000-00000"])
        self.assertIn("claim-0000-00001", qrels["query-claim-0000-00000"])

    def test_conflicting_relation_labels_remove_the_pair(self):
        fixture = records(61)
        fixture[0]["relations"].append({
            "from_text": "证据0", "to_text": "论点0", "relation": "against",
        })
        documents, queries, qrels, evidence, audit = MODULE._parse_records(fixture)
        self.assertEqual(audit["counts"]["conflicting_relation_pairs"], 1)
        self.assertNotIn("evidence-0000-00000", qrels["query-claim-0000-00000"])
        self.assertIn("claim-0000-00001", qrels["query-claim-0000-00000"])

    def test_official_pin_is_fixed_before_download(self):
        self.assertIn(MODULE.REVISION, MODULE.ARCHIVE_URL)
        self.assertEqual(MODULE.ARCHIVE_BYTES, 21_326_349)
        self.assertEqual(len(MODULE.ARCHIVE_SHA256), 64)
        self.assertEqual(len(MODULE.ARCHIVE_BLOB_SHA1), 40)

    def test_reviewed_bundle_excludes_ambiguous_queries_collapses_aliases_and_keeps_self_match_identity(self):
        documents, queries, qrels, evidence, _ = MODULE._parse_records(records(61))
        duplicate = copy.deepcopy(documents[0])
        duplicate["_id"] = "duplicate-evidence-alias"
        duplicate["metadata"]["source_annotation_id"] = "duplicate-evidence-alias"
        duplicate["metadata"]["source_json_pointer"] = "CEDAR.json/0/evidences/99"
        documents.append(duplicate)
        parent = {
            "documents": documents,
            "queries": queries,
            "qrels": qrels,
            "evidence": evidence,
            "provenance": {},
        }
        parent_audit = {
            "review_required": [{
                "code": "equivalent_unlabelled_document",
                "query_id": "query-claim-0000-00000",
            }],
        }

        reviewed, report, aliases = MODULE._reviewed_bundle(parent, parent_audit)

        self.assertEqual(len(reviewed["queries"]), 60)
        self.assertNotIn("query-claim-0000-00000", {row["_id"] for row in reviewed["queries"]})
        self.assertEqual(report["counts"]["ambiguous_queries_excluded"], 1)
        self.assertEqual(report["counts"]["collapsed_alias_occurrences"], 1)
        self.assertEqual(len(reviewed["documents"]), 244)
        self.assertEqual(len(aliases[documents[0]["_id"]]), 2)
        for query in reviewed["queries"]:
            metadata = query["metadata"]
            self.assertEqual(metadata["group_id"], query["group_id"])
            self.assertEqual(metadata["self_match_document_ids"], [metadata["canonical_query_document_id"]])
            self.assertEqual(metadata["source_query_document_id"], metadata["source_annotation_id"])
        self.assertEqual(
            MODULE._verify_reviewed_query_identity(reviewed),
            {
                "queries_checked": 60,
                "source_annotation_identity_retained": True,
                "canonical_self_match_resolvable": True,
            },
        )
        sampling = MODULE._verify_reviewed_sampling(reviewed)
        self.assertEqual(sampling["counts"], {"dev": 10, "test": 50, "smoke": 1})


if __name__ == "__main__":
    unittest.main()
