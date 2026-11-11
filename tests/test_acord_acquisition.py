import csv
import json
import tempfile
import unittest
from pathlib import Path

import scripts.acquire_acord_20261031 as acquire


class AcordAcquisitionTests(unittest.TestCase):
    def test_git_blob_sha1_helper(self):
        self.assertEqual(acquire.DATASET_BYTES, 3_131_196)
        self.assertEqual(acquire.DATASET_SHA256, "c9e851456a14aed161033b856e9f40f3cc9f46cf7282c2007b143420e93a65c3")
        self.assertEqual(acquire.DATASET_BLOB_SHA1, "6de8c6c01a5221fe7669bc3117cfc02e5bc02fbc")
        payload = b"ACORD test archive"
        self.assertEqual(
            acquire._git_blob_sha1(payload),
            "cb8397eefad773e52b903cee92f42ca230587a4b",
        )

    def test_grade_mapping_keeps_raw_labels_and_only_positive_qrels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "qrels").mkdir()
            query_map = {
                "q-train": {"metadata": {"split": "train"}},
                "q-valid": {"metadata": {"split": "valid"}},
                "q-test": {"metadata": {"split": "test"}},
            }
            for split, query_id in (("train", "q-train"), ("valid", "q-valid"), ("test", "q-test")):
                with (root / "qrels" / f"{split}.tsv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                    writer.writerow(("query-id", "corpus-id", "score"))
                    for index, grade in enumerate(range(5)):
                        writer.writerow((query_id, f"d-{split}-{index}", grade))
            qrels, judgments, exclusions, counts = acquire._parse_qrels(
                root, query_map, {f"d-{split}-{index}" for split in ("train", "valid", "test") for index in range(5)}
            )
            self.assertEqual(qrels["q-train"], {"d-train-2": 1, "d-train-3": 2, "d-train-4": 3})
            self.assertEqual(len(judgments), 15)
            self.assertEqual({row["raw_grade"] for row in judgments}, set(range(5)))
            self.assertEqual(sum(row["canonical_relevance"] > 0 for row in judgments), 9)
            self.assertEqual(counts["test"]["positive_pairs"], 3)
            self.assertEqual(len(exclusions), 6)

    def test_canonical_inputs_exclude_supplement_variants_and_preserve_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            (dataset / "qrels").mkdir(parents=True)
            (dataset / "supplemental materials").mkdir(parents=True)
            (dataset / "corpus.jsonl").write_text(
                json.dumps({"_id": "d1", "text": "Original clause"}) + "\n",
                encoding="utf-8",
            )
            (dataset / "queries.jsonl").write_text(
                json.dumps(
                    {
                        "_id": "q1",
                        "text": "Attorney query",
                        "metadata": {"category": "Term", "parent_query_id": "", "type": "cuad1", "split": "train"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            for split in ("train", "valid", "test"):
                (dataset / "qrels" / f"{split}.tsv").write_text(
                    "query-id\tcorpus-id\tscore\n" + ("q1\td1\t3\n" if split == "train" else ""),
                    encoding="utf-8",
                )
            (dataset / "supplemental materials" / "acord query (short_medium_long).tsv").write_text(
                "ACORD Query\tMedium Format\tLong Format\n"
                "Attorney query\tmedium\tlong\n",
                encoding="utf-8",
            )
            old_raw = acquire.RAW
            acquire.RAW = root / "raw"
            try:
                result = acquire._build_canonical_inputs(dataset)
            finally:
                acquire.RAW = old_raw
            self.assertEqual(len(result["documents"]), 1)
            self.assertEqual(len(result["queries"]), 1)
            self.assertEqual(result["qrels"], {"q1": {"d1": 2}})
            self.assertEqual(result["queries"][0]["metadata"]["group_id"], "category:Term")
            self.assertEqual(result["queries"][0]["metadata"]["supplement_query_variant"], False)
            self.assertEqual(sum(item["kind"] == "query_variant" for item in result["exclusions"]), 1)


if __name__ == "__main__":
    unittest.main()
