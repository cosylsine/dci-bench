import json
import tempfile
import unittest
from pathlib import Path


try:
    from dci_bench.tasks.mteb_llm_retrieval import _make_sample
except (ImportError, OSError):
    _make_sample = None


@unittest.skipUnless(_make_sample is not None, "a working inspect_ai environment is required")
class InspectTaskLabelTest(unittest.TestCase):
    def test_sample_target_does_not_contain_qrels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata" / "Task"
            metadata.mkdir(parents=True)
            (metadata / "queries.jsonl").write_text(
                json.dumps({"_id": "Q1", "text": "query"}) + "\n",
                encoding="utf-8",
            )
            (metadata / "qrels.json").write_text(
                json.dumps({"Q1": {"GOLD_DOC": 1.0}}),
                encoding="utf-8",
            )

            assert _make_sample is not None
            sample = _make_sample("Task", "Q1", root / "metadata", root / "workspaces")

            self.assertEqual(sample.target, "")
            self.assertNotIn("GOLD_DOC", json.dumps(sample.model_dump(), default=str))


if __name__ == "__main__":
    unittest.main()
