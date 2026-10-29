"""Regression for legal Unicode separators observed in the original FiQA corpus."""
import json
from pathlib import Path
import tempfile
import unittest

from dci_bench.data.pipeline.core import load_bundle, publish_bundle, verify_release


class JsonlRoundtripTests(unittest.TestCase):
    def test_unicode_separators_survive_publication_and_verification(self):
        text = "Original financial text: A\u0085B\u2028C\u2029D."
        source = {
            "schema_version": "dci-data-bundle-v1", "input_kind": "retrieval",
            "source": {"dataset": "regression/fiqa-jsonl", "revision": "test"},
            "documents": [{"_id": "d1", "title": "Source", "text": text,
                           "language": "en", "metadata": {}}],
            "queries": [{"_id": "q1", "text": "Which financial text?", "language": "en", "metadata": {}}],
            "qrels": {"q1": {"d1": 2}}, "evidence": [], "judgments": [],
            "provenance": {"domain": "finance", "language": "en"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary) / "release"
            manifest = publish_bundle(source, release)
            self.assertEqual(verify_release(release), manifest)
            restored = load_bundle(release)
            self.assertEqual(restored["documents"][0]["text"], text)
            self.assertEqual(restored["qrels"], source["qrels"])
            # Exercise the literal separator representation written by the pipeline.
            self.assertIn("\u2028".encode(), (release / "documents.jsonl").read_bytes())

    def test_malformed_jsonl_still_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bundle_metadata.json").write_text('{}')
            (root / "documents.jsonl").write_text('{"text": "unterminated\n')
            with self.assertRaises(json.JSONDecodeError):
                load_bundle(root, verify=False)


if __name__ == "__main__":
    unittest.main()
