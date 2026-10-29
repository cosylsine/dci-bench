"""Targeted checks for the bounded no-content-hash verification path."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dci_bench.data.expansion import build_document_task, verify_document_task
from dci_bench.data.pipeline.core import (
    digest,
    export_dci,
    file_digest,
    load_bundle,
    publish_bundle,
    verify_release,
)


def _bundle():
    return {
        "schema_version": "dci-data-bundle-v1",
        "input_kind": "retrieval",
        "source": {"dataset": "test/nohash", "revision": "r1"},
        "documents": [
            {"_id": "doc/a", "title": "A", "text": "Mars has two moons."},
            {"_id": "doc/b", "title": "B", "text": "Venus has no moons."},
        ],
        "queries": [{"_id": "q/1", "text": "Which planet has two moons?"}],
        "qrels": {"q/1": {"doc/a": 2}},
        "evidence": [],
        "judgments": [],
        "provenance": {"domain": "science", "language": "en"},
    }


class NoHashVerificationTests(unittest.TestCase):
    def test_release_nohash_skips_expensive_hashes_but_strict_detects_same_size_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary) / "release"
            publish_bundle(_bundle(), release)
            document_path = release / "documents.jsonl"
            original = document_path.read_bytes()
            document_path.write_bytes(original.replace(b"Mars", b"Jars", 1))
            with self.assertRaises(ValueError):
                verify_release(release)
            with patch("dci_bench.data.pipeline.core.file_digest", side_effect=AssertionError("file hash")), \
                    patch("dci_bench.data.pipeline.core.digest", side_effect=AssertionError("bundle hash")):
                receipt = verify_release(release, verify_content_hashes=False)
                restored = load_bundle(release, verify_content_hashes=False)
            self.assertFalse(receipt["content_hashes_verified"])
            self.assertIn("stored digests declared only", receipt["verification_boundary"])
            self.assertEqual(restored["qrels"], {"q/1": {"doc/a": 2}})

    def test_nohash_allows_unlabelled_review_required_evidence_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary) / "release"
            bundle = _bundle()
            bundle["input_kind"] = "evidence_qa"
            bundle["review_queue"] = [{
                "query_id": "q/1",
                "doc_id": "doc/a",
                "reason": "manual review",
            }]
            manifest = publish_bundle(bundle, release)
            self.assertEqual(manifest["state"], "review_required")

            # Keep the release structurally valid while modeling an evidence
            # QA review queue with a retained query awaiting labels.
            queries_path = release / "queries.jsonl"
            with queries_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"_id": "q/2", "text": "Awaiting labels"}) + "\n")
            manifest_path = release / "manifest.json"
            release_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            release_manifest["num_queries"] = 2
            release_manifest["bundle_hash"] = digest(load_bundle(release, verify=False))
            manifest_path.write_text(
                json.dumps(release_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            marker_path = release / "_COMPLETE.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["files"]["queries.jsonl"]["bytes"] = queries_path.stat().st_size
            marker["files"]["queries.jsonl"]["sha256"] = file_digest(queries_path)
            marker["files"]["manifest.json"]["bytes"] = manifest_path.stat().st_size
            marker["files"]["manifest.json"]["sha256"] = file_digest(manifest_path)
            marker_path.write_text(
                json.dumps(marker, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

            # The generic strict release verifier intentionally does not apply
            # this no-hash structural checker to review-required evidence.
            self.assertEqual(verify_release(release)["state"], "review_required")

            with patch("dci_bench.data.pipeline.core.file_digest", side_effect=AssertionError("file hash")), \
                    patch("dci_bench.data.pipeline.core.digest", side_effect=AssertionError("bundle hash")):
                receipt = verify_release(release, verify_content_hashes=False)
            self.assertEqual(receipt["state"], "review_required")
            self.assertEqual(receipt["num_queries"], 2)

    def test_export_nohash_returns_boundary_and_rejects_bad_qrels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "release"
            publish_bundle(_bundle(), release)
            with patch("dci_bench.data.pipeline.core.file_digest", side_effect=AssertionError("file hash")), \
                    patch("dci_bench.data.pipeline.core.digest", side_effect=AssertionError("bundle hash")):
                result = export_dci(
                    release,
                    root / "suite",
                    "nohash-task",
                    verify_content_hashes=False,
                    io_workers=1,
                )
            self.assertFalse(result["content_hashes_verified"])
            self.assertEqual(result["num_queries"], 1)

            # Structural qrel validation remains active even when content
            # digests are deliberately not recomputed.
            metadata = root / "suite" / "metadata" / "nohash-task"
            qrels_path = metadata / "qrels.json"
            qrels = json.loads(qrels_path.read_text())
            qrels["q/1"] = {"missing-alias": 1.0}
            qrels_path.write_text(json.dumps(qrels))
            with self.assertRaises(ValueError):
                verify_document_task(
                    root / "suite",
                    "nohash-task",
                    io_workers=1,
                    verify_content_hashes=False,
                )

    def test_expansion_nohash_keeps_sizes_ids_and_symlink_checks(self):
        task = {
            "name": "nohash",
            "dataset": "fixture/nohash",
            "revision": "a" * 40,
            "track": "document",
            "corpus": [
                {"_id": "source/a", "title": "A", "text": "Document A"},
                {"_id": "source/b", "title": "B", "text": "Document B"},
            ],
            "queries": [{"_id": "q1", "text": "Find A"}],
            "qrels": {"q1": {"source/a": 1.0}},
            "host_labels": {},
            "provenance": {"exclusions": []},
        }
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            build_document_task(task, suite, {"raw_directory": "/raw", "source_fingerprint": "b" * 64}, io_workers=1)
            corpus = suite / "workspaces/nohash/corpus/d000001.md"
            original = corpus.read_bytes()
            corpus.write_bytes(original.replace(b"Document", b"Changed!", 1))
            with self.assertRaises(ValueError):
                verify_document_task(suite, "nohash")
            with patch("dci_bench.data.expansion.sha256_file", side_effect=AssertionError("file hash")):
                receipt = verify_document_task(suite, "nohash", io_workers=1, verify_content_hashes=False)
            self.assertFalse(receipt["content_hashes_verified"])

            corpus.write_bytes(original + b"x")
            with self.assertRaises(ValueError):
                verify_document_task(suite, "nohash", io_workers=1, verify_content_hashes=False)

            corpus.write_bytes(original)
            external = suite / "external.md"
            external.write_bytes(original)
            corpus.unlink()
            corpus.symlink_to(external)
            with self.assertRaises(ValueError):
                verify_document_task(suite, "nohash", io_workers=1, verify_content_hashes=False)


if __name__ == "__main__":
    unittest.main()
