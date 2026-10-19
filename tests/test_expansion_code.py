"""CPU-only ARB archive adaptation checks using tiny self-contained fixtures."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import zstandard

from dci_bench.data import expansion_code as adapter
from dci_bench.scoring.code import file_metadata, validate_reference


class ARBExpansionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.output = self.root / "export"
        self.release = "v2_trace2code"

    def tearDown(self):
        self.temp.cleanup()

    def sample(self, query_id="q1", commit="a" * 40, gold=None, spans=None):
        row = {"id": query_id, "repo": "owner/repo", "base_commit": commit,
               "task_type": "trace2code", "query": {"failure": "observed failure"},
               "gold": {"root_cause_files": gold or ["src/a.py"]}}
        if spans is not None:
            row["gold_spans"] = spans
        return row

    def chunk(self, path="src/a.py", text="def f():\n    return 1\n", commit="a" * 40):
        return {"repo": "owner/repo", "base_commit": commit, "path": path,
                "kind": "file", "start_line": 1, "end_line": len(text.splitlines()), "text": text}

    def archive(self, samples, snapshots, extras=()):
        filename = f"agent_retrieval_bench_{self.release}.tar.zst"
        target = self.raw / "releases" / self.release / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        tar_bytes = io.BytesIO()
        with tarfile.open(fileobj=tar_bytes, mode="w") as archive:
            root = tarfile.TarInfo(".")
            root.type = tarfile.DIRTYPE
            archive.addfile(root)
            rows = [(f"benchmark/{self.release}/samples.jsonl", samples)]
            rows.extend((f"corpus/{self.release}/owner__repo/{commit}.chunks.jsonl", chunks)
                        for commit, chunks in snapshots.items())
            for name, records in rows:
                data = ("".join(json.dumps(row) + "\n" for row in records)).encode()
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            for member, data in extras:
                archive.addfile(member, io.BytesIO(data))
        target.write_bytes(zstandard.ZstdCompressor().compress(tar_bytes.getvalue()))
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return {self.release: (len(samples), digest)}

    def build(self, catalog):
        with patch.dict(adapter.RELEASES, catalog, clear=True):
            return adapter.build_arb(self.raw, self.output, adapter.ARB_REVISION, (self.release,))

    def labels(self):
        return json.loads((self.output / "host/labels.json").read_text())["queries"]

    def test_snapshot_isolation_and_verbatim_all_file_view(self):
        text_a = "\r\n  preserve whitespace\r\n"
        text_b = "different revision\n"
        samples = [self.sample(), self.sample("q2", "b" * 40)]
        catalog = self.archive(samples, {
            "a" * 40: [self.chunk(text=text_a), self.chunk("README.md", "non-gold candidate")],
            "b" * 40: [self.chunk(text=text_b, commit="b" * 40)],
        })
        summary = self.build(catalog)
        self.assertEqual((summary["samples"], summary["snapshots"], summary["files"]), (2, 2, 3))
        labels = self.labels()
        self.assertNotEqual(labels["q1"]["snapshot_id"], labels["q2"]["snapshot_id"])
        for query_id, expected in [("q1", text_a), ("q2", text_b)]:
            label = labels[query_id]
            ref = adapter.load_reference(label, self.output / "snapshots")
            validate_reference(ref)
            self.assertEqual(ref["files"]["src/a.py"], file_metadata(expected))
            actual = self.output / "snapshots" / label["snapshot_id"] / "corpus/src/a.py"
            self.assertEqual(actual.read_bytes(), expected.encode())
        self.assertIn("README.md", adapter.load_reference(labels["q1"], self.output / "snapshots")["files"])
        self.assertTrue((self.output / "_COMPLETE.json").is_file())
        adapter.export_scoring_labels(self.output, self.root / "expanded.json")
        expanded = json.loads((self.root / "expanded.json").read_text())
        self.assertEqual(expanded["schema_version"], "dci-code-labels-v1")
        self.assertEqual(set(expanded["queries"]), {"q1", "q2"})
        with self.assertRaises(FileExistsError):
            adapter.export_scoring_labels(self.output, self.root / "expanded.json")

    def test_truncation_and_optional_spans_are_disclosed_not_scored(self):
        samples = [self.sample("absent"), self.sample("unverified", spans=[
            {"path": "src/a.py", "start_line": 1, "end_line": 1}]),
            self.sample("invalid", spans=[{"path": "src/a.py", "start_line": 0, "end_line": 2}]),
            self.sample("beyond", spans=[{"path": "src/a.py", "start_line": 1, "end_line": 999}])]
        text = "x" * 8000 + "\n...[truncated]"
        chunk = self.chunk(text=text)
        chunk["end_line"] = 999
        summary = self.build(self.archive(samples, {"a" * 40: [chunk]}))
        self.assertEqual(summary["scope"], "arb_file_view")
        self.assertFalse(summary["source_complete"])
        self.assertEqual(summary["truncated_files"], 1)
        expected = {"absent": "source_spans_absent", "unverified": "source_line_correspondence_unverified",
                    "invalid": "source_spans_malformed", "beyond": "source_spans_exceed_released_file_view"}
        for query_id, label in self.labels().items():
            self.assertEqual(label["span_status"]["reason"], expected[query_id])
            ref = adapter.load_reference(label, self.output / "snapshots")
            self.assertNotIn("gold_spans", ref)
            self.assertNotIn("gold_functions", ref)
        raw = json.loads((self.output / "host/raw_samples.json").read_text())
        self.assertEqual(raw["beyond"]["gold_spans"][0]["end_line"], 999)

    def test_unknown_gold_rejected_without_published_artifact(self):
        catalog = self.archive([self.sample(gold=["missing.py"])], {"a" * 40: [self.chunk()]})
        with self.assertRaises(ValueError):
            self.build(catalog)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".export.building-*")))

    def test_existing_artifact_is_never_modified(self):
        self.output.mkdir()
        owned = self.output / "user-owned"
        owned.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            adapter.build_arb(self.raw, self.output)
        self.assertEqual(owned.read_bytes(), b"keep")

    def test_links_and_traversal_rejected_even_in_ignored_reports(self):
        for name, kind in [("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                           ("reports/sym", tarfile.SYMTYPE), ("reports/hard", tarfile.LNKTYPE)]:
            with self.subTest(name=name):
                member = tarfile.TarInfo(name)
                member.type = kind
                if kind != tarfile.REGTYPE:
                    member.linkname = "/tmp/escape"
                catalog = self.archive([self.sample()], {"a" * 40: [self.chunk()]}, [(member, b"")])
                with self.assertRaises(adapter.CodeExpansionError):
                    self.build(catalog)
                self.assertFalse(self.output.exists())

    def test_checksum_is_enforced_before_parsing(self):
        catalog = self.archive([self.sample()], {"a" * 40: [self.chunk()]})
        catalog[self.release] = (1, "0" * 64)
        with self.assertRaisesRegex(adapter.CodeExpansionError, "SHA256 mismatch"):
            self.build(catalog)
        self.assertFalse(self.output.exists())

    def test_candidate_path_traversal_and_duplicate_rows_rejected(self):
        for chunks in [[self.chunk("../outside")], [self.chunk(), self.chunk()]]:
            catalog = self.archive([self.sample()], {"a" * 40: chunks})
            with self.assertRaises(adapter.CodeExpansionError):
                self.build(catalog)
            self.assertFalse(self.output.exists())

    def test_snapshot_tamper_fails_reference_expansion(self):
        self.build(self.archive([self.sample()], {"a" * 40: [self.chunk()]}))
        label = self.labels()["q1"]
        path = self.output / "snapshots" / label["snapshot_id"] / "manifest.json"
        snapshot = json.loads(path.read_text())
        snapshot["files"]["src/a.py"]["line_count"] += 1
        path.write_text(json.dumps(snapshot))
        with self.assertRaisesRegex(adapter.CodeExpansionError, "hash mismatch"):
            adapter.load_reference(label, self.output / "snapshots")

    def test_corpus_byte_tamper_blocks_verification_and_scoring_export(self):
        self.build(self.archive([self.sample()], {"a" * 40: [self.chunk()]}))
        label = self.labels()["q1"]
        file = self.output / "snapshots" / label["snapshot_id"] / "corpus/src/a.py"
        self.assertTrue(adapter.verify_arb(self.output)["corpus_bytes_verified"])
        file.write_bytes(b"changed contents\n")
        with self.assertRaisesRegex(adapter.CodeExpansionError, "corpus bytes/metadata mismatch"):
            adapter.verify_arb(self.output)
        destination = self.root / "cannot_publish.json"
        with self.assertRaises(adapter.CodeExpansionError):
            adapter.export_scoring_labels(self.output, destination)
        self.assertFalse(destination.exists())
        report = adapter.verify_arb(self.output, verify_corpus=False)
        self.assertFalse(report["corpus_bytes_verified"])

    def test_snapshot_nonreference_metadata_tamper_checks_manifest_hash(self):
        self.build(self.archive([self.sample()], {"a" * 40: [self.chunk()]}))
        label = self.labels()["q1"]
        path = self.output / "snapshots" / label["snapshot_id"] / "manifest.json"
        snapshot = json.loads(path.read_text())
        snapshot["source_member"] = "changed provenance"
        path.write_text(json.dumps(snapshot))
        with self.assertRaisesRegex(adapter.CodeExpansionError, "snapshot manifest hash mismatch"):
            adapter.verify_arb(self.output)
        with self.assertRaises(adapter.CodeExpansionError):
            adapter.export_scoring_labels(self.output, self.root / "cannot_publish.json")

    def test_extra_candidate_and_corpus_links_are_rejected(self):
        self.build(self.archive([self.sample()], {"a" * 40: [self.chunk()]}))
        label = self.labels()["q1"]
        corpus = self.output / "snapshots" / label["snapshot_id"] / "corpus"
        extra = corpus / "unexpected.py"
        extra.write_text("extra")
        with self.assertRaisesRegex(adapter.CodeExpansionError, "path set mismatch"):
            adapter.verify_arb(self.output)
        extra.unlink()
        source = corpus / "src/a.py"
        target = self.root / "saved_source"
        target.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(target)
        with self.assertRaisesRegex(adapter.CodeExpansionError, "regular unlinked"):
            adapter.verify_arb(self.output)
        source.unlink()
        os.link(target, source)
        with self.assertRaisesRegex(adapter.CodeExpansionError, "regular unlinked"):
            adapter.verify_arb(self.output)

    def test_all_artifact_hashes_checked(self):
        self.build(self.archive([self.sample()], {"a" * 40: [self.chunk()]}))
        (self.output / "queries.json").write_text("[]")
        with self.assertRaisesRegex(adapter.CodeExpansionError, "artifact hash mismatch: queries.json"):
            adapter.verify_arb(self.output)

    def test_query_id_closure_checked_even_with_updated_hash_chain(self):
        self.build(self.archive([self.sample()], {"a" * 40: [self.chunk()]}))
        (self.output / "queries.json").write_text("[]")
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_hashes"]["queries.json"] = adapter._sha(self.output / "queries.json")
        manifest_path.write_text(json.dumps(manifest))
        complete_path = self.output / "_COMPLETE.json"
        complete = json.loads(complete_path.read_text())
        complete["manifest_sha256"] = adapter._sha(manifest_path)
        complete_path.write_text(json.dumps(complete))
        with self.assertRaisesRegex(adapter.CodeExpansionError, "IDs do not close"):
            adapter.verify_arb(self.output)

    def test_one_and_four_workers_produce_identical_artifacts(self):
        chunks = [self.chunk(f"src/module_{i}.py", f"def f_{i}():\n    return {i}\n") for i in range(25)]
        samples = [self.sample(gold=["src/module_1.py"])]
        catalog = self.archive(samples, {"a" * 40: chunks})
        second = self.root / "parallel_export"
        with patch.dict(adapter.RELEASES, catalog, clear=True):
            first_summary = adapter.build_arb(self.raw, self.output, adapter.ARB_REVISION,
                                              (self.release,), io_workers=1)
            second_summary = adapter.build_arb(self.raw, second, adapter.ARB_REVISION,
                                               (self.release,), io_workers=4)
        self.assertEqual(first_summary, second_summary)
        first_files = {p.relative_to(self.output).as_posix(): p.read_bytes()
                       for p in self.output.rglob("*") if p.is_file()}
        second_files = {p.relative_to(second).as_posix(): p.read_bytes()
                        for p in second.rglob("*") if p.is_file()}
        self.assertEqual(first_files, second_files)
        self.assertEqual(adapter.verify_arb(self.output, io_workers=1),
                         adapter.verify_arb(second, io_workers=4))

    def test_writer_failure_joins_pending_work_before_cleanup(self):
        chunks = [self.chunk(f"src/module_{i}.py", "some content") for i in range(20)]
        catalog = self.archive([self.sample(gold=["src/module_0.py"])], {"a" * 40: chunks})
        original = adapter._NativeWriter._write_bytes
        def fail_one(destination, content):
            if destination.name == "module_3.py":
                raise OSError("synthetic write failure")
            original(destination, content)
        with patch.object(adapter._NativeWriter, "_write_bytes", side_effect=fail_one):
            with self.assertRaisesRegex(OSError, "synthetic write failure"):
                self.build(catalog)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".export.building-*")))


if __name__ == "__main__":
    unittest.main()
