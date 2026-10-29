import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dci_bench.data.document_pools as document_pools
from dci_bench.data.document_pools import (
    DOCUMENT_ID_POLICY,
    prepare_document_pools,
    stable_json_sha256,
    verify_document_pool_suite,
)


class NoHashDocumentPoolTests(unittest.TestCase):
    task = "ToyTask"

    def _source(self, root: Path, *, with_inventory: bool = True) -> Path:
        source = root / "source"
        corpus = source / "workspaces" / self.task / "corpus"
        metadata = source / "metadata" / self.task
        corpus.mkdir(parents=True)
        metadata.mkdir(parents=True)
        mapping = {
            "d000001": "original-000001",
            "d000002": "original-000002",
            "d000003": "original-000003",
        }
        qrels = {
            "Q0": {mapping["d000001"]: 1.0},
            "Q1": {mapping["d000002"]: 1.0},
        }
        query_rows = [{"_id": query_id, "text": "query"} for query_id in qrels]
        doc_id_files = {alias: f"{alias}.md" for alias in mapping}
        (metadata / "source_doc_ids.json").write_text(
            json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
        )
        (metadata / "doc_id_files.json").write_text(
            json.dumps(doc_id_files, indent=2) + "\n", encoding="utf-8"
        )
        (metadata / "original_qrels.json").write_text(
            json.dumps(qrels, indent=2) + "\n", encoding="utf-8"
        )
        (metadata / "queries.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in query_rows), encoding="utf-8"
        )
        for alias in mapping:
            (corpus / f"{alias}.md").write_bytes(f"document {alias}\n".encode())
        canonical_qrels = {
            query_id: {
                alias: score
                for alias, original in mapping.items()
                for document_id, score in qrels[query_id].items()
                if original == document_id
            }
            for query_id in qrels
        }
        (metadata / "qrels.json").write_text(
            json.dumps(canonical_qrels, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "task": self.task,
            "dataset": "toy/dataset",
            "revision": "rev1",
            "resolved_revision": "rev1-resolved",
            "task_instruction": "",
            "document_id_policy": DOCUMENT_ID_POLICY,
            "qrels_hash": stable_json_sha256(canonical_qrels),
            "queries_hash": stable_json_sha256(query_rows),
            "source_doc_ids_hash": stable_json_sha256(mapping),
            "original_qrels_hash": stable_json_sha256(qrels),
        }
        (metadata / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if with_inventory:
            rows = []
            for alias, original in mapping.items():
                path = corpus / doc_id_files[alias]
                rows.append(
                    {
                        "doc_id": alias,
                        "source_doc_id": original,
                        "filename": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            (metadata / "corpus_files.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
        return source

    def _build_nohash(self, root: Path, *, with_inventory: bool = True) -> Path:
        source = self._source(root, with_inventory=with_inventory)
        output = root / "prepared"
        prepare_document_pools(
            source,
            output,
            self.task,
            capacity=3,
            verify_content_hashes=False,
        )
        return output

    def test_nohash_build_reuses_declared_digests_without_hashing_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls: list[Path] = []
            real_hash = document_pools.sha256_file

            def record_hash(path: Path) -> str:
                calls.append(Path(path))
                return real_hash(path)

            source = self._source(root)
            with patch.object(document_pools, "sha256_file", side_effect=record_hash), patch.object(
                document_pools, "_copy_bytes", wraps=document_pools._copy_bytes
            ) as copy_bytes:
                output = root / "prepared"
                prepare_document_pools(
                    source,
                    output,
                    self.task,
                    capacity=3,
                    verify_content_hashes=False,
                )

            self.assertTrue(calls)
            self.assertTrue(all("workspaces" not in path.parts for path in calls))
            self.assertTrue(all(call.kwargs["compute_hash"] is False for call in copy_bytes.call_args_list))
            inventory = json.loads(
                (output / "metadata" / self.task / "file_inventory.json").read_text(
                    encoding="utf-8"
                )
            )
            source_inventory = {
                row["doc_id"]: row["sha256"]
                for row in (
                    json.loads(line)
                    for line in (
                        source / "metadata" / self.task / "corpus_files.jsonl"
                    ).read_text(encoding="utf-8").splitlines()
                )
            }
            self.assertEqual(
                {row["alias"]: row["sha256"] for row in inventory.values()},
                source_inventory,
            )
            manifest = json.loads(
                (output / "metadata" / self.task / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            completion = json.loads(
                (output / "metadata" / self.task / "_COMPLETE.json").read_text(
                    encoding="utf-8"
                )
            )
            for payload in (manifest, completion):
                self.assertEqual(payload["verification_policy"], "declared-source-digests")
                self.assertFalse(payload["content_hashes_verified"])

    def test_strict_verification_detects_same_size_corruption_nohash_accepts_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self._build_nohash(root)
            document = output / "workspaces" / self.task / "corpus" / "d000001.md"
            document.write_bytes(b"X" * document.stat().st_size)
            with self.assertRaisesRegex(ValueError, "workspace hash mismatch"):
                verify_document_pool_suite(output, task=self.task)
            result = verify_document_pool_suite(
                output,
                task=self.task,
                verify_content_hashes=False,
            )
            self.assertFalse(result["content_hashes_verified"])

    def test_nohash_verification_keeps_size_path_and_label_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self._build_nohash(root)
            document = output / "workspaces" / self.task / "corpus" / "d000001.md"

            document.write_bytes(document.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "file inventory size mismatch"):
                verify_document_pool_suite(output, task=self.task, verify_content_hashes=False)

            document.unlink()
            with self.assertRaisesRegex(ValueError, "missing or non-regular"):
                verify_document_pool_suite(output, task=self.task, verify_content_hashes=False)

    def test_nohash_verification_rejects_invalid_inventory_path_and_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self._build_nohash(root)
            metadata = output / "metadata" / self.task
            inventory_path = metadata / "file_inventory.json"
            completion_path = metadata / "_COMPLETE.json"
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            old_relative = next(iter(inventory))
            invalid_relative = "workspaces/%s/corpus/../d000001.md" % self.task
            inventory[invalid_relative] = inventory.pop(old_relative)
            completion["workspace_hashes"][invalid_relative] = completion["workspace_hashes"].pop(
                old_relative
            )
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            completion_path.write_text(json.dumps(completion), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid relative artifact path"):
                verify_document_pool_suite(output, task=self.task, verify_content_hashes=False)

            # Restore the path and corrupt only the declared alias.  The
            # nohash verifier still checks labels and namespace closure.
            inventory[old_relative] = inventory.pop(invalid_relative)
            completion["workspace_hashes"][old_relative] = completion["workspace_hashes"].pop(
                invalid_relative
            )
            inventory[old_relative]["alias"] = "d999999"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            completion_path.write_text(json.dumps(completion), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file inventory row does not match"):
                verify_document_pool_suite(output, task=self.task, verify_content_hashes=False)

    def test_nohash_verification_rejects_inconsistent_declared_inventory_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = self._build_nohash(root)
            inventory_path = output / "metadata" / self.task / "file_inventory.json"
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            first_relative = next(iter(inventory))
            inventory[first_relative]["sha256"] = "declared-but-inconsistent"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file inventory hash mismatch"):
                verify_document_pool_suite(output, task=self.task, verify_content_hashes=False)

    def test_nohash_build_requires_declared_source_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source(root, with_inventory=False)
            with self.assertRaisesRegex(ValueError, "requires a source corpus inventory"):
                prepare_document_pools(
                    source,
                    root / "prepared",
                    self.task,
                    capacity=3,
                    verify_content_hashes=False,
                )


if __name__ == "__main__":
    unittest.main()
