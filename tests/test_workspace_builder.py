from pathlib import Path
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from dci_bench.data.loader import load_qrels
from dci_bench.data.registry import TaskSpec
from dci_bench.data.workspace_builder import build_workspace, doc_id_to_filename, render_doc_markdown


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


class WorkspaceBuilderTest(unittest.TestCase):
    def test_doc_id_filename_is_reversible_for_unicode_ids(self):
        self.assertEqual(doc_id_to_filename("pégase_23_6"), "p%C3%A9gase_23_6.md")
        self.assertEqual(doc_id_to_filename("D/1"), "D%2F1.md")

    def test_render_doc_markdown_preserves_text_trailing_whitespace(self):
        self.assertEqual(render_doc_markdown("", "alpha text \n"), "alpha text \n")

    def test_load_qrels_rejects_duplicate_pairs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_table(
                root / "data" / "test-00000-of-00001.parquet",
                [
                    {"query-id": "Q1", "corpus-id": "D1", "score": 1},
                    {"query-id": "Q1", "corpus-id": "D1", "score": 1},
                ],
            )
            with self.assertRaises(ValueError):
                load_qrels(root)

    def test_build_workspace_writes_corpus_and_keeps_qrels_host_side(self):
        with self.subTest("tmp workspace"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                task = TaskSpec(
                    name="ToyTask",
                    hf_dataset="mteb/llm-eval-public-health-qa",
                    revision="test-revision",
                    local_dir_name="llm-eval-public-health-qa",
                )
                source = tmp_path / "raw" / task.local_dir_name
                write_table(
                    source / "corpus" / "corpus-00000-of-00001.parquet",
                    [
                        {"_id": "D/1", "title": "First", "text": "alpha text"},
                        {"_id": "D2", "title": "", "text": "beta text"},
                    ],
                )
                write_table(
                    source / "queries" / "queries-00000-of-00001.parquet",
                    [{"_id": "Q1", "text": "alpha query"}],
                )
                write_table(
                    source / "data" / "test-00000-of-00001.parquet",
                    [{"query-id": "Q1", "corpus-id": "D/1", "score": 1}],
                )

                result = build_workspace(
                    task,
                    data_root=tmp_path / "raw",
                    output_root=tmp_path / "workspaces",
                    metadata_root=tmp_path / "metadata",
                )

                doc_path = result.workspace_dir / "corpus" / doc_id_to_filename("D/1")
                self.assertTrue((result.workspace_dir / "README.md").exists())
                self.assertTrue(doc_path.read_text(encoding="utf-8").startswith("# First"))
                self.assertFalse((result.workspace_dir / "data").exists())
                self.assertFalse((result.workspace_dir / "queries").exists())
                self.assertFalse(list(result.workspace_dir.rglob("*qrels*")))
                self.assertTrue((result.metadata_dir / "qrels.json").exists())
                self.assertTrue((result.metadata_dir / "queries.jsonl").exists())
                manifest = (result.metadata_dir / "manifest.json").read_text(encoding="utf-8")
                self.assertIn('"num_docs": 2', manifest)
                self.assertIn('"num_queries": 1', manifest)
