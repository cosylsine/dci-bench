import json
from pathlib import Path
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from dci_bench.data.loader import load_qrels
from dci_bench.data.registry import TaskSpec
from dci_bench.data.workspace_builder import (
    CORPUS_MAX_LINE_CHARS,
    WORKSPACE_FORMAT_VERSION,
    assert_pinned_revision,
    build_workspace,
    doc_id_to_filename,
    render_doc_markdown,
    wrap_corpus_text,
)


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


class WorkspaceBuilderTest(unittest.TestCase):
    @staticmethod
    def _write_repo_metadata(source: Path, task: TaskSpec) -> None:
        metadata = source / ".hfd" / "repo_metadata.json"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            '{"id": "' + task.hf_dataset + '", "sha": "' + task.revision + '-resolved"}\n',
            encoding="utf-8",
        )

    def test_doc_id_filename_is_reversible_for_unicode_ids(self):
        self.assertEqual(doc_id_to_filename("pégase_23_6"), "p%C3%A9gase_23_6.md")
        self.assertEqual(doc_id_to_filename("D/1"), "D%2F1.md")

    def test_render_doc_markdown_preserves_text_trailing_whitespace(self):
        self.assertEqual(render_doc_markdown("", "alpha text \n"), "alpha text \n")

    def test_wrap_corpus_text_bounds_lines_and_retains_source_characters(self):
        source = "x" * 11 + " alpha beta gamma delta"
        wrapped = wrap_corpus_text(source, width=11)
        self.assertTrue(all(len(line) <= 11 for line in wrapped.splitlines()))
        self.assertEqual(wrapped.replace("\n", ""), source)

    def test_render_doc_markdown_wraps_long_corpus_lines(self):
        rendered = render_doc_markdown("Title", "word " * 200)
        self.assertLessEqual(max(map(len, rendered.splitlines())), CORPUS_MAX_LINE_CHARS)

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

    def test_missing_revision_metadata_fails_closed(self):
        import tempfile

        task = TaskSpec("ToyTask", "mteb/toy", "abc123", "toy")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "Missing revision metadata"):
                assert_pinned_revision(task, Path(tmp))

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
                    task_instruction="Search the toy concept and stop after verification.",
                )
                source = tmp_path / "raw" / task.local_dir_name
                self._write_repo_metadata(source, task)
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
                self.assertIn('"resolved_revision": "test-revision-resolved"', manifest)
                self.assertIn(f'"version": "{WORKSPACE_FORMAT_VERSION}"', manifest)
                manifest_payload = json.loads(manifest)
                self.assertEqual(
                    manifest_payload["task_instruction"],
                    "Search the toy concept and stop after verification.",
                )
                self.assertEqual(
                    manifest_payload["protocol"]["task_context"],
                    {
                        "task_name": "ToyTask",
                        "instruction": "Search the toy concept and stop after verification.",
                    },
                )

                reused = build_workspace(
                    task,
                    data_root=tmp_path / "raw",
                    output_root=tmp_path / "workspaces",
                    metadata_root=tmp_path / "metadata",
                    skip_existing=True,
                )
                self.assertEqual(reused.manifest_path, result.manifest_path)

    def test_refresh_existing_rebuilds_an_older_workspace_format(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task = TaskSpec("ToyTask", "mteb/toy", "abc123", "toy")
            source = tmp_path / "raw" / task.local_dir_name
            self._write_repo_metadata(source, task)
            write_table(
                source / "corpus" / "corpus-00000-of-00001.parquet",
                [{"_id": "D1", "title": "", "text": "word " * 200}],
            )
            write_table(
                source / "queries" / "queries-00000-of-00001.parquet",
                [{"_id": "Q1", "text": "query"}],
            )
            write_table(
                source / "data" / "test-00000-of-00001.parquet",
                [{"query-id": "Q1", "corpus-id": "D1", "score": 1}],
            )
            first = build_workspace(
                task,
                data_root=tmp_path / "raw",
                output_root=tmp_path / "workspaces",
                metadata_root=tmp_path / "metadata",
            )
            manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
            manifest.pop("workspace_format")
            first.manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            document = first.workspace_dir / "corpus" / "D1.md"
            document.write_text("legacy single line", encoding="utf-8")

            refreshed = build_workspace(
                task,
                data_root=tmp_path / "raw",
                output_root=tmp_path / "workspaces",
                metadata_root=tmp_path / "metadata",
                refresh_existing=True,
            )

            self.assertNotEqual(document.read_text(encoding="utf-8"), "legacy single line")
            refreshed_manifest = json.loads(refreshed.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(refreshed_manifest["workspace_format"]["version"], WORKSPACE_FORMAT_VERSION)
