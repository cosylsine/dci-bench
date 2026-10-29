import gzip
import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

from dci_bench.data.pipeline.connectors import (
    ConnectorError,
    SCHEMA_VERSION,
    SnapshotConflictError,
    fetch_source,
    list_connectors,
    load_source,
    _worksurface_span_locator,
)
from dci_bench.data.pipeline.quality import prepare_bundle


def _spec(source_id, connector, input_kind, options, **extra):
    return {
        "id": source_id,
        "connector": connector,
        "input_kind": input_kind,
        "domain": extra.pop("domain", "test"),
        "language": extra.pop("language", "en"),
        "revision": extra.pop("revision", "fixture"),
        "license": extra.pop("license", "fixture-license"),
        "options": options,
        **extra,
    }


class ConnectorTest(unittest.TestCase):
    def test_registry_is_complete_and_stable(self):
        self.assertEqual(
            list_connectors(),
            ["beir", "edgar", "eurlex", "generic_jsonl", "github_markdown", "pmc_jats", "qasper", "squad", "wixqa", "worksurface"],
        )

    def test_generic_retrieval_bundle_preserves_ids_and_manifest_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "documents.jsonl").write_text(
                json.dumps({"_id": "native/doc", "title": "Title", "text": "Body", "label": "metadata-only"}) + "\n",
                encoding="utf-8",
            )
            (source / "queries.jsonl").write_text(json.dumps({"_id": "q/1", "text": "Question"}) + "\n", encoding="utf-8")
            (source / "qrels.json").write_text(json.dumps({"q/1": {"native/doc": 2}}), encoding="utf-8")
            spec = _spec("generic", "generic_jsonl", "retrieval", {"path": str(source)})
            snapshot = fetch_source(spec, root / "raw")
            bundle = load_source(spec, snapshot)
            self.assertEqual(bundle["schema_version"], SCHEMA_VERSION)
            self.assertEqual(bundle["documents"][0]["_id"], "native/doc")
            self.assertEqual(bundle["queries"][0]["_id"], "q/1")
            self.assertEqual(bundle["qrels"], {"q/1": {"native/doc": 2.0}})
            self.assertNotIn("label", bundle["documents"][0]["text"])
            self.assertEqual(fetch_source(spec, root / "raw"), snapshot)
            (snapshot / "data" / "documents.jsonl").write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(SnapshotConflictError):
                load_source(spec, snapshot)

    def test_corpus_has_no_synthetic_queries_or_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "docs"
            source.mkdir()
            (source / "corpus.jsonl").write_text(json.dumps({"_id": "d1", "text": "only source"}) + "\n")
            spec = _spec("corpus", "generic_jsonl", "corpus", {"path": str(source)})
            bundle = load_source(spec, fetch_source(spec, root / "raw"))
            self.assertEqual(len(bundle["documents"]), 1)
            self.assertEqual(bundle["queries"], [])
            self.assertEqual(bundle["qrels"], {})

    def test_wixqa_split_filter_uses_snapshot_relative_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "wix"
            (source / "wix_kb_corpus").mkdir(parents=True)
            (source / "wixqa_expertwritten").mkdir()
            (source / "wixqa_simulated").mkdir()
            (source / "wix_kb_corpus/wix_kb_corpus.jsonl").write_text(
                json.dumps({"id": "d1", "title": "Doc", "contents": "body"}) + "\n",
                encoding="utf-8",
            )
            (source / "wixqa_expertwritten/test.jsonl").write_text(
                json.dumps({"question": "expert", "article_ids": ["d1"]}) + "\n",
                encoding="utf-8",
            )
            (source / "wixqa_simulated/test.jsonl").write_text(
                json.dumps({"question": "simulated", "article_ids": ["d1"]}) + "\n",
                encoding="utf-8",
            )
            spec = _spec("wixqa_expertwritten", "wixqa", "retrieval", {"path": str(source), "split": "expertwritten"}, revision="d662dc42479c14e202eccd832f8c4b66a035c4cc")
            bundle = load_source(spec, fetch_source(spec, root / "raw"))
            self.assertEqual(len(bundle["queries"]), 1)
            self.assertEqual(bundle["queries"][0]["metadata"]["split"], "expertwritten")

    def test_beir_archive_split_and_qrels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "scifact.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("corpus.jsonl", json.dumps({"_id": "d1", "title": "T", "text": "text"}) + "\n")
                handle.writestr(
                    "queries.jsonl",
                    json.dumps({"_id": "q1", "text": "question"}) + "\n"
                    + json.dumps({"_id": "q2", "text": "train-only"}) + "\n",
                )
                handle.writestr("qrels/test.tsv", "q1\t0\td1\t1\n")
                handle.writestr("qrels/train.tsv", "q2\t0\td1\t1\n")
            spec = _spec("beir", "beir", "retrieval", {"archive": str(archive), "split": "test"})
            bundle = load_source(spec, fetch_source(spec, root / "raw"))
            self.assertEqual(bundle["qrels"], {"q1": {"d1": 1.0}})
            self.assertEqual([query["_id"] for query in bundle["queries"]], ["q1"])
            self.assertEqual(bundle["provenance"]["queries_without_target_split_qrels"], 1)

    def test_qasper_exact_evidence_and_missing_evidence_excludes_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "qasper.json.gz"
            row = {
                "id": "paper-1",
                "title": "Paper",
                "full_text": [{"section_name": "Intro", "paragraphs": ["Exact evidence.", "Another paragraph."]}],
                "qas": [{"question": "What?", "answers": [{"answer": "answer text", "evidence": ["Exact evidence.", "Another paragraph."]}]}],
            }
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                json.dump([row], handle)
            spec = _spec("qasper", "qasper", "evidence_qa", {"files": [str(source)]})
            bundle = load_source(spec, fetch_source(spec, root / "raw"))
            self.assertEqual(bundle["qrels"], {"paper-1::q1": {"paper-1": 1.0}})
            self.assertEqual(len(bundle["evidence"]), 2)
            self.assertNotIn("answer text", bundle["documents"][0]["text"])

            row["qas"][0]["answers"][0]["evidence"] = ["not in paper"]
            bad = root / "bad.json.gz"
            with gzip.open(bad, "wt", encoding="utf-8") as handle:
                json.dump([row], handle)
            bad_spec = _spec("qasper-bad", "qasper", "evidence_qa", {"files": [str(bad)]})
            bad_bundle = load_source(bad_spec, fetch_source(bad_spec, root / "raw"))
            self.assertEqual(bad_bundle["queries"], [])
            self.assertEqual(bad_bundle["qrels"], {})
            self.assertEqual(bad_bundle["provenance"]["excluded"][0]["reason"], "unmapped_evidence")
            self.assertEqual(bad_bundle["provenance"]["excluded"][0]["original_evidence"], ["not in paper"])

    def test_qasper_official_keyed_nested_answers_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "qasper.json"
            source.write_text(
                json.dumps(
                    {
                        "paper-1": {
                            "title": "Paper",
                            "abstract": "Abstract text.",
                            "full_text": {
                                "section_name": ["Intro", "Results"],
                                "paragraphs": [["Exact paragraph."], ["Second paragraph."]],
                            },
                            "qas": [
                                {
                                    "question": "What is supported?",
                                    "question_id": "q-valid",
                                    "answers": [
                                        {
                                            "annotation_id": "a1",
                                            "answer": {
                                                "unanswerable": False,
                                                "extractive_spans": ["Exact"],
                                                "free_form_answer": "Exact",
                                                "evidence": ["Intro", "Exact paragraph."],
                                            },
                                        },
                                        {
                                            "annotation_id": "a2",
                                            "answer": {
                                                "unanswerable": False,
                                                "extractive_spans": ["Second"],
                                                "free_form_answer": "Second",
                                                "evidence": ["Second paragraph."],
                                            },
                                        },
                                    ],
                                },
                                {
                                    "question": "No answer?",
                                    "question_id": "q-unanswerable",
                                    "answers": [
                                        {
                                            "annotation_id": "a3",
                                            "answer": {"unanswerable": True, "evidence": []},
                                        }
                                    ],
                                },
                                {
                                    "question": "Figure only?",
                                    "question_id": "q-figure",
                                    "answers": [
                                        {
                                            "annotation_id": "a4",
                                            "answer": {
                                                "unanswerable": False,
                                                "evidence": ["FLOAT SELECTED: Figure 1"],
                                            },
                                        }
                                    ],
                                },
                                {
                                    "question": "No text?",
                                    "question_id": "q-no-text",
                                    "answers": [
                                        {
                                            "annotation_id": "a5",
                                            "answer": {"unanswerable": False, "evidence": []},
                                        }
                                    ],
                                },
                                {
                                    "question": "Which heading is present?",
                                    "question_id": "q-heading",
                                    "answers": [
                                        {
                                            "annotation_id": "a6",
                                            "answer": {
                                                "unanswerable": False,
                                                "evidence": ["Intro"],
                                            },
                                        }
                                    ],
                                },
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            spec = _spec("qasper-official", "qasper", "evidence_qa", {"files": [str(source)]})
            bundle = load_source(spec, fetch_source(spec, root / "raw"))
            self.assertEqual([item["_id"] for item in bundle["queries"]], ["q-heading"])
            self.assertEqual(bundle["qrels"], {"q-heading": {"paper-1": 1.0}})
            self.assertEqual({item["quote"] for item in bundle["evidence"]}, {"Intro"})
            self.assertEqual(
                {item["reason"] for item in bundle["provenance"]["excluded"]},
                {"annotation_disagreement", "unanswerable", "figure_only", "no_text_evidence"},
            )
            disagreement = next(item for item in bundle["provenance"]["excluded"] if item["reason"] == "annotation_disagreement")
            self.assertEqual(disagreement["query_id"], "q-valid")
            self.assertEqual(disagreement["valid_annotation_ids"], ["a1", "a2"])

    def test_qasper_official_dev_snapshot_counts_and_exclusions(self):
        default_snapshot = Path(
            "/mnt/afs2/202609/embedding_models/dci-bench/raw_data/qasper_dev/"
            "2afb05053bce7c7aa61add963277d9222c5507806da98a8bdd275efaf6c889d4"
        )
        snapshot = Path(os.environ.get("DCI_QASPER_DEV_SNAPSHOT", default_snapshot))
        manifest_path = snapshot / "source_manifest.json"
        if not manifest_path.is_file():
            self.skipTest(f"official QASPER dev snapshot is unavailable: {snapshot}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle = load_source(manifest["requested_spec"], snapshot)
        self.assertEqual(
            (len(bundle["documents"]), len(bundle["queries"]), len(bundle["qrels"]), len(bundle["evidence"])),
            (281, 465, 465, 1075),
        )
        self.assertEqual(len(bundle["provenance"]["excluded"]), 540)
        self.assertEqual(
            Counter(item["reason"] for item in bundle["provenance"]["excluded"]),
            Counter({
                "annotation_disagreement": 388,
                "unanswerable": 60,
                "unmapped_evidence": 39,
                "no_text_evidence": 30,
                "figure_only": 23,
            }),
        )
        self.assertEqual(bundle["provenance"]["annotation_review"], [])
        self.assertIn("section headings", bundle["provenance"]["evidence_policy"])

    def test_worksurface_maps_native_span_and_preserves_unresolved_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "worksurface"
            docs = source / "resources/profiles/backend_developer/kb_docs"
            docs.mkdir(parents=True)
            text = "Version 10\n# Dependency Management Report 10\nPython 3.11\n"
            (docs / "doc.md").write_text(text, encoding="utf-8")
            data = source / "data"
            data.mkdir()
            questions = [("unique", "Which Python?", "3.11"),
                         ("context", "Complete: “# Dependency Management Report ___”?", "10"),
                         ("ambiguous", "Which version?", "10")]
            rows = [{"id": qid, "source": {"persona": "Backend Developer"},
                     "question": question, "task_type": "rag_only", "required_surfaces": ["rag"],
                     "gold_evidence": [{"surface": "rag", "file": "doc.md", "span": span, "claim": "verbatim"}]}
                    for qid, question, span in questions]
            (data / "tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            spec = _spec("worksurface-span", "worksurface", "evidence_qa", {"path": str(source), "persona": "Backend Developer"})
            bundle = load_source(spec, fetch_source(spec, root / "raw"))
            self.assertEqual(len(bundle["evidence"]), 3)
            cleaned, audit = prepare_bundle(bundle)
            self.assertEqual(len(cleaned["evidence"]), 2)
            self.assertEqual(cleaned["qrels"], bundle["qrels"])
            self.assertEqual(audit["status"], "review_required")
            self.assertEqual(cleaned["review_queue"][0]["query_id"], "ambiguous")
            contextual = next(item for item in cleaned["evidence"] if item["query_id"] == "context")
            self.assertEqual(contextual["start"], text.rindex("10"))
            for item in cleaned["evidence"]:
                self.assertEqual(text[item["start"]:item["end"]], item["quote"])

    def test_worksurface_locator_never_guesses_ambiguous_context(self):
        self.assertEqual(_worksurface_span_locator("Complete “A ___”", "A 10\nA 10", "10"), {})
        self.assertEqual(_worksurface_span_locator("Complete “___ ___”", "10 10", "10"), {})
        self.assertEqual(_worksurface_span_locator("Complete “A ___” or “B ___”", "A 10 B 10", "10"), {})

    def test_worksurface_ignores_hidden_partial_tasks_and_excludes_unlabelled_equivalents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "worksurface"
            docs = source / "resources/profiles/operations_manager/kb_docs"
            docs.mkdir(parents=True)
            header1 = "<!-- source_task: 1 | source_file: shared.md | surface: rag -->\n\n"
            header2 = "<!-- source_task: 2 | source_file: shared.md | surface: rag -->\n\n"
            (docs / "t1__shared.md").write_text(header1 + "same body", encoding="utf-8")
            (docs / "t2__shared.md").write_text(header2 + "same body", encoding="utf-8")
            data = source / "data"
            data.mkdir()
            (data / "tasks.jsonl").write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "source": {"persona": "Operations Manager"},
                        "question": "Which file?",
                        "task_type": "rag_only",
                        "required_surfaces": ["rag"],
                        "gold_evidence": [{"surface": "rag", "file": "t1__shared.md", "claim": "claim"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            partial = source / ".hfd/partials-before-http2-switch/data"
            partial.mkdir(parents=True)
            (partial / "tasks.jsonl").write_text("{not-json\n", encoding="utf-8")
            spec = _spec(
                "worksurface-equivalent",
                "worksurface",
                "evidence_qa",
                {"path": str(source), "persona": "Operations Manager"},
                revision="3b3aef45f1e8f69416d6112c46e6e873cf522315",
            )
            snapshot = fetch_source(spec, root / "raw")
            bundle = load_source(spec, snapshot)
            self.assertEqual(len(bundle["documents"]), 2)
            self.assertEqual(bundle["queries"], [])
            self.assertEqual(bundle["qrels"], {})
            self.assertEqual(bundle["provenance"]["excluded"][0]["reason"], "unlabelled_equivalent_documents")
            manifest = json.loads((snapshot / "source_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(any(".hfd" in item["path"] for item in manifest["files"]))

    def test_github_translation_metadata_and_root_genre(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "repo/content/zh-cn/docs/concepts"
            source.mkdir(parents=True)
            (source / "example.md").write_text("# Example\n", encoding="utf-8")
            spec = _spec(
                "github-meta",
                "github_markdown",
                "corpus",
                {
                    "path": str(root / "repo"),
                    "paths": ["content/zh-cn/docs"],
                    "extensions": [".md"],
                    "repo": "kubernetes/website",
                    "translation_status": "translated_parallel",
                },
                revision="bc7f3479bc8b0af48097b155c22c8d58e8e1afe4",
                genre="technical_documentation",
                language="zh",
            )
            bundle = load_source(spec, fetch_source(spec, root / "raw"))
            metadata = bundle["documents"][0]["metadata"]
            self.assertEqual(metadata["translation_status"], "translated_parallel")
            self.assertEqual(metadata["translation_of"], "content/en/docs/concepts/example.md")
            self.assertEqual(metadata["family_id"], "kubernetes/website:docs/concepts/example.md")
            self.assertEqual(bundle["provenance"]["genre"], "technical_documentation")

    def test_fixed_revision_and_secret_redaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "d.jsonl"
            source.write_text(json.dumps({"_id": "d", "text": "x"}) + "\n")
            floating = _spec("hf", "wixqa", "corpus", {"files": [str(source)], "split": "expertwritten"}, revision="main")
            with self.assertRaises(ConnectorError):
                fetch_source(floating, root / "raw")

            docs = root / "documents.jsonl"
            queries = root / "queries.jsonl"
            qrels = root / "qrels.json"
            docs.write_text(json.dumps({"_id": "d", "text": "x"}) + "\n")
            queries.write_text(json.dumps({"_id": "q", "text": "x"}) + "\n")
            qrels.write_text(json.dumps({"q": {"d": 1}}))
            spec = _spec(
                "secret",
                "generic_jsonl",
                "retrieval",
                {
                    "files": {
                        "documents": {"path": str(docs)},
                        "queries": {"path": str(queries)},
                        "qrels": {"path": str(qrels)},
                    },
                    "headers": {"Authorization": "Bearer dont-write"},
                },
            )
            snapshot = fetch_source(spec, root / "raw")
            manifest = (snapshot / "source_manifest.json").read_text(encoding="utf-8")
            self.assertNotIn("dont-write", manifest)

    def test_html_strips_script_and_pmc_requires_actual_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            html_path = root / "filing.html"
            html_path.write_text("<html><title>Filing</title><script>SECRET</script><p>Visible text</p></html>")
            html_spec = _spec("filing", "edgar", "corpus", {"files": [str(html_path)]})
            bundle = load_source(html_spec, fetch_source(html_spec, root / "raw"))
            self.assertEqual(bundle["documents"][0]["title"], "Filing")
            self.assertIn("Visible text", bundle["documents"][0]["text"])
            self.assertNotIn("SECRET", bundle["documents"][0]["text"])

            xml = root / "article.xml"
            xml.write_text("<article><front><article-meta><article-title>T</article-title></article-meta></front><body><p>Text</p></body></article>")
            pmc_spec = _spec("pmc", "pmc_jats", "corpus", {"files": [str(xml)]})
            with self.assertRaises(ConnectorError):
                load_source(pmc_spec, fetch_source(pmc_spec, root / "raw"))
