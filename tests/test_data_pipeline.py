"""Integration invariants for canonical releases and the existing DCI protocol."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from dci_bench.data.pipeline.core import (
    check_roots, export_dci, load_bundle, load_config, publish_bundle, read_json,
    select_queries, verify_release, write_dashboard,
)
from dci_bench.data.pipeline.connectors import _spec_fingerprint


def bundle(kind="retrieval"):
    docs = [{"_id": "source/a", "title": "Mars", "text": "Mars has two moons: Phobos and Deimos.",
             "language": "en", "metadata": {"translation_status": "native"}},
            {"_id": "source/b", "title": "Venus", "text": "Venus has no natural moons.",
             "language": "en", "metadata": {"translation_status": "native"}}]
    return {"schema_version": "dci-data-bundle-v1", "input_kind": kind,
            "source": {"dataset": "test/astronomy", "revision": "abc123"},
            "documents": docs,
            "queries": [] if kind == "corpus" else [
                {"_id": "q/1", "text": "Which planet has Phobos and Deimos?", "language": "en", "metadata": {}}],
            "qrels": {} if kind == "corpus" else {"q/1": {"source/a": 2}},
            "evidence": [] if kind == "corpus" else [
                {"query_id": "q/1", "doc_id": "source/a", "quote": "Mars has two moons: Phobos and Deimos."}],
            "judgments": [], "provenance": {"domain": "science", "language": "en", "genre": "encyclopedia"}}


def test_canonical_roundtrip_no_overwrite_and_extra_file_detection(tmp_path):
    source = bundle()
    before = copy.deepcopy(source)
    target = tmp_path / "release"
    m = publish_bundle(source, target)
    assert source == before
    assert m["state"] == "labels_ready"
    assert verify_release(target) == m
    assert load_bundle(target)["qrels"] == source["qrels"]
    with pytest.raises(FileExistsError):
        publish_bundle(source, target)
    (target / "unexpected").write_text("data")
    with pytest.raises(ValueError, match="inventory"):
        verify_release(target)


@pytest.mark.parametrize("which", ["documents.jsonl", "queries.jsonl", "qrels.json", "audit.json"])
def test_changed_artifacts_fail_verification(tmp_path, which):
    target = tmp_path / "release"
    publish_bundle(bundle(), target)
    with (target / which).open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="inventory"):
        verify_release(target)


def test_corpus_completion_does_not_claim_query_or_eval_readiness(tmp_path):
    target = tmp_path / "corpus"
    m = publish_bundle(bundle("corpus"), target)
    assert m["state"] == "corpus_ready"
    assert m["num_queries"] == 0
    with pytest.raises(ValueError, match="cannot be evaluated"):
        export_dci(target, tmp_path / "dci", "astronomy")


def test_corpus_with_only_some_queries_labelled_cannot_export(tmp_path):
    b = bundle()
    b["input_kind"] = "corpus"
    b["queries"].append({"_id": "unlabelled", "text": "An unfinished query", "language": "en", "metadata": {}})
    result = publish_bundle(b, tmp_path / "partial")
    assert result["state"] == "corpus_ready"
    with pytest.raises(ValueError, match="cannot be evaluated"):
        export_dci(tmp_path / "partial", tmp_path / "export", "partial")


def test_article_groups_do_not_split_within_a_release(tmp_path):
    from dci_bench.data.pipeline.quality import assign_splits
    b = bundle("corpus")
    for row in b["documents"]:
        row["metadata"]["group_id"] = "same-original-article"
    splits = assign_splits(b)
    assert len(set(splits["document_splits"].values())) == 1


def test_corpus_queries_without_qrels_remain_corpus_ready(tmp_path):
    b = bundle("corpus")
    b["queries"] = [{"_id": "q/seed", "text": "A query without labels", "language": "en", "metadata": {}}]
    manifest = publish_bundle(b, tmp_path / "corpus")
    assert manifest["state"] == "corpus_ready"


def test_raw_snapshot_source_identity_is_checked_but_memory_fixture_is_allowed(tmp_path):
    spec = {
        "id": "astronomy-source", "connector": "generic_jsonl", "input_kind": "retrieval",
        "revision": "r1", "options": {"dataset": "test/astronomy"},
    }
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "source_manifest.json").write_text(json.dumps({
        "schema_version": "dci-source-snapshot-v1", "complete": True,
        "id": spec["id"], "connector": spec["connector"], "input_kind": spec["input_kind"],
        "requested_spec_fingerprint": _spec_fingerprint(spec), "spec_fingerprint": _spec_fingerprint(spec),
        "requested_spec": spec, "resolved_revision": "r1", "files": [],
    }))
    source_bundle = bundle()
    source_bundle["source"] = {"dataset": "test/astronomy", "revision": "r1", "connector": "generic_jsonl"}
    source_bundle["provenance"]["source_id"] = spec["id"]
    publish_bundle(source_bundle, tmp_path / "matching", specification=spec, raw_snapshot=raw)
    bad = copy.deepcopy(source_bundle)
    bad["source"]["revision"] = "wrong"
    with pytest.raises(ValueError, match="source dataset/revision"):
        publish_bundle(bad, tmp_path / "mismatch", specification=spec, raw_snapshot=raw)
    # Existing unit fixtures intentionally have no raw snapshot closure.
    assert publish_bundle(bundle(), tmp_path / "memory-only")["state"] == "labels_ready"


def test_parent_release_source_identity_cannot_be_reused_for_another_dataset(tmp_path):
    parent = tmp_path / "parent"
    publish_bundle(bundle(), parent)
    child = bundle()
    child["source"]["dataset"] = "different/dataset"
    with pytest.raises(ValueError, match="Parent release source dataset"):
        publish_bundle(child, tmp_path / "child", parent_release=parent)


def test_export_keeps_supervision_host_only_and_pools_share_documents(tmp_path):
    from dci_bench.data.document_pools import prepare_document_pools, verify_document_pool_suite
    b = bundle("evidence_qa")
    b["queries"][0]["metadata"]["answer"] = "HOST_ONLY_CANARY"
    b["queries"].append({"_id": "q/2", "text": "Which planet has no moons?", "language": "en", "metadata": {}})
    b["qrels"]["q/2"] = {"source/b": 1}
    target = tmp_path / "release"
    publish_bundle(b, target)
    m = export_dci(target, tmp_path / "dci", "astronomy")
    assert m["num_docs"] == 2
    assert m["num_queries"] == 2
    for p in (tmp_path / "dci/workspaces").rglob("*"):
        if p.is_file():
            assert "HOST_ONLY_CANARY" not in p.read_text()
    result = prepare_document_pools(tmp_path / "dci", tmp_path / "pools", "astronomy", capacity=200)
    verify_document_pool_suite(result.suite_root, task="astronomy")
    pool = read_json(result.candidate_pools_path)
    assert len(list((result.workspace_dir / "corpus").glob("*.md"))) == 2
    assert all(row["num_docs"] == 2 for row in pool["pools"].values())
    assert all(row["candidate_scores"] is None for row in pool["pools"].values())


def test_selection_preserves_full_corpus_and_selected_qrels(tmp_path):
    b = bundle()
    for i in range(10):
        qid = f"q/{i + 2}"
        b["queries"].append({"_id": qid, "text": f"Question {i}", "language": "en", "metadata": {}})
        b["qrels"][qid] = {"source/a": 1, "source/b": 2}
    selected = select_queries(b, limit=3, seed=42)
    assert selected == select_queries(b, limit=3, seed=42)
    assert selected["documents"] == b["documents"]
    assert len(selected["queries"]) == 3
    for q in selected["queries"]:
        assert selected["qrels"][q["_id"]] == b["qrels"][q["_id"]]


def test_failed_write_preserved_without_success_marker(tmp_path):
    import dci_bench.data.pipeline.core as core
    original = core.write_json
    def fail_once(path, data):
        if path.name == "manifest.json":
            raise OSError("simulated disk failure")
        original(path, data)
    with patch.object(core, "write_json", side_effect=fail_once), pytest.raises(OSError):
        publish_bundle(bundle(), tmp_path / "release")
    assert not (tmp_path / "release").exists()
    stages = list(tmp_path.glob(".release.staging-*"))
    assert len(stages) == 1
    assert (stages[0] / "_FAILED.json").is_file()
    assert not (stages[0] / "_COMPLETE.json").exists()


def test_roots_duplicate_json_and_symlink_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        check_roots(tmp_path, tmp_path / "clean")
    path = tmp_path / "duplicate.json"
    path.write_text('{"sources":[],"sources":[]}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(path)
    path.write_text(json.dumps({"sources": [], "raw_root": {"escape": True}}))
    with pytest.raises(ValueError, match="filesystem paths"):
        load_config(path)
    publish_bundle(bundle(), tmp_path / "release")
    (tmp_path / "linked").symlink_to(tmp_path / "release", target_is_directory=True)
    with pytest.raises(ValueError):
        verify_release(tmp_path / "linked")


def test_automatic_labels_need_explicit_export_policy(tmp_path):
    b = bundle()
    b["provenance"]["generation"] = {"model": "mock-generator"}
    manifest = publish_bundle(b, tmp_path / "release")
    assert manifest["state"] == "review_required"
    with pytest.raises(ValueError, match="cannot be evaluated"):
        export_dci(tmp_path / "release", tmp_path / "export", "test")


def _judged_model_bundle():
    b = bundle()
    b["queries"][0]["metadata"]["generated_by"] = "llm"
    b["provenance"]["models"] = {
        "generate": {"model": "mock-generator", "revision": "g1"},
        "judge": {"model": "mock-judge", "revision": "j1"},
    }
    b["judgments"] = [{
        "query_id": "q/1", "doc_id": "source/a", "status": "judged", "score": 2,
        "quotes": ["Mars has two moons: Phobos and Deimos."], "source": "llm_judge",
    }]
    return b


def test_model_qrels_need_judged_positive_exact_quote_and_allow_policy(tmp_path):
    b = _judged_model_bundle()
    manifest = publish_bundle(b, tmp_path / "release")
    assert manifest["state"] == "labels_ready"
    with pytest.raises(ValueError, match="Automatic labels"):
        export_dci(tmp_path / "release", tmp_path / "suite", "model-task")
    export_dci(tmp_path / "release", tmp_path / "suite", "model-task", allow_automatic_labels=True)

    missing = _judged_model_bundle()
    missing["judgments"][0]["quotes"] = []
    missing_manifest = publish_bundle(missing, tmp_path / "missing")
    assert missing_manifest["state"] == "review_required"

    # A model judgment added to a human query does not force its pre-existing
    # human positive to be rejudged.
    mixed = _judged_model_bundle()
    mixed["queries"][0]["metadata"] = {}
    mixed["qrels"]["q/1"]["source/b"] = 1
    mixed["judgments"][0].update({"doc_id": "source/b", "quotes": ["Venus has no natural moons."]})
    mixed_manifest = publish_bundle(mixed, tmp_path / "mixed")
    assert mixed_manifest["state"] == "labels_ready"


def test_fake_model_marker_and_review_queue_cannot_claim_ready(tmp_path):
    fake = bundle()
    fake["provenance"]["label_source"] = "model"
    fake_manifest = publish_bundle(fake, tmp_path / "fake")
    assert fake_manifest["state"] == "review_required"

    queued = _judged_model_bundle()
    queued["review_queue"] = [{"query_id": "q/1", "doc_id": "source/a", "reason": "manual review"}]
    queued_manifest = publish_bundle(queued, tmp_path / "queued")
    assert queued_manifest["state"] == "review_required"


def test_export_reuses_suite_for_distinct_tasks_but_rejects_reentry_and_symlinks(tmp_path):
    release = tmp_path / "release"
    publish_bundle(bundle(), release)
    suite = tmp_path / "suite"
    export_dci(release, suite, "task-one")
    export_dci(release, suite, "task-two")
    with pytest.raises(FileExistsError, match="existing task"):
        export_dci(release, suite, "task-one")
    linked = tmp_path / "linked-suite"
    linked.symlink_to(suite, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        export_dci(release, linked, "task-three")


def test_dashboard_escapes_script_payload(tmp_path):
    b = bundle()
    b["provenance"]["source_id"] = "</script><script>alert(1)</script>"
    publish_bundle(b, tmp_path / "release")
    report = write_dashboard([tmp_path / "release"], tmp_path / "report")
    text = (tmp_path / "report/index.html").read_text()
    assert "</script><script>alert(1)" not in text
    assert report["releases"][0]["queries"] == 1
