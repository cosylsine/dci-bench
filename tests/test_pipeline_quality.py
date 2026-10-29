from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from dci_bench.data.pipeline.quality import (
    assign_splits,
    audit_bundle,
    coverage_report,
    prepare_bundle,
    write_report,
)


def make_bundle(*, input_kind: str = "retrieval") -> dict:
    return {
        "schema_version": "dci-data-bundle-v1",
        "input_kind": input_kind,
        "source": {"dataset": "toy", "revision": "r1", "url": "host-only"},
        "documents": [
            {
                "_id": "d1",
                "title": "One",
                "text": "Paris is the capital of France.",
                "language": "en",
                "metadata": {"domain": "geo", "family_id": "f1", "annotation_source": "source"},
            },
            {
                "_id": "d2",
                "title": "Two",
                "text": "Berlin is the capital of Germany.",
                "language": "de",
                "metadata": {"domain": "geo", "native": True, "annotation_source": "model"},
            },
        ],
        "queries": [
            {
                "_id": "q1",
                "text": "What is the capital of France?",
                "language": "en",
                "metadata": {"family_id": "f1", "annotation_source": "source"},
            }
        ],
        "qrels": {"q1": {"d1": 1}},
        "evidence": [{"query_id": "q1", "doc_id": "d1", "quote": "capital of France", "start": 13, "end": 30}],
        "judgments": [],
        "provenance": {"domain": "geo", "language": "en", "genre": "fact"},
    }


def test_valid_audit_is_serializable_and_preserves_source_origin() -> None:
    bundle = make_bundle()
    before = copy.deepcopy(bundle)
    report = audit_bundle(bundle)

    assert report["status"] == "pass"
    assert report["valid"] is True
    assert report["errors"] == []
    assert report["statistics"]["documents"]["annotation_source"]["source"] == 1
    assert report["statistics"]["documents"]["annotation_source"].get("human", 0) == 0
    assert bundle == before
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def test_schema_duplicates_nan_bool_unknown_gold_and_query_consistency_are_reported() -> None:
    bundle = make_bundle()
    bundle["documents"].append(copy.deepcopy(bundle["documents"][0]))
    bundle["documents"][-1]["text"] = "conflicting body"
    bundle["queries"].append({"_id": "q2", "text": "unlabelled"})
    bundle["qrels"]["q1"].update({"missing": float("nan"), "d2": True})
    report = audit_bundle(bundle)

    codes = {item["code"] for item in report["errors"]}
    assert report["status"] == "invalid"
    assert {"duplicate_document_id", "unknown_qrel_document", "invalid_qrel_score", "query_without_qrels"} <= codes
    assert report["unknown_qrel_documents"]["q1"] == ["missing"]


def test_evidence_requires_exact_locator_and_rejects_ambiguous_or_mismatched_offsets() -> None:
    bundle = make_bundle()
    bundle["documents"][0]["text"] = "needle appears; needle appears"
    bundle["evidence"] = [
        {"query_id": "q1", "doc_id": "d1", "quote": "needle"},
        {"query_id": "q1", "doc_id": "d1", "quote": "needle", "start": 0, "end": 6},
        {"query_id": "q1", "doc_id": "d1", "quote": "needle", "start": 1, "end": 7},
        {"query_id": "q1", "doc_id": "d1", "start": 0, "end": 99},
    ]
    report = audit_bundle(bundle)
    codes = {item["code"] for item in report["errors"]}

    assert "ambiguous_evidence_quote" in codes
    assert "evidence_offset_mismatch" in codes
    assert "evidence_offset_bounds" in codes
    assert report["evidence_report"]["ambiguous"] == 1
    assert report["review_required"]


def test_prepare_excludes_whole_query_when_positive_doc_is_oversized() -> None:
    bundle = make_bundle()
    bundle["documents"].append({"_id": "huge", "text": "x" * (1_048_576 + 1), "metadata": {}})
    bundle["qrels"]["q1"] = {"d1": 1, "huge": 1}
    before = copy.deepcopy(bundle)

    cleaned, report = prepare_bundle(bundle)

    assert bundle == before
    assert "huge" not in {row["_id"] for row in cleaned["documents"]}
    assert "q1" not in {row["_id"] for row in cleaned["queries"]}
    assert "q1" not in cleaned["qrels"]
    reasons = {entry["reason"] for entry in report["exclusions"]}
    assert "document_too_large" in reasons
    assert "dependent_on_invalid_document" in reasons


@pytest.mark.parametrize("quote", ["absent", "Paris"])
def test_prepare_invalid_evidence_stays_in_review_with_original_labels(quote: str) -> None:
    bundle = make_bundle(input_kind="evidence_qa")
    bundle["documents"][0]["text"] = "Paris and Paris"
    bundle["evidence"] = [{"query_id": "q1", "doc_id": "d1", "quote": quote}]
    original = copy.deepcopy(bundle)
    cleaned, report = prepare_bundle(bundle)
    assert bundle == original
    assert cleaned["queries"] == bundle["queries"]
    assert cleaned["qrels"] == bundle["qrels"]
    assert cleaned["evidence"] == []
    assert cleaned["review_queue"][0]["evidence"] == bundle["evidence"][0]
    assert report["status"] == "review_required"
    assert report["review_required"][0]["query_id"] == "q1"
    # Re-auditing a published/reloaded bundle must retain the review state.
    assert audit_bundle(cleaned)["status"] == "review_required"


def test_prepare_uses_decimal_one_million_document_limit_and_keeps_boundary() -> None:
    def bundle_with_size(size: int) -> dict:
        bundle = make_bundle()
        bundle["documents"] = [{"_id": "boundary", "text": "x" * size, "metadata": {}}]
        bundle["queries"] = [{"_id": "q-boundary", "text": "which document?", "metadata": {}}]
        bundle["qrels"] = {"q-boundary": {"boundary": 1}}
        bundle["evidence"] = []
        return bundle

    retained, retained_report = prepare_bundle(bundle_with_size(1_000_000))
    assert [row["_id"] for row in retained["documents"]] == ["boundary"]
    assert [row["_id"] for row in retained["queries"]] == ["q-boundary"]
    assert retained_report["status"] == "pass"

    for size in (1_000_001, 1_048_576):
        cleaned, report = prepare_bundle(bundle_with_size(size))
        assert cleaned["documents"] == []
        assert cleaned["queries"] == []
        assert cleaned["qrels"] == {}
        assert "q-boundary" in report["preparation"]["excluded_query_ids"]
        assert any(entry["reason"] == "document_too_large" for entry in report["exclusions"])


def test_exact_equivalent_unlabelled_document_is_reviewed_or_excludes_query_without_new_gold() -> None:
    bundle = make_bundle()
    bundle["documents"].append({"_id": "d1-copy", "text": bundle["documents"][0]["text"], "metadata": {}})

    cleaned, report = prepare_bundle(bundle)
    assert "q1" not in cleaned["qrels"]
    assert any(entry["reason"] == "equivalent_unlabelled_document" for entry in report["exclusions"])
    assert "d1-copy" not in cleaned["qrels"].get("q1", {})

    reviewed, review_report = prepare_bundle(bundle, {"exclude_review_required": False})
    assert reviewed["qrels"]["q1"] == {"d1": 1}
    assert review_report["status"] == "review_required"
    assert "d1-copy" not in reviewed["qrels"]["q1"]


def test_duplicates_are_grouped_and_near_duplicates_are_candidates_without_default_deletion() -> None:
    bundle = make_bundle()
    duplicate = copy.deepcopy(bundle["documents"][0])
    duplicate["_id"] = "d1-copy"
    near = {
        "_id": "d-near",
        "text": "Paris is the capital of France and this sentence adds a small note.",
        "metadata": {},
    }
    bundle["documents"].extend([duplicate, near])

    report = audit_bundle(bundle, strict=False)
    cleaned, prep_report = prepare_bundle(bundle)

    assert report["full_duplicate_groups"][0]["document_ids"] == ["d1", "d1-copy"]
    assert report["near_duplicate"]["method"] == "token-shingles-minhash-candidates"
    assert {row["_id"] for row in cleaned["documents"]} == {"d1", "d2", "d1-copy", "d-near"}
    assert prep_report["preparation"]["near_duplicates_removed"] is False


def test_corpus_without_qrels_is_needs_queries_and_missing_language_is_unknown() -> None:
    bundle = make_bundle(input_kind="corpus")
    bundle["queries"] = []
    bundle.pop("qrels")
    bundle["evidence"] = []
    bundle["documents"][0].pop("language")

    report = audit_bundle(bundle)

    assert report["status"] == "needs_queries"
    assert report["needs_queries"] is True
    assert report["language_stats"]["documents"]["unknown"] == 1


def test_coverage_report_handles_empty_input_and_aggregate_overlap() -> None:
    assert coverage_report([])["totals"]["documents"] == 0
    first = make_bundle()
    second = make_bundle(input_kind="corpus")
    second["source"] = {"dataset": "other", "revision": "r2"}
    second["queries"] = []
    second.pop("qrels")
    second["evidence"] = []

    report = coverage_report([first, second])

    assert report["bundle_count"] == 2
    assert report["totals"]["documents"] == 4
    assert report["id_overlap"]["documents"]["d1"] == [0, 1]
    assert report["language_stats"]["documents"]["en"] == 2


def test_assign_splits_is_reproducible_and_keeps_family_components_together() -> None:
    bundle = make_bundle()
    bundle["documents"].append({"_id": "d3", "text": "isolated", "metadata": {"family_id": "f2"}})
    bundle["queries"].append({"_id": "q2", "text": "same family", "metadata": {"family_id": "f1"}})
    bundle["qrels"]["q2"] = {"d2": 1}

    first = assign_splits(bundle, seed=17)
    second = assign_splits(bundle, seed=17)

    assert first == second
    assert first["query_splits"]["q1"] == first["document_splits"]["d1"]
    assert first["query_splits"]["q1"] == first["query_splits"]["q2"]
    assert first["cross_split_components"] == []


def test_assign_splits_connects_exact_duplicate_bodies_and_translation_of_across_queries() -> None:
    bundle = make_bundle()
    bundle["documents"][0]["metadata"] = {}
    bundle["queries"][0]["metadata"] = {}
    bundle["documents"].append({"_id": "d-copy", "text": bundle["documents"][0]["text"], "metadata": {}})
    bundle["queries"].append({"_id": "q-copy", "text": "same body query", "metadata": {}})
    bundle["qrels"]["q-copy"] = {"d-copy": 1}
    bundle["documents"].append(
        {"_id": "d-translation", "text": "translated body", "metadata": {"translation_of": "d1"}}
    )
    bundle["queries"].append({"_id": "q-translation", "text": "translated query", "metadata": {}})
    bundle["qrels"]["q-translation"] = {"d-translation": 1}

    split = assign_splits(bundle, seed=23)

    assert split["query_splits"]["q1"] == split["query_splits"]["q-copy"]
    assert split["query_splits"]["q1"] == split["query_splits"]["q-translation"]
    assert split["document_splits"]["d1"] == split["document_splits"]["d-copy"]
    assert split["document_splits"]["d1"] == split["document_splits"]["d-translation"]
    assert split["cross_split_components"] == []


def test_explicit_normalization_is_traceable_and_write_report_supports_json(tmp_path: Path) -> None:
    bundle = make_bundle()
    bundle["documents"][0]["text"] = "  Paris\n is the capital of France.  "
    before = copy.deepcopy(bundle)
    unchanged, unchanged_report = prepare_bundle(bundle)
    normalized, report = prepare_bundle(bundle, {"normalize_whitespace": True})

    assert unchanged == before
    assert unchanged_report["normalization_changes"] == []
    assert normalized["documents"][0]["text"] == "Paris is the capital of France."
    assert report["normalization_changes"][0]["before_sha256"] != report["normalization_changes"][0]["after_sha256"]
    output = tmp_path / "quality.json"
    write_report(report, output)
    assert json.loads(output.read_text(encoding="utf-8"))["report_version"] == "dci-data-quality-v1"


def test_assign_splits_reports_large_shared_component() -> None:
    bundle = make_bundle()
    for index in range(12):
        query_id = f"family-q{index}"
        doc_id = f"family-d{index}"
        bundle["queries"].append({"_id": query_id, "text": query_id, "metadata": {"family_id": "huge"}})
        bundle["documents"].append({"_id": doc_id, "text": doc_id, "metadata": {"family_id": "huge"}})
        bundle["qrels"][query_id] = {doc_id: 1}

    split = assign_splits(bundle, seed=1)

    assert split["large_components"]
    assert split["warnings"][0]["code"] == "large_shared_component"
