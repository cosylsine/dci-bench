from __future__ import annotations

import copy
import json

from dci_bench.data.pipeline.collection import assign_collection_splits, collection_coverage


def bundle(*, source: str, document_id: str = "d", query_id: str = "q", document_text: str = "shared") -> dict:
    return {
        "schema_version": "dci-data-bundle-v1",
        "input_kind": "retrieval",
        "source": {"dataset": source, "revision": "r1"},
        "documents": [{"_id": document_id, "title": "D", "text": document_text, "language": "en", "metadata": {}}],
        "queries": [{"_id": query_id, "text": f"question {source}", "language": "en", "metadata": {}}],
        "qrels": {query_id: {document_id: 1}},
        "evidence": [{"query_id": query_id, "doc_id": document_id, "quote": document_text}],
        "judgments": [],
        "provenance": {},
    }


def split_for(result: dict, kind: str, source_index: int, original_id: str) -> str:
    return result[f"{kind}_splits"][str(source_index)][original_id]


def test_shared_family_across_languages_is_one_global_component() -> None:
    english = bundle(source="xquad-en", document_id="d", query_id="q", document_text="English body")
    french = bundle(source="xquad-fr", document_id="d", query_id="q", document_text="French body")
    english["queries"][0]["metadata"] = {"family_id": "xquad-question-7"}
    french["queries"][0]["metadata"] = {"family_id": "xquad-question-7"}
    english["documents"][0]["metadata"] = {"family_id": "xquad-doc-7"}
    french["documents"][0]["metadata"] = {"family_id": "xquad-doc-7"}

    result = assign_collection_splits([english, french], seed=17)

    assert split_for(result, "query", 0, "q") == split_for(result, "query", 1, "q")
    assert split_for(result, "document", 0, "d") == split_for(result, "document", 1, "d")
    assert result["component_stats"]["cross_source_count"] >= 1
    assert any(component["cross_source"] for component in result["components"])


def test_exact_duplicate_document_body_connects_two_query_sources() -> None:
    first = bundle(source="wix-expert", document_id="article", query_id="q1", document_text="same KB article")
    second = bundle(source="wix-simulated", document_id="article", query_id="q2", document_text="same KB article")

    result = assign_collection_splits([first, second], seed=29)

    assert split_for(result, "document", 0, "article") == split_for(result, "document", 1, "article")
    assert split_for(result, "query", 0, "q1") == split_for(result, "query", 1, "q2")
    assert any(component["cross_source"] for component in result["components"])


def test_equal_raw_ids_with_different_content_stay_separate_without_explicit_link() -> None:
    first = bundle(source="one", document_id="same", query_id="same-q", document_text="body one")
    second = bundle(source="two", document_id="same", query_id="same-q", document_text="body two")

    result = assign_collection_splits([first, second], seed=31)

    matching_components = [
        component
        for component in result["components"]
        if any(ref["source_index"] == 0 and ref["original_id"] == "same" for ref in component["documents"])
        or any(ref["source_index"] == 1 and ref["original_id"] == "same" for ref in component["documents"])
    ]
    assert len(matching_components) == 2
    assert {ref["source_index"] for component in matching_components for ref in component["documents"]} == {0, 1}
    assert not any(component["cross_source"] for component in matching_components)

    first["documents"][0]["metadata"] = {"family_id": "explicit-shared"}
    second["documents"][0]["metadata"] = {"family_id": "explicit-shared"}
    linked = assign_collection_splits([first, second], seed=31)
    assert split_for(linked, "document", 0, "same") == split_for(linked, "document", 1, "same")


def test_translation_of_and_group_id_connect_across_sources_without_mutation() -> None:
    first = bundle(source="base", document_id="base-doc", query_id="base-q", document_text="base")
    second = bundle(source="translated", document_id="translated-doc", query_id="translated-q", document_text="translated")
    second["documents"][0]["metadata"] = {"translation_of": "base-doc", "group_id": "shared-group"}
    second["queries"][0]["metadata"] = {"translation_of": "base-q", "group_id": "shared-group"}
    before = copy.deepcopy([first, second])

    result = assign_collection_splits([first, second], seed=43)

    assert split_for(result, "document", 0, "base-doc") == split_for(result, "document", 1, "translated-doc")
    assert split_for(result, "query", 0, "base-q") == split_for(result, "query", 1, "translated-q")
    assert [first, second] == before
    json.dumps(result, ensure_ascii=False, allow_nan=False)


def test_collection_coverage_counts_explicit_translations() -> None:
    native = bundle(source="native")
    translated = bundle(source="translated")
    unknown = bundle(source="unknown")
    native["queries"][0]["metadata"] = {"translation_status": "native"}
    # XQuAD's released bundles use this explicit status without necessarily
    # carrying a translation_of target in the query metadata.
    translated["queries"][0]["metadata"] = {"translation_status": "translated"}

    report = collection_coverage([native, translated, unknown])

    assert report["query_counts"] == {"total": 3, "native": 1, "translated": 1, "unknown": 1}
    assert report["native_query_count"] == 1
    assert report["translated_query_count"] == 1
    assert report["unknown_query_count"] == 1
    assert report["source_query_counts"][1]["translated_queries"] == 1
    assert report["source_query_counts"][2]["unknown_queries"] == 1
