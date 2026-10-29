import json

import pytest

from dci_bench.data.pipeline.models import (
    LocalChatClient,
    LocalModelLengthError,
    LocalModelResponseError,
    generate_queries,
    judge_queries,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class _Opener:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def open(self, _request, timeout):
        del timeout
        self.calls += 1
        return _Response(self.payloads.pop(0))


def _completion(content, *, finish_reason="stop", usage=None, model="m"):
    result = {
        "model": model,
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
    }
    if usage is not None:
        result["usage"] = usage
    return result


def _bundle():
    return {
        "schema_version": "dci-data-bundle-v1",
        "input_kind": "retrieval",
        "source": {"dataset": "tiny", "revision": "r1"},
        "documents": [
            {"_id": "d1", "title": "One", "text": "Paris is the capital of France.", "language": "en", "metadata": {}},
            {"_id": "d2", "title": "Two", "text": "Tokyo is the capital of Japan.", "language": "en", "metadata": {}},
        ],
        "queries": [{"_id": "q0", "text": "seed", "language": "en", "metadata": {}}],
        "qrels": {},
        "evidence": [],
        "judgments": [],
        "provenance": {},
    }


def test_client_retries_invalid_json_and_uses_atomic_cache(tmp_path):
    opener = _Opener([
        {"unexpected": True},
        _completion('{"ok": true}', usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}),
    ])
    client = LocalChatClient(model="m", cache_dir=tmp_path, retries=1)
    client._opener = opener
    schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    messages = [{"role": "system", "content": "fixed"}, {"role": "user", "content": "data"}]
    assert client.complete_json(messages, schema) == {"ok": True}
    assert opener.calls == 2
    assert client.complete_json(messages, schema) == {"ok": True}
    assert opener.calls == 2
    assert client.audit_log[-1]["from_cache"] is True


def test_client_rejects_length_and_missing_usage():
    length_client = LocalChatClient(model="m", retries=0)
    length_client._opener = _Opener([_completion('{"ok": true}', finish_reason="length", usage={"total_tokens": 1})])
    with pytest.raises(LocalModelLengthError):
        length_client.complete_json([], {"type": "object"})

    missing_client = LocalChatClient(model="m", retries=0)
    missing_client._opener = _Opener([_completion('{"ok": true}', usage=None)])
    with pytest.raises(LocalModelResponseError, match="usage"):
        missing_client.complete_json([], {"type": "object"})


@pytest.mark.parametrize(
    "content",
    [
        '{"ok": true, "ok": false}',
        '{"outer": {"value": 1, "value": 2}}',
    ],
)
def test_client_rejects_duplicate_json_keys_including_nested(content):
    client = LocalChatClient(model="m", retries=0)
    client._opener = _Opener([_completion(content, usage={"total_tokens": 1})])
    with pytest.raises(LocalModelResponseError, match="duplicate object key"):
        client.complete_json([], {"type": "object"})


class _FakeClient:
    model = "generator"
    revision = "local"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.audit_log = []

    def complete_json(self, *_args, **_kwargs):
        self.audit_log.append({"cache_key": str(len(self.audit_log)), "usage": {"total_tokens": 1}})
        return self.outputs.pop(0)


def test_generate_queries_validates_quote_and_marks_generated_qrels():
    client = _FakeClient([{
        "queries": [{
            "text": "What is the capital of France?",
            "language": "en",
            "constraints": [],
            "version": "1",
            "multidoc": False,
            "evidence": [{"doc_id": "d1", "quote": "Paris is the capital of France."}],
        }]
    }])
    result = generate_queries(_bundle(), client, {"selected_doc_ids": ["d1"]})
    generated = result["queries"][-1]
    assert result["qrels"][generated["_id"]] == {"d1": 1}
    assert result["provenance"]["qrels_label"] == "llm_generated"
    assert result["evidence"][-1]["start"] == 0
    generation = result["provenance"]["query_generation"]
    assert generation["selected_doc_ids"] == ["d1"]
    assert generation["generated_query_ids"] == [generated["_id"]]
    assert generation["release_readiness"]["human_review_complete"] is False
    assert generation["release_readiness"]["gold_status"] == "not_human_gold"

    bad = _FakeClient([{
        "queries": [{
            "text": "bad",
            "language": "en",
            "constraints": [],
            "version": "1",
            "multidoc": False,
            "evidence": [{"doc_id": "d1", "quote": "invented"}],
        }]
    }])
    with pytest.raises(ValueError, match="exact document substring"):
        generate_queries(_bundle(), bad, {"selected_doc_ids": ["d1"]})


def test_judge_unknown_and_omitted_candidates_enter_review_queue():
    client = _FakeClient([{"judgments": [{"doc_id": "d1", "relevance": "maybe", "quotes": []}]}])
    result = judge_queries(
        _bundle(),
        client,
        {"candidate_doc_ids": {"q0": ["d1", "d2"]}},
    )
    by_id = {row["doc_id"]: row for row in result["judgments"]}
    assert by_id["d1"]["score"] is None
    assert by_id["d2"]["status"] == "review_required"
    assert {row["reason"] for row in result["review_queue"]} == {"unknown_label", "candidate_omitted"}
    assert "generator and judge use the same model" not in result["provenance"].get("warnings", [])
    judge = result["provenance"]["judge"]
    assert judge["candidate_count"] == 2
    assert judge["judgment_count"] == 2
    assert judge["review_queue_count"] == 2
    assert judge["release_readiness"]["requires_human_review"] is True


def test_judge_disagreement_is_not_collapsed_to_a_label():
    client = _FakeClient([
        {"judgments": [{"doc_id": "d1", "relevance": 1, "quotes": ["Paris is the capital of France."]}]},
        {"judgments": [{"doc_id": "d1", "relevance": 2, "quotes": ["Paris is the capital of France."]}]},
    ])
    result = judge_queries(
        _bundle(), client, {"candidate_doc_ids": {"q0": ["d1"]}, "votes": 2}
    )
    assert result["judgments"][-1]["score"] is None
    assert result["review_queue"][-1]["reason"] == "disagreement"


def test_judge_preserves_existing_qrels_and_records_positive_conflict():
    source = _bundle()
    source["qrels"] = {"q0": {"d1": 2}}
    source["judgments"] = [{"query_id": "q0", "doc_id": "d1", "label": "human", "score": 2}]
    client = _FakeClient([{
        "judgments": [
            {"doc_id": "d1", "relevance": 0, "quotes": []},
            {"doc_id": "d2", "relevance": 2, "quotes": ["Tokyo is the capital of Japan."]},
        ]
    }])
    result = judge_queries(
        source,
        client,
        {"candidate_doc_ids": {"q0": ["d1", "d2"]}, "stage_id": "judge-stage-2"},
    )
    assert result["qrels"]["q0"] == {"d1": 2, "d2": 2}
    assert len(result["judgments"]) == 3
    assert result["judgments"][-1]["stage_id"] == "judge-stage-2"
    assert any(row["reason"] == "human_positive_conflict" for row in result["review_queue"])
    assert result["provenance"]["judge"]["existing_qrels_preserved"] is True


def test_generation_stops_calls_once_new_query_budget_is_reached():
    client = _FakeClient([{"queries": [{"text": "What is the capital of France?", "language": "en",
        "constraints": [], "version": "1", "multidoc": False,
        "evidence": [{"doc_id": "d1", "quote": "Paris is the capital of France."}]}]}])
    result = generate_queries(_bundle(), client, {"selected_doc_ids": ["d1", "d2"], "max_queries": 1})
    assert len(result["provenance"]["query_generation"]["generated_query_ids"]) == 1
