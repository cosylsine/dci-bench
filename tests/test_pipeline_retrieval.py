import hashlib
import json

import numpy as np
import pytest

from dci_bench.data.pipeline.retrieval import (
    DEFAULT_DOCUMENT_PROMPT,
    DEFAULT_QUERY_PROMPT,
    EmbeddingCache,
    EmbeddingConfig,
    EmbeddingShardCache,
    SentenceTransformerEncoder,
    bm25_search,
    build_source_identity_for_bundle,
    dense_search,
    export_saved_retrieval,
    load_prepared_source,
    model_weight_identity,
    unicode_tokens,
)


def _bundle():
    return {
        "schema_version": "dci-data-bundle-v1",
        "input_kind": "retrieval",
        "source": {"dataset": "tiny", "revision": "r1"},
        "documents": [
            {"_id": "d1", "title": "one", "text": "one", "language": "en", "metadata": {}},
            {"_id": "d2", "title": "two", "text": "two", "language": "en", "metadata": {}},
            {"_id": "d3", "title": "three", "text": "three", "language": "en", "metadata": {}},
        ],
        "queries": [{"_id": "q1", "text": "one", "language": "en", "metadata": {}}],
        "qrels": {"q1": {"d3": 1}},
        "evidence": [],
        "judgments": [],
        "provenance": {},
    }


def test_dense_search_preserves_ties_and_adds_gold_score_outside_topk():
    result = dense_search(
        {"q1": [1.0, 0.0]},
        {"d1": [1.0, 0.0], "d2": [1.0, 0.0], "d3": [0.0, 1.0]},
        top_k=1,
        gold_doc_ids={"q1": ["d3"]},
        chunk_size=1,
    )["q1"]
    assert result["ranked_doc_ids"] == ["d1"]
    assert set(result["scores"]) == {"d1", "d3"}
    assert result["scores"]["d3"] == 0.0


def test_dense_search_stable_topk_across_chunks_matches_reference_order():
    query = np.array([[1.0, 0.0]], dtype="float32")
    docs = np.array([
        [0.90, 0.0], [0.80, 0.0], [0.80, 0.0], [0.80, 0.0], [0.70, 0.0],
    ], dtype="float32")
    ids = ["d0", "d1", "d2", "d3", "d4"]
    result = dense_search(query, docs, ids, query_ids=["q"], top_k=3, chunk_size=2)["q"]
    assert result["ranked_doc_ids"] == ["d0", "d1", "d2"]
    expected_scores = docs @ query[0]
    reference = sorted(range(len(ids)), key=lambda index: (-float(expected_scores[index]), index))[:3]
    assert result["ranked_doc_ids"] == [ids[index] for index in reference]


def test_dense_search_gold_score_uses_position_map_without_document_id_index():
    class NoIndexIds(list):
        def index(self, _value):
            raise AssertionError("dense search must not linearly search document IDs")

    ids = NoIndexIds(["d0", "d1", "gold"])
    result = dense_search(
        np.array([[1.0, 0.0]], dtype="float32"),
        np.array([[1.0, 0.0], [0.9, 0.0], [0.1, 0.0]], dtype="float32"),
        ids,
        query_ids=["q"],
        top_k=1,
        gold_doc_ids={"q": ["gold"]},
        chunk_size=1,
    )["q"]
    assert result["ranked_doc_ids"] == ["d0"]
    assert result["scores"]["gold"] == pytest.approx(0.1)


def test_unicode_lexical_search_supports_cjk_bigrams():
    assert "東京" in unicode_tokens("東京駅")
    result = bm25_search("東京", {"a": "東京駅の案内", "b": "京都の案内"}, top_k=1)
    assert result["ranked_doc_ids"] == ["a"]


def test_export_calls_source_identity_and_keeps_real_gold_score():
    bundle = _bundle()
    identity = build_source_identity_for_bundle(bundle)
    assert set(identity) == {"queries_hash", "corpus_files_sha256", "source_doc_ids_hash"}
    saved = export_saved_retrieval(
        bundle,
        {"q1": {"ranked_doc_ids": ["d1"], "scores": {"d1": 0.9, "d3": 0.2}}},
        model=EmbeddingConfig(model_path="/local/kal", weight_identity="sha256:test-weight"),
        source_identity=identity,
    )
    assert saved["schema_version"] == "dci-saved-retrieval-v1"
    assert saved["rankings"]["q1"] == ["d1"]
    assert saved["scores"]["q1"]["d3"] == 0.2


def test_missing_local_model_never_downloads():
    encoder = SentenceTransformerEncoder(EmbeddingConfig(model_path="/definitely/missing/local-model"))
    with pytest.raises(FileNotFoundError, match="refusing download"):
        _ = encoder.model


def test_embedding_cache_identity_change_and_incomplete_entry(tmp_path):
    first = EmbeddingCache(tmp_path, EmbeddingConfig(model_path="/local/a"), weight_identity="w1")
    first.set("hello", np.array([1.0, 2.0], dtype="float32"), kind="query")
    assert np.allclose(first.get("hello", kind="query"), [1.0, 2.0])
    changed = EmbeddingCache(tmp_path, EmbeddingConfig(model_path="/local/a", normalize=False), weight_identity="w1")
    assert changed.get("hello", kind="query") is None
    metadata = next(tmp_path.glob("*.json"))
    metadata.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        first.get("hello", kind="query")


def test_all_cache_hits_keep_the_correct_query_audit(tmp_path):
    cfg = EmbeddingConfig(model_path="/local/test", weight_identity="test-weights")
    cache = EmbeddingCache(tmp_path, cfg)
    cache.set("cached query", np.array([1., 0.]), kind="query")
    encoder = SentenceTransformerEncoder(cfg)
    encoder.last_metadata = {"kind": "document", "truncation": {"count": 42}}
    result = encoder.encode_with_cache(["cached query"], kind="query", cache_dir=tmp_path)
    assert result.shape == (1, 2)
    assert encoder.last_metadata["kind"] == "query"
    assert encoder.last_metadata["truncation"]["count"] is None
    assert encoder.last_metadata["inference_executed"] is False


def test_local_kalm_config_reads_prompt_and_token_limit(tmp_path):
    (tmp_path / "config_sentence_transformers.json").write_text(json.dumps({
        "prompts": {"query": DEFAULT_QUERY_PROMPT, "document": DEFAULT_DOCUMENT_PROMPT},
    }))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"max_length": 7}))
    pooling = tmp_path / "1_Pooling"
    pooling.mkdir()
    (pooling / "config.json").write_text(json.dumps({"pooling_mode_mean_tokens": True}))
    config = EmbeddingConfig.from_local_model(tmp_path)
    assert config.query_prompt == DEFAULT_QUERY_PROMPT
    assert config.document_prompt == DEFAULT_DOCUMENT_PROMPT
    assert config.max_length == 7
    assert config.pooling == "mean"
    assert config.truncation_unit == "tokens"


def test_encoder_uses_token_limit_without_character_slice(tmp_path):
    class Tokenizer:
        def __call__(self, text, **_kwargs):
            return {"input_ids": list(range(len(text)))}

    class Model:
        tokenizer = Tokenizer()

        def __init__(self):
            self.max_seq_length = None
            self.seen = []

        def encode(self, texts, **_kwargs):
            self.seen.extend(texts)
            return np.ones((len(texts), 2), dtype="float32")

    model = Model()
    encoder = SentenceTransformerEncoder(
        EmbeddingConfig(
            model_path=tmp_path,
            query_prompt="Q:",
            document_prompt="",
            max_length=3,
            normalize=False,
        )
    )
    encoder._model = model
    encoder.encode(["abcdef"], kind="query")
    assert model.max_seq_length == 3
    assert model.seen == ["Q:abcdef"]
    assert encoder.last_metadata["truncation"] == {
        "max_length": 3,
        "unit": "tokens",
        "applied_by": "sentence_transformers_tokenizer",
        "count": 1,
        "character_max_length": None,
        "character_count": 0,
    }


def test_weight_identity_hashes_same_size_content(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"aaaa")
    first = model_weight_identity(tmp_path)
    weights.write_bytes(b"bbbb")
    second = model_weight_identity(tmp_path)
    assert first != second


def test_cache_identity_binds_tokenizer_files(tmp_path):
    tokenizer = tmp_path / "tokenizer_config.json"
    tokenizer.write_text(json.dumps({"max_length": 3}))
    first = EmbeddingCache(tmp_path / "cache", EmbeddingConfig(model_path=tmp_path), weight_identity="sha256:pinned")
    first.set("hello", np.array([1.0, 2.0], dtype="float32"), kind="query")
    tokenizer.write_text(json.dumps({"max_length": 4}))
    changed = EmbeddingCache(tmp_path / "cache", EmbeddingConfig(model_path=tmp_path), weight_identity="sha256:pinned")
    assert changed.get("hello", kind="query") is None


def test_sharded_cache_resumes_without_overwriting_completed_shards_and_rejects_corruption(tmp_path):
    class Model:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **_kwargs):
            self.calls.append(list(texts))
            return np.array([[float(len(text)), 1.0] for text in texts], dtype="float32")

    config = EmbeddingConfig(model_path=tmp_path / "model", weight_identity="pinned", normalize=False)
    encoder = SentenceTransformerEncoder(config)
    encoder._model = Model()
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    first = encoder.encode_with_cache(texts, kind="document", cache_dir=tmp_path / "cache", cache_shard_size=2)
    assert first.shape == (5, 2)
    assert len(encoder._model.calls) == 3
    second = encoder.encode_with_cache(texts, kind="document", cache_dir=tmp_path / "cache", cache_shard_size=2)
    assert np.array_equal(first, second)
    assert len(encoder._model.calls) == 3
    shard = next((tmp_path / "cache" / "shards").rglob("document-00000001.npz"))
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="embedding shard"):
        encoder.encode_with_cache(texts, kind="document", cache_dir=tmp_path / "cache", cache_shard_size=2)


def test_sharded_cache_prompt_change_selects_new_identity_namespace(tmp_path):
    class Model:
        def __init__(self):
            self.calls = 0

        def encode(self, texts, **_kwargs):
            self.calls += 1
            return np.ones((len(texts), 2), dtype="float32")

    cache_dir = tmp_path / "cache"
    first_config = EmbeddingConfig(model_path=tmp_path / "model", weight_identity="pinned", query_prompt="Q1:", normalize=False)
    first = SentenceTransformerEncoder(first_config)
    first._model = Model()
    first.encode_with_cache(["hello"], kind="query", cache_dir=cache_dir, cache_shard_size=2)
    changed_config = EmbeddingConfig(model_path=tmp_path / "model", weight_identity="pinned", query_prompt="Q2:", normalize=False)
    changed = SentenceTransformerEncoder(changed_config)
    changed._model = Model()
    changed.encode_with_cache(["hello"], kind="query", cache_dir=cache_dir, cache_shard_size=2)
    assert first._model.calls == changed._model.calls == 1
    assert len(list((cache_dir / "shards").iterdir())) == 2


def test_sharded_cache_resumes_after_encode_failure_without_losing_prior_shard(tmp_path):
    class Model:
        def __init__(self):
            self.calls = 0
            self.fail_once = True

        def encode(self, texts, **_kwargs):
            self.calls += 1
            if self.fail_once and self.calls == 2:
                self.fail_once = False
                raise RuntimeError("synthetic interruption")
            return np.array([[float(len(text)), 1.0] for text in texts], dtype="float32")

    config = EmbeddingConfig(model_path=tmp_path / "model", weight_identity="pinned", normalize=False)
    model = Model()
    encoder = SentenceTransformerEncoder(config)
    encoder._model = model
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        encoder.encode_with_cache(texts, kind="document", cache_dir=tmp_path / "cache", cache_shard_size=2)
    assert len(list((tmp_path / "cache" / "shards").rglob("document-*.npz"))) == 1
    result = encoder.encode_with_cache(texts, kind="document", cache_dir=tmp_path / "cache", cache_shard_size=2)
    assert result.shape == (5, 2)
    assert len(list((tmp_path / "cache" / "shards").rglob("document-*.npz"))) == 3


def test_sentence_transformer_default_prompt_is_not_applied_twice(tmp_path):
    class Model:
        default_prompt_name = "query"
        prompts = {"query": "MODEL:"}

        def __init__(self):
            self.seen = []
            self.prompt_args = []

        def encode(self, texts, **kwargs):
            self.prompt_args.append(kwargs.get("prompt"))
            prompt = kwargs.get("prompt")
            if prompt is None and self.default_prompt_name is not None:
                texts = [self.prompts[self.default_prompt_name] + text for text in texts]
            self.seen.extend(texts)
            return np.ones((len(texts), 2), dtype="float32")

    model = Model()
    encoder = SentenceTransformerEncoder(EmbeddingConfig(
        model_path=tmp_path, query_prompt="WRAPPER:", document_prompt="", normalize=False,
    ))
    encoder._model = model
    encoder.encode(["query"], kind="query")
    assert model.prompt_args == [""]
    assert model.seen == ["WRAPPER:query"]


def test_encoder_retries_cuda_oom_by_halving_batch_and_records_audit(tmp_path):
    class Model:
        def __init__(self):
            self.batch_sizes = []

        def encode(self, texts, *, batch_size, **_kwargs):
            self.batch_sizes.append(batch_size)
            if batch_size > 2:
                raise RuntimeError("CUDA out of memory")
            return np.ones((len(texts), 2), dtype="float32")

    model = Model()
    encoder = SentenceTransformerEncoder(EmbeddingConfig(
        model_path=tmp_path,
        device="cuda:0",
        batch_size=8,
        normalize=False,
    ))
    encoder._model = model
    result = encoder.encode(["one", "two", "three"], kind="query")
    assert result.shape == (3, 2)
    assert model.batch_sizes == [8, 4, 2]
    assert encoder.last_metadata["requested_batch_size"] == 8
    assert encoder.last_metadata["effective_batch_size"] == 2
    assert encoder.last_metadata["oom_retries"] == 2


def test_sharded_cache_hit_preserves_batch_oom_audit(tmp_path):
    class Model:
        def __init__(self):
            self.batch_sizes = []

        def encode(self, texts, *, batch_size, **_kwargs):
            self.batch_sizes.append(batch_size)
            if batch_size > 1:
                raise RuntimeError("CUDA out of memory")
            return np.ones((len(texts), 2), dtype="float32")

    config = EmbeddingConfig(
        model_path=tmp_path / "model",
        device="cuda:0",
        batch_size=4,
        normalize=False,
    )
    model = Model()
    encoder = SentenceTransformerEncoder(config)
    encoder._model = model
    texts = ["one", "two", "three"]
    encoder.encode_with_cache(texts, kind="document", cache_dir=tmp_path / "cache", cache_shard_size=2)
    assert model.batch_sizes == [4, 2, 1, 4, 2, 1]
    model.batch_sizes.clear()
    encoder.encode_with_cache(texts, kind="document", cache_dir=tmp_path / "cache", cache_shard_size=2)
    assert model.batch_sizes == []
    assert encoder.last_metadata["requested_batch_size"] == 4
    assert encoder.last_metadata["effective_batch_size"] == 1
    assert encoder.last_metadata["oom_retries"] == 4
    assert all(row["cache_hit"] for row in encoder.last_metadata["cache_shard_audit"])
    assert all(row["requested_batch_size"] == 4 for row in encoder.last_metadata["cache_shard_audit"])
    assert all(row["effective_batch_size"] == 1 for row in encoder.last_metadata["cache_shard_audit"])
    assert all(row["oom_retries"] == 2 for row in encoder.last_metadata["cache_shard_audit"])


def test_sharded_cache_collision_does_not_overwrite_original_vector(tmp_path):
    config = EmbeddingConfig(model_path=tmp_path / "model", weight_identity="pinned", normalize=False)
    cache = EmbeddingShardCache(tmp_path / "cache", config, 2)
    original = np.array([[1.0, 2.0]], dtype="float32")
    replacement = np.array([[9.0, 8.0]], dtype="float32")
    cache.set_shard(["text"], original, kind="document", shard_index=0)
    with pytest.raises(ValueError, match="different vectors"):
        cache.set_shard(["text"], replacement, kind="document", shard_index=0)
    assert np.array_equal(cache.get_shard(["text"], kind="document", shard_index=0), original)


def test_prepared_pool_jsonl_identity_and_task_are_preserved(tmp_path):
    task = "different-task-name"
    metadata = tmp_path / "metadata" / task
    corpus = tmp_path / "workspaces" / task / "corpus"
    metadata.mkdir(parents=True)
    corpus.mkdir(parents=True)
    text = "# Title\nRendered body"
    document = corpus / "d000001.md"
    document.write_text(text)
    sha256 = hashlib.sha256(text.encode()).hexdigest()
    (metadata / "manifest.json").write_text(json.dumps({"task": task, "dataset": "test", "revision": "r1"}))
    (metadata / "source_doc_ids.json").write_text(json.dumps({"d000001": "original-1"}))
    (metadata / "qrels.json").write_text(json.dumps({"q1": {"d000001": 1}}))
    query_text = "body\u2028with separator"
    (metadata / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": query_text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (metadata / "corpus_files.jsonl").write_text(json.dumps({
        "doc_id": "d000001", "source_doc_id": "original-1", "filename": "d000001.md",
        "bytes": len(text.encode()), "sha256": sha256,
    }) + "\n")
    bundle = load_prepared_source(tmp_path, task)
    assert bundle["queries"][0]["text"] == query_text
    saved = export_saved_retrieval(
        bundle,
        {"q1": {"ranked_doc_ids": ["original-1"], "scores": {"original-1": 0.5}}},
        model={"path": "/local/model", "weight_identity": "sha256:pinned", "trust_remote_code": True},
    )
    assert saved["task"] == task
    document.write_text("# Title\nChanged")
    with pytest.raises(ValueError, match="hash mismatch"):
        export_saved_retrieval(
            bundle,
            {"q1": {"ranked_doc_ids": ["original-1"], "scores": {"original-1": 0.5}}},
            model={"path": "/local/model", "weight_identity": "sha256:pinned", "trust_remote_code": True},
        )
