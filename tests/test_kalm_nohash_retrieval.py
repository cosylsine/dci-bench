import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from dci_bench.data.pipeline import retrieval, stages


TASK = "fixture-task"


def _write_suite(root: Path, *, text: str = "# Title\nabcde") -> Path:
    metadata = root / "metadata" / TASK
    corpus = root / "workspaces" / TASK / "corpus"
    metadata.mkdir(parents=True)
    corpus.mkdir(parents=True)
    document = corpus / "d000001.md"
    document.write_text(text, encoding="utf-8")
    (metadata / "manifest.json").write_text(
        json.dumps({"task": TASK, "dataset": "fixture", "revision": "r1"}),
        encoding="utf-8",
    )
    (metadata / "source_doc_ids.json").write_text(
        json.dumps({"d000001": "original-1"}), encoding="utf-8"
    )
    (metadata / "qrels.json").write_text(
        json.dumps({"q1": {"d000001": 1}}), encoding="utf-8"
    )
    (metadata / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "question"}) + "\n", encoding="utf-8"
    )
    (metadata / "corpus_files.jsonl").write_text(
        json.dumps(
            {
                "doc_id": "d000001",
                "source_doc_id": "original-1",
                "filename": "d000001.md",
                "bytes": len(text.encode("utf-8")),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_nohash_load_uses_saved_identity_without_sha_scan(tmp_path):
    suite = _write_suite(tmp_path / "suite")
    strict = retrieval.load_prepared_source(suite, TASK)
    with patch.object(retrieval, "_sha256_file", side_effect=AssertionError("unexpected corpus hash scan")):
        fast = retrieval.load_prepared_source(suite, TASK, verify_content_hashes=False)
    assert fast["documents"] == strict["documents"]
    assert fast["queries"] == strict["queries"]
    assert fast["provenance"]["source_identity"] == strict["provenance"]["source_identity"]
    assert fast["provenance"]["content_hashes_verified"] is False
    assert fast["provenance"]["content_hash_verification"] == "declared_saved_digest"


def test_nohash_load_rejects_missing_files_ids_and_digests(tmp_path):
    suite = _write_suite(tmp_path / "missing")
    (suite / "workspaces" / TASK / "corpus" / "d000001.md").unlink()
    with pytest.raises(FileNotFoundError, match="corpus file is missing"):
        retrieval.load_prepared_source(suite, TASK, verify_content_hashes=False)

    suite = _write_suite(tmp_path / "ids")
    (suite / "metadata" / TASK / "source_doc_ids.json").write_text(
        json.dumps({"d000001": "different-original"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="inventory ID mismatch"):
        retrieval.load_prepared_source(suite, TASK, verify_content_hashes=False)

    suite = _write_suite(tmp_path / "digest")
    inventory = suite / "metadata" / TASK / "corpus_files.jsonl"
    row = json.loads(inventory.read_text(encoding="utf-8"))
    row.pop("sha256")
    inventory.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing sha256"):
        retrieval.load_prepared_source(suite, TASK, verify_content_hashes=False)


def test_strict_detects_same_size_tamper_but_nohash_uses_declared_digest(tmp_path):
    suite = _write_suite(tmp_path / "tamper", text="# Title\nabcde")
    original = retrieval.load_prepared_source(suite, TASK)
    document = suite / "workspaces" / TASK / "corpus" / "d000001.md"
    document.write_text("# Title\nvwxyz", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        retrieval.load_prepared_source(suite, TASK)
    fast = retrieval.load_prepared_source(suite, TASK, verify_content_hashes=False)
    assert fast["documents"][0]["text"] == "# Title\nvwxyz"
    assert fast["provenance"]["source_identity"] == original["provenance"]["source_identity"]


def test_rank_source_forwards_nohash_and_pinned_weight_identity(tmp_path, monkeypatch):
    bundle = {
        "schema_version": "dci-data-bundle-v1",
        "source": {"dataset": "fixture", "revision": "r1"},
        "documents": [{"_id": "d1", "text": "document", "title": "", "language": "en", "metadata": {}}],
        "queries": [{"_id": "q1", "text": "query", "language": "en", "metadata": {}}],
        "qrels": {"q1": {"d1": 1}},
        "evidence": [],
        "judgments": [],
        "provenance": {},
        "input_kind": "retrieval",
    }
    load_calls = []
    config_calls = []

    def load_source(*args, **kwargs):
        load_calls.append((args, kwargs))
        return bundle

    class Encoder:
        def __init__(self, config):
            config_calls.append(config)
            self.config = config
            self.last_metadata = {"encoder": "test"}

        def _identities(self):
            return ("sha256:pinned", "tokenizer", "config")

        def encode_with_cache(self, texts, *, kind, cache_dir, **kwargs):
            self.last_metadata = {"kind": kind, "cache_dir": str(cache_dir)}
            return np.ones((len(texts), 2), dtype="float32")

    def dense(*args, **kwargs):
        return {"q1": {"ranked_doc_ids": ["d1"], "scores": {"d1": 1.0}}}

    def export(bundle, results, *, model):
        return {"schema_version": "dci-saved-retrieval-v1", "model": model.to_dict(), "results": results}

    monkeypatch.setattr(retrieval, "load_prepared_source", load_source)
    monkeypatch.setattr(retrieval, "SentenceTransformerEncoder", Encoder)
    monkeypatch.setattr(retrieval, "dense_search", dense)
    monkeypatch.setattr(retrieval, "export_saved_retrieval", export)
    monkeypatch.setattr(retrieval, "model_weight_identity", lambda *_args, **_kwargs: pytest.fail("weight hash"))
    output = tmp_path / "rankings.json"
    stages.rank_source(
        output,
        suite=tmp_path / "suite",
        task=TASK,
        cache_dir=tmp_path / "cache",
        verify_content_hashes=False,
        weight_identity="sha256:pinned",
    )
    assert load_calls[0][1] == {"verify_content_hashes": False}
    assert config_calls[0].weight_identity == "sha256:pinned"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model"]["weight_identity"] == "sha256:pinned"
    assert payload["verification_audit"]["content_hashes_verified"] is False
    assert payload["verification_audit"]["weight_identity_supplied"] is True
    assert payload["verification_audit"]["weight_identity_verified"] is False


def test_rank_source_propagates_release_verification_option(tmp_path, monkeypatch):
    bundle = {
        "schema_version": "dci-data-bundle-v1",
        "source": {"dataset": "fixture", "revision": "r1"},
        "documents": [{"_id": "d1", "text": "document", "title": "", "language": "en", "metadata": {}}],
        "queries": [{"_id": "q1", "text": "query", "language": "en", "metadata": {}}],
        "qrels": {"q1": {"d1": 1}},
        "evidence": [],
        "judgments": [],
        "provenance": {},
        "input_kind": "retrieval",
    }
    load_calls = []

    def load_release(*args, **kwargs):
        load_calls.append((args, kwargs))
        return bundle

    class Encoder:
        def __init__(self, config):
            self.config = config
            self.last_metadata = {"encoder": "test"}

        def _identities(self):
            return ("sha256:pinned", "tokenizer", "config")

        def encode_with_cache(self, texts, *, kind, cache_dir, **kwargs):
            self.last_metadata = {"kind": kind, "cache_dir": str(cache_dir)}
            return np.ones((len(texts), 2), dtype="float32")

    def dense(*args, **kwargs):
        return {"q1": {"ranked_doc_ids": ["d1"], "scores": {"d1": 1.0}}}

    monkeypatch.setattr(stages, "load_bundle", load_release)
    monkeypatch.setattr(retrieval, "SentenceTransformerEncoder", Encoder)
    monkeypatch.setattr(retrieval, "dense_search", dense)
    output = tmp_path / "rankings.json"
    stages.rank_source(
        output,
        release=tmp_path / "release",
        cache_dir=tmp_path / "cache",
        verify_content_hashes=False,
        weight_identity="sha256:pinned",
    )
    assert load_calls[0][0] == (tmp_path / "release",)
    assert load_calls[0][1] == {"verify_content_hashes": False}
