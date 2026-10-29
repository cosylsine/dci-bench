import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from dci_bench.data.pipeline.connectors import fetch_source, load_source
from dci_bench.data.pipeline.core import digest, export_dci, load_bundle, publish_bundle, write_json
from dci_bench.data.pipeline.stages import mine_review, partition_languages, rank_source, run_model_stage
from test_data_pipeline import bundle


def test_squad_preserves_unicode_offsets_ids_and_rejects_bad_span(tmp_path):
    source = tmp_path / "qa.json"
    raw = {"data": [{"title": "서울", "paragraphs": [{"context": "서울은 한국의 수도다.",
             "qas": [{"id": "korean-123", "question": "한국의 수도는?",
                      "answers": [{"answer_start": 0, "text": "서울"}]}]}]}]}
    source.write_text(json.dumps(raw, ensure_ascii=False))
    spec = {"id": "tydi", "connector": "squad", "input_kind": "evidence_qa", "revision": "v1",
            "options": {"files": [{"path": str(source), "name": "data/qa.json"}],
                        "language_from_id": "tydiqa", "translation_status": "native"}}
    snapshot = fetch_source(spec, tmp_path / "raw")
    b = load_source(spec, snapshot)
    assert b["queries"][0]["_id"] == "korean-123"
    assert b["documents"][0]["language"] == "ko"
    assert b["evidence"][0]["end"] == 2
    publish_bundle(b, tmp_path / "clean", specification=spec, raw_snapshot=snapshot)
    raw["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"] = 1
    source.write_text(json.dumps(raw))
    spec["revision"] = "v2"
    snapshot = fetch_source(spec, tmp_path / "raw")
    with pytest.raises(ValueError, match="offsets"):
        load_source(spec, snapshot)


def test_mining_unions_views_and_retains_seed_gold_without_false_negatives(tmp_path):
    b = bundle()
    b["queries"][0]["text"] = "Venus natural moons"
    release = tmp_path / "release"
    publish_bundle(b, release)
    view = {"bundle_hash": digest(load_bundle(release)), "model": "independent-test-encoder",
            "results": {"q/1": {"ranked_doc_ids": ["source/a"]}}}
    write_json(tmp_path / "dense.json", view)
    mine_review(release, tmp_path / "review.json", top_k=1, dense_files=[tmp_path / "dense.json"])
    result = json.loads((tmp_path / "review.json").read_text())
    assert result["candidate_doc_ids"]["q/1"] == ["source/a", "source/b"]
    assert result["unjudged_is_negative"] is False
    assert result["review"][0]["unjudged_candidates"] == ["source/b"]
    view["bundle_hash"] = "stale"
    write_json(tmp_path / "stale.json", view)
    with pytest.raises(ValueError, match="different bundle"):
        mine_review(release, tmp_path / "rejected.json", dense_files=[tmp_path / "stale.json"])
    assert not (tmp_path / "rejected.json").exists()


def test_partition_preserves_all_documents_and_rejects_cross_language_gold(tmp_path):
    b = bundle()
    other = copy.deepcopy(b["documents"][1])
    other.update(_id="zhdoc", language="zh", text="金星没有天然卫星。")
    b["documents"].append(other)
    b["queries"].append({"_id": "zhq", "text": "金星有卫星吗？", "language": "zh", "metadata": {}})
    b["qrels"]["zhq"] = {"zhdoc": 1}
    publish_bundle(b, tmp_path / "release")
    result = partition_languages(tmp_path / "release", tmp_path / "languages")
    assert result["en"]["num_documents"] == 2
    assert result["zh"]["num_documents"] == 1
    assert load_bundle(tmp_path / "languages/zh")["qrels"] == {"zhq": {"zhdoc": 1}}
    b["qrels"]["q/1"]["zhdoc"] = 1
    publish_bundle(b, tmp_path / "cross")
    with pytest.raises(ValueError, match="cross-language"):
        partition_languages(tmp_path / "cross", tmp_path / "cross-languages")
    assert not (tmp_path / "cross-languages").exists()


def test_generation_requires_explicit_scope_before_network(tmp_path):
    publish_bundle(bundle("corpus"), tmp_path / "release")
    config = Path(__file__).resolve().parents[1] / "configs/data_pipeline/models.json"
    with pytest.raises(ValueError, match="explicit document IDs"):
        run_model_stage(tmp_path / "release", tmp_path / "output", role="generator",
                        models_config=config, cache_dir=tmp_path / "cache", stage_config={})


@pytest.mark.parametrize("mapping", [{}, {"q/1": []}, {"q/1": ["source/b"]}])
def test_judge_rejects_incomplete_plan_before_network(tmp_path, mapping):
    publish_bundle(bundle(), tmp_path / "release")
    write_json(tmp_path / "plan.json", {"bundle_hash": digest(load_bundle(tmp_path / "release")),
                                        "candidate_doc_ids": mapping})
    config = Path(__file__).resolve().parents[1] / "configs/data_pipeline/models.json"
    with pytest.raises(ValueError, match="Candidate|candidates"):
        run_model_stage(tmp_path / "release", tmp_path / "output", role="judge",
                        models_config=config, cache_dir=tmp_path / "cache", stage_config={},
                        candidates=tmp_path / "plan.json")
    assert not (tmp_path / "output").exists()


def test_partition_rejects_cross_language_judgments(tmp_path):
    b = bundle()
    b["documents"][1]["language"] = "fr"
    b["judgments"] = [{"query_id": "q/1", "doc_id": "source/b", "score": 0, "source": "source_human"}]
    publish_bundle(b, tmp_path / "release")
    with pytest.raises(ValueError, match="cross-language evidence/judgment"):
        partition_languages(tmp_path / "release", tmp_path / "languages")
    assert not (tmp_path / "languages").exists()


def test_ranked_export_consumed_by_real_candidate_pool_builder(tmp_path):
    from dci_bench.data.document_pools import prepare_document_pools
    b = bundle()
    b["documents"].extend({"_id": f"d-{i:04}", "title": f"title {i}", "text": f"unique {i}",
                            "language": "en", "metadata": {}} for i in range(1030))
    publish_bundle(b, tmp_path / "release")
    export_dci(tmp_path / "release", tmp_path / "suite", "different-task-name")
    class Encoder:
        def __init__(self, config): self.last_metadata = {"test_double": True}
        def encode_with_cache(self, texts, *, kind, cache_dir):
            if kind == "query": return np.array([[1., 0.]], dtype="float32")
            return np.array([[0. if "Phobos" in text else float(i + 1) / len(texts), 1.]
                             for i, text in enumerate(texts)], dtype="float32")
    with patch("dci_bench.data.pipeline.retrieval.SentenceTransformerEncoder", Encoder):
        rank_source(tmp_path / "rankings.json", suite=tmp_path / "suite", task="different-task-name",
                    cache_dir=tmp_path / "cache", top_k=1000)
    saved = json.loads((tmp_path / "rankings.json").read_text())
    assert saved["task"] == "different-task-name"
    for capacity in (200, 500, 1000):
        result = prepare_document_pools(tmp_path / "suite", tmp_path / "pools" / str(capacity), "different-task-name",
                                         capacity=capacity, rankings_path=tmp_path / "rankings.json")
        pool = json.loads(result.candidate_pools_path.read_text())
        assert pool["pools"]["q/1"]["num_docs"] == capacity
