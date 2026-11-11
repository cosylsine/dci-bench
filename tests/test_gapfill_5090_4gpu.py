from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_gapfill_5090_4gpu as launcher


def _bundle_loader(_suite: Path, _source: str, *, verify_content_hashes: bool = False):
    return {"documents": [{"_id": f"d{i}", "text": f"document {i}"} for i in range(2050)]}


def _entry(tmp_path: Path) -> dict:
    source = "gapfill_browsecomp_plus_en"
    state = tmp_path / "state"
    return {
        "source": source,
        "suite": str(tmp_path / "suite"),
        "cache_dir": str(tmp_path / "canonical"),
        "canonical_cache_dir": str(tmp_path / "canonical"),
        "worker_cache_dir": str(state / "workers" / "worker-00" / source / "embedding-cache"),
        "state_root": str(state),
        "model_path": str(tmp_path / "model"),
        "weight_identity": "fixture-weight-v1",
    }


def test_four_worker_partition_is_complete_disjoint_and_stable():
    sources = [
        {"source": "gapfill_browsecomp_plus_en", "status": "pending", "missing_shards": [0, 2, 3, 6, 7, 10]},
        {"source": "gapfill_cedar_zh_v2", "status": "pending", "missing_shards": [1, 4, 8, 9]},
        {"source": "already-complete", "status": "complete", "missing_shards": [0, 1]},
    ]
    first = launcher.partition_missing_shards(sources)
    second = launcher.partition_missing_shards(sources)
    assert first == second
    flattened = [(item["source"], item["shard"]) for rows in first.values() for item in rows]
    expected = [(sources[0]["source"], i) for i in sources[0]["missing_shards"]]
    expected += [(sources[1]["source"], i) for i in sources[1]["missing_shards"]]
    assert sorted(flattened) == sorted(expected)
    assert len(flattened) == len(set(flattened))
    assert all(first[index] for index in range(4))


def test_worker_cache_reuse_and_sequential_merge_preserve_identity(tmp_path):
    entry = _entry(tmp_path)
    Path(entry["model_path"]).mkdir()
    calls = []

    class FakeEncoder:
        def __init__(self, config):
            assert config.device == "cuda:0"
            assert config.batch_size == 8
            assert config.normalize is True
            self.last_metadata = {"truncation": {"count": 0, "character_count": 0}}

        def encode(self, texts, *, kind):
            calls.append((kind, len(texts)))
            return np.asarray([[float(len(text)), float(index + 1)] for index, text in enumerate(texts)], dtype="float32")

    first = launcher.encode_document_shards(
        entry, [0, 1, 2], encoder_factory=FakeEncoder, bundle_loader=_bundle_loader,
        runtime={"runtime": "first"},
    )
    assert first["encoded_shards"] == [0, 1, 2]
    call_count = len(calls)
    second = launcher.encode_document_shards(
        entry, [0, 1, 2], encoder_factory=FakeEncoder, bundle_loader=_bundle_loader,
        runtime={"runtime": "second"},
    )
    assert second["reused_shards"] == [0, 1, 2]
    assert second["encoded_shards"] == []
    assert len(calls) == call_count

    worker_cache = launcher._cache_for({**entry, "cache_dir": entry["worker_cache_dir"]})
    bundle = _bundle_loader(Path(entry["suite"]), entry["source"])
    texts = [row["text"] for row in bundle["documents"]]
    saved, metadata = worker_cache.get_shard(texts[:1024], kind="document", shard_index=0, with_metadata=True)
    assert metadata["audit"]["truncation"] == {"count": 0, "character_count": 0}
    canonical_cache = launcher._cache_for(entry)
    canonical_cache.set_shard(texts[:1024], saved, kind="document", shard_index=0, audit=metadata["audit"])
    merged = launcher.merge_document_shards(
        entry, {0: [0, 1, 2], 1: [], 2: [], 3: []},
        runtime={"gpu": "5090"}, bundle_loader=_bundle_loader,
    )
    assert merged["reused_shards"] == [0]
    assert merged["merged_shards"] == [1, 2]
    assert launcher._cache_for(entry).get_shard(texts[:1024], kind="document", shard_index=0) is not None
    assert launcher._cache_for(entry).get_shard(texts[1024:2048], kind="document", shard_index=1) is not None
    assert launcher._cache_for(entry).get_shard(texts[2048:], kind="document", shard_index=2) is not None
    _, merged_metadata = launcher._cache_for(entry).get_shard(
        texts[1024:2048], kind="document", shard_index=1, with_metadata=True)
    assert merged_metadata["audit"]["truncation"] == {"count": 0, "character_count": 0}


def test_default_prepared_reader_uses_run_root_and_source_layout(tmp_path):
    source = "gapfill_browsecomp_plus_en"
    run_root = tmp_path / "run"
    metadata = run_root / "source" / "metadata" / source
    corpus = run_root / "source" / "workspaces" / source / "corpus"
    metadata.mkdir(parents=True)
    corpus.mkdir(parents=True)
    mapping = {"d000001": "original-1", "d000002": "original-2"}
    (metadata / "source_doc_ids.json").write_text(json.dumps(mapping), encoding="utf-8")
    rows = []
    for alias, text in (("d000001", "# first\nbody"), ("d000002", "# second\nbody")):
        path = corpus / f"{alias}.md"
        path.write_text(text, encoding="utf-8")
        rows.append({"doc_id": alias, "filename": path.name, "bytes": path.stat().st_size,
                     "sha256": "fixture"})
    (metadata / "corpus_files.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (metadata / "queries.jsonl").write_text('{"_id":"q1","text":"query"}\n', encoding="utf-8")
    reader = launcher._prepared_shard_reader(run_root, source)
    assert reader.document_count == 2
    assert reader.query_count == 1
    assert reader.texts_for_shard(0) == ["# first\nbody", "# second\nbody"]

    model = tmp_path / "model"
    model.mkdir()
    state = tmp_path / "state"
    entry = {
        "source": source,
        "suite": str(run_root / "source"),
        "cache_dir": str(tmp_path / "canonical"),
        "canonical_cache_dir": str(tmp_path / "canonical"),
        "worker_cache_dir": str(state / "workers" / "worker-00" / source / "embedding-cache"),
        "state_root": str(state),
        "model_path": str(model),
        "weight_identity": "fixture-weight-v1",
    }

    class FakeEncoder:
        def __init__(self, config):
            assert config.device == "cuda:0"

        def encode(self, texts, *, kind):
            assert kind == "document"
            return np.asarray([[float(index + 1), 1.0] for index, _ in enumerate(texts)], dtype="float32")

    encoded = launcher.encode_document_shards(
        entry, [0], encoder_factory=FakeEncoder, runtime={"test": True}
    )
    assert encoded["encoded_shards"] == [0]
    merged = launcher.merge_document_shards(
        entry, {0: [0], 1: [], 2: [], 3: []}, runtime={"test": True}
    )
    assert merged["merged_shards"] == [0]


def test_cached_vector_runtime_compatibility_gate_is_tight():
    expected = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    result = launcher.compare_embedding_vectors(expected, expected.copy())
    assert result["passed"] is True
    with pytest.raises(RuntimeError, match="compatibility smoke failed"):
        launcher.compare_embedding_vectors(expected, np.asarray([[0.99, 0.01], [0.0, 1.0]], dtype="float32"))


def test_completed_source_receipt_is_validated_and_skipped(tmp_path):
    spec = launcher.SOURCE_SPECS["gapfill_cedar_zh_v2"]
    root = tmp_path / spec.name
    root.mkdir(parents=True)
    ranking = root / spec.ranking_name
    ranking.write_text("ranking", encoding="utf-8")
    pools = []
    for capacity, name in zip(launcher.POOL_CAPACITIES, spec.pool_names):
        pool = root / name
        marker = pool / "metadata" / spec.name / "_COMPLETE.json"
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps({"capacity": capacity}), encoding="utf-8")
        pools.append({"capacity": capacity, "path": str(pool),
                      "completion_sha256": launcher.sha256_file(marker)})
    receipt = {"source": spec.name, "task": spec.name, "ranking": str(ranking),
               "ranking_sha256": launcher.sha256_file(ranking), "pools": pools}
    (root / spec.receipt_name).write_text(json.dumps(receipt), encoding="utf-8")
    checked = []

    def validator(path, *, task, verify_content_hashes):
        checked.append((Path(path), task, verify_content_hashes))

    result = launcher.completed_source_status(spec, tmp_path, pool_validator=validator)
    assert result["status"] == "complete"
    assert len(checked) == 2
    assert all(value[2] is False for value in checked)


def test_check_only_does_not_enter_gpu_or_inference_path(monkeypatch, capsys):
    plan = {
        "mode": "check-only-plan",
        "state_root": "/tmp/state",
        "workers": [{"worker_index": i, "physical_gpu": str(i),
                      "environment": {"CUDA_VISIBLE_DEVICES": str(i)}, "tasks": []}
                     for i in range(4)],
        "sources": [],
    }
    monkeypatch.setattr(launcher, "build_plan", lambda **kwargs: plan)
    monkeypatch.setattr(launcher, "_gpu_preflight", lambda *_: pytest.fail("GPU preflight in check-only"))
    monkeypatch.setattr(launcher, "run_full", lambda *_args, **_kwargs: pytest.fail("full run in check-only"))
    assert launcher.main(["--check-only"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["mode"] == "check-only-plan"


def test_shell_uses_required_python_and_fixed_launcher_command():
    shell = (Path(__file__).parents[1] / "scripts" / "launch_gapfill_5090_4gpu.sh").read_text()
    assert launcher.REQUIRED_PYTHON in shell
    assert re.search(r'exec "\$EMBED_PY" .*run_gapfill_5090_4gpu\.py', shell)
    assert launcher._worker_command(Path("/tmp/plan.json"), 2)[:3] == [
        launcher.REQUIRED_PYTHON, str(Path(launcher.__file__).resolve()), "--worker"
    ]


def _saved_merge_plan(tmp_path):
    entry = {**_entry(tmp_path), "status": "pending", "documents": 2050, "queries": 1}
    Path(entry["model_path"]).mkdir()
    plan = {
        "schema_version": launcher.SCHEMA, "mode": "full",
        "run_root": str(tmp_path), "state_root": entry["state_root"], "sources": [entry],
        "workers": [{"worker_index": i, "tasks": [
            {"source": entry["source"], "shard": index} for index in range(3)
        ] if i == 0 else []} for i in range(4)],
    }
    plan["plan_sha256"] = launcher.stable_digest(plan)
    path = Path(entry["state_root"]) / "attempts" / "original" / "plan.json"
    launcher.write_json_new(path, plan)
    for worker in plan["workers"]:
        launcher.write_json_new(path.parent / f"worker-{worker['worker_index']:02d}.json", worker)
    return path, entry


def test_merge_only_recovers_interruption_without_gpu_or_reencoding(tmp_path, monkeypatch):
    path, entry = _saved_merge_plan(tmp_path)
    texts = [row["text"] for row in _bundle_loader(None, None)["documents"]]
    worker = launcher._cache_for({**entry, "cache_dir": entry["worker_cache_dir"]})
    canonical = launcher._cache_for(entry)
    for index in range(3):
        chunk = texts[index * 1024:(index + 1) * 1024]
        values = np.ones((len(chunk), 2), dtype="float32")
        worker.set_shard(chunk, values, kind="document", shard_index=index,
                         audit={"runtime": {"original_worker": 0}, "truncation": {"count": 7}})
        if index == 0:
            canonical.set_shard(chunk, values, kind="document", shard_index=index)
    original = {p: p.read_bytes() for p in path.parent.iterdir()}
    first_shard = canonical._path("document", 0).read_bytes()
    monkeypatch.setattr(launcher, "_validate_prepared_metadata", lambda *_: {"num_docs": 2050, "num_queries": 1})
    from types import SimpleNamespace
    monkeypatch.setattr(launcher, "_prepared_shard_reader", lambda *_: SimpleNamespace(
        document_count=len(texts), texts_for_shard=lambda index: texts[index * 1024:(index + 1) * 1024]))
    monkeypatch.setattr(launcher, "_gpu_preflight", lambda *_: pytest.fail("GPU in merge-only"))
    monkeypatch.setattr(launcher, "_encoder_for", lambda *_: pytest.fail("encoding in merge-only"))
    monkeypatch.setattr(launcher, "_finalize_source", lambda *_: pytest.fail("ranking in merge-only"))
    result = launcher.recover_completed_workers(path)
    assert result["status"] == "document_cache_complete"
    assert result["inference_executed"] is False
    assert result["sources"][0]["merged_shards"] == [1, 2]
    assert result["sources"][0]["reused_shards"] == [0]
    assert all(p.read_bytes() == value for p, value in original.items())
    assert canonical._path("document", 0).read_bytes() == first_shard
    _, metadata = canonical.get_shard(texts[1024:2048], kind="document", shard_index=1, with_metadata=True)
    assert metadata["audit"]["runtime"] == {"original_worker": 0}
    assert metadata["audit"]["truncation"] == {"count": 7}
    assert (Path(result["attempt_root"]) / "_MERGE_COMPLETE.json").is_file()
    assert not (Path(result["attempt_root"]) / "_COMPLETE.json").exists()
    again = launcher.recover_completed_workers(path)
    assert again["sources"][0]["merged_shards"] == []
    assert again["sources"][0]["reused_shards"] == [0, 1, 2]


@pytest.mark.parametrize("failure", ["changed-plan", "missing-receipt", "wrong-assignment"])
def test_merge_only_rejects_invalid_attempt_before_writing(tmp_path, failure):
    path, entry = _saved_merge_plan(tmp_path)
    if failure == "changed-plan":
        plan = launcher.read_json(path)
        plan["sources"][0]["documents"] = 42
        path.write_text(json.dumps(plan))
    elif failure == "missing-receipt":
        (path.parent / "worker-01.json").unlink()
    else:
        (path.parent / "worker-01.json").write_text(json.dumps({"worker_index": 1, "tasks": ["wrong"]}))
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    with pytest.raises((ValueError, FileNotFoundError)):
        launcher.recover_completed_workers(path)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert not Path(entry["canonical_cache_dir"]).exists()


def test_shell_preserves_controller_error_and_exit_status_without_logging_dry_run(tmp_path):
    fake = tmp_path / "fake-python"
    fake.write_text(f"#!{sys.executable}\nimport os, sys\nprint('controller output')\n"
                    "print('controller failure detail', file=sys.stderr)\n"
                    "sys.exit(int(os.environ['FAKE_EXIT_CODE']))\n")
    fake.chmod(0o755)
    shell = Path(launcher.__file__).with_name("launch_gapfill_5090_4gpu.sh").read_text()
    shell = shell.replace(launcher.REQUIRED_PYTHON, str(fake))
    script = tmp_path / "launch.sh"
    script.write_text(shell)
    logs = tmp_path / "controller-logs"
    env = dict(os.environ, DCI_GAPFILL_LOG_DIR=str(logs), DRY_RUN="0", FAKE_EXIT_CODE="7")
    dry = subprocess.run(["bash", str(script), "--check-only"], env=env, capture_output=True, text=True)
    assert dry.returncode == 7
    assert not logs.exists()
    failed = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert failed.returncode == 7
    log = next(logs.glob("*.log")).read_text()
    assert "controller output" in log and "controller failure detail" in log
    assert "exit_code=7" in log
    passed = subprocess.run(["bash", str(script)], env={**env, "FAKE_EXIT_CODE": "0"}, capture_output=True, text=True)
    assert passed.returncode == 0
    assert len(list(logs.glob("*.log"))) == 2
