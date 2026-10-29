import copy
from pathlib import Path
import sys
import types

import pytest

from dci_bench.data.pipeline import batch, batch_resume, stages
from test_kalm_pool_batch import ready_source, fake_rank
from test_kalm_batch_resume import tiny_parent, seal_parent
from test_kalm_phase_scheduler import _resume_prepare_fixture


def test_nohash_rank_receipt_avoids_reading_large_file(tmp_path, monkeypatch):
    path = tmp_path / 'rankings.json'
    path.write_text('{"results": {}}')
    plan = {'validation': {'verify_content_hashes': False}}
    receipt = {'path': str(path), 'bytes': path.stat().st_size,
               'sha256': None, 'content_hashes_verified': False}
    monkeypatch.setattr(batch, 'sha', lambda p: pytest.fail('nohash rank must not reread bytes'))
    assert batch._ranking_receipt_valid(receipt, plan)
    assert batch._ranking_receipt_valid({'path': str(path), 'sha256': 'declared-old-sha'}, plan)
    path.write_text('truncated')
    assert not batch._ranking_receipt_valid(receipt, plan)
    assert not batch._ranking_receipt_valid({'path': str(path)}, plan)


def test_prepare_sets_128_and_nohash_without_changing_semantic_parameters(tmp_path, monkeypatch):
    old, new, old_root, old_plan, parameters, sources, matrix = _resume_prepare_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(batch, 'load_plan', lambda _: old_plan)
    monkeypatch.setattr(batch, 'preflight', lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(batch, 'snapshot_code', lambda _: {})
    batch.prepare(new, matrix, run_root=tmp_path / 'new-run', resume_bundle=old,
                  batch_size=128, verify_content_hashes=False)
    plan = batch.read(new / 'plan.json')
    assert plan['sources'] == sources
    assert plan['parameters']['batch_size'] == 128
    assert batch.compatible_parameters(parameters, plan['parameters'])
    assert not batch.content_hashes_enabled(plan)
    assert calls[0]['verify_content_hashes'] is False
    changed = copy.deepcopy(plan['parameters'])
    changed['max_length'] = 1024
    assert not batch.compatible_parameters(parameters, changed)


def test_nohash_preflight_does_not_hash_model_or_source_content(ready_source, monkeypatch, tmp_path):
    from dci_bench.data.pipeline import retrieval
    entry, plan, root = ready_source
    plan = {**plan, 'sources': [entry], 'run_root': str(root),
            'validation': {'verify_content_hashes': False}}
    monkeypatch.setattr(batch, 'load_plan', lambda _: plan)
    monkeypatch.setattr(batch.importlib.metadata, 'version', lambda _: '4.0.0')
    monkeypatch.setitem(sys.modules, 'sentence_transformers', types.SimpleNamespace(SentenceTransformer=object))
    monkeypatch.setattr(retrieval, 'model_weight_identity', lambda *a: pytest.fail('weights must not be hashed'))
    original_sha = batch.sha
    monkeypatch.setattr(batch, 'sha', lambda p: pytest.fail('source content must not be rehashed')
                        if Path(p).name in {'manifest.json', 'qrels.json', 'queries.jsonl'} else original_sha(p))
    receipt = batch.preflight(tmp_path / 'bundle')
    assert receipt['content_hashes_verified'] is False
    assert 'not recomputed' in receipt['boundary']


def test_reuse_partial_cache_without_content_hashing(tiny_parent, monkeypatch):
    entry, plan, old, new = tiny_parent
    plan['validation'] = {'verify_content_hashes': False}
    cache = old / 'embedding_cache/shards/config'
    cache.mkdir(parents=True)
    source = cache / 'document-00000000.npz'
    source.write_bytes(b'closed-shard')
    original_sha = batch.sha
    def no_cache_hash(path):
        if '.npz' in Path(path).name:
            pytest.fail('cache copies must use size checks without hashing')
        return original_sha(path)
    monkeypatch.setattr(batch, 'sha', no_cache_hash)
    batch_resume.import_source_receipts(entry, plan, new)
    target = new / 'embedding_cache/shards/config' / source.name
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_nlink == 1
    assert target.stat().st_ino != source.stat().st_ino
    snapshot = batch.read(new / 'reuse.json')
    assert snapshot['cache_files']['shards/config/' + source.name]['sha256'] is None
    target.write_bytes(b'truncated')
    with pytest.raises(ValueError, match='cache shard differs'):
        batch_resume.import_source_receipts(entry, plan, new)


def test_batch128_reuses_completed_rank_and_transitively_reused_pools(ready_source, monkeypatch, tmp_path):
    entry, original_plan, original_root = ready_source
    monkeypatch.setattr(stages, 'rank_source', fake_rank)
    first = batch.execute_source(entry, original_plan, original_root, device='cpu')
    first_parent = tmp_path / 'first-parent'
    first_parent.mkdir()
    second_plan = seal_parent(first_parent, entry, original_plan['parameters'], original_root)
    second_root = tmp_path / 'second-run'
    def never_rank(*args, **kwargs):
        pytest.fail('completed ranking must be reused across batch-size change')
    monkeypatch.setattr(stages, 'rank_source', never_rank)
    second = batch.execute_source(entry, second_plan, second_root, device='cpu')
    assert second['pools'] == first['pools']
    assert not (second_root / 'sources/fixture/pools').exists()
    second_parent = tmp_path / 'second-parent'
    second_parent.mkdir()
    third_plan = seal_parent(second_parent, entry, second_plan['parameters'], second_root)
    third_plan['parameters'] = {**third_plan['parameters'], 'batch_size': 128}
    third_plan['validation'] = {'verify_content_hashes': False}
    original_sha = batch.sha
    monkeypatch.setattr(batch, 'sha', lambda p: pytest.fail('reused rank must not be hashed')
                        if Path(p).name == 'rankings.json' else original_sha(p))
    third_root = tmp_path / 'third-run'
    third = batch.execute_source(entry, third_plan, third_root, device='cpu')
    assert third['ranking'] == first['ranking']
    assert third['pools'] == first['pools']
    assert third['content_hashes_verified'] is False
    assert batch._source_complete_valid(entry, third_plan, third_root)
    assert batch.execute_source(entry, third_plan, third_root, device='cpu') == third


def test_cpu_resources_are_explicitly_configurable(monkeypatch, tmp_path):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.delenv(key, raising=False)
    env = batch._worker_env(tmp_path, {'source_id': 'fixture'}, 'export', None,
                            cpu_threads=12, io_workers=32)
    assert env['CUDA_VISIBLE_DEVICES'] == ''
    assert env['OMP_NUM_THREADS'] == '12'
    assert env['DCI_KALM_IO_WORKERS'] == '32'


def test_source_completion_does_not_mislabel_validation_policy(tmp_path):
    entry = {'source_id': 'fixture', 'documents': 1, 'queries': 1}
    plan = {'parameters': {}}
    binding = {'entry_hash': batch.digest(entry), 'parameters_hash': batch.digest({})}
    batch._complete_source(entry, plan, tmp_path, binding, {}, [])
    nohash = {**plan, 'validation': {'verify_content_hashes': False}}
    with pytest.raises(ValueError, match='validation policy differs'):
        batch._complete_source(entry, nohash, tmp_path, binding, {}, [])


def test_batch_change_skips_incompatible_partial_embedding_cache(tiny_parent):
    entry, plan, old, new = tiny_parent
    plan['parameters'] = {**plan['parameters'], 'batch_size': 128}
    cache = old / 'embedding_cache/shards/config'
    cache.mkdir(parents=True)
    (cache / 'document-00000000.npz').write_bytes(b'old-batch-cache')
    batch_resume.import_source_receipts(entry, plan, new)
    snapshot = batch.read(new / 'reuse.json')
    assert snapshot['cache_files'] == {}
    assert snapshot['cache_reuse_skipped'].startswith('batch_size_changed')
    assert not (new / 'embedding_cache').exists()
