import copy
from pathlib import Path

import pytest

from dci_bench.data.pipeline import batch, batch_resume, stages
from test_kalm_pool_batch import ready_source, fake_rank


def seal_parent(tmp_path, entry, parameters, root):
    bundle = tmp_path / 'parent_bundle'
    bundle.mkdir()
    (bundle / 'launch.sh').write_text('#!/bin/sh\nexit 0\n')
    plan = {'sources': [entry], 'parameters': parameters, 'run_root': str(root), 'code_files': {}}
    batch.write(bundle / 'plan.json', plan)
    batch.write(bundle / '_COMPLETE.json', {
        'schema_version': batch.SCHEMA, 'plan_sha256': batch.sha(bundle / 'plan.json'),
        'launcher_sha256': batch.sha(bundle / 'launch.sh')})
    batch.write(root / 'run.json', {'bundle': str(bundle), 'plan_sha256': batch.sha(bundle / 'plan.json')})
    return {**plan, 'reuse': {'bundle': str(bundle), 'plan_sha256': batch.sha(bundle / 'plan.json'),
                              'run_root': str(root)}}


@pytest.fixture
def tiny_parent(tmp_path):
    entry = {'source_id': 'fixture', 'task': 'fixture', 'bundle_hash': 'original'}
    parameters = {'trust_remote_code': True, 'query_prompt': batch.QUERY_PROMPT,
                  'capacities': [200, 500, 1000]}
    root = tmp_path / 'old_run'
    plan = seal_parent(tmp_path, entry, parameters, root)
    source = root / 'sources/fixture'
    batch.write(source / 'binding.json', batch_resume._binding(entry, plan))
    suite = source / 'attempts/export-ok/suite'
    marker = suite / 'metadata/fixture/_COMPLETE.json'
    batch.write(marker, {'complete': True})
    batch.write(source / 'phases/export.json', {'suite': str(suite), 'completion_sha256': batch.sha(marker)})
    return entry, plan, source, tmp_path / 'new_run/sources/fixture'


def test_partial_attempts_ignored_and_cache_copied_independently(tiny_parent):
    entry, plan, old, new = tiny_parent
    cache = old / 'embedding_cache/shards/config'
    cache.mkdir(parents=True)
    (cache / 'document-00000000.npz').write_bytes(b'closed shard; encoder validates format later')
    (cache / '.document-00000001.npz.tmp').write_bytes(b'partial')
    (old / 'pools/.1000.staging-partial').mkdir(parents=True)
    batch_resume.import_source_receipts(entry, plan, new)
    assert batch.read(new / 'phases/export.json') == batch.read(old / 'phases/export.json')
    target = new / 'embedding_cache/shards/config/document-00000000.npz'
    original = cache / target.name
    assert target.read_bytes() == original.read_bytes()
    assert target.stat().st_ino != original.stat().st_ino
    assert target.stat().st_nlink == original.stat().st_nlink == 1
    assert len(list((new / 'embedding_cache').rglob('*npz*'))) == 1
    assert batch_resume.pool_target(entry, plan, new, 1000) == new / 'pools/1000'
    before = target.stat().st_mtime_ns
    batch_resume.import_source_receipts(entry, plan, new)
    assert target.stat().st_mtime_ns == before


def test_interrupted_import_replays_pinned_receipts(tiny_parent, monkeypatch):
    entry, plan, old, new = tiny_parent
    original_write = batch.write
    def fail(path, value, **kwargs):
        if Path(path) == new / 'phases/export.json':
            raise RuntimeError('interrupted after reuse snapshot')
        return original_write(path, value, **kwargs)
    monkeypatch.setattr(batch, 'write', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        batch_resume.import_source_receipts(entry, plan, new)
    assert (new / 'reuse.json').exists()
    monkeypatch.setattr(batch, 'write', original_write)
    batch_resume.import_source_receipts(entry, plan, new)
    assert batch.read(new / 'phases/export.json') == batch.read(old / 'phases/export.json')


def test_reuse_rejects_changed_parameters_and_receipt(tiny_parent):
    entry, plan, old, new = tiny_parent
    changed = copy.deepcopy(plan)
    changed['parameters']['query_prompt'] = 'different'
    with pytest.raises(ValueError, match='different source content or model'):
        batch_resume.import_source_receipts(entry, changed, new)
    batch_resume.import_source_receipts(entry, plan, new)
    batch.write(new / 'phases/export.json', {'suite': 'wrong'}, replace=True)
    with pytest.raises(ValueError, match='receipt differs'):
        batch_resume.import_source_receipts(entry, plan, new)


def test_changed_completed_pool_is_rejected(tiny_parent):
    entry, plan, old, new = tiny_parent
    rank = old / 'attempts/rank-ok/rankings.json'
    batch.write(rank, {'saved': 'ranking'})
    batch.write(old / 'phases/rank.json', {'path': str(rank), 'sha256': batch.sha(rank)})
    marker = old / 'pools/200/metadata/fixture/_COMPLETE.json'
    batch.write(marker, {'complete': True})
    batch_resume.import_source_receipts(entry, plan, new)
    assert batch_resume.pool_target(entry, plan, new, 200) == old / 'pools/200'
    marker.write_text('{}')
    with pytest.raises(ValueError, match='completion changed'):
        batch_resume.pool_target(entry, plan, new, 200)


def test_reuse_does_not_accept_wrong_old_run_identity(tiny_parent):
    entry, plan, old, new = tiny_parent
    batch.write(Path(plan['reuse']['run_root']) / 'run.json', {'plan_sha256': 'foreign'}, replace=True)
    with pytest.raises(ValueError, match='run identity'):
        batch_resume.import_source_receipts(entry, plan, new)


def test_reuse_rejects_nested_run_and_modified_snapshot(tiny_parent):
    entry, plan, old, new = tiny_parent
    nested = Path(plan['reuse']['run_root']) / 'nested/sources/fixture'
    with pytest.raises(ValueError, match='independent new run'):
        batch_resume.import_source_receipts(entry, plan, nested)
    assert not nested.exists()
    batch_resume.import_source_receipts(entry, plan, new)
    snapshot = batch.read(new / 'reuse.json')
    snapshot['phases']['export']['suite'] = '/another/source'
    batch.write(new / 'reuse.json', snapshot, replace=True)
    with pytest.raises(ValueError, match='immutable origin'):
        batch_resume.import_source_receipts(entry, plan, new)


def test_reuse_rejects_linked_completion_marker(tiny_parent):
    entry, plan, old, new = tiny_parent
    marker = Path(batch.read(old / 'phases/export.json')['suite']) / 'metadata/fixture/_COMPLETE.json'
    target = marker.with_name('original.json')
    marker.rename(target)
    marker.symlink_to(target)
    with pytest.raises(ValueError, match='independent regular'):
        batch_resume.import_source_receipts(entry, plan, new)


def test_completed_legacy_results_reused_without_encoding_or_mutating_old(ready_source, monkeypatch, tmp_path):
    entry, old_plan, old_root = ready_source
    monkeypatch.setattr(stages, 'rank_source', fake_rank)
    result = batch.execute_source(entry, old_plan, old_root, device='cpu')
    # v1 published source completion but had no separate pools phase receipt.
    (old_root / 'sources' / entry['source_id'] / 'phases/pools.json').unlink()
    plan = seal_parent(tmp_path, entry, old_plan['parameters'], old_root)
    old_source = old_root / 'sources' / entry['source_id']
    paths = [old_source / 'status.json', old_source / '_COMPLETE.json',
             old_source / 'phases/export.json', old_source / 'phases/rank.json']
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    def no_rerank(*a, **kw):
        raise AssertionError('completed ranking must not run on GPU again')
    monkeypatch.setattr(stages, 'rank_source', no_rerank)
    new_root = tmp_path / 'new_run'
    resumed = batch.execute_source(entry, plan, new_root, device='cpu')
    assert resumed['ranking'] == result['ranking']
    assert resumed['pools'] == result['pools']
    assert not (new_root / 'sources/fixture/pools').exists()
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
