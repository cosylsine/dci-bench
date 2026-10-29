import json
from pathlib import Path

import pytest

from dci_bench.data.pipeline import batch


def _scheduler_plan(root, entries):
    return {
        'run_root': str(root),
        'sources': entries,
        'parameters': {'capacities': [200, 500, 1000]},
    }


def _write_rank_receipt(root, entry):
    ranked = root / 'sources' / entry['source_id'] / 'old-rank.json'
    batch.write(ranked, {'source': entry['source_id']})
    batch.write(root / 'sources' / entry['source_id'] / 'phases' / 'rank.json',
               {'path': str(ranked), 'sha256': batch.sha(ranked)})


def _write_pool_completion(root, entry, parameters):
    source_root = root / 'sources' / entry['source_id']
    rank = batch.read(source_root / 'phases' / 'rank.json')
    pools = []
    for capacity in parameters['capacities']:
        target = source_root / 'pools' / str(capacity)
        marker = target / 'metadata' / entry['task'] / '_COMPLETE.json'
        batch.write(marker, {'capacity': capacity})
        pools.append({'capacity': capacity, 'path': str(target),
                      'completion_sha256': batch.sha(marker)})
    batch.write(source_root / '_COMPLETE.json', {
        'schema_version': batch.SCHEMA,
        'entry_hash': batch.digest(entry),
        'parameters_hash': batch.digest(parameters),
        'ranking': rank,
        'pools': pools,
    })


def test_phase_queues_keep_cpu_pools_running_and_refill_gpu_ranks(tmp_path, monkeypatch):
    root, bundle = tmp_path / 'run', tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'plan.json').write_text('{}')
    resumed = {'source_id': 'resumed', 'task': 'resumed', 'documents': 2000, 'queries': 1}
    fresh = [
        {'source_id': f'fresh{i}', 'task': f'fresh{i}', 'documents': 1000 + i, 'queries': 1}
        for i in range(5)
    ]
    entries = [resumed, *fresh]
    plan = _scheduler_plan(root, entries)
    _write_rank_receipt(root, resumed)
    monkeypatch.setattr(batch, 'load_plan', lambda _: plan)
    monkeypatch.setattr(batch, 'preflight', lambda *args, **kwargs: {'test': True})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,5,6,7')
    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.delenv('SLURM_PROCID', raising=False)
    monkeypatch.setattr(batch.time, 'sleep', lambda _: None)

    launched = []
    processes = []
    rank_started_with_pool = []

    class Process:
        def __init__(self, command, *, env, **kwargs):
            self.pid = 500 + len(processes)
            self.command = command
            self.phase = command[command.index('--phase') + 1]
            self.source = command[command.index('--source') + 1]
            self.env = env
            self.poll_count = 0
            self.done = False
            processes.append(self)
            launched.append(self)
            if self.phase == 'rank':
                rank_started_with_pool.append(any(pool.phase == 'pools' and not pool.done
                                                  for pool in processes))
            if self.phase == 'export':
                entry = next(item for item in entries if item['source_id'] == self.source)
                marker = (root / 'sources' / self.source / 'attempts' / 'export-fixture' /
                          'suite' / 'metadata' / entry['task'] / '_COMPLETE.json')
                batch.write(marker, {'source': self.source})
                batch.write(root / 'sources' / self.source / 'phases' / 'export.json',
                            {'suite': str(marker.parents[2]), 'completion_sha256': batch.sha(marker)})
            elif self.phase == 'rank':
                ranked = root / 'sources' / self.source / 'rankings.json'
                batch.write(ranked, {'source': self.source})
                batch.write(root / 'sources' / self.source / 'phases' / 'rank.json',
                            {'path': str(ranked), 'sha256': batch.sha(ranked)})

        def poll(self):
            self.poll_count += 1
            if self.phase == 'pools' and self.poll_count < 10:
                return None
            if self.phase == 'rank' and self.poll_count < 2:
                return None
            if self.done:
                return 0
            self.done = True
            if self.phase == 'pools':
                _write_pool_completion(
                    root,
                    next(item for item in entries if item['source_id'] == self.source),
                    plan['parameters'])
            return 0

    monkeypatch.setattr(batch.subprocess, 'Popen', Process)
    result = batch.run(bundle)

    assert result['source_views'] == len(entries)
    rank_workers = [p for p in launched if p.phase == 'rank']
    assert len(rank_workers) == len(fresh)
    assert {p.env['CUDA_VISIBLE_DEVICES'] for p in rank_workers} == {'3', '5', '6', '7'}
    assert all(p.env['CUDA_VISIBLE_DEVICES'] == '' for p in launched if p.phase != 'rank')
    assert all(p.env['OMP_NUM_THREADS'] == '4' for p in launched if p.phase != 'rank')
    assert all(p.env['OPENBLAS_NUM_THREADS'] == '4' for p in launched if p.phase != 'rank')
    assert not any(p.source == 'resumed' and p.phase == 'rank' for p in launched)
    assert any(rank_started_with_pool)
    assert (root / '_COMPLETE.json').is_file()


def test_failed_phase_does_not_publish_batch_completion_or_stop_other_sources(tmp_path, monkeypatch):
    root, bundle = tmp_path / 'run', tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'plan.json').write_text('{}')
    entries = [
        {'source_id': 'bad', 'task': 'bad', 'documents': 1001, 'queries': 1},
        {'source_id': 'good', 'task': 'good', 'documents': 1000, 'queries': 1},
    ]
    plan = _scheduler_plan(root, entries)
    monkeypatch.setattr(batch, 'load_plan', lambda _: plan)
    monkeypatch.setattr(batch, 'preflight', lambda *args, **kwargs: {'test': True})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.delenv('SLURM_PROCID', raising=False)
    monkeypatch.setattr(batch.time, 'sleep', lambda _: None)
    processes = []

    class Process:
        def __init__(self, command, *, env, **kwargs):
            self.pid = 700 + len(processes)
            self.phase = command[command.index('--phase') + 1]
            self.source = command[command.index('--source') + 1]
            processes.append(self)

        def poll(self):
            if self.source == 'bad':
                return 2
            if self.phase == 'export':
                entry = next(item for item in entries if item['source_id'] == self.source)
                marker = (root / 'sources' / self.source / 'suite' / 'metadata' /
                          entry['task'] / '_COMPLETE.json')
                if not marker.exists():
                    batch.write(marker, {'source': self.source})
                    batch.write(root / 'sources' / self.source / 'phases' / 'export.json',
                                {'suite': str(marker.parents[2]), 'completion_sha256': batch.sha(marker)})
                return 0
            if self.phase == 'rank':
                ranked = root / 'sources' / self.source / 'rankings.json'
                if not ranked.exists():
                    batch.write(ranked, {'source': self.source})
                    batch.write(root / 'sources' / self.source / 'phases' / 'rank.json',
                                {'path': str(ranked), 'sha256': batch.sha(ranked)})
                return 0
            marker = root / 'sources' / self.source / '_COMPLETE.json'
            if not marker.exists():
                _write_pool_completion(
                    root,
                    next(item for item in entries if item['source_id'] == self.source),
                    plan['parameters'])
            return 0

    monkeypatch.setattr(batch.subprocess, 'Popen', Process)
    with pytest.raises(RuntimeError, match='1 source'):
        batch.run(bundle)
    assert (root / 'sources' / 'good' / '_COMPLETE.json').is_file()
    assert not (root / '_COMPLETE.json').exists()
    assert any(p.source == 'bad' for p in processes)


@pytest.mark.parametrize('kind', ['same', 'ancestor', 'child'])
def test_run_rejects_reuse_root_overlap_before_mkdir(tmp_path, monkeypatch, kind):
    old_root = tmp_path / 'old-run'
    old_root.mkdir()
    if kind == 'same':
        new_root = old_root
    elif kind == 'ancestor':
        new_root = tmp_path
    else:
        new_root = old_root / 'new-run'
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    plan = {
        'run_root': str(new_root),
        'reuse': {'run_root': str(old_root)},
        'sources': [],
        'parameters': {'capacities': [200, 500, 1000]},
    }
    monkeypatch.setattr(batch, 'load_plan', lambda _: plan)
    with pytest.raises(ValueError, match='overlaps'):
        batch.run(bundle)
    if kind == 'child':
        assert not new_root.exists()
    else:
        assert old_root.exists()


def _resume_prepare_fixture(tmp_path):
    old_bundle = tmp_path / 'old-bundle'
    new_bundle = tmp_path / 'new-bundle'
    old_bundle.mkdir()
    old_root = tmp_path / 'old-run'
    matrix = tmp_path / 'matrix.json'
    matrix.write_text('{}')
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config_sentence_transformers.json').write_text(json.dumps({
        'prompts': {'query': batch.QUERY_PROMPT, 'document': ''},
    }))
    parameters = {
        'model_path': str(model), 'weight_identity': 'old-weights',
        'trust_remote_code': True, 'query_prompt': batch.QUERY_PROMPT,
        'document_prompt': '', 'max_length': 8192, 'batch_size': 2,
        'cache_shard_size': 128, 'search_query_batch_size': 2,
        'top_k': 1000, 'capacities': [200, 500, 1000], 'seed': 20261021,
    }
    sources = [{'source_id': 'exact-source', 'task': 'exact-source',
                'release': str(tmp_path / 'release'), 'documents': 1, 'queries': 1}]
    old_plan = {
        'schema_version': batch.SCHEMA, 'matrix': str(matrix),
        'matrix_sha256': batch.sha(matrix), 'sources': sources,
        'parameters': parameters, 'summary': {'source_views': 1},
        'scifact': {'rerun_required': False}, 'run_root': str(old_root),
    }
    (old_bundle / 'plan.json').write_text(json.dumps(old_plan))
    return old_bundle, new_bundle, old_root, old_plan, parameters, sources, matrix


def test_prepare_resume_copies_exact_sources_parameters_and_rejects_nested_root(tmp_path, monkeypatch):
    old_bundle, new_bundle, old_root, old_plan, parameters, sources, matrix = _resume_prepare_fixture(tmp_path)
    monkeypatch.setattr(batch, 'load_plan', lambda _: old_plan)
    monkeypatch.setattr(batch, 'preflight', lambda *args, **kwargs: {'resume': True})
    monkeypatch.setattr(batch, 'snapshot_code', lambda _: {})

    batch.prepare(new_bundle, matrix=matrix, run_root=tmp_path / 'new-run',
                  resume_bundle=old_bundle)
    copied = batch.read(new_bundle / 'plan.json')
    assert copied['sources'] == sources
    assert copied['parameters'] == parameters
    assert copied['reuse'] == {
        'bundle': str(old_bundle.absolute()),
        'plan_sha256': batch.sha(old_bundle / 'plan.json'),
        'run_root': old_plan['run_root'],
    }
    assert Path(copied['run_root']) == (tmp_path / 'new-run').absolute()

    nested_bundle = tmp_path / 'nested-bundle'
    with pytest.raises(ValueError, match='disjoint run root'):
        batch.prepare(nested_bundle, matrix=matrix, run_root=old_root / 'nested',
                      resume_bundle=old_bundle)
    assert not nested_bundle.exists()


def test_reuse_without_resume_module_fails_closed(tmp_path, monkeypatch):
    entry = {'source_id': 's', 'task': 's'}
    plan = {'reuse': {'bundle': 'old', 'run_root': 'old'}}
    def missing_resume_module(*, required=False):
        if required:
            raise RuntimeError('This reuse plan requires batch_resume')
        return None
    monkeypatch.setattr(batch, '_resume_module', missing_resume_module)
    with pytest.raises(RuntimeError, match='requires .*batch_resume'):
        batch._import_source_receipts(entry, plan, tmp_path / 'source')
    with pytest.raises(RuntimeError, match='requires .*batch_resume'):
        batch._pool_target(entry, plan, tmp_path / 'source', 200)


def test_valid_source_complete_is_requeued_for_cpu_pool_validation(tmp_path):
    root = tmp_path / 'run'
    entry = {'source_id': 'complete', 'task': 'complete', 'documents': 10, 'queries': 1}
    plan = _scheduler_plan(root, [entry])
    _write_rank_receipt(root, entry)
    _write_pool_completion(root, entry, plan['parameters'])
    assert batch._source_complete_valid(entry, plan, root)
    assert batch._source_next_phase(entry, plan, root) == 'pools'
