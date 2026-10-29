import copy
import json
from pathlib import Path

import numpy as np
import pytest

from dci_bench.data.pipeline import batch
from dci_bench.data.pipeline.core import publish_bundle
from dci_bench.data.pipeline.retrieval import (
    EmbeddingConfig, dense_search, export_saved_retrieval, load_prepared_source,
)


def test_selection_is_exact_and_does_not_admit_legacy_review_or_scifact():
    good = [{'source_id': f'source{i}', 'origin': 'pipeline', 'inventory_role': 'three_tier_source',
             'release_state': 'labels_ready'} for i in range(53)]
    sci = {**good[0], 'source_id': 'beir_scifact_test'}
    rows = good + [sci, {**good[0], 'source_id': 'old', 'origin': 'legacy_frozen_only'},
                   {**good[0], 'source_id': 'review', 'release_state': 'review_required'},
                   {**good[0], 'source_id': 'small', 'inventory_role': 'small_corpus'}]
    assert len(batch.select_sources({'sources': rows}, scifact={'rerun_required': False})) == 53
    assert len(batch.select_sources({'sources': rows}, scifact={'rerun_required': True})) == 54
    with pytest.raises(ValueError, match='exactly'):
        batch.select_sources({'sources': rows[:-1] + [good[1]]}, scifact={'rerun_required': False})


def test_scifact_decision_uses_executed_prompt_audits(tmp_path):
    config = {'trust_remote_code': True, 'query_prompt': batch.QUERY_PROMPT, 'document_prompt': ''}
    saved = {'model': config, 'encoding_audit': {'queries': config, 'documents': config}}
    path = tmp_path / 'rankings.json'
    path.write_text(json.dumps(saved))
    assert batch.audit_scifact(path)['rerun_required'] is False
    saved = copy.deepcopy(saved)
    saved['encoding_audit']['queries'] = {**config, 'query_prompt': ''}
    path.write_text(json.dumps(saved))
    assert batch.audit_scifact(path)['rerun_required'] is True


def test_gpu_masks_preserve_scheduler_assignment():
    assert batch.cuda_slots(4, '3,5,6,7') == ['3', '5', '6', '7']
    assert batch.cuda_slots(2, 'GPU-aa,GPU-bb') == ['GPU-aa', 'GPU-bb']
    with pytest.raises(ValueError):
        batch.cuda_slots(4, '0,1')
    with pytest.raises(ValueError):
        batch.cuda_slots(2, '0,0')


def test_lock_prevents_duplicate_job_and_receipts_are_not_overwritten(tmp_path):
    path = tmp_path / '.lock'
    with batch.lock(path):
        with pytest.raises(RuntimeError, match='Another process'):
            with batch.lock(path):
                pass
    receipt = tmp_path / 'receipt.json'
    batch.write(receipt, {'a': 1})
    with pytest.raises(FileExistsError):
        batch.write(receipt, {'a': 2})
    assert batch.read(receipt) == {'a': 1}


def test_child_exit_race_does_not_skip_remaining_cleanup(monkeypatch):
    from unittest.mock import Mock
    first, second = Mock(pid=101), Mock(pid=102)
    first.poll.return_value = second.poll.return_value = None
    logs = [Mock(), Mock()]
    killed = []
    def kill(pid, sig):
        killed.append(pid)
        if pid == 101:
            raise ProcessLookupError('child just exited')
    monkeypatch.setattr(batch.os, 'killpg', kill)
    batch.terminate_children({101: (first, '0', {}, logs[0]), 102: (second, '1', {}, logs[1])})
    assert killed == [101, 102]
    first.wait.assert_called_once()
    second.wait.assert_called_once()
    assert all(log.close.called for log in logs)


@pytest.fixture
def ready_source(tmp_path):
    release = tmp_path / 'release'
    docs = [{'_id': f'original-{i:04d}', 'title': f'Title {i}', 'text': f'Document {i}',
             'language': 'en', 'metadata': {}} for i in range(1001)]
    bundle = {'schema_version': 'dci-data-bundle-v1', 'input_kind': 'retrieval',
              'source': {'dataset': 'batch-fixture', 'revision': 'fixed'}, 'documents': docs,
              'queries': [{'_id': 'q1', 'text': 'question\u2028with separator', 'language': 'en', 'metadata': {}}],
              'qrels': {'q1': {'original-1000': 2}}, 'evidence': [], 'judgments': [],
              'provenance': {}}
    publish_bundle(bundle, release, specification={'id': 'fixture', 'input_kind': 'retrieval'})
    manifest = batch.read(release / 'manifest.json')
    # Publication may flag lexical near-duplicates only as diagnostics, never silently alter fixtures.
    assert manifest['state'] == 'labels_ready'
    row = {'source_id': 'fixture', 'source_path': str(release), 'completion_sha256': batch.sha(release / '_COMPLETE.json'),
           'domain': 'test', 'language': 'en', 'documents': 1001, 'queries': 1, 'positive_pairs': 1}
    entry = batch.inspect_release(row)
    plan = {'parameters': {'model_path': batch.MODEL, 'trust_remote_code': True,
                          'query_prompt': batch.QUERY_PROMPT, 'document_prompt': '',
                          'weight_identity': 'fixture-weights', 'max_length': 8192, 'batch_size': 2,
                          'cache_shard_size': 128, 'search_query_batch_size': 2,
                          'top_k': 1000, 'capacities': [200, 500, 1000], 'seed': 20261021}}
    return entry, plan, tmp_path / 'run'


def fake_rank(output, *, suite, task, **kwargs):
    bundle = load_prepared_source(suite, task)
    ids = [d['_id'] for d in bundle['documents']]
    # One positive falls outside the scored top 1000 and must be injected with its true score.
    documents = np.array([[1., 0.] for _ in ids], dtype=np.float32)
    documents[-1] = [0., 1.]
    results = dense_search(np.array([[1., 0.]], dtype=np.float32), documents, ids,
                           query_ids=['q1'], top_k=1000, qrels=bundle['qrels'])
    config = EmbeddingConfig(model_path=batch.MODEL, weight_identity='fixture-weights',
                             query_prompt=kwargs['query_prompt'], document_prompt=kwargs['document_prompt'],
                             max_length=kwargs['max_length'])
    payload = export_saved_retrieval(bundle, results, model=config)
    batch.write(output, payload)


def test_cpu_export_rank_pool_roundtrip_resume_and_tamper_rejection(ready_source, monkeypatch):
    from dci_bench.data.pipeline import stages
    entry, plan, root = ready_source
    calls = []
    def ranking(*a, **kw):
        calls.append(kw)
        fake_rank(*a, **kw)
    monkeypatch.setattr(stages, 'rank_source', ranking)
    result = batch.execute_source(entry, plan, root, device='cpu')
    assert len(result['pools']) == 3
    assert calls[0]['query_prompt'] == batch.QUERY_PROMPT
    source = root / 'sources/fixture'
    for cap in (200, 500, 1000):
        pool = batch.read(source / f'pools/{cap}/metadata/fixture/candidate_pools.json')['pools']['q1']
        assert pool['num_docs'] == cap
        where = pool['candidate_original_doc_ids'].index('original-1000')
        assert pool['candidate_scores'][where] == 0.
    batch.execute_source(entry, plan, root, device='cpu')
    assert len(calls) == 1
    changed = copy.deepcopy(plan)
    changed['parameters']['query_prompt'] = 'Different prompt'
    with pytest.raises(ValueError, match='resume identity'):
        batch.execute_source(entry, changed, root, device='cpu')
    ranking_path = Path(batch.read(source / 'phases/rank.json')['path'])
    ranking_path.write_text('{}')
    with pytest.raises(ValueError, match='ranking changed'):
        batch.execute_source(entry, plan, root, device='cpu')


def test_failed_ranking_attempt_preserved_and_export_reused(ready_source, monkeypatch):
    from dci_bench.data.pipeline import stages
    entry, plan, root = ready_source
    def failure(output, **kwargs):
        output.parent.mkdir(parents=True)
        output.write_text('partial response')
        raise RuntimeError('injected ranking failure')
    monkeypatch.setattr(stages, 'rank_source', failure)
    with pytest.raises(RuntimeError, match='injected'):
        batch.execute_source(entry, plan, root, device='cpu')
    source = root / 'sources/fixture'
    partial = list((source / 'attempts').glob('rank-*/rankings.json'))
    assert len(partial) == 1 and partial[0].read_text() == 'partial response'
    export_receipt = (source / 'phases/export.json').read_bytes()
    monkeypatch.setattr(stages, 'rank_source', fake_rank)
    batch.execute_source(entry, plan, root, device='cpu')
    assert partial[0].read_text() == 'partial response'
    assert (source / 'phases/export.json').read_bytes() == export_receipt
    assert len(list((source / 'failures').glob('*.json'))) == 1


def test_invalid_successful_rank_does_not_poison_resume(ready_source, monkeypatch):
    from dci_bench.data.pipeline import stages
    entry, plan, root = ready_source
    def invalid(output, **kwargs):
        fake_rank(output, **kwargs)
        value = batch.read(output)
        value['model']['query_prompt'] = ''
        output.write_text(json.dumps(value))
    monkeypatch.setattr(stages, 'rank_source', invalid)
    with pytest.raises(ValueError, match='configuration mismatch'):
        batch.execute_source(entry, plan, root, device='cpu')
    source = root / 'sources/fixture'
    assert not (source / 'phases/rank.json').exists()
    monkeypatch.setattr(stages, 'rank_source', fake_rank)
    batch.execute_source(entry, plan, root, device='cpu')
    assert len(list((source / 'attempts').glob('rank-*/rankings.json'))) == 2


def test_export_retention_rejects_same_count_foreign_bundle(ready_source):
    from dci_bench.data.pipeline.core import export_dci
    entry, _, root = ready_source
    suite = root / 'export'
    export_dci(Path(entry['release']), suite, entry['task'])
    foreign = {**entry, 'bundle_hash': 'different-source-content'}
    with pytest.raises(ValueError, match='different source content'):
        batch._assert_export_retention(suite, foreign)


def test_real_rank_stage_interface_and_pool_roundtrip(ready_source, monkeypatch):
    from dci_bench.data.pipeline import retrieval
    entry, plan, root = ready_source
    class Encoder:
        def __init__(self, config):
            assert config.trust_remote_code is True
            assert config.query_prompt == batch.QUERY_PROMPT
            self.last_metadata = {}
        def _identities(self):
            return 'fixture-weights', 'fixture-tokenizer', 'fixture-config'
        def encode_with_cache(self, texts, *, kind, cache_dir, cache_shard_size=0):
            assert cache_shard_size == 128
            self.last_metadata = {'kind': kind, 'cache_hits': 0}
            return np.array([[0., 1.] if 'Document 1000' in text else [1., 0.] for text in texts], dtype=np.float32)
    monkeypatch.setattr(retrieval, 'SentenceTransformerEncoder', Encoder)
    # Keep the real rank_source signature, prompt propagation, scoring and export.
    result = batch.execute_source(entry, plan, root, device='cpu')
    assert [p['capacity'] for p in result['pools']] == [200, 500, 1000]


def test_parent_refills_four_scheduler_gpu_slots(tmp_path, monkeypatch):
    bundle, root = tmp_path / 'bundle', tmp_path / 'run'
    bundle.mkdir()
    (bundle / 'plan.json').write_text('{}')
    entries = [{'source_id': f's{i}', 'task': f's{i}', 'documents': 1000 + i, 'queries': 1} for i in range(6)]
    parameters = {'capacities': [200, 500, 1000]}
    plan = {'run_root': str(root), 'sources': entries, 'parameters': parameters}
    monkeypatch.setattr(batch, 'load_plan', lambda _: plan)
    monkeypatch.setattr(batch, 'preflight', lambda *a, **kw: {'test': True})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,5,6,7')
    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.delenv('SLURM_PROCID', raising=False)
    monkeypatch.setattr(batch.time, 'sleep', lambda _: None)
    launched, processes = [], []
    class Process:
        def __init__(self, command, *, env, **kwargs):
            self.pid = 100 + len(processes)
            self.polls = 0
            processes.append(self)
            source = command[command.index('--source') + 1]
            phase = command[command.index('--phase') + 1]
            launched.append((source, env['CUDA_VISIBLE_DEVICES'], phase))
            entry = next(e for e in entries if e['source_id'] == source)
            if phase == 'export':
                marker = root / 'sources' / source / 'attempts' / 'export-fixture' / 'suite' / 'metadata' / entry['task'] / '_COMPLETE.json'
                batch.write(marker, {'source': source})
                batch.write(root / 'sources' / source / 'phases' / 'export.json',
                            {'suite': str(marker.parents[2]), 'completion_sha256': batch.sha(marker)})
            elif phase == 'rank':
                ranked = root / 'sources' / source / 'attempts' / 'rank-fixture' / 'rankings.json'
                batch.write(ranked, {'source': source})
                batch.write(root / 'sources' / source / 'phases' / 'rank.json',
                            {'path': str(ranked), 'sha256': batch.sha(ranked)})
            else:
                rank = batch.read(root / 'sources' / source / 'phases' / 'rank.json')
                pools = []
                for capacity in parameters['capacities']:
                    target = root / 'sources' / source / 'pools' / str(capacity)
                    marker = target / 'metadata' / entry['task'] / '_COMPLETE.json'
                    batch.write(marker, {'capacity': capacity})
                    pools.append({'capacity': capacity, 'path': str(target),
                                  'completion_sha256': batch.sha(marker)})
                batch.write(root / 'sources' / source / '_COMPLETE.json', {
                    'schema_version': batch.SCHEMA,
                    'entry_hash': batch.digest(entry),
                    'parameters_hash': batch.digest(parameters),
                    'ranking': rank, 'pools': pools})
        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0
    monkeypatch.setattr(batch.subprocess, 'Popen', Process)
    result = batch.run(bundle)
    assert result['source_views'] == 6 and result['pools'] == 18
    assert [slot for _, slot, phase in launched if phase == 'rank'][:4] == ['3', '5', '6', '7']
    assert len({source for source, _, _ in launched}) == 6
    assert (root / '_COMPLETE.json').is_file()
