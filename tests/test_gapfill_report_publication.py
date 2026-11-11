"""Publication gates use synthetic receipts; no report or data is published."""
import json
from unittest.mock import patch

import pytest

from scripts.publish_gapfill_report_20261031 import SOURCES, publish


@pytest.fixture
def receipts(tmp_path):
    root = tmp_path
    gap = root / 'results/implementation/20261031-gapfill'
    gap.mkdir(parents=True)
    domains = ['encyclopedia', 'general_qa', 'scientific', 'medical', 'finance',
               'legal', 'enterprise', 'web', 'software', 'argumentation']
    tasks = {}
    for domain in domains:
        for lang in ('en', 'zh'):
            task = f'{domain}_{lang}'
            ready = task not in {'software_zh', 'enterprise_zh'}
            ids = {'dev': [f'{task}-d{i}' for i in range(10)],
                   'test': [f'{task}-t{i}' for i in range(50)]} if ready else {'dev': [], 'test': []}
            # ArguAna legitimately has multiple queries per topic group.
            groups = {qid: (f'{task}-{split}' if domain == 'argumentation' else qid)
                      for split, rows in ids.items() for qid in rows}
            tasks[task] = {'status': 'ready' if ready else 'blocked', 'source_id': task,
                           'query_ids': ids, 'query_groups': groups}
    source_rows = [{'source': source, 'documents': d, 'queries': q, 'positive_pairs': p,
                    'content_hashes_verified': True} for source, (d, q, p) in SOURCES.items()]
    values = {
        'prepared-manifest.json': {'tasks': tasks},
        'data-audit.json': {'task_count': 20, 'ready_for_formal': False},
        'suite-validation.json': {'old_ready': 13, 'new_status': {'ready': 18, 'blocked': 2},
            'stable_old_ready_tasks': 13, 'dev_groups': 162, 'test_groups': 802,
            'pool_records_checked': 2160, 'self_match_records_checked': 240, 'acord_train_excluded': 51},
        'retrieval-quality.json': {'sources': source_rows, 'embedding_gpu_inference_completed': True,
                                   'benchmark_model_inference_completed': False},
        'source-review.json': {'sources': source_rows, 'benchmark_model_inference_completed': False,
            'software_zh': {'status': 'blocked', 'native_provenance_verified': False,
                            'general_knowledge_with_tags': {'documents': 7481, 'queries': 7490, 'positive_pairs': 7490}},
            'enterprise_zh': {'status': 'deferred'}},
        'sampling-preview.json': {'ready_source_count': 18, 'old_13_query_selections_unchanged': True,
                                  'group_leakage': False},
        'data-preflight/preflight.json': {'jobs': 36, 'model_calls': 0, 'status': 'data_preflight_passed'},
    }
    for name, value in values.items():
        path = gap / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value))
    (gap / 'data-unit-tests.log').write_text('59 passed in 1.00s\n')
    old = root / 'results/implementation/20261029-lite-full/status.json'
    old.parent.mkdir()
    old.write_text(json.dumps({'data_tasks': [{'task': task, 'detail': 'historical'} for task in tasks],
                               'timing_rows': []}))
    return root, gap


def test_publication_accepts_fewer_groups_than_queries(receipts):
    root, _ = receipts
    with patch('scripts.update_benchmark_progress.update') as render:
        publish(root)
    status = render.call_args.args[1]
    assert '18/20' in status['summary']
    assert status['data_preparation']['benchmark_model_inference_completed'] is False
    assert all('待pool审计' not in row['detail'] for row in status['data_tasks'])


@pytest.mark.parametrize('failure', ['partial', 'unverified_bodies', 'failed_tests'])
def test_publication_rejects_incomplete_evidence_before_render(receipts, failure):
    root, gap = receipts
    if failure == 'failed_tests':
        (gap / 'data-unit-tests.log').write_text('1 failed, 59 passed in 1.00s\n')
    else:
        path = gap / ('prepared-manifest.json' if failure == 'partial' else 'retrieval-quality.json')
        payload = json.loads(path.read_text())
        if failure == 'partial':
            payload['tasks']['web_en']['status'] = 'blocked'
        else:
            payload['sources'][0]['content_hashes_verified'] = False
        path.write_text(json.dumps(payload))
    with patch('scripts.update_benchmark_progress.update') as render, pytest.raises(ValueError):
        publish(root)
    render.assert_not_called()
