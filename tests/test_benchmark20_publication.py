"""Report gates require complete CPU evidence and keep browser checks separate."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import publish_benchmark20_20261103 as publisher


@pytest.fixture
def accepted(tmp_path):
    evidence, suite = tmp_path / 'evidence', tmp_path / 'suite'
    evidence.mkdir()
    suite.mkdir()
    tasks = [f'task_{i}' for i in range(19)] + ['web_en']
    prepared = {'overall_sha256': 'fixture-suite', 'tasks': {task: {'status': 'ready'} for task in tasks}}
    payload = json.dumps(prepared)
    (suite / 'manifest.json').write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    values = {
        'prepared-manifest.json': prepared,
        'status-ready.json': {'status': 'data_accepted', 'suite': str(suite), 'ready_tasks': 20,
                              'blocked_tasks': [], 'manifest_sha256': digest},
        'data-audit.json': {'ready_tasks': tasks, 'blocked_tasks': [], 'ready_for_formal': True,
                            'overall_sha256': 'fixture-suite'},
        'data-preflight/preflight.json': {'status': 'data_preflight_passed', 'jobs': 40, 'model_calls': 0},
        'data-preflight/launch.json': {'preflight_only': True, 'selection_sha256': digest,
                                      'execution_count': 40, 'complete_scope': True, 'blocked_tasks': []},
        'old-19-stability.json': {'stable_old_task_count': 19, 'changed': {}},
        'suite-validation.json': {'passed': True, 'pool_records_checked': 2400, 'global_group_leakage': False},
        'selected-body-validation.json': {'query_count': 60, 'verified_unique_body_count': 1234},
        'config.json': {'sources': {}}, 'catalog.json': {}, 'source-closure.json': {},
    }
    for name, value in values.items():
        path = evidence / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value))
    return evidence, suite, tasks


@pytest.mark.parametrize('failure', [None, 'jobs', 'model_calls', 'selection', 'missing_body_check'])
def test_20_task_publication_acceptance_gate(accepted, failure):
    evidence, suite, _ = accepted
    if failure in ('jobs', 'model_calls'):
        path = evidence / 'data-preflight/preflight.json'
        value = publisher.read(path)
        value[failure] = 38 if failure == 'jobs' else 1
        path.write_text(json.dumps(value))
    elif failure == 'selection':
        path = evidence / 'data-preflight/launch.json'
        value = publisher.read(path)
        value['selection_sha256'] = 'different-suite'
        path.write_text(json.dumps(value))
    elif failure == 'missing_body_check':
        path = evidence / 'selected-body-validation.json'
        path.write_text('{"query_count":60,"verified_unique_body_count":0}')
    if failure:
        with pytest.raises(ValueError):
            publisher.accepted_data(evidence)
    else:
        assert publisher.accepted_data(evidence)[0] == suite


def test_report_publication_does_not_run_browser_or_mark_model_complete(accepted, tmp_path):
    evidence, _, tasks = accepted
    report = tmp_path / 'docs/project-progress-20261028'
    report.mkdir(parents=True)
    (report / 'index.html').write_text('<p>old report</p>')
    (report / 'benchmark-v2-status.json').write_text(json.dumps({
        'data_tasks': [{'task': task, 'source': task, 'status': '阻塞' if task == 'web_en' else '已准备'} for task in tasks],
        'limitations': [], 'evidence': []}))
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/benchmark_v2.json').write_text((evidence / 'config.json').read_text())
    with (patch.object(publisher, 'ROOT', tmp_path), patch.object(publisher, 'build_coverage'),
          patch.object(publisher, 'update') as render,
          patch('subprocess.run', side_effect=AssertionError('browser must run independently'))):
        result = publisher.publish(evidence)
    assert result['ready_tasks'] == 20
    assert publisher.read(evidence / 'integration-complete.json')['status'] == 'data_complete'
    status = render.call_args.args[1]
    assert status['data_ready_for_formal'] is True
    assert status['formal_model_execution_ready'] is False
    assert status['data_preparation']['benchmark_model_inference_completed'] is False
    web = next(row for row in status['data_tasks'] if row['task'] == 'web_en')
    assert web['status'] == '已准备' and web['source'] == publisher.SOURCE


def test_stale_browser_receipt_cannot_change_data_acceptance(accepted, tmp_path):
    evidence, _, _ = accepted
    report = tmp_path / 'docs/project-progress-20261028'
    report.mkdir(parents=True)
    (report / 'index.html').write_text('<p>current report</p>')
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/benchmark_v2.json').write_text((evidence / 'config.json').read_text())
    browser = tmp_path / 'browser.json'
    browser.write_text('{"passed":true,"html_sha256":"different"}')
    before = (evidence / 'status-ready.json').read_bytes()
    with patch.object(publisher, 'ROOT', tmp_path), pytest.raises(ValueError, match='published HTML'):
        publisher.publish(evidence, browser)
    assert (evidence / 'status-ready.json').read_bytes() == before
