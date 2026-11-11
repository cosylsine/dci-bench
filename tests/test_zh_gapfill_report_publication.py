"""CPU publication recovery preserves accepted data and failed browser evidence."""
import json
import hashlib
import subprocess
from unittest.mock import patch

import pytest

from scripts import publish_zh_gapfill_report_20261102 as publisher


@pytest.mark.parametrize('changed', [None, 'ranking', 'completion', 'metadata'])
def test_bcp_report_checks_hashes_without_reading_corpus(tmp_path, changed):
    source = 'gapfill_browsecomp_plus_en'
    root = tmp_path / source
    root.mkdir()
    ranking = root / 'ranking.json'
    ranking.write_text('{}')
    row = {'source': source, 'task': source, 'ranking': str(ranking),
           'ranking_sha256': hashlib.sha256(ranking.read_bytes()).hexdigest(), 'pools': []}
    for capacity in (200, 1000):
        suite = root / f'pools-{capacity}'
        metadata = suite / 'metadata' / source
        metadata.mkdir(parents=True)
        inventory = metadata / 'file_inventory.json'
        # Bodies are deliberately unavailable: publication checks only metadata.
        inventory.write_text('{"corpus/unavailable.txt":"sealed-body"}')
        marker = metadata / '_COMPLETE.json'
        marker.write_text(json.dumps({'content_hashes_verified': True, 'metadata_hashes': {
            inventory.name: hashlib.sha256(inventory.read_bytes()).hexdigest()}}))
        row['pools'].append({'capacity': capacity, 'path': str(suite),
                             'completion_sha256': hashlib.sha256(marker.read_bytes()).hexdigest()})
    receipt = root / 'catalog-row.json'
    receipt.write_text(json.dumps(row))
    if changed:
        {'ranking': ranking, 'completion': marker, 'metadata': inventory}[changed].write_text('{}\n')
    with patch.object(publisher, 'RUN', tmp_path):
        if changed:
            with pytest.raises(ValueError, match='BrowseComp\\+'):
                publisher.bcp_completion_metadata()
        else:
            assert publisher.bcp_completion_metadata() == receipt


@pytest.fixture
def publication(tmp_path):
    data = tmp_path / 'data'
    evidence = data / 'evidence/suite-acceptance'
    report = tmp_path / 'docs/project-progress-20261028'
    audit = {'ready_tasks': [f'task-{i}' for i in range(19)],
             'blocked_tasks': ['web_en'], 'overall_sha256': 'accepted-suite',
             'body_hashes_checked': False}
    sources = [{'task': task, 'source': task, 'documents': 2000, 'queries': 60,
                'positive_pairs': 60, 'positive_min': 1, 'positive_max': 1}
               for task in ('software_zh', 'enterprise_zh')]
    config = {'sources': {row['task']: row['source'] for row in sources}}
    files = {
        evidence / 'status-ready.json': {'status': 'ready'},
        evidence / 'prepared-manifest.json': {'overall_sha256': 'accepted-suite'},
        evidence / 'config.json': config,
        evidence / 'report-audit.json': audit,
        data / 'evidence/data-preflight/preflight.json': {
            'jobs': 38, 'model_calls': 0, 'status': 'data_preflight_passed'},
        data / 'source-readiness.json': {'sources': sources},
        report / 'benchmark-v2-status.json': {
            'data_tasks': [{'task': 'web_en', 'detail': 'pending'}], 'evidence': []},
        tmp_path / 'configs/benchmark_v2.json': config,
    }
    for path, value in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    browser_script = tmp_path / 'results/implementation/20261102-gapfill-suite-v2/report-ui/check_browser.py'
    browser_script.parent.mkdir(parents=True)
    browser_script.write_text('# fixture browser check\n')
    failed_log = tmp_path / 'results/implementation/20261102-zh-gapfill-accepted/report-ui/browser-check.log'
    failed_log.parent.mkdir(parents=True)
    failed_log.write_text('libdbus-1.so.3: cannot open shared object file\n')
    with (patch.object(publisher, 'ROOT', tmp_path),
          patch.object(publisher, 'audit_benchmark_v2', return_value=audit),
          patch.object(publisher, 'bcp_completion_metadata',
                       return_value=tmp_path / 'bcp/catalog-row.json'),
          patch.object(publisher, 'build_coverage'),
          patch.object(publisher, 'update') as render,
          patch.object(publisher.subprocess, 'run') as run):
        yield tmp_path, data, evidence, report, failed_log, render, run


def test_resume_reuses_matching_audit_and_preserves_failed_attempt(publication):
    root, data, evidence, report, failed_log, render, run = publication
    audit_before = (evidence / 'report-audit.json').read_bytes()
    failed_before = failed_log.read_bytes()
    publisher.publish(data)
    assert run.call_count == 2
    assert render.call_count == 2
    assert (evidence / 'report-audit.json').read_bytes() == audit_before
    assert failed_log.read_bytes() == failed_before
    assert len(list((root / 'results/implementation').glob('zh-gapfill-report-*/report-ui/browser-check.log'))) == 1
    status = render.call_args.args[1]
    assert '19/20' in status['summary']
    assert '源池已完成' in status['summary']
    web = next(row for row in status['data_tasks'] if row['task'] == 'web_en')
    assert web['source'] == 'BrowseComp+（源池完成，待suite合入）'
    assert web['pending_reason'] == 'source pools complete; awaiting suite integration'
    assert status['data_preparation']['benchmark_model_inference_completed'] is False
    # Simulate a second invocation after the first report was written.
    (report / 'benchmark-v2-status.json').write_text(json.dumps(status))
    publisher.publish(data)
    assert len(list((root / 'results/implementation').glob('zh-gapfill-report-*/report-ui/browser-check.log'))) == 2
    paths = [row['path'] for row in render.call_args.args[1]['evidence']]
    assert len(paths) == len(set(paths))


def test_different_existing_audit_stops_before_report_writes(publication):
    root, data, evidence, report, _, render, run = publication
    (evidence / 'report-audit.json').write_text('{"overall_sha256":"different"}')
    with pytest.raises(ValueError, match='Existing report audit differs'):
        publisher.publish(data)
    render.assert_not_called()
    run.assert_not_called()
    assert not (report / 'history').exists()


def test_missing_browser_library_fails_before_publishing(publication):
    _, data, evidence, report, failed_log, render, run = publication
    before = {p: p.read_bytes() for p in (evidence / 'report-audit.json',
                                         report / 'benchmark-v2-status.json', failed_log)}
    run.side_effect = subprocess.CalledProcessError(1, ['browser-preflight'])
    with pytest.raises(subprocess.CalledProcessError):
        publisher.publish(data)
    render.assert_not_called()
    assert not (report / 'history').exists()
    assert all(p.read_bytes() == value for p, value in before.items())
