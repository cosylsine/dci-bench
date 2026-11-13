import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('benchmark_runner', ROOT / 'scripts/run_benchmark_v2.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BenchmarkV2RunnerTest(unittest.TestCase):
    def test_preflight_runs_all_40_jobs_with_bounded_overlap_and_isolated_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'logs').mkdir()
            args = argparse.Namespace(output=root, preflight_concurrency=4)
            jobs = [{'task_id': f'task{i}', 'tier': tier}
                    for i in range(20) for tier in ('lite', 'full')]
            script = '''
import json, os, pathlib, sys, time
root = pathlib.Path(sys.argv[1])
name = sys.argv[2]
started = time.monotonic()
(root / (name + '.ready')).touch()
deadline = started + 5
while len(list(root.glob('*.ready'))) < 4:
    assert time.monotonic() < deadline, 'first four jobs must overlap'
    time.sleep(.01)
time.sleep(.03)
print(json.dumps({'start': started, 'end': time.monotonic(),
                  'profile': os.environ['DCI_PROTOCOL_PROFILE'],
                  'budget': os.environ['DCI_DOCUMENT_TOKEN_BUDGET']}))
'''
            env = dict(os.environ, DCI_DOCUMENT_TOKEN_BUDGET='64000')
            with patch.object(runner, 'command_for', side_effect=lambda job, *_args, **_kw: [
                    runner.sys.executable, '-c', script, tmp, job['task_id'] + '-' + job['tier']]):
                runner.run_data_preflight(jobs, args, env=env)
            self.assertNotIn('DCI_PROTOCOL_PROFILE', env)
            events = []
            for job in jobs:
                row = json.loads((root / 'logs' / (job['task_id'] + '-' + job['tier'] + '.log')).read_text())
                self.assertEqual(row['profile'], 'document-pool-v2-' + job['tier'])
                self.assertEqual(row['budget'], '64000')
                events.extend([(row['start'], 1), (row['end'], -1)])
            active = peak = 0
            for _, change in sorted(events):
                active += change
                peak = max(peak, active)
            self.assertEqual(peak, 4)
            self.assertEqual(active, 0)

    def test_preflight_failure_timeout_and_interrupt_reap_inflight_children(self):
        for failure in ('exit', 'timeout', 'interrupt', 'spawn'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'logs').mkdir()
                args = argparse.Namespace(output=root, preflight_concurrency=2)
                jobs = [{'task_id': str(i), 'tier': 'lite'} for i in range(5)]
                children = []
                popen = runner.subprocess.Popen

                def spawn(*a, **kw):
                    if failure == 'spawn' and children:
                        raise OSError('spawn fixture failed')
                    child = popen(*a, **kw)
                    children.append(child)
                    return child

                def command(job, *_args, **_kw):
                    code = "import time; time.sleep(60)"
                    if failure == 'exit' and job['task_id'] == '0':
                        code = "import sys; print('bad data', flush=True); sys.exit(7)"
                    return [runner.sys.executable, '-c', code]

                with patch.object(runner, 'command_for', side_effect=command), \
                        patch.object(runner.subprocess, 'Popen', side_effect=spawn):
                    if failure == 'interrupt':
                        with patch.object(runner, 'time', wraps=time) as clock, \
                                self.assertRaises(KeyboardInterrupt):
                            clock.sleep.side_effect = KeyboardInterrupt
                            runner.run_data_preflight(jobs, args, env=dict(os.environ))
                    else:
                        error = OSError if failure == 'spawn' else RuntimeError
                        pattern = {'exit': r'0-lite \(exit 7\).*log:',
                                   'timeout': 'timed out.*log:', 'spawn': 'spawn fixture failed'}[failure]
                        with self.assertRaisesRegex(error, pattern):
                            runner.run_data_preflight(jobs, args, env=dict(os.environ),
                                                      timeout=.15 if failure == 'timeout' else 10)
                self.assertEqual(len(children), 1 if failure == 'spawn' else 2)
                self.assertTrue(all(child.poll() is not None for child in children))
                if failure == 'exit':
                    self.assertIn('bad data', (root / 'logs/0-lite.log').read_text())

    def test_preflight_publishes_success_only_after_all_checks_and_records_concurrency(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'run'
                with patch.object(runner, 'build_plan', return_value={'mode': 'local', 'jobs': []}), \
                        patch.object(runner, 'run_data_preflight',
                                     side_effect=RuntimeError('bad data') if fail else None):
                    argv = ['--suite', tmp, '--preflight-only', '--preflight-concurrency', '2',
                            '--output', str(output)]
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'bad data'):
                            runner.main(argv)
                        self.assertFalse((output / 'preflight.json').exists())
                    else:
                        self.assertEqual(runner.main(argv), 0)
                        report = json.loads((output / 'preflight.json').read_text())
                        self.assertEqual(report['preflight_concurrency'], 2)
                        self.assertEqual(report['model_calls'], 0)
                        self.assertGreaterEqual(report['elapsed_seconds'], 0)
                    self.assertEqual(json.loads((output / 'launch.json').read_text())['preflight_concurrency'], 2)
                    self.assertNotIn('active_session_started_at', json.loads((output / 'ledger.json').read_text()))

    def test_one_batch_contains_all_selected_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(concurrency=8, split='test', output=Path(tmp),
                                      backend_manifest=Path(tmp) / 'backend.json', bridge_port=28100)
            job = {'task_id': 'medical_en', 'tier': 'lite', 'suite': tmp,
                   'query_ids': [str(i) for i in range(50)]}
            command = runner.command_for(job, args, 'fixture')
            begin = command.index('--query-ids') + 1
            end = command.index('--max-concurrency')
            self.assertEqual(command[begin:end], job['query_ids'])
            self.assertIn('--run-id', command)
            (Path(tmp) / 'evaluation/fixture/medical_en/runs/test-lite').mkdir(parents=True)
            self.assertIn('--resume-run', runner.command_for(job, args, 'fixture'))

    def test_command_for_passes_runtime_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                concurrency=8, split='test', output=Path(tmp),
                backend_manifest=Path(tmp) / 'backend.json', bridge_port=28100,
                runtime_config=Path(tmp) / 'runtime.json',
            )
            job = {'task_id': 'medical_en', 'tier': 'lite', 'suite': tmp,
                   'query_ids': ['q1']}
            command = runner.command_for(job, args, 'fixture')
            self.assertEqual(command[command.index('--runtime-config') + 1], str(args.runtime_config))

    def test_plan_does_not_spawn_or_require_endpoint(self):
        with patch.object(runner, 'build_plan', return_value={'mode': 'local', 'execution_count': 2000}), \
                patch.object(runner.subprocess, 'Popen') as spawn:
            self.assertEqual(runner.main(['--suite', '/tmp/fixture']), 0)
            spawn.assert_not_called()

    def test_execute_requires_backend_before_spawning(self):
        with patch.object(runner, 'build_plan', return_value={'mode': 'local'}), \
                patch.object(runner.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'backend-manifest'):
                runner.main(['--suite', '/tmp/fixture', '--execute', '--output', '/tmp/fixture-output'])
            spawn.assert_not_called()

    def test_nonfinite_spending_limit_rejected(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            runner.main(['--suite', '/tmp/fixture', '--budget-cny', 'nan'])

    def test_source_fingerprint_resolves_runner_path(self):
        original = runner.__file__
        try:
            runner.__file__ = 'scripts/run_benchmark_v2.py'
            self.assertRegex(runner.source_fingerprint(), r'^[0-9a-f]{64}$')
        finally:
            runner.__file__ = original

    def test_formal_calibration_cost_rejects_unavailable_evidence(self):
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            runner._report_accounted_cost(
                {'cost_cny': None, 'spending_guard': {}}, label='32000'
            )

    def test_formal_api_prior_cost_exhausting_budget_is_rejected(self):
        from dci_bench.results.benchmark_v2 import fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = {}
            for budget in (32000, 64000, 96000):
                report = {
                    'split': 'dev', 'completed': 400, 'valid': 400,
                    'complete_scope': True, 'token_budget': budget,
                    'selection_sha256': 'selection', 'backend_sha256': 'backend',
                    'concurrency': 4, 'source_fingerprint': 'source',
                    'cost_cny': 100.0, 'spending_guard': {'accounted_or_reserved_cny': 100.0},
                }
                path = root / f'{budget}.json'
                path.write_text(json.dumps(report), encoding='utf-8')
                reports[budget] = path
            decision = {
                'selected_budget': 64000,
                'candidates': [],
                'reports': {
                    str(budget): {'path': str(path.resolve()), 'sha256': runner.sha256_file(path)}
                    for budget, path in reports.items()
                },
            }
            decision['sha256'] = fingerprint(decision)
            decision_path = root / 'selection.json'
            decision_path.write_text(json.dumps(decision), encoding='utf-8')
            prices = root / 'prices.json'
            prices.write_text(json.dumps({
                'currency': 'CNY', 'model_key': 'fixture', 'as_of': 'today', 'source': 'fixture',
                'input_per_million': 1, 'output_per_million': 1,
                'cache_read_per_million': 1, 'cache_write_per_million': 1,
            }), encoding='utf-8')
            backend_path = root / 'backend.json'
            backend_path.write_text('{}', encoding='utf-8')
            plan = {
                'mode': 'api', 'model_key': 'fixture', 'split': 'test', 'execution_count': 1,
                'selection_sha256': 'selection', 'backend_sha256': 'backend', 'concurrency': 4,
                'token_budget': 64000, 'source_fingerprint': 'source', 'jobs': [],
                'expected_tasks': [], 'complete_scope': True,
            }
            with patch.object(runner, 'build_plan', return_value=plan), \
                    patch.object(runner, 'validate_complete_report'), \
                    patch.object(runner, 'select_budget', return_value={'selected_budget': 64000, 'candidates': []}), \
                    patch.object(runner, 'cost_gate', return_value={
                        'passed': True, 'failures': [], 'projections': {'formal_cost_cny': 700.0}
                    }), \
                    patch('dci_bench.backends.manifest.load_backend_manifest', return_value={
                        'backend': {'context_length': 65536}, 'model': {'model_key': 'fixture'}
                    }):
                with self.assertRaisesRegex(ValueError, 'prior cost'):
                    runner.main([
                        '--suite', str(root), '--split', 'test', '--execute',
                        '--backend-manifest', str(backend_path), '--prices', str(prices),
                        '--calibration', str(reports[64000]), '--budget-selection', str(decision_path),
                        '--prior-cost-cny', '100', '--budget-cny', '1000',
                        '--output', str(root / 'output'),
                    ])

    def test_formal_api_first_attempt_reservation_uses_eighty_percent_ceiling(self):
        from dci_bench.results.benchmark_v2 import fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = {}
            for budget in (32000, 64000, 96000):
                report = {
                    'split': 'dev', 'completed': 400, 'valid': 400,
                    'complete_scope': True, 'token_budget': budget,
                    'selection_sha256': 'selection', 'backend_sha256': 'backend',
                    'concurrency': 4, 'source_fingerprint': 'source',
                    'cost_cny': 0.0, 'spending_guard': {'accounted_or_reserved_cny': 0.0},
                }
                path = root / f'{budget}.json'
                path.write_text(json.dumps(report), encoding='utf-8')
                reports[budget] = path
            decision = {
                'selected_budget': 64000,
                'candidates': [],
                'reports': {
                    str(budget): {'path': str(path.resolve()), 'sha256': runner.sha256_file(path)}
                    for budget, path in reports.items()
                },
            }
            decision['sha256'] = fingerprint(decision)
            decision_path = root / 'selection.json'
            decision_path.write_text(json.dumps(decision), encoding='utf-8')
            prices = root / 'prices.json'
            prices.write_text(json.dumps({
                'currency': 'CNY', 'model_key': 'fixture', 'as_of': 'today', 'source': 'fixture',
                'input_per_million': 15000, 'output_per_million': 15000,
                'cache_read_per_million': 15000, 'cache_write_per_million': 15000,
            }), encoding='utf-8')
            backend_path = root / 'backend.json'
            backend_path.write_text('{}', encoding='utf-8')
            plan = {
                'mode': 'api', 'model_key': 'fixture', 'split': 'test', 'execution_count': 1,
                'selection_sha256': 'selection', 'backend_sha256': 'backend', 'concurrency': 4,
                'token_budget': 64000, 'source_fingerprint': 'source',
                'jobs': [{'task_id': 'fixture', 'tier': 'lite', 'query_ids': ['q1'], 'groups': {}}],
                'expected_tasks': ['fixture'], 'complete_scope': True,
            }
            with patch.object(runner, 'build_plan', return_value=plan), \
                    patch.object(runner, 'validate_complete_report'), \
                    patch.object(runner, 'select_budget', return_value={'selected_budget': 64000, 'candidates': []}), \
                    patch.object(runner, 'cost_gate', return_value={
                        'passed': True, 'failures': [], 'projections': {'formal_cost_cny': 0.0}
                    }), \
                    patch('dci_bench.backends.manifest.load_backend_manifest', return_value={
                        'backend': {'context_length': 65536}, 'model': {'model_key': 'fixture'}
                    }), patch('scripts.inspect_local_runtime.prepare'), \
                    patch.object(runner, 'source_fingerprint', return_value='source'), \
                    patch.object(runner, 'run_child') as child:
                with self.assertRaisesRegex(ValueError, '80% ceiling'):
                    runner.main([
                        '--suite', str(root), '--split', 'test', '--execute',
                        '--backend-manifest', str(backend_path), '--prices', str(prices),
                        '--calibration', str(reports[64000]), '--budget-selection', str(decision_path),
                        '--prior-cost-cny', '0', '--output', str(root / 'output'),
                    ])
                child.assert_not_called()

    def test_formal_budget_selection_requires_all_three_report_references(self):
        from dci_bench.results.benchmark_v2 import fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / 'calibration.json'
            calibration.write_text(json.dumps({
                'split': 'dev', 'completed': 400, 'valid': 400,
                'complete_scope': True, 'token_budget': 64000,
                'selection_sha256': 'selection', 'backend_sha256': 'backend',
                'concurrency': 4, 'source_fingerprint': 'source',
                'cost_cny': 100.0,
            }), encoding='utf-8')
            decision = {'selected_budget': 64000, 'candidates': [], 'reports': {}}
            decision['sha256'] = fingerprint(decision)
            decision_path = root / 'selection.json'
            decision_path.write_text(json.dumps(decision), encoding='utf-8')
            prices = root / 'prices.json'
            prices.write_text(json.dumps({
                'currency': 'CNY', 'model_key': 'fixture', 'as_of': 'today', 'source': 'fixture',
                'input_per_million': 1, 'output_per_million': 1,
                'cache_read_per_million': 1, 'cache_write_per_million': 1,
            }), encoding='utf-8')
            backend_path = root / 'backend.json'
            backend_path.write_text('{}', encoding='utf-8')
            plan = {
                'mode': 'api', 'model_key': 'fixture', 'split': 'test', 'execution_count': 1,
                'selection_sha256': 'selection', 'backend_sha256': 'backend', 'concurrency': 4,
                'token_budget': 64000, 'source_fingerprint': 'source', 'jobs': [],
                'expected_tasks': [], 'complete_scope': True,
            }
            with patch.object(runner, 'build_plan', return_value=plan), \
                    patch.object(runner, 'validate_complete_report'), \
                    patch('dci_bench.backends.manifest.load_backend_manifest', return_value={
                        'backend': {'context_length': 65536}, 'model': {'model_key': 'fixture'}
                    }):
                with self.assertRaisesRegex(ValueError, 'all three'):
                    runner.main([
                        '--suite', str(root), '--split', 'test', '--execute',
                        '--backend-manifest', str(backend_path), '--prices', str(prices),
                        '--calibration', str(calibration), '--budget-selection', str(decision_path),
                        '--prior-cost-cny', '0', '--output', str(root / 'output'),
                    ])

    def test_completed_batch_rejects_forged_task_identity(self):
        from test_phase4_runner import Phase4RunnerTest
        from dci_bench.results.phase4_runner import execute_phase4
        from dci_bench.results.protocol import sha256_json

        fixture = Phase4RunnerTest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = fixture._fixture(root)
            execution = execute_phase4(
                args,
                selected_query_ids=['Q1'],
                inspect_eval_fn=fixture._fake_eval(),
                task_factory=fixture._task_factory,
            )
            task_result_path = execution.run_dir / 'task-result.json'
            task_result = json.loads(task_result_path.read_text())
            task_result['task'] = 'OtherTask'
            task_result_path.write_text(json.dumps(task_result), encoding='utf-8')
            marker_path = execution.run_dir / '_COMPLETE'
            marker = json.loads(marker_path.read_text())
            marker['task_result_sha256'] = sha256_json(task_result)
            marker_path.write_text(json.dumps(marker), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'task identity'):
                runner.completed_samples(execution.run_dir, {
                    'query_ids': ['Q1'], 'task_id': args.task, 'tier': 'lite', 'groups': {'Q1': 'g1'}
                })

    def test_completed_batch_requires_matching_hash_and_selection(self):
        from dci_bench.results.protocol import sha256_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'task-result.json').write_text(json.dumps({'query_ids': ['q1'], 'samples': []}))
            (root / '_COMPLETE').write_text(json.dumps({'task_result_sha256': sha256_json(json.loads((root / 'task-result.json').read_text()))}))
            with self.assertRaisesRegex(ValueError, 'selection'):
                runner.completed_samples(root, {'query_ids': ['different']})
            (root / 'task-result.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                runner.completed_samples(root, {'query_ids': ['q1']})

    def test_resume_cannot_change_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            output.mkdir()
            (output / 'launch.json').write_text('{}')
            with patch.object(runner, 'build_plan', return_value={'mode': 'local'}), \
                    patch.object(runner.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    runner.main(['--suite', '/tmp/fixture', '--preflight-only', '--resume', '--output', str(output)])
                spawn.assert_not_called()

    def test_unclean_session_elapsed_time_is_not_reset_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                runner, 'build_plan', side_effect=lambda *args, **kwargs: {'mode': 'local', 'jobs': []}):
            output = Path(tmp) / 'run'
            args = ['--suite', '/tmp/fixture', '--preflight-only', '--output', str(output)]
            self.assertEqual(runner.main(args), 0)
            ledger = json.loads((output / 'ledger.json').read_text())
            ledger.update(elapsed_seconds=100, active_session_started_at=time.time() - 30)
            (output / 'ledger.json').write_text(json.dumps(ledger))
            self.assertEqual(runner.main([*args, '--resume']), 0)
            ledger = json.loads((output / 'ledger.json').read_text())
            self.assertGreaterEqual(ledger['elapsed_seconds'], 130)
            self.assertNotIn('active_session_started_at', ledger)

    def test_reader_accepts_the_actual_phase4_canonical_publication(self):
        from test_phase4_runner import Phase4RunnerTest
        from dci_bench.results.phase4_runner import execute_phase4
        fixture = Phase4RunnerTest()
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture._fixture(Path(tmp))
            execution = execute_phase4(args, selected_query_ids=['Q1'],
                inspect_eval_fn=fixture._fake_eval(), task_factory=fixture._task_factory)
            samples = runner.completed_samples(execution.run_dir, {
                'query_ids': ['Q1'], 'task_id': args.task, 'tier': 'lite', 'groups': {'Q1': 'g1'}})
            self.assertEqual(len(samples), 1)
            self.assertTrue(samples[0]['valid'])
            self.assertEqual(samples[0]['usage']['total_tokens'], 15)
            with self.assertRaisesRegex(ValueError, 'agent_runtime'):
                runner.completed_samples(execution.run_dir, {
                    'query_ids': ['Q1'], 'task_id': args.task, 'tier': 'lite', 'groups': {'Q1': 'g1'},
                    'expected_agent_runtime': {
                        'schema_version': 'dci-agent-runtime-v1',
                        'mode': 'frozen-rootfs',
                        'rootfs_tree_sha256': 'a' * 64,
                    },
                })

    def test_host_reader_rejects_frozen_manifest_and_result(self):
        from test_phase4_runner import Phase4RunnerTest
        from dci_bench.results.phase4_runner import execute_phase4
        from dci_bench.results.protocol import compute_run_fingerprint, sha256_json

        fixture = Phase4RunnerTest()
        runtime = {
            'schema_version': 'dci-agent-runtime-v1',
            'mode': 'frozen-rootfs',
            'rootfs_tree_sha256': 'a' * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture._fixture(Path(tmp))
            execution = execute_phase4(
                args,
                selected_query_ids=['Q1'],
                inspect_eval_fn=fixture._fake_eval(),
                task_factory=fixture._task_factory,
            )
            manifest_path = execution.run_dir / 'run-manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest.setdefault('code', {})['agent_runtime'] = runtime
            manifest['run_fingerprint'] = compute_run_fingerprint(
                model=manifest['model'],
                backend=manifest['backend'],
                data=manifest['data'],
                benchmark_contract=manifest['benchmark_contract'],
                metric_ks=manifest['metric_ks'],
                query_ids=manifest['query_ids'],
                code=manifest['code'],
            )
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            result_path = execution.run_dir / 'task-result.json'
            result = json.loads(result_path.read_text())
            result.setdefault('code', {})['agent_runtime'] = runtime
            result_path.write_text(json.dumps(result), encoding='utf-8')
            marker_path = execution.run_dir / '_COMPLETE'
            marker = json.loads(marker_path.read_text())
            marker['task_result_sha256'] = sha256_json(result)
            marker['run_fingerprint'] = manifest['run_fingerprint']
            marker_path.write_text(json.dumps(marker), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'host completed batch'):
                runner.completed_samples(
                    execution.run_dir,
                    {'query_ids': ['Q1'], 'task_id': args.task, 'tier': 'lite', 'groups': {'Q1': 'g1'}},
                )

    def test_frozen_reader_requires_final_and_trace_artifacts(self):
        from test_phase4_runner import Phase4RunnerTest
        from dci_bench.results.phase4_runner import execute_phase4
        from dci_bench.results.protocol import compute_run_fingerprint, sha256_json

        fixture = Phase4RunnerTest()
        runtime = {
            'schema_version': 'dci-agent-runtime-v1',
            'mode': 'frozen-rootfs',
            'rootfs_tree_sha256': 'a' * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture._fixture(Path(tmp))
            execution = execute_phase4(
                args,
                selected_query_ids=['Q1'],
                inspect_eval_fn=fixture._fake_eval(),
                task_factory=fixture._task_factory,
            )
            manifest_path = execution.run_dir / 'run-manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest.setdefault('code', {})['agent_runtime'] = runtime
            manifest['run_fingerprint'] = compute_run_fingerprint(
                model=manifest['model'],
                backend=manifest['backend'],
                data=manifest['data'],
                benchmark_contract=manifest['benchmark_contract'],
                metric_ks=manifest['metric_ks'],
                query_ids=manifest['query_ids'],
                code=manifest['code'],
            )
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            sample_path = execution.run_dir / 'Q1' / 'result.json'
            sample = json.loads(sample_path.read_text())
            sample.setdefault('provenance', {})['agent_runtime'] = runtime
            sample['artifacts'].pop('trace')
            sample_path.write_text(json.dumps(sample), encoding='utf-8')
            result_path = execution.run_dir / 'task-result.json'
            result = json.loads(result_path.read_text())
            result.setdefault('code', {})['agent_runtime'] = runtime
            result['samples'][0]['result']['sha256'] = sha256_json(sample)
            result_path.write_text(json.dumps(result), encoding='utf-8')
            marker_path = execution.run_dir / '_COMPLETE'
            marker = json.loads(marker_path.read_text())
            marker['task_result_sha256'] = sha256_json(result)
            marker['run_fingerprint'] = manifest['run_fingerprint']
            marker_path.write_text(json.dumps(marker), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'final/trace artifact'):
                runner.completed_samples(
                    execution.run_dir,
                    {
                        'query_ids': ['Q1'],
                        'task_id': args.task,
                        'tier': 'lite',
                        'groups': {'Q1': 'g1'},
                        'expected_agent_runtime': runtime,
                    },
                )


if __name__ == '__main__':
    unittest.main()
