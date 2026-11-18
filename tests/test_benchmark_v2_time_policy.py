from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_benchmark_v2 as runner
from tests.test_benchmark_v2_results import row


class BenchmarkTimePolicyTest(unittest.TestCase):
    def test_execution_records_hardware_and_time_without_default_deadline(self):
        for split, hours in (('smoke', None), ('smoke', '48'), ('test', None), ('test', '48')):
            with self.subTest(split=split, hours=hours), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / 'output'
                backend = {'backend': {'context_length': 65536, 'kind': 'vllm', 'base_url': 'http://127.0.0.1:29099/v1'},
                           'model': {'served_model_name': 'fixture'}}
                backend_path = root / 'backend.json'
                backend_path.write_text(json.dumps(backend))
                job = {'task_id': 'scientific_en', 'tier': 'lite', 'query_ids': ['1']}
                plan = {'mode': 'local', 'model_key': 'fixture', 'jobs': [job], 'execution_count': 1,
                        'expected_tasks': ['scientific_en'], 'source_fingerprint': 'fixture-source',
                        'split': split, 'selection_sha256': 'selection', 'backend_sha256': 'backend',
                        'concurrency': 4, 'token_budget': 64000, 'complete_scope': False}
                run = output / f'evaluation/fixture/scientific_en/runs/{split}-lite'
                hardware = {'runner_host': {'cpu_models': ['fixture CPU']}, 'model_server': {'gpus': None}}

                def execute(command, *, env, log, timeout):
                    self.assertEqual(env['DCI_TOKEN_COUNT_BASE_URL'], backend['backend']['base_url'])
                    self.assertEqual(env['DCI_TOKEN_COUNT_MODEL'], 'fixture')
                    self.assertEqual(env['DCI_TOKEN_COUNT_PYTHON'], runner.sys.executable)
                    if hours is None:
                        self.assertIsNone(timeout)
                    else:
                        self.assertGreater(timeout, 47 * 3600)
                        self.assertLessEqual(timeout, 48 * 3600)
                    run.mkdir(parents=True)
                    (run / '_COMPLETE').write_text('{}')
                    return 0

                with patch.object(runner, 'build_plan', side_effect=lambda *a, **kw: deepcopy(plan)), \
                     patch('dci_bench.backends.manifest.load_backend_manifest', return_value=backend), \
                     patch.object(runner, 'source_fingerprint', return_value='fixture-source'), \
                     patch('scripts.inspect_local_runtime.prepare'), \
                     patch('shutil.which', return_value=None), \
                     patch.object(runner, 'command_for', return_value=['fixture']), \
                     patch.object(runner, 'run_child', side_effect=execute), \
                     patch.object(runner, 'completed_samples', return_value=[row()]), \
                     patch.object(runner, 'execution_hardware', return_value=hardware):
                    args = ['--suite', str(root), '--execute', '--backend-manifest', str(backend_path),
                            '--output', str(output), '--split', split]
                    if split == 'test':
                        args += ['--fixed-budget-reason', 'User requested 64k']
                    if hours is not None:
                        args += ['--wall-time-limit-hours', hours]
                    self.assertEqual(runner.main(args), 0)
                    if split == 'test':
                        self.assertEqual(runner.main([*args, '--resume']), 0)
                        changed_args = [a if a != 'User requested 64k' else 'Different reason' for a in args]
                        with self.assertRaisesRegex(ValueError, 'fingerprint/configuration differs'):
                            runner.main([*changed_args, '--resume'])
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual(summary['hardware_sessions'], [hardware])
                self.assertEqual(summary['task_executions'][0]['hardware_session'], 0)
                self.assertGreaterEqual(summary['task_executions'][0]['elapsed_seconds'], 0)
                self.assertEqual(summary['samples'][0]['wall_seconds'], 120)
                self.assertEqual(summary['wall_time_limit_hours'], float(hours) if hours else None)
                if split == 'test':
                    policy = summary['budget_policy']
                    self.assertEqual(policy['mode'], 'user-fixed')
                    self.assertEqual(policy['token_budget'], 64000)
                    self.assertFalse(policy['development_selection_applied'])
                    self.assertEqual(json.loads((output/'launch.json').read_text())['budget_policy'], policy)

    def test_invalid_operational_limit_rejected(self):
        for value in ('0', '-1', 'nan', 'inf'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite and positive'):
                runner.main(['--suite', '/unused', '--wall-time-limit-hours', value])
