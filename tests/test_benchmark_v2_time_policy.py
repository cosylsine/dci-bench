import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_benchmark_v2 as runner
from tests.test_benchmark_v2_results import row


class BenchmarkTimePolicyTest(unittest.TestCase):
    def test_execution_records_hardware_and_time_without_default_deadline(self):
        for hours in (None, '48'):
            with self.subTest(hours=hours), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / 'output'
                backend = {'backend': {'context_length': 65536, 'kind': 'vllm'}}
                backend_path = root / 'backend.json'
                backend_path.write_text(json.dumps(backend))
                job = {'task_id': 'scientific_en', 'tier': 'lite', 'query_ids': ['1']}
                plan = {'mode': 'local', 'model_key': 'fixture', 'jobs': [job], 'execution_count': 1,
                        'expected_tasks': ['scientific_en'], 'source_fingerprint': 'fixture-source',
                        'split': 'smoke', 'selection_sha256': 'selection', 'backend_sha256': 'backend',
                        'concurrency': 4, 'token_budget': 64000, 'complete_scope': False}
                run = output / 'evaluation/fixture/scientific_en/runs/smoke-lite'
                hardware = {'runner_host': {'cpu_models': ['fixture CPU']}, 'model_server': {'gpus': None}}

                def execute(command, *, env, log, timeout):
                    if hours is None:
                        self.assertIsNone(timeout)
                    else:
                        self.assertGreater(timeout, 47 * 3600)
                        self.assertLessEqual(timeout, 48 * 3600)
                    run.mkdir(parents=True)
                    (run / '_COMPLETE').write_text('{}')
                    return 0

                with patch.object(runner, 'build_plan', return_value=plan), \
                     patch('dci_bench.backends.manifest.load_backend_manifest', return_value=backend), \
                     patch.object(runner, 'source_fingerprint', return_value='fixture-source'), \
                     patch('scripts.inspect_local_runtime.prepare'), \
                     patch('shutil.which', return_value=None), \
                     patch.object(runner, 'command_for', return_value=['fixture']), \
                     patch.object(runner, 'run_child', side_effect=execute), \
                     patch.object(runner, 'completed_samples', return_value=[row()]), \
                     patch.object(runner, 'execution_hardware', return_value=hardware):
                    args = ['--suite', str(root), '--execute', '--backend-manifest', str(backend_path),
                            '--output', str(output)]
                    if hours is not None:
                        args += ['--wall-time-limit-hours', hours]
                    self.assertEqual(runner.main(args), 0)
                summary = json.loads((output / 'summary.json').read_text())
                self.assertEqual(summary['hardware_sessions'], [hardware])
                self.assertEqual(summary['task_executions'][0]['hardware_session'], 0)
                self.assertGreaterEqual(summary['task_executions'][0]['elapsed_seconds'], 0)
                self.assertEqual(summary['samples'][0]['wall_seconds'], 120)
                self.assertEqual(summary['wall_time_limit_hours'], float(hours) if hours else None)

    def test_invalid_operational_limit_rejected(self):
        for value in ('0', '-1', 'nan', 'inf'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite and positive'):
                runner.main(['--suite', '/unused', '--wall-time-limit-hours', value])
