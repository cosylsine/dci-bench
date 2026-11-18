import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_benchmark20_resume as checker


class Benchmark20ResumeTests(unittest.TestCase):
    def test_restart_before_test_needs_no_prior_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(checker.main([
                '--run', directory, '--suite', directory,
                '--token-budget', '64000', '--concurrency', '16',
            ]), 0)

    def test_original_launch_and_backend_are_preserved_and_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'test').mkdir()
            (root/'config.json').write_text('{}')
            (root/'backend.json').write_text('{"original":true}')
            backend = {
                'kind': 'vllm', 'context_length': 65536, 'tensor_parallel_size': 4,
                'dtype': 'bfloat16', 'tool_call_parser': 'qwen3_coder',
                'reasoning_parser': 'qwen3', 'language_model_only': True,
                'max_num_seqs': 16, 'gpu_memory_utilization': .8,
                'base_url': 'http://127.0.0.1:29099/v1', 'version': '0.17.0',
            }
            model = {'model_key': 'Qwen3.8-27B', 'served_model_name': 'Qwen3.8-27B',
                     'source_path': str(root), 'config_sha256': checker.runner.sha256_file(root/'config.json')}
            base_plan = {'source_fingerprint': 'source', 'agent_runtime': {'mode': 'frozen-rootfs'},
                         'backend_sha256': checker.runner.sha256_file(root/'backend.json'),
                         'token_budget': 64000, 'concurrency': 16, 'selection_sha256': 'data'}
            launch = dict(base_plan, wall_time_limit_hours=None, price_quote=None,
                          budget_cny=1000, preflight_only=False, budget_policy={
                              'mode': 'user-fixed', 'token_budget': 64000,
                              'reason': 'User-requested fixed cumulative budget: 64000 tokens',
                              'development_selection_applied': False, 'calibration_projection_applied': False,
                          })
            launch['fingerprint'] = checker.runner.fingerprint(launch)
            (root/'test/launch.json').write_text(json.dumps(launch))
            original = {p: p.read_bytes() for p in (root/'backend.json', root/'test/launch.json')}
            args = ['--run', directory, '--suite', directory, '--token-budget', '64000', '--concurrency', '16']
            with patch.dict(os.environ, MODEL_PATH=directory, VLLM_PORT='29099', VLLM_GPU_MEMORY_UTILIZATION='.8'), \
                 patch.object(checker, 'load_backend_manifest', return_value={'backend': backend, 'model': model}), \
                 patch.object(checker.runner, 'verify_runtime_config', return_value='verified-runtime'), \
                 patch.object(checker.runner, 'build_plan', side_effect=lambda *a, **kw: dict(base_plan)) as build:
                self.assertEqual(checker.main(args), 0)
                self.assertEqual(build.call_args.kwargs['agent_runtime'], 'verified-runtime')
                for field in ('source_fingerprint', 'agent_runtime', 'backend_sha256', 'selection_sha256'):
                    with self.subTest(field=field), patch.dict(base_plan, {field: 'changed'}):
                        with self.assertRaisesRegex(ValueError, 'fingerprint/configuration differs'):
                            checker.main(args)
                with patch.dict(backend, max_num_seqs=8):
                    with self.assertRaisesRegex(ValueError, 'max_num_seqs'):
                        checker.main(args)
                with self.assertRaisesRegex(ValueError, 'version'):
                    checker.main([*args, '--serving-version', 'different'])
                self.assertEqual(checker.main([*args, '--serving-version', '0.17.0']), 0)
            for path, content in original.items():
                self.assertEqual(path.read_bytes(), content)

    def test_missing_launch_or_backend_fails_without_repairing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'test').mkdir()
            args = ['--run', directory, '--suite', directory, '--token-budget', '64000', '--concurrency', '16']
            with self.assertRaisesRegex(ValueError, 'no launch.json'):
                checker.main(args)
            (root/'test/launch.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'original backend.json'):
                checker.main(args)
            self.assertFalse((root/'backend.json').exists())
