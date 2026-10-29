"""CPU checks for pilot scope, resource guards, and unbiased result aggregation."""
from copy import deepcopy
from contextlib import redirect_stdout
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('expansion_pilot', SCRIPTS / 'run_selective_expansion_pilot.py')
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class ExpansionPilotTests(unittest.TestCase):
    def setUp(self):
        self.tasks = sorted(P.TASKS)
        self.plan = {'task_order': self.tasks}
        self.splits = {t: {'pilot': [f'original/{i}' for i in range(10)], 'smoke': ['original/0']}
                       for t in self.tasks}

    def available_port_schedule(self):
        for _ in range(50):
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                base = probe.getsockname()[1]
            try:
                work = P.schedule_bridge_ports(P.batches(self.plan, self.splits), base)
                P.check_ports(base - 1, work)
            except ValueError:
                continue
            return base, work
        self.fail('Could not find a free contiguous port range for the CPU regression')

    def test_smoke_and_tail_are_70_unique_queries_and_original_ids_survive(self):
        work = P.batches(self.plan, self.splits)
        self.assertEqual(len(work), 14)
        self.assertTrue(all(b['stage'] == 'smoke' and len(b['query_ids']) == 1 for b in work[:7]))
        self.assertTrue(all(b['stage'] == 'pilot_tail' and len(b['query_ids']) == 9 for b in work[7:]))
        pairs = [(b['task'], q) for b in work for q in b['query_ids']]
        self.assertEqual(len(pairs), 70)
        self.assertEqual(len(set(pairs)), 70)
        for task in self.tasks:
            self.assertEqual([q for t, q in pairs if t == task], self.splits[task]['pilot'])
        for b in work:
            self.assertEqual(b['max_concurrency'], 1 if b['stage'] == 'smoke' else 8)

    def test_wrong_task_duplicate_and_changed_smoke_are_rejected(self):
        with self.assertRaises(ValueError):
            P.batches({'task_order': [*self.tasks[:-1], 'LLMAILAStatutes']}, self.splits)
        splits = deepcopy(self.splits)
        splits[self.tasks[0]]['pilot'][-1] = 'original/0'
        with self.assertRaises(ValueError):
            P.batches(self.plan, splits)
        splits = deepcopy(self.splits)
        splits[self.tasks[0]]['smoke'] = ['original/1']
        with self.assertRaises(ValueError):
            P.batches(self.plan, splits)

    def test_configured_capacity_controls_tail_ports_and_serving_together(self):
        for capacity in [1, 4, 8]:
            work = P.schedule_bridge_ports(P.batches(self.plan, self.splits, capacity), 28131)
            self.assertTrue(all(b['max_concurrency'] == capacity for b in work[7:]))
            self.assertTrue(all(b['bridge_port_count'] == capacity for b in work))
            self.assertEqual(work[-1]['bridge_port'] + capacity - 1, 28131 + 14 * capacity - 1)
            env = P.launch_environment(Path('/runtime'), {'model': {'source_path': '/model'}},
                                       concurrency=capacity)
            self.assertEqual(env['VLLM_MAX_NUM_SEQS'], str(capacity))

    def test_all_fourteen_batch_commands_have_disjoint_port_blocks(self):
        original = P.batches(self.plan, self.splits)
        before = deepcopy(original)
        work = P.schedule_bridge_ports(original, 28131)
        self.assertEqual(original, before)
        reserved = []
        for index, batch in enumerate(work):
            self.assertEqual(batch['bridge_port'], 28131 + index * 8)
            reserved.extend(range(batch['bridge_port'], batch['bridge_port'] + 8))
            argv = P.command(dict(batch, inspect_tools_directory='/private/tools'),
                             {'suite': '/suite'}, Path('/runtime'), Path('/output'))
            self.assertEqual(int(argv[argv.index('--bridge-port') + 1]), batch['bridge_port'])
        self.assertEqual(reserved, list(range(28131, 28243)))
        self.assertEqual(len(set(reserved)), 112)
        for base in [1023, 65425]:
            with self.subTest(base=base), self.assertRaises(ValueError):
                P.schedule_bridge_ports(original, base)
        self.assertEqual(P.schedule_bridge_ports(original, 65424)[-1]['bridge_port'] + 7, 65535)
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            P.check_ports(28186, work)

    def test_port_preflight_checks_the_last_reserved_port_even_for_serial_batch(self):
        self.plan['task_order'] = [t for t in self.tasks if t != 'enterprise_rag'] + ['enterprise_rag']
        base, work = self.available_port_schedule()
        last = work[-1]['bridge_port'] + 7
        self.assertEqual(work[-1]['max_concurrency'], 8)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', last))
            with self.assertRaisesRegex(ValueError, f'Pilot port {last} is unavailable'):
                P.check_ports(base - 1, work)
        # Failed checks release all the sockets they opened.
        P.check_ports(base - 1, work)

    def test_batch_transitions_pass_frozen_preflight_while_old_ports_remain_in_time_wait(self):
        runtime = P.expansion.code_path(P.expansion.read(P.PLAN / 'plan.json')['execution_runtime'])
        frozen = runtime / 'dci_bench/results/phase4_runner.py'
        # Execute the actual frozen guard without importing a second package tree.
        function = next(node for node in ast.parse(frozen.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'preflight_bridge_ports')
        namespace = {'socket': socket}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(frozen), 'exec'), namespace)
        preflight_bridge_ports = namespace['preflight_bridge_ports']
        _, work = self.available_port_schedule()
        for batch in work:
            port = batch['bridge_port']
            preflight_bridge_ports(port, batch['max_concurrency'])
            with socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.settimeout(2)
                listener.bind(('127.0.0.1', port))
                listener.listen()
                with socket.create_connection(('127.0.0.1', port), timeout=2) as client:
                    connection, _ = listener.accept()
                    # The server closes first, leaving its port in TIME_WAIT.
                    connection.close()
                    self.assertEqual(client.recv(1), b'')
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                preflight_bridge_ports(port, 1)
            # The next iteration immediately uses a different block, without sleeps.

    def test_output_is_new_and_cannot_escape_fixed_model_root(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(P, 'RESULTS', Path(tmp)):
            for name in ['../escape', '/absolute', 'a/b', 'two words', '.hidden']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    P.output_path(name)
            path = P.output_path('fresh-1')
            path.mkdir()
            with self.assertRaises(FileExistsError):
                P.output_path('fresh-1')

    def test_environment_accepts_mount_override_but_pins_protocol_and_runtime(self):
        expected = {'model': {'source_path': '/frozen/model'}}
        with patch.dict(os.environ, {'MODEL_PATH': '/wrong', 'PYTHONHOME': '/wrong', 'NODE_OPTIONS': '--wrong',
                                    'DCI_EVALUATION_TRACK': 'code', 'TENSOR_PARALLEL_SIZE': '1',
                                    'CUDA_VISIBLE_DEVICES': '4,5,6,7'}):
            env = P.launch_environment(Path('/frozen/runtime'), expected)
        self.assertEqual(env['MODEL_PATH'], '/wrong')
        self.assertEqual(env['PYTHONPATH'], '/frozen/runtime')
        self.assertEqual(env['TENSOR_PARALLEL_SIZE'], '4')
        self.assertEqual(env['DCI_EVALUATION_TRACK'], 'document')
        self.assertEqual(env['DCI_PROTOCOL_CONTRACT_SHA256'], P.expansion.CONTRACT)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '4,5,6,7')
        self.assertNotIn('NODE_OPTIONS', env)
        self.assertNotIn('PYTHONHOME', env)
        with patch.dict(os.environ, {'MODEL_PATH': '/environment/model'}):
            self.assertEqual(P.launch_environment(Path('/runtime'), expected, '/cli/model')['MODEL_PATH'],
                             '/cli/model')

    def test_model_location_requires_matching_config_and_complete_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alias = root / 'mounted checkpoint'
            alias.mkdir()
            (alias / 'config.json').write_text('{"model_type":"fixture"}')
            (alias / 'model.safetensors').write_bytes(b'fixture')
            expected = {'model': {'source_path': str(root / 'missing'),
                'config_sha256': P.expansion.sha(alias / 'config.json'),
                'weights': {'files': [{'name': 'model.safetensors', 'size_bytes': 7}]}}}
            location = P.model_location(expected, alias)
            self.assertEqual(location['selected_path'], str(alias))
            self.assertEqual([c['status'] for c in location['checks']],
                             ['config_and_weight_sizes_match'])
            self.assertFalse(location['weights_identity_verified'])
            # Explicitly requested mounts never silently fall back to another copy.
            missing = P.model_location(expected, root / 'typo')
            self.assertIsNone(missing['selected_path'])
            self.assertEqual(len(missing['checks']), 1)
            self.assertEqual(missing['checks'][0]['status'], 'missing_config')
            (alias / 'model.safetensors').write_bytes(b'incomplete')
            wrong = P.model_location(expected, alias)
            self.assertIsNone(wrong['selected_path'])
            self.assertEqual(wrong['checks'][-1]['status'], 'weight_names_or_sizes_mismatch')
            (alias / 'config.json').write_text('{"model_type":"other"}')
            wrong = P.model_location(expected, alias)
            self.assertEqual(wrong['checks'][-1]['status'], 'config_mismatch')

    def test_relocation_preserves_frozen_checkpoint_and_backend_identity_checks(self):
        frozen_scripts = P.expansion.SOURCE_RUNTIME / 'scripts'
        spec = importlib.util.spec_from_file_location('frozen_confirmation',
                                                     frozen_scripts / 'prepare_harness_confirmation.py')
        confirmation = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', [str(frozen_scripts), *sys.path]):
            spec.loader.exec_module(confirmation)
        expected = P.expansion.read(P.expansion.SOURCE / 'expected-backend.json')
        actual = deepcopy(expected)
        actual['model']['source_path'] = '/new/mount/checkpoint'
        before = deepcopy(actual)
        P.check_backend_identity(confirmation, expected, actual)
        self.assertEqual(actual, before)
        for section, field, value in [('model', 'config_sha256', '0' * 64),
                                      ('model', 'model_key', 'another-model'),
                                      ('model', 'revision', 'another-revision'),
                                      ('backend', 'dtype', 'float16'),
                                      ('backend', 'tensor_parallel_size', 1)]:
            bad = deepcopy(actual)
            bad[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                P.check_backend_identity(confirmation, expected, bad)
        bad = deepcopy(actual)
        bad['model']['weights']['files'][0]['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            P.check_backend_identity(confirmation, expected, bad)

    def test_actual_four_h100s_and_pinned_serving_software_required(self):
        info = dict(packages={'vllm': '0.17.0'}, torch_cuda='12.8', python={'version': '3.12.14'},
                    cuda={'kernel_probe': 'passed-on-every-visible-device',
                          'gpus': [{'name': 'NVIDIA H100 PCIe', 'capability': [9, 0],
                                    'total_memory_bytes': 80 * 1024**3} for _ in range(4)]})
        expected = {'backend': {'runtime': info}}
        P.validate_runtime(info, expected)
        changed = deepcopy(info)
        changed['cuda']['gpus'].pop()
        with self.assertRaises(ValueError):
            P.validate_runtime(changed, expected)
        changed = deepcopy(info)
        changed['packages']['vllm'] = 'changed'
        with self.assertRaises(ValueError):
            P.validate_runtime(changed, expected)

    def test_only_explicit_backend_capacity_change_is_allowed(self):
        expected = P.expansion.read(P.expansion.SOURCE / 'expected-backend.json')
        actual = deepcopy(expected)
        actual['backend']['max_num_seqs'] = 8
        actual['backend']['runtime']['launch_environment']['VLLM_MAX_NUM_SEQS'] = '8'
        seen = []
        confirmation = SimpleNamespace(check_backend=lambda _, value: seen.append(value))
        P.check_backend_identity(confirmation, expected, actual, 8)
        self.assertEqual(seen, [expected])
        self.assertEqual(actual['backend']['max_num_seqs'], 8)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            P.check_backend_identity(confirmation, expected, actual, 4)
        actual['backend']['runtime']['launch_environment']['VLLM_EXTRA_ARGS'] = '--quantization fp8'
        with self.assertRaisesRegex(ValueError, 'launch environment'):
            P.check_backend_identity(confirmation, expected, actual, 8)

    def test_smoke_gates_validity_and_real_calls_but_never_retrieval_score(self):
        records = [{'sample': {'task': t, 'valid_output': True, 'metrics': {'recall_at_10': 0},
                    'execution': {'tool_calls': 1, 'model_calls': 1}}} for t in self.tasks]
        self.assertTrue(P.smoke_passed(records, self.tasks))
        self.assertFalse(P.smoke_passed(records[:-1], self.tasks))
        bad = deepcopy(records)
        bad[0]['sample']['valid_output'] = False
        self.assertFalse(P.smoke_passed(bad, self.tasks))
        bad = deepcopy(records)
        bad[0]['sample']['execution']['model_calls'] = 0
        self.assertFalse(P.smoke_passed(bad, self.tasks))

    def test_summary_counts_invalid_as_zero_and_reports_missing_usage(self):
        def record(qid, valid, metric, usage, wall):
            return {'run': 'source/' + qid, 'precision_at_10': metric if valid else 0,
                'sample': {'task': 'task', 'query_id': qid, 'valid_output': valid,
                    'metrics': {'f1_at_10': metric, 'recall_at_10': metric},
                    'failure': None if valid else {'kind': 'timeout'},
                    'usage': {'available': usage is not None, 'total_tokens': usage},
                    'timing': {'wall_time_seconds': wall},
                    'execution': {'repair_attempts': 1, 'repair_reasons': ['invalid JSON']}}}
        result = P.summarize([record('a', True, 1., 100, 10), record('b', False, 1., None, None)], ['task'])
        task = result['tasks']['task']
        self.assertEqual(task['all_sample']['f1_at_10'], .5)
        self.assertEqual(task['valid_only']['f1_at_10'], 1.)
        self.assertEqual(task['all_sample']['precision_at_10'], .5)
        self.assertEqual(task['failure_counts'], {'timeout': 1})
        self.assertEqual(task['usage_available_samples'], 1)
        self.assertEqual(task['timing_available_samples'], 1)
        self.assertEqual(task['mean_wall_seconds'], 10)
        absent = P.summarize([record('a', False, 0., None, None)], ['task'])['tasks']['task']
        self.assertIsNone(absent['total_tokens'])
        self.assertIsNone(absent['mean_wall_seconds'])

    def test_tail_command_uses_frozen_expansion_runner_and_separate_outputs(self):
        tail = P.schedule_bridge_ports(P.batches(self.plan, self.splits), 28131)[7]
        argv = P.command(dict(tail, inspect_tools_directory='/private/tools'),
                         {'suite': '/frozen/suite'}, Path('/frozen/runtime'), Path('/new/output'))
        self.assertIn('/frozen/runtime/scripts/run_expansion_task.py', argv)
        self.assertEqual(argv[argv.index('--tools-directory') + 1], '/private/tools')
        self.assertIn(str(P.ROOT / 'releases/run-launchers/new/output/inspect_local_runtime.py'), argv)
        self.assertEqual(argv[argv.index('--query-ids') + 1:argv.index('--model-key')], tail['query_ids'])
        self.assertEqual(argv[argv.index('--run-id') + 1], 'pilot_tail')
        self.assertEqual(argv[argv.index('--results-root') + 1], '/new/output/evaluation')
        self.assertIn('--require-valid-output', argv)

    def test_preflight_only_never_probes_cuda_or_starts_a_server(self):
        plan = {**self.plan, 'execution_runtime': '/frozen/runtime', 'dataset_index': '/frozen/index',
                'selection_seed': 'fixture', 'candidate_policy': 'all candidates',
                'task_inventory': [{'task': t, 'suite': '/frozen/suite', 'metadata': '/frozen/metadata/' + t,
                                    'workspace': '/frozen/workspaces/' + t} for t in self.tasks]}
        expected = {'model': {'source_path': '/deliberately-missing-model'}}
        inputs = {str(P.PLAN / 'plan.json'): plan, str(P.PLAN / 'query-splits.json'): self.splits,
                  str(P.expansion.SOURCE / 'expected-backend.json'): expected}
        identity = {'python': 'fixture'}
        full = SimpleNamespace(verify_bundle=lambda _: {'bench_identity': identity, 'node_version': 'fixture'},
                               confirmation=SimpleNamespace(python_identity=lambda _: identity))
        with tempfile.TemporaryDirectory() as tmp, patch.object(P, 'RESULTS', Path(tmp)), \
             patch.object(P.sys, 'argv', ['driver', '--preflight-only', '--run-id', 'cpu']), \
             patch.object(P.sys, 'path', list(sys.path)), patch.dict(os.environ, {'DRY_RUN': '0'}), \
             patch.object(P.expansion, 'verify'), \
             patch.object(P.expansion, 'read', side_effect=lambda p: inputs[str(p)]), \
             patch.object(P.importlib, 'import_module', return_value=full), \
             patch.object(P.subprocess, 'check_output', return_value='fixture\n'), \
             patch.object(P.subprocess, 'Popen', side_effect=AssertionError('Unexpected subprocess/GPU')), \
             patch.object(P.os, 'access', return_value=True), patch.object(P, 'step') as step, \
             patch.object(P.staging, 'stage', return_value=({r['task']: r for r in plan['task_inventory']}, {})), \
             patch.object(P.staging, 'local_filesystem'), patch.object(P.inspect_runtime, 'prepare', return_value={}), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(P.main(), 0)
            names = [call.args[0] for call in step.call_args_list]
            self.assertEqual(sum(n.startswith('corpus-') for n in names), 7)
            self.assertEqual(sum(n.startswith('sandbox-') for n in names), 7)
            self.assertIn('enterprise-startup', names)
            self.assertIn('inspect-parallel', names)
            self.assertLess(names.index('inspect-parallel'), names.index('snapshot'))
            self.assertNotIn('runtime', names)
            self.assertNotIn('backend', names)
            self.assertNotIn('warmup', names)
            self.assertTrue((Path(tmp) / 'cpu/preflight.json').exists())
            self.assertFalse((Path(tmp) / 'cpu/_COMPLETE').exists())

    def test_port_conflict_stops_normal_run_before_weight_hashing_or_model_calls(self):
        plan = {**self.plan, 'execution_runtime': '/frozen/runtime', 'dataset_index': '/frozen/index',
                'selection_seed': 'fixture', 'candidate_policy': 'all candidates',
                'task_inventory': [{'task': t} for t in self.tasks]}
        expected = {'model': {'source_path': '/frozen/model'}}
        inputs = {str(P.PLAN / 'plan.json'): plan, str(P.PLAN / 'query-splits.json'): self.splits,
                  str(P.expansion.SOURCE / 'expected-backend.json'): expected}
        location = dict(requested_path='/mount/model', selected_path='/mount/model')
        with tempfile.TemporaryDirectory() as tmp, patch.object(P, 'RESULTS', Path(tmp)), \
             patch.object(P.sys, 'argv', ['driver', '--run-id', 'occupied-port']), \
             patch.dict(os.environ, {'DRY_RUN': '0'}), patch.object(P.expansion, 'verify'), \
             patch.object(P.expansion, 'read', side_effect=lambda p: inputs[str(p)]), \
             patch.object(P, 'model_location', return_value=location), \
             patch.object(P, 'check_ports', side_effect=ValueError('Pilot port 28186 is unavailable')) as check, \
             patch.object(P.importlib, 'import_module', side_effect=AssertionError('Unexpected weight verification')), \
             patch.object(P, 'step', side_effect=AssertionError('Unexpected subprocess/GPU')):
            with self.assertRaisesRegex(ValueError, '28186 is unavailable'):
                P.main()
            self.assertEqual(len(check.call_args.args[1]), 14)
            failure = json.loads((Path(tmp) / 'occupied-port/_FAILED.json').read_text())
            self.assertFalse(failure['gpu_started'])
            self.assertFalse(failure['model_server_started'])
            self.assertEqual(failure['completed'], 0)

    def test_inspect_failure_stops_before_weights_corpus_or_cuda(self):
        from dci_bench.backends import manifest

        plan = {**self.plan, 'execution_runtime': '/frozen/runtime', 'dataset_index': '/frozen/index',
                'selection_seed': 'fixture', 'candidate_policy': 'all candidates',
                'task_inventory': [{'task': t} for t in self.tasks]}
        expected = {'model': {'source_path': '/frozen/model'}}
        inputs = {str(P.PLAN / 'plan.json'): plan, str(P.PLAN / 'query-splits.json'): self.splits,
                  str(P.expansion.SOURCE / 'expected-backend.json'): expected}
        identity = {'python': 'fixture'}
        full = SimpleNamespace(verify_bundle=lambda _: {'bench_identity': identity, 'node_version': 'fixture'},
                               confirmation=SimpleNamespace(python_identity=lambda _: identity))
        location = dict(requested_path='/mount/model', selected_path='/mount/model')
        with tempfile.TemporaryDirectory() as tmp, patch.object(P, 'RESULTS', Path(tmp)), \
             patch.object(P.sys, 'argv', ['driver', '--run-id', 'inspect-failure']), \
             patch.object(P.sys, 'path', list(sys.path)), patch.dict(os.environ, {'DRY_RUN': '0'}), \
             patch.object(P.expansion, 'verify'), \
             patch.object(P.expansion, 'read', side_effect=lambda p: inputs[str(p)]), \
             patch.object(P, 'model_location', return_value=location), patch.object(P, 'check_ports'), \
             patch.object(P.importlib, 'import_module', return_value=full), \
             patch.object(P.subprocess, 'check_output', return_value='fixture\n'), \
             patch.object(P.subprocess, 'Popen', side_effect=AssertionError('Unexpected subprocess/GPU')), \
             patch.object(P.staging, 'local_filesystem'), \
             patch.object(P.inspect_runtime, 'prepare', return_value={}), \
             patch.object(P.staging, 'stage', side_effect=AssertionError('Unexpected corpus work')), \
             patch.object(manifest, 'build_vllm_manifest',
                   side_effect=AssertionError('Unexpected weight hashing')), \
             patch.object(P, 'step', side_effect=RuntimeError('parallel Inspect failed')) as step:
            with self.assertRaisesRegex(RuntimeError, 'parallel Inspect failed'):
                P.main()
            self.assertEqual([call.args[0] for call in step.call_args_list], ['inspect-parallel'])
            failure = json.loads((Path(tmp) / 'inspect-failure/_FAILED.json').read_text())
            self.assertFalse(failure['gpu_started'])
            self.assertFalse(failure['model_server_started'])
            self.assertEqual(failure['completed'], 0)
            self.assertFalse((Path(tmp) / 'inspect-failure/_COMPLETE').exists())

    def test_model_only_preflight_preserves_actual_mount_and_never_starts_gpu_or_scans_corpus(self):
        from dci_bench.backends import manifest

        plan = {**self.plan, 'execution_runtime': '/frozen/runtime', 'dataset_index': '/frozen/index',
                'selection_seed': 'fixture', 'candidate_policy': 'all candidates',
                'task_inventory': [{'task': t} for t in self.tasks]}
        expected = P.expansion.read(P.expansion.SOURCE / 'expected-backend.json')
        actual = deepcopy(expected)
        actual['model']['source_path'] = '/new mount/model'
        actual['backend']['max_num_seqs'] = 8
        actual['backend']['runtime']['launch_environment'].update(MODEL_PATH='/new mount/model', VLLM_MAX_NUM_SEQS='8')
        inputs = {str(P.PLAN / 'plan.json'): plan, str(P.PLAN / 'query-splits.json'): self.splits,
                  str(P.expansion.SOURCE / 'expected-backend.json'): expected}
        identity = {'python': 'fixture'}
        def check_backend(_, payload):
            self.assertEqual(payload, expected)
        full = SimpleNamespace(verify_bundle=lambda _: {'bench_identity': identity, 'node_version': 'fixture'},
            confirmation=SimpleNamespace(python_identity=lambda _: identity, check_backend=check_backend))
        location = dict(requested_path='/new mount/model', selected_path='/new mount/model',
                        weights_identity_verified=False, checks=[])
        with tempfile.TemporaryDirectory() as tmp, patch.object(P, 'RESULTS', Path(tmp)), \
             patch.object(P.sys, 'argv', ['driver', '--model-preflight-only', '--run-id', 'model-check',
                                         '--model-path', '/new mount/model']), \
             patch.object(P.sys, 'path', list(sys.path)), patch.dict(os.environ, {'DRY_RUN': '0'}), \
             patch.object(P.expansion, 'verify'), \
             patch.object(P.expansion, 'read', side_effect=lambda p: inputs[str(p)]), \
             patch.object(P.importlib, 'import_module', return_value=full), \
             patch.object(P.subprocess, 'check_output', return_value='fixture\n'), \
             patch.object(P.subprocess, 'Popen', side_effect=AssertionError('Unexpected subprocess/GPU')), \
             patch.object(P, 'step', side_effect=AssertionError('Unexpected corpus/GPU check')), \
             patch.object(P, 'model_location', return_value=location) as resolver, \
             patch.object(manifest, 'build_vllm_manifest', return_value=actual) as build, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(P.main(), 0)
            resolver.assert_called_once_with(expected, Path('/new mount/model'))
            self.assertEqual(build.call_args.kwargs['model_path'], Path('/new mount/model'))
            output = Path(tmp) / 'model-check'
            self.assertEqual(json.loads((output / 'model-preflight.json').read_text())['model']['source_path'],
                             '/new mount/model')
            self.assertTrue(json.loads((output / 'model-location.json').read_text())['weights_identity_verified'])
            self.assertFalse((output / '_COMPLETE').exists())

    def test_percentiles_use_available_observations(self):
        self.assertIsNone(P.percentile([], .5))
        self.assertEqual(P.percentile([10, 20], .5), 15)
        self.assertEqual(P.percentile([10, 20], .95), 19.5)


if __name__ == '__main__':
    unittest.main()
