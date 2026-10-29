"""Scope, isolation and acceptance checks for the 159-execution launcher."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location('pool_smoke', Path(__file__).resolve().parents[1] / 'scripts/run_pool_smoke.py')
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class PoolSmokeLauncherTests(unittest.TestCase):
    def setUp(self):
        P.STOPPING.clear()
        self.addCleanup(P.STOPPING.clear)

    def rows(self):
        return [dict(task=f'task_{i}', query_id=f'original/{i}', capacity=k,
                     profile='document-pool-v1-' + tier, suite=f'/data/task_{i}/pools/{k}')
                for i in range(53) for k, tier in P.CAPACITIES.items()]

    def test_driver_imports_its_runtime_without_relying_on_cwd_or_pythonpath(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / 'work'
            work.mkdir()
            foreign = Path(tmp) / 'foreign'
            foreign.mkdir()
            (foreign / 'dci_bench').mkdir()
            (foreign / 'dci_bench/__init__.py').write_text("raise RuntimeError('wrong runtime imported')\n")
            code = '''
import pathlib, runpy, sys
driver = pathlib.Path(sys.argv[1])
runpy.run_path(str(driver), run_name='driver_import_probe')
from dci_bench.backends.manifest import validate_backend_manifest
from dci_bench.results.phase4_runner import load_completed_sample
import dci_bench
assert pathlib.Path(dci_bench.__file__).resolve().is_relative_to(driver.parents[1])
'''
            for inherited_path in (None, str(foreign)):
                with self.subTest(pythonpath=inherited_path):
                    env = dict(os.environ)
                    env.pop('PYTHONPATH', None)
                    if inherited_path is not None:
                        env['PYTHONPATH'] = inherited_path
                    result = subprocess.run([sys.executable, '-B', '-c', code, str(Path(P.__file__).resolve())],
                                            cwd=work, env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_exact_159_disjoint_ports_and_paired_original_ids(self):
        work = P.schedule(self.rows())
        self.assertEqual(len(work), 159)
        self.assertEqual({r['bridge_port'] for r in work}, set(range(29131, 29290)))
        self.assertEqual(len({(r['task'], r['query_id']) for r in work}), 53)

    def test_reject_missing_duplicate_and_nonpaired_rows(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            P.schedule(rows[:-1])
        rows[-1] = dict(rows[-2])
        with self.assertRaises(ValueError):
            P.schedule(rows)
        rows = self.rows()
        rows[-1]['query_id'] = 'wrong-query'
        with self.assertRaises(ValueError):
            P.schedule(rows)

    def test_reject_profile_mismatch_and_port_overflow(self):
        rows = self.rows()
        rows[0]['profile'] = 'budgeted-selective'
        with self.assertRaises(ValueError):
            P.schedule(rows)
        with self.assertRaises(ValueError):
            P.schedule(self.rows(), 65500)

    def test_all_commands_use_pool_entry_and_separate_run_directory(self):
        for row in P.schedule(self.rows()):
            command = P.task_command(row, Path('/sealed'), Path('/out'), Path('/tools'))
            self.assertIn('/sealed/scripts/run_document_pool.py', command)
            self.assertEqual(command[command.index('--query-ids') + 1], row['query_id'])
            self.assertEqual(command[command.index('--run-id') + 1], f"smoke-{row['capacity']}")
            self.assertEqual(command[command.index('--max-concurrency') + 1], '1')
            self.assertIn('--require-valid-output', command)

    def test_data_preflight_has_no_backend_or_model_calls(self):
        command = P.task_command(P.schedule(self.rows())[0], Path('/sealed'), Path('/out'), preflight=True)
        self.assertIn('--preflight-only', command)
        self.assertNotIn('--backend-manifest', command)
        self.assertNotIn('--all-queries', command)

    def test_complete_requires_all_real_model_and_tool_executions(self):
        row = dict(valid=True, real_model_and_tools=True, failure=None)
        self.assertEqual(P.summarize([row] * 159)['status'], 'passed')
        self.assertEqual(P.summarize([row] * 158)['status'], 'not_passed')
        bad = dict(row, real_model_and_tools=False)
        self.assertEqual(P.summarize([row] * 158 + [bad])['status'], 'not_passed')
        bad = dict(row, valid=False, failure={'kind': 'timeout'})
        self.assertEqual(P.summarize([row] * 158 + [bad])['failure_counts'], {'timeout': 1})

    def test_inventory_detects_mutation_and_rejects_external_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'file').write_text('before')
            old = P.tree_inventory(root)
            (root / 'file').write_text('after')
            self.assertNotEqual(old, P.tree_inventory(root))
            (root / 'escape').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):
                P.tree_inventory(root)

    def test_infrastructure_failure_does_not_start_the_remaining_queue(self):
        started = []
        def run(row):
            started.append(row)
            raise RuntimeError('infrastructure failed')
        with self.assertRaisesRegex(RuntimeError, 'infrastructure'):
            list(P.parallel_checked(run, range(159), 1))
        self.assertEqual(started, [0])

    def test_listener_must_belong_to_new_server_process_group(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        code = 'import socket,time; s=socket.socket(); s.bind(("127.0.0.1",'+str(port)+')); s.listen(); time.sleep(30)'
        server = subprocess.Popen([sys.executable, '-c', code], start_new_session=True)
        try:
            deadline = time.monotonic() + 5
            while not P.server_owns_port(server, port) and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(P.server_owns_port(server, port))
            self.assertFalse(P.server_owns_port(SimpleNamespace(pid=1, poll=lambda: None), port))
        finally:
            P.terminate(server)

    def test_child_registered_during_shutdown_is_terminated(self):
        P.STOPPING.set()
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        with self.assertRaisesRegex(RuntimeError, 'stopping'):
            P.track(child)
        self.assertIsNotNone(child.poll())
        self.assertNotIn(child, P.CHILDREN)


if __name__ == '__main__':
    unittest.main()
