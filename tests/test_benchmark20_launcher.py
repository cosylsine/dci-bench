"""Run the real shell control flow with synthetic processes, without GPUs."""
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class Benchmark20LauncherTests(unittest.TestCase):
    def launch(self, *, mode=None, fail_stage="", memory="0.80", frozen=False,
               prepare_fail=False, ensure_fail=False, dry_run=False, warm=False, preflight_concurrency="4",
               full=False, concurrency="4", selected_budget=64000, inspect_timeout="600", fixed_budget=None,
               resume=False, named_run=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            bindir = root / "env/bin"
            bindir.mkdir(parents=True)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            text = (ROOT / "scripts/launch_benchmark20_qwen38_5090.sh").read_text()
            text = text.replace("BENCH_ENV=/mnt/afs/250010100/miniconda/envs/dci-bench", f"BENCH_ENV={root}/env")
            text = text.replace("SERVING_ENV=/mnt/afs/250010100/miniconda/envs/dci-vllm-qwen38", f"SERVING_ENV={root}/env")
            text = text.replace("MODEL_PATH=/mnt/afs2/202610/Qwen3.8-27B", f"MODEL_PATH={model}")
            launcher = scripts / "launch_benchmark20_qwen38_5090.sh"
            launcher.write_text(text)
            shutil.copyfile(ROOT / "scripts/serve_vllm_qwen38.sh", scripts / "serve_vllm_qwen38.sh")
            (scripts / "ensure_pool_smoke_tools.sh").write_text(
                '#!/bin/sh\nexit "${TEST_DEP_FAIL:-0}"\n')
            prepare = scripts / "prepare_frozen_runtime.sh"
            prepare.write_text("""#!/bin/sh
printf '{\"stage\":\"frozen_prepare\",\"config\":\"%s\",\"bundle\":\"%s\"}\\n' \\
  \"$DCI_RUNTIME_CONFIG\" \"${DCI_RUNTIME_BUNDLE:-}\" >> \"$TEST_TRACE\"
exit \"${TEST_PREPARE_FAIL:-0}\"
""")
            prepare.chmod(0o755)
            fake = bindir / "python"
            fake.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys, time
args = sys.argv[1:]
if args[0] == '-':
    source = sys.stdin.read()
    if 'scripts.inspect_local_runtime' not in source:
        sys.argv = ['-', *args[1:]]
        exec(compile(source, '<launcher-inline>', 'exec'))
    sys.exit(0)
if args[0] == '-c':
    print('0.17.0')
    sys.exit(0)
stage = pathlib.Path(args[0]).name
if stage == 'run_benchmark_v2.py':
    stage = 'execute' if '--execute' in args else 'data'
elif args[:2] == ['-m', 'vllm.entrypoints.openai.api_server']:
    stage = 'serve'
elif args[:2] == ['-m', 'dci_bench.backends.manifest']:
    stage = 'manifest'
with open(os.environ['TEST_TRACE'], 'a') as f:
    f.write(json.dumps({'stage': stage, 'args': args,
                        'worker': os.environ.get('VLLM_WORKER_CLS'),
                        'memory': os.environ.get('VLLM_GPU_MEMORY_UTILIZATION')}) + '\\n')
if stage == 'execute':
    output = pathlib.Path(args[args.index('--output')+1])
    output.mkdir(parents=True, exist_ok=True)
    launch = output / 'launch.json'
    if not launch.exists():
        launch.write_text('{"fixture":"original launch"}')
    if os.environ.get('TEST_FAIL_STAGE') == 'kill-execute':
        import signal
        (output / 'sealed-sample').write_text('preserve completed result')
        os.kill(os.getpid(), signal.SIGKILL)
if os.environ.get('TEST_FAIL_STAGE') == stage:
    sys.exit(23)
if stage == 'execute' and os.environ.get('TEST_FAIL_STAGE') == 'dev-' + args[args.index('--token-budget')+1]:
    sys.exit(23)
if stage == 'analyze_benchmark_v2.py':
    pathlib.Path(args[args.index('--output')+1]).write_text(json.dumps(
        {'selected_budget': json.loads(os.environ['TEST_SELECTED_BUDGET'])}))
if stage == 'harness_pilot_runtime.py':
    pathlib.Path(args[args.index('--output')+1]).write_text(json.dumps(
        {'cuda': {'gpus': [{'name': 'RTX 5090', 'capability': [12, 0]}] * 4}}))
if stage == 'manifest':
    pathlib.Path(args[args.index('--output')+1]).write_text('{"fixture":"original backend"}')
if stage == 'serve':
    while True:
        time.sleep(1)
''')
            fake.chmod(0o755)
            curl = bindir / "curl"
            curl.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys, time
for _ in range(100):
    if any(json.loads(line)['stage'] == 'serve' for line in pathlib.Path(os.environ['TEST_TRACE']).read_text().splitlines()):
        break
    time.sleep(.01)
pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_text(json.dumps({'data':[{'id':'Qwen3.8-27B'}]}))
''')
            curl.chmod(0o755)
            trace = root / "trace.jsonl"
            env = {**os.environ, "DRY_RUN": "1" if dry_run else "0", "TEST_TRACE": str(trace),
                   "TEST_FAIL_STAGE": fail_stage, "TEST_PREPARE_FAIL": "23" if prepare_fail else "0",
                   "TEST_DEP_FAIL": "19" if ensure_fail else "0",
                   "VLLM_GPU_MEMORY_UTILIZATION": memory, "PREFLIGHT_CONCURRENCY": preflight_concurrency,
                   "INSPECT_PREFLIGHT_TIMEOUT_SECONDS": inspect_timeout,
                   "CONCURRENCY": concurrency, "TEST_SELECTED_BUDGET": json.dumps(selected_budget)}
            env.pop('DCI_RUNTIME_CONFIG', None)
            env.pop('DCI_RUNTIME_BUNDLE', None)
            if frozen:
                config = root/'runtime.json'
                config.write_text('{}')
                env['DCI_RUNTIME_CONFIG'] = str(config)
                if not warm:
                    bundle = root/'rootfs.tar'
                    bundle.write_bytes(b'fixture bundle')
                    env['DCI_RUNTIME_BUNDLE'] = str(bundle)
            command = ["bash", str(launcher), *(['--full'] if full else []), *([mode] if mode else [])]
            if fixed_budget is not None:
                command += ['--token-budget', str(fixed_budget)]
            if named_run:
                command += ['--run-dir', str(root/'named-run')]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
            if resume:
                self.assertNotEqual(result.returncode, 0)
                run = root/'named-run' if named_run else next((root/'results/Qwen3.8-27B/benchmark-v2-full-runs').iterdir())
                original = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
                trace.write_text('')
                env['TEST_FAIL_STAGE'] = ''
                resume_command = ["bash", str(launcher), '--full', '--token-budget', str(fixed_budget),
                                  '--resume-run', str(run)]
                result = subprocess.run(resume_command, env=env, capture_output=True, text=True, timeout=30)
                for path, content in original.items():
                    self.assertEqual(path.read_bytes(), content, str(path))
            rows = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
            return result, rows

    def test_complete_launch_orders_kernel_probe_profile_worker_and_evaluation(self):
        result, rows = self.launch(memory="0.75")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([r["stage"] for r in rows], [
            "data", "check_inspect_parallel.py", "harness_pilot_runtime.py", "check_gdn_kernels.py",
            "manifest", "serve", "harness_pilot_warmup.py", "check_budget_tokenizer.py", "execute"])
        serve = next(r for r in rows if r["stage"] == "serve")
        args = serve["args"]
        for name, value in {"--worker-cls": "dci_bench.backends.vllm_gdn_worker.GDNWarmupWorker",
                            "--max-num-batched-tokens": "2048", "--max-model-len": "65536",
                            "--gpu-memory-utilization": "0.75", "--tensor-parallel-size": "4",
                            "--dtype": "bfloat16"}.items():
            self.assertEqual(args[args.index(name) + 1], value)
        manifest = next(r["args"] for r in rows if r["stage"] == "manifest")
        self.assertEqual(manifest[manifest.index("--gpu-memory-utilization") + 1], "0.75")
        warmup = next(r["args"] for r in rows if r["stage"] == "harness_pilot_warmup.py")
        self.assertEqual(warmup[warmup.index("--backend-manifest") + 1],
                         next(r["args"] for r in rows if r["stage"] == "manifest")[
                             manifest.index("--output") + 1])

    def test_serving_diagnostic_skips_data_and_benchmark_but_requires_warmup(self):
        result, rows = self.launch(mode="--serving-smoke-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([r["stage"] for r in rows], ["harness_pilot_runtime.py", "check_gdn_kernels.py",
            "manifest", "serve", "harness_pilot_warmup.py", "check_budget_tokenizer.py"])
        self.assertIn("benchmark not executed", result.stdout)

    def test_preflight_concurrency_is_forwarded_independently_of_model_concurrency(self):
        for workers in ('1', '4', '8'):
            with self.subTest(workers=workers):
                result, rows = self.launch(mode="--preflight-only", preflight_concurrency=workers)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                args = next(row['args'] for row in rows if row['stage'] == 'data')
                self.assertEqual(args[args.index('--preflight-concurrency') + 1], workers)
                self.assertEqual(args[args.index('--concurrency') + 1], '4')
                self.assertNotIn('serve', [row['stage'] for row in rows])

    def test_frozen_serving_diagnostic_skips_runtime_and_data_preparation(self):
        result, rows = self.launch(mode="--serving-smoke-only", frozen=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([r["stage"] for r in rows], ["harness_pilot_runtime.py", "check_gdn_kernels.py",
            "manifest", "serve", "harness_pilot_warmup.py", "check_budget_tokenizer.py"])

    def test_failed_kernel_probe_stops_before_manifest_or_model_load(self):
        result, rows = self.launch(mode="--serving-smoke-only", fail_stage="check_gdn_kernels.py")
        self.assertEqual(result.returncode, 23)
        self.assertEqual([r["stage"] for r in rows], ["harness_pilot_runtime.py", "check_gdn_kernels.py"])

    def test_failed_service_warmup_never_starts_benchmark(self):
        result, rows = self.launch(fail_stage="harness_pilot_warmup.py")
        self.assertEqual(result.returncode, 23)
        self.assertNotIn("execute", [r["stage"] for r in rows])

    def test_tokenizer_mismatch_stops_before_benchmark(self):
        result, rows = self.launch(fail_stage="check_budget_tokenizer.py")
        self.assertEqual(result.returncode, 23)
        self.assertNotIn("execute", [r["stage"] for r in rows])

    def test_cpu_only_mode_never_probes_or_loads_gpu(self):
        result, rows = self.launch(mode="--preflight-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([r["stage"] for r in rows], ["data", "check_inspect_parallel.py"])

    def test_inspect_timeout_reaches_probe_and_is_logged(self):
        for seconds in ('600', '900'):
            with self.subTest(seconds=seconds):
                result, rows = self.launch(mode='--preflight-only', inspect_timeout=seconds)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                args = next(row['args'] for row in rows if row['stage'] == 'check_inspect_parallel.py')
                self.assertEqual(args[args.index('--sample-timeout-seconds') + 1], seconds)
                self.assertIn(f'inspect_preflight_timeout_seconds={seconds}', result.stdout)

    def test_invalid_inspect_timeout_stops_before_preflight(self):
        for seconds in ('0', '-1', 'abc'):
            with self.subTest(seconds=seconds):
                result, rows = self.launch(mode='--preflight-only', inspect_timeout=seconds)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(rows, [])

    def test_frozen_configuration_reaches_data_bridge_and_evaluation(self):
        result, rows = self.launch(frozen=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([r["stage"] for r in rows][:2], ["frozen_prepare", "data"])
        prepare = rows[0]
        self.assertTrue(prepare["config"].endswith("runtime.json"))
        self.assertTrue(prepare["bundle"].endswith("rootfs.tar"))
        paths = []
        for row in rows:
            if row['stage'] in ('data', 'check_inspect_parallel.py', 'execute'):
                paths.append(row['args'][row['args'].index('--runtime-config') + 1])
        self.assertEqual(len(paths), 3)
        self.assertEqual(len(set(paths)), 1)

    def test_frozen_prepare_failure_stops_before_data_or_gpu(self):
        result, rows = self.launch(frozen=True, prepare_fail=True)
        self.assertEqual(result.returncode, 23)
        self.assertEqual([r["stage"] for r in rows], ["frozen_prepare"])

    def test_frozen_warm_cache_reuses_without_bundle(self):
        result, rows = self.launch(frozen=True, warm=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(rows[0]["stage"], "frozen_prepare")
        self.assertEqual(rows[0]["bundle"], "")

    def test_dependency_failure_stops_before_frozen_prepare_or_gpu(self):
        result, rows = self.launch(frozen=True, ensure_fail=True)
        self.assertEqual(result.returncode, 19)
        self.assertEqual(rows, [])

    def test_dry_run_does_not_prepare_frozen_runtime_or_start_gpu(self):
        result, rows = self.launch(frozen=True, dry_run=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([r["stage"] for r in rows], ["data"])
        self.assertNotIn("frozen_prepare", result.stdout + result.stderr)

    def test_full_run_calibrates_all_budgets_before_selected_formal_test(self):
        result, rows = self.launch(full=True, frozen=True, selected_budget=96000)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        data = [r['args'] for r in rows if r['stage'] == 'data']
        self.assertEqual([a[a.index('--split')+1] for a in data], ['dev', 'test'])
        executions = [r['args'] for r in rows if r['stage'] == 'execute']
        self.assertEqual([(a[a.index('--split')+1], a[a.index('--token-budget')+1]) for a in executions],
                         [('dev', '32000'), ('dev', '64000'), ('dev', '96000'), ('test', '96000')])
        self.assertEqual(len({a[a.index('--backend-manifest')+1] for a in executions}), 1)
        self.assertEqual(len({a[a.index('--runtime-config')+1] for a in executions}), 1)
        outputs = [Path(a[a.index('--output')+1]) for a in executions]
        self.assertEqual([p.name for p in outputs], ['dev-32000', 'dev-64000', 'dev-96000', 'test'])
        self.assertTrue(all('benchmark-v2-full-runs' in str(p) for p in outputs))
        analysis = next(r['args'] for r in rows if r['stage'] == 'analyze_benchmark_v2.py')
        self.assertEqual(analysis[analysis.index('--reports')+1:analysis.index('--output')],
                         [str(p/'summary.json') for p in outputs[:3]])
        formal = executions[-1]
        self.assertEqual(formal[formal.index('--calibration')+1], str(outputs[2]/'summary.json'))
        self.assertEqual(formal[formal.index('--budget-selection')+1], analysis[analysis.index('--output')+1])
        self.assertEqual([r['stage'] for r in rows][-5:],
                         ['execute', 'execute', 'execute', 'analyze_benchmark_v2.py', 'execute'])
        self.assertEqual(sum(r['stage'] == 'serve' for r in rows), 1)

    def test_model_concurrency_reaches_service_manifest_bridge_warmup_and_all_runs(self):
        result, rows = self.launch(full=True, concurrency='8', preflight_concurrency='4')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        flags = {'serve': '--max-num-seqs', 'manifest': '--max-num-seqs',
                 'check_inspect_parallel.py': '--max-concurrency', 'harness_pilot_warmup.py': '--concurrency',
                 'data': '--concurrency', 'execute': '--concurrency'}
        for row in rows:
            if row['stage'] in flags:
                args = row['args']
                self.assertEqual(args[args.index(flags[row['stage']])+1], '8', row)
                if row['stage'] == 'data':
                    self.assertEqual(args[args.index('--preflight-concurrency')+1], '4')

    def test_full_preflight_and_dry_run_never_start_model(self):
        for dry_run in (True, False):
            with self.subTest(dry_run=dry_run):
                result, rows = self.launch(full=True, dry_run=dry_run,
                                         mode=None if dry_run else '--preflight-only')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn('serve', [r['stage'] for r in rows])
                self.assertNotIn('execute', [r['stage'] for r in rows])
                data = [r['args'] for r in rows if r['stage'] == 'data']
                self.assertEqual([a[a.index('--split')+1] for a in data],
                                 ['test'] if dry_run else ['dev', 'test'])

    def test_failed_calibration_or_budget_selection_never_enters_test(self):
        for options in ({'fail_stage': 'dev-64000'}, {'selected_budget': None}):
            with self.subTest(options=options):
                result, rows = self.launch(full=True, **options)
                self.assertNotEqual(result.returncode, 0)
                executions = [r['args'] for r in rows if r['stage'] == 'execute']
                self.assertTrue(executions)
                self.assertTrue(all(a[a.index('--split')+1] == 'dev' for a in executions))

    def test_fixed_budget_runs_only_test_and_keeps_preflights(self):
        result, rows = self.launch(full=True, frozen=True, fixed_budget=64000, selected_budget=None)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        executions = [r['args'] for r in rows if r['stage'] == 'execute']
        self.assertEqual(len(executions), 1)
        args = executions[0]
        self.assertEqual(args[args.index('--split')+1], 'test')
        self.assertEqual(args[args.index('--token-budget')+1], '64000')
        self.assertIn('--fixed-budget-reason', args)
        self.assertNotIn('--calibration', args)
        stages = [r['stage'] for r in rows]
        self.assertNotIn('analyze_benchmark_v2.py', stages)
        self.assertIn('check_inspect_parallel.py', stages)
        self.assertIn('check_budget_tokenizer.py', stages)
        data = [r['args'] for r in rows if r['stage'] == 'data']
        self.assertEqual([a[a.index('--split')+1] for a in data], ['test'])

    def test_fixed_budget_requires_full_and_supported_value(self):
        for options in ({'fixed_budget': 64000}, {'full': True, 'fixed_budget': 128000}):
            result, rows = self.launch(**options)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(rows, [])

    def test_resume_after_killed_execution_preserves_original_artifacts(self):
        result, rows = self.launch(full=True, fixed_budget=64000, fail_stage='kill-execute',
                                   resume=True, named_run=True, frozen=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('manifest', [r['stage'] for r in rows])
        self.assertEqual(sum(r['stage'] == 'check_benchmark20_resume.py' for r in rows), 2)
        execute = next(r['args'] for r in rows if r['stage'] == 'execute')
        self.assertIn('--resume', execute)
        output = Path(execute[execute.index('--output')+1])
        self.assertEqual(output.parts[-2:], ('named-run', 'test'))
        preflight = next(r['args'] for r in rows if r['stage'] == 'data')
        self.assertIn('resume-attempts', preflight[preflight.index('--output')+1])

    def test_resume_before_backend_creation_restarts_preparation(self):
        result, rows = self.launch(full=True, fixed_budget=64000, fail_stage='check_gdn_kernels.py', resume=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('manifest', [r['stage'] for r in rows])
        execute = next(r['args'] for r in rows if r['stage'] == 'execute')
        self.assertNotIn('--resume', execute)

    def test_invalid_concurrency_stops_before_preflight_or_gpu(self):
        for options in ({'concurrency': '3'}, {'preflight_concurrency': '0'}):
            with self.subTest(options=options):
                result, rows = self.launch(full=True, **options)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(rows, [])

    def test_inspect_probe_parser_accepts_all_launcher_concurrency_values(self):
        spec = importlib.util.spec_from_file_location('inspect_parallel_probe', ROOT/'scripts/check_inspect_parallel.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for concurrency in ('4', '8', '16'):
            with self.subTest(concurrency=concurrency):
                argv = ['check_inspect_parallel.py', '--runtime', str(ROOT), '--tools-directory', '/tmp/tools',
                        '--output', '/tmp/probe-unused', '--max-concurrency', concurrency, '--child']
                with patch.object(sys, 'argv', argv), patch.object(module, 'child') as child:
                    module.main()
                child.assert_called_once()
                self.assertEqual(child.call_args.args[0].max_concurrency, int(concurrency))


if __name__ == "__main__":
    unittest.main()
