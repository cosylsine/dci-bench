"""Run the real shell control flow with synthetic processes, without GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Benchmark20LauncherTests(unittest.TestCase):
    def launch(self, *, mode=None, fail_stage="", memory="0.80", frozen=False,
               prepare_fail=False, ensure_fail=False, dry_run=False, warm=False, preflight_concurrency="4"):
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
if os.environ.get('TEST_FAIL_STAGE') == stage:
    sys.exit(23)
if stage == 'harness_pilot_runtime.py':
    pathlib.Path(args[args.index('--output')+1]).write_text(json.dumps(
        {'cuda': {'gpus': [{'name': 'RTX 5090', 'capability': [12, 0]}] * 4}}))
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
                   "VLLM_GPU_MEMORY_UTILIZATION": memory, "PREFLIGHT_CONCURRENCY": preflight_concurrency}
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
            command = ["bash", str(launcher), *([mode] if mode else [])]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
            rows = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
            return result, rows

    def test_complete_launch_orders_kernel_probe_profile_worker_and_evaluation(self):
        result, rows = self.launch(memory="0.75")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([r["stage"] for r in rows], [
            "data", "check_inspect_parallel.py", "harness_pilot_runtime.py", "check_gdn_kernels.py",
            "manifest", "serve", "harness_pilot_warmup.py", "execute"])
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
            "manifest", "serve", "harness_pilot_warmup.py"])
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
            "manifest", "serve", "harness_pilot_warmup.py"])

    def test_failed_kernel_probe_stops_before_manifest_or_model_load(self):
        result, rows = self.launch(mode="--serving-smoke-only", fail_stage="check_gdn_kernels.py")
        self.assertEqual(result.returncode, 23)
        self.assertEqual([r["stage"] for r in rows], ["harness_pilot_runtime.py", "check_gdn_kernels.py"])

    def test_failed_service_warmup_never_starts_benchmark(self):
        result, rows = self.launch(fail_stage="harness_pilot_warmup.py")
        self.assertEqual(result.returncode, 23)
        self.assertNotIn("execute", [r["stage"] for r in rows])

    def test_cpu_only_mode_never_probes_or_loads_gpu(self):
        result, rows = self.launch(mode="--preflight-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([r["stage"] for r in rows], ["data", "check_inspect_parallel.py"])

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


if __name__ == "__main__":
    unittest.main()
