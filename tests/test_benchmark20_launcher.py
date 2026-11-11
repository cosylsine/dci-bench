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
    def launch(self, *, mode=None, fail_stage="", memory="0.80"):
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
            (scripts / "ensure_pool_smoke_tools.sh").write_text("#!/bin/sh\nexit 0\n")
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
            env = {**os.environ, "DRY_RUN": "0", "TEST_TRACE": str(trace),
                   "TEST_FAIL_STAGE": fail_stage, "VLLM_GPU_MEMORY_UTILIZATION": memory}
            command = ["bash", str(launcher), *([mode] if mode else [])]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
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

    def test_serving_diagnostic_skips_data_and_benchmark_but_requires_warmup(self):
        result, rows = self.launch(mode="--serving-smoke-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([r["stage"] for r in rows], ["harness_pilot_runtime.py", "check_gdn_kernels.py",
            "manifest", "serve", "harness_pilot_warmup.py"])
        self.assertIn("benchmark not executed", result.stdout)

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


if __name__ == "__main__":
    unittest.main()
