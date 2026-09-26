import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from dci_bench.data.registry import TASK_NAMES


class Phase5VllmLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = Path(__file__).resolve().parents[1] / "scripts/run_phase5_qwen38_vllm_node.sh"

    def test_shell_syntax(self):
        result = subprocess.run(["bash", "-n", str(self.script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sensecore_ranks_cover_each_task_once(self):
        selected = []
        with tempfile.TemporaryDirectory() as tmp:
            for rank in range(4):
                env = {
                    **os.environ,
                    "SENSECORE_PYTORCH_NODE_RANK": str(rank),
                    "SENSECORE_PYTORCH_NNODES": "4",
                    "SENSECORE_ACCELERATE_DEVICE_COUNT": "4",
                    "DRY_RUN": "1",
                    "RUN_LOG": str(Path(tmp) / f"rank{rank}.log"),
                    "SERVICE_LOG": str(Path(tmp) / f"service{rank}.log"),
                    "DIAGNOSTIC_DIR": str(Path(tmp) / f"artifacts{rank}"),
                }
                env.pop("MAX_MODEL_LEN", None)
                result = subprocess.run(
                    ["bash", str(self.script)], env=env, capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("dry-run complete", result.stdout)
                self.assertIn("vllm_max_num_seqs=4 vllm_gpu_memory_utilization=0.80", result.stdout)
                self.assertIn("max_model_len=65536", result.stdout)
                task_line = next(line for line in result.stdout.splitlines() if " tasks=" in line)
                selected.extend(task_line.split(" tasks=", 1)[1].split(" model=", 1)[0].split())
        self.assertCountEqual(selected, TASK_NAMES)

    def test_invalid_cluster_size_fails_before_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "SENSECORE_PYTORCH_NODE_RANK": "0",
                "SENSECORE_PYTORCH_NNODES": "3",
                "DRY_RUN": "1",
                "RUN_LOG": str(Path(tmp) / "run.log"),
            }
            result = subprocess.run(
                ["bash", str(self.script)], env=env, capture_output=True, text=True
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("WORLD_SIZE=4", result.stderr)

    def test_single_node_runs_only_selected_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "WORLD_SIZE": "1",
                "NODE_RANK": "0",
                "ONLY_TASK": "LLMTwitterHjerneRetrieval",
                "SENSECORE_ACCELERATE_DEVICE_COUNT": "4",
                "DRY_RUN": "1",
                "RUN_LOG": str(Path(tmp) / "run.log"),
                "SERVICE_LOG": str(Path(tmp) / "service.log"),
                "DIAGNOSTIC_DIR": str(Path(tmp) / "artifacts"),
            }
            result = subprocess.run(
                ["bash", str(self.script)], env=env, capture_output=True, text=True
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("tasks=LLMTwitterHjerneRetrieval", result.stdout)
        self.assertNotIn("tasks=LLMHC3FinanceRetrieval", result.stdout)
        self.assertIn("dry-run complete", result.stdout)

    def test_single_node_requires_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "WORLD_SIZE": "1",
                "NODE_RANK": "0",
                "ONLY_TASK": "",
                "DRY_RUN": "1",
                "RUN_LOG": str(Path(tmp) / "run.log"),
            }
            result = subprocess.run(
                ["bash", str(self.script)], env=env, capture_output=True, text=True
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires ONLY_TASK", result.stderr)

    def test_vllm_service_receives_memory_limits(self):
        service = self.script.with_name("serve_vllm_qwen38.sh")
        with tempfile.TemporaryDirectory() as tmp:
            fake_python = Path(tmp) / "python"
            fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n', encoding="utf-8")
            fake_python.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{tmp}:{os.environ['PATH']}",
                "VLLM_MAX_NUM_SEQS": "4",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.80",
                "VLLM_EXTRA_ARGS": "--max-num-seqs 256",
            }
            env.pop("MAX_MODEL_LEN", None)
            result = subprocess.run(
                ["bash", str(service)], env=env, capture_output=True, text=True
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertEqual(args[args.index("--max-model-len") + 1], "65536")
        self.assertEqual(args[-4:], ["--max-num-seqs", "4", "--gpu-memory-utilization", "0.80"])

    def test_engine_capacity_cannot_be_below_benchmark_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "SENSECORE_PYTORCH_NODE_RANK": "0",
                "SENSECORE_PYTORCH_NNODES": "4",
                "MAX_CONCURRENCY": "5",
                "DRY_RUN": "1",
                "RUN_LOG": str(Path(tmp) / "run.log"),
                "SERVICE_LOG": str(Path(tmp) / "service.log"),
                "DIAGNOSTIC_DIR": str(Path(tmp) / "artifacts"),
            }
            result = subprocess.run(
                ["bash", str(self.script)], env=env, capture_output=True, text=True
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("VLLM_MAX_NUM_SEQS must be at least MAX_CONCURRENCY", result.stdout)


if __name__ == "__main__":
    unittest.main()
