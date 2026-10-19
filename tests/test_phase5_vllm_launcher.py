import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from dci_bench.data.registry import TASK_NAMES


def load_sync_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "cluster_run_sync.py"
    spec = importlib.util.spec_from_file_location("cluster_run_sync_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Phase5VllmLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.script = cls.root / "scripts/run_phase5_qwen38_vllm_node.sh"
        cls.service = cls.root / "scripts/serve_vllm_qwen38.sh"

    def _dry_run(self, tmp: str, rank: int, **extra):
        tmp_path = Path(tmp)
        env = {
            **os.environ,
            "SENSECORE_PYTORCH_NODE_RANK": str(rank),
            "SENSECORE_PYTORCH_NNODES": "3",
            "SENSECORE_ACCELERATE_DEVICE_COUNT": "4",
            "DCI_RUN_ID": "test-cluster-run",
            "DRY_RUN": "1",
            "RESULTS_ROOT": str(tmp_path / "results"),
            "RUN_LOG": str(tmp_path / f"rank{rank}.log"),
            "SERVICE_LOG": str(tmp_path / f"service{rank}.log"),
            "DIAGNOSTIC_DIR": str(tmp_path / f"diag{rank}"),
            **extra,
        }
        return subprocess.run(
            ["bash", str(self.script)], env=env, capture_output=True, text=True
        )

    def test_shell_syntax(self):
        for script in (self.script, self.service):
            result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_three_node_assignments_cover_each_task_once(self):
        selected = []
        with tempfile.TemporaryDirectory() as tmp:
            for rank in range(3):
                result = self._dry_run(tmp, rank)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("dry-run complete", result.stdout)
                self.assertIn("world_size=3", result.stdout)
                self.assertIn("vllm_max_num_seqs=4 vllm_gpu_memory_utilization=0.80", result.stdout)
                self.assertIn("max_model_len=65536", result.stdout)
                task_line = next(line for line in result.stdout.splitlines() if " tasks=" in line)
                selected.extend(task_line.split(" tasks=", 1)[1].split(" model=", 1)[0].split())
            self.assertFalse((Path(tmp) / "results").exists())
        self.assertCountEqual(selected, TASK_NAMES)
        self.assertEqual(len(selected), 6)

    def test_four_node_compatibility_layout_remains_available(self):
        selected = []
        with tempfile.TemporaryDirectory() as tmp:
            for rank in range(4):
                result = self._dry_run(tmp, rank, SENSECORE_PYTORCH_NNODES="4")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                task_line = next(line for line in result.stdout.splitlines() if " tasks=" in line)
                selected.extend(task_line.split(" tasks=", 1)[1].split(" model=", 1)[0].split())
        self.assertCountEqual(selected, TASK_NAMES)

    def test_invalid_cluster_size_and_rank_fail_before_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_size = self._dry_run(tmp, 0, SENSECORE_PYTORCH_NNODES="2")
            self.assertEqual(bad_size.returncode, 2)
            self.assertIn("WORLD_SIZE=3", bad_size.stderr)
            bad_rank = self._dry_run(tmp, 3)
            self.assertEqual(bad_rank.returncode, 2)
            self.assertIn("WORLD_SIZE=3", bad_rank.stderr)

    def test_scheduler_job_id_is_shared_run_id_and_explicit_id_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "SENSECORE_PYTORCH_NODE_RANK": "0",
                "SENSECORE_PYTORCH_NNODES": "3",
                "SLURM_JOB_ID": "slurm-123/7",
                "DRY_RUN": "1",
                "RESULTS_ROOT": str(Path(tmp) / "results"),
            }
            scheduler = subprocess.run(
                ["bash", str(self.script)], env=env, capture_output=True, text=True
            )
            self.assertEqual(scheduler.returncode, 0, scheduler.stdout + scheduler.stderr)
            self.assertIn("scheduler=slurm", scheduler.stdout)
            self.assertIn("run_id=slurm-123_7", scheduler.stdout)
            env["DCI_RUN_ID"] = "explicit-cluster-1"
            explicit = subprocess.run(
                ["bash", str(self.script)], env=env, capture_output=True, text=True
            )
            self.assertEqual(explicit.returncode, 0, explicit.stdout + explicit.stderr)
            self.assertIn("run_id=explicit-cluster-1", explicit.stdout)

    def test_dry_run_writes_only_an_explicit_temporary_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._dry_run(tmp, 0)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            root = Path(tmp)
            self.assertTrue((root / "rank0.log").is_file())
            self.assertFalse((root / "results").exists())
            self.assertFalse((root / "diag0").exists())
            self.assertIn("sync_dir=", result.stdout)
            self.assertIn("task_path task=LLMAILAStatutes", result.stdout)

    def test_single_node_runs_only_selected_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "WORLD_SIZE": "1",
                "NODE_RANK": "0",
                "ONLY_TASK": "LLMTwitterHjerneRetrieval",
                "SENSECORE_ACCELERATE_DEVICE_COUNT": "4",
                "DCI_RUN_ID": "single-task-run",
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

    def test_vllm_service_preserves_independent_defaults_and_memory_limits(self):
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
            result = subprocess.run(
                ["bash", str(self.service)], env=env, capture_output=True, text=True
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertEqual(args[args.index("--max-model-len") + 1], "65536")
        self.assertEqual(args[args.index("--port") + 1], "8000")
        self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "1")
        self.assertEqual(args[args.index("--served-model-name") + 1], "MiniCPM5-2B")
        self.assertEqual(args[-4:], ["--max-num-seqs", "4", "--gpu-memory-utilization", "0.80"])
        self.assertIn('CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"', self.service.read_text())

    def test_launcher_boundary_configuration_is_explicit(self):
        text = self.script.read_text(encoding="utf-8")
        self.assertIn('BARRIER_TIMEOUT_SECONDS="${BARRIER_TIMEOUT_SECONDS:-172800}"', text)
        self.assertIn('INIT_WAIT_TIMEOUT_SECONDS="${INIT_WAIT_TIMEOUT_SECONDS:-1800}"', text)
        self.assertIn('BARRIER_HEARTBEAT_TIMEOUT_SECONDS="${BARRIER_HEARTBEAT_TIMEOUT_SECONDS:-1800}"', text)
        self.assertIn('VLLM_PORT="${VLLM_PORT:-17892}"', text)
        self.assertIn('TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"', text)
        self.assertIn('CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"', text)
        self.assertIn('BENCH_PYTHON="${BENCH_PYTHON:-${CONDA_ROOT}/envs/${BENCH_ENV}/bin/python}"', text)
        self.assertIn('CONDA_SH="${CONDA_SH:-${CONDA_ROOT}/etc/profile.d/conda.sh}"', text)
        self.assertIn('apt-get install -y --no-install-recommends', text)
        self.assertIn('setsid "${RUNNER_ARGS[@]}" &', text)
        self.assertIn('stop_evaluation_tree', text)
        self.assertIn('--runtime-info "${RUNTIME_ARTIFACT_PATH}"', text)
        self.assertNotIn('--runtime-artifact', text)
        self.assertIn('for task in "${TASKS[@]}"; do', text)
        self.assertIn('RUN_LOG="${RUN_LOG:-${RESULTS_ROOT}/job-logs/', text)
        self.assertIn('SERVICE_LOG="${SERVICE_LOG:-${RESULTS_ROOT}/job-logs/', text)
        self.assertIn('DIAGNOSTIC_DIR="${DIAGNOSTIC_DIR:-${RESULTS_ROOT}/job-artifacts/', text)

    def test_engine_capacity_cannot_be_below_benchmark_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._dry_run(tmp, 0, MAX_CONCURRENCY="5")
        self.assertEqual(result.returncode, 2)
        self.assertIn("VLLM_MAX_NUM_SEQS must be at least MAX_CONCURRENCY", result.stdout)

    def test_runtime_requires_shared_id_outside_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                **os.environ,
                "SENSECORE_PYTORCH_NODE_RANK": "0",
                "SENSECORE_PYTORCH_NNODES": "3",
                "DRY_RUN": "0",
                "RESULTS_ROOT": str(Path(tmp) / "results"),
                "RUN_LOG": str(Path(tmp) / "run.log"),
                "CONDA_SH": str(Path(tmp) / "missing-conda.sh"),
            }
            for name in (
                "DCI_RUN_ID",
                "RUN_ID",
                "SLURM_JOB_ID",
                "SENSECORE_JOB_ID",
                "SENSECORE_PYTORCH_JOB_ID",
                "JOB_ID",
                "LSB_JOBID",
                "PBS_JOBID",
            ):
                env.pop(name, None)
            result = subprocess.run(
                ["bash", str(self.script)], env=env, capture_output=True, text=True
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("require explicit DCI_RUN_ID", result.stdout)


class ClusterRunSyncTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = Path(__file__).resolve().parents[1] / "scripts/cluster_run_sync.py"

    def _base(self, state: Path, action: str, *, world_size: int = 3, timeout: float = 3):
        command = [
            sys.executable,
            str(self.script),
            action,
            "--state-dir",
            str(state),
            "--world-size",
            str(world_size),
            "--run-id",
            "barrier-test",
            "--model-key",
            "Qwen3.8-27B",
            "--tasks",
            "task-a",
            "task-b",
            "--poll-seconds",
            "0.02",
        ]
        if action in {"init", "wait"}:
            command.extend(["--timeout", str(timeout)])
        return command

    def test_default_startup_and_terminal_timeouts_are_separate(self):
        module = load_sync_module()
        common = [
            "--state-dir", "/tmp/dci-sync-test", "--world-size", "1", "--rank", "0",
            "--run-id", "timeout-test",
        ]
        self.assertEqual(module._parser().parse_args(["init", *common]).timeout, 1800.0)
        self.assertEqual(module._parser().parse_args(["wait", *common]).timeout, 172800.0)

    def test_scheduler_detection_and_run_id_selection(self):
        module = load_sync_module()
        detected = module.detect_scheduler({"SLURM_JOB_ID": "1234"})
        self.assertEqual(detected, {"scheduler": "slurm", "job_id": "1234", "env": "SLURM_JOB_ID"})
        self.assertEqual(module.choose_run_id(env={"SLURM_JOB_ID": "1234"})[0], "1234")
        self.assertEqual(module.choose_run_id("explicit-1", {"SLURM_JOB_ID": "1234"}), ("explicit-1", "DCI_RUN_ID"))

    def test_subprocess_barrier_publishes_unique_terminal_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "cluster-run"
            processes = []
            for rank in range(3):
                command = self._base(state, "init") + ["--rank", str(rank)]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                processes.append(process)
            for process in processes:
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, stdout + stderr)
            for rank, status in enumerate(("success", "failure", "aborted")):
                result = subprocess.run(
                    self._base(state, "terminal")
                    + [
                        "--rank",
                        str(rank),
                        "--status",
                        status,
                        "--exit-code",
                        "0" if status == "success" else "2",
                        "--reason",
                        status,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            barrier = subprocess.run(
                self._base(state, "wait") + ["--rank", "0"], capture_output=True, text=True
            )
            self.assertEqual(barrier.returncode, 1, barrier.stdout + barrier.stderr)
            self.assertEqual(
                json.loads((state / "terminal/rank-0.json").read_text())["status"], "success"
            )
            self.assertEqual(
                json.loads((state / "terminal/rank-1.json").read_text())["status"], "failure"
            )
            self.assertEqual(
                json.loads((state / "terminal/rank-2.json").read_text())["status"], "aborted"
            )

    def test_terminal_marker_is_immutable_on_repeated_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "cluster-run"
            for rank in range(3):
                result = subprocess.run(
                    self._base(state, "init") + ["--rank", str(rank)], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            command = self._base(state, "terminal") + ["--rank", "0", "--status", "success"]
            self.assertEqual(subprocess.run(command, capture_output=True, text=True).returncode, 0)
            repeat = subprocess.run(
                self._base(state, "terminal") + ["--rank", "0", "--status", "failure"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(repeat.returncode, 0, repeat.stdout + repeat.stderr)
            self.assertEqual(
                json.loads((state / "terminal/rank-0.json").read_text())["status"], "success"
            )

    def test_init_rejects_duplicate_same_rank_participant(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "cluster-run"
            first = subprocess.run(
                self._base(state, "init") + ["--rank", "0"], capture_output=True, text=True
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            first = subprocess.run(
                self._base(state, "init") + ["--rank", "1"], capture_output=True, text=True
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            repeat = subprocess.run(
                self._base(state, "init") + ["--rank", "1"], capture_output=True, text=True
            )
            self.assertEqual(repeat.returncode, 2)
            self.assertIn("refusing rank reuse", repeat.stderr)

    def test_stale_heartbeat_propagates_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "cluster-run"
            for rank in range(3):
                result = subprocess.run(
                    self._base(state, "init") + ["--rank", str(rank)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = subprocess.run(
                    self._base(state, "heartbeat") + ["--rank", str(rank)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            time.sleep(0.05)
            result = subprocess.run(
                self._base(state, "check-abort")
                + ["--rank", "0", "--heartbeat-timeout", "0.001"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["stale_ranks"])

    def test_never_joined_rank_uses_group_creation_for_bounded_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "cluster-run"
            result = subprocess.run(
                self._base(state, "init") + ["--rank", "0"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            time.sleep(0.05)
            result = subprocess.run(
                self._base(state, "check-abort")
                + ["--rank", "0", "--heartbeat-timeout", "0.001"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            self.assertIn(1, payload["stale_ranks"])
            self.assertIn(2, payload["stale_ranks"])
            self.assertEqual(payload["heartbeats"]["1"]["source"], "group_created_at")

    def test_model_key_mismatch_cannot_join_existing_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "cluster-run"
            first = subprocess.run(
                self._base(state, "init") + ["--rank", "0"], capture_output=True, text=True
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            mismatch = self._base(state, "init") + ["--rank", "1", "--model-key", "other-model"]
            result = subprocess.run(mismatch, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("model key mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
