"""CPU-only regressions for new-query selection and immutable pilot launch."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import prepare_harness_confirmation as C
from prepare_harness_pilot_continuation import execution_order

ARCHIVE = REPO / "results/harness-continuations/20261014T043719Z-validation-remaining/runtime"


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="dci-confirmation-test-")
        self.addCleanup(temp.cleanup)
        self.bundle = Path(temp.name)
        self.runtime = self.bundle / "runtime"
        (self.runtime / "scripts").mkdir(parents=True)
        self.tasks = [f"Task{i}" for i in range(6)]
        self.history = {"selection_inputs": [
            {"path": "old/plan.json", "sha256": "fixture", "query_ids": {t: ["q0", "q1"] for t in self.tasks}},
            {"path": "primer/run-manifest.json", "sha256": "fixture", "query_ids": {self.tasks[0]: ["q2"]}},
        ]}
        self.save(self.runtime / "scripts/harness_pilot_queries.json", {t: ["q0"] for t in self.tasks})
        for task in self.tasks:
            path = self.runtime / "data/metadata" / task / "queries.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps({"_id": f"q{i}", "text": "fixture"}) + "\n" for i in range(25)))
        self.freeze = C.compute_freeze(self.runtime, self.history, 20)
        self.save(self.runtime / C.HISTORY_FILE, self.history)
        self.save(self.runtime / C.FREEZE_FILE, self.freeze)
        self.backend = C.read_json(REPO / "results/harness-replay-runs/20261014T163113Z-final-replay-1/backend.json")
        self.save(self.bundle / "expected-backend.json", self.backend)
        self.manifest = {"query_ids": self.freeze["query_ids"], "take": 20,
                         "planned_model_evaluations": 240, "contract_sha256": C.CONTRACTS,
                         "bench_python": C.BENCH_PYTHON, "serving_python": C.SERVING_PYTHON,
                         "bench_identity": {"fixture": True}, "node_version": "fixture",
                         "output_parent": str(self.bundle / "outputs")}
        self.plan = {"query_ids": self.freeze["query_ids"], "query_suite": "confirmation",
                     "profiles": {p: {"sha256": sha} for p, sha in C.CONTRACTS.items()},
                     "max_concurrency": 4, "profile_order": "alternating",
                     "execution_order": execution_order(self.tasks, C.PROFILES, "alternating")}

    def save(self, path, value):
        path.write_text(json.dumps(value))

    def test_selection_excludes_history_and_never_needs_qrels(self):
        # No qrels exist in this fixture. Prior primer IDs must also be excluded.
        for task, ids in self.freeze["query_ids"].items():
            self.assertEqual(len(ids), 20)
            self.assertTrue(set(ids).isdisjoint(self.freeze["excluded_query_ids"][task]))
        self.assertNotIn("q2", self.freeze["query_ids"][self.tasks[0]])
        self.assertEqual(C.load_confirmation(self.runtime), self.freeze)

    def test_query_metadata_order_does_not_change_selection(self):
        before = self.freeze["query_ids"]
        for task in self.tasks:
            path = self.runtime / "data/metadata" / task / "queries.jsonl"
            path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
        self.assertEqual(C.compute_freeze(self.runtime, self.history, 20)["query_ids"], before)
        with self.assertRaisesRegex(ValueError, "freeze changed"):
            C.load_confirmation(self.runtime)

    def test_duplicate_ids_or_insufficient_pool_fail_closed(self):
        task = self.tasks[0]
        path = self.runtime / "data/metadata" / task / "queries.jsonl"
        original = path.read_text()
        path.write_text(original + original.splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            C.compute_freeze(self.runtime, self.history, 20)
        path.write_text("\n".join(original.splitlines()[:10]))
        with self.assertRaisesRegex(ValueError, "only"):
            C.compute_freeze(self.runtime, self.history, 20)

    def test_saved_selection_or_exclusion_tampering_is_rejected(self):
        changed = deepcopy(self.freeze)
        changed["query_ids"][self.tasks[0]][0] = "q0"
        self.save(self.runtime / C.FREEZE_FILE, changed)
        with self.assertRaisesRegex(ValueError, "freeze changed"):
            C.load_confirmation(self.runtime)
        self.save(self.runtime / C.FREEZE_FILE, self.freeze)
        changed_history = deepcopy(self.history)
        changed_history["selection_inputs"].pop()
        self.save(self.runtime / C.HISTORY_FILE, changed_history)
        with self.assertRaisesRegex(ValueError, "freeze changed"):
            C.load_confirmation(self.runtime)

    def test_historical_command_and_manifest_ids_keep_canonical_strings(self):
        self.assertEqual(C.query_ids_from_record({"argv": ["python", "run_task.py", "--task", "Task0",
                         "--query-ids", "00001", "Q72", "--run-id", "x"]}, self.tasks),
                         {"Task0": ["00001", "Q72"]})
        self.assertEqual(C.query_ids_from_record({"task": "Task0", "query_ids": ["00000"]}, self.tasks),
                         {"Task0": ["00000"]})

    def test_generated_pilot_runs_new_suite_and_rejects_partial_selection(self):
        source = C.patch_pilot((ARCHIVE / "scripts/run_harness_pilot.py").read_text())
        target = self.runtime / "scripts/run_harness_pilot.py"
        target.write_text(source)
        spec = importlib.util.spec_from_file_location("confirmation_fixture_pilot", target)
        pilot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot)
        # Real plan generation, with only protocol subprocesses mocked.
        contracts = {p: {"sha256": sha, "contract": {"generation_config": {}}} for p, sha in C.CONTRACTS.items()}
        argv = ["--backend-manifest", str(self.bundle / "expected-backend.json"), "--profiles", *C.PROFILES,
                "--query-suite", "confirmation", "--query-limit", "20", "--profile-order", "alternating",
                "--max-concurrency", "4", "--dry-run"]
        stream = io.StringIO()
        with patch.object(pilot, "_describe_profiles", return_value=contracts), redirect_stdout(stream):
            self.assertEqual(pilot.main(argv), 0)
        actual = json.loads(stream.getvalue())
        C.verify_plan(self.manifest, actual)
        self.assertEqual(actual["confirmation_freeze"], self.freeze)
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
            pilot.main([*argv, "--tasks", "Task0"])
        with self.assertRaisesRegex(ValueError, "patch anchor"):
            C.patch_pilot(source)

    def test_runtime_and_weights_drift_rejected(self):
        C.check_runtime(self.bundle, self.backend["backend"]["runtime"])
        C.check_backend(self.bundle, self.backend)
        changed = deepcopy(self.backend)
        changed["backend"]["runtime"]["packages"]["vllm"] = "different"
        with self.assertRaisesRegex(ValueError, "runtime changed"):
            C.check_runtime(self.bundle, changed["backend"]["runtime"])
        changed = deepcopy(self.backend)
        changed["model"]["weights"]["manifest_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "weights changed"):
            C.check_backend(self.bundle, changed)

    def test_occupied_serving_or_overlapping_bridge_port_fails_before_start(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            with self.assertRaises(OSError):
                C.check_ports({"VLLM_PORT": str(sock.getsockname()[1])})
        with self.assertRaisesRegex(ValueError, "overlapping"):
            C.check_ports({"VLLM_PORT": "28131"})

    def test_environment_cannot_change_model_protocol_or_concurrency(self):
        with patch.dict(os.environ, {"TP_SIZE": "1", "MAX_CONCURRENCY": "99", "MODEL_PATH": "/wrong",
                                    "PYTHONPATH": "/live", "NODE_OPTIONS": "--wrong", "DCI_EVALUATION_TRACK": "code"}):
            env = C.launch_environment(self.bundle, self.manifest)
        self.assertEqual(env["TP_SIZE"], "4")
        self.assertEqual(env["MAX_CONCURRENCY"], "4")
        self.assertEqual(env["MODEL_PATH"], self.backend["model"]["source_path"])
        self.assertEqual(env["PYTHONPATH"], str(self.runtime))
        self.assertEqual(env["DCI_EVALUATION_TRACK"], "document")
        self.assertNotIn("NODE_OPTIONS", env)

    def test_cli_and_environment_dry_run_never_launch_gpu(self):
        for cli, environmental in [(True, "0"), (False, "1")]:
            with self.subTest(cli=cli), patch.dict(os.environ, {"DRY_RUN": environmental}), \
                 patch.object(C, "verify_bundle", return_value=self.manifest), \
                 patch.object(C, "python_identity", return_value=self.manifest["bench_identity"]), \
                 patch.object(C.subprocess, "check_output", side_effect=["fixture", json.dumps(self.plan)]), \
                 patch.object(C.subprocess, "run") as run, patch.object(C, "finish_run") as finish, \
                 patch.object(C.os, "execvpe") as execute, \
                 redirect_stdout(io.StringIO()):
                C.run_bundle(self.bundle, dry_run=cli)
            finish.assert_not_called()
            execute.assert_not_called()
            shell_calls = [call for call in run.call_args_list if call.args[0][0] == "bash"]
            self.assertEqual(len(shell_calls), 1)
            self.assertEqual(shell_calls[0].kwargs["env"]["DRY_RUN"], "1")

    def test_frozen_data_or_dependencies_tampering_stops_before_subprocess(self):
        self.manifest.update(files={"expected-backend.json": C.file_hash(self.bundle / "expected-backend.json")},
                             runtime_inventory=C.inventory(self.runtime, self.runtime / "data/workspaces"))
        self.save(self.bundle / "confirmation.json", self.manifest)
        (self.bundle / "confirmation.sha256").write_text(C.file_hash(self.bundle / "confirmation.json"))
        C.verify_bundle(self.bundle)
        (self.runtime / "data/metadata/Task0/queries.jsonl").write_text("changed")
        with patch.object(C.subprocess, "run") as run, self.assertRaisesRegex(ValueError, "frozen code"):
            C.run_bundle(self.bundle)
        run.assert_not_called()

    def test_frozen_plan_rejects_profile_and_order_drift(self):
        C.verify_plan(self.manifest, self.plan)
        changed = deepcopy(self.plan)
        changed["execution_order"].reverse()
        with self.assertRaisesRegex(ValueError, "paired order"):
            C.verify_plan(self.manifest, changed)
        changed = deepcopy(self.plan)
        changed["profiles"][C.PROFILES[0]]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "profile contracts"):
            C.verify_plan(self.manifest, changed)

    def test_prepared_entry_does_not_write_bytecode_and_live_edits_are_isolated(self):
        # Small physical-copy fixture exercises the real generated shell entry.
        source = self.bundle / "source"
        source.mkdir()
        original = source / "runtime"
        shutil.copytree(self.runtime, original)
        (original / C.FREEZE_FILE).unlink()
        (original / C.HISTORY_FILE).unlink()
        workspaces = self.bundle / "shared-workspaces"
        for task in self.tasks:
            corpus = workspaces / task / "corpus"
            corpus.mkdir(parents=True)
            (corpus / "doc.md").write_text("original evidence")
        (original / "data/workspaces").symlink_to(workspaces)
        for name in ("scripts/run_harness_pilot.py", "scripts/run_harness_pilot.sh"):
            shutil.copyfile(ARCHIVE / name, original / name)
        shutil.copyfile(REPO / "scripts/prepare_harness_pilot_continuation.py", source / "continue.py")
        C.write_json(source / "continuation.json", {"shared_workspaces": str(workspaces),
                     "runtime_inventory": C.inventory(original, workspaces)})
        output = self.bundle / "prepared"
        with patch.object(C, "collect_history", return_value=self.history), redirect_stdout(io.StringIO()):
            C.prepare(source, REPO, output, self.bundle / "expected-backend.json", 20)
        env = os.environ.copy()
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        result = subprocess.run(["bash", str(output / "launch.sh"), "--help"],
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(list((output / "runtime").rglob("__pycache__")))
        (workspaces / "Task0/corpus/doc.md").write_text("edited live document")
        (original / "scripts/run_harness_pilot.py").write_text("edited live code")
        C.verify_bundle(output)
        self.assertEqual((output / "runtime/data/workspaces/Task0/corpus/doc.md").read_text(), "original evidence")
        self.assertFalse((output / "runtime/data/workspaces").is_symlink())

    def test_existing_output_is_never_reused_or_launched(self):
        existing = self.bundle / "existing-output"
        existing.mkdir()
        with patch.dict(os.environ, {"DRY_RUN": "0", "PILOT_ROOT": str(existing)}), \
             patch.object(C, "verify_bundle", return_value=self.manifest), \
             patch.object(C, "python_identity", return_value=self.manifest["bench_identity"]), \
             patch.object(C.subprocess, "check_output", side_effect=["fixture", json.dumps(self.plan)]), \
             patch.object(C.subprocess, "run"), patch.object(C, "check_ports"), \
             patch.object(C.os, "execvpe") as execute, self.assertRaises(FileExistsError):
            C.run_bundle(self.bundle)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
