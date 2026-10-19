"""CPU regressions for frozen 527-query selective launch and completion gates."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import prepare_harness_full as F

ARCHIVE = REPO / "results/harness-confirmations/20261015-selection-confirmation-v2"


class FullTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="dci-full-test-")
        self.addCleanup(temp.cleanup)
        self.bundle = Path(temp.name)
        self.runtime = self.bundle / "runtime"
        (self.runtime / "scripts").mkdir(parents=True)
        for task, count in F.COUNTS.items():
            directory = self.runtime / "data/metadata" / task
            directory.mkdir(parents=True)
            (directory / "queries.jsonl").write_text(''.join(
                json.dumps({"_id": f"{i:05d}", "text": "fixture"}) + '\n' for i in range(count)))
        F.write_json(self.runtime / "scripts/harness_pilot_queries.json", {t: ["00000"] for t in F.COUNTS})
        self.freeze = F.compute_full_freeze(self.runtime)
        F.write_json(self.runtime / F.FREEZE_FILE, self.freeze)
        self.backend = F.read_json(ARCHIVE / "expected-backend.json")
        F.write_json(self.bundle / "expected-backend.json", self.backend)
        self.manifest = {
            "query_ids": self.freeze["query_ids"], "planned_model_evaluations": 527, "take": 100,
            "contract_sha256": {F.PROFILE: F.CONTRACT_SHA256}, "bench_python": F.BENCH_PYTHON,
            "serving_python": F.SERVING_PYTHON, "bench_identity": {"fixture": True}, "node_version": "fixture",
            "results_root": str(self.bundle / "results/Qwen3.8-27B"),
            "output_parent": str(self.bundle / "results/Qwen3.8-27B/full-selective-runs"),
        }
        self.plan = {"model_key": F.MODEL_KEY, "query_ids": self.freeze["query_ids"], "query_suite": "full",
                     "profiles": {F.PROFILE: {"sha256": F.CONTRACT_SHA256, "contract": {"generation_config": {}}}},
                     "profile_order": "as-listed", "max_concurrency": 4,
                     "execution_order": F.execution_order(list(F.COUNTS), [F.PROFILE], "as-listed")}

    def pilot(self):
        path = self.runtime / "scripts/run_harness_pilot.py"
        path.write_text(F.patch_pilot((ARCHIVE / "runtime/scripts/run_harness_pilot.py").read_text()))
        spec = importlib.util.spec_from_file_location("full_fixture_pilot", path)
        pilot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot)
        return pilot

    def args(self):
        return ["--backend-manifest", str(self.bundle / "expected-backend.json"), "--profiles", F.PROFILE,
                "--query-suite", "full", "--query-limit", "100", "--profile-order", "as-listed",
                "--max-concurrency", "4", "--dry-run"]

    def test_all_queries_include_previous_pilot_ids_and_keep_metadata_order(self):
        self.assertEqual(sum(map(len, self.freeze["query_ids"].values())), 527)
        self.assertEqual(self.freeze["counts"], F.COUNTS)
        for task, ids in self.freeze["query_ids"].items():
            self.assertEqual(ids[0], "00000")
            self.assertEqual(ids[-1], f"{F.COUNTS[task] - 1:05d}")
        self.assertEqual(F.load_full(self.runtime), self.freeze)

    def test_missing_duplicate_or_changed_queries_fail_closed(self):
        task = next(iter(F.COUNTS))
        path = self.runtime / "data/metadata" / task / "queries.jsonl"
        original = path.read_text()
        for content in [original + original.splitlines()[0] + '\n', '\n'.join(original.splitlines()[:-1])]:
            path.write_text(content)
            with self.assertRaisesRegex(ValueError, "count/identity"):
                F.compute_full_freeze(self.runtime)
        path.write_text(original.replace('"fixture"', '"changed"', 1))
        with self.assertRaisesRegex(ValueError, "freeze changed"):
            F.load_full(self.runtime)

    def test_actual_orchestrator_plan_is_527_single_arm(self):
        pilot = self.pilot()
        stream = io.StringIO()
        with patch.object(pilot, "_describe_profiles", return_value=self.plan["profiles"]), redirect_stdout(stream):
            self.assertEqual(pilot.main(self.args()), 0)
        actual = json.loads(stream.getvalue())
        F.verify_plan(self.manifest, actual)
        self.assertEqual(actual["full_freeze"], self.freeze)
        self.assertEqual(len(actual["execution_order"]), 6)

    def test_partial_query_or_task_arguments_are_rejected(self):
        pilot = self.pilot()
        for extra in [["--tasks", next(iter(F.COUNTS))], ["--query-limit", "20"]]:
            with self.subTest(extra=extra), self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
                pilot.main([*self.args(), *extra])

    def test_another_or_extra_profile_is_rejected(self):
        pilot = self.pilot()
        for profiles in [{"budgeted-stable": {"sha256": "other"}},
                         {**self.plan["profiles"], "budgeted-stable": {"sha256": "other"}}]:
            with self.subTest(profiles=profiles), patch.object(pilot, "_describe_profiles", return_value=profiles), \
                 self.assertRaisesRegex(ValueError, "profile contracts"), redirect_stdout(io.StringIO()):
                pilot.main(self.args())

    def test_manifest_plan_rejects_wrong_scope_model_profile_and_concurrency(self):
        F.verify_plan(self.manifest, self.plan)
        mutations = [("model_key", "other"), ("query_suite", "confirmation"), ("max_concurrency", 1),
                     ("profile_order", "alternating"), ("execution_order", self.plan["execution_order"][:-1])]
        for key, value in mutations:
            changed = deepcopy(self.plan)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                F.verify_plan(self.manifest, changed)
        changed = deepcopy(self.plan)
        changed["query_ids"][next(iter(F.COUNTS))].pop()
        with self.assertRaisesRegex(ValueError, "query scope"):
            F.verify_plan(self.manifest, changed)

    def test_environment_cannot_redirect_outputs_or_change_protocol_runtime(self):
        with patch.dict(os.environ, {"PILOT_ROOT": "/wrong", "RESULTS_ROOT": "/wrong", "MODEL_PATH": "/wrong",
                                    "MAX_CONCURRENCY": "99", "TP_SIZE": "1", "DCI_EVALUATION_TRACK": "code",
                                    "NODE_OPTIONS": "--wrong", "PYTHONPATH": "/live", "PYTHONHOME": "/wrong"}):
            env = F.launch_environment(self.bundle, self.manifest)
        for name in ["PILOT_ROOT", "RESULTS_ROOT", "NODE_OPTIONS", "PYTHONHOME"]:
            self.assertNotIn(name, env)
        self.assertEqual(env["MODEL_PATH"], self.backend["model"]["source_path"])
        self.assertEqual(env["MAX_CONCURRENCY"], "4")
        self.assertEqual(env["TP_SIZE"], "4")
        self.assertEqual(env["DCI_EVALUATION_TRACK"], "document")
        self.assertEqual(env["PYTHONPATH"], str(self.runtime))

    def test_output_is_unique_and_stays_inside_model_root(self):
        output = F.output_path(self.manifest, {"FULL_RUN_ID": "new-full-1"})
        self.assertTrue(output.is_relative_to(Path(self.manifest["results_root"])))
        output.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            F.output_path(self.manifest, {"FULL_RUN_ID": "new-full-1"})
        for run_id in ["../escape", "/absolute", "two words"]:
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                F.output_path(self.manifest, {"FULL_RUN_ID": run_id})
        escaped = deepcopy(self.manifest)
        escaped["output_parent"] = str(self.bundle / "outside")
        with self.assertRaisesRegex(ValueError, "escapes"):
            F.output_path(escaped, {})

    def test_six_sandbox_preflights_use_selective(self):
        with patch.object(F.subprocess, "run") as run:
            F.check_sandboxes(self.bundle, self.manifest)
        self.assertEqual(run.call_count, 6)
        for task, call in zip(F.COUNTS, run.call_args_list):
            self.assertEqual(call.args[0][-1], str(self.runtime / "data/workspaces" / task))
            self.assertEqual(call.kwargs["env"]["DCI_PROTOCOL_PROFILE"], F.PROFILE)
            self.assertIn("--preflight-only", call.args[0])

    def test_cpu_modes_never_exec_gpu_and_preflight_checks_sandboxes(self):
        for dry, preflight, environmental in [(True, False, "0"), (False, True, "0"), (False, False, "1")]:
            with self.subTest(dry=dry, preflight=preflight, env=environmental), \
                 patch.dict(os.environ, {"DRY_RUN": environmental}), \
                 patch.object(F, "verify_bundle", return_value=self.manifest), \
                 patch.object(F.confirmation, "python_identity", return_value=self.manifest["bench_identity"]), \
                 patch.object(F.subprocess, "check_output", side_effect=["fixture", json.dumps(self.plan)]), \
                 patch.object(F.subprocess, "run") as run, patch.object(F, "check_sandboxes") as sandbox, \
                 patch.object(F, "check_ports"), patch.object(F.os, "execvpe") as execute, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(F.run_bundle(self.bundle, dry_run=dry, preflight_only=preflight), 0)
            execute.assert_not_called()
            self.assertEqual(sandbox.call_count, int(preflight))
            shell = [c for c in run.call_args_list if c.args[0][0] == "bash"]
            self.assertEqual(len(shell), 1)
            self.assertEqual(shell[0].kwargs["env"]["DRY_RUN"], "1")

    def test_frozen_data_tamper_stops_before_subprocess(self):
        self.manifest.update(files={}, runtime_inventory=F.inventory(self.runtime, self.runtime / "data/workspaces"))
        F.write_json(self.bundle / "frozen.json", self.manifest)
        (self.bundle / "frozen.sha256").write_text(F.file_hash(self.bundle / "frozen.json"))
        F.verify_bundle(self.bundle)
        (self.runtime / "data/metadata" / next(iter(F.COUNTS)) / "queries.jsonl").write_text("changed")
        with patch.object(F.subprocess, "run") as run, self.assertRaisesRegex(ValueError, "frozen code"):
            F.run_bundle(self.bundle)
        run.assert_not_called()

    def test_patched_shell_dry_run_does_not_create_output(self):
        shell = self.runtime / "scripts/run_harness_pilot.sh"
        shell.write_text(F.patch_launcher((ARCHIVE / "runtime/scripts/run_harness_pilot.sh").read_text()))
        subprocess.run(["bash", "-n", str(shell)], check=True)
        output = self.bundle / "not-created"
        result = subprocess.run(["bash", str(shell), *self.args()[2:]],
                                env={**os.environ, "DRY_RUN": "1", "PILOT_ROOT": str(output)},
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(output.exists())

    def test_finish_preserves_partial_failures_and_returns_nonzero(self):
        for valid in [2, 1]:
            output = Path(self.manifest["output_parent"]) / f"finish-{valid}"
            evaluation = output / "evaluation"
            run = evaluation / "runs" / F.MODEL_KEY / "Fixture" / "runs" / f"pilot-{F.PROFILE}"
            run.mkdir(parents=True)
            F.write_json(evaluation / "plan.json", self.plan)
            F.write_json(evaluation / "summary.json", {})
            F.write_json(evaluation / "_COMPLETE", {"rows": 1, "summary_sha256": F.file_hash(evaluation / "summary.json")})
            F.write_json(run / "run-manifest.json", {"benchmark_contract": {
                "task_context": {"task_name": "Fixture"}, "generation_config": {}}})
            F.write_json(run / "task-result.json", {"counts": {"completed": 2, "valid": valid}})
            if not (self.bundle / "frozen.json").exists():
                F.write_json(self.bundle / "frozen.json", self.manifest)
            manifest = {**self.manifest, "planned_model_evaluations": 2}
            with patch.object(F, "COUNTS", {"Fixture": 2}), patch.object(F, "verify_bundle"), \
                 patch.object(F, "verify_plan"), patch.object(F, "verify_group", return_value={"samples": 2}), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(F.finish_run(self.bundle, output, manifest), 0 if valid == 2 else 1)
            marker = F.read_json(output / "_COMPLETE")
            self.assertEqual(marker["invalid"], 2 - valid)
            self.assertEqual(marker["status"], "complete" if valid == 2 else "complete_with_sample_failures")


if __name__ == "__main__":
    unittest.main()
