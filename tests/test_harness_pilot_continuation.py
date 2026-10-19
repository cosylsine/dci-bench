"""No-GPU regressions for isolated, whole-task pilot continuation."""

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pilot_continuation", REPO / "scripts/prepare_harness_pilot_continuation.py")
CONT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONT)


class ContinuationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dci-continuation-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "checkout"
        self.pilot = self.root / "pilot-$"
        self.evaluation = self.pilot / "evaluation"
        self.evaluation.mkdir(parents=True)
        self.bundle = self.root / "bundle"
        (self.repo / "data/metadata").mkdir(parents=True)
        (self.repo / "data/workspaces").mkdir()
        (self.repo / "data/metadata/queries.json").write_text('{"original":true}\n')
        (self.repo / "pi-dci/packages/agent/dist").mkdir(parents=True)
        (self.repo / "pi-dci/packages/agent/dist/index.js").write_text("export const value = 1;\n")
        (self.repo / "pi-dci/node_modules/@fixture").mkdir(parents=True)
        (self.repo / "pi-dci/node_modules/@fixture/agent").symlink_to("../../packages/agent")
        self.tasks = [f"Task{i}" for i in range(6)]
        self.profiles = ["budgeted-stable", "budgeted-selective"]
        self.plan = {
            "model_key": "fixture", "query_suite": "validation", "profile_order": "alternating",
            "query_ids": {t: [f"Q{i}" for i in range(10)] for t in self.tasks},
            "profiles": {p: {"sha256": p} for p in self.profiles},
            "validation_freeze": {"frozen": True}, "max_concurrency": 4,
            "execution_groups": [{"task": t, "profiles": self.profiles[::-1] if i % 2 == 0 else self.profiles}
                                 for i, t in enumerate(self.tasks)],
            "execution_order": CONT.execution_order(self.tasks, self.profiles, "alternating"),
        }
        self.save(self.evaluation / "plan.json", self.plan)
        self.save(self.pilot / "backend.json", {
            "model": {"source_path": "/fixture/model", "served_model_name": "fixture"},
            "backend": {"tensor_parallel_size": 4, "max_num_seqs": 4, "gpu_memory_utilization": .9,
                        "runtime": {"launch_environment": {"VLLM_EXTRA_ARGS": "--enforce-eager"}}},
        })
        self.sources = {"dci_bench/data/expansion_documents.py": b"original = True\n",
                        "scripts/run_harness_pilot.sh": b"#!/bin/bash\nexit 0\n"}
        self.archive(self.sources)
        for task in self.tasks[:2]:
            for profile in self.profiles:
                self.complete(task, profile)

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def archive(self, sources):
        with tarfile.open(self.evaluation / "sources.tar.gz", "w:gz") as archive:
            for name, payload in sources.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(payload))
        self.save(self.evaluation / "source-hashes.json",
                  {name: hashlib.sha256(payload).hexdigest() for name, payload in sources.items()})

    def complete(self, task, profile):
        run = self.evaluation / "runs/fixture" / task / "runs" / f"pilot-{profile}"
        ids = self.plan["query_ids"][task]
        index = []
        for q in ids:
            self.save(run / q / "trace.json", {"query": q})
            ref = {"path": f"{q}/trace.json", "sha256": CONT.file_hash(run / q / "trace.json")}
            sample = {"query_id": q, "artifacts": {"trace": ref}}
            self.save(run / q / "result.json", sample)
            index.append({"query_id": q, "result": {"path": f"{q}/result.json", "sha256": CONT.json_hash(sample)},
                          "trace": ref})
        manifest = {"query_ids": ids, "task": task, "run_id": f"pilot-{profile}",
                    "run_fingerprint": "fingerprint", "data": {"task": task}}
        result = {"query_ids": ids, "task": task, "run_id": f"pilot-{profile}", "counts": {"completed": len(ids)},
                  "samples": index, "data": {"task": task}, "code": {"source_tree_fingerprint": "source"}}
        self.save(run / "run-manifest.json", manifest)
        self.save(run / "task-result.json", result)
        self.save(run / "_COMPLETE", {"run_id": result["run_id"], "run_fingerprint": "fingerprint",
                                      "task_result_sha256": CONT.json_hash(result)})
        return run

    def prepare(self):
        with redirect_stdout(io.StringIO()):
            CONT.prepare(self.pilot, self.bundle, self.repo)
        return CONT.read_json(self.bundle / "continuation.json")

    def test_selects_only_remaining_80_and_preserves_original_order(self):
        _, profiles, tasks, completed, expected = CONT.remaining_plan(self.pilot)
        self.assertEqual(tasks, self.tasks[2:])
        self.assertEqual(profiles, self.profiles)
        self.assertEqual(sum(x["samples"] for x in completed), 40)
        self.assertEqual(expected, self.plan["execution_order"][4:])
        self.assertEqual(CONT.execution_order(tasks, profiles, "alternating"), expected)

    def test_odd_completed_prefix_preserves_alternating_order(self):
        for profile in self.profiles:
            self.complete(self.tasks[2], profile)
        _, profiles, tasks, _, expected = CONT.remaining_plan(self.pilot)
        self.assertEqual(profiles, self.profiles[::-1])
        self.assertEqual(CONT.execution_order(tasks, profiles, "alternating"), expected)

    def test_completed_artifact_tampering_is_rejected(self):
        run = self.evaluation / "runs/fixture/Task0/runs/pilot-budgeted-stable"
        (run / "Q0/trace.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            CONT.remaining_plan(self.pilot)

    def test_half_completed_pair_is_not_silently_rerun(self):
        self.complete(self.tasks[2], self.profiles[0])
        with self.assertRaisesRegex(ValueError, "partially started task"):
            CONT.remaining_plan(self.pilot)

    def test_incomplete_task_directory_is_not_silently_discarded(self):
        run = self.evaluation / "runs/fixture/Task2/runs/pilot-budgeted-stable"
        run.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "partially started task"):
            CONT.remaining_plan(self.pilot)

    def test_source_archive_digest_and_traversal_are_rejected(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        self.save(self.evaluation / "source-hashes.json", {name: "0" * 64 for name in self.sources})
        with self.assertRaisesRegex(ValueError, "source archive hash mismatch"):
            CONT.restore_sources(self.evaluation, runtime)
        self.archive({"../outside": b"bad"})
        with self.assertRaisesRegex(ValueError, "unsafe relative"):
            CONT.restore_sources(self.evaluation, runtime)

    def test_development_edits_do_not_change_frozen_runtime(self):
        manifest = self.prepare()
        runtime = self.bundle / "runtime"
        # Reproduce the exact class of failure from the two cluster runs.
        source = self.repo / "dci_bench/data/expansion_documents.py"
        source.parent.mkdir(parents=True)
        source.write_text("edited while evaluation is running\n")
        (self.repo / "pi-dci/packages/agent/dist/index.js").write_text("changed dependency\n")
        frozen = runtime / "dci_bench/data/expansion_documents.py"
        self.assertEqual(frozen.read_bytes(), self.sources["dci_bench/data/expansion_documents.py"])
        linked = runtime / "pi-dci/node_modules/@fixture/agent/dist/index.js"
        self.assertTrue(linked.resolve().is_relative_to(runtime))
        self.assertEqual(linked.read_text(), "export const value = 1;\n")
        self.assertNotEqual(linked.stat().st_ino, (self.repo / "pi-dci/packages/agent/dist/index.js").stat().st_ino)
        self.assertEqual(CONT.inventory(runtime, self.repo / "data/workspaces"), manifest["runtime_inventory"])
        self.assertEqual(manifest["planned_model_evaluations"], 80)

    def test_external_dependency_symlink_is_rejected(self):
        (self.repo / "pi-dci/node_modules/outside").symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, "symlink escapes"):
            self.prepare()

    def test_frozen_copy_tampering_fails_before_any_subprocess(self):
        self.prepare()
        source = self.bundle / "runtime/dci_bench/data/expansion_documents.py"
        source.chmod(0o644)
        source.write_text("tampered\n")
        with patch.object(CONT.subprocess, "run") as run, self.assertRaisesRegex(ValueError, "frozen runtime changed"):
            CONT.run_bundle(self.bundle, dry_run=True)
        run.assert_not_called()

    def test_dry_run_stays_in_copy_and_does_not_invoke_serving(self):
        manifest = self.prepare()
        planned = {**self.plan, "query_ids": {t: self.plan["query_ids"][t] for t in self.tasks[2:]},
                   "execution_order": self.plan["execution_order"][4:]}
        with patch.object(CONT.subprocess, "run") as run, \
             patch.object(CONT.subprocess, "check_output", return_value=json.dumps(planned)) as plan, \
             patch.object(CONT.os, "execvpe") as execute, redirect_stdout(io.StringIO()):
            CONT.run_bundle(self.bundle, dry_run=True)
        execute.assert_not_called()
        runtime = self.bundle / "runtime"
        self.assertEqual(plan.call_args.kwargs["cwd"], runtime)
        self.assertEqual(plan.call_args.kwargs["env"]["PYTHONPATH"], str(runtime))
        argv = plan.call_args.args[0]
        self.assertEqual(argv[argv.index("--tasks") + 1:], manifest["remaining_tasks"])
        self.assertTrue(all("serve_vllm" not in str(call) for call in run.call_args_list))

    def test_dry_plan_query_drift_is_rejected(self):
        self.prepare()
        planned = {**self.plan, "query_ids": {t: ["changed"] for t in self.tasks[2:]}}
        with patch.object(CONT.subprocess, "run"), \
             patch.object(CONT.subprocess, "check_output", return_value=json.dumps(planned)), \
             self.assertRaisesRegex(ValueError, "changed frozen query IDs"):
            CONT.run_bundle(self.bundle, dry_run=True)

    def test_reversed_profile_execution_order_is_rejected(self):
        planned = {**self.plan, "query_ids": {t: self.plan["query_ids"][t] for t in self.tasks[2:]},
                   "execution_order": CONT.execution_order(self.tasks[2:], self.profiles[::-1], "alternating")}
        with self.assertRaisesRegex(ValueError, "changed paired execution order"):
            CONT.check_plan(self.plan, planned, self.tasks[2:], self.plan["execution_order"][4:])

    def test_existing_pilot_output_stops_before_model_launch(self):
        self.prepare()
        planned = {**self.plan, "query_ids": {t: self.plan["query_ids"][t] for t in self.tasks[2:]},
                   "execution_order": self.plan["execution_order"][4:]}
        with patch.object(CONT.subprocess, "run"), \
             patch.object(CONT.subprocess, "check_output", return_value=json.dumps(planned)), \
             patch.object(CONT.os, "execvpe") as execute, \
             patch.dict(os.environ, {"PILOT_ROOT": str(self.pilot)}), \
             redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(FileExistsError, "refusing to reuse pilot output"):
            CONT.run_bundle(self.bundle)
        execute.assert_not_called()

    def test_existing_bundle_is_never_overwritten(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
