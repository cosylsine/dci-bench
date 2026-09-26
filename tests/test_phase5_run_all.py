import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dci_bench.data.registry import TASK_NAMES, TASKS


def load_run_all_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_all.py"
    spec = importlib.util.spec_from_file_location("run_all_phase5", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Phase5RegistryTest(unittest.TestCase):
    def test_registry_contains_the_six_pinned_tasks_in_canonical_order(self):
        self.assertEqual(
            TASK_NAMES,
            (
                "LLMAILAStatutes",
                "LLMFQuADRetrieval",
                "LLMHC3FinanceRetrieval",
                "LLMLegalBenchConsumerContractsQA",
                "LLMPublicHealthQA",
                "LLMTwitterHjerneRetrieval",
            ),
        )
        self.assertEqual(len(TASKS), 6)
        self.assertTrue(all(spec.task_instruction.strip() for spec in TASKS.values()))
        self.assertEqual(
            {name: spec.revision for name, spec in TASKS.items()},
            {
                "LLMAILAStatutes": "a2acf12d293e",
                "LLMFQuADRetrieval": "dc5443dbfad5",
                "LLMHC3FinanceRetrieval": "8733760a6f3e",
                "LLMLegalBenchConsumerContractsQA": "642870c78f65",
                "LLMPublicHealthQA": "b05938525381",
                "LLMTwitterHjerneRetrieval": "31f9b918c30e",
            },
        )


class Phase5RunAllTest(unittest.TestCase):
    def _args(self, module, root: Path):
        manifest = root / "backend.json"
        manifest.write_text("{}\n", encoding="utf-8")
        args = module.parse_args(
            [
                "--backend-manifest",
                str(manifest),
                "--results-root",
                str(root / "results"),
                "--workspace-root",
                str(root / "workspaces"),
                "--metadata-root",
                str(root / "metadata"),
            ]
        )
        for task_name in TASK_NAMES:
            spec = TASKS[task_name]
            (args.workspace_root / task_name / "corpus").mkdir(parents=True)
            metadata = args.metadata_root / task_name
            metadata.mkdir(parents=True)
            (metadata / "manifest.json").write_text(
                json.dumps(
                    {
                        "task": task_name,
                        "dataset": spec.hf_dataset,
                        "revision": spec.revision,
                        "resolved_revision": spec.revision + "-resolved",
                        "num_queries": 1,
                        "num_docs": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (metadata / "queries.jsonl").write_text(
                '{"_id":"Q1","text":"query"}\n', encoding="utf-8"
            )
            (metadata / "qrels.json").write_text(
                '{"Q1":{"D1":1}}\n', encoding="utf-8"
            )
            (metadata / "doc_id_files.json").write_text(
                '{"D1":"D1.md"}\n', encoding="utf-8"
            )
            (args.workspace_root / task_name / "corpus" / "D1.md").write_text(
                "document\n", encoding="utf-8"
            )
        return args

    def test_run_all_invokes_every_task_in_registry_order(self):
        module = load_run_all_module()
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def fake_run(command, **kwargs):
                calls.append((command, kwargs))
                return SimpleNamespace(returncode=0)

            exit_code, records = module.run_all(
                self._args(module, Path(tmp)), run_command=fake_run
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual([record["task"] for record in records], list(TASK_NAMES))
        invoked_tasks = [command[command.index("--task") + 1] for command, _ in calls]
        self.assertEqual(invoked_tasks, list(TASK_NAMES))
        self.assertTrue(all("--all-queries" in command for command, _ in calls))
        invoked_ports = [
            int(command[command.index("--bridge-port") + 1]) for command, _ in calls
        ]
        self.assertEqual(invoked_ports, [18891 + index for index in range(6)])

    def test_task_bridge_port_ranges_are_disjoint_for_concurrent_runs(self):
        module = load_run_all_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(module, Path(tmp))
            args.max_concurrency = 4
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0)

            exit_code, _ = module.run_all(args, run_command=fake_run)

        self.assertEqual(exit_code, 0)
        ports = [int(command[command.index("--bridge-port") + 1]) for command in calls]
        self.assertEqual(ports, [18891, 18895, 18899, 18903, 18907, 18911])

    def test_task_subset_keeps_registry_port_ranges(self):
        module = load_run_all_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(module, Path(tmp))
            args.tasks = ["LLMPublicHealthQA", "LLMAILAStatutes"]
            args.max_concurrency = 2
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0)

            exit_code, records = module.run_all(args, run_command=fake_run)

        self.assertEqual(exit_code, 0)
        self.assertEqual([item["task"] for item in records], args.tasks)
        self.assertEqual(
            [int(command[command.index("--bridge-port") + 1]) for command in calls],
            [18899, 18891],
        )

    def test_duplicate_task_subset_is_rejected(self):
        module = load_run_all_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(module, Path(tmp))
            args.tasks = ["LLMPublicHealthQA", "LLMPublicHealthQA"]
            with self.assertRaisesRegex(ValueError, "duplicate"):
                module.run_all(args, run_command=lambda *a, **k: None)

    def test_all_bridge_port_ranges_are_validated_before_launch(self):
        module = load_run_all_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(module, Path(tmp))
            args.bridge_port = 65531
            calls = []
            with self.assertRaisesRegex(ValueError, "exceed TCP port 65535"):
                module.run_all(
                    args,
                    run_command=lambda command, **kwargs: calls.append(command),
                )
        self.assertEqual(calls, [])

    def test_sample_failures_continue_but_configuration_failure_stops(self):
        module = load_run_all_module()
        with tempfile.TemporaryDirectory() as tmp:
            return_codes = iter((1, 0, 2, 0, 0, 0))

            def fake_run(_command, **_kwargs):
                return SimpleNamespace(returncode=next(return_codes))

            exit_code, records = module.run_all(
                self._args(module, Path(tmp)), run_command=fake_run
            )

        self.assertEqual(exit_code, 2)
        self.assertEqual(len(records), 3)
        self.assertEqual([record["exit_code"] for record in records], [1, 0, 2])

    def test_preflight_rejects_bad_revision_before_any_task_runs(self):
        module = load_run_all_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(module, Path(tmp))
            manifest_path = args.metadata_root / TASK_NAMES[-1] / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["resolved_revision"] = "wrong"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            calls = []

            def fake_run(command, **kwargs):
                calls.append((command, kwargs))
                return SimpleNamespace(returncode=0)

            with self.assertRaisesRegex(ValueError, "does not match pinned"):
                module.run_all(args, run_command=fake_run)
            self.assertEqual(calls, [])

    def test_preflight_only_does_not_require_backend_manifest(self):
        module = load_run_all_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(module, Path(tmp))
            args.backend_manifest = None
            args.preflight_only = True
            with patch.object(module, "parse_args", return_value=args):
                with patch.object(module, "run_all") as run_all_mock:
                    self.assertEqual(module.main([]), 0)
            run_all_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
