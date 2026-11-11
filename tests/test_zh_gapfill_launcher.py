from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_gapfill_5090_4gpu as launcher


ZH_SOURCES = ("gapfill_t2ranking_software_zh", "gapfill_cfqa_enterprise_zh")


class ZhGapfillLauncherTests(unittest.TestCase):
    def test_old_defaults_and_explicit_chinese_source_selection(self):
        self.assertEqual(launcher.DEFAULT_SOURCES,
                         ("gapfill_browsecomp_plus_en", "gapfill_cedar_zh_v2"))
        root = "/mnt/afs/250010100/zxp/dci-bench-zh-data/20261102_v1/run"
        args = launcher.build_parser().parse_args([
            "--run-root", root, "--source", ZH_SOURCES[0], "--source", ZH_SOURCES[1],
            "--check-only",
        ])
        self.assertEqual(args.run_root, Path(root))
        self.assertEqual(args.sources, list(ZH_SOURCES))
        self.assertTrue(args.check_only)
        for source in ZH_SOURCES:
            spec = launcher.SOURCE_SPECS[source]
            self.assertEqual(spec.batch_size, 8)
            self.assertFalse(spec.exclude_self)
            self.assertEqual((spec.receipt_name, spec.ranking_name, spec.raw_ranking_name),
                             ("catalog-row.json", "ranking.json", "ranking.json"))
            self.assertEqual(spec.pool_names, ("pools-200", "pools-1000"))

    def test_plan_is_read_only_and_commands_reproduce_actual_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            run = root / "new-run"
            state = root / "new-state"
            reader = SimpleNamespace(document_count=4097, query_count=60,
                                     texts_for_shard=lambda _index: [])
            before = sorted(path.relative_to(root) for path in root.rglob("*"))
            with (
                patch.object(launcher, "completed_source_status", return_value=None),
                patch.object(launcher, "_validate_prepared_metadata",
                             return_value={"num_docs": 4097, "num_queries": 60}),
                patch.object(launcher, "_prepared_shard_reader", return_value=reader),
                patch.object(launcher, "_model_weight_identity", return_value="fixture-weights"),
                patch.object(launcher, "_validate_canonical_shards", return_value=[]),
                patch.object(launcher, "runtime_provenance", return_value={}),
                patch.object(launcher, "_gpu_preflight", side_effect=AssertionError("GPU in plan")),
                patch.object(launcher, "_encoder_for", side_effect=AssertionError("encode in plan")),
                patch.object(launcher, "write_json_new", side_effect=AssertionError("write in plan")),
            ):
                plan = launcher.build_plan(run_root=run, state_root=state,
                                           source_names=ZH_SOURCES,
                                           visible_devices=["4", "5", "6", "7"],
                                           model_path=str(model))
            self.assertEqual(before, sorted(path.relative_to(root) for path in root.rglob("*")))
            self.assertEqual(plan["source_names"], list(ZH_SOURCES))
            for command in (plan["controller_command"], plan["check_only_command"],
                            plan["shell_command"]):
                parsed = launcher.build_parser().parse_args(command[2:])
                self.assertEqual((parsed.run_root, parsed.state_root, parsed.model_path),
                                 (run, state, str(model)))
                self.assertEqual(parsed.sources, list(ZH_SOURCES))
            self.assertTrue(launcher.build_parser().parse_args(
                plan["check_only_command"][2:]).check_only)
            actual = [(task["source"], task["shard"])
                      for worker in plan["workers"] for task in worker["tasks"]]
            expected = [(source, shard) for source in ZH_SOURCES for shard in range(5)]
            self.assertCountEqual(actual, expected)
            self.assertEqual(len(actual), len(set(actual)))
            caches = {worker["cache_root"] for worker in plan["workers"]}
            self.assertEqual(len(caches), 4)
            self.assertTrue(all(worker["tasks"] for worker in plan["workers"]))
            for entry in plan["sources"]:
                self.assertEqual(entry["suite"], str(run / "source"))
                self.assertEqual(entry["canonical_cache_dir"],
                                 str(run / entry["source"] / "embedding-cache"))
                self.assertNotIn(entry["canonical_cache_dir"], caches)

    def test_check_only_and_dry_run_environment_skip_full_execution(self):
        for flag, value in ((["--check-only"], "0"), ([], "1")):
            with (
                self.subTest(flag=flag, dry_run=value),
                patch.dict(os.environ, {"DRY_RUN": value, "CUDA_VISIBLE_DEVICES": "4,5,6,7"}),
                patch.object(launcher, "_validate_python_environment"),
                patch.object(launcher, "build_plan", return_value={}) as build,
                patch.object(launcher, "_print_plan") as output,
                patch.object(launcher, "run_full", side_effect=AssertionError("full execution in dry-run")),
            ):
                self.assertEqual(launcher.main([
                    "--run-root", "/tmp/zh-data/run", "--source", ZH_SOURCES[0],
                    "--source", ZH_SOURCES[1], *flag,
                ]), 0)
                self.assertEqual(build.call_args.kwargs["run_root"], Path("/tmp/zh-data/run"))
                self.assertEqual(build.call_args.kwargs["source_names"], ZH_SOURCES)
                self.assertEqual(build.call_args.kwargs["visible_devices"], ["4", "5", "6", "7"])
                output.assert_called_once_with({})

    def test_dry_run_zero_dispatches_full_mode_with_old_default_sources(self):
        with (
            patch.dict(os.environ, {"DRY_RUN": "0", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}),
            patch.object(launcher, "_validate_python_environment"),
            patch.object(launcher, "build_plan", return_value={}) as build,
            patch.object(launcher, "run_full", return_value={"status": "fixture"}) as full,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(launcher.main([]), 0)
            self.assertEqual(build.call_args.kwargs["source_names"], launcher.DEFAULT_SOURCES)
            full.assert_called_once_with({}, visible_devices=["0", "1", "2", "3"])

    def test_real_execution_gate_smokes_before_workers_and_sequential_merge(self):
        for returncode in (0, 1):
            with self.subTest(smoke_returncode=returncode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                events = []
                entries = [{"source": source, "status": "pending", "missing_shards": [0]}
                           for source in ZH_SOURCES]
                plan = {"run_root": str(root / "run"), "state_root": str(root / "state"),
                        "sources": entries,
                        "workers": [{"worker_index": index, "tasks":
                                     [{"source": ZH_SOURCES[index], "shard": 0}] if index < 2 else []}
                                    for index in range(4)]}

                def smoke(command, **kwargs):
                    events.append("smoke")
                    self.assertIn("--smoke", command)
                    self.assertEqual(command[-1], ZH_SOURCES[0])
                    kwargs["stdout"].write(json.dumps({"inference_executed": True}) + "\n")
                    return SimpleNamespace(returncode=returncode)

                with (
                    patch.object(launcher, "_validate_python_environment"),
                    patch.object(launcher, "_gpu_preflight",
                                 side_effect=lambda _slots: events.append("gpu-preflight") or []),
                    patch.object(launcher, "runtime_provenance", return_value={}),
                    patch.object(launcher.subprocess, "run", side_effect=smoke),
                    patch.object(launcher, "_launch_workers",
                                 side_effect=lambda *_args: events.append("workers") or {}),
                    patch.object(launcher, "merge_document_shards",
                                 side_effect=lambda entry, *_args, **_kwargs:
                                 events.append("merge:" + entry["source"]) or {}),
                    patch.object(launcher, "_finalize_source",
                                 side_effect=lambda _plan, entry, **_kwargs:
                                 events.append("finalize:" + entry["source"]) or {}),
                ):
                    if returncode:
                        with self.assertRaisesRegex(RuntimeError, "real encoder smoke failed"):
                            launcher.run_full(plan, visible_devices=["0", "1", "2", "3"])
                        self.assertEqual(events, ["gpu-preflight", "smoke"])
                    else:
                        result = launcher.run_full(plan, visible_devices=["0", "1", "2", "3"])
                        self.assertEqual(result["status"], "complete")
                        self.assertEqual(events, ["gpu-preflight", "smoke", "workers",
                                                 *["merge:" + source for source in ZH_SOURCES],
                                                 *["finalize:" + source for source in ZH_SOURCES]])
                self.assertFalse((root / "run").exists())


if __name__ == "__main__":
    unittest.main()
