import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anyio


try:
    from dci_bench.tasks.mteb_llm_retrieval import (
        BridgePortPool,
        _make_samples,
        _select_query_ids,
        _serialized_bridge_start,
    )
except (ImportError, OSError):
    BridgePortPool = None
    _make_samples = None
    _select_query_ids = None
    _serialized_bridge_start = None


def load_run_task_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_task.py"
    spec = importlib.util.spec_from_file_location("run_task_phase4", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(_make_samples is not None, "a working inspect_ai environment is required")
class Phase4BatchTaskTest(unittest.TestCase):
    def _metadata(self, root: Path) -> Path:
        metadata = root / "metadata" / "Task"
        metadata.mkdir(parents=True)
        (metadata / "queries.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"_id": "Q2", "text": "second"}),
                    json.dumps({"_id": "Q1", "text": "first"}),
                    json.dumps({"_id": "Q3", "text": "third"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (metadata / "qrels.json").write_text(
            json.dumps({"Q1": {"D1": 1}, "Q2": {"D2": 1}, "Q3": {"D3": 1}}),
            encoding="utf-8",
        )
        return root / "metadata"

    def test_all_queries_follows_queries_jsonl_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = self._metadata(root)
            samples = _make_samples(
                "Task",
                None,
                metadata,
                root / "workspaces",
                all_queries=True,
            )
            self.assertEqual([sample.id for sample in samples], ["Q2", "Q1", "Q3"])
            self.assertTrue(all("bridge_port" not in sample.metadata for sample in samples))
            self.assertTrue(all(sample.target == "" for sample in samples))

    def test_explicit_selection_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = self._metadata(Path(tmp))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                _select_query_ids(
                    "Task", metadata_root=metadata, query_ids=["Q1", "Q1"]
                )
            with self.assertRaisesRegex(ValueError, "Unknown"):
                _select_query_ids(
                    "Task", metadata_root=metadata, query_ids=["Q9"]
                )
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                _select_query_ids(
                    "Task",
                    metadata_root=metadata,
                    query_ids=["Q1"],
                    all_queries=True,
                )

    def test_port_lease_reuses_only_after_out_of_order_release(self):
        async def exercise():
            pool = BridgePortPool(13131, 2)
            first_ready = anyio.Event()
            second_ready = anyio.Event()
            release_first = anyio.Event()
            observed = {}

            async def first_worker():
                async with pool.lease() as port:
                    observed["first"] = port
                    first_ready.set()
                    await release_first.wait()

            async def second_worker():
                async with pool.lease() as port:
                    observed["second"] = port
                    second_ready.set()
                    # This worker completes while the first lease is active.
                    await anyio.sleep(0.01)

            async def third_worker():
                await first_ready.wait()
                await second_ready.wait()
                async with pool.lease() as port:
                    observed["third"] = port
                    # Keep the third lease alive long enough to overlap the
                    # first lease; it must not receive the first port.
                    await anyio.sleep(0.01)

            async with anyio.create_task_group() as group:
                group.start_soon(first_worker)
                await first_ready.wait()
                group.start_soon(second_worker)
                await second_ready.wait()
                group.start_soon(third_worker)
                while "third" not in observed:
                    await anyio.sleep(0)
                release_first.set()

            self.assertNotEqual(observed["first"], observed["second"])
            self.assertEqual(observed["second"], observed["third"])

        anyio.run(exercise)

    def test_bridge_startup_is_serialized_while_sample_bodies_overlap(self):
        async def exercise():
            pool = BridgePortPool(13131, 2)
            active_enters = 0
            max_enters = 0
            active_bodies = 0
            max_bodies = 0

            @asynccontextmanager
            async def bridge_context():
                nonlocal active_enters, max_enters, active_bodies, max_bodies
                active_enters += 1
                max_enters = max(max_enters, active_enters)
                await anyio.sleep(0.02)
                active_enters -= 1
                active_bodies += 1
                max_bodies = max(max_bodies, active_bodies)
                try:
                    yield object()
                finally:
                    active_bodies -= 1

            async def worker():
                async with _serialized_bridge_start(pool, bridge_context()):
                    await anyio.sleep(0.05)

            async with anyio.create_task_group() as group:
                group.start_soon(worker)
                group.start_soon(worker)

            self.assertEqual(max_enters, 1)
            self.assertEqual(max_bodies, 2)

        anyio.run(exercise)


@unittest.skipUnless(_make_samples is not None, "a working inspect_ai environment is required")
class Phase4BatchCliTest(unittest.TestCase):
    def test_cli_invokes_inspect_eval_once_for_multiple_queries(self):
        module = load_run_task_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata" / "LLMPublicHealthQA"
            metadata.mkdir(parents=True)
            (metadata / "queries.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"_id": "Q2", "text": "two"}),
                        json.dumps({"_id": "Q1", "text": "one"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (metadata / "qrels.json").write_text(
                json.dumps({"Q1": {"D1": 1}, "Q2": {"D2": 1}}),
                encoding="utf-8",
            )
            args = module.parse_args(
                [
                    "--model-key",
                    "MiniCPM5-2B",
                    "--task",
                    "LLMPublicHealthQA",
                    "--query-ids",
                    "Q2",
                    "Q1",
                    "--metadata-root",
                    str(root / "metadata"),
                    "--workspace-root",
                    str(root / "workspaces"),
                    "--results-root",
                    str(root / "results"),
                    "--run-id",
                    "run-1",
                    "--max-concurrency",
                    "2",
                ]
            )
            calls = []

            def fake_eval(task, **kwargs):
                calls.append((task, kwargs))
                return [SimpleNamespace(location="logs/batch.eval")]

            with patch.object(module, "mteb_llm_retrieval", return_value=object()) as make_task:
                invocation = module.run_batch(args, inspect_eval_fn=fake_eval)
            self.assertEqual(len(calls), 1)
            self.assertEqual(invocation.query_ids, ("Q2", "Q1"))
            self.assertEqual(calls[0][1]["max_samples"], 2)
            self.assertEqual(calls[0][1]["max_connections"], 2)
            self.assertEqual(make_task.call_args.kwargs["query_ids"], ["Q2", "Q1"])
            self.assertEqual(make_task.call_args.kwargs["model_context_window"], 128000)
            self.assertIn(
                "public-health questions",
                make_task.call_args.kwargs["task_instruction"],
            )


class Phase4BatchCliOfflineTest(unittest.TestCase):
    """Exercise the one-eval CLI boundary without installing Inspect."""

    def test_cli_invokes_injected_eval_once(self):
        module = load_run_task_module()
        original_selector = module._select_query_ids
        original_task = module.mteb_llm_retrieval
        try:
            module._select_query_ids = lambda *args, **kwargs: ["Q2", "Q1"]
            module.mteb_llm_retrieval = lambda **kwargs: kwargs
            with tempfile.TemporaryDirectory() as results_root:
                args = module.parse_args(
                    [
                        "--task",
                        "LLMPublicHealthQA",
                        "--query-ids",
                        "Q2",
                        "Q1",
                        "--results-root",
                        results_root,
                        "--run-id",
                        "offline-run",
                        "--max-concurrency",
                        "2",
                    ]
                )
                calls = []

                def fake_eval(task, **kwargs):
                    calls.append((task, kwargs))
                    return []

                invocation = module.run_batch(args, inspect_eval_fn=fake_eval)
                self.assertEqual(len(calls), 1)
                self.assertEqual(invocation.query_ids, ("Q2", "Q1"))
                self.assertEqual(calls[0][1]["max_samples"], 2)
                self.assertEqual(calls[0][1]["max_connections"], 2)
        finally:
            module._select_query_ids = original_selector
            module.mteb_llm_retrieval = original_task


if __name__ == "__main__":
    unittest.main()
