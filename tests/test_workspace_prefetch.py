"""CPU-only checks for bounded query workspace prefetch and cleanup."""

from __future__ import annotations

import hashlib
from concurrent.futures import Future
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import anyio
from inspect_ai.util import LimitExceededError

from dci_bench.data.document_runtime import (
    WorkspacePrefetchConfig,
    WorkspacePrefetchError,
    WorkspacePrefetcher,
    materialize_query_workspace,
    workspace_prefetch_config,
)
from dci_bench.tasks.mteb_llm_retrieval import pi_dci_solver, workspace_prefetch_lifecycle


class WorkspacePrefetchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus = self.root / "source-corpus"
        self.corpus.mkdir()
        inventory = {}
        for index in range(1, 7):
            path = self.corpus / f"d{index:06d}.md"
            path.write_text(f"body-{index}\n", encoding="utf-8")
            inventory[path.stem] = {
                "alias": path.stem,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        self.pool = {
            "_runtime_corpus": str(self.corpus),
            "_runtime_documents": inventory,
            "pools": {
                "Q1": {"candidate_doc_ids": ["d000001", "d000002"]},
                "Q2": {"candidate_doc_ids": ["d000003", "d000004"]},
                "Q3": {"candidate_doc_ids": ["d000002", "d000005"]},
                "Q4": {"candidate_doc_ids": ["d000006"]},
            },
        }
        self.shared = self.root / "workspaces" / "Task"
        self.shared.mkdir(parents=True)

    def test_prefetch_is_disabled_by_default_and_legacy_materialization_stays_isolated(self) -> None:
        config = workspace_prefetch_config(2, environ={})
        self.assertFalse(config.enabled)
        with materialize_query_workspace(self.shared, "Q1", self.pool) as workspace:
            self.assertNotEqual(workspace, self.shared)
            self.assertEqual(
                {path.stem for path in (workspace / "corpus").iterdir()},
                {"d000001", "d000002"},
            )
        self.assertFalse(workspace.exists())

    def test_prefetch_caps_active_plus_one_batch_and_keeps_exact_ids(self) -> None:
        async def exercise() -> None:
            manager = WorkspacePrefetcher(
                self.shared,
                ["Q1", "Q2", "Q3", "Q4"],
                self.pool,
                WorkspacePrefetchConfig(True, 2, 2, None),
            )
            first = await manager.acquire("Q1")
            second = await manager.acquire("Q2")
            self.assertEqual(manager.active_count, 2)
            self.assertLessEqual(manager.inflight_count, 4)
            self.assertEqual(
                {path.stem for path in (first.path / "corpus").iterdir()},
                {"d000001", "d000002"},
            )
            self.assertEqual(
                {path.stem for path in (second.path / "corpus").iterdir()},
                {"d000003", "d000004"},
            )
            self.assertNotEqual(first.path, second.path)
            await manager.release(first)
            await manager.release(second)
            third = await manager.acquire("Q3")
            await manager.release(third)
            fourth = await manager.acquire("Q4")
            await manager.release(fourth)
            self.assertEqual(manager.inflight_count, 0)

        anyio.run(exercise)

    def test_concurrent_queries_get_independent_workspaces(self) -> None:
        async def exercise() -> None:
            manager = WorkspacePrefetcher(
                self.shared,
                ["Q1", "Q2"],
                self.pool,
                WorkspacePrefetchConfig(True, 2, 2, None),
            )
            ready = anyio.Event()
            release = anyio.Event()
            observed: dict[str, Path] = {}

            async def worker(query_id: str) -> None:
                workspace = await manager.acquire(query_id)
                observed[query_id] = workspace.path
                ready.set()
                await release.wait()
                await manager.release(workspace)

            async with anyio.create_task_group() as group:
                group.start_soon(worker, "Q1")
                group.start_soon(worker, "Q2")
                while len(observed) < 2:
                    await anyio.sleep(0)
                self.assertNotEqual(observed["Q1"], observed["Q2"])
                release.set()
            self.assertFalse(observed["Q1"].exists())
            self.assertFalse(observed["Q2"].exists())

        anyio.run(exercise)

    def test_tampered_body_is_rejected_before_pi(self) -> None:
        self.corpus.joinpath("d000001.md").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed since preflight"):
            with materialize_query_workspace(self.shared, "Q1", self.pool):
                self.fail("tampered source was materialized")

    def test_prefetch_rejects_same_size_target_body_tamper(self) -> None:
        original_copyfile = shutil.copyfile

        def copy_then_tamper(source: str | Path, target: str | Path) -> str:
            result = original_copyfile(source, target)
            Path(target).write_text("tamper\n", encoding="utf-8")
            return result

        async def exercise() -> None:
            manager = WorkspacePrefetcher(
                self.shared,
                ["Q1"],
                self.pool,
                WorkspacePrefetchConfig(True, 1, 1, None),
            )
            with patch(
                "dci_bench.data.document_runtime.shutil.copyfile",
                side_effect=copy_then_tamper,
            ), self.assertRaisesRegex(ValueError, "changed since preflight"):
                await manager.acquire("Q1")
            self.assertEqual(manager.inflight_count, 0)
            await manager.close()

        anyio.run(exercise)

    def test_disk_cap_has_a_clear_diagnostic(self) -> None:
        config = WorkspacePrefetchConfig(True, 2, 2, 1)

        async def exercise() -> None:
            manager = WorkspacePrefetcher(self.shared, ["Q1"], self.pool, config)
            with self.assertRaisesRegex(WorkspacePrefetchError, "DCI_WORKSPACE_PREFETCH_MAX_BYTES"):
                await manager.acquire("Q1")
            await manager.close()

        anyio.run(exercise)

    def test_exception_and_cancellation_cleanup_pending_and_active_workspaces(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with materialize_query_workspace(self.shared, "Q1", self.pool) as workspace:
                raise RuntimeError("synthetic")
        self.assertFalse(workspace.exists())

        async def exercise() -> None:
            manager = WorkspacePrefetcher(
                self.shared,
                ["Q1", "Q2", "Q3"],
                self.pool,
                WorkspacePrefetchConfig(True, 2, 2, None),
            )
            active = await manager.acquire("Q1")
            await manager.cancel_pending()
            self.assertEqual(manager.pending_count, 0)
            await manager.release(active)
            await manager.close()
            self.assertEqual(manager.inflight_count, 0)

        anyio.run(exercise)

    def test_completed_optional_workspace_is_evicted_before_next_capped_reservation(self) -> None:
        max_bytes = sum(self.pool["_runtime_documents"][alias]["bytes"] for alias in ("d000001", "d000002"))

        async def exercise() -> None:
            manager = WorkspacePrefetcher(
                self.shared,
                ["Q1", "Q2", "Q3"],
                self.pool,
                WorkspacePrefetchConfig(True, 1, 1, max_bytes),
            )
            first = await manager.acquire("Q1")
            await manager.release(first)
            with anyio.fail_after(2):
                while True:
                    with manager._condition:
                        optional = manager._futures.get("Q2")
                        if optional is not None and optional.future.done():
                            break
                    await anyio.sleep(0.01)
            third = await manager.acquire("Q3")
            self.assertEqual(third.query_id, "Q3")
            self.assertEqual(
                {path.stem for path in (third.path / "corpus").iterdir()},
                {"d000002", "d000005"},
            )
            await manager.release(third)
            await manager.close()
            self.assertEqual(manager.reserved_bytes, 0)

        anyio.run(exercise)

    def test_release_holds_reservation_until_workspace_cleanup_finishes(self) -> None:
        max_bytes = sum(self.pool["_runtime_documents"][alias]["bytes"] for alias in ("d000001", "d000002"))
        cleanup_started = threading.Event()
        allow_cleanup = threading.Event()

        async def exercise() -> None:
            manager = WorkspacePrefetcher(
                self.shared,
                ["Q1", "Q2"],
                self.pool,
                WorkspacePrefetchConfig(True, 1, 1, max_bytes),
            )
            first = await manager.acquire("Q1")
            original_cleanup = first.cleanup

            def blocked_cleanup() -> None:
                cleanup_started.set()
                self.assertTrue(allow_cleanup.wait(2))
                original_cleanup()

            first.cleanup = blocked_cleanup
            async with anyio.create_task_group() as group:
                group.start_soon(manager.release, first)
                await anyio.to_thread.run_sync(cleanup_started.wait)
                self.assertEqual(manager.reserved_bytes, max_bytes)
                self.assertEqual(manager.pending_count, 0)
                allow_cleanup.set()
            with anyio.fail_after(2):
                while manager.pending_count == 0:
                    await anyio.sleep(0.01)
            await manager.close()

        anyio.run(exercise)

    def test_cancelled_acquisition_does_not_cancel_other_samples(self) -> None:
        max_bytes = sum(self.pool["_runtime_documents"][alias]["bytes"] for alias in ("d000001", "d000002"))

        async def exercise() -> None:
            manager = WorkspacePrefetcher(
                self.shared,
                ["Q1", "Q2", "Q3"],
                self.pool,
                WorkspacePrefetchConfig(True, 1, 1, max_bytes),
            )
            first = await manager.acquire("Q1")
            with anyio.fail_after(2):
                with anyio.move_on_after(0.05) as scope:
                    await manager.acquire("Q2")
                self.assertTrue(scope.cancel_called)
            with anyio.fail_after(2):
                while True:
                    with manager._condition:
                        if manager._acquiring == 0:
                            break
                    await anyio.sleep(0.01)
            await manager.release(first)
            third = await manager.acquire("Q3")
            self.assertEqual(third.query_id, "Q3")
            await manager.release(third)
            await manager.close()
            self.assertEqual(manager.reserved_bytes, 0)

        anyio.run(exercise)

    def test_cancel_pending_releases_already_cancelled_future_once(self) -> None:
        manager = WorkspacePrefetcher(
            self.shared,
            ["Q1"],
            self.pool,
            WorkspacePrefetchConfig(True, 1, 1, None),
        )
        future = Future()
        future.cancel()
        reserved = sum(
            self.pool["_runtime_documents"][alias]["bytes"]
            for alias in self.pool["pools"]["Q1"]["candidate_doc_ids"]
        )
        with manager._condition:
            manager._futures["Q1"] = SimpleNamespace(
                query_id="Q1",
                reserved_bytes=reserved,
                future=future,
            )
            manager._submitted.add("Q1")
            manager._reserved_count = 1
            manager._reserved_bytes = reserved

        async def exercise() -> None:
            await manager.cancel_pending()
            self.assertEqual(manager.inflight_count, 0)
            self.assertEqual(manager.reserved_bytes, 0)
            self.assertEqual(manager._cleaning, 0)
            await manager.close()

        anyio.run(exercise)

    def test_partial_epoch_cleanup_and_refresh_reuses_task_manager(self) -> None:
        manager = WorkspacePrefetcher(
            self.shared,
            ["Q1"],
            self.pool,
            WorkspacePrefetchConfig(True, 1, 1, None),
        )
        fake_task = SimpleNamespace(_dci_workspace_prefetcher=manager)
        paths: list[Path] = []

        async def acquire_without_release() -> None:
            prepared = await manager.acquire("Q1")
            paths.append(prepared.path)

        with workspace_prefetch_lifecycle(fake_task):
            anyio.run(acquire_without_release)
        self.assertFalse(paths[0].exists())
        self.assertEqual(manager.inflight_count, 0)

        async def acquire_and_release() -> None:
            prepared = await manager.acquire("Q1")
            paths.append(prepared.path)
            await manager.release(prepared)

        with workspace_prefetch_lifecycle(fake_task):
            anyio.run(acquire_and_release)
        self.assertFalse(paths[1].exists())
        self.assertEqual(manager.inflight_count, 0)

    def test_bridge_failure_is_cleaned_by_evaluation_lifecycle(self) -> None:
        manager = WorkspacePrefetcher(
            self.shared,
            ["Q1", "Q2"],
            self.pool,
            WorkspacePrefetchConfig(True, 1, 1, None),
        )
        fake_task = SimpleNamespace(_dci_workspace_prefetcher=manager)

        @asynccontextmanager
        async def failing_bridge():
            raise RuntimeError("bridge startup failed")
            yield object()

        async def exercise() -> None:
            prepared = await manager.acquire("Q1")
            await manager.release(prepared)
            with patch(
                "dci_bench.tasks.mteb_llm_retrieval.sandbox_agent_bridge",
                return_value=failing_bridge(),
            ):
                state = SimpleNamespace(
                    metadata={
                        "task": "Task",
                        "query_id": "Q2",
                        "workspace": str(self.shared),
                    },
                    input_text="query",
                    output=SimpleNamespace(),
                )
                solve = pi_dci_solver(
                    output_dir=str(self.root / "results"),
                    candidate_pool=self.pool,
                    workspace_prefetcher=manager,
                )
                with self.assertRaisesRegex(RuntimeError, "bridge startup failed"):
                    await solve(state, None)

        with workspace_prefetch_lifecycle(fake_task):
            anyio.run(exercise)
        self.assertEqual(manager.inflight_count, 0)

    def test_preparation_exhausting_deadline_skips_pi_and_releases_workspace(self) -> None:
        manager = WorkspacePrefetcher(
            self.shared,
            ["Q1"],
            self.pool,
            WorkspacePrefetchConfig(True, 1, 1, None),
        )

        @asynccontextmanager
        async def bridge():
            yield object()

        async def exercise() -> None:
            state = SimpleNamespace(
                metadata={"task": "Task", "query_id": "Q1", "workspace": str(self.shared)},
                input_text="query",
                output=SimpleNamespace(),
            )
            solve = pi_dci_solver(
                output_dir=str(self.root / "results"),
                candidate_pool=self.pool,
                workspace_prefetcher=manager,
            )
            with (
                patch(
                    "dci_bench.tasks.mteb_llm_retrieval.sandbox_agent_bridge",
                    return_value=bridge(),
                ),
                patch(
                    "dci_bench.tasks.mteb_llm_retrieval._remaining_pi_timeout_seconds",
                    side_effect=LimitExceededError("time", value=60, limit=60),
                ),
                patch("dci_bench.tasks.mteb_llm_retrieval.run_pi_dci") as run_pi,
            ):
                with self.assertRaises(LimitExceededError):
                    await solve(state, None)
            run_pi.assert_not_called()
            self.assertEqual(manager.active_count, 0)
            self.assertEqual(manager.inflight_count, 0)

        anyio.run(exercise)


if __name__ == "__main__":
    unittest.main()
