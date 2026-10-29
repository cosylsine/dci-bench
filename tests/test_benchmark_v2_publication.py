"""CPU coverage for query-level publication and v2 resume behavior."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dci_bench.protocol.contracts import BENCHMARK_CONTRACT_VERSION, protocol_contract
from dci_bench.results import phase4_runner
from dci_bench.results.incremental import IncrementalPublicationError, IncrementalPublicationHook
from dci_bench.results.protocol import ResultIntegrityError, sample_directory_name

def _metrics() -> dict[str, float]:
    return {
        f"{family}_at_{cutoff}": 1.0
        for family in ("recall", "f1", "ndcg")
        for cutoff in (1, 3, 5, 10, 20)
    }


class BenchmarkV2PublicationTest(unittest.TestCase):
    def _fixture(self, root: Path):
        from importlib import import_module

        helper = import_module("tests.test_phase4_runner").Phase4RunnerTest()
        args = helper._fixture(root, query_id="Q1")
        metadata = args.metadata_root / args.task
        (metadata / "queries.jsonl").write_text(
            json.dumps({"_id": "Q1", "text": "query one"})
            + "\n"
            + json.dumps({"_id": "Q2", "text": "query two"})
            + "\n",
            encoding="utf-8",
        )
        (metadata / "qrels.json").write_text(
            json.dumps({"Q1": {"D1": 1}, "Q2": {"D1": 1}}) + "\n",
            encoding="utf-8",
        )
        return args

    @staticmethod
    def _contract(profile: str) -> dict[str, object]:
        contract = protocol_contract(
            task_name="LLMPublicHealthQA",
            task_instruction="Return ranked public-health documents.",
        )
        contract["protocol_profile"] = profile
        return contract

    @staticmethod
    def _sample(query_id: str) -> SimpleNamespace:
        usage = {
            "openai-api/sglang/MiniCPM5-2B": SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                input_tokens_cache_read=None,
                input_tokens_cache_write=None,
            )
        }
        return SimpleNamespace(
            id=query_id,
            model_usage=usage,
            scores={"dci_retrieval_scorer": SimpleNamespace(value=_metrics())},
            started_at="2026-09-15T00:00:00Z",
            completed_at="2026-09-15T00:00:02Z",
            total_time=2.0,
            working_time=1.5,
            error=None,
        )

    def _write_sample(self, task: dict[str, object], query_id: str) -> SimpleNamespace:
        output = Path(str(task["output_dir"]))
        sample_dir = output / sample_directory_name(query_id)
        sample_dir.mkdir(parents=True, exist_ok=True)
        usage_summary = {
            "available": True,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "turns": 1,
        }
        (sample_dir / "final.json").write_text(
            json.dumps(
                {
                    "contract_version": BENCHMARK_CONTRACT_VERSION,
                    "valid_output": True,
                    "failure_reason": None,
                    "failure": None,
                    "ranked_doc_ids": ["D1"],
                    "agent_steps": 1,
                    "tool_calls": 0,
                    "repair_attempts": 0,
                    "repair_reasons": [],
                    "usage_summary": usage_summary,
                    "inspect_sandbox": "local",
                    "tool_sandbox": "bubblewrap",
                }
            ),
            encoding="utf-8",
        )
        (sample_dir / "trace.json").write_text(
            json.dumps(
                {
                    "events": [],
                    "messages": [],
                    "usage_summary": usage_summary,
                    "usage_by_turn": [],
                }
            ),
            encoding="utf-8",
        )
        return self._sample(query_id)

    @staticmethod
    def _emit_sample(sample: SimpleNamespace) -> None:
        from inspect_ai.hooks._hooks import emit_sample_end, get_all_hooks

        assert any(isinstance(hook, IncrementalPublicationHook) for hook in get_all_hooks())
        asyncio.run(
            emit_sample_end(
                eval_set_id=None,
                run_id="inspect-run",
                eval_id="inspect-eval",
                sample_id="inspect-sample-uuid",
                sample=sample,
            )
        )

    @staticmethod
    def _task_factory(**kwargs):
        return kwargs

    @staticmethod
    def _stable_runner_patches(profile: str):
        return (
            patch.object(
                phase4_runner,
                "protocol_contract",
                return_value=BenchmarkV2PublicationTest._contract(profile),
            ),
            patch.object(phase4_runner, "code_audit", return_value={"test": "publication"}),
            patch.object(phase4_runner, "preflight_bridge_ports"),
        )

    def _interrupted_v2_run(self, root: Path):
        args = self._fixture(root)
        run_dir = root / "results" / "MiniCPM5-2B" / "LLMPublicHealthQA" / "runs" / "test-run"

        def interrupted_eval(task, **_kwargs):
            self._emit_sample(self._write_sample(task, "Q1"))
            raise KeyboardInterrupt("midbatch")

        patches = self._stable_runner_patches("document-pool-v2-lite")
        with patches[0], patches[1], patches[2]:
            with self.assertRaises(KeyboardInterrupt):
                phase4_runner.execute_phase4(
                    args,
                    selected_query_ids=["Q1", "Q2"],
                    inspect_eval_fn=interrupted_eval,
                    task_factory=self._task_factory,
                )
        self.assertTrue((run_dir / sample_directory_name("Q1") / "_COMPLETE").is_file())
        return args, run_dir

    def test_v2_publishes_finished_query_before_interrupt_and_resumes_only_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._fixture(root)
            run_dir = root / "results" / "MiniCPM5-2B" / "LLMPublicHealthQA" / "runs" / "test-run"
            calls: list[tuple[str, ...]] = []

            def interrupted_eval(task, **_kwargs):
                self.assertEqual(_kwargs["max_retries"], 0)
                self.assertEqual(_kwargs["model_args"]["max_retries"], 0)
                calls.append(tuple(task["query_ids"]))
                sample = self._write_sample(task, "Q1")
                self._emit_sample(sample)
                raise KeyboardInterrupt("midbatch")

            patches = self._stable_runner_patches("document-pool-v2-lite")
            with patches[0], patches[1], patches[2]:
                with self.assertRaises(KeyboardInterrupt):
                    phase4_runner.execute_phase4(
                        args,
                        selected_query_ids=["Q1", "Q2"],
                        inspect_eval_fn=interrupted_eval,
                        task_factory=self._task_factory,
                    )

                q1 = run_dir / sample_directory_name("Q1")
                self.assertTrue((q1 / "result.json").is_file())
                q1_result = json.loads((q1 / "result.json").read_text(encoding="utf-8"))
                self.assertNotIn("inspect_log", q1_result["artifacts"])
                q1_bytes = {
                    path.relative_to(q1): path.read_bytes()
                    for path in q1.rglob("*")
                    if path.is_file()
                }

                args.resume_run = "test-run"
                args.run_id = None

                def resumed_eval(task, **kwargs):
                    calls.append(tuple(task["query_ids"]))
                    self.assertEqual(task["query_ids"], ["Q2"])
                    sample = self._write_sample(task, "Q2")
                    self._emit_sample(sample)
                    log_path = Path(kwargs["log_dir"]) / "batch.eval"
                    log_path.write_text("inspect-log", encoding="utf-8")
                    return [SimpleNamespace(location=str(log_path), samples=[sample])]

                execution = phase4_runner.execute_phase4(
                    args,
                    selected_query_ids=["Q1", "Q2"],
                    inspect_eval_fn=resumed_eval,
                    task_factory=self._task_factory,
                )

                self.assertEqual(calls, [("Q1", "Q2"), ("Q2",)])
                self.assertEqual(execution.valid_samples, 2)
                self.assertTrue((run_dir / "_COMPLETE").is_file())
                self.assertEqual(
                    q1_bytes,
                    {
                        path.relative_to(q1): path.read_bytes()
                        for path in q1.rglob("*")
                        if path.is_file()
                    },
                )

    def test_v2_hook_failure_is_propagated_before_finalization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._fixture(root)

            def evaluate(task, **_kwargs):
                self._emit_sample(self._write_sample(task, "Q1"))
                return []

            patches = self._stable_runner_patches("document-pool-v2-lite")
            with patches[0], patches[1], patches[2], patch.object(
                phase4_runner,
                "build_and_publish_sample",
                side_effect=RuntimeError("cannot persist Q1"),
            ):
                with self.assertRaisesRegex(
                    IncrementalPublicationError,
                    "incremental query publication failed.*Q1",
                ):
                    phase4_runner.execute_phase4(
                        args,
                        selected_query_ids=["Q1"],
                        inspect_eval_fn=evaluate,
                        task_factory=self._task_factory,
                    )

            run_dir = root / "results" / "MiniCPM5-2B" / "LLMPublicHealthQA" / "runs" / "test-run"
            self.assertFalse((run_dir / "Q1").exists())
            self.assertFalse((run_dir / "task-result.json").exists())
            self.assertFalse((run_dir / "_COMPLETE").exists())

    def test_v2_interrupted_resume_rejects_tampered_result_or_missing_marker(self):
        for mutation in ("tamper", "remove-marker"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                args, run_dir = self._interrupted_v2_run(Path(temp))
                q1 = run_dir / sample_directory_name("Q1")
                marker = q1 / "_COMPLETE"
                if mutation == "tamper":
                    result_path = q1 / "result.json"
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    result["ranked_doc_ids"] = ["D2"]
                    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
                    expected_error = "query result digest mismatch"
                else:
                    marker.unlink()
                    expected_error = "missing query completion marker"

                args.resume_run = "test-run"
                args.run_id = None
                called = False

                def should_not_run(*_args, **_kwargs):
                    nonlocal called
                    called = True
                    raise AssertionError("integrity failure must happen before Inspect")

                patches = self._stable_runner_patches("document-pool-v2-lite")
                with patches[0], patches[1], patches[2]:
                    with self.assertRaisesRegex(ResultIntegrityError, expected_error):
                        phase4_runner.execute_phase4(
                            args,
                            selected_query_ids=["Q1", "Q2"],
                            inspect_eval_fn=should_not_run,
                            task_factory=self._task_factory,
                        )
                self.assertFalse(called)

    def test_v1_does_not_install_incremental_publication_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._fixture(root)
            run_dir = root / "results" / "MiniCPM5-2B" / "LLMPublicHealthQA" / "runs" / "test-run"

            def evaluate(task, **kwargs):
                sample = self._write_sample(task, "Q1")
                self._emit_sample(sample)
                self.assertFalse((run_dir / sample_directory_name("Q1")).exists())
                log_path = Path(kwargs["log_dir"]) / "batch.eval"
                log_path.write_text("inspect-log", encoding="utf-8")
                return [SimpleNamespace(location=str(log_path), samples=[sample])]

            patches = self._stable_runner_patches("v3")
            with patches[0], patches[1], patches[2]:
                execution = phase4_runner.execute_phase4(
                    args,
                    selected_query_ids=["Q1"],
                    inspect_eval_fn=evaluate,
                    task_factory=self._task_factory,
                )

            self.assertEqual(execution.exit_code, 0)
            self.assertTrue((run_dir / "_COMPLETE").is_file())


if __name__ == "__main__":
    unittest.main()
