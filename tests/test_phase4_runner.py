import argparse
import json
import tempfile
import unittest
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dci_bench.backends.manifest import build_sglang_manifest, write_json_atomic
from dci_bench.results.phase4_runner import data_audit, execute_phase4, preflight_bridge_ports


def _metrics(value: float = 1.0):
    return {
        f"{family}_at_{cutoff}": value
        for family in ("recall", "f1", "ndcg")
        for cutoff in (1, 3, 5, 10, 20)
    }


class Phase4RunnerTest(unittest.TestCase):
    def test_bridge_port_preflight_rejects_an_occupied_port(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        try:
            with self.assertRaisesRegex(ValueError, "unavailable"):
                preflight_bridge_ports(port, 1)
        finally:
            listener.close()

    def _fixture(self, root: Path):
        metadata = root / "metadata" / "LLMPublicHealthQA"
        metadata.mkdir(parents=True)
        (metadata / "queries.jsonl").write_text(
            json.dumps({"_id": "Q1", "text": "query one"}) + "\n",
            encoding="utf-8",
        )
        (metadata / "qrels.json").write_text(
            json.dumps({"Q1": {"D1": 1}}) + "\n", encoding="utf-8"
        )
        (metadata / "manifest.json").write_text(
            json.dumps(
                {
                    "task": "LLMPublicHealthQA",
                    "dataset": "mteb/llm-eval-public-health-qa",
                    "revision": "b05938525381",
                    "resolved_revision": "b05938525381-resolved",
                    "corpus_hash": "a" * 64,
                    "source_parquet_hash": "b" * 64,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        model = root / "model"
        (model / ".hfd").mkdir(parents=True)
        (model / "config.json").write_text('{"model_type":"test"}\n', encoding="utf-8")
        (model / "weights.safetensors").write_bytes(b"weights")
        (model / ".hfd" / "repo_metadata.json").write_text(
            json.dumps(
                {
                    "id": "org/model",
                    "sha": "revision",
                    "siblings": [
                        {
                            "rfilename": "weights.safetensors",
                            "lfs": {"sha256": "c" * 64},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        backend = build_sglang_manifest(
            model_key="MiniCPM5-2B",
            model_path=model,
            served_model_name="MiniCPM5-2B",
            base_url="http://127.0.0.1:30000/v1",
            sglang_version="test",
            sglang_git_revision="revision",
            tool_call_parser="minicpm5",
            sampling_backend="pytorch",
            tensor_parallel_size=1,
            context_length=32768,
            dtype="bfloat16",
        )
        backend_path = root / "backend.json"
        write_json_atomic(backend_path, backend)
        args = argparse.Namespace(
            model_key="MiniCPM5-2B",
            task="LLMPublicHealthQA",
            backend_manifest=backend_path,
            metadata_root=root / "metadata",
            workspace_root=root / "workspaces",
            results_root=root / "results",
            metric_ks=[1, 3, 5, 10, 20],
            bridge_port=13131,
            max_concurrency=1,
            api_key="SECRET_TEST_KEY",
            run_id="test-run",
            resume_run=None,
            backend_artifact=[],
        )
        return args

    def _fake_eval(self, *, with_usage=True):
        def evaluate(task, **kwargs):
            self.assertEqual(task["model_context_window"], 32768)
            self.assertIn("public-health questions", task["task_instruction"])
            output = Path(task["output_dir"])
            sample_dir = output / "Q1"
            sample_dir.mkdir(parents=True)
            usage_summary = {
                "available": with_usage,
                "input_tokens": 10 if with_usage else None,
                "output_tokens": 5 if with_usage else None,
                "total_tokens": 15 if with_usage else None,
                "turns": 1,
            }
            (sample_dir / "final.json").write_text(
                json.dumps(
                    {
                        "contract_version": "dci-mvp-v2",
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
            log_path = Path(kwargs["log_dir"]) / "batch.eval"
            log_path.write_text("inspect-log", encoding="utf-8")
            usage = (
                {
                    "openai-api/sglang/MiniCPM5-2B": SimpleNamespace(
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        input_tokens_cache_read=None,
                        input_tokens_cache_write=None,
                    )
                }
                if with_usage
                else {}
            )
            sample = SimpleNamespace(
                id="Q1",
                model_usage=usage,
                scores={"dci_retrieval_scorer": SimpleNamespace(value=_metrics())},
                started_at="2026-09-15T00:00:00Z",
                completed_at="2026-09-15T00:00:02Z",
                total_time=2.0,
                working_time=1.5,
                error=None,
            )
            return [SimpleNamespace(location=str(log_path), samples=[sample])]

        return evaluate

    def test_data_audit_rejects_empty_resolved_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._fixture(root)
            manifest_path = args.metadata_root / args.task / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["resolved_revision"] = None
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty"):
                data_audit(args.metadata_root, args.task)

    def test_data_audit_accepts_legacy_manifest_without_instruction(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self._fixture(Path(temp))
            audit = data_audit(args.metadata_root, args.task)
            self.assertEqual(audit["dataset"], "mteb/llm-eval-public-health-qa")

    def test_data_audit_rejects_mismatched_manifest_instruction(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self._fixture(Path(temp))
            manifest_path = args.metadata_root / args.task / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["task_instruction"] = "stale instruction"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                data_audit(args.metadata_root, args.task)

    @staticmethod
    def _task_factory(**kwargs):
        return kwargs

    def test_complete_run_has_canonical_artifacts_and_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._fixture(root)
            execution = execute_phase4(
                args,
                selected_query_ids=["Q1"],
                inspect_eval_fn=self._fake_eval(with_usage=True),
                task_factory=self._task_factory,
            )
            self.assertEqual(execution.exit_code, 0)
            self.assertTrue((execution.run_dir / "_COMPLETE").is_file())
            self.assertFalse((execution.run_dir / ".staging").exists())
            self.assertTrue((execution.run_dir / "Q1" / "result.json").is_file())
            self.assertTrue((execution.run_dir / "Q1" / "final.json").is_file())
            trace = json.loads((execution.run_dir / "Q1" / "trace.json").read_text())
            self.assertEqual(trace["schema_version"], "dci-trajectory-v1")
            task_result = json.loads((execution.run_dir / "task-result.json").read_text())
            self.assertEqual(task_result["counts"]["valid"], 1)
            self.assertEqual(task_result["usage"]["total_tokens"]["sum"], 15.0)
            run_manifest = json.loads((execution.run_dir / "run-manifest.json").read_text())
            task_context = run_manifest["benchmark_contract"]["task_context"]
            self.assertEqual(task_context["task_name"], "LLMPublicHealthQA")
            self.assertIn("public-health questions", task_context["instruction"])
            index = json.loads((root / "results" / "MiniCPM5-2B" / "LLMPublicHealthQA" / "index.json").read_text())
            self.assertEqual(index["runs"][0]["status"], "complete")
            self.assertNotIn("SECRET_TEST_KEY", json.dumps(task_result))
            json_paths = list(execution.run_dir.rglob("*.json")) + [
                root / "results" / "MiniCPM5-2B" / "LLMPublicHealthQA" / "index.json"
            ]
            for path in json_paths:
                with self.subTest(path=path):
                    text = path.read_text(encoding="utf-8")
                    self.assertEqual(
                        text,
                        json.dumps(
                            json.loads(text),
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                    )

    def test_missing_inspect_usage_is_terminal_failure_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._fixture(root)
            execution = execute_phase4(
                args,
                selected_query_ids=["Q1"],
                inspect_eval_fn=self._fake_eval(with_usage=False),
                task_factory=self._task_factory,
            )
            self.assertEqual(execution.exit_code, 1)
            result = json.loads((execution.run_dir / "Q1" / "result.json").read_text())
            self.assertFalse(result["valid_output"])
            self.assertEqual(result["failure"]["kind"], "usage_unavailable")
            self.assertIsNone(result["usage"]["total_tokens"])
            self.assertEqual(result["metrics"]["recall_at_1"], 0.0)

    def test_resume_discards_staging_and_preserves_original_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._fixture(root)
            diagnostic = root / "models.json"
            diagnostic.write_text('{"instance":"first"}\n', encoding="utf-8")
            args.backend_artifact = [diagnostic]

            with patch(
                "dci_bench.results.phase4_runner.finalize_run",
                side_effect=RuntimeError("simulated launcher interruption"),
            ):
                with self.assertRaisesRegex(RuntimeError, "launcher interruption"):
                    execute_phase4(
                        args,
                        selected_query_ids=["Q1"],
                        inspect_eval_fn=self._fake_eval(with_usage=True),
                        task_factory=self._task_factory,
                    )

            run_dir = root / "results" / "MiniCPM5-2B" / "LLMPublicHealthQA" / "runs" / "test-run"
            stale = run_dir / ".staging" / "aborted" / "Q1"
            stale.mkdir(parents=True)
            (stale / "trace.json").write_text("truncated", encoding="utf-8")
            diagnostic.write_text('{"instance":"second"}\n', encoding="utf-8")

            args.resume_run = "test-run"
            args.run_id = None
            execution = execute_phase4(
                args,
                selected_query_ids=["Q1"],
                inspect_eval_fn=lambda *_args, **_kwargs: self.fail("completed sample was rerun"),
                task_factory=self._task_factory,
            )

            self.assertEqual(execution.exit_code, 0)
            self.assertFalse((run_dir / ".staging").exists())
            self.assertEqual(
                (run_dir / "artifacts" / "backend" / "models.json").read_text(encoding="utf-8"),
                '{"instance":"first"}\n',
            )
            self.assertTrue((run_dir / "_COMPLETE").is_file())


if __name__ == "__main__":
    unittest.main()
