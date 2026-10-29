import argparse
import json
import tempfile
import unittest
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dci_bench.backends.manifest import build_sglang_manifest, write_json_atomic
from dci_bench.protocol.contracts import BENCHMARK_CONTRACT_VERSION
from dci_bench.results.phase4_runner import (
    _package_lock_audit,
    _normalize_trajectory,
    data_audit,
    execute_phase4,
    preflight_bridge_ports,
)
from dci_bench.results.protocol import sha256_json


def _metrics(value: float = 1.0):
    return {
        f"{family}_at_{cutoff}": value
        for family in ("recall", "f1", "ndcg")
        for cutoff in (1, 3, 5, 10, 20)
    }


class Phase4RunnerTest(unittest.TestCase):
    def test_package_lock_audit_prunes_large_runtime_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in (
                "package-lock.json",
                "pi-dci/package-lock.json",
                "data/corpus/package-lock.json",
                "results/old-run/requirements.txt",
                "node_modules/dependency/package-lock.json",
                "__pycache__/requirements.txt",
                ".venv/lib/requirements.txt",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")

            records = _package_lock_audit(root)
            self.assertEqual(
                [record["path"] for record in records],
                ["package-lock.json", "pi-dci/package-lock.json"],
            )

    def test_bridge_port_preflight_rejects_an_occupied_port(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        try:
            with self.assertRaisesRegex(ValueError, "unavailable"):
                preflight_bridge_ports(port, 1)
        finally:
            listener.close()

    def test_bridge_port_preflight_accepts_closed_server_time_wait(self):
        with socket.socket() as listener, socket.socket() as client:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            client.settimeout(2)
            client.connect(("127.0.0.1", port))
            connection, _ = listener.accept()
            with connection:
                connection.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1), b"")
        self.assertEqual(preflight_bridge_ports(port, 1), (port,))

    def _fixture(self, root: Path, query_id="Q1"):
        metadata = root / "metadata" / "LLMPublicHealthQA"
        metadata.mkdir(parents=True)
        (metadata / "queries.jsonl").write_text(
            json.dumps({"_id": query_id, "text": "query one"}) + "\n",
            encoding="utf-8",
        )
        (metadata / "qrels.json").write_text(
            json.dumps({query_id: {"D1": 1}}) + "\n", encoding="utf-8"
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

    def _fake_eval(self, *, with_usage=True, final_failure=None, sample_limit=None, query_id="Q1"):
        def evaluate(task, **kwargs):
            self.assertEqual(task["model_context_window"], 32768)
            self.assertIn("public-health questions", task["task_instruction"])
            output = Path(task["output_dir"])
            from dci_bench.results.protocol import sample_directory_name
            sample_dir = output / sample_directory_name(query_id)
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
                        "contract_version": BENCHMARK_CONTRACT_VERSION,
                        "valid_output": final_failure is None,
                        "failure_reason": None if final_failure is None else final_failure["message"],
                        "failure": final_failure,
                        "ranked_doc_ids": [] if final_failure is not None else ["D1"],
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
            sample_kwargs = {
                "id": query_id,
                "model_usage": usage,
                "scores": {"dci_retrieval_scorer": SimpleNamespace(value=_metrics())},
                "started_at": "2026-09-15T00:00:00Z",
                "completed_at": "2026-09-15T00:00:02Z",
                "total_time": 2.0,
                "working_time": 1.5,
                "error": None,
            }
            if sample_limit is not None:
                sample_kwargs["limit"] = sample_limit
                sample_kwargs["events"] = [
                    SimpleNamespace(
                        event="sample_limit",
                        type=sample_limit.type,
                        message=f"{sample_limit.type} limit exceeded",
                        limit=sample_limit.limit,
                    )
                ]
            sample = SimpleNamespace(**sample_kwargs)
            return [SimpleNamespace(location=str(log_path), samples=[sample])]

        return evaluate

    def test_inspect_time_limit_overrides_fetch_failed_agent_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._fixture(root)
            execution = execute_phase4(
                args,
                selected_query_ids=["Q1"],
                inspect_eval_fn=self._fake_eval(
                    with_usage=True,
                    final_failure={"kind": "agent_error", "message": "fetch failed"},
                    sample_limit=SimpleNamespace(type="time", limit=600),
                ),
                task_factory=self._task_factory,
            )
            result = json.loads((execution.run_dir / "Q1" / "result.json").read_text())
            self.assertEqual(execution.exit_code, 1)
            self.assertFalse(result["valid_output"])
            self.assertEqual(result["failure"]["kind"], "timeout")
            self.assertEqual(result["failure"]["message"], "time limit exceeded")
            self.assertEqual(result["metrics"]["recall_at_1"], 0.0)

    def test_inspect_time_limit_rejects_late_valid_pi_final(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._fixture(root)
            execution = execute_phase4(
                args,
                selected_query_ids=["Q1"],
                inspect_eval_fn=self._fake_eval(
                    with_usage=True,
                    sample_limit=SimpleNamespace(type="time", limit=600),
                ),
                task_factory=self._task_factory,
            )
            result = json.loads((execution.run_dir / "Q1" / "result.json").read_text())
            self.assertEqual(execution.exit_code, 1)
            self.assertFalse(result["valid_output"])
            self.assertEqual(result["failure"]["kind"], "timeout")
            self.assertEqual(result["ranked_doc_ids"], [])
            self.assertTrue(all(value == 0.0 for value in result["metrics"].values()))

    def test_missing_inspect_limit_preserves_pi_token_limit_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._fixture(root)
            execution = execute_phase4(
                args,
                selected_query_ids=["Q1"],
                inspect_eval_fn=self._fake_eval(
                    with_usage=True,
                    final_failure={"kind": "token_limit", "message": "budget reached"},
                ),
                task_factory=self._task_factory,
            )
            result = json.loads((execution.run_dir / "Q1" / "result.json").read_text())
            self.assertEqual(result["failure"]["kind"], "token_limit")
            self.assertEqual(result["failure"]["message"], "budget reached")

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

    def test_data_audit_checks_actual_qrels_hash_when_manifest_has_one(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._fixture(root)
            manifest_path = args.metadata_root / args.task / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["qrels_hash"] = sha256_json({"Q1": {"D1": 1.0}})
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            audit = data_audit(args.metadata_root, args.task)
            self.assertEqual(audit["qrels_hash"], manifest["qrels_hash"])
            (args.metadata_root / args.task / "qrels.json").write_text(
                json.dumps({"Q1": {"D2": 1.0}}) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "qrels_hash"):
                data_audit(args.metadata_root, args.task)

    def test_data_audit_accepts_integer_source_hash_and_rejects_changed_grade(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self._fixture(Path(temp))
            task_root = args.metadata_root / args.task
            manifest_path = task_root / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['qrels_hash'] = sha256_json({'Q1': {'D1': 1}})
            manifest_path.write_text(json.dumps(manifest))
            audit = data_audit(args.metadata_root, args.task)
            self.assertEqual(audit['qrels_hash_expected'], manifest['qrels_hash'])
            self.assertEqual(audit['qrels_hash'], sha256_json({'Q1': {'D1': 1.0}}))
            for changed_grade in [2, True, '1']:
                (task_root / 'qrels.json').write_text(json.dumps({'Q1': {'D1': changed_grade}}))
                with self.assertRaises(ValueError):
                    data_audit(args.metadata_root, args.task)

    @staticmethod
    def _task_factory(**kwargs):
        return kwargs

    def test_original_query_ids_with_path_characters_complete_without_rewriting(self):
        from dci_bench.results.phase4_runner import load_completed_sample
        from dci_bench.results.protocol import sample_directory_name
        for query_id in ("expertwritten/0", "simulated/199", "../escape", "~q~literal"):
            with self.subTest(query_id=query_id), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args = self._fixture(root, query_id=query_id)
                execution = execute_phase4(
                    args, selected_query_ids=[query_id],
                    inspect_eval_fn=self._fake_eval(query_id=query_id),
                    task_factory=self._task_factory,
                )
                self.assertEqual(execution.exit_code, 0)
                directory = sample_directory_name(query_id)
                self.assertNotIn("/", directory)
                self.assertTrue((execution.run_dir / directory / "result.json").is_file())
                self.assertTrue((execution.run_dir / "_COMPLETE").is_file())
                sample = load_completed_sample(execution.run_dir, query_id)
                self.assertEqual(sample["query_id"], query_id)
                self.assertEqual(sample["ranked_doc_ids"], ["D1"])
                result = json.loads((execution.run_dir / "task-result.json").read_text())
                self.assertEqual(result["query_ids"], [query_id])
                self.assertEqual(result["samples"][0]["result"]["path"], directory + "/result.json")

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
            self.assertEqual(
                run_manifest["backend"]["resolved_generation_config"]["max_model_output_tokens"],
                1024,
            )
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

    def test_inspect_eval_receives_contract_generation_kwargs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self._fixture(root)
            captured = {}
            fake = self._fake_eval(with_usage=True)

            def evaluate(task, **kwargs):
                captured.update(kwargs)
                return fake(task, **kwargs)

            execute_phase4(
                args,
                selected_query_ids=["Q1"],
                inspect_eval_fn=evaluate,
                task_factory=self._task_factory,
            )
            self.assertEqual(captured["temperature"], 0)
            self.assertEqual(captured["max_tokens"], 1024)
            self.assertEqual(captured["seed"], 0)
            self.assertEqual(
                captured["extra_body"],
                {"chat_template_kwargs": {"enable_thinking": False}},
            )

    def test_trace_normalizer_preserves_truncated_trace_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.json"
            partial = '{"events":[{"kind":"tool"}'
            path.write_text(partial, encoding="utf-8")
            normalized = _normalize_trajectory(
                path=path,
                run_id="run",
                task="Task",
                query_id="Q1",
                sample=None,
                failure={"kind": "timeout", "message": "stopped"},
            )
            self.assertEqual(normalized["partial_trace"], partial)
            self.assertEqual(normalized["events"], [])

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
