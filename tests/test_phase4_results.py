import json
import math
import tempfile
import unittest
from pathlib import Path

from dci_bench.results import (
    ResultIntegrityError,
    ResultValidationError,
    SAMPLE_RESULT_SCHEMA,
    TASK_RESULT_SCHEMA,
    aggregate_task_results,
    append_task_index_run,
    artifact_record,
    build_run_manifest,
    build_sample_result,
    build_task_index,
    build_task_result,
    canonical_json,
    finalize_run,
    r7_p50,
    r7_p95,
    run_directory,
    sha256_json,
    validate_no_sensitive_fields,
    validate_run_manifest,
    validate_sample_result,
    validate_task_index,
    validate_task_result,
    verify_artifact_record,
    write_json_atomic,
    write_run_manifest,
    write_sample_result,
    write_task_index,
    write_task_result,
    update_task_index_run,
)


def _sample(run_id: str, query_id: str, *, valid: bool, value: float = 1.0):
    return build_sample_result(
        run_id=run_id,
        task="Task",
        query_id=query_id,
        query_sha256=sha256_json(f"query:{query_id}"),
        valid_output=valid,
        ranked_doc_ids=["D1"] if valid else None,
        metrics={
            "recall_at_1": value,
            "recall_at_3": value,
            "recall_at_5": value,
            "recall_at_10": value,
            "recall_at_20": value,
            "f1_at_1": value,
            "f1_at_3": value,
            "f1_at_5": value,
            "f1_at_10": value,
            "f1_at_20": value,
            "ndcg_at_1": value,
            "ndcg_at_3": value,
            "ndcg_at_5": value,
            "ndcg_at_10": value,
            "ndcg_at_20": value,
        },
        status="success" if valid else "token_limit",
        failure=None if valid else {"kind": "token_limit", "message": "budget reached"},
        usage={
            "available": valid,
            "input_tokens": 10 if valid else None,
            "output_tokens": 5 if valid else None,
            "total_tokens": 15 if valid else None,
        },
        timing={
            "started_at": "2026-09-15T00:00:00Z" if valid else None,
            "ended_at": "2026-09-15T00:00:02Z" if valid else None,
            "wall_time_seconds": 2.0 if valid else None,
            "working_time_seconds": 1.5 if valid else None,
        },
        execution={"agent_steps": 2, "tool_calls": 1, "model_calls": 1, "repair_attempts": 0},
    )


class Phase4ResultsTest(unittest.TestCase):
    def test_canonical_hash_and_atomic_json_are_deterministic(self):
        value = {"b": 2, "a": [1, "中文"]}
        self.assertEqual(canonical_json(value), '{"a":[1,"中文"],"b":2}')
        with tempfile.TemporaryDirectory() as temp:
            path = write_json_atomic(Path(temp) / "nested" / "value.json", value)
            rendered = path.read_text(encoding="utf-8")
            self.assertEqual(
                rendered,
                '{\n  "a": [\n    1,\n    "中文"\n  ],\n  "b": 2\n}\n',
            )
            self.assertEqual(json.loads(rendered), value)
            self.assertEqual(sha256_json(value), sha256_json({"a": [1, "中文"], "b": 2}))

    def test_r7_percentiles_and_population_stddev(self):
        self.assertEqual(r7_p50([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(r7_p95([1, 2, 3, 4]), 3.85)
        sample = _sample("run", "Q1", valid=True)
        task = build_task_result(run_id="run", task="Task", samples=[sample])
        stats = task["metrics"]["macro"]["recall_at_1"]
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["population_stddev"], 0.0)

    def test_sample_result_has_version_and_rejects_sensitive_fields(self):
        sample = _sample("run", "Q1", valid=True)
        self.assertEqual(sample["schema_version"], SAMPLE_RESULT_SCHEMA)
        self.assertTrue(validate_sample_result(sample))
        with self.assertRaises(ResultValidationError):
            validate_no_sensitive_fields({"qrels": {"Q1": {"D1": 1}}})
        with self.assertRaises(ResultValidationError):
            validate_no_sensitive_fields({"api_key": "do-not-store"})
        # Token usage is an allowed audit field, unlike a credential key.
        self.assertTrue(validate_no_sensitive_fields({"input_tokens": 12, "total_tokens": 12}))

    def test_success_requires_usage_time_and_strict_sandbox_attestation(self):
        sample = _sample("run", "Q1", valid=True)
        sample["usage"]["available"] = False
        with self.assertRaises(ResultValidationError):
            validate_sample_result(sample)
        sample = _sample("run", "Q1", valid=True)
        sample["security"]["tool_sandbox"] = "none"
        with self.assertRaises(ResultValidationError):
            validate_sample_result(sample)

    def test_task_aggregation_uses_invalid_as_zero_and_reports_valid_only(self):
        valid = _sample("run", "Q1", valid=True, value=1.0)
        invalid = _sample("run", "Q2", valid=False, value=1.0)
        task = aggregate_task_results(run_id="run", task="Task", sample_results=[valid, invalid])
        self.assertEqual(task["schema_version"], TASK_RESULT_SCHEMA)
        self.assertEqual(task["counts"]["planned"], 2)
        self.assertEqual(task["counts"]["valid"], 1)
        self.assertEqual(task["counts"]["invalid"], 1)
        self.assertEqual(task["failure_rate"], 0.5)
        macro = task["metrics"]["macro"]["recall_at_1"]
        valid_only = task["metrics"]["valid_only"]["recall_at_1"]
        self.assertEqual(macro["mean"], 0.5)
        self.assertEqual(macro["count"], 2)
        self.assertEqual(valid_only["mean"], 1.0)
        self.assertEqual(valid_only["count"], 1)
        self.assertEqual(task["usage"]["input_tokens"]["available_count"], 1)
        self.assertEqual(task["usage"]["input_tokens"]["missing_count"], 1)
        self.assertEqual(task["usage"]["input_tokens"]["sum"], 10.0)
        self.assertEqual(task["timing"]["wall_time_seconds"]["available_count"], 1)
        self.assertEqual(task["counts"]["failure_counts"]["token_limit"], 1)
        self.assertTrue(validate_task_result(task))

    def test_run_fingerprint_manifest_and_index_validation(self):
        manifest = build_run_manifest(
            run_id="2026-run",
            model_key="MiniCPM5-2B",
            task="Task",
            query_ids=["Q1", "Q2"],
            model={"model_id": "MiniCPM5-2B", "revision": "abc"},
            backend={"name": "sglang", "revision": "def"},
            data={"manifest_sha256": "data"},
            benchmark_contract="dci-mvp-v1",
            code={"revision": "gitsha"},
        )
        self.assertTrue(validate_run_manifest(manifest))
        changed = dict(manifest)
        changed["model"] = {"model_id": "different"}
        with self.assertRaises(ResultValidationError):
            validate_run_manifest(changed)
        index = build_task_index(
            model_key="MiniCPM5-2B",
            task="Task",
            runs=[
                {
                    "run_id": manifest["run_id"],
                    "status": "running",
                    "run_fingerprint": manifest["run_fingerprint"],
                }
            ],
        )
        self.assertTrue(validate_task_index(index))

    def test_task_index_appends_and_updates_without_losing_history(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "Task" / "index.json"
            first = {"run_id": "run-1", "status": "running", "run_fingerprint": "a" * 64}
            second = {"run_id": "run-2", "status": "running", "run_fingerprint": "b" * 64}
            append_task_index_run(path, model_key="Model", task="Task", run=first)
            append_task_index_run(path, model_key="Model", task="Task", run=second)
            with self.assertRaises(FileExistsError):
                append_task_index_run(path, model_key="Model", task="Task", run=first)
            update_task_index_run(
                path,
                run={"run_id": "run-1", "status": "complete", "run_fingerprint": "a" * 64},
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([item["run_id"] for item in payload["runs"]], ["run-1", "run-2"])
            self.assertEqual(payload["runs"][0]["status"], "complete")

    def test_artifact_hash_and_immutable_finalize(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_root = run_directory(root / "results", model_key="MiniCPM5-2B", task="Task", run_id="run")
            (run_root / "Q1").mkdir(parents=True)
            (run_root / "Q1" / "final.json").write_text('{"ranked_doc_ids":["D1"]}', encoding="utf-8")
            (run_root / "Q1" / "trace.json").write_text('{"events":[]}', encoding="utf-8")
            final_artifact = artifact_record(run_root / "Q1" / "final.json", root=run_root)
            trace_artifact = artifact_record(run_root / "Q1" / "trace.json", root=run_root)
            sample = _sample("run", "Q1", valid=True)
            sample["artifacts"] = {"final": final_artifact, "trace": trace_artifact}
            # Revalidate after adding artifacts; hashes and relative paths are required.
            validate_sample_result(sample)
            task = build_task_result(run_id="run", task="Task", samples=[sample])
            manifest = build_run_manifest(
                run_id="run", model_key="MiniCPM5-2B", task="Task", query_ids=["Q1"]
            )
            write_run_manifest(run_root, manifest)
            write_sample_result(run_root, sample)
            write_task_result(run_root, task)
            self.assertTrue(verify_artifact_record(final_artifact, root=run_root))
            marker = finalize_run(run_root, manifest=manifest, task_result=task)
            self.assertTrue(marker.exists())
            with self.assertRaises(FileExistsError):
                finalize_run(run_root, manifest=manifest, task_result=task)

    def test_artifact_hash_mismatch_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "trace.json"
            path.write_text("one", encoding="utf-8")
            record = artifact_record(path, root=root)
            path.write_text("two", encoding="utf-8")
            with self.assertRaises(ResultIntegrityError):
                verify_artifact_record(record, root=root)


if __name__ == "__main__":
    unittest.main()
