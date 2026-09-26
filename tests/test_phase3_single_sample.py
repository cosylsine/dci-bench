import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_phase3_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_single_sample.py"
    spec = importlib.util.spec_from_file_location("phase3_single_sample", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase3SingleSampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_phase3_module()

    def test_summary_reports_valid_output_and_metrics_without_qrels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            sample_dir = output_dir / "Task" / "Q1"
            sample_dir.mkdir(parents=True)
            (sample_dir / "final.json").write_text(
                json.dumps(
                    {
                        "valid_output": True,
                        "failure_reason": None,
                        "ranked_doc_ids": ["D1"],
                        "agent_steps": 3,
                        "tool_calls": 2,
                        "repair_attempts": 0,
                        "inspect_sandbox": "local",
                        "tool_sandbox": "bubblewrap",
                    }
                ),
                encoding="utf-8",
            )
            metadata_dir = root / "metadata" / "Task"
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "qrels.json").write_text(json.dumps({"Q1": {"D1": 1.0}}), encoding="utf-8")

            summary = self.module.build_sample_summary(
                task_name="Task",
                query_id="Q1",
                output_dir=output_dir,
                metadata_root=root / "metadata",
                model_path="/models/minicpm",
                served_model_name="MiniCPM5-2B",
                openai_base_url="http://127.0.0.1:30000/v1",
                openai_service="sglang",
                inspect_log_paths=["logs/eval.eval"],
            )

            self.assertTrue(summary["valid_output"])
            self.assertEqual(summary["ndcg_at_10"], 1.0)
            self.assertEqual(summary["recall_at_10"], 1.0)
            self.assertEqual(summary["f1_at_1"], 1.0)
            self.assertEqual(summary["f1_at_10"], 1.0)
            self.assertEqual(summary["metric_ks"], [1, 3, 5, 10, 20])
            self.assertEqual(summary["openai_service"], "sglang")
            self.assertNotIn("qrels", summary)
            self.assertNotIn("ranked_doc_ids", summary)

    def test_summary_marks_missing_final_as_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_dir = root / "metadata" / "Task"
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "qrels.json").write_text(json.dumps({"Q1": {"D1": 1.0}}), encoding="utf-8")

            summary = self.module.build_sample_summary(
                task_name="Task",
                query_id="Q1",
                output_dir=root / "output",
                metadata_root=root / "metadata",
                model_path="/models/minicpm",
                served_model_name="MiniCPM5-2B",
                openai_base_url="http://127.0.0.1:30000/v1",
                openai_service="sglang",
                inspect_log_paths=[],
            )

            self.assertFalse(summary["valid_output"])
            self.assertIn("did not produce final.json", summary["failure_reason"])

    def test_summary_marks_non_object_final_as_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            sample_dir = output_dir / "Task" / "Q1"
            sample_dir.mkdir(parents=True)
            (sample_dir / "final.json").write_text("[]", encoding="utf-8")
            metadata_dir = root / "metadata" / "Task"
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "qrels.json").write_text(json.dumps({"Q1": {"D1": 1.0}}), encoding="utf-8")

            summary = self.module.build_sample_summary(
                task_name="Task",
                query_id="Q1",
                output_dir=output_dir,
                metadata_root=root / "metadata",
                model_path="/models/minicpm",
                served_model_name="MiniCPM5-2B",
                openai_base_url="http://127.0.0.1:30000/v1",
                openai_service="sglang",
                inspect_log_paths=[],
            )

            self.assertFalse(summary["valid_output"])
            self.assertEqual(summary["failure_reason"], "final.json must contain a JSON object")

    def test_summary_rejects_result_without_sandbox_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            sample_dir = output_dir / "Task" / "Q1"
            sample_dir.mkdir(parents=True)
            (sample_dir / "final.json").write_text(
                json.dumps(
                    {
                        "valid_output": True,
                        "failure_reason": None,
                        "ranked_doc_ids": ["D1"],
                    }
                ),
                encoding="utf-8",
            )
            metadata_dir = root / "metadata" / "Task"
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "qrels.json").write_text(
                json.dumps({"Q1": {"D1": 1.0}}),
                encoding="utf-8",
            )

            summary = self.module.build_sample_summary(
                task_name="Task",
                query_id="Q1",
                output_dir=output_dir,
                metadata_root=root / "metadata",
                model_path="/models/minicpm",
                served_model_name="MiniCPM5-2B",
                openai_base_url="http://127.0.0.1:30000/v1",
                openai_service="sglang",
                inspect_log_paths=[],
            )

            self.assertFalse(summary["valid_output"])
            self.assertIn("sandbox boundary", summary["failure_reason"])


if __name__ == "__main__":
    unittest.main()
