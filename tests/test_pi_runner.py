import json
import shutil
import tempfile
import unittest
from pathlib import Path

from dci_bench.agents.pi_agent import run_pi_dci


class PiRunnerTest(unittest.TestCase):
    def test_repairs_final_answer_without_body_evidence(self):
        if shutil.which("npm") is None:
            self.skipTest("npm is required for the Pi runner smoke test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            corpus = workspace / "corpus"
            corpus.mkdir(parents=True)
            (corpus / "doc_00.md").write_text("A document about apples.\n", encoding="utf-8")
            (corpus / "doc_02.md").write_text("Polar bears rely on sea ice to hunt seals.\n", encoding="utf-8")
            mock_script = root / "mock-script.json"
            mock_script.write_text(
                json.dumps(
                    [
                        {
                            "type": "tool_call",
                            "name": "bash",
                            "arguments": {"command": "ls /workspace/corpus"},
                            "id": "call_ls",
                        },
                        {"type": "final", "text": '{"ranked_doc_ids":["doc_00"]}'},
                        {
                            "type": "tool_call",
                            "name": "read",
                            "arguments": {"path": "/workspace/corpus/doc_02.md"},
                            "id": "call_read",
                        },
                        {"type": "final", "text": '{"ranked_doc_ids":["doc_02"]}'},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = root / "final.json"
            result = run_pi_dci(
                query="Which document explains how polar bears hunt seals?",
                workspace=workspace,
                output_path=output_path,
                trace_path=root / "trace.json",
                mock_script=mock_script,
                max_agent_steps=8,
            )

            self.assertTrue(result.valid_output)
            self.assertEqual(result.ranked_doc_ids, ["doc_02"])
            self.assertEqual(result.repair_attempts, 1)
            self.assertIn("corpus/doc_02.md", result.body_evidence["read_files"])
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["contract"]["max_agent_steps"], 8)


if __name__ == "__main__":
    unittest.main()
