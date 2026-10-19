import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from dci_bench.agents.pi_agent import run_pi_dci


@unittest.skipUnless(shutil.which("node") and shutil.which("bwrap"), "Node and bubblewrap required")
class PiHardeningTest(unittest.TestCase):
    def _run(self, steps, *, timeout=90):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        corpus = root / "workspace" / "corpus"
        corpus.mkdir(parents=True)
        (corpus / "report.md.md").write_text("😀 relevant corpus evidence\n", encoding="utf-8")
        for step in steps:
            step.setdefault("usage", {"input": 10, "output": 5})
        mock = root / "mock.json"
        mock.write_text(json.dumps(steps), encoding="utf-8")
        result = run_pi_dci(
            query="relevant", workspace=corpus.parent, output_path=root / "final.json",
            trace_path=root / "trace.json", mock_script=mock, timeout_seconds=timeout,
            max_agent_steps=20,
        )
        return result, json.loads((root / "trace.json").read_text())

    def test_filename_search_spoof_and_empty_read_require_repair(self):
        steps = [
            {"type": "tool_call", "name": "bash", "arguments": {"command": command}}
            for command in (
                "rg -l relevant /workspace/corpus",
                "rg --files /workspace/corpus",
                "echo cat /workspace/corpus",
            )
        ]
        steps.extend([
            {"type": "tool_call", "name": "read", "arguments": {"path": "/workspace/corpus/report.md.md", "offset": 999}},
            {"type": "final", "text": '{"ranked_doc_ids":["report.md"]}'},
            {"type": "tool_call", "name": "read", "arguments": {"path": "/workspace/corpus/report.md.md"}},
            {"type": "final", "text": '{"ranked_doc_ids":["report.md"]}'},
        ])
        result, trace = self._run(steps)
        self.assertTrue(result.valid_output)
        self.assertEqual(result.ranked_doc_ids, ["report.md"])
        self.assertEqual(result.repair_attempts, 1)
        self.assertEqual(result.body_evidence["content_search_commands"], [])
        self.assertEqual(len(result.body_evidence["read_receipts"]), 1)
        for event in trace["events"]:
            if event["type"] == "tool_execution_end" and event["toolName"] == "bash":
                self.assertFalse(event["result"]["details"]["bodyEvidence"])

    def test_sample_timeout_preserves_read_and_kills_sleep_descendant(self):
        probe = f"dci-cancel-{os.getpid()}"
        result, trace = self._run([
            {"type": "tool_call", "name": "read", "arguments": {"path": "/workspace/corpus/report.md.md"}},
            {"type": "tool_call", "name": "bash", "arguments": {"command": f"bash -c 'exec -a {probe} sleep 60' & wait", "timeout_seconds": 60}},
            {"type": "final", "text": '{"ranked_doc_ids":["report.md"]}'},
        ], timeout=45)
        self.assertFalse(result.valid_output)
        self.assertEqual(result.failure_kind, "timeout")
        self.assertTrue(trace["messages"])
        self.assertTrue(any(event["type"] == "tool_execution_start" for event in trace["events"]))
        self.assertTrue(any(message["role"] == "toolResult" for message in trace["messages"]))
        residual = subprocess.run(["pgrep", "-f", rf"^{probe}( |$)"], capture_output=True, text=True)
        self.assertEqual(residual.returncode, 1, residual.stdout)

    def test_bash_unicode_output_is_bounded_and_not_evidence(self):
        result, trace = self._run([
            {"type": "tool_call", "name": "bash", "arguments": {"command": "yes '中文😀'"}},
            {"type": "tool_call", "name": "read", "arguments": {"path": "/workspace/corpus/report.md.md"}},
            {"type": "final", "text": '{"ranked_doc_ids":["report.md"]}'},
        ])
        self.assertTrue(result.valid_output)
        message = next(m for m in trace["messages"] if m["role"] == "toolResult" and m["toolName"] == "bash")
        text = message["content"][0]["text"]
        self.assertLessEqual(len(text.encode("utf-8")), 20000)
        self.assertNotIn("\ufffd", text)
        self.assertIn("process terminated", text)


if __name__ == "__main__":
    unittest.main()
