import json
import os
import shutil
import subprocess
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

    def test_read_and_bash_cannot_escape_current_corpus(self):
        if shutil.which("npm") is None or shutil.which("bwrap") is None:
            self.skipTest("npm and bubblewrap are required for the sandbox smoke test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            corpus = workspace / "corpus"
            corpus.mkdir(parents=True)
            (corpus / "doc_02.md").write_text(
                "Polar bears rely on sea ice to hunt seals.\n",
                encoding="utf-8",
            )
            outside = root / "qrels.json"
            secret = "GOLD_LABEL_MUST_NOT_LEAK"
            api_secret = "API_KEY_MUST_NOT_LEAK"
            timeout_probe = f"dci-timeout-probe-{os.getpid()}"
            outside.write_text(secret, encoding="utf-8")
            mock_script = root / "mock-script.json"
            mock_script.write_text(
                json.dumps(
                    [
                        {
                            "type": "tool_call",
                            "name": "bash",
                            "arguments": {"command": "yes OUTPUT_LIMIT_PROBE"},
                            "id": "call_output_limit",
                        },
                        {
                            "type": "tool_call",
                            "name": "read",
                            "arguments": {"path": str(outside)},
                            "id": "call_read_escape",
                        },
                        {
                            "type": "tool_call",
                            "name": "bash",
                            "arguments": {
                                "command": (
                                    f"cat {outside}; "
                                    "test ! -e /mnt; test ! -e /root; test ! -e /proc; "
                                    "printf scratch-ok > /scratch/probe; "
                                    "cat /workspace/corpus/doc_02.md"
                                )
                            },
                            "id": "call_bash_escape",
                        },
                        {
                            "type": "tool_call",
                            "name": "bash",
                            "arguments": {
                                "command": (
                                    "cat /scratch/probe; "
                                    "if bash -c 'exec 3<>/dev/tcp/127.0.0.1/1' 2>/dev/null; "
                                    "then echo NETWORK_OPEN; else echo NETWORK_BLOCKED; fi; "
                                    "grep -n 'sea ice' /workspace/corpus/doc_02.md"
                                )
                            },
                            "id": "call_scratch_and_network",
                        },
                        {
                            "type": "tool_call",
                            "name": "bash",
                            "arguments": {
                                "command": f"bash -c 'exec -a {timeout_probe} sleep 120' & wait",
                                "timeout_seconds": 1,
                            },
                            "id": "call_timeout",
                        },
                        {
                            "type": "tool_call",
                            "name": "bash",
                            "arguments": {
                                "command": (
                                    "for i in $(seq 1 30); do "
                                    "dd if=/dev/zero of=/scratch/quota-$i bs=1M count=1 status=none || break; "
                                    "done"
                                )
                            },
                            "id": "call_scratch_limit",
                        },
                        {"type": "final", "text": '{"ranked_doc_ids":["doc_02"]}'},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = root / "final.json"
            trace_path = root / "trace.json"
            result = run_pi_dci(
                query="Which document explains how polar bears hunt seals?",
                workspace=workspace,
                output_path=output_path,
                trace_path=trace_path,
                mock_script=mock_script,
                openai_api_key=api_secret,
                max_agent_steps=8,
            )

            trace = trace_path.read_text(encoding="utf-8")
            trace_payload = json.loads(trace)
            network_result = next(
                event["result"]["content"][0]["text"]
                for event in trace_payload["events"]
                if event.get("type") == "tool_execution_end"
                and event.get("toolCallId") == "call_scratch_and_network"
            )
            output_limit_result = next(
                event["result"]["content"][0]["text"]
                for event in trace_payload["events"]
                if event.get("type") == "tool_execution_end"
                and event.get("toolCallId") == "call_output_limit"
            )
            scratch_limit_result = next(
                event["result"]["content"][0]["text"]
                for event in trace_payload["events"]
                if event.get("type") == "tool_execution_end"
                and event.get("toolCallId") == "call_scratch_limit"
            )
            timeout_result = next(
                event["result"]["content"][0]["text"]
                for event in trace_payload["events"]
                if event.get("type") == "tool_execution_end"
                and event.get("toolCallId") == "call_timeout"
            )
            self.assertTrue(result.valid_output)
            self.assertEqual(result.tool_sandbox, "bubblewrap")
            self.assertNotIn(secret, trace)
            self.assertNotIn(api_secret, trace)
            self.assertIn("scratch-ok", network_result)
            self.assertIn("NETWORK_BLOCKED", network_result)
            self.assertNotIn("NETWORK_OPEN", network_result)
            self.assertIn("[output limit exceeded; process terminated]", output_limit_result)
            self.assertIn("[scratch limit exceeded; process terminated]", scratch_limit_result)
            self.assertIn("[timeout exceeded; process terminated]", timeout_result)
            residual = subprocess.run(
                ["pgrep", "-f", rf"^{timeout_probe}( |$)"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(residual.returncode, 1, residual.stdout)

    def test_corpus_symlink_fails_closed(self):
        if shutil.which("npm") is None or shutil.which("bwrap") is None:
            self.skipTest("npm and bubblewrap are required for the sandbox smoke test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            corpus = workspace / "corpus"
            corpus.mkdir(parents=True)
            outside = root / "qrels.json"
            outside.write_text("secret", encoding="utf-8")
            (corpus / "escape.md").symlink_to(outside)
            mock_script = root / "mock-script.json"
            mock_script.write_text("[]\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Corpus entries must be regular files"):
                run_pi_dci(
                    query="query",
                    workspace=workspace,
                    output_path=root / "final.json",
                    mock_script=mock_script,
                    max_agent_steps=2,
                )

    def test_corpus_hard_link_fails_closed(self):
        if shutil.which("npm") is None or shutil.which("bwrap") is None:
            self.skipTest("npm and bubblewrap are required for the sandbox smoke test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            corpus = workspace / "corpus"
            corpus.mkdir(parents=True)
            outside = root / "qrels.json"
            outside.write_text("secret", encoding="utf-8")
            os.link(outside, corpus / "escape.md")
            mock_script = root / "mock-script.json"
            mock_script.write_text("[]\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "must not be hard links"):
                run_pi_dci(
                    query="query",
                    workspace=workspace,
                    output_path=root / "final.json",
                    mock_script=mock_script,
                    max_agent_steps=2,
                )


if __name__ == "__main__":
    unittest.main()
