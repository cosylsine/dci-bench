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
        if shutil.which("npm") is None or shutil.which("bwrap") is None:
            self.skipTest("npm and bubblewrap are required for the Pi runner smoke test")

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
                            "usage": {"input": 100, "output": 10},
                        },
                        {
                            "type": "final",
                            "text": '{"ranked_doc_ids":["doc_00"]}',
                            "usage": {"input": 100, "output": 10},
                        },
                        {
                            "type": "tool_call",
                            "name": "read",
                            "arguments": {"path": "/workspace/corpus/doc_02.md"},
                            "id": "call_read",
                            "usage": {"input": 100, "output": 10},
                        },
                        {
                            "type": "final",
                            "text": '{"ranked_doc_ids":["doc_02"]}',
                            "usage": {"input": 100, "output": 10},
                        },
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
                context_window=65536,
                task_instruction="Search climate and hunting concepts, then stop.",
            )

            self.assertTrue(result.valid_output)
            self.assertEqual(result.ranked_doc_ids, ["doc_02"])
            self.assertEqual(result.repair_attempts, 1)
            self.assertIn("corpus/doc_02.md", result.body_evidence["read_files"])
            final_text = output_path.read_text(encoding="utf-8")
            trace_text = (root / "trace.json").read_text(encoding="utf-8")
            self.assertEqual(json.loads(final_text)["contract"]["max_agent_steps"], 8)
            self.assertEqual(json.loads(final_text)["contract"]["context_window"], 65536)
            self.assertEqual(
                json.loads(final_text)["contract"]["task_instruction"],
                "Search climate and hunting concepts, then stop.",
            )
            first_user_text = json.loads(trace_text)["messages"][0]["content"][0]["text"]
            self.assertIn("Task-aware retrieval guidance:", first_user_text)
            self.assertIn("Search climate and hunting concepts, then stop.", first_user_text)
            self.assertEqual(final_text, json.dumps(json.loads(final_text), indent=2, sort_keys=True) + "\n")
            self.assertEqual(trace_text, json.dumps(json.loads(trace_text), indent=2, sort_keys=True) + "\n")

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
                            "usage": {"input": 1, "output": 1},
                        },
                        {
                            "type": "tool_call",
                            "name": "read",
                            "arguments": {"path": str(outside)},
                            "id": "call_read_escape",
                            "usage": {"input": 1, "output": 1},
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
                            "usage": {"input": 1, "output": 1},
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
                            "usage": {"input": 1, "output": 1},
                        },
                        {
                            "type": "tool_call",
                            "name": "bash",
                            "arguments": {
                                "command": f"bash -c 'exec -a {timeout_probe} sleep 120' & wait",
                                "timeout_seconds": 1,
                            },
                            "id": "call_timeout",
                            "usage": {"input": 1, "output": 1},
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
                            "usage": {"input": 1, "output": 1},
                        },
                        {
                            "type": "final",
                            "text": '{"ranked_doc_ids":["doc_02"]}',
                            "usage": {"input": 1, "output": 1},
                        },
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

    def test_missing_mock_usage_is_a_structured_failure(self):
        if shutil.which("npm") is None or shutil.which("bwrap") is None:
            self.skipTest("npm and bubblewrap are required for the Pi runner smoke test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            corpus = workspace / "corpus"
            corpus.mkdir(parents=True)
            (corpus / "doc_02.md").write_text("Polar bears rely on sea ice.\n", encoding="utf-8")
            mock_script = root / "mock-script.json"
            mock_script.write_text(
                json.dumps([{"type": "final", "text": '{"ranked_doc_ids":["doc_02"]}'}]) + "\n",
                encoding="utf-8",
            )

            output_path = root / "final.json"
            result = run_pi_dci(
                query="Which document explains polar bears?",
                workspace=workspace,
                output_path=output_path,
                trace_path=root / "trace.json",
                mock_script=mock_script,
                max_agent_steps=4,
            )

            self.assertFalse(result.valid_output)
            self.assertEqual(result.failure_kind, "usage_unavailable")
            self.assertFalse(result.usage_summary["available"])
            self.assertIsNone(result.usage_summary["total_model_tokens"])
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["contract_version"], "dci-mvp-v2")
            self.assertEqual(payload["pi_result_version"], "dci-pi-final-v1")
            self.assertTrue(payload["started_at"].endswith("Z"))
            self.assertTrue(payload["completed_at"].endswith("Z"))
            self.assertEqual(payload["failure"]["kind"], "usage_unavailable")
            self.assertIn("usage_summary", payload)
            trace_payload = json.loads((root / "trace.json").read_text(encoding="utf-8"))
            self.assertEqual(trace_payload["schema_version"], "dci-trajectory-v1")
            self.assertTrue(trace_payload["started_at"].endswith("Z"))
            self.assertTrue(trace_payload["completed_at"].endswith("Z"))

    def test_token_limit_rejects_a_valid_boundary_final(self):
        if shutil.which("npm") is None or shutil.which("bwrap") is None:
            self.skipTest("npm and bubblewrap are required for the Pi runner smoke test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            corpus = workspace / "corpus"
            corpus.mkdir(parents=True)
            (corpus / "doc_02.md").write_text("Polar bears rely on sea ice.\n", encoding="utf-8")
            mock_script = root / "mock-script.json"
            mock_script.write_text(
                json.dumps(
                    [
                        {
                            "type": "tool_call",
                            "name": "read",
                            "arguments": {"path": "/workspace/corpus/doc_02.md"},
                            "id": "call_read",
                            "usage": {"input": 64000, "output": 0},
                        },
                        {
                            "type": "final",
                            "text": '{"ranked_doc_ids":["doc_02"]}',
                            "usage": {"input": 64000, "output": 0},
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            output_path = root / "final.json"
            result = run_pi_dci(
                query="Which document explains polar bears?",
                workspace=workspace,
                output_path=output_path,
                trace_path=root / "trace.json",
                mock_script=mock_script,
                max_agent_steps=4,
            )

            self.assertFalse(result.valid_output)
            self.assertEqual(result.failure_kind, "token_limit")
            self.assertEqual(result.ranked_doc_ids, [])
            self.assertEqual(result.usage_summary["total_model_tokens"], 128000)
            self.assertEqual(result.usage_summary["total_tokens"], 128000)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["failure"]["kind"], "token_limit")

    def test_token_limit_allows_a_valid_final_below_boundary(self):
        if shutil.which("npm") is None or shutil.which("bwrap") is None:
            self.skipTest("npm and bubblewrap are required for the Pi runner smoke test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "workspace" / "corpus"
            corpus.mkdir(parents=True)
            (corpus / "doc.md").write_text("relevant body evidence\n", encoding="utf-8")
            mock_script = root / "mock-script.json"
            mock_script.write_text(
                json.dumps(
                    [
                        {
                            "type": "tool_call",
                            "name": "read",
                            "arguments": {"path": "/workspace/corpus/doc.md"},
                            "usage": {"input": 63999, "output": 0},
                        },
                        {
                            "type": "final",
                            "text": '{"ranked_doc_ids":["doc"]}',
                            "usage": {"input": 64000, "output": 0},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            result = run_pi_dci(
                query="query",
                workspace=root / "workspace",
                output_path=root / "final.json",
                trace_path=root / "trace.json",
                mock_script=mock_script,
                max_agent_steps=4,
            )
            self.assertTrue(result.valid_output)
            self.assertEqual(result.usage_summary["total_model_tokens"], 127999)

    def test_token_limit_rejects_single_turn_overflow(self):
        if shutil.which("npm") is None or shutil.which("bwrap") is None:
            self.skipTest("npm and bubblewrap are required for the Pi runner smoke test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "workspace" / "corpus"
            corpus.mkdir(parents=True)
            (corpus / "doc.md").write_text("relevant body evidence\n", encoding="utf-8")
            mock_script = root / "mock-script.json"
            mock_script.write_text(
                json.dumps(
                    [
                        {
                            "type": "tool_call",
                            "name": "read",
                            "arguments": {"path": "/workspace/corpus/doc.md"},
                            "usage": {"input": 64000, "output": 0},
                        },
                        {
                            "type": "final",
                            "text": '{"ranked_doc_ids":["doc"]}',
                            "usage": {"input": 64001, "output": 0},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            result = run_pi_dci(
                query="query",
                workspace=root / "workspace",
                output_path=root / "final.json",
                trace_path=root / "trace.json",
                mock_script=mock_script,
                max_agent_steps=4,
            )
            self.assertFalse(result.valid_output)
            self.assertEqual(result.failure_kind, "token_limit")
            self.assertEqual(result.usage_summary["total_model_tokens"], 128001)

    def test_token_limit_is_enforced_during_repair(self):
        if shutil.which("npm") is None or shutil.which("bwrap") is None:
            self.skipTest("npm and bubblewrap are required for the Pi runner smoke test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "workspace" / "corpus"
            corpus.mkdir(parents=True)
            (corpus / "doc.md").write_text("relevant body evidence\n", encoding="utf-8")
            mock_script = root / "mock-script.json"
            mock_script.write_text(
                json.dumps(
                    [
                        {
                            "type": "tool_call",
                            "name": "read",
                            "arguments": {"path": "/workspace/corpus/doc.md"},
                            "usage": {"input": 40000, "output": 0},
                        },
                        {"type": "final", "text": "not json", "usage": {"input": 40000, "output": 0}},
                        {
                            "type": "final",
                            "text": '{"ranked_doc_ids":["doc"]}',
                            "usage": {"input": 48000, "output": 0},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            result = run_pi_dci(
                query="query",
                workspace=root / "workspace",
                output_path=root / "final.json",
                trace_path=root / "trace.json",
                mock_script=mock_script,
                max_agent_steps=5,
            )
            self.assertFalse(result.valid_output)
            self.assertEqual(result.failure_kind, "token_limit")
            self.assertEqual(result.repair_attempts, 1)
            self.assertEqual(result.usage_summary["total_model_tokens"], 128000)

    def test_usage_total_excludes_cache_breakdown(self):
        if shutil.which("npm") is None or shutil.which("bwrap") is None:
            self.skipTest("npm and bubblewrap are required for the Pi runner smoke test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            corpus = workspace / "corpus"
            corpus.mkdir(parents=True)
            (corpus / "doc_02.md").write_text("Polar bears rely on sea ice.\n", encoding="utf-8")
            mock_script = root / "mock-script.json"
            usage = {"input": 10, "output": 5, "cacheRead": 2, "cacheWrite": 3}
            mock_script.write_text(
                json.dumps(
                    [
                        {
                            "type": "tool_call",
                            "name": "read",
                            "arguments": {"path": "/workspace/corpus/doc_02.md"},
                            "id": "call_read",
                            "usage": usage,
                        },
                        {
                            "type": "final",
                            "text": '{"ranked_doc_ids":["doc_02"]}',
                            "usage": usage,
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_pi_dci(
                query="Which document explains polar bears?",
                workspace=workspace,
                output_path=root / "final.json",
                trace_path=root / "trace.json",
                mock_script=mock_script,
                max_agent_steps=4,
            )

            self.assertTrue(result.valid_output)
            self.assertEqual(result.usage_summary["input_tokens"], 20)
            self.assertEqual(result.usage_summary["output_tokens"], 10)
            self.assertEqual(result.usage_summary["total_tokens"], 30)
            self.assertEqual(result.usage_summary["cache_read_tokens"], 4)
            self.assertEqual(result.usage_summary["cache_write_tokens"], 6)

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
