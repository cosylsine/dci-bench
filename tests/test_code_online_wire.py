"""CPU acceptance of code runner -> Inspect -> Pi -> localhost HTTP -> scorer."""

import hashlib
import json
import os
import shutil
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import dci_bench
from dci_bench.data.code_online import load_code_samples
from dci_bench.protocol.code_runtime import code_runtime_contract, runtime_contract_sha256
from dci_bench.results.code_runner import execute_code_run
from dci_bench.scoring.code import score_query, snapshot_identity


@unittest.skipUnless(shutil.which("node") and shutil.which("bwrap"), "Node and Bubblewrap required")
class CodeOnlineWireTest(unittest.TestCase):
    def test_online_nested_read_spans_and_host_scoring(self):
        requests = []
        selected = {"ranked_files": ["src/arithmetic.py"], "abstain": False,
                    "spans": [{"path": "src/arithmetic.py", "start_line": 5, "end_line": 6}]}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if len(requests) == 1:
                    message = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "read_source_1", "type": "function", "function": {
                            "name": "read", "arguments": json.dumps({
                                "path": "/workspace/corpus/src/arithmetic.py",
                                "start_line": 5, "end_line": 6})}}]}
                    reason = "tool_calls"
                else:
                    message = {"role": "assistant", "content": json.dumps(selected)}
                    reason = "stop"
                payload = json.dumps({
                    "id": f"reply_{len(requests)}", "object": "chat.completion", "created": 0,
                    "model": "dci-code-wire-test",
                    "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="dci-code-wire-") as tmp:
                examples = Path(dci_bench.__file__).resolve().parent.parent / "examples" / "code_retrieval"
                samples = load_code_samples(queries_path=examples / "online_queries.json",
                                            labels_path=examples / "labels.json", query_ids=["demo-trace"])
                sample = samples[0]
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    bridge_port = listener.getsockname()[1]
                run_dir = Path(tmp) / "run"
                with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost",
                                             "no_proxy": "127.0.0.1,localhost"}):
                    report = execute_code_run(
                        samples, run_dir=run_dir, model="openai-api/local/dci-code-wire-test",
                        base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="test",
                        bridge_port=bridge_port,
                    )
                self.assertEqual(report["status"], "complete", json.dumps(report, indent=2))
                self.assertEqual(report["valid_outputs"], 1)
                self.assertEqual(report["invalid_outputs"], 0)
                self.assertEqual(len(requests), 2)
                self.assertIn("Retrieval task type: trace2code", json.dumps(requests[0]["messages"]))
                for request in requests:
                    self.assertEqual(request["temperature"], 0)
                    self.assertEqual(request["max_tokens"], 1024)
                    self.assertEqual(request["seed"], 0)
                    self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": False})
                    wire = json.dumps(request)
                    for forbidden in ("gold_files", "gold_spans", "gold_functions", "gold_functionalities",
                                      "no_gold_kind", "functionality_id", "arithmetic.divide",
                                      "labels.json", "content_tokens", "synthetic/arithmetic-demo"):
                        self.assertNotIn(forbidden, wire)
                    for metadata in sample.reference["files"].values():
                        self.assertNotIn(metadata["sha256"], wire)
                tool_messages = [m for m in requests[1]["messages"] if m["role"] == "tool"]
                self.assertEqual(len(tool_messages), 1)
                tool_content = tool_messages[0]["content"]
                self.assertIsInstance(tool_content, str)
                self.assertIn("def divide(a, b):", tool_content)
                self.assertIn("return a / b", tool_content)
                self.assertNotIn("def add", tool_content)
                self.assertIn('"start_line":5', tool_content.replace(" ", ""))
                self.assertIn('"end_line":6', tool_content.replace(" ", ""))
                directory = run_dir / "samples" / sample.directory_name
                final = json.loads((directory / "final.json").read_text())
                self.assertTrue(final["valid_output"])
                self.assertIsNone(final["failure"])
                self.assertEqual(final["code_output"], selected)
                self.assertEqual(final["evaluation_track"], "code")
                self.assertEqual(final["protocol_profile"], "code-v1")
                self.assertEqual(final["contract_version"], "dci-code-online-v1")
                self.assertEqual(final["inspect_sandbox"], "local")
                self.assertEqual(final["tool_sandbox"], "bubblewrap")
                self.assertEqual(final["body_evidence"]["read_files"], ["corpus/src/arithmetic.py"])
                self.assertNotIn("budget_control", final["contract"])
                expected_prediction = {**snapshot_identity(sample.reference), "output": selected}
                expected_score = score_query(expected_prediction, sample.reference)
                for filename in ("prediction.json", "scorer-prediction.json"):
                    self.assertEqual(json.loads((directory / filename).read_text()), expected_prediction)
                for filename in ("score.json", "scorer-score.json"):
                    self.assertEqual(json.loads((directory / filename).read_text()), expected_score)
                self.assertEqual(report["per_query"][sample.query_id], expected_score)
                for metric in ("file_recall_at_20", "file_precision_at_20", "line_coverage",
                               "line_precision", "function_recall", "functionality_recall",
                               "selective_success_at_20"):
                    self.assertEqual(report["metrics"][metric]["mean"], 1.0, metric)
                manifest = json.loads((run_dir / "run-manifest.json").read_text())
                contract_digest = runtime_contract_sha256(code_runtime_contract())
                self.assertEqual(manifest["runtime_contract_sha256"], contract_digest)
                self.assertEqual(final["base_contract_sha256"], contract_digest)
                self.assertFalse(manifest["mock"])
                marker = json.loads((run_dir / "_COMPLETE.json").read_text())
                self.assertEqual(marker["status"], "complete")
                self.assertEqual(marker["run_fingerprint"], manifest["fingerprint"])
                self.assertEqual(report["run_fingerprint"], manifest["fingerprint"])
                for artifact, digest in marker["artifact_sha256"].items():
                    self.assertEqual(hashlib.sha256((run_dir / artifact).read_bytes()).hexdigest(), digest, artifact)
                self.assertEqual(json.loads((run_dir / "code-result.json").read_text()), report)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
