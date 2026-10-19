"""CPU acceptance across Pi -> Inspect bridge -> OpenAI-compatible HTTP."""

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

from dci_bench.backends.manifest import inspect_generation_kwargs


@unittest.skipUnless(shutil.which("node") and shutil.which("bwrap"), "Node and Bubblewrap required")
class GenerationWireTest(unittest.TestCase):
    def test_inspect_sends_canonical_config_on_every_model_turn(self):
        from inspect_ai import eval as inspect_eval
        from dci_bench.tasks.mteb_llm_retrieval import mteb_llm_retrieval

        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                if len(requests) == 1:
                    message = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "read_1", "type": "function", "function": {
                            "name": "read", "arguments": json.dumps({"path": "/workspace/corpus/report.md.md"})}}
                    ]}
                    reason = "tool_calls"
                else:
                    message = {"role": "assistant", "content": '{"ranked_doc_ids":["report.md"]}'}
                    reason = "stop"
                payload = json.dumps({
                    "id": f"reply_{len(requests)}", "object": "chat.completion", "created": 0,
                    "model": "dci-wire-test", "choices": [{"index": 0, "message": message, "finish_reason": reason}],
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
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                task_name = "LLMPublicHealthQA"
                metadata = root / "metadata" / task_name
                corpus = root / "workspaces" / task_name / "corpus"
                metadata.mkdir(parents=True)
                corpus.mkdir(parents=True)
                (metadata / "queries.jsonl").write_text('{"_id":"Q1","text":"Which document concerns public health?"}\n')
                (metadata / "qrels.json").write_text('{"Q1":{"report.md":1}}')
                (corpus / "report.md.md").write_text("Public health evidence for this query.\n")
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    bridge_port = listener.getsockname()[1]
                task = mteb_llm_retrieval(
                    task_name=task_name, query_ids=["Q1"], workspace_root=str(root / "workspaces"),
                    metadata_root=str(root / "metadata"), output_dir=str(root / "output"),
                    bridge_port=bridge_port, model_context_window=65536,
                )
                with patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}):
                    logs = inspect_eval(
                        task, model="openai-api/local/dci-wire-test",
                        model_base_url=f"http://127.0.0.1:{server.server_port}/v1",
                        model_args={"api_key": "test", "responses_api": False},
                        log_dir=str(root / "logs"), display="none", max_retries=0,
                        **inspect_generation_kwargs(),
                    )
                self.assertEqual(logs[0].status, "success", str(logs[0].error))
                final = json.loads((root / "output" / "Q1" / "final.json").read_text())
                self.assertTrue(final["valid_output"], f"{final.get('failure_reason')}; requests={requests}")
                self.assertEqual(final["ranked_doc_ids"], ["report.md"])
                self.assertEqual(len(requests), 2)
                for request in requests:
                    self.assertEqual(request["temperature"], 0)
                    self.assertEqual(request["max_tokens"], 1024)
                    self.assertEqual(request["seed"], 0)
                    self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": False})
                    self.assertNotIn("qrels", json.dumps(request))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
