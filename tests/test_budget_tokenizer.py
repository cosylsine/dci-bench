import asyncio
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dci_bench.agents.token_count import normalize_request
from scripts.check_budget_tokenizer import probe


class TokenizerTests(unittest.TestCase):
    def test_normalization_retains_tool_history_and_uses_inspect_schema(self):
        raw = {"messages": [{"role": "user", "content": "正文😀"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "x", "type": "function",
                 "function": {"name": "read", "arguments": '{"path":"a.md"}'}}]},
                {"role": "tool", "tool_call_id": "x", "name": "read", "content": "evidence"}],
               "tools": [{"type": "function", "function": {"name": "read", "description": "read document",
                 "parameters": {"type": "object", "properties": {"path": {"type": "string", "minLength": 1}}, "required": ["path"]}}}],
               "chat_template_kwargs": {"enable_thinking": False}}
        result = asyncio.run(normalize_request(raw, "Qwen3.8-27B"))
        self.assertEqual(result["messages"][0]["content"], "正文😀")
        self.assertEqual(json.loads(result["messages"][1]["tool_calls"][0]["function"]["arguments"]), {"path": "a.md"})
        self.assertTrue(result["tools"][0]["function"]["strict"])
        self.assertNotIn("minLength", result["tools"][0]["function"]["parameters"]["properties"]["path"])
        self.assertEqual(result["chat_template_kwargs"], {"enable_thinking": False})
        self.assertFalse(result["add_special_tokens"])
        self.assertTrue(result["add_generation_prompt"])

    def test_serving_probe_checks_all_shapes_and_stops_on_first_mismatch(self):
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmp:
                calls = []

                def respond(request, timeout):
                    body = json.loads(request.data)
                    calls.append((request.full_url, body))
                    count = 100 + len(body["messages"])
                    if request.full_url.endswith("/tokenize"):
                        payload = {"count": count}
                    else:
                        self.assertNotIn("add_generation_prompt", body)
                        payload = {"usage": {"prompt_tokens": count + int(mismatch), "completion_tokens": 1, "total_tokens": count + 1}}
                    return io.BytesIO(json.dumps(payload).encode())

                output = Path(tmp) / "probe.json"
                with patch("scripts.check_budget_tokenizer.urlopen", side_effect=respond):
                    if mismatch:
                        with self.assertRaisesRegex(RuntimeError, "benchmark not started"):
                            asyncio.run(probe("http://127.0.0.1:1/v1", "fixture", output))
                    else:
                        asyncio.run(probe("http://127.0.0.1:1/v1", "fixture", output))
                report = json.loads(output.read_text())
                self.assertEqual(report["success"], not mismatch)
                self.assertEqual(len(calls), 2 if mismatch else 8)
                if not mismatch:
                    self.assertEqual([row["shape"] for row in report["checks"]], ["search", "read", "final", "repair"])
                    self.assertTrue(all(row["delta"] == 0 for row in report["checks"]))
                    self.assertNotIn("tools", calls[-1][1])
                    self.assertNotIn("tool_choice", calls[-1][1])


if __name__ == "__main__":
    unittest.main()
