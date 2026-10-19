"""CPU-only replay checks. All generation requests go to a local fake server."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
SPEC = importlib.util.spec_from_file_location("final_replay", REPO / "scripts/prepare_harness_final_replay.py")
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
RUNTIME = REPO / "results/harness-continuations/20261014T043719Z-validation-remaining/runtime"
NODE = Path(R.BENCH_PYTHON).with_name("node")


class ReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="dci-replay-tests-")
        cls.root = Path(cls.temp.name)
        cls.engine = cls.root / "pi-dci/scripts/dci-run.ts"
        cls.engine.parent.mkdir(parents=True)
        cls.engine.write_text(R.generate_engine((RUNTIME / "pi-dci/scripts/dci-run.ts").read_text(),
                                                (REPO / "scripts/harness_final_replay_entry.ts.inc").read_text()))
        shutil.copytree(RUNTIME / "dci_bench/protocol", cls.root / "dci_bench/protocol")
        shutil.copyfile(RUNTIME / "pi-dci/package.json", cls.root / "pi-dci/package.json")
        (cls.root / "pi-dci/node_modules").symlink_to(RUNTIME / "pi-dci/node_modules")
        cls.contracts = R.contracts(RUNTIME)

    @classmethod
    def tearDownClass(cls):
        # Temporary generated test files only, never the frozen runtime link.
        for path in (cls.root / "dci_bench/protocol").rglob("*"):
            if path.is_file():
                path.chmod(0o600)
        cls.temp.cleanup()

    def fixture(self):
        receipt = {"path": "/workspace/corpus/report.md.md", "doc_id": "report.md", "returned_bytes": 8,
                   "content_sha256": "fixture", "offset_codepoints": 0, "returned_codepoints": 8}
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Find a report"}], "timestamp": 1},
            {"role": "assistant", "content": [{"type": "toolCall", "id": "call1", "name": "read",
                                               "arguments": {"path": "report.md.md"}}],
             "api": "openai-completions", "provider": "inspect-bridge", "model": "inspect", "stopReason": "toolUse",
             "timestamp": 2, "usage": {"input": 90, "output": 10, "totalTokens": 100, "usageAvailable": True}},
            {"role": "toolResult", "toolName": "read", "toolCallId": "call1", "isError": False,
             "content": [{"type": "text", "text": "Evidence"}],
             "details": {"bodyEvidence": True, "receipt": receipt}, "timestamp": 3},
        ]
        control = {"action": "finalize", "turn": 1, "remaining_tokens": 127900, "remaining_seconds": 60,
                   "reason": "identical tool calls repeated"}
        contract = self.contracts[R.PROFILES[0]]
        prompt = contract["budget_control"]["final_prompt"].replace("{reason}", control["reason"]).replace(
            "{remaining_tokens}", str(control["remaining_tokens"]))
        trace = {"task": "fixture", "query_id": "q1", "protocol_profile": R.PROFILES[0], "control_events": [control],
                 "messages": messages + [{"role": "user", "content": prompt}],
                 "body_evidence": {"read_receipts": [receipt]}, "failure": None}
        value = R.extract_prefix(trace, contract, ["report.md.md", "other.md"])
        wire = [{"role": "system", "content": contract["system_prompt"]}, {"role": "user", "content": "Find a report"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call1", "type": "function",
                 "function": {"name": "read", "arguments": '{"path": "report.md.md"}'}}]},
                {"role": "tool", "tool_call_id": "call1", "content": "Evidence"}]
        template = {"model": "Qwen3.8-27B", "max_tokens": 1024, "temperature": 0, "seed": 0, "tools": None,
                    "tool_choice": None, "chat_template_kwargs": {"enable_thinking": False}}
        value.update(wire_prefix=wire, wire_prefix_sha256=R.json_hash(wire), request_template=template,
                     original_request_sha256=R.json_hash({**template, "messages": wire + [{"role": "user", "content": prompt}]}))
        return trace, value

    def run_engine(self, replies, *, profile=R.PROFILES[0], value=None):
        if value is None:
            _, value = self.fixture()
        directory = self.root / self._testMethodName / str(len(list(self.root.glob(self._testMethodName + "/*"))))
        directory.mkdir(parents=True)
        input_path, output_path = directory / "input.json", directory / "output.json"
        R.write_json(input_path, value)
        captured = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if len(captured) > len(replies):
                    self.send_error(500, "unexpected request")
                    return
                reply = replies[len(captured) - 1]
                time.sleep(reply.get("delay", 0))
                response = {"choices": [{"finish_reason": "stop", "message": reply.get("message", {"content": reply.get("text", "")})}],
                            "usage": reply.get("usage", {"prompt_tokens": 110, "completion_tokens": 10})}
                raw = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except BrokenPipeError:
                    pass  # Expected when testing the client's deadline abort.

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run([str(NODE), "--import", "tsx", str(self.engine), str(input_path), str(output_path)],
                                    cwd=self.root / "pi-dci", env={**os.environ, "TSX_DISABLE_CACHE": "1",
                                    "DCI_EVALUATION_TRACK": "document", "DCI_PROTOCOL_PROFILE": profile,
                                    "DCI_PROTOCOL_CONTRACT_SHA256": R.CONTRACT_HASHES[profile],
                                    "BACKEND_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1"},
                                    capture_output=True, text=True, timeout=30)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        return result, R.read_json(output_path) if output_path.exists() else None, captured

    def test_exact_wire_prefix_and_only_final_prompt_changes(self):
        _, value = self.fixture()
        outputs = []
        for profile in R.PROFILES:
            process, result, requests = self.run_engine([{"text": '{"ranked_doc_ids":["report.md"]}'}], profile=profile)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue(result["valid_output"])
            self.assertEqual(result["tool_executions"], 0)
            self.assertEqual(requests[0]["messages"][:-1], value["wire_prefix"])
            self.assertEqual({k: v for k, v in requests[0].items() if k != "messages"}, value["request_template"])
            self.assertEqual(result["prefix_usage"]["total_tokens"], 100)
            self.assertEqual(result["usage"]["total_tokens"], 120)
            outputs.append(requests[0]["messages"][-1]["content"])
        self.assertNotEqual(*outputs)

    def test_unknown_filename_uses_original_mapping_and_selective_repair(self):
        process, result, requests = self.run_engine([
            {"text": '{"ranked_doc_ids":["report.md.md"]}'}, {"text": '{"ranked_doc_ids":["report.md"]}'}],
            profile=R.PROFILES[1])
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(result["valid_output"])
        self.assertEqual(result["repair_attempts"], 1)
        reason = result["repair_reasons"][0]
        self.assertIn('"doc_id":"report.md"', reason)
        self.assertEqual(requests[1]["messages"][-1]["content"],
                         self.contracts[R.PROFILES[1]]["budget_control"]["final_repair_prompt"].replace("{reason}", reason))
        self.assertEqual(requests[0]["messages"], requests[1]["messages"][:-2])

    def test_late_tool_call_blocked_and_repaired_without_execution(self):
        process, result, requests = self.run_engine([
            {"message": {"content": None, "tool_calls": [{"id": "late", "type": "function", "function": {
                "name": "bash", "arguments": '{"command":"must-not-run"}'}}]}},
            {"text": '{"ranked_doc_ids":["report.md"]}'}])
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(result["valid_output"])
        self.assertIn("blocked_tool_calls", requests[1]["messages"][-2]["content"])
        self.assertEqual(result["tool_executions"], 0)

    def test_three_repairs_maximum(self):
        process, result, requests = self.run_engine([{"text": "not JSON"}] * 4)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(requests), 4)
        self.assertEqual(result["repair_attempts"], 3)
        self.assertEqual(result["failure"]["kind"], "invalid_json")

    def test_cumulative_crossing_invalidates_even_valid_json(self):
        process, result, requests = self.run_engine([{"text": '{"ranked_doc_ids":["report.md"]}',
                                                    "usage": {"prompt_tokens": 127890, "completion_tokens": 10}}])
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(len(requests), 1)
        self.assertEqual(result["failure"]["kind"], "token_limit")
        self.assertFalse(result["valid_output"])

    def test_pre_dispatch_budget_guard_sends_no_request(self):
        _, value = self.fixture()
        value["messages"][1]["usage"].update(input=120000, output=10, totalTokens=120010)
        value["control"]["remaining_tokens"] = 7990
        value["prefix_sha256"] = R.json_hash(value["messages"])
        value["origin_profile"] = R.PROFILES[1]  # old prompt isn't this treatment's baseline
        process, result, requests = self.run_engine([], value=value)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(requests, [])
        self.assertEqual(result["failure"]["kind"], "token_limit")

    def test_missing_usage_is_not_scored_as_retrieval_failure(self):
        process, result, _ = self.run_engine([{"text": '{"ranked_doc_ids":["report.md"]}', "usage": None}])
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(result["failure"]["kind"], "usage_unavailable")

    def test_remaining_deadline_aborts_delayed_response(self):
        _, value = self.fixture()
        value["control"]["remaining_seconds"] = 0.2
        process, result, requests = self.run_engine([{"delay": 0.5, "text": '{"ranked_doc_ids":["report.md"]}'}], value=value)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertFalse(result["valid_output"])
        self.assertEqual(result["failure"]["kind"], "timeout")
        self.assertLessEqual(len(requests), 1)

    def test_tampered_prefix_rejected_before_network(self):
        _, value = self.fixture()
        value["messages"][0]["content"][0]["text"] = "changed"
        process, result, requests = self.run_engine([], value=value)
        self.assertNotEqual(process.returncode, 0)
        self.assertIsNone(result)
        self.assertEqual(requests, [])

    def test_extraction_excludes_natural_and_rejects_post_final_tools(self):
        trace, _ = self.fixture()
        natural = deepcopy(trace)
        natural["control_events"] = []
        self.assertIsNone(R.extract_prefix(natural, self.contracts[R.PROFILES[0]], ["report.md.md"]))
        trace["messages"].append(deepcopy(trace["messages"][2]))
        with self.assertRaisesRegex(ValueError, "after the final"):
            R.extract_prefix(trace, self.contracts[R.PROFILES[0]], ["report.md.md"])

    def test_entry_patch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "entrypoint changed"):
            R.generate_engine("unexpected source", "fragment")

    def test_saved_inspect_wire_preserves_argument_spacing_and_normalizes_temperature(self):
        _, value = self.fixture()
        request = deepcopy(value["request_template"])
        request["temperature"] = 0.0
        request["extra_headers"] = {"x-irid": "audit-only"}
        request["extra_body"] = {"chat_template_kwargs": request.pop("chat_template_kwargs")}
        wire = deepcopy(value["wire_prefix"]) + [{"role": "user", "content": value["original_final_prompt"]}]
        original_args = wire[2]["tool_calls"][0]["function"]["arguments"]
        wire[2]["tool_calls"][0]["function"]["arguments"] = "attachment://arguments"
        sample = {"id": "q1", "attachments": {"arguments": original_args}, "events_data": {"calls": wire},
                  "events": [{"event": "model"}, {"event": "model", "call": {"call_key": "messages",
                             "call_refs": [[0, 2], [2, 5]], "request": request}}]}
        path = self.root / "fixture.eval"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("samples/q1_epoch_1.json", json.dumps(sample))
        recovered = R.saved_wire_request(path, value, self.contracts[R.PROFILES[0]])
        self.assertEqual(recovered["wire_prefix"], value["wire_prefix"])
        self.assertEqual(recovered["original_request_sha256"], value["original_request_sha256"])
        self.assertIsInstance(recovered["request_template"]["temperature"], int)

    def test_preflight_never_starts_serving(self):
        bundle = self.root / "minimal-bundle"
        bundle.mkdir()
        R.write_json(bundle / "plan.json", {"planned_final_chains": 236, "unique_queries": 59, "runtime": "frozen"})
        with patch.object(R, "verify_bundle"), patch.object(R, "check_inputs"), \
                patch.object(R, "bundle_env", return_value={"TENSOR_PARALLEL_SIZE": "4"}), patch.object(R.os, "execvpe") as execute:
            R.main(["--bundle", str(bundle), "--preflight-only"])
        execute.assert_not_called()

    def test_environment_dry_run_never_starts_serving(self):
        bundle = self.root / "env-dry-bundle"
        bundle.mkdir()
        R.write_json(bundle / "plan.json", {"planned_final_chains": 236, "unique_queries": 59, "runtime": "frozen"})
        with patch.object(R, "verify_bundle"), patch.object(R, "check_inputs"), \
                patch.object(R, "bundle_env", return_value={"TENSOR_PARALLEL_SIZE": "4"}), \
                patch.dict(os.environ, {"DRY_RUN": "1"}), patch.object(R.os, "execvpe") as execute:
            R.main(["--bundle", str(bundle)])
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
