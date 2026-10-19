"""CPU-only checks for the bounded synthetic serving warmup."""

from __future__ import annotations

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import unittest


REPO = Path(__file__).resolve().parents[1]
PYTHON = "/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python"


def load_warmup_module():
    script = REPO / "scripts/harness_pilot_warmup.py"
    spec = importlib.util.spec_from_file_location("harness_pilot_warmup_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WarmupServer:
    def __init__(self, *, mode: str = "success", delay: float = 0.0, named_finish: str = "stop"):
        self.mode = mode
        self.delay = delay
        self.named_finish = named_finish
        self.requests: list[dict] = []
        self.responses: list[bytes] = []
        self.active = 0
        self.peak_active = 0
        self.lock = threading.Lock()

        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_args):
                pass

            def do_POST(self):
                self.close_connection = True
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with owner.lock:
                    owner.requests.append(body)
                    owner.active += 1
                    owner.peak_active = max(owner.peak_active, owner.active)
                try:
                    if owner.delay:
                        time.sleep(owner.delay)
                    if owner.mode in ("http_error", "large_http_error"):
                        error_body = json.dumps({"error": "synthetic failure"}).encode()
                        if owner.mode == "large_http_error":
                            error_body = ("失败" * 10000).encode("utf-8")
                        self.send_response(500)
                        self.send_header("Content-Length", str(len(error_body)))
                        self.end_headers()
                        try:
                            self.wfile.write(error_body)
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                        return
                    if owner.mode == "no_usage":
                        payload = {"choices": [{"message": {"role": "assistant", "content": "bad"},
                                                 "finish_reason": "stop"}]}
                    else:
                        if body.get("tool_choice") == "none":
                            message = {"role": "assistant", "content": "synthetic final"}
                            finish_reason = "stop"
                        else:
                            named = isinstance(body.get("tool_choice"), dict)
                            name = "read" if named else "bash"
                            if owner.mode == "unexpected_tool":
                                name = "wrong_tool"
                            if owner.mode == "missing_args":
                                arguments = "{}"
                            elif name == "read":
                                arguments = json.dumps({"path": "/workspace/corpus/__warmup__.md"})
                            else:
                                arguments = json.dumps({"command": "printf warmup"})
                            message = {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": "synthetic-call",
                                    "type": "function",
                                    "function": {"name": name, "arguments": arguments},
                                }],
                            }
                            # vLLM 0.17 uses stop for named calls and tool_calls
                            # for auto calls. Keep this distinction in fixtures.
                            finish_reason = owner.named_finish if named else "tool_calls"
                        payload = {
                            "id": "synthetic-warmup",
                            "object": "chat.completion",
                            "choices": [{"index": 0, "message": message,
                                          "finish_reason": finish_reason}],
                            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                        }
                    encoded = json.dumps(payload).encode("utf-8")
                    if owner.mode == "not_json":
                        encoded = b"synthetic invalid JSON response"
                    with owner.lock:
                        owner.responses.append(encoded)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    try:
                        self.wfile.write(encoded)
                        self.wfile.flush()
                    except BrokenPipeError:
                        pass
                finally:
                    with owner.lock:
                        owner.active -= 1

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class HarnessPilotWarmupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.warmup = load_warmup_module()

    def run_main(self, server: WarmupServer, output: Path, *extra: str) -> int:
        return self.warmup.main([
            "--base-url", server.base_url,
            "--model", "synthetic-model",
            "--output", str(output),
            "--concurrency", "2",
            "--deadline-seconds", "5",
            "--http-timeout-seconds", "1",
            *extra,
        ])

    def test_success_runs_two_rounds_and_reaches_configured_batch_concurrency(self):
        with tempfile.TemporaryDirectory() as temporary, WarmupServer(delay=0.04) as server:
            output = Path(temporary) / "warmup.json"
            self.assertEqual(self.run_main(server, output), 0)
            artifact = json.loads(output.read_text())
        self.assertTrue(artifact["synthetic"])
        self.assertEqual(artifact["phase"], "pre_evaluation")
        self.assertTrue(artifact["success"])
        self.assertEqual(artifact["rounds"], 2)
        self.assertEqual(artifact["concurrency"], 2)
        self.assertEqual(artifact["requested_requests"], 12)
        self.assertEqual(artifact["completed_requests"], 12)
        self.assertEqual(artifact["shapes"], ["auto_tool", "named_read", "tools_disabled_final"])
        self.assertEqual(len(artifact["batches"]), 6)
        self.assertEqual(artifact["usage"], {"prompt_tokens": 132, "completion_tokens": 84, "total_tokens": 216})
        self.assertGreaterEqual(server.peak_active, 2)
        self.assertEqual(len(server.requests), 12)
        for request in server.requests:
            self.assertEqual(request["temperature"], 0)
            self.assertEqual(request["max_tokens"], 1024)
            self.assertEqual(request["seed"], 0)
            self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": False})
            self.assertNotIn("qrels", json.dumps(request))
        self.assertEqual(
            [request["tool_choice"] for request in server.requests[:2]], ["auto", "auto"]
        )
        self.assertEqual(
            [tool["function"]["name"] for tool in server.requests[0]["tools"]],
            ["read", "bash"],
        )
        self.assertIn("command='printf warmup'", server.requests[0]["messages"][-1]["content"])
        self.assertEqual(
            [request["tool_choice"] for request in server.requests[2:4]],
            [{"type": "function", "function": {"name": "read"}}] * 2,
        )
        self.assertIn(
            "path='/workspace/corpus/__warmup__.md'",
            server.requests[2]["messages"][-1]["content"],
        )
        disabled = server.requests[4:6] + server.requests[10:12]
        self.assertTrue(all("tools" not in request for request in disabled))
        self.assertTrue(all(request["tool_choice"] == "none" for request in disabled))

    def test_http_failure_writes_false_artifact_and_aborts(self):
        with tempfile.TemporaryDirectory() as temporary, WarmupServer(mode="http_error") as server:
            output = Path(temporary) / "warmup.json"
            self.assertEqual(self.run_main(server, output), 2)
            artifact = json.loads(output.read_text())
        self.assertFalse(artifact["success"])
        self.assertEqual(artifact["error"]["kind"], "http_error")
        self.assertEqual(artifact["error"]["details"]["http_status"], 500)
        self.assertIn("synthetic failure", artifact["error"]["details"]["response_excerpt"])

    def test_named_tool_calls_finish_remains_compatible(self):
        with tempfile.TemporaryDirectory() as temporary, WarmupServer(named_finish="tool_calls") as server:
            output = Path(temporary) / "warmup.json"
            self.assertEqual(self.run_main(server, output), 0)
            artifact = json.loads(output.read_text())
        self.assertTrue(artifact["success"])
        self.assertEqual(artifact["completed_requests"], 12)

    def test_named_length_failure_records_actual_response_and_cluster_log(self):
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, WarmupServer(named_finish="length") as server:
            output = Path(temporary) / "warmup.json"
            with redirect_stderr(stderr):
                self.assertEqual(self.run_main(server, output, "--concurrency", "4"), 2)
            artifact = json.loads(output.read_text())
        self.assertFalse(artifact["success"])
        self.assertEqual(artifact["completed_requests"], 4)
        self.assertEqual(artifact["accounting_scope"], "successful_batches_only")
        self.assertEqual(artifact["error"]["kind"], "unexpected_finish_reason")
        detail = artifact["error"]["details"]
        self.assertEqual(detail["round"], 1)
        self.assertEqual(detail["shape"], "named_read")
        self.assertIn(detail["request_index"], range(4))
        self.assertEqual(detail["http_status"], 200)
        self.assertEqual(detail["finish_reason"], "length")
        self.assertEqual(detail["tool_names"], ["read"])
        self.assertFalse(detail["response_truncated"])
        self.assertIn(detail["response_excerpt"].encode(), server.responses)
        response = json.loads(detail["response_excerpt"])
        self.assertEqual(response["usage"]["total_tokens"], 18)
        self.assertEqual(response["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "read")
        log = stderr.getvalue()
        self.assertIn('"shape": "named_read"', log)
        self.assertIn('"finish_reason": "length"', log)
        self.assertIn("response_excerpt", log)
        self.assertIn(str(output), log)
        self.assertTrue(all(request["tool_choice"] != "none" for request in server.requests))

    def test_named_stop_does_not_waive_tool_and_usage_validation(self):
        valid = {
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "", "tool_calls": [{
                "id": "test-read", "type": "function", "function": {
                    "name": "read", "arguments": '{"path":"/workspace/corpus/__warmup__.md"}',
                },
            }]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        self.assertEqual(self.warmup._validate_response(valid, "named_read"), valid["usage"])
        for case in ("no_call", "wrong_tool", "missing_args", "bad_json", "wrong_path", "no_usage", "length", "missing_finish", "error"):
            with self.subTest(case=case):
                invalid = deepcopy(valid)
                choice = invalid["choices"][0]
                function = choice["message"]["tool_calls"][0]["function"]
                if case == "no_call":
                    choice["message"].pop("tool_calls")
                    choice["message"]["content"] = "I cannot call tools"
                elif case == "wrong_tool":
                    function["name"] = "bash"
                elif case == "missing_args":
                    function["arguments"] = "{}"
                elif case == "bad_json":
                    function["arguments"] = '{"path":'
                elif case == "wrong_path":
                    function["arguments"] = '{"path":"/other.md"}'
                elif case == "no_usage":
                    invalid.pop("usage")
                elif case == "missing_finish":
                    choice.pop("finish_reason")
                else:
                    choice["finish_reason"] = case
                with self.assertRaises(self.warmup.WarmupError):
                    self.warmup._validate_response(invalid, "named_read")
        # The compatibility exception belongs only to named tools.
        auto = deepcopy(valid)
        auto["choices"][0]["message"]["tool_calls"][0]["function"] = {
            "name": "bash", "arguments": '{"command":"printf warmup"}',
        }
        with self.assertRaises(self.warmup.WarmupError):
            self.warmup._validate_response(auto, "auto_tool")

    def test_large_http_error_response_capture_is_bounded(self):
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, WarmupServer(mode="large_http_error") as server:
            output = Path(temporary) / "warmup.json"
            with redirect_stderr(stderr):
                self.assertEqual(self.run_main(server, output, "--api-key", "synthetic-secret"), 2)
            artifact_text = output.read_text()
            detail = json.loads(artifact_text)["error"]["details"]
        self.assertTrue(detail["response_truncated"])
        self.assertLessEqual(len(detail["response_excerpt"].encode("utf-8")), self.warmup.RESPONSE_CAPTURE_MAX_BYTES)
        self.assertNotIn("\ufffd", detail["response_excerpt"])
        self.assertNotIn("synthetic-secret", artifact_text + stderr.getvalue())
        line = next(line for line in stderr.getvalue().splitlines() if line.startswith("[warmup] failure details: "))
        console = json.loads(line.removeprefix("[warmup] failure details: "))
        self.assertLessEqual(len(console["response_excerpt"].encode("utf-8")), 2048)
        self.assertTrue(console["response_truncated"])

    def test_non_json_response_is_retained_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as temporary, WarmupServer(mode="not_json") as server:
            output = Path(temporary) / "warmup.json"
            self.assertEqual(self.run_main(server, output), 2)
            artifact = json.loads(output.read_text())
        self.assertEqual(artifact["error"]["kind"], "response_structure")
        self.assertEqual(artifact["error"]["details"]["response_excerpt"], "synthetic invalid JSON response")

    def test_timeout_writes_false_artifact_without_retrying(self):
        with tempfile.TemporaryDirectory() as temporary, WarmupServer(delay=0.4) as server:
            output = Path(temporary) / "warmup.json"
            started = time.monotonic()
            self.assertEqual(self.run_main(server, output, "--http-timeout-seconds", "0.05"), 2)
            self.assertLess(time.monotonic() - started, 1.0)
            artifact = json.loads(output.read_text())
        self.assertFalse(artifact["success"])
        self.assertEqual(artifact["error"]["kind"], "timeout")
        self.assertEqual(len(server.requests), 2)

    def test_total_deadline_bounds_request_when_http_timeout_is_longer(self):
        with tempfile.TemporaryDirectory() as temporary, WarmupServer(delay=0.4) as server:
            output = Path(temporary) / "warmup.json"
            started = time.monotonic()
            self.assertEqual(
                self.run_main(
                    server,
                    output,
                    "--deadline-seconds",
                    "0.1",
                    "--http-timeout-seconds",
                    "5",
                ),
                2,
            )
            elapsed = time.monotonic() - started
            artifact = json.loads(output.read_text())
        self.assertLess(elapsed, 0.8)
        self.assertLess(artifact["duration_seconds"], 0.5)
        self.assertIn(artifact["error"]["kind"], {"timeout", "deadline"})

    def test_missing_usage_writes_false_artifact(self):
        with tempfile.TemporaryDirectory() as temporary, WarmupServer(mode="no_usage") as server:
            output = Path(temporary) / "warmup.json"
            self.assertEqual(self.run_main(server, output), 2)
            artifact = json.loads(output.read_text())
        self.assertFalse(artifact["success"])
        self.assertEqual(artifact["error"]["kind"], "usage_missing")

    def test_unexpected_tool_writes_false_artifact(self):
        with tempfile.TemporaryDirectory() as temporary, WarmupServer(mode="unexpected_tool") as server:
            output = Path(temporary) / "warmup.json"
            self.assertEqual(self.run_main(server, output), 2)
            artifact = json.loads(output.read_text())
        self.assertFalse(artifact["success"])
        self.assertEqual(artifact["error"]["kind"], "unexpected_tool")

    def test_missing_required_tool_argument_is_rejected_by_contract_schema(self):
        with tempfile.TemporaryDirectory() as temporary, WarmupServer(mode="missing_args") as server:
            output = Path(temporary) / "warmup.json"
            self.assertEqual(self.run_main(server, output), 2)
            artifact = json.loads(output.read_text())
        self.assertFalse(artifact["success"])
        self.assertEqual(artifact["error"]["kind"], "response_structure")

    def test_dry_run_does_not_create_artifact_or_connect(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "warmup.json"
            result = self.warmup.main([
                "--base-url", "http://127.0.0.1:1/v1",
                "--model", "synthetic-model",
                "--output", str(output),
                "--concurrency", "2",
                "--dry-run",
            ])
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())

    def test_direct_script_bootstraps_repo_without_pythonpath(self):
        with tempfile.TemporaryDirectory() as temporary, WarmupServer() as server:
            output = Path(temporary) / "warmup.json"
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [
                    PYTHON,
                    str(REPO / "scripts/harness_pilot_warmup.py"),
                    "--base-url",
                    server.base_url,
                    "--model",
                    "synthetic-model",
                    "--output",
                    str(output),
                    "--concurrency",
                    "1",
                    "--rounds",
                    "1",
                    "--deadline-seconds",
                    "5",
                    "--http-timeout-seconds",
                    "1",
                ],
                cwd=REPO,
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(output.read_text())["success"])

class HarnessPilotLauncherWarmupTest(unittest.TestCase):
    """Exercise the launcher ordering with a local fake OpenAI endpoint."""

    def test_launcher_runs_warmup_before_evaluation_and_dry_run_skips_it(self):
        with tempfile.TemporaryDirectory(prefix="dci-pilot-warmup-launcher-") as temporary:
            root = Path(temporary) / "checkout"
            (root / "scripts").mkdir(parents=True)
            shutil.copyfile(REPO / "scripts/run_harness_pilot.sh", root / "scripts/run_harness_pilot.sh")
            shutil.copyfile(REPO / "scripts/harness_pilot_warmup.py", root / "scripts/harness_pilot_warmup.py")
            (root / "scripts/run_all.py").write_text("# fixture\n")
            (root / "pi-dci/scripts").mkdir(parents=True)
            (root / "pi-dci/scripts/dci-run.ts").write_text("// fixture\n")
            events = root / "events.log"
            (root / "fake_server.py").write_text(_FAKE_SERVER)
            (root / "scripts/serve_vllm_qwen38.sh").write_text(
                "#!/bin/sh\nexec \"$FAKE_PYTHON\" \"$FAKE_SERVER\"\n"
            )
            (root / "scripts/serve_vllm_qwen38.sh").chmod(0o755)
            trusted = Path(temporary) / "trusted"
            trusted.mkdir()
            for name in ("bwrap", "rg"):
                path = trusted / name
                path.write_text("#!/bin/sh\nexit 0\n")
                path.chmod(0o755)
            bindir = Path(temporary) / "bin"
            bindir.mkdir()
            fake_bench = bindir / "bench-python"
            fake_bench.write_text(_FAKE_BENCH)
            fake_bench.chmod(0o755)
            fake_serving = bindir / "serving-python"
            fake_serving.write_text(_FAKE_SERVING)
            fake_serving.chmod(0o755)
            fake_npm = bindir / "npm"
            fake_npm.write_text(f'#!/bin/sh\necho "sandbox:$DCI_PROTOCOL_PROFILE" >> {events}\nexit 0\n')
            fake_npm.chmod(0o755)
            for name in ("node",):
                node = bindir / name
                node.write_text("#!/bin/sh\nexit 0\n")
                node.chmod(0o755)
            for missing in ("bwrap", "rg"):
                original = root / "scripts/run_harness_pilot.sh"
                original.write_text(original.read_text().replace(f"/usr/bin/{missing}", str(trusted / missing)))
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            output = root / "pilot-output"
            env = {
                **os.environ,
                "PATH": f"{bindir}:{os.environ['PATH']}",
                "BENCH_PYTHON": str(fake_bench),
                "SERVING_PYTHON": str(fake_serving),
                "REAL_PYTHON": PYTHON,
                "FAKE_PYTHON": PYTHON,
                "FAKE_SERVER": str(root / "fake_server.py"),
                "PILOT_ROOT": str(output),
                "VLLM_PORT": str(port),
                "MODEL_PATH": "/synthetic/model",
                "SERVED_MODEL_NAME": "synthetic-model",
                "DRY_RUN": "0",
                "EVENTS": str(events),
                "PYTHONPATH": str(REPO),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            result = subprocess.run(
                ["bash", str(root / "scripts/run_harness_pilot.sh"), "--profiles", "v3", "budgeted"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(events.read_text().splitlines(),
                             ["preflight", "data", "sandbox:v3", "sandbox:budgeted",
                              "runtime", "manifest", "warmup", "eval"])
            artifact = json.loads((output / "warmup.json").read_text())
            self.assertTrue(artifact["success"])
            self.assertFalse((output / "evaluation" / "_COMPLETE").exists())

            dry_events = root / "dry-events.log"
            dry_env = {**env, "EVENTS": str(dry_events), "PILOT_ROOT": str(root / "dry-output"), "DRY_RUN": "1"}
            dry_result = subprocess.run(
                ["bash", str(root / "scripts/run_harness_pilot.sh"), "--profiles", "v3"],
                cwd=root,
                env=dry_env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(dry_result.returncode, 0, dry_result.stdout + dry_result.stderr)
            self.assertFalse(dry_events.exists())
            self.assertFalse((root / "dry-output").exists())


_FAKE_SERVER = r'''import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *_args):
        pass
    def _send(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()
    def do_GET(self):
        if self.path == "/v1/models":
            self._send({"data": [{"id": os.environ["SERVED_MODEL_NAME"]}]})
        else:
            self.send_error(404)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("tool_choice") == "none":
            message = {"role": "assistant", "content": "synthetic final"}
            finish = "stop"
        else:
            name = "read" if isinstance(body.get("tool_choice"), dict) else "bash"
            arguments = (
                {"path": "/workspace/corpus/__warmup__.md"}
                if name == "read"
                else {"command": "printf warmup"}
            )
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "synthetic", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)}}]}
            finish = "stop" if name == "read" else "tool_calls"
        self._send({"choices": [{"message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

ThreadingHTTPServer(("127.0.0.1", int(os.environ["VLLM_PORT"])), Handler).serve_forever()
'''


_FAKE_BENCH = r'''#!/bin/sh
set -eu
if [ "$1" = scripts/run_harness_pilot.py ] && [ "${2:-}" = --preflight-only ]; then
  echo preflight >> "$EVENTS"
  exit 0
fi
if [ "$1" = scripts/run_all.py ]; then
  echo data >> "$EVENTS"
  exit 0
fi
if [ "$1" = -m ]; then
  output=""
  previous=""
  for arg in "$@"; do
    if [ "$previous" = --output ]; then output="$arg"; fi
    previous="$arg"
  done
  printf '%s\n' '{"model":{"model_key":"synthetic-model"},"backend":{"max_num_seqs":2},"manifest_sha256":"synthetic"}' > "$output"
  echo manifest >> "$EVENTS"
  exit 0
fi
if [ "$1" = scripts/harness_pilot_warmup.py ]; then
  echo warmup >> "$EVENTS"
  exec "$REAL_PYTHON" "$@"
fi
if [ "$1" = scripts/run_harness_pilot.py ]; then
  found_warmup=0
  for arg in "$@"; do
    if [ "$arg" = "${PILOT_ROOT}/warmup.json" ]; then found_warmup=1; fi
  done
  [ "$found_warmup" = 1 ]
  echo eval >> "$EVENTS"
  exit 0
fi
exit 0
'''


_FAKE_SERVING = r'''#!/bin/sh
set -eu
if [ "$1" = scripts/harness_pilot_runtime.py ]; then
  echo runtime >> "$EVENTS"
  previous=""
  for arg in "$@"; do
    if [ "$previous" = --output ]; then printf '%s\n' '{}' > "$arg"; fi
    previous="$arg"
  done
  exit 0
fi
if [ "$1" = -c ]; then echo 0.0.0; exit 0; fi
exit 0
'''


if __name__ == "__main__":
    unittest.main()
