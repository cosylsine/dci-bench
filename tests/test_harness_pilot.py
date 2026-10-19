"""CPU regression acceptance for profile -> Pi -> Inspect -> HTTP.

Each arm uses a fresh process because protocol selection is immutable at import.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def wire_case(case, destination):
    from inspect_ai import eval as inspect_eval
    from dci_bench.backends.manifest import inspect_generation_kwargs
    from dci_bench.protocol.contracts import CONTRACT_SHA256, GENERATION_CONFIG
    from dci_bench.tasks.mteb_llm_retrieval import mteb_llm_retrieval

    requests = []
    final_responses = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            nonlocal final_responses
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            turn = len(requests)
            name = None
            arguments = {}
            tools_enabled = body.get("tools") and body.get("tool_choice") != "none"
            read_only = [tool["function"]["name"] for tool in body.get("tools", [])] == ["read"]
            if case in ("forced_read", "ignored_read", "empty_required_read", "varied_search"):
                if tools_enabled:
                    name = "read" if read_only and case != "ignored_read" else "bash"
                    if case == "forced_read" and turn >= 5:
                        name = "read" if turn == 5 else None
                    if case == "varied_search" and turn >= 8:
                        name = None
            elif case in ("cold_first_turn", "near_deadline", "past_deadline"):
                name = "bash" if turn == 1 else "read" if turn == 2 else None
            elif case in ("search_growth", "calibrated_continuation"):
                if tools_enabled:
                    name = "read" if turn == 1 else "bash"
                else:
                    final_responses += 1
            elif case == "read_reserve":
                if tools_enabled:
                    last_user = next(message["content"] for message in reversed(body["messages"])
                                     if message["role"] == "user")
                    has_read = any(call["function"]["name"] == "read"
                                   for message in body["messages"]
                                   for call in message.get("tool_calls", []))
                    name = "read" if "Inspect a relevant candidate with read now" in last_user and not has_read else "bash"
                else:
                    final_responses += 1
            elif case == "stable_search":
                name = "bash" if turn <= 4 else "read" if turn == 5 else None
            elif case == "bash_stdin":
                name = "bash" if turn <= 5 else "read" if turn == 6 else None
            elif case in ("budget", "repair_budget"):
                if turn <= 2 and tools_enabled:
                    name = "read" if turn == 1 else "bash"
            elif case == "no_read":
                name = "bash" if turn <= 3 else None
            elif case == "calibrated_overflow":
                name = "read" if turn == 1 else None
            elif turn <= 3:
                name = "read"
            elif case == "blocked" and turn == 4:
                name = "bash"
            if name == "read":
                arguments = {"path": "/workspace/corpus/report.md.md"}
                if case == "forced_read":
                    # Real regression: bash cd output supplies a bare filename.
                    arguments = {"path": "report.md.md" if turn == 4 else "l%27exemple.md"}
                if case == "varied_search":
                    arguments = {"path": "./report.md.md"}
                if case == "encoded_id_repair":
                    arguments = {"path": "/workspace/corpus/l%27exemple.md"}
                if case == "empty_required_read":
                    arguments["offset"] = 99999
            elif name == "bash":
                arguments = {"command": "printf 'listing only'"}
                if case == "read_reserve":
                    arguments = {"command": f"printf 'search-{turn}'"}
                if case == "varied_search":
                    arguments = {"command": f"printf 'candidate-{turn}: report.md.md'"}
                if case == "search_growth":
                    arguments = {"command": "printf 'x%.0s' {1..19000}" if turn == 2 else f"printf 'search-{turn}'"}
                if case == "calibrated_continuation":
                    arguments = {"command": f"printf 'search-{turn}'"}
                if case == "bash_stdin":
                    commands = [
                        "cd /workspace/corpus && rg -l 'Public health' | head -5",
                        "printf 'needle\\n' | rg needle",
                        "rg -l 'Public health' /workspace/corpus",
                        "grep -l 'Public health' /workspace/corpus/*.md",
                        "cd /workspace/corpus && rg -l NO_MATCH",
                    ]
                    arguments = {"command": commands[turn - 1], "timeout_seconds": 1}
                if case == "stable_search":
                    commands = [
                        "cd /workspace/corpus && rg -il 'public health' | head -1",
                        "rg -il 'public health' /workspace/corpus | head -1",
                        "rg --files /workspace/corpus | head -1",
                        'test -r "$RIPGREP_CONFIG_PATH" && test ! -w "$RIPGREP_CONFIG_PATH" '
                        '&& printf "config-read-only\\n" && printf "z\\na\\n" | rg .',
                    ]
                    arguments = {"command": commands[turn - 1]}
            if name:
                message = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"tool_{turn}", "type": "function", "function": {
                        "name": name, "arguments": json.dumps(arguments)}}]}
                reason = "tool_calls"
            else:
                text = "Here is the answer." if (case == "repair" and turn == 4) or case == "repair_budget" else '{"ranked_doc_ids":["report.md"]}'
                if case in ("read_reserve", "search_growth", "calibrated_continuation") and final_responses == 1:
                    text = "Here is the answer."
                if case == "id_repair" and turn == 4:
                    text = '{"ranked_doc_ids":["report.md.md"]}'
                if case == "encoded_id_repair":
                    text = json.dumps({"ranked_doc_ids": ["l%27exemple" if turn == 4 else "l'exemple"]})
                if case == "unknown_id":
                    text = '{"ranked_doc_ids":["invented"]}'
                if case == "forced_read":
                    text = json.dumps({"ranked_doc_ids": ["report.md", "l'exemple"]})
                message = {"role": "assistant", "content": text}
                reason = "stop"
            input_tokens = 35000 if case in ("budget", "repair_budget") else 10000 if case == "read_reserve" else 100
            if case == "search_growth":
                input_tokens = 10000 if turn <= 2 else 30000
            if case == "calibrated_continuation":
                input_tokens = 5000
            if case == "calibrated_overflow" and turn == 2:
                input_tokens = 127875  # Returned usage crosses the limit despite valid JSON.
            payload = json.dumps({
                "id": f"reply_{turn}", "object": "chat.completion", "created": 0,
                "model": "pilot-wire", "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": {"prompt_tokens": input_tokens, "completion_tokens": 25, "total_tokens": input_tokens + 25},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    root = Path(destination)
    task_name = "LLMPublicHealthQA"
    metadata = root / "metadata" / task_name
    corpus = root / "workspaces" / task_name / "corpus"
    metadata.mkdir(parents=True)
    corpus.mkdir(parents=True)
    (metadata / "queries.jsonl").write_text('{"_id":"Q1","text":"Which document concerns public health?"}\n')
    (metadata / "qrels.json").write_text('{"Q1":{"report.md":1}}')
    (corpus / "report.md.md").write_text("Public health evidence.\n")
    (corpus / "l%27exemple.md").write_text("A second public health document.\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            bridge_port = listener.getsockname()[1]
        task = mteb_llm_retrieval(
            task_name=task_name, query_ids=["Q1"], workspace_root=str(root / "workspaces"),
            metadata_root=str(root / "metadata"), output_dir=str(root / "output"),
            bridge_port=bridge_port, model_context_window=65536,
        )
        logs = inspect_eval(
            task, model="openai-api/local/pilot-wire", model_base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model_args={"api_key": "test", "responses_api": False}, log_dir=str(root / "logs"),
            display="none", max_retries=0, **inspect_generation_kwargs(),
        )
        assert logs[0].status == "success", str(logs[0].error)
        final = json.loads((root / "output" / "Q1" / "final.json").read_text())
        assert final["base_contract_sha256"] == CONTRACT_SHA256
        for request in requests:
            assert request["temperature"] == GENERATION_CONFIG["temperature"]
            assert request["max_tokens"] == 1024
            assert request["seed"] == 0
            assert request["chat_template_kwargs"] == {"enable_thinking": False}
            assert "qrels" not in json.dumps(request)
        (root / "wire.json").write_text(json.dumps(requests))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class PilotContractTest(unittest.TestCase):
    def test_stable_and_selective_have_only_declared_contract_changes(self):
        repo = Path(__file__).resolve().parents[1]
        plan = json.loads((repo / "scripts/harness_selection_round3_plan.json").read_text())
        snapshots = {}
        for profile in ("budgeted", "budgeted-stable", "budgeted-selective"):
            output = subprocess.check_output([
                sys.executable, str(repo / "scripts/run_harness_pilot.py"), "--describe-profile",
            ], env={**os.environ, "DCI_PROTOCOL_PROFILE": profile}, text=True)
            payload = json.loads(output)
            snapshots[profile] = payload["contract"]
            if profile in plan["profiles"]:
                self.assertEqual(payload["sha256"], plan["profiles"][profile]["base_contract_sha256"])
        baseline, stable, selective = (snapshots[name] for name in snapshots)
        self.assertEqual(stable["benchmark_contract_version"], "dci-mvp-v3.4-pilot")
        self.assertEqual(stable["budget_control"], baseline["budget_control"])
        self.assertEqual(stable["search_policy"], selective["search_policy"])
        self.assertEqual(stable["search_policy"]["ripgrep_config"], "--sort=path\n")
        for field in ("final_prompt", "final_repair_prompt"):
            self.assertNotEqual(selective["budget_control"][field], stable["budget_control"][field])
            selective["budget_control"][field] = stable["budget_control"][field]
        selective["budget_control"]["version"] = stable["budget_control"]["version"]
        selective["protocol_profile"] = stable["protocol_profile"]
        self.assertEqual(selective, stable)
        stable.pop("search_policy")
        for field in ("benchmark_contract_version", "protocol_profile"):
            stable[field] = baseline[field]
        self.assertEqual(stable, baseline)

    def test_stable_config_drift_fails_before_any_model_request(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("contracts.py", "contract.json", "pilot_profiles.json"):
                shutil.copyfile(repo / "dci_bench/protocol" / name, root / name)
            (root / "stable-rg.conf").write_text("--sort=none\n")
            result = subprocess.run([sys.executable, str(root / "contracts.py")],
                                    env={**os.environ, "DCI_PROTOCOL_PROFILE": "budgeted-stable"},
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stable-rg.conf does not match", result.stderr)

    def test_calibration_changes_only_forecast_parameters_and_preserves_baseline_hash(self):
        repo = Path(__file__).resolve().parents[1]
        snapshots = {}
        for profile in ("budgeted", "budgeted-calibrated"):
            env = {**os.environ, "DCI_PROTOCOL_PROFILE": profile}
            output = subprocess.check_output([
                sys.executable, "-c",
                "import json; from dci_bench.protocol.contracts import CONTRACT_SHA256, protocol_contract; "
                "print(json.dumps({'sha256': CONTRACT_SHA256, 'contract': protocol_contract()}))",
            ], cwd=repo, env=env, text=True)
            snapshots[profile] = json.loads(output)
        self.assertEqual(snapshots["budgeted"]["sha256"],
                         "e3cbd15c8014c3cd66b350237a6e9c1cdae369c5e1b6d78cf5a022d1b1a8a22d")
        baseline = snapshots["budgeted"]["contract"]
        candidate = snapshots["budgeted-calibrated"]["contract"]
        self.assertNotEqual(snapshots["budgeted"]["sha256"], snapshots["budgeted-calibrated"]["sha256"])
        candidate["protocol_profile"] = baseline["protocol_profile"]
        policy = candidate["budget_control"]
        self.assertEqual(policy.pop("forecast_bytes_per_token"), 2)
        self.assertEqual(policy.pop("forecast_request_safety_tokens"), 1024)
        self.assertEqual(policy["version"], "budgeted-finalize-v6-calibrated")
        policy["version"] = baseline["budget_control"]["version"]
        self.assertEqual(candidate, baseline)


@unittest.skipUnless(shutil.which("node") and shutil.which("bwrap"), "Node and bubblewrap required")
class HarnessPilotTest(unittest.TestCase):
    def tool_choices(self, requests):
        # Inspect normalizes "none" by omitting both tools and tool_choice.
        return [r.get("tool_choice", "none" if not r.get("tools") else "auto") for r in requests]

    def run_case(self, profile, case):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "DCI_PROTOCOL_PROFILE": profile,
                   "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
            repo = Path(__file__).resolve().parents[1]
            env["PYTHONPATH"] = str(repo)
            if case in ("cold_first_turn", "near_deadline", "past_deadline"):
                # Simulate elapsed request time in Pi without sleeping
                # or changing the production clock/deadline configuration.
                deadline = "Number(process.argv[process.argv.indexOf('--deadline-at-ms') + 1])"
                elapsed_ms = {
                    "cold_first_turn": "220000",
                    "near_deadline": f"{deadline} - nativeNow() - 60000",
                    "past_deadline": f"{deadline} - nativeNow() + 1000",
                }[case]
                preload = Path(tmp) / "cold-clock.cjs"
                preload.write_text(
                    "const nativeNow = Date.now; const nativeFetch = globalThis.fetch;\n"
                    "let elapsed = 0; Date.now = () => nativeNow() + elapsed;\n"
                    "globalThis.fetch = async (...args) => {\n"
                    "  const response = await nativeFetch(...args);\n"
                    f"  if (!elapsed && String(args[0]).endsWith('/chat/completions')) elapsed = {elapsed_ms};\n"
                    "  return response;\n"
                    "};\n"
                )
                env["NODE_OPTIONS"] = f"{env.get('NODE_OPTIONS', '')} --require={preload}".strip()
            process = subprocess.run([sys.executable, __file__, "--wire-case", case, tmp],
                                     cwd=repo, env=env, text=True, capture_output=True, timeout=90)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            root = Path(tmp)
            return (json.loads((root / "output/Q1/final.json").read_text()),
                    json.loads((root / "wire.json").read_text()))

    def test_v3_stays_auto_and_has_no_budget_prompt(self):
        final, requests = self.run_case("v3", "plain")
        self.assertTrue(final["valid_output"])
        self.assertEqual(final["contract_version"], "dci-mvp-v3.3")
        self.assertEqual(final["control_events"], [])
        self.assertTrue(all(r["tool_choice"] == "auto" for r in requests))
        self.assertNotIn("Execution budget:", json.dumps(requests[0]))

    def test_stable_and_selective_final_prompts_reach_http_with_identical_search(self):
        outputs = [self.run_case(profile, "repair") for profile in ("budgeted-stable", "budgeted-selective")]
        for final, requests in outputs:
            self.assertTrue(final["valid_output"], final)
            self.assertEqual(final["contract_version"], "dci-mvp-v3.4-pilot")
            self.assertEqual(final["contract"]["search_policy"]["version"], "stable-rg-v1")
            self.assertEqual(final["contract"]["budget_control"]["reserve_tokens"], 32000)
            self.assertNotIn("forecast_bytes_per_token", final["contract"]["budget_control"])
            self.assertEqual(self.tool_choices(requests), ["auto"] * 3 + ["none"] * 2)
            self.assertEqual(final["repair_attempts"], 1)
            self.assertEqual(final["usage_summary"]["total_model_tokens"], 625)
            self.assertIn("stable-rg-v1", json.dumps(requests[0]))
            self.assertIn("--sort=path", json.dumps(requests[0]))
        self.assertEqual(outputs[0][1][:3], outputs[1][1][:3])
        for request in outputs[1][1][-2:]:
            self.assertIn("Keep multiple relevant documents" if request == outputs[1][1][-1]
                          else "Retain multiple relevant documents", request["messages"][-1]["content"])
            self.assertIn("shared keywords", request["messages"][-1]["content"])

    def test_stable_search_sorts_files_before_head_preserving_pipe_line_order(self):
        final, requests = self.run_case("budgeted-stable", "stable_search")
        self.assertTrue(final["valid_output"], final)
        outputs = [request["messages"][-1]["content"] for request in requests[1:5]]
        for output in outputs[:3]:
            self.assertIn("l%27exemple.md", output)
            self.assertNotIn("report.md.md", output)
            self.assertIn("exit_code=0", output)
        self.assertIn("config-read-only\nz\na\n", outputs[3])

    def test_selective_retains_read_gate_and_actual_usage_limit(self):
        final, _ = self.run_case("budgeted-selective", "no_read")
        self.assertFalse(final["valid_output"])
        self.assertEqual(final["failure"]["kind"], "tool_failure")
        final, _ = self.run_case("budgeted-selective", "calibrated_overflow")
        self.assertFalse(final["valid_output"])
        self.assertEqual(final["failure"]["kind"], "token_limit")

    def test_repeat_finalization_and_repair_disable_tools_and_count_usage(self):
        final, requests = self.run_case("budgeted", "repair")
        self.assertTrue(final["valid_output"])
        self.assertEqual(self.tool_choices(requests), ["auto"] * 3 + ["none"] * 2)
        self.assertEqual(final["repair_attempts"], 1)
        self.assertEqual(final["usage_summary"]["total_model_tokens"], 625)
        self.assertEqual(final["control_events"][0]["action"], "finalize")
        self.assertEqual(final["contract"]["max_total_model_tokens"], 128000)
        self.assertIn("Tools remain disabled.", requests[-1]["messages"][-1]["content"])
        self.assertNotIn("Inspect /workspace/corpus now", requests[-1]["messages"][-1]["content"])

    def test_missing_read_reserves_read_final_and_repair_requests(self):
        # At 10025 tokens per turn, the old two-request reserve delayed the read
        # until turn 11, leaving no affordable repair after the invalid final.
        final, requests = self.run_case("budgeted", "read_reserve")
        self.assertTrue(final["valid_output"], final)
        self.assertEqual(final["repair_attempts"], 1)
        self.assertEqual(final["usage_summary"]["total_model_tokens"], len(requests) * 10025)
        self.assertLess(final["usage_summary"]["total_model_tokens"], 128000)
        self.assertTrue(final["body_evidence"]["read_receipts"])
        controls = final["control_events"]
        self.assertEqual([event["action"] for event in controls], ["require_read", "finalize"])
        self.assertLess(controls[0]["turn"], controls[1]["turn"])
        self.assertGreater(controls[0]["remaining_tokens"], controls[1]["remaining_tokens"])
        choices = self.tool_choices(requests)
        self.assertEqual(choices[-2:], ["none"] * 2)
        self.assertEqual(choices[-3], {"type": "function", "function": {"name": "read"}})
        self.assertIn("Tools remain disabled.", requests[-1]["messages"][-1]["content"])
        self.assertNotIn("Inspect /workspace/corpus now", requests[-1]["messages"][-1]["content"])

    def test_search_result_growth_preserves_a_final_repair(self):
        # A 19 KB tool result expands later requests after a valid read. The old
        # reserve paid for more searching, then had no room to repair a bad final.
        final, requests = self.run_case("budgeted", "search_growth")
        self.assertTrue(final["valid_output"], final)
        self.assertEqual(self.tool_choices(requests), ["auto", "auto", "none", "none"])
        self.assertEqual(final["repair_attempts"], 1)
        self.assertEqual(final["usage_summary"]["total_model_tokens"], 80100)
        self.assertTrue(final["body_evidence"]["read_receipts"])
        self.assertGreater(len(requests[2]["messages"][-2]["content"]), 19000)

    def test_calibrated_profile_continues_search_and_keeps_final_repair(self):
        baseline, old_requests = self.run_case("budgeted", "calibrated_continuation")
        calibrated, new_requests = self.run_case("budgeted-calibrated", "calibrated_continuation")
        for final, requests in ((baseline, old_requests), (calibrated, new_requests)):
            self.assertTrue(final["valid_output"], final)
            self.assertEqual(final["repair_attempts"], 1)
            self.assertEqual(self.tool_choices(requests)[-2:], ["none", "none"])
            self.assertLess(final["usage_summary"]["total_model_tokens"], 128000)
        self.assertEqual(len(old_requests), 13)
        self.assertEqual(len(new_requests), 19)
        self.assertEqual(new_requests[:11], old_requests[:11])
        self.assertEqual(calibrated["control_events"][0]["forecast"]["bytesPerToken"], 2)

    def test_calibrated_profile_handles_large_tool_growth_with_a_repair(self):
        final, requests = self.run_case("budgeted-calibrated", "search_growth")
        self.assertTrue(final["valid_output"], final)
        self.assertEqual(final["repair_attempts"], 1)
        self.assertEqual(self.tool_choices(requests)[-2:], ["none", "none"])
        self.assertLess(final["usage_summary"]["total_model_tokens"], 128000)

    def test_calibrated_profile_retains_read_and_hard_usage_guards(self):
        final, _requests = self.run_case("budgeted-calibrated", "no_read")
        self.assertFalse(final["valid_output"])
        self.assertEqual(final["failure"]["kind"], "tool_failure")
        final, requests = self.run_case("budgeted-calibrated", "calibrated_overflow")
        self.assertFalse(final["valid_output"])
        self.assertEqual(final["failure"]["kind"], "token_limit")
        self.assertEqual(final["ranked_doc_ids"], [])
        self.assertGreaterEqual(final["usage_summary"]["total_model_tokens"], 128000)
        self.assertEqual(len(requests), 2)

    def test_calibrated_profile_retains_conservative_repair_dispatch_guard(self):
        final, requests = self.run_case("budgeted-calibrated", "repair_budget")
        self.assertFalse(final["valid_output"])
        self.assertEqual(final["failure"]["kind"], "token_limit")
        self.assertEqual(len(requests), 3)
        self.assertEqual(final["usage_summary"]["total_model_tokens"], 105075)

    def test_budget_reserves_final_turn_before_hard_limit(self):
        final, requests = self.run_case("budgeted", "budget")
        self.assertTrue(final["valid_output"])
        self.assertEqual(self.tool_choices(requests), ["auto", "none"])
        self.assertEqual(final["usage_summary"]["total_model_tokens"], 70050)

    def test_nudge_does_not_waive_read_requirement(self):
        final, requests = self.run_case("budgeted", "no_read")
        self.assertFalse(final["valid_output"])
        self.assertEqual(final["body_evidence"]["read_receipts"], [])
        self.assertEqual(final["control_events"][0]["action"], "require_read")
        self.assertEqual(final["failure"]["kind"], "tool_failure")
        self.assertEqual(len(requests), 5)
        self.assertTrue(all([tool["function"]["name"] for tool in r["tools"]] == ["read"] for r in requests[3:]))

    def test_filename_ids_are_repaired_without_normalizing_canonical_suffixes(self):
        final, requests = self.run_case("budgeted", "id_repair")
        self.assertTrue(final["valid_output"], final)
        self.assertEqual(final["ranked_doc_ids"], ["report.md"])
        self.assertEqual(final["repair_attempts"], 1)
        self.assertIn("Unknown original document IDs", final["repair_reasons"][0])
        self.assertEqual(self.tool_choices(requests)[-2:], ["none", "none"])
        self.assertIn('"doc_id":"report.md"', requests[1]["messages"][-1]["content"])

    def test_percent_encoded_ids_are_repaired_using_exact_corpus_mapping(self):
        final, _requests = self.run_case("budgeted", "encoded_id_repair")
        self.assertTrue(final["valid_output"], final)
        self.assertEqual(final["ranked_doc_ids"], ["l'exemple"])
        self.assertEqual(final["repair_attempts"], 1)

    def test_invented_ids_never_become_valid_outputs(self):
        final, _requests = self.run_case("budgeted", "unknown_id")
        self.assertFalse(final["valid_output"])
        self.assertEqual(final["ranked_doc_ids"], [])
        self.assertEqual(final["repair_attempts"], 3)
        self.assertIn("Unknown original document IDs", final["failure_reason"])

    def test_required_read_accepts_basename_and_resumes_candidate_verification(self):
        final, requests = self.run_case("budgeted", "forced_read")
        self.assertTrue(final["valid_output"], final)
        self.assertEqual(len(requests), 6)
        self.assertEqual(self.tool_choices(requests)[3], {"type": "function", "function": {"name": "read"}})
        self.assertEqual([tool["function"]["name"] for tool in requests[3]["tools"]], ["read"])
        self.assertEqual(final["body_evidence"]["read_receipts"][0]["doc_id"], "report.md")
        self.assertEqual([event["action"] for event in final["control_events"]], ["require_read", "resume_search"])
        self.assertEqual(self.tool_choices(requests)[4:], ["auto", "auto"])
        self.assertEqual({tool["function"]["name"] for tool in requests[4]["tools"]}, {"read", "bash"})
        self.assertEqual(final["ranked_doc_ids"], ["report.md", "l'exemple"])
        self.assertEqual(len(final["body_evidence"]["read_receipts"]), 2)

    def test_backend_ignoring_read_phase_cannot_execute_more_bash(self):
        final, requests = self.run_case("budgeted", "ignored_read")
        self.assertFalse(final["valid_output"])
        self.assertEqual(final["failure"]["kind"], "tool_failure")
        self.assertEqual(len(requests), 5)
        # The fifth request carries the rejection of the fourth bash call.
        self.assertIn("not found", requests[4]["messages"][-2]["content"].lower())

    def test_empty_reads_cannot_satisfy_required_read_phase(self):
        final, requests = self.run_case("budgeted", "empty_required_read")
        self.assertFalse(final["valid_output"])
        self.assertEqual(final["failure"]["kind"], "tool_failure")
        self.assertEqual(final["body_evidence"]["read_receipts"], [])
        self.assertEqual(len(requests), 5)

    def test_varied_searches_cannot_avoid_the_read_phase(self):
        final, requests = self.run_case("budgeted", "varied_search")
        self.assertTrue(final["valid_output"], final)
        self.assertEqual(final["control_events"][0]["reason"], "search_turn_limit")
        self.assertEqual(len(requests), 8)
        self.assertEqual(final["control_events"][-1]["action"], "resume_search")
        self.assertEqual(self.tool_choices(requests)[-1], "auto")

    def test_slow_first_request_does_not_end_search_with_350_seconds_left(self):
        final, requests = self.run_case("budgeted", "cold_first_turn")
        self.assertTrue(final["valid_output"], final)
        self.assertEqual(len(requests), 3)
        self.assertEqual(self.tool_choices(requests), ["auto", "auto", "auto"])
        self.assertEqual(final["control_events"], [])
        self.assertEqual(final["ranked_doc_ids"], ["report.md"])

    def test_less_than_90_seconds_remaining_does_not_force_finalization(self):
        final, requests = self.run_case("budgeted", "near_deadline")
        self.assertTrue(final["valid_output"], final)
        self.assertEqual(len(requests), 3)
        self.assertEqual(self.tool_choices(requests), ["auto"] * 3)
        self.assertEqual(final["control_events"], [])

    def test_hard_wall_deadline_still_rejects_late_responses(self):
        final, requests = self.run_case("budgeted", "past_deadline")
        self.assertFalse(final["valid_output"])
        self.assertEqual(final["failure"]["kind"], "timeout")
        self.assertEqual(final["ranked_doc_ids"], [])
        self.assertEqual(len(requests), 1)

    def test_repair_does_not_issue_request_without_remaining_budget(self):
        final, requests = self.run_case("budgeted", "repair_budget")
        self.assertFalse(final["valid_output"])
        self.assertEqual(final["failure"]["kind"], "token_limit")
        self.assertEqual(len(requests), 3)
        self.assertEqual(final["usage_summary"]["total_model_tokens"], 105075)

    def test_temperature_profile_reaches_actual_http(self):
        final, requests = self.run_case("v3-temp1", "plain")
        self.assertTrue(final["valid_output"])
        self.assertEqual(final["control_events"], [])
        self.assertTrue(all(r["temperature"] == 1 for r in requests))

    def test_backend_cannot_execute_tools_after_finalization(self):
        final, requests = self.run_case("budgeted", "blocked")
        self.assertTrue(final["valid_output"])
        self.assertEqual(final["tool_calls"], 3)
        self.assertEqual(final["repair_attempts"], 1)
        self.assertEqual(final["usage_summary"]["total_model_tokens"], 625)
        self.assertEqual(self.tool_choices(requests)[-1], "none")

    def test_bash_searches_cwd_without_stdin_and_preserves_shell_pipelines(self):
        final, requests = self.run_case("v3", "bash_stdin")
        self.assertTrue(final["valid_output"])
        self.assertEqual(len(requests), 7)
        outputs = [request["messages"][-1]["content"] for request in requests[1:6]]
        for index, output in enumerate(outputs):
            self.assertNotIn("timeout", output)
            self.assertIn(f"exit_code={1 if index == 4 else 0}", output)
        for index in (0, 2, 3):
            self.assertIn("report.md.md", outputs[index])
        self.assertIn("needle", outputs[1])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--wire-case":
        wire_case(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
