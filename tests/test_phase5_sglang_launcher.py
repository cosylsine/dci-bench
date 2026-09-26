import subprocess
import unittest
from pathlib import Path


class Phase5SglangLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_phase5_with_sglang.sh"
        )
        cls.text = cls.script.read_text(encoding="utf-8")
        cls.service_script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "serve_sglang_minicpm5.sh"
        )
        cls.service_text = cls.service_script.read_text(encoding="utf-8")

    def test_shell_syntax(self):
        completed = subprocess.run(
            ["bash", "-n", str(self.script)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_readiness_and_tool_smoke_precede_six_task_evaluation(self):
        data_preflight = self.text.index("stage=phase5-data-preflight begin")
        service_launch = self.text.index("stage=sglang-launch begin")
        readiness_wait = self.text.index("waiting for SGLang readiness")
        tool_smoke = self.text.index("[DCI] tool parser smoke passed")
        runner = self.text.index("RUNNER_ARGS=(")
        self.assertLess(data_preflight, service_launch)
        self.assertLess(service_launch, readiness_wait)
        self.assertLess(readiness_wait, tool_smoke)
        self.assertLess(tool_smoke, runner)
        self.assertIn("kill -0 \"${SGLANG_PID}\"", self.text)
        self.assertIn('[[ -r "${BACKEND_MANIFEST_PATH}" ]] || return 1', self.text)

    def test_service_is_isolated_and_cleaned_up(self):
        self.assertIn("setsid scripts/serve_sglang_minicpm5.sh &", self.text)
        self.assertIn('kill -TERM -- "-${SGLANG_PGID}"', self.text)
        self.assertIn("trap cleanup EXIT", self.text)
        self.assertIn("trap 'handle_signal TERM 143' TERM", self.text)
        self.assertIn('wait "${SGLANG_PID}"', self.text)
        self.assertIn('stop_watchdog_pid=$!', self.text)
        self.assertIn('wait "${stop_watchdog_pid}"', self.text)

    def test_completed_sample_failures_do_not_fail_cluster_job_by_default(self):
        self.assertIn(
            'FAIL_JOB_ON_SAMPLE_FAILURES="${FAIL_JOB_ON_SAMPLE_FAILURES:-0}"',
            self.text,
        )
        self.assertIn(
            '[[ "${RUNNER_EXIT}" == "1" && "${FAIL_JOB_ON_SAMPLE_FAILURES}" == "0" ]]',
            self.text,
        )
        self.assertIn('EVAL_STATUS="complete_with_sample_failures"', self.text)
        self.assertIn(
            'runner_exit=${RUNNER_EXIT} job_exit=${JOB_EXIT}',
            self.text,
        )

    def test_multigpu_and_query_concurrency_are_separate(self):
        self.assertIn('TP_SIZE="${TP_SIZE:-1}"', self.text)
        self.assertIn('CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"', self.text)
        self.assertIn('MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"', self.text)
        self.assertIn('TP_SIZE="${TP_SIZE}"', self.text)
        self.assertIn('--max-concurrency "${MAX_CONCURRENCY}"', self.text)

    def test_context_default_matches_the_shared_model_window(self):
        self.assertIn('CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"', self.text)
        self.assertIn('CONTEXT_LENGTH="${CONTEXT_LENGTH:-65536}"', self.service_text)

    def test_bridge_preflight_reserves_disjoint_ranges_for_all_tasks(self):
        self.assertIn("BRIDGE_PORT + 6 * MAX_CONCURRENCY - 1", self.text)

    def test_phase5_uses_a_dedicated_default_sglang_port(self):
        self.assertIn('SGLANG_PORT="${SGLANG_PORT:-17891}"', self.text)
        self.assertIn('BRIDGE_PORT="${BRIDGE_PORT:-18891}"', self.text)
        self.assertIn(
            'BACKEND_BASE_URL="${BACKEND_BASE_URL:-http://127.0.0.1:${SGLANG_PORT}/v1}"',
            self.text,
        )

    def test_required_sandbox_tools_are_installed_together(self):
        self.assertIn("MISSING_OS_PACKAGES+=(bubblewrap)", self.text)
        self.assertIn("MISSING_OS_PACKAGES+=(ripgrep)", self.text)
        self.assertIn('apt-get install -y --no-install-recommends "${MISSING_OS_PACKAGES[@]}"', self.text)
        self.assertIn('trusted ripgrep must be available at /usr/bin/rg', self.text)

    def test_http_port_is_not_reused_as_sglang_internal_port_base(self):
        self.assertIn('--port "${SGLANG_PORT}"', self.service_text)
        self.assertIn(
            'exec env -u SGLANG_PORT "${CMD[@]}"',
            self.service_text,
        )

    def test_runner_receives_backend_evidence_and_all_roots(self):
        runner_block = self.text[
            self.text.index("RUNNER_ARGS=(") : self.text.index(
                'log "stage=phase5-evaluation begin'
            )
        ]
        for required in (
            '--backend-manifest "${BACKEND_MANIFEST_PATH}"',
            '--backend-artifact "${MODELS_RESPONSE_PATH}"',
            '--backend-artifact "${TOOL_SMOKE_REQUEST_PATH}"',
            '--backend-artifact "${TOOL_SMOKE_RESPONSE_PATH}"',
            '--workspace-root "${WORKSPACE_ROOT}"',
            '--metadata-root "${METADATA_ROOT}"',
            '--results-root "${RESULTS_ROOT}"',
            '--bridge-port "${BRIDGE_PORT}"',
            '--metric-ks "${METRIC_K_ARRAY[@]}"',
        ):
            self.assertIn(required, runner_block)

    def test_tool_smoke_checks_the_parser_contract(self):
        self.assertIn('function.get("name") != "get_weather"', self.text)
        self.assertIn("json.loads(arguments)", self.text)
        self.assertIn('choices[0].get("finish_reason") != "tool_calls"', self.text)
        self.assertIn('usage.get("total_tokens") is None', self.text)


if __name__ == "__main__":
    unittest.main()
