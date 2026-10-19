import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from dci_bench.agents.pi_agent import run_pi_dci
from dci_bench.protocol.contracts import TRAJECTORY_SCHEMA_VERSION


class PiTimeoutFallbackTest(unittest.TestCase):
    def test_timeout_preserves_receipts_and_handles_exit_during_kill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.json"
            evidence = {"read_receipts": [{"path": "corpus/report.md", "returned_bytes": 12}]}
            trace.write_text(json.dumps({"events": [{"type": "tool_execution_end"}],
                                         "body_evidence": evidence}))
            process = Mock(pid=12345, returncode=-9)
            process.communicate.side_effect = [
                subprocess.TimeoutExpired("node", 1),
                subprocess.TimeoutExpired("node", 5), ("", ""),
            ]
            with patch("dci_bench.agents.pi_agent.subprocess.Popen", return_value=process), \
                 patch("dci_bench.agents.pi_agent.os.killpg", side_effect=ProcessLookupError) as kill:
                result = run_pi_dci(query="q", workspace=root, output_path=root / "final.json",
                                    trace_path=trace, timeout_seconds=1)
            self.assertEqual(result.failure_kind, "timeout")
            self.assertFalse(result.valid_output)
            self.assertEqual(result.body_evidence, evidence)
            saved = json.loads(trace.read_text())
            self.assertEqual(saved["schema_version"], TRAJECTORY_SCHEMA_VERSION)
            self.assertEqual(len(saved["events"]), 1)
            self.assertIn(unittest.mock.call(12345, signal.SIGKILL), kill.call_args_list)

    def test_timeout_before_first_snapshot_is_structured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = Mock(pid=12345, returncode=-15)
            process.communicate.side_effect = [subprocess.TimeoutExpired("node", 1), ("", "")]
            with patch("dci_bench.agents.pi_agent.subprocess.Popen", return_value=process) as popen, \
                 patch("dci_bench.agents.pi_agent.time.time", return_value=123.0), \
                 patch("dci_bench.agents.pi_agent.os.killpg"):
                result = run_pi_dci(query="q", workspace=root, output_path=root / "final.json",
                                    timeout_seconds=1)
            self.assertEqual(result.failure_kind, "timeout")
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--deadline-at-ms") + 1], "124000.0")
            saved = json.loads(result.trace_path.read_text())
            self.assertEqual(saved["schema_version"], TRAJECTORY_SCHEMA_VERSION)
            self.assertTrue(saved["partial"])
            self.assertEqual(saved["events"], [])
