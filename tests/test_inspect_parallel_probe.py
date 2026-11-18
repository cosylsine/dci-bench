"""Regression coverage for timed-out probes and their enclosing deadline."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import check_inspect_parallel as probe

ROOT = Path(__file__).resolve().parents[1]


class InspectParallelProbeTests(unittest.TestCase):
    def test_sample_limit_and_empty_output_report_sample_instead_of_json_error(self):
        cases = [
            (SimpleNamespace(type='time', limit=120), '', 'q7 hit time limit \\(120\\)'),
            (None, '', 'q7 returned empty completion'),
            (None, ' \n', 'q7 returned empty completion'),
        ]
        for limit, completion, message in cases:
            with self.subTest(limit=limit, completion=completion), tempfile.TemporaryDirectory() as tmp:
                args = argparse.Namespace(runtime=ROOT, runtime_config=None, output=Path(tmp) / 'probe',
                                          samples=1, port=19131, sample_timeout_seconds=600)
                sample = SimpleNamespace(id='q7', limit=limit, error=None,
                                         output=SimpleNamespace(completion=completion))
                log = SimpleNamespace(status='success', samples=[sample])
                with patch.dict('os.environ'), patch('inspect_ai.eval', return_value=[log]) as evaluate:
                    with self.assertRaisesRegex(RuntimeError, message):
                        probe.child(args)
                self.assertEqual(evaluate.call_args.args[0].time_limit, 600)
                self.assertFalse((args.output / 'result.json').exists())

    def test_configured_deadline_reaches_children_and_extends_process_watchdog(self):
        for timeout in (600, 900):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'probe'
                argv = ['probe', '--runtime', str(ROOT), '--tools-directory', tmp,
                        '--output', str(output), '--max-concurrency', '1',
                        '--sample-timeout-seconds', str(timeout)]

                def run(command, **kwargs):
                    self.assertEqual(kwargs['timeout'], timeout + 240)
                    self.assertEqual(command[command.index('--sample-timeout-seconds') + 1], str(timeout))
                    target = Path(command[command.index('--output') + 1])
                    target.mkdir()
                    (target / 'result.json').write_text(json.dumps({'status': 'passed', 'samples': [{}]}))
                    return SimpleNamespace(returncode=0)

                with patch('sys.argv', argv), patch.object(probe, 'free_blocks', return_value=[19131, 19132]), \
                        patch.object(probe.subprocess, 'run', side_effect=run) as process:
                    probe.main()
                self.assertEqual(process.call_count, 2)
                report = json.loads((output / 'result.json').read_text())
                self.assertEqual(report['sample_timeout_seconds'], timeout)
                self.assertEqual(report['process_timeout_seconds'], timeout + 240)

    def test_invalid_deadline_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'probe'
            argv = ['probe', '--runtime', str(ROOT), '--tools-directory', tmp,
                    '--output', str(output), '--sample-timeout-seconds', '0']
            with patch('sys.argv', argv), self.assertRaises(SystemExit) as error:
                probe.main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
