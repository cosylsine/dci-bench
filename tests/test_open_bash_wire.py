"""Actual Pi HTTP requests and sandbox commands for the open-Bash prompt contract."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node') and Path('/usr/bin/bwrap').exists(), 'Node and bwrap required')
class OpenBashWireTests(unittest.TestCase):
    def run_case(self, tier, *, runtime=None, forced_read=False, malformed_final=False):
        requests = []
        commands = [
            "printf '%s\\n' 'for f in /workspace/corpus/*.md; do basename \"$f\"; done' > /scratch/list.sh; bash /scratch/list.sh | sort > /scratch/files",
            "test -s /scratch/files && cat /scratch/files",
        ]
        # Six varied Bash turns trigger the existing read-only control phase.
        if forced_read:
            commands.extend(f"printf 'inspection-{i}\\n'" for i in range(4))

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(request)
                turn = len(requests)
                call = None
                if turn <= len(commands):
                    call = ('bash', {'command': commands[turn - 1]})
                elif turn == len(commands) + 1:
                    call = ('read', {'path': 'd000001.md'})
                elif forced_read and turn == len(commands) + 2:
                    call = ('bash', {'command': 'cat /scratch/files'})
                final_turn = len(commands) + (3 if forced_read else 2)
                if call:
                    message = {'content': None, 'tool_calls': [{'id': f'call-{turn}', 'type': 'function',
                        'function': {'name': call[0], 'arguments': json.dumps(call[1])}}]}
                else:
                    # A parsed but unknown ID must be repaired with tools disabled.
                    ids = ['unknown'] if turn == final_turn else ['d000001', 'd000002']
                    message = {'content': 'Here is the answer.' if malformed_final and turn == final_turn
                               else json.dumps({'ranked_doc_ids': ids})}
                payload = json.dumps({'choices': [{'message': message, 'finish_reason': 'tool_calls' if call else 'stop'}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / 'workspace/corpus'
            corpus.mkdir(parents=True)
            for name in ('d000001', 'd000002'):
                (corpus / f'{name}.md').write_text('Relevant evidence.\n')
            (root / 'query.txt').write_text('Find the relevant evidence.')
            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            env = {**os.environ, 'DCI_PROTOCOL_PROFILE': f'document-pool-v2-{tier}', 'DCI_EVALUATION_TRACK': 'document', 'DCI_DOCUMENT_TOKEN_BUDGET': '64000'}
            env.pop('DCI_PROTOCOL_CONTRACT_SHA256', None)
            command = ['node', '--import', 'tsx', 'scripts/dci-run.ts', '--workspace', str(corpus.parent),
                '--query-file', str(root / 'query.txt'), '--output', str(root / 'final.json'),
                '--trace', str(root / 'trace.json'), '--openai-base-url', f'http://127.0.0.1:{server.server_port}/v1']
            if runtime is not None:
                command.extend(['--runtime-json', json.dumps(runtime)])
            try:
                process = subprocess.run(command, cwd=ROOT / 'pi-dci', env=env, text=True, capture_output=True, timeout=180)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
            self.assertEqual(process.returncode, 0, process.stderr)
            final = json.loads((root / 'final.json').read_text())
            trace = json.loads((root / 'trace.json').read_text())
        self.assertTrue(final['valid_output'])
        self.assertEqual(final['ranked_doc_ids'], ['d000001', 'd000002'])
        self.assertEqual(len(final['body_evidence']['read_receipts']), 1)
        self.assertEqual(final['repair_attempts'], 1)
        first = json.dumps(requests[0])
        self.assertIn('no command-name allowlist', first)
        self.assertIn('If you choose rg', first)
        for stale in ('First use rg', 'Start with rg', 'then bounded rg'):
            self.assertNotIn(stale, first)
        self.assertEqual(requests[0]['tool_choice'], 'auto')
        self.assertEqual(requests[-1]['tools'], [])
        self.assertEqual(requests[-1]['tool_choice'], 'none')
        bash_results = [event['result'] for event in trace['events']
                        if event['type'] == 'tool_execution_end' and event['toolName'] == 'bash']
        self.assertTrue(bash_results)
        for result in bash_results:
            self.assertEqual(result['details']['exitCode'], 0)
            self.assertFalse(result['details']['bodyEvidence'])
        if forced_read:
            self.assertEqual([event['action'] for event in final['control_events']], ['require_read', 'resume_search'])
            self.assertEqual([tool['function']['name'] for tool in requests[6]['tools']], ['read'])
            self.assertEqual({tool['function']['name'] for tool in requests[7]['tools']}, {'read', 'bash'})
            self.assertIn('Both read and bash are available again', requests[7]['messages'][-1]['content'])
        if runtime is not None:
            self.assertEqual(final['agent_runtime'], runtime['identity'])
            self.assertEqual(trace['agent_runtime'], runtime['identity'])

    def test_host_lite_commands_read_receipt_and_final_repair(self):
        self.run_case('lite')

    def test_v2_malformed_final_repairs_cannot_reopen_tools(self):
        self.run_case('lite', malformed_final=True)

    @unittest.skipUnless(os.environ.get('DCI_TEST_FROZEN_RUNTIME_CONFIG'), 'Prepared Frozen config required')
    def test_frozen_lite_and_full_reopen_tools_only_with_budget_and_caps(self):
        from dci_bench.runtime import verify_runtime

        runtime = verify_runtime(Path(os.environ['DCI_TEST_FROZEN_RUNTIME_CONFIG']))
        for tier in ('lite', 'full'):
            with self.subTest(tier=tier):
                self.run_case(tier, runtime=runtime.to_payload(), forced_read=True, malformed_final=True)


if __name__ == '__main__':
    unittest.main()
