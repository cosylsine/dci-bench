#!/usr/bin/env python3
"""CPU acceptance of two real v2 batches, with a synthetic loopback provider.

No GPU or external model calls. Exercises candidate isolation, Bash Bubblewrap,
read receipts, result publication, phase timing and query-level resume.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dci_bench.backends.manifest import build_api_manifest
from dci_bench.results.protocol import write_json_atomic
from scripts.check_inspect_parallel import free_blocks
from scripts.inspect_local_runtime import prepare
from scripts.run_benchmark_v2 import build_plan, command_for, completed_samples, run_child


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    for executable in ('/usr/bin/bwrap', '/usr/bin/rg'):
        if not os.access(executable, os.X_OK):
            raise RuntimeError('sandbox dependency missing; run bash scripts/ensure_pool_smoke_tools.sh')
    args.output = args.output.resolve()
    plan = build_plan(args.suite.resolve(), split='dev', concurrency=4, token_budget=64000,
                      backend_path=None, allow_partial=True)
    jobs = [job for job in plan['jobs'] if job['task_id'] == args.task]
    if len(jobs) != 2:
        raise ValueError('probe requires one ready task with both tiers')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'scratch').mkdir()
    prepare(args.output / 'inspect-tools')
    args.concurrency, args.split = 4, 'dev'
    args.bridge_port = free_blocks([4])[0]
    args.backend_manifest = args.output / 'backend.json'
    model = 'CPU-Synthetic-V2-Probe'
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_error(404)
                return
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if request.get('stream'):
                self.send_error(400, 'CPU probe uses nonstreaming responses')
                return
            tool_messages = [message for message in request['messages'] if message['role'] == 'tool']
            if not tool_messages:
                function, arguments = 'bash', {'command':
                    'test ! -e /mnt && test ! -e /root && test ! -e /var/tmp && '
                    'rg --files /workspace/corpus | sort | head -1'}
            elif len(tool_messages) == 1:
                # Use the visible corpus filename only; never consult qrels.
                content = tool_messages[0]['content']
                import re
                candidates = re.findall(r'/workspace/corpus/[^\s"<>]+\.md', content)
                if not candidates:
                    self.send_error(500, 'sandbox did not return a candidate path')
                    return
                function, arguments = 'read', {'path': candidates[0]}
            else:
                read_call = next(call for message in request['messages'] for call in message.get('tool_calls', [])
                                 if call['function']['name'] == 'read')
                path = json.loads(read_call['function']['arguments'])['path']
                doc_id = Path(path).name.removesuffix('.md')
                function, arguments = None, None
            message = {'role': 'assistant', 'content': ''}
            if function:
                message['tool_calls'] = [{'id': f'probe-{len(tool_messages)}', 'type': 'function',
                    'function': {'name': function, 'arguments': json.dumps(arguments)}}]
            else:
                message['content'] = json.dumps({'ranked_doc_ids': [doc_id]})
            response = {'id': 'synthetic-cpu', 'object': 'chat.completion', 'created': 0, 'model': model,
                'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if function else 'stop'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 30, 'total_tokens': 130}}
            requests.append({'model': request['model'], 'phase': function or 'final'})
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = build_api_manifest(model_key=model, served_model_name=model, provider='synthetic',
            revision='cpu-fixture-v1', base_url=f'http://127.0.0.1:{server.server_port}/v1')
        write_json_atomic(args.backend_manifest, backend)
        samples = []
        for job in jobs:
            env = dict(os.environ, DCI_PROTOCOL_PROFILE='document-pool-v2-' + job['tier'],
                       DCI_DOCUMENT_TOKEN_BUDGET='64000', DCI_WORKSPACE_PREFETCH='1',
                       DCI_WORKSPACE_PREFETCH_BATCH_SIZE='4', DCI_API_KEY='synthetic-not-a-secret',
                       TMPDIR=str(args.output / 'scratch'))
            env['NO_PROXY'] = env['no_proxy'] = 'localhost,127.0.0.1,::1'
            code = run_child(command_for(job, args, model), env=env,
                             log=args.output / (job['tier'] + '.log'), timeout=600)
            if code:
                raise RuntimeError(f'{job["tier"]} CPU batch failed; inspect retained log')
            run = args.output / 'evaluation' / model / job['task_id'] / 'runs' / ('dev-' + job['tier'])
            rows = completed_samples(run, job)
            if not all(row['valid'] and row['usage']['available']
                       and row['stage_timing'].get('bash_calls') == 1
                       and all(row['stage_timing'].get(key, -1) >= 0 for key in (
                           'workspace_prepare_seconds', 'workspace_cleanup_seconds',
                           'sandbox_validate_seconds', 'sandbox_preflight_seconds', 'bash_wall_seconds'))
                       for row in rows):
                raise RuntimeError('batch did not publish valid measured samples')
            count = len(requests)
            # Simulate interruption after query publication but before batch
            # aggregation, using a disposable copy; preserve original results.
            resume_args = argparse.Namespace(**vars(args))
            resume_args.output = args.output / 'resume-check' / job['tier']
            shutil.copytree(args.output / 'evaluation', resume_args.output / 'evaluation')
            copied_run = resume_args.output / run.relative_to(args.output)
            (copied_run / '_COMPLETE').unlink()
            (copied_run / 'task-result.json').unlink()
            code = run_child(command_for(job, resume_args, model), env=env,
                             log=args.output / (job['tier'] + '-resume.log'), timeout=180)
            if code or len(requests) != count:
                raise RuntimeError('completed query resume made additional requests or failed')
            if completed_samples(copied_run, job) != [dict(row, result_path=str(
                    copied_run / Path(row['result_path']).relative_to(run))) for row in rows]:
                raise RuntimeError('resume changed a published query result')
            if list((args.output / 'scratch').glob('dci-query-*')):
                raise RuntimeError('query workspace remains after evaluation lifecycle cleanup')
            samples.extend(rows)
        result = {'status': 'passed', 'evaluations': len(samples), 'http_requests': len(requests),
                  'capacities': [200, 1000], 'concurrency': 4, 'resume_new_requests': 0,
                  'resume_scenario': 'after_query_publication_before_batch_aggregation_in_disposable_copy',
                  'query_workspace_cleanup_verified': True,
                  'external_model_calls': 0, 'gpu_inference_started': False,
                  'interpretation': 'CPU integration diagnostic with scripted provider; not model quality evidence.',
                  'samples': samples}
        write_json_atomic(args.output / 'result.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, ensure_ascii=False))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        write_json_atomic(args.output / 'requests.json', requests)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
