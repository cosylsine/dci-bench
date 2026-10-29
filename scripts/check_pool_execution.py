#!/usr/bin/env python3
"""CPU acceptance through the real pool CLI, HTTP provider, Pi, scoring and publication.

The provider returns scripted responses; these artifacts are diagnostics only.
Candidate IDs are taken from the selected pool, never from relevance labels.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_pool_smoke as driver
from scripts import inspect_local_runtime
from scripts.check_inspect_parallel import free_blocks
from dci_bench.backends.manifest import build_vllm_manifest, validate_backend_manifest

PROBE_MODEL = 'CPU-Synthetic-Pool-Probe'


def run_row(row, args, tools_directory, model_directory, contracts):
    output = args.output / str(row['capacity'])
    output.mkdir()
    (output / 'logs').mkdir()
    metadata = Path(row['suite']) / 'metadata' / row['task']
    pool = driver.read(metadata / 'candidate_pools.json')['pools'][row['query_id']]
    ids = sorted(pool['candidate_doc_ids'])[:contracts[row['profile']]['contract']['top_k']]
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_error(404)
                return
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if request.get('stream'):
                self.send_error(400, 'CPU probe expects non-streaming Inspect requests')
                return
            read_done = any(m['role'] == 'tool' for m in request['messages'])
            message = {'role': 'assistant', 'content': json.dumps({'ranked_doc_ids': ids}) if read_done else ''}
            if not read_done:
                message['tool_calls'] = [{'id': 'probe-read', 'type': 'function', 'function': {
                    'name': 'read', 'arguments': json.dumps({'path': '/workspace/corpus/' + ids[0] + '.md'})}}]
            response = {'id': 'cpu-probe', 'object': 'chat.completion', 'created': 0, 'model': PROBE_MODEL,
                        'choices': [{'index': 0, 'message': message,
                                     'finish_reason': 'stop' if read_done else 'tool_calls'}],
                        'usage': {'prompt_tokens': 100, 'completion_tokens': 30, 'total_tokens': 130}}
            requests.append({'model': request['model'], 'read_done': read_done})
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
        backend = build_vllm_manifest(
            model_key=PROBE_MODEL, model_path=model_directory, served_model_name=PROBE_MODEL,
            base_url=f'http://127.0.0.1:{server.server_port}/v1', vllm_version='synthetic-http-fixture',
            tool_call_parser='qwen3_coder', reasoning_parser='qwen3', language_model_only=True,
            tensor_parallel_size=1, context_length=65536, dtype='bfloat16', max_num_seqs=1,
            gpu_memory_utilization=0.9,
        )
        backend['backend'].update(kind='synthetic', openai_service='synthetic')
        backend['manifest_sha256'] = driver.canonical({k: v for k, v in backend.items() if k != 'manifest_sha256'})
        validate_backend_manifest(backend)
        driver.write(output / 'backend.json', backend)
        for name in ('runtime.json', 'models.json', 'warmup.json'):
            driver.write(output / name, {'provider': 'synthetic-http-cpu', 'gpu_started': False})
        command = driver.task_command(row, ROOT, output, tools_directory)
        driver.step('evaluation', command, output, ROOT, driver.environment(ROOT, row['profile']), timeout=600)
        record = driver.verify_batch(row, output, contracts[row['profile']])
        if not record['valid'] or not record['real_model_and_tools'] or len(requests) < 2:
            raise RuntimeError(f'CPU publication probe failed: {record}')
        record.pop('real_model_and_tools')
        record.update(synthetic_provider=True, http_requests=len(requests), external_model_calls=0,
                      gpu_started=False, completed_artifacts_verified=True)
        driver.write(output / 'result.json', record)
        return record
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tools-directory', type=Path)
    parser.add_argument('--task')
    parser.add_argument('--concurrency', type=int, default=3, choices=range(1, 4))
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    work = driver.schedule(driver.read(args.bundle / 'plan.json')['rows'])
    task = args.task or work[0]['task']
    rows = [row for row in work if row['task'] == task]
    if len(rows) != 3 or {r['capacity'] for r in rows} != {200, 500, 1000}:
        raise ValueError('CPU execution probe requires all three tiers of one source')
    for row, port in zip(rows, free_blocks([1, 1, 1])):
        row['bridge_port'] = port
    contracts = driver.read(args.bundle / 'contracts.json')
    model = args.output / 'synthetic-model'
    model.mkdir()
    driver.write(model / 'config.json', {'model_type': 'synthetic-http-fixture'})
    (model / 'model.safetensors').write_bytes(b'CPU synthetic fixture; never load as model weights.\n')
    driver.MODEL = PROBE_MODEL
    with tempfile.TemporaryDirectory(prefix='dci-pool-publication-') as temporary:
        tools_directory = args.tools_directory or Path(temporary) / 'inspect-tools'
        if args.tools_directory is None:
            inspect_local_runtime.prepare(tools_directory)
        results = list(driver.parallel_checked(
            lambda row: run_row(row, args, tools_directory, model, contracts), rows, args.concurrency))
    result = dict(status='passed', task=task, evaluations=3, capacities=[200, 500, 1000],
                  provider='synthetic-http-cpu', gpu_started=False, external_model_calls=0,
                  interpretation='CPU integration diagnostic; not model evaluation or benchmark evidence.',
                  checks=results)
    driver.write(args.output / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}), flush=True)


if __name__ == '__main__':
    def interrupted(signum, _frame):
        driver.stop_children()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
