#!/usr/bin/env python3
"""Offline acceptance through real Inspect local tools, bridge and frozen Pi.

Exercise seven simultaneous independent Inspect processes, then eight samples
within one Inspect process. Only the model provider is synthetic; all sandbox
startup, loopback HTTP, bash, read, final validation and cleanup are real.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def child(args):
    sys.path.insert(0, str(args.runtime))
    from inspect_ai import Task, eval as evaluate
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, ModelUsage, get_model
    from inspect_ai.tool import ToolCall
    from dci_bench.tasks.mteb_llm_retrieval import BridgePortPool, pi_dci_solver
    import dci_bench

    if not Path(dci_bench.__file__).resolve().is_relative_to(args.runtime.resolve()):
        raise ValueError('Probe did not import the frozen runtime')
    args.output.mkdir(parents=True, exist_ok=False)
    workspace = args.output / 'workspace'
    (workspace / 'corpus').mkdir(parents=True)
    (workspace / 'corpus/probe.md').write_text('Independent synthetic document for bridge acceptance.\n')

    def response(messages, tools, tool_choice, config):
        count = sum(m.role == 'tool' for m in messages)
        if count == 0:
            call = ToolCall(id='isolation', function='bash', arguments={
                'command': 'test -r /workspace/corpus/probe.md && test ! -e /mnt && test ! -e /root && test ! -e /var/tmp',
                'timeout_seconds': 10}, type='function')
        elif count == 1:
            call = ToolCall(id='receipt', function='read', arguments={'path': '/workspace/corpus/probe.md'},
                            type='function')
        else:
            call = None
        message = ChatMessageAssistant(content='' if call else '{"ranked_doc_ids":["probe"]}',
                                       tool_calls=[call] if call else None)
        return ModelOutput(model='mockllm/bridge-probe',
            choices=[ChatCompletionChoice(message=message, stop_reason='tool_calls' if call else 'stop')],
            usage=ModelUsage(input_tokens=100, output_tokens=10, total_tokens=110))

    samples = [Sample(id=f'q{i}', input='Synthetic bridge startup acceptance; no benchmark query',
                      metadata={'task': 'bridge_probe', 'query_id': f'q{i}', 'workspace': str(workspace)})
               for i in range(args.samples)]
    pool = BridgePortPool(args.port, args.samples)
    task = Task(dataset=samples, solver=pi_dci_solver(output_dir=str(args.output / 'samples'),
                bridge_port_pool=pool, model_context_window=65536), sandbox='local', time_limit=120)
    logs = evaluate(task, model=get_model('mockllm/bridge-probe', custom_outputs=response),
                    max_samples=args.samples, max_connections=args.samples,
                    log_dir=str(args.output / 'inspect'), display='none', fail_on_error=False)
    if len(logs) != 1 or logs[0].status != 'success' or not logs[0].samples:
        raise RuntimeError('Inspect bridge probe did not complete')
    records = []
    for sample in logs[0].samples:
        if sample.error:
            raise RuntimeError(f'Inspect sample {sample.id} failed: {sample.error}')
        result = json.loads(sample.output.completion)
        trace = json.loads(Path(result['trace_path']).read_text())
        bash = [e for e in trace['events'] if e.get('type') == 'tool_execution_end' and e.get('toolName') == 'bash']
        model_requests = sum(event.event == 'model' for event in sample.events)
        if (not result['valid_output'] or result['ranked_doc_ids'] != ['probe']
                or result['tool_sandbox'] != 'bubblewrap'
                or len(result['body_evidence'].get('read_receipts', [])) != 1
                or len(bash) != 1 or bash[0]['result']['details']['exitCode'] != 0
                or len(sample.model_usage) != 1 or model_requests != 3):
            raise RuntimeError(f'Bridge/tool/final acceptance failed: {sample.id}')
        records.append(dict(id=sample.id, valid=True, read_receipts=1, bash_exit_code=0,
                            model_requests=model_requests, provider='mockllm', external_model_calls=0))
    if len(records) != args.samples:
        raise RuntimeError('Missing probe samples')
    (args.output / 'result.json').write_text(json.dumps(dict(status='passed', samples=records), indent=2) + '\n')


def free_blocks(widths):
    held, bases = [], []
    try:
        for width in widths:
            for _ in range(100):
                probe = socket.socket()
                probe.bind(('127.0.0.1', 0))
                base = probe.getsockname()[1]
                block = [probe]
                try:
                    for offset in range(1, width):
                        sock = socket.socket()
                        block.append(sock)
                        sock.bind(('127.0.0.1', base + offset))
                except (OSError, OverflowError):
                    for sock in block:
                        sock.close()
                    continue
                held.extend(block)
                bases.append(base)
                break
            else:
                raise RuntimeError('Cannot reserve distinct probe ports')
        return bases
    finally:
        for sock in held:
            sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--tools-directory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-concurrency', type=int, default=8, choices=range(1, 9))
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    if args.child:
        child(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    wrapper = Path(__file__).with_name('inspect_local_runtime.py')
    processes = min(7, args.max_concurrency)
    ports = free_blocks([1] * processes + [args.max_concurrency])

    def run(index, count):
        target = args.output / f'process-{index}'
        command = [sys.executable, '-B', str(wrapper), '--tools-directory', str(args.tools_directory),
                   '--runner', str(Path(__file__).resolve()), '--', '--child', '--runtime', str(args.runtime),
                   '--tools-directory', str(args.tools_directory), '--output', str(target),
                   '--samples', str(count), '--port', str(ports[index])]
        with (args.output / f'process-{index}.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240, env=os.environ.copy())
        if result.returncode:
            raise RuntimeError(f'Inspect parallel probe failed; inspect {log.name}')
        return json.loads((target / 'result.json').read_text())

    with ThreadPoolExecutor(max_workers=processes) as executor:
        results = list(executor.map(lambda i: run(i, 1), range(processes)))
    results.append(run(processes, args.max_concurrency))
    record = dict(status='passed', parallel_processes=processes, batch_concurrency=args.max_concurrency,
                  samples=sum(len(r['samples']) for r in results), external_model_calls=0, gpu_started=False,
                  elapsed_seconds=time.monotonic() - start, results=results)
    (args.output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: value for key, value in record.items() if key != 'results'}), flush=True)


if __name__ == '__main__':
    main()
