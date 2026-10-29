#!/usr/bin/env python3
"""Run 70 frozen expansion queries on one four-H100 Qwen3.8-27B replica.

The first query per task is a smoke test and is included in the final 70.
This driver changes orchestration only; the accepted solver and data stay frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener

import prepare_selective_expansion as expansion
import stage_expansion_corpora as staging
import inspect_local_runtime as inspect_runtime

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'results/expansion-plans/20261015-selective-document-v2'
RESULTS = ROOT / 'results/Qwen3.8-27B/expansion-pilot-runs'
MODEL = 'Qwen3.8-27B'
CONCURRENCY = 8
TASKS = {'worksurface_backend_developer_rag', 'worksurface_logistics_manager_rag',
         'worksurface_operations_manager_rag', 'wixqa_expertwritten', 'wixqa_simulated',
         'browsecomp_plus', 'enterprise_rag'}


def batches(plan, splits, concurrency=CONCURRENCY):
    """Partition exactly ten original IDs per task into disjoint 1 + 9 batches."""
    result = []
    if len(plan['task_order']) != 7 or set(plan['task_order']) != TASKS:
        raise ValueError('Expected exactly seven new document tasks')
    for stage in ['smoke', 'pilot_tail']:
        for task in plan['task_order']:
            ids = splits[task]['pilot']
            if len(ids) != 10 or len(set(ids)) != 10 or splits[task]['smoke'] != ids[:1]:
                raise ValueError(f'Invalid frozen pilot partition: {task}')
            result.append(dict(stage=stage, task=task, query_ids=ids[:1] if stage == 'smoke' else ids[1:],
                               max_concurrency=1 if stage == 'smoke' else concurrency))
    return result


def schedule_bridge_ports(work, base_port):
    """Give every subprocess a fresh block sized for the configured concurrency."""
    width = max((b['max_concurrency'] for b in work), default=1)
    if not work or not 1024 <= base_port <= 65536 - len(work) * width:
        raise ValueError('The complete bridge port range must fit within 1024..65535')
    return [dict(batch, bridge_port=base_port + index * width, bridge_port_count=width)
            for index, batch in enumerate(work)]


def output_path(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id):
        raise ValueError('--run-id must be one safe path component')
    output = (RESULTS / run_id).resolve()
    if not output.is_relative_to(RESULTS.resolve()):
        raise ValueError('Output escapes the Qwen3.8-27B result root')
    if output.exists():
        raise FileExistsError(f'Refusing to reuse existing output: {output}')
    return output


def launch_environment(runtime, expected, model_path=None, concurrency=CONCURRENCY):
    env = expansion.environment()
    for name in ['PYTHONHOME', 'NODE_PATH', 'NODE_OPTIONS', 'KV_CACHE_DTYPE',
                 'VLLM_NVFP4_GEMM_BACKEND', 'DCI_CODE_RUNTIME_CONTRACT']:
        env.pop(name, None)
    env.update(PYTHONPATH=str(runtime), TSX_DISABLE_CACHE='1',
               MODEL_PATH=str(model_path or env.get('MODEL_PATH') or expected['model']['source_path']),
               SERVED_MODEL_NAME=MODEL,
               TENSOR_PARALLEL_SIZE='4', TP_SIZE='4', MAX_MODEL_LEN='65536',
               DTYPE='bfloat16', VLLM_HOST='127.0.0.1', TOOL_CALL_PARSER='qwen3_coder',
               REASONING_PARSER='qwen3', LANGUAGE_MODEL_ONLY='1',
               VLLM_MAX_NUM_SEQS=str(concurrency), VLLM_GPU_MEMORY_UTILIZATION='0.9',
               VLLM_EXTRA_ARGS='--enforce-eager',
               NVIDIA_DRIVER_CAPABILITIES='compute,utility')
    env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    for key in ['NO_PROXY', 'no_proxy']:
        env[key] = env.get(key, '') + ',127.0.0.1,localhost'
    return env


def model_location(expected, override=None):
    """Check the requested mount cheaply; complete weight identity is checked separately."""
    requested = str(override or expected['model']['source_path'])
    record = dict(requested_path=requested, explicit_override=bool(override),
                  frozen_source_path=expected['model']['source_path'], selected_path=None,
                  checks=[], weights_identity_verified=False)
    expected_files = {f['name']: f['size_bytes'] for f in expected['model']['weights']['files']}
    check = dict(path=requested)
    record['checks'].append(check)
    try:
        path = Path(requested).expanduser().resolve()
        check['path'] = str(path)
        if not (path / 'config.json').is_file():
            check['status'] = 'missing_config'
        elif expansion.sha(path / 'config.json') != expected['model']['config_sha256']:
            check['status'] = 'config_mismatch'
        else:
            files = {p.name: p.stat().st_size for p in [*path.glob('*.safetensors'), *path.glob('*.bin')]}
            if files != expected_files:
                check['status'] = 'weight_names_or_sizes_mismatch'
            else:
                check['status'] = 'config_and_weight_sizes_match'
                record['selected_path'] = str(path)
    except OSError as exc:
        check.update(status='unreadable', error=str(exc))
    return record


def check_backend_identity(confirmation, expected, actual, concurrency=None):
    """Allow declared scheduling capacity and relocation, preserving model identity."""
    # Keep the actual mount and capacity in the saved manifest. Project only
    # those declared orchestration changes for the original strict comparison;
    # every checkpoint digest and all other serving settings remain pinned.
    comparison = dict(actual, model=dict(actual['model'], source_path=expected['model']['source_path']))
    if concurrency is not None:
        if actual['backend']['max_num_seqs'] != concurrency:
            raise ValueError('Backend capacity differs from the declared concurrency')
        comparison['backend'] = dict(actual['backend'], max_num_seqs=expected['backend']['max_num_seqs'])
        runtime = actual['backend'].get('runtime')
        if runtime:
            previous = expected['backend']['runtime']
            launch = dict(previous['launch_environment'], MODEL_PATH=actual['model']['source_path'],
                          VLLM_MAX_NUM_SEQS=str(concurrency))
            if runtime['launch_environment'] != launch:
                raise ValueError('Serving launch environment differs beyond declared mount/capacity')
            for field in ['packages', 'torch_cuda', 'python']:
                if runtime[field] != previous[field]:
                    raise ValueError(f'Serving software differs: {field}')
            comparison['backend']['runtime'] = dict(runtime, launch_environment=previous['launch_environment'])
    confirmation.check_backend(expansion.SOURCE, comparison)


def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # Reap workers even if their process-group leader has already exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def step(name, argv, *, output, cwd, env, timeout=1800, accepted=(0,)):
    log = output / 'logs' / (name + '.log')
    start = time.monotonic()
    print(f'[pilot] {name}; log={log}', flush=True)
    with log.open('x') as handle:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            terminate(process)
            raise
    elapsed = time.monotonic() - start
    record = dict(argv=argv, cwd=str(cwd), exit_code=code, elapsed_seconds=elapsed,
                  log=str(log.relative_to(output)))
    expansion.write(output / 'logs' / (name + '.json'), record)
    if code not in accepted:
        raise RuntimeError(f'{name} failed ({code}); inspect {log}')
    print(f'[pilot] {name}: exit={code}, seconds={elapsed:.2f}', flush=True)
    return record


def command(batch, row, runtime, output):
    argv = [expansion.PYTHON, '-B', str(ROOT / 'releases/run-launchers' / output.parent.name / output.name / 'inspect_local_runtime.py'),
            '--tools-directory', batch['inspect_tools_directory'],
            '--runner', str(runtime / 'scripts/run_expansion_task.py'), '--',
            '--suite', row['suite'], '--task', batch['task'], '--query-ids', *batch['query_ids'],
            '--model-key', MODEL, '--backend-manifest', str(output / 'backend.json'),
            '--results-root', str(output / 'evaluation'), '--run-id', batch['stage'],
            '--max-concurrency', str(batch['max_concurrency']), '--bridge-port', str(batch['bridge_port']),
            '--metric-ks', '1', '3', '5', '10', '20', '--require-valid-output']
    for name in ['runtime.json', 'models.json', 'warmup.json']:
        argv.extend(['--backend-artifact', str(output / name)])
    return argv


def validate_runtime(actual, expected, launch_environment=None):
    previous = expected['backend']['runtime']
    for key in ['packages', 'torch_cuda', 'python']:
        if actual[key] != previous[key]:
            raise ValueError(f'Serving software differs from the accepted run: {key}')
    if launch_environment is not None:
        wanted = {key: launch_environment.get(key) for key in previous['launch_environment']}
        if actual['launch_environment'] != wanted:
            raise ValueError('Serving runtime did not capture the declared launch environment')
    gpus = actual['cuda']['gpus']
    if (actual['cuda'].get('kernel_probe') != 'passed-on-every-visible-device'
            or len(gpus) != 4 or any('H100' not in g['name'] or g['capability'] != [9, 0]
                                    or g.get('total_memory_bytes', 0) < 75 * 1024**3 for g in gpus)):
        raise ValueError('This pilot requires exactly four visible H100 GPUs')


def check_ports(serving, work):
    ports = [serving, *(batch['bridge_port'] + offset
                       for batch in work for offset in range(batch['bridge_port_count']))]
    if len(set(ports)) != len(ports) or not all(1024 <= p <= 65535 for p in ports):
        raise ValueError('Invalid or overlapping serving/bridge ports')
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            try:
                sock.bind(('127.0.0.1', port))
            except OSError as exc:
                raise ValueError(f'Pilot port {port} is unavailable: {exc}') from exc
    finally:
        for sock in sockets:
            sock.close()


def smoke_passed(records, task_order):
    return (len(records) == len(task_order)
            and {r['sample']['task'] for r in records} == set(task_order)
            and all(r['sample']['valid_output'] and r['sample']['execution']['tool_calls'] > 0
                    and r['sample']['execution']['model_calls'] > 0 for r in records))


STARTUP_PROBE = '''import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dci_bench.agents.pi_agent import run_pi_dci
metadata, workspace, output = map(Path, sys.argv[1:4])
concurrency = int(sys.argv[4])
output.mkdir()
mapping = json.loads((metadata / 'doc_id_files.json').read_text())
doc_id, filename = next(iter(mapping.items()))
mock = [
 {'type':'tool_call','name':'bash','id':'isolation',
  'arguments':{'command':'test -r /workspace/corpus && test ! -e /mnt && test ! -e /root && test ! -e /proc && test ! -e /tmp',
               'timeout_seconds':10},'usage':{'input':1,'output':1}},
 {'type':'tool_call','name':'bash','id':'full-scan',
  'arguments':{'command':'rg -l -F -- __DCI_QUERY_INDEPENDENT_IO_PROBE_20261020__ /workspace/corpus >/dev/null; rc=$?; test "$rc" -le 1',
               'timeout_seconds':60},'usage':{'input':1,'output':1}},
 {'type':'tool_call','name':'read','id':'receipt',
  'arguments':{'path':'/workspace/corpus/' + filename},'usage':{'input':1,'output':1}},
 {'type':'final','text':json.dumps({'ranked_doc_ids':[doc_id]}),'usage':{'input':1,'output':1}}]
mock_path = output / 'mock.json'
mock_path.write_text(json.dumps(mock) + '\\n')
def probe(index):
 target = output / str(index)
 target.mkdir()
 started = time.monotonic()
 result = run_pi_dci(query='CPU startup probe; no benchmark query or relevance assertion',
  workspace=workspace, output_path=target/'final.json', trace_path=target/'trace.json',
  mock_script=mock_path, model='mock', max_agent_steps=5, context_window=65536, timeout_seconds=120)
 trace = json.loads((target/'trace.json').read_text())
 bash = [e for e in trace['events'] if e.get('type') == 'tool_execution_end' and e.get('toolName') == 'bash']
 clean = len(bash) == 2 and all('exit_code=0' in json.dumps(e['result']) and
                              not e.get('isError') for e in bash)
 row = {'elapsed_seconds':time.monotonic()-started, 'valid_output':result.valid_output,
  'failure_kind':result.failure_kind, 'read_receipts':len(result.body_evidence.get('read_receipts', [])),
  'tool_sandbox':result.tool_sandbox, 'isolation_and_full_scan_passed':clean}
 assert result.valid_output and row['read_receipts'] == 1 and result.tool_sandbox == 'bubblewrap' and clean, row
 return row
with ThreadPoolExecutor(max_workers=concurrency) as pool:
 rows = list(pool.map(probe, range(concurrency)))
record = {'samples':rows, 'concurrency':concurrency, 'max_elapsed_seconds':max(r['elapsed_seconds'] for r in rows),
 'external_model_calls':0, 'benchmark_queries':0, 'probe_timeout_seconds':120}
(output/'measurement.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''


def verify_batch(batch, row, runtime, output):
    from dci_bench.results.phase4_runner import load_completed_sample
    from dci_bench.results.protocol import verify_artifact_record, validate_run_manifest, validate_task_result
    from dci_bench.scoring.retrieval import precision_at_k

    run = output / 'evaluation' / MODEL / batch['task'] / 'runs' / batch['stage']
    result = expansion.read(run / 'task-result.json')
    manifest = expansion.read(run / 'run-manifest.json')
    marker = expansion.read(run / '_COMPLETE')
    validate_run_manifest(manifest)
    validate_task_result(result)
    if (marker['task_result_sha256'] != expansion.canonical(result)
            or marker['run_fingerprint'] != manifest['run_fingerprint']
            or result['query_ids'] != batch['query_ids'] or manifest['query_ids'] != batch['query_ids']
            or result['task'] != batch['task'] or manifest['task'] != batch['task']
            or result['run_id'] != batch['stage'] or manifest['run_id'] != batch['stage']
            or marker['run_id'] != batch['stage'] or manifest['data'] != result['data']
            or manifest['model_key'] != MODEL or manifest['model'] != result['model']
            or manifest['backend'] != result['backend'] or manifest['code'] != result['code']
            or manifest['code']['max_concurrency'] != batch['max_concurrency']
            or result['metric_ks'] != [1, 3, 5, 10, 20]
            or result['counts']['completed'] != len(batch['query_ids'])
            or [s['query_id'] for s in result['samples']] != batch['query_ids']):
        raise ValueError(f'Completed batch identity/integrity mismatch: {run}')
    contract = dict(manifest['benchmark_contract'])
    context = contract.pop('task_context')
    if (expansion.canonical(contract) != expansion.CONTRACT
            or context != {'task_name': batch['task'], 'instruction': row['task_instruction'].strip()}):
        raise ValueError(f'Completed batch protocol changed: {run}')
    records = []
    qrels = expansion.read(Path(row['metadata']) / 'qrels.json')
    for entry in result['samples']:
        sample = load_completed_sample(run, entry['query_id'])
        if sample is None or expansion.canonical(sample) != entry['result']['sha256']:
            raise ValueError('Completed sample differs from task index')
        records.append(dict(sample=sample, run=str(run.relative_to(output)),
            precision_at_10=precision_at_k(sample['ranked_doc_ids'], qrels[entry['query_id']], 10)
                            if sample['valid_output'] else 0.0))
    integrity = result['integrity']
    for ref in [integrity.get('backend_manifest'), *integrity.get('backend_diagnostics', [])]:
        if ref:
            verify_artifact_record(ref, root=run)
    return records


def summarize(records, task_order):
    tasks = {}
    for task in task_order:
        rows = [r['sample'] for r in records if r['sample']['task'] == task]
        if not rows:
            continue
        valid = [r for r in rows if r['valid_output']]
        metrics = {key: sum(r['metrics'].get(key, 0) if r['valid_output'] else 0 for r in rows) / len(rows)
                   for key in rows[0]['metrics']}
        task_records = [r for r in records if r['sample']['task'] == task]
        metrics['precision_at_10'] = sum(r['precision_at_10'] for r in task_records) / len(rows)
        walls = sorted(r['timing']['wall_time_seconds'] for r in rows
                       if r['timing'].get('wall_time_seconds') is not None)
        tokens = [r['usage']['total_tokens'] for r in rows if r['usage'].get('total_tokens') is not None]
        valid_metrics = {key: sum(r['metrics'].get(key, 0) for r in valid) / len(valid) if valid else None
                         for key in metrics if key != 'precision_at_10'}
        valid_metrics['precision_at_10'] = (sum(r['precision_at_10'] for r in task_records
                                                if r['sample']['valid_output']) / len(valid)) if valid else None
        tasks[task] = dict(completed=len(rows), valid=len(valid), invalid=len(rows) - len(valid),
            all_sample=metrics,
            valid_only=valid_metrics,
            failure_counts=dict(Counter(r['failure']['kind'] for r in rows if r['failure'])),
            total_tokens=sum(tokens) if tokens else None,
            usage_available_samples=sum(r['usage']['available'] is True for r in rows),
            timing_available_samples=len(walls), mean_wall_seconds=sum(walls) / len(walls) if walls else None,
            p50_wall_seconds=percentile(walls, .5), p95_wall_seconds=percentile(walls, .95),
            samples_requiring_repair=sum(r['execution']['repair_attempts'] > 0 for r in rows),
            repair_attempts=sum(r['execution']['repair_attempts'] for r in rows),
            repair_reasons=dict(Counter(reason for r in rows for reason in r['execution']['repair_reasons'])),
            repaired_zero_recall=sum(r['execution']['repair_attempts'] > 0 and
                                    r['metrics'].get('recall_at_10', 0) == 0 for r in rows))
    keys = next(iter(tasks.values()))['all_sample'] if tasks else {}
    return dict(schema_version='dci-selective-expansion-pilot-summary-v1', tasks=tasks,
        completed=len(records), valid=sum(r['sample']['valid_output'] for r in records),
        equal_task_macro={k: sum(t['all_sample'][k] for t in tasks.values()) / len(tasks) for k in keys},
        query_weighted={k: sum(t['all_sample'][k] * t['completed'] for t in tasks.values()) / len(records)
                        for k in keys},
        interpretation='70-query diagnostic only; smoke samples included once; full task corpora retained. '
                       'Missing usage/timing is reported by sample validity and availability, not inferred as success.',
        sample_sources=[dict(task=r['sample']['task'], query_id=r['sample']['query_id'], run=r['run']) for r in records])


def percentile(values, fraction):
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--preflight-only', action='store_true')
    modes.add_argument('--inspect-preflight-only', action='store_true',
                       help='Test real parallel Inspect/bridge/Pi startup with synthetic responses; no weights/corpus/GPU')
    modes.add_argument('--model-preflight-only', action='store_true',
                       help='Verify the mounted checkpoint identity without CUDA, corpus scans, or model calls')
    parser.add_argument('--model-path', type=Path,
                        help='Mounted copy of the frozen checkpoint; takes precedence over MODEL_PATH')
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'-{os.getpid()}')
    parser.add_argument('--sandbox-timeout', type=int, default=1800)
    parser.add_argument('--io-workers', type=int, choices=range(1, 257), default=16)
    parser.add_argument('--max-concurrency', type=int, choices=range(1, 9),
                        default=int(os.environ.get('MAX_CONCURRENCY', CONCURRENCY)))
    parser.add_argument('--local-corpus-root', type=Path,
                        default=Path(os.environ.get('LOCAL_CORPUS_ROOT', '/tmp/dci-expansion-corpora')),
                        help='Node-local SSD/tmpfs parent for a fresh, full, verified corpus copy')
    args = parser.parse_args()
    if args.sandbox_timeout < 1:
        parser.error('--sandbox-timeout must be positive')
    if not 1 <= args.max_concurrency <= 8:
        parser.error('--max-concurrency / MAX_CONCURRENCY must be between 1 and 8')
    cpu_only = args.preflight_only or args.inspect_preflight_only
    expansion.verify(PLAN)
    plan, splits = expansion.read(PLAN / 'plan.json'), expansion.read(PLAN / 'query-splits.json')
    work = batches(plan, splits, args.max_concurrency)
    runtime = expansion.code_path(plan['execution_runtime'])
    expected = expansion.read(expansion.SOURCE / 'expected-backend.json')
    output = output_path(args.run_id)
    override = args.model_path or os.environ.get('MODEL_PATH')
    location = (dict(requested_path=str(override or expected['model']['source_path']),
                     selected_path=None, status='not_checked_cpu_preflight')
                if cpu_only else model_location(expected, override))
    env = launch_environment(runtime, expected,
                             location['selected_path'] or location['requested_path'], args.max_concurrency)
    for name in ['DCI_PROTOCOL_PROFILE', 'DCI_EVALUATION_TRACK', 'DCI_PROTOCOL_CONTRACT_SHA256']:
        os.environ[name] = env[name]
    port = int(env.get('VLLM_PORT', '28099'))
    bridge = int(env.get('BRIDGE_PORT', '28131'))
    work = schedule_bridge_ports(work, bridge)
    inventory = {r['task']: r for r in plan['task_inventory']}
    local_destination = args.local_corpus_root.resolve() / args.run_id
    tools_directory = args.local_corpus_root.resolve() / (args.run_id + '-inspect-tools')
    work = [dict(batch, inspect_tools_directory=str(tools_directory)) for batch in work]
    launch_plan = dict(schema_version='dci-selective-expansion-pilot-launch-v1',
        frozen_plan=str(PLAN), frozen_plan_sha256=expansion.sha(PLAN / '_COMPLETE.json'),
        protocol_sha256=expansion.CONTRACT, execution_runtime=str(runtime), output=str(output),
        model=MODEL, model_location=location,
        tensor_parallel_size=4, dtype='bfloat16', context_length=65536,
        serving_port=port, bridge_port=bridge, cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],
        bridge_port_range=[bridge, bridge + len(work) * args.max_concurrency - 1],
        bridge_port_policy='Dedicated concurrency-sized block per subprocess; no reuse between batches',
        max_concurrency=args.max_concurrency, serving_max_num_seqs=args.max_concurrency,
        local_corpus_destination=str(local_destination),
        storage_policy='Complete byte-identical local corpora; frozen runtime and metadata unchanged',
        inspect_tools_directory=str(tools_directory),
        inspect_tools_policy='Private verified bundle prepared once; path-only bootstrap; real parallel bridge preflight',
        orchestration_changes={'baseline_max_num_seqs': expected.get('backend', {}).get('max_num_seqs', 4),
                               'parallel_smoke_tasks': min(7, args.max_concurrency)},
        cumulative_token_budget=128000, sample_timeout_seconds=600, stages=work,
        unique_queries=sum(len(b['query_ids']) for b in work), smoke_queries=7,
        task_order=plan['task_order'], query_ids={t: splits[t]['pilot'] for t in plan['task_order']},
        selection_seed=plan['selection_seed'], candidate_policy=plan['candidate_policy'],
        synthetic_warmup='Separate from the 70 benchmark queries and their usage totals',
        server_command=['bash', str(expansion.SOURCE_RUNTIME / 'scripts/serve_vllm_qwen38.sh')])
    if args.dry_run or os.environ.get('DRY_RUN') == '1':
        print(json.dumps({**launch_plan, 'mode': 'dry-run', 'gpu_started': False}, indent=2))
        return 0
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    expansion.write(output / 'launch-plan.json', launch_plan)
    shutil.copyfile(__file__, output / 'driver-source.py')
    shutil.copyfile(staging.__file__, output / 'staging-source.py')
    launcher_runtime = ROOT / 'releases/run-launchers' / output.parent.name / output.name
    launcher_runtime.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(inspect_runtime.__file__, launcher_runtime / 'inspect_local_runtime.py')
    shutil.copyfile(ROOT / 'scripts/check_inspect_parallel.py', launcher_runtime / 'check_inspect_parallel.py')
    expansion.write(output / 'launcher-runtime.json', {'path': str(launcher_runtime)})
    expansion.write(output / 'driver-source.sha256.json', {'sha256': expansion.sha(Path(__file__)),
                    'staging_sha256': expansion.sha(Path(staging.__file__)),
                    'inspect_runtime_sha256': expansion.sha(Path(inspect_runtime.__file__)),
                    'inspect_probe_sha256': expansion.sha(ROOT / 'scripts/check_inspect_parallel.py')})
    server = None
    gpu_probe_started = False
    model_server_started = False
    records = []
    try:
        if not cpu_only and not args.model_preflight_only:
            # Detect conflicts before reading weights or scanning full corpora.
            check_ports(port, work)
        if not cpu_only and location['selected_path'] is None:
            detail = '; '.join(f'{c["path"]}: {c["status"]}' for c in location['checks'])
            raise RuntimeError(f'No mounted copy matches the frozen checkpoint ({detail}). '
                               'Mount the original checkpoint and set MODEL_PATH or --model-path. '
                               'A directory name alone does not establish model identity.')
        # Source/dependency inventory, metadata, all candidate bytes and sandboxes precede CUDA.
        full_scripts = expansion.SOURCE_RUNTIME / 'scripts'
        sys.path.insert(0, str(full_scripts))
        full = importlib.import_module('prepare_harness_full')
        frozen = full.verify_bundle(expansion.SOURCE)
        bench_identity = full.confirmation.python_identity(expansion.PYTHON)
        node_version = subprocess.check_output([str(Path(expansion.PYTHON).with_name('node')),
                                                '--version'], text=True).strip()
        if bench_identity != frozen['bench_identity'] or node_version != frozen['node_version']:
            raise ValueError('Benchmark Python packages or Node version differ from the accepted run')
        sys.path.insert(0, str(runtime))
        if not args.model_preflight_only:
            tools_directory.parent.mkdir(parents=True, exist_ok=True)
            staging.local_filesystem(tools_directory.parent)
            tools_record = inspect_runtime.prepare(tools_directory)
            expansion.write(output / 'inspect-tools.json', tools_record)
            step('inspect-parallel', [expansion.PYTHON, '-B', str(launcher_runtime / 'check_inspect_parallel.py'),
                 '--runtime', str(runtime), '--tools-directory', str(tools_directory),
                 '--output', str(output / 'inspect-preflight'), '--max-concurrency', str(args.max_concurrency)],
                 output=output, cwd=runtime, env=env, timeout=600)
            if args.inspect_preflight_only:
                print(f'[pilot] Parallel Inspect preflight passed; no weights/corpus/GPU: {output}', flush=True)
                return 0
        if not cpu_only:
            print(f'[pilot] model mount: {env["MODEL_PATH"]}; verifying frozen weight identity', flush=True)
            from dci_bench.backends.manifest import build_vllm_manifest
            identity = build_vllm_manifest(model_key=MODEL, model_path=Path(env['MODEL_PATH']),
                served_model_name=MODEL, base_url=f'http://127.0.0.1:{port}/v1',
                vllm_version=expected['backend']['version'], tool_call_parser='qwen3_coder',
                reasoning_parser='qwen3', language_model_only=True, tensor_parallel_size=4,
                context_length=65536, dtype='bfloat16', max_num_seqs=args.max_concurrency, gpu_memory_utilization=.9)
            check_backend_identity(full.confirmation, expected, identity, args.max_concurrency)
            expansion.write(output / 'model-preflight.json', identity)
            expansion.write(output / 'model-location.json', dict(location, weights_identity_verified=True,
                             identity_status='passed', gpu_started=False, model_calls=0))
            print('[pilot] frozen model config and weight identity: passed', flush=True)
            if args.model_preflight_only:
                print(f'[pilot] Model preflight passed; no GPU/model calls: {output}', flush=True)
                return 0
        step('snapshot', [expansion.PYTHON, '-B', str(runtime / 'verify_snapshot.py'),
             '--data-index', plan['dataset_index']], output=output, cwd=runtime, env=env)
        for name in ['/usr/bin/bwrap', '/usr/bin/rg']:
            if not os.access(name, os.X_OK):
                raise RuntimeError(f'Missing trusted dependency: {name}; use the shell launcher')
        print(f'[pilot] staging complete frozen corpora on local storage: {local_destination}', flush=True)
        inventory, stage_record = staging.stage(inventory, local_destination, workers=args.io_workers)
        expansion.write(output / 'local-corpora.json', stage_record)
        expansion.write(output / 'execution-inventory.json', inventory)
        for task in plan['task_order']:
            row = inventory[task]
            code = ('from pathlib import Path; from dci_bench.data.expansion import verify_document_task; '
                    f'm=verify_document_task(Path({row["suite"]!r}), {task!r}, io_workers={args.io_workers}); '
                    'print({k:m[k] for k in ["task", "num_docs", "num_queries"]})')
            step('corpus-' + task, [expansion.PYTHON, '-B', '-c', code],
                 output=output, cwd=runtime, env=env, timeout=3600)
            step('sandbox-' + task, [str(Path(expansion.PYTHON).with_name('node')),
                 str(runtime / 'pi-dci/node_modules/tsx/dist/cli.mjs'), 'scripts/dci-run.ts',
                 '--preflight-only', '--workspace', row['workspace']], output=output,
                 cwd=runtime / 'pi-dci', env=env, timeout=args.sandbox_timeout)
        (output / 'cpu-mock').mkdir()
        enterprise = inventory['enterprise_rag']
        step('enterprise-startup', [expansion.PYTHON, '-B', '-c', STARTUP_PROBE,
             enterprise['metadata'], enterprise['workspace'], str(output / 'cpu-mock/enterprise_rag'),
             str(args.max_concurrency)], output=output, cwd=runtime, env=env, timeout=240)
        expansion.write(output / 'preflight.json', dict(status='passed', tasks=plan['task_order'],
                         source_corpus_bytes_verified=True, gpu_started=False, model_calls=0,
                         inspect_parallel_preflight='passed',
                         bench_identity=bench_identity, node_version=node_version))
        if args.preflight_only:
            print(f'[pilot] CPU preflight passed; no GPU/model calls: {output}', flush=True)
            return 0
        # CPU preflights can take many minutes; check again immediately before CUDA.
        check_ports(port, work)
        serving_python = str(expansion.read(expansion.SOURCE / 'frozen.json')['serving_python'])
        serving_env = dict(env, PATH=str(Path(serving_python).parent) + os.pathsep + env['PATH'],
                           LD_LIBRARY_PATH='/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu')
        gpu_probe_started = True
        step('runtime', [serving_python, str(full_scripts / 'harness_pilot_runtime.py'),
             '--output', str(output / 'runtime.json')], output=output, cwd=runtime, env=serving_env)
        actual = expansion.read(output / 'runtime.json')
        validate_runtime(actual, expected, serving_env)
        serving_env['VLLM_PORT'] = str(port)
        with (output / 'server.log').open('x') as handle:
            server = subprocess.Popen(launch_plan['server_command'], cwd=expansion.SOURCE_RUNTIME,
                env=serving_env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            model_server_started = True
        opener = build_opener(ProxyHandler({}))
        deadline = time.monotonic() + 1800
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Model server exited; inspect {output / "server.log"}')
            try:
                with opener.open(f'http://127.0.0.1:{port}/v1/models', timeout=5) as response:
                    models = json.load(response)
                if MODEL not in [r['id'] for r in models['data']]:
                    raise ValueError('Endpoint returned an unexpected served model')
                expansion.write(output / 'models.json', models)
                break
            except (OSError, ValueError):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Model server readiness timed out')
                time.sleep(5)
        step('backend', [expansion.PYTHON, '-B', '-m', 'dci_bench.backends.manifest',
             '--output', str(output / 'backend.json'), '--model-key', MODEL,
             '--model-path', env['MODEL_PATH'], '--served-model-name', MODEL,
             '--base-url', f'http://127.0.0.1:{port}/v1', '--vllm-version', actual['packages']['vllm'],
             '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3', '--language-model-only',
             '--sampling-backend', 'auto', '--tensor-parallel-size', '4', '--context-length', '65536',
             '--dtype', 'bfloat16', '--max-num-seqs', str(args.max_concurrency), '--gpu-memory-utilization', '0.9',
             '--runtime-info', str(output / 'runtime.json')], output=output, cwd=runtime, env=env)
        check_backend_identity(full.confirmation, expected, expansion.read(output / 'backend.json'), args.max_concurrency)
        step('warmup', [expansion.PYTHON, '-B', str(full_scripts / 'harness_pilot_warmup.py'),
             '--base-url', f'http://127.0.0.1:{port}/v1', '--model', MODEL,
             '--output', str(output / 'warmup.json'), '--concurrency', str(args.max_concurrency),
             '--deadline-seconds', '600', '--http-timeout-seconds', '600'],
             output=output, cwd=runtime, env=env)
        def run_batch(batch):
            step(batch['stage'] + '-' + batch['task'],
                 command(batch, inventory[batch['task']], runtime, output),
                 output=output, cwd=runtime, env=env,
                 timeout=(len(batch['query_ids']) + 1) * 660 + 600, accepted=(0, 1))
            return verify_batch(batch, inventory[batch['task']], runtime, output)
        # Independent smoke processes have distinct result directories and ports.
        # Collect in the frozen task order regardless of completion order.
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
            futures = [pool.submit(run_batch, batch) for batch in work[:7]]
            for future in futures:
                records.extend(future.result())
        if not smoke_passed(records, plan['task_order']):
            expansion.write(output / 'smoke-summary.json', summarize(records, plan['task_order']))
            failed = [r['sample'] for r in records if not r['sample']['valid_output']
                      or not r['sample']['execution']['model_calls'] or not r['sample']['execution']['tool_calls']]
            details = '; '.join(f'{s["task"]}/{s["query_id"]}: '
                                f'{(s.get("failure") or {}).get("kind", "missing real model/tool calls")}'
                                for s in failed)
            raise RuntimeError(f'Smoke failed ({details}); preserved seven samples and did not launch '
                               f'the remaining 63; inspect {output / "smoke-summary.json"}')
        for batch in work[7:]:
            records.extend(run_batch(batch))
        if len(records) != 70 or len({(r['sample']['task'], r['sample']['query_id']) for r in records}) != 70:
            raise ValueError('Final sample set differs from the frozen 70 unique queries')
        summary = summarize(records, plan['task_order'])
        expansion.write(output / 'summary.json', summary)
        status = 'complete' if summary['valid'] == 70 else 'complete_with_sample_failures'
        # Stop all writers before hashing the service log and sealing completion.
        terminate(server)
        server = None
        artifacts = {p.name: expansion.sha(p) for p in sorted(output.iterdir()) if p.is_file()}
        expansion.write(output / '_COMPLETE', dict(status=status, completed=70, valid=summary['valid'],
             invalid=70 - summary['valid'], summary_sha256=expansion.sha(output / 'summary.json'),
             launch_plan_sha256=expansion.sha(output / 'launch-plan.json'), artifact_sha256=artifacts))
        print(f'[pilot] {status}: {output}; valid={summary["valid"]}/70', flush=True)
        return 0 if summary['valid'] == 70 else 1
    except BaseException as exc:
        expansion.write(output / '_FAILED.json', dict(error=str(exc), completed=len(records),
            gpu_probe_started=gpu_probe_started, gpu_started=gpu_probe_started,
            model_server_started=model_server_started, local_corpus_destination=str(local_destination),
            local_corpus_preserved=local_destination.exists(),
            inspect_tools_directory=str(tools_directory),
            partial_summary=summarize(records, plan['task_order'])))
        raise
    finally:
        if server is not None:
            terminate(server)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    raise SystemExit(main())
