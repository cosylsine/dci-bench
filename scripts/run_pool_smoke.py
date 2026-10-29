#!/usr/bin/env python3
"""Seal and run the paired 53-source, three-tier Qwen3.8-27B smoke.

The data preparer preserves existing pools. This driver owns orchestration only.
Every invocation creates a new result directory; no model runs during sealing,
dry-run or CPU preflight. Failed attempts and partial results are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
# Direct frozen launchers must resolve their own package without inherited PYTHONPATH.
sys.path.insert(0, str(ROOT))
BENCH = '/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python'
SERVING = '/mnt/afs/250010100/miniconda/envs/dci-vllm-qwen38/bin/python'
MODEL = 'Qwen3.8-27B'
DEFAULT_MODEL = '/mnt/afs2/202610/Qwen3.8-27B'
CAPACITIES = {200: 'small', 500: 'medium', 1000: 'large'}
CHILDREN = set()
CHILDREN_LOCK = threading.RLock()
STOPPING = threading.Event()


def track(process):
    with CHILDREN_LOCK:
        CHILDREN.add(process)
    if STOPPING.is_set():
        terminate(process)
        raise RuntimeError('Driver is stopping; new child was terminated')
    return process


def stop_children():
    STOPPING.set()
    while True:
        with CHILDREN_LOCK:
            children = list(CHILDREN)
        if not children:
            break
        for process in children:
            terminate(process)


def parallel_checked(function, rows, concurrency):
    """Keep only concurrency jobs queued and stop all children on infrastructure failure."""
    iterator = iter(rows)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        pending = {executor.submit(function, row) for row in [next(iterator, None) for _ in range(concurrency)]
                   if row is not None}
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                completed = [future.result() for future in done]
                for result in completed:
                    yield result
                for _ in done:
                    row = next(iterator, None)
                    if row is not None:
                        pending.add(executor.submit(function, row))
        except BaseException:
            for future in pending:
                future.cancel()
            stop_children()
            raise


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def environment(runtime, profile=None):
    env = dict(os.environ)
    for key in ['DCI_PROTOCOL_PROFILE', 'DCI_PROTOCOL_CONTRACT_SHA256', 'DCI_CODE_RUNTIME_CONTRACT',
                'PYTHONHOME', 'NODE_PATH', 'NODE_OPTIONS', 'KV_CACHE_DTYPE', 'VLLM_NVFP4_GEMM_BACKEND']:
        env.pop(key, None)
    env.update(PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE='1', TSX_DISABLE_CACHE='1',
               DCI_EVALUATION_TRACK='document',
               PATH=str(Path(BENCH).parent) + os.pathsep + env.get('PATH', ''),
               LD_LIBRARY_PATH=str(Path(BENCH).parent.parent / 'lib'))
    if profile:
        env['DCI_PROTOCOL_PROFILE'] = profile
    for key in ['NO_PROXY', 'no_proxy']:
        env[key] = env.get(key, '') + ',127.0.0.1,localhost'
    return env


def validate_rows(rows):
    if len(rows) != 159:
        raise ValueError('Expected exactly 159 task/tier executions')
    tasks = {}
    for row in rows:
        task, query, capacity = row['task'], row['query_id'], row['capacity']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', task) or not isinstance(query, str) or not query:
            raise ValueError('Unsafe task or missing original query ID')
        if capacity not in CAPACITIES or row['profile'] != 'document-pool-v1-' + CAPACITIES[capacity]:
            raise ValueError('Pool capacity/profile mismatch')
        group = tasks.setdefault(task, {})
        if capacity in group:
            raise ValueError('Duplicate task/tier')
        group[capacity] = query
    if len(tasks) != 53 or any(set(g) != set(CAPACITIES) or len(set(g.values())) != 1
                               for g in tasks.values()):
        raise ValueError('Each of 53 sources must have one identical query across three tiers')


def schedule(rows, bridge_base=29131):
    validate_rows(rows)
    if not 1024 <= bridge_base <= 65535 - len(rows) + 1:
        raise ValueError('Bridge range exceeds TCP bounds')
    return [dict(row, bridge_port=bridge_base + index, run_id=f"smoke-{row['capacity']}")
            for index, row in enumerate(sorted(rows, key=lambda r: (r['capacity'], r['task'])))]


def check_ports(port, work):
    ports = [port, *(r['bridge_port'] for r in work)]
    if len(set(ports)) != len(ports) or not all(1024 <= p <= 65535 for p in ports):
        raise ValueError('Invalid or overlapping server/bridge ports')
    sockets = []
    try:
        for value in ports:
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(('127.0.0.1', value))
    finally:
        for sock in sockets:
            sock.close()


def tree_inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f'Snapshot dependency escapes runtime: {relative}')
            result[relative] = {'symlink': os.readlink(path)}
        elif path.is_file():
            result[relative] = {'sha256': sha(path)}
    return result


def seal(data_root, bundle, results_root):
    if {'results', 'eval_data', 'raw_data', 'eval_data_code'} & set(bundle.resolve().parts):
        raise ValueError('Executable bundles belong under the code repository releases/ directory')
    plan = read(data_root / 'plan.json')
    validate_rows(plan['rows'])
    if not (data_root / '_COMPLETE.json').is_file():
        raise ValueError('Data projection is not complete')
    bundle.mkdir(parents=True, exist_ok=False)
    runtime = bundle / 'runtime'
    runtime.mkdir()
    ignore = shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '.pytest_cache')
    for name in ['dci_bench', 'schemas', 'scripts']:
        shutil.copytree(ROOT / name, runtime / name, symlinks=True, ignore=ignore)
    pi = runtime / 'pi-dci'
    pi.mkdir()
    for name in ['scripts', 'packages', 'node_modules']:
        shutil.copytree(ROOT / 'pi-dci' / name, pi / name, symlinks=True, ignore=ignore)
    for path in (ROOT / 'pi-dci').iterdir():
        if path.is_file() and path.suffix in {'.json', '.ts', '.js', '.mjs'}:
            shutil.copy2(path, pi / path.name)
    expected = ROOT / 'releases/protocol-freezes/20261015-selective-full-v1/expected-backend.json'
    shutil.copy2(expected, bundle / 'expected-backend.json')
    shutil.copy2(data_root / 'plan.json', bundle / 'plan.json')
    contracts = {}
    for tier in CAPACITIES.values():
        profile = 'document-pool-v1-' + tier
        payload = json.loads(subprocess.check_output([BENCH, '-B', '-c',
            'import json; from dci_bench.protocol.contracts import CONTRACT_SHA256, protocol_contract; '
            'print(json.dumps(dict(sha256=CONTRACT_SHA256,contract=protocol_contract())))'],
            cwd=runtime, env=environment(runtime, profile), text=True))
        if canonical(payload['contract']) != payload['sha256']:
            raise ValueError('Protocol hash does not match contract')
        contracts[profile] = payload
    write(bundle / 'contracts.json', contracts)
    frozen = dict(schema_version='dci-pool-smoke-freeze-v1', created_at=datetime.now(timezone.utc).isoformat(),
                  data_root=str(data_root), data_completion_sha256=sha(data_root / '_COMPLETE.json'),
                  data_plan_sha256=sha(data_root / 'plan.json'), planned_evaluations=159,
                  results_root=str(results_root), bench_python=BENCH, serving_python=SERVING,
                  model_path=DEFAULT_MODEL, max_concurrency=4,
                  packages={p: version(p) for p in ['inspect-ai', 'openai', 'httpx', 'aiohttp']},
                  node_version=subprocess.check_output([str(Path(BENCH).with_name('node')), '--version'],
                                                       text=True, env=environment(runtime)).strip(),
                  files={name: sha(bundle / name) for name in ['plan.json', 'contracts.json', 'expected-backend.json']},
                  pool_completion_sha256={str(Path(r['suite']) / 'metadata' / r['task'] / '_COMPLETE.json'):
                      sha(Path(r['suite']) / 'metadata' / r['task'] / '_COMPLETE.json') for r in plan['rows']},
                  runtime_files=tree_inventory(runtime))
    write(bundle / 'frozen.json', frozen)
    (bundle / 'frozen.sha256').write_text(sha(bundle / 'frozen.json') + '\n')
    launch = '#!/usr/bin/env bash\nset -Eeuo pipefail\nbundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
    launch += f'export DCI_POOL_SMOKE_LOG_DIR={shlex.quote(str(results_root.parent / "job-logs"))}\n'
    launch += f'exec bash "$bundle/runtime/scripts/run_pool_smoke.sh" --bundle "$bundle" "$@"\n'
    (bundle / 'launch.sh').write_text(launch)
    (bundle / 'launch.sh').chmod(0o755)
    print(json.dumps({'status': 'sealed', 'bundle': str(bundle), 'evaluations': 159, 'gpu_started': False}))


def verify_bundle(bundle):
    frozen = read(bundle / 'frozen.json')
    if sha(bundle / 'frozen.json') != (bundle / 'frozen.sha256').read_text().strip():
        raise ValueError('Frozen manifest changed')
    for name, digest in frozen['files'].items():
        if sha(bundle / name) != digest:
            raise ValueError(f'Frozen attachment changed: {name}')
    if tree_inventory(bundle / 'runtime') != frozen['runtime_files']:
        raise ValueError('Frozen runtime source/dependency inventory changed')
    data = Path(frozen['data_root'])
    if (sha(data / '_COMPLETE.json') != frozen['data_completion_sha256']
            or sha(data / 'plan.json') != frozen['data_plan_sha256']):
        raise ValueError('Data projection anchor changed')
    for path, digest in frozen['pool_completion_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'Frozen selected pool identity changed: {path}')
    if {p: version(p) for p in frozen['packages']} != frozen['packages']:
        raise ValueError('Benchmark Python packages changed')
    env = environment(bundle / 'runtime')
    if subprocess.check_output([str(Path(BENCH).with_name('node')), '--version'], text=True,
                               env=env).strip() != frozen['node_version']:
        raise ValueError('Node version changed')
    return frozen


def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        with CHILDREN_LOCK:
            CHILDREN.discard(process)


def step(name, command, output, runtime, env, timeout=1800, accepted=(0,)):
    if STOPPING.is_set():
        raise RuntimeError('Driver is stopping; refusing another step')
    log = output / 'logs' / (name + '.log')
    print(f'[smoke] {name}: {log}', flush=True)
    started = time.monotonic()
    with log.open('x') as handle:
        process = track(subprocess.Popen(command, cwd=runtime, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True))
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            terminate(process)
            raise
        finally:
            with CHILDREN_LOCK:
                CHILDREN.discard(process)
    write(output / 'logs' / (name + '.json'), dict(command=command, exit_code=code,
          elapsed_seconds=time.monotonic() - started))
    if code not in accepted:
        raise RuntimeError(f'{name} failed ({code}); see {log}')
    return code


def task_command(row, runtime, output, tools_directory=None, preflight=False):
    suite = Path(row['suite'])
    runner = runtime / 'scripts/run_document_pool.py'
    prefix = [BENCH, '-B', str(runner)]
    if tools_directory is not None:
        prefix = [BENCH, '-B', str(runtime / 'scripts/inspect_local_runtime.py'),
                  '--tools-directory', str(tools_directory), '--runner', str(runner), '--']
    command = [*prefix, '--task', row['task'], '--metadata-root', str(suite / 'metadata'),
               '--workspace-root', str(suite / 'workspaces'), '--query-ids', row['query_id'],
               '--model-key', MODEL, '--run-id', row['run_id'], '--results-root', str(output / 'evaluation'),
               '--max-concurrency', '1', '--bridge-port', str(row['bridge_port'])]
    if preflight:
        command.append('--preflight-only')
    else:
        command += ['--backend-manifest', str(output / 'backend.json'), '--require-valid-output']
        for name in ['runtime.json', 'models.json', 'warmup.json']:
            command += ['--backend-artifact', str(output / name)]
    return command


def data_preflight(work, runtime, output, concurrency):
    def run(row):
        name = f"data-{row['task']}-{row['capacity']}"
        step(name, task_command(row, runtime, output, preflight=True), output, runtime,
             environment(runtime, row['profile']))
        result = read(output / 'logs' / (name + '.log'))
        if (result['query_ids'] != [row['query_id']] or result['capacity'] != row['capacity']
                or result['profile'] != row['profile'] or result['model_calls'] != 0):
            raise ValueError(f'Pool routing mismatch: {name}')
        return {'task': row['task'], 'capacity': row['capacity'], 'query_id': row['query_id'],
                'contract_sha256': result['contract_sha256']}
    checks = list(parallel_checked(run, work, concurrency))
    write(output / 'data-preflight.json', dict(status='passed', evaluations=len(checks), checks=checks,
          selected_pool_content_hashes_verified=True, full_source_content_hashes_verified=False,
          sandbox_verified=False, external_model_calls=0, gpu_started=False))


def model_size_preflight(model_path, expected):
    if sha(model_path / 'config.json') != expected['model']['config_sha256']:
        raise ValueError('Model config differs from the accepted Qwen3.8-27B checkpoint')
    files = {p.name: p.stat().st_size for p in model_path.iterdir() if p.suffix in {'.safetensors', '.bin'}}
    if files != {r['name']: r['size_bytes'] for r in expected['model']['weights']['files']}:
        raise ValueError('Model weight file names/sizes differ from accepted checkpoint')
    return dict(config_sha256=sha(model_path / 'config.json'), weight_files=len(files),
                weight_bytes=sum(files.values()), weight_content_hashes_verified=False)


def verify_batch(row, output, contract):
    from dci_bench.results.phase4_runner import load_completed_sample
    from dci_bench.results.protocol import validate_task_result, validate_run_manifest, verify_artifact_record
    run = output / 'evaluation' / MODEL / row['task'] / 'runs' / row['run_id']
    result, manifest, marker = (read(run / n) for n in ['task-result.json', 'run-manifest.json', '_COMPLETE'])
    validate_task_result(result)
    validate_run_manifest(manifest)
    actual_contract = dict(manifest['benchmark_contract'])
    context = actual_contract.pop('task_context')
    if (actual_contract != contract['contract'] or context['task_name'] != row['task']
            or marker['task_result_sha256'] != canonical(result)
            or marker['run_fingerprint'] != manifest['run_fingerprint']
            or result['query_ids'] != [row['query_id']] or manifest['query_ids'] != [row['query_id']]
            or result['task'] != row['task'] or result['run_id'] != row['run_id']
            or result['counts']['completed'] != 1 or len(result['samples']) != 1):
        raise ValueError(f'Smoke completion mismatch: {run}')
    sample = load_completed_sample(run, row['query_id'])
    if sample is None or canonical(sample) != result['samples'][0]['result']['sha256']:
        raise ValueError(f'Sample hash mismatch: {run}')
    for ref in [result['integrity'].get('backend_manifest'), *result['integrity'].get('backend_diagnostics', [])]:
        if ref:
            verify_artifact_record(ref, root=run)
    exercised = sample['execution']['model_calls'] > 0 and sample['execution']['tool_calls'] > 0
    return dict(task=row['task'], capacity=row['capacity'], query_id=row['query_id'],
                run=str(run), valid=sample['valid_output'], real_model_and_tools=exercised,
                failure=sample['failure'], usage=sample['usage'], timing=sample['timing'], metrics=sample['metrics'])


def summarize(records):
    records = sorted(records, key=lambda r: (r.get('capacity', 0), r.get('task', '')))
    passed = sum(r['valid'] and r['real_model_and_tools'] for r in records)
    return dict(planned=159, completed=len(records), valid=sum(r['valid'] for r in records),
                smoke_passed=passed, status='passed' if len(records) == passed == 159 else 'not_passed',
                failure_counts=dict(Counter((r['failure'] or {}).get('kind', 'missing_model_or_tools')
                    for r in records if not (r['valid'] and r['real_model_and_tools']))), samples=records,
                interpretation='159 diagnostic executions, 53 paired queries; not a completed benchmark.')


def server_owns_port(server, port):
    """Reject a same-name model served by another process after a bind race."""
    inodes = set()
    for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
        parts = line.split()
        if int(parts[1].split(':')[1], 16) == port and parts[3] == '0A':
            inodes.add(parts[9])
    owned = set()
    for process in Path('/proc').glob('[0-9]*'):
        try:
            if os.getpgid(int(process.name)) != server.pid:
                continue
            for link in (process / 'fd').iterdir():
                try:
                    match = re.fullmatch(r'socket:\[(\d+)\]', os.readlink(link))
                    if match:
                        owned.add(match[1])
                except OSError:
                    continue
        except (OSError, ProcessLookupError):
            continue
    return server.poll() is None and bool(inodes) and inodes.issubset(owned)


def run(args):
    bundle = args.bundle.resolve()
    frozen = verify_bundle(bundle)
    runtime = bundle / 'runtime'
    # Always execute the sealed driver, including preflight and orchestration.
    if Path(__file__).resolve() != runtime / 'scripts/run_pool_smoke.py':
        os.execve(BENCH, [BENCH, '-B', str(runtime / 'scripts/run_pool_smoke.py'), *sys.argv[1:]],
                  environment(runtime))
    # Exercise the parent-process import during dry-run/CPU preflight too.
    from dci_bench.backends.manifest import validate_backend_manifest
    work = schedule(read(bundle / 'plan.json')['rows'], args.bridge_port)
    contracts = read(bundle / 'contracts.json')
    expected = read(bundle / 'expected-backend.json')
    model_path = Path(args.model_path or os.environ.get('MODEL_PATH', DEFAULT_MODEL)).resolve()
    launch = dict(bundle=str(bundle), bundle_sha256=sha(bundle / 'frozen.json'), work=work,
                  model=MODEL, model_path=str(model_path), concurrency=args.concurrency,
                  tensor_parallel_size=4, context_length=65536, dtype='bfloat16', server_port=args.port,
                  bridge_ports=[work[0]['bridge_port'], work[-1]['bridge_port']],
                  budgets={200: 96000, 500: 128000, 1000: 192000}, sample_timeout_seconds=600,
                  source_content_hash_scope='Selected candidate documents only; original full corpus not rehashed')
    if args.dry_run or os.environ.get('DRY_RUN') == '1':
        print(json.dumps(dict(launch, mode='dry-run', gpu_started=False), indent=2))
        return 0
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', args.run_id):
        raise ValueError('Unsafe run ID')
    output = Path(frozen['results_root']) / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    write(output / 'launch-plan.json', launch)
    server = None
    gpu_started = False
    records = []
    try:
        check_ports(args.port, work)
        data_preflight(work, runtime, output, args.concurrency)
        checks = read(output / 'data-preflight.json')['checks']
        for row in checks:
            profile = 'document-pool-v1-' + CAPACITIES[row['capacity']]
            if row['contract_sha256'] != contracts[profile]['sha256']:
                raise ValueError('Data preflight used a changed tier contract')
        write(output / 'model-location.json', model_size_preflight(model_path, expected))
        if args.data_preflight_only:
            print(f'[smoke] CPU data preflight passed: {output}; no GPU/model calls', flush=True)
            return 0
        for path in ['/usr/bin/bwrap', '/usr/bin/rg']:
            if not os.access(path, os.X_OK):
                raise RuntimeError(f'Missing trusted dependency {path}; install bubblewrap/ripgrep before launch')
        with tempfile.TemporaryDirectory(prefix='dci-pool-smoke-') as temporary:
            local = Path(temporary)
            import inspect_local_runtime
            tools_directory = local / 'inspect-tools'
            write(output / 'inspect-tools.json', inspect_local_runtime.prepare(tools_directory))
            # Real Inspect, HTTP bridge and Pi sandbox; synthetic provider only.
            step('inspect-sandbox', [BENCH, '-B', str(runtime / 'scripts/check_inspect_parallel.py'),
                 '--runtime', str(runtime), '--tools-directory', str(tools_directory),
                 '--output', str(output / 'inspect-preflight'), '--max-concurrency', str(args.concurrency)],
                 output, runtime, environment(runtime), timeout=600)
            # Include the actual pool runner and result publication, not only
            # the solver, in the no-GPU acceptance boundary.
            step('pool-publication', [BENCH, '-B', str(runtime / 'scripts/check_pool_execution.py'),
                 '--bundle', str(bundle), '--output', str(output / 'pool-publication-preflight'),
                 '--tools-directory', str(tools_directory), '--concurrency', str(min(args.concurrency, 3))],
                 output, runtime, environment(runtime), timeout=900)
            # Each tier also traverses real selected-pool materialization and Pi preflight.
            def sandbox(row):
                step(f"sandbox-{row['task']}-{row['capacity']}", [BENCH, '-B', str(__file__),
                     '--bundle', str(bundle), '--sandbox-row', str(work.index(row)),
                     '--bridge-port', str(args.bridge_port)], output, runtime,
                     environment(runtime, row['profile']), timeout=300)
            list(parallel_checked(sandbox, work, args.concurrency))
            write(output / 'preflight.json', dict(status='passed', evaluations=159,
                  selected_pool_content_hashes_verified=True, sandbox_verified=True,
                  external_model_calls=0, gpu_started=False))
            if args.preflight_only:
                print(f'[smoke] CPU + real sandbox preflight passed: {output}', flush=True)
                return 0
            check_ports(args.port, work)
            env = environment(runtime)
            env.update(MODEL_PATH=str(model_path), SERVED_MODEL_NAME=MODEL, TENSOR_PARALLEL_SIZE='4',
                       MAX_MODEL_LEN='65536', DTYPE='bfloat16', TOOL_CALL_PARSER='qwen3_coder',
                       REASONING_PARSER='qwen3', LANGUAGE_MODEL_ONLY='1', VLLM_HOST='127.0.0.1',
                       VLLM_PORT=str(args.port), VLLM_MAX_NUM_SEQS=str(args.concurrency),
                       VLLM_GPU_MEMORY_UTILIZATION='0.9', VLLM_EXTRA_ARGS='--enforce-eager',
                       CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3'),
                       NVIDIA_DRIVER_CAPABILITIES='compute,utility')
            serving_env = dict(env, PATH=str(Path(SERVING).parent) + os.pathsep + env['PATH'],
                               LD_LIBRARY_PATH='/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu')
            # Hash checkpoint bytes before touching CUDA, then bind that identity
            # to the measured serving runtime without a second weight scan.
            step('model-identity', [BENCH, '-B', '-m', 'dci_bench.backends.manifest', '--output', str(output / 'model-preflight.json'),
                 '--model-key', MODEL, '--model-path', str(model_path), '--served-model-name', MODEL,
                 '--base-url', f'http://127.0.0.1:{args.port}/v1', '--vllm-version', expected['backend']['version'],
                 '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3', '--language-model-only',
                 '--sampling-backend', 'auto', '--tensor-parallel-size', '4', '--context-length', '65536',
                 '--dtype', 'bfloat16', '--max-num-seqs', str(args.concurrency), '--gpu-memory-utilization', '0.9'],
                 output, runtime, env)
            backend = read(output / 'model-preflight.json')
            if (backend['model']['weights'] != expected['model']['weights']
                    or backend['model']['config_sha256'] != expected['model']['config_sha256']):
                raise ValueError('Mounted model weight identity differs from frozen baseline')
            gpu_started = True
            step('runtime', [SERVING, '-B', str(runtime / 'scripts/harness_pilot_runtime.py'),
                 '--output', str(output / 'runtime.json')], output, runtime, serving_env)
            actual = read(output / 'runtime.json')
            gpus = actual['cuda']['gpus']
            if len(gpus) != 4 or any('H100' not in g['name'] or g['total_memory_bytes'] < 75 * 1024**3 for g in gpus):
                raise ValueError('Frozen launch requires exactly four visible H100 80GB GPUs')
            if actual['packages'] != expected['backend']['runtime']['packages']:
                raise ValueError('Serving packages differ from accepted backend')
            backend['backend']['runtime'] = actual
            backend['manifest_sha256'] = canonical({k: v for k, v in backend.items() if k != 'manifest_sha256'})
            validate_backend_manifest(backend)
            write(output / 'backend.json', backend)
            with (output / 'server.log').open('x') as log:
                server = track(subprocess.Popen(['bash', str(runtime / 'scripts/serve_vllm_qwen38.sh')],
                         cwd=runtime, env=serving_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Model server exited; inspect server.log')
                try:
                    with opener.open(f'http://127.0.0.1:{args.port}/v1/models', timeout=5) as response:
                        models = json.load(response)
                    if MODEL not in [r['id'] for r in models['data']]:
                        raise ValueError('Unexpected served model')
                    if not server_owns_port(server, args.port):
                        raise RuntimeError('Endpoint is not owned by the newly started server process group')
                    write(output / 'models.json', models)
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Model readiness timed out')
                    time.sleep(5)
            step('warmup', [BENCH, '-B', str(runtime / 'scripts/harness_pilot_warmup.py'),
                 '--base-url', f'http://127.0.0.1:{args.port}/v1', '--model', MODEL,
                 '--output', str(output / 'warmup.json'), '--concurrency', str(args.concurrency),
                 '--deadline-seconds', '600', '--http-timeout-seconds', '600'], output, runtime, env)
            def evaluate(row):
                step(f"eval-{row['task']}-{row['capacity']}", task_command(row, runtime, output, tools_directory),
                     output, runtime, environment(runtime, row['profile']), timeout=1200, accepted=(0, 1))
                return verify_batch(row, output, contracts[row['profile']])
            for result in parallel_checked(evaluate, work, args.concurrency):
                records.append(result)
                print(f'[smoke] audited {len(records)}/159', flush=True)
            terminate(server)
            server = None
            summary = summarize(records)
            write(output / 'summary.json', summary)
            write(output / '_COMPLETE', dict(status='complete' if summary['status'] == 'passed' else 'complete_with_sample_failures',
                  completed=len(records), valid=summary['valid'], smoke_passed=summary['smoke_passed'],
                  summary_sha256=sha(output / 'summary.json'), launch_plan_sha256=sha(output / 'launch-plan.json'),
                  artifacts={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
            print(f"[smoke] {summary['status']}: {output}; passed={summary['smoke_passed']}/159", flush=True)
            return 0 if summary['status'] == 'passed' else 1
    except BaseException as exc:
        write(output / '_FAILED.json', dict(error=str(exc), gpu_started=gpu_started,
              model_server_started=server is not None, partial_summary=summarize(records),
              partial_count_scope='Rows audited and collected by driver; additional completed task artifacts may exist in evaluation.'))
        raise
    finally:
        if server is not None:
            terminate(server)


def sandbox_row(args):
    # Child uses the parent's already-verified sealed source, one tier per process.
    runtime = args.bundle.resolve() / 'runtime'
    row = schedule(read(args.bundle / 'plan.json')['rows'], args.bridge_port)[args.sandbox_row]
    os.environ['DCI_PROTOCOL_PROFILE'] = row['profile']
    from dci_bench.data.document_runtime import load_pool_for_runtime, materialize_query_workspace
    from dci_bench.protocol.contracts import protocol_contract
    suite = Path(row['suite'])
    pool = load_pool_for_runtime(suite / 'metadata', suite / 'workspaces', row['task'], protocol_contract())
    with materialize_query_workspace(suite / 'workspaces' / row['task'], row['query_id'], pool) as workspace:
        if len(list((workspace / 'corpus').iterdir())) != row['capacity']:
            raise ValueError('Materialized candidate count differs from tier')
        subprocess.run(['node', '--import', 'tsx', 'scripts/dci-run.ts', '--preflight-only',
                        '--workspace', str(workspace)], cwd=runtime / 'pi-dci', check=True)
    if workspace.exists():
        raise ValueError('Temporary candidate workspace was not cleaned')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--seal', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--data-preflight-only', action='store_true')
    modes.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'results/Qwen3.8-27B/pool-smoke-runs')
    parser.add_argument('--model-path')
    parser.add_argument('--concurrency', type=int, default=4, choices=range(1, 5))
    parser.add_argument('--port', type=int, default=29099)
    parser.add_argument('--bridge-port', type=int, default=29131)
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'-{os.getpid()}')
    parser.add_argument('--sandbox-row', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sandbox_row is not None:
        sandbox_row(args)
        return 0
    if args.seal:
        if args.data_root is None:
            parser.error('--seal requires --data-root')
        seal(args.data_root.resolve(), args.bundle.resolve(), args.results_root.resolve())
        return 0
    return run(args)


if __name__ == '__main__':
    def interrupted(signum, _frame):
        stop_children()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
