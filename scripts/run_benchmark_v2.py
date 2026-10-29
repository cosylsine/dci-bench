#!/usr/bin/env python3
"""Plan, CPU-preflight, or execute a frozen Lite/Full suite against one running service.

Default is a non-inference plan. --execute starts batches; test execution requires
a complete suite, a measured development report, and a matching cost gate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.results.benchmark_v2 import (
    TIERS,
    PriceQuote,
    cost_gate,
    fingerprint,
    select_budget,
    summarize,
    validate_complete_report,
)
from dci_bench.results.protocol import (
    sha256_file,
    sha256_json,
    validate_run_manifest,
    validate_sample_result,
    validate_task_result,
    verify_artifact_record,
    write_json_atomic,
)


def source_fingerprint() -> str:
    paths = [*ROOT.joinpath('dci_bench').rglob('*.py'), *ROOT.joinpath('dci_bench/protocol').glob('*.json'),
             ROOT / 'pi-dci/scripts/dci-run.ts', ROOT / 'scripts/run_document_pool.py',
             ROOT / 'scripts/run_task.py', ROOT / 'scripts/inspect_local_runtime.py', Path(__file__).resolve()]
    return fingerprint({str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)})


def build_plan(suite: Path, *, split: str, concurrency: int, token_budget: int,
               backend_path: Path | None, allow_partial: bool = False) -> dict:
    from dci_bench.data.benchmark_v2 import audit_benchmark_v2
    audit_benchmark_v2(suite)
    selection = json.loads((suite / 'benchmark-manifest.json').read_text())
    if split not in ('smoke', 'dev', 'test') or concurrency not in (4, 8, 16):
        raise ValueError('split must be smoke/dev/test and concurrency must be 4/8/16')
    if token_budget not in (32000, 64000, 96000):
        raise ValueError('token budget must be 32000/64000/96000')
    if split == 'test' and not selection['ready_for_formal']:
        raise ValueError('formal evaluation requires all 20 tasks ready; missing cells cannot be skipped')
    blocked = [task for task, row in selection['tasks'].items() if row['status'] != 'ready']
    if blocked and not allow_partial:
        raise ValueError('incomplete suite; only smoke/dev can use --allow-partial: ' + ', '.join(blocked))
    jobs = []
    backend = json.loads(backend_path.read_text()) if backend_path else None
    expected_model_key = backend['model']['model_key'] if backend else None
    expected_backend_sha256 = sha256_file(backend_path) if backend_path else None
    for task_id, row in sorted(selection['tasks'].items()):
        if row['status'] != 'ready':
            continue
        ids = row['query_ids'][split]
        if len(ids) != {'smoke': 1, 'dev': 10, 'test': 50}[split]:
            raise ValueError(f'{task_id}: invalid {split} count')
        for tier, capacity in TIERS.items():
            pool = (suite / row['pools'][tier]['suite']).resolve()
            if not pool.is_relative_to(suite.resolve()):
                raise ValueError('prepared pool is outside the frozen suite')
            pool_manifest_path = pool / 'metadata' / task_id / 'manifest.json'
            pool_manifest = json.loads(pool_manifest_path.read_text())
            if not isinstance(pool_manifest, dict):
                raise ValueError(f'{task_id}/{tier}: frozen pool manifest must be an object')
            source_manifest_sha256 = pool_manifest.get('source_manifest_sha256')
            if (not isinstance(source_manifest_sha256, str) or len(source_manifest_sha256) != 64
                    or any(character not in '0123456789abcdef' for character in source_manifest_sha256)):
                raise ValueError(f'{task_id}/{tier}: frozen pool source_manifest_sha256 is invalid')
            if row.get('source_manifest_sha256') not in (None, source_manifest_sha256):
                raise ValueError(f'{task_id}/{tier}: task and pool source manifests differ')
            pool_manifest_sha256 = sha256_file(pool_manifest_path)
            expected_profile = row['pools'][tier].get('profile', f'document-pool-v2-{tier}')
            jobs.append({'task_id': task_id, 'tier': tier, 'capacity': capacity,
                         'suite': str(pool), 'query_ids': ids,
                         'groups': {qid: row['query_groups'][qid] for qid in ids},
                         'expected_profile': expected_profile,
                         'expected_budget': token_budget,
                         'expected_model_key': expected_model_key,
                         'expected_backend_sha256': expected_backend_sha256,
                         'source_manifest_sha256': source_manifest_sha256,
                         'pool_manifest_sha256': pool_manifest_sha256})
    if not jobs:
        raise ValueError('no runnable tasks')
    return {'schema_version': 'dci-benchmark-v2-launch', 'split': split, 'concurrency': concurrency,
            'token_budget': token_budget, 'selection_sha256': sha256_file(suite / 'benchmark-manifest.json'),
            'source_fingerprint': source_fingerprint(), 'backend_sha256': expected_backend_sha256,
            'model_key': expected_model_key,
            'mode': 'api' if backend and backend['backend']['kind'] == 'remote_api' else 'local',
            'jobs': jobs, 'blocked_tasks': blocked, 'complete_scope': not blocked,
            'execution_count': sum(len(job['query_ids']) for job in jobs),
            'expected_tasks': sorted(selection['tasks'])}


def command_for(job: dict, args: argparse.Namespace, model_key: str | None, *, preflight: bool = False) -> list[str]:
    prefix = [sys.executable, str(ROOT / 'scripts/run_document_pool.py')]
    if not preflight and (args.output / 'inspect-tools/_READY.json').exists():
        prefix = [sys.executable, str(ROOT / 'scripts/inspect_local_runtime.py'),
                  '--tools-directory', str(args.output / 'inspect-tools'),
                  '--runner', str(ROOT / 'scripts/run_document_pool.py'), '--']
    command = [*prefix, '--task', job['task_id'],
               '--metadata-root', str(Path(job['suite']) / 'metadata'),
               '--workspace-root', str(Path(job['suite']) / 'workspaces'),
               '--query-ids', *job['query_ids'], '--max-concurrency', str(args.concurrency)]
    if preflight:
        return [*command, '--preflight-only']
    run_id = args.split + '-' + job['tier']
    task_run = args.output / 'evaluation' / model_key / job['task_id'] / 'runs' / run_id
    return [*command, '--model-key', model_key, '--backend-manifest', str(args.backend_manifest),
            '--results-root', str(args.output / 'evaluation'), '--bridge-port', str(args.bridge_port),
            '--resume-run' if task_run.exists() else '--run-id', run_id]


def completed_samples(run: Path, job: dict) -> list[dict]:
    marker = json.loads((run / '_COMPLETE').read_text())
    result = json.loads((run / 'task-result.json').read_text())
    if marker.get('task_result_sha256') != sha256_json(result):
        raise ValueError('completed batch result hash mismatch')
    result_query_ids = result.get('query_ids')
    if (not isinstance(result_query_ids, list) or any(not isinstance(value, str) for value in result_query_ids)
            or set(result_query_ids) != set(job['query_ids'])):
        raise ValueError('completed batch query selection changed')
    manifest_path = run / 'run-manifest.json'
    if not manifest_path.is_file():
        raise ValueError('completed batch run-manifest.json is missing')
    manifest = json.loads(manifest_path.read_text())
    validate_run_manifest(manifest)
    validate_task_result(result)
    if marker.get('schema_version') != 'dci-run-complete-v1':
        raise ValueError('completed batch marker schema mismatch')
    if marker.get('run_id') != manifest.get('run_id'):
        raise ValueError('completed batch marker run identity mismatch')
    if marker.get('run_fingerprint') != manifest.get('run_fingerprint'):
        raise ValueError('completed batch marker run fingerprint mismatch')
    if result.get('run_id') != manifest.get('run_id'):
        raise ValueError('completed batch task result run identity mismatch')
    task_id = job.get('task_id')
    if task_id is not None:
        if manifest.get('task') != task_id or result.get('task') != task_id:
            raise ValueError('completed batch task identity mismatch')
    if manifest.get('model_key') != manifest.get('model', {}).get('model_key'):
        raise ValueError('completed batch run manifest model identity mismatch')
    if result.get('task') != manifest.get('task'):
        raise ValueError('completed batch task result task identity mismatch')
    if list(manifest.get('query_ids', [])) != list(result.get('query_ids', [])):
        raise ValueError('completed batch manifest/task query IDs differ')
    if list(result.get('query_ids', [])) != list(job['query_ids']):
        raise ValueError('completed batch query ID order changed')

    expected_profile = job.get('expected_profile')
    is_v2 = isinstance(expected_profile, str) and expected_profile.startswith('document-pool-v2-')
    contract = manifest.get('benchmark_contract', {})
    if is_v2:
        if contract.get('protocol_profile') != expected_profile:
            raise ValueError('completed batch protocol profile mismatch')
        if contract.get('benchmark_contract_version') != 'dci-document-pool-v2':
            raise ValueError('completed batch benchmark contract mismatch')
        expected_budget = job.get('expected_budget')
        if expected_budget is not None and manifest.get('budgets', {}).get('max_total_model_tokens') != expected_budget:
            raise ValueError('completed batch token budget mismatch')
    expected_model = job.get('expected_model_key')
    if expected_model is not None:
        if manifest.get('model_key') != expected_model or manifest.get('model', {}).get('model_key') != expected_model:
            raise ValueError('completed batch model identity mismatch')
        if result.get('model', {}).get('model_key') != expected_model:
            raise ValueError('completed batch task result model identity mismatch')

    expected_backend_sha256 = job.get('expected_backend_sha256')
    if expected_backend_sha256 is not None:
        backend_copy = run / 'artifacts' / 'backend' / 'backend-manifest.json'
        if not backend_copy.is_file() or sha256_file(backend_copy) != expected_backend_sha256:
            raise ValueError('completed batch backend artifact hash mismatch')
        backend_record = result.get('integrity', {}).get('backend_manifest')
        if not isinstance(backend_record, dict):
            raise ValueError('completed batch backend artifact record is missing')
        verify_artifact_record(backend_record, root=run)
        if backend_record.get('sha256') != expected_backend_sha256:
            raise ValueError('completed batch task backend artifact hash mismatch')

    pool_manifest_sha256 = job.get('pool_manifest_sha256')
    if pool_manifest_sha256 is not None:
        data = manifest.get('data', {})
        if data.get('source_manifest_sha256') != pool_manifest_sha256:
            raise ValueError('completed batch frozen pool manifest hash mismatch')
        if result.get('data', {}).get('source_manifest_sha256') != pool_manifest_sha256:
            raise ValueError('completed batch task frozen pool manifest hash mismatch')
    source_manifest_sha256 = job.get('source_manifest_sha256')
    if source_manifest_sha256 is not None:
        if task_id is None:
            raise ValueError('completed batch source manifest binding requires task identity')
        pool_manifest_path = Path(job['suite']) / 'metadata' / task_id / 'manifest.json'
        pool_manifest = json.loads(pool_manifest_path.read_text())
        if pool_manifest.get('source_manifest_sha256') != source_manifest_sha256:
            raise ValueError('completed batch source manifest identity mismatch')

    groups = job.get('groups', {})
    samples = []
    seen_query_ids = set()
    for entry in result['samples']:
        if not isinstance(entry, dict) or entry.get('query_id') in seen_query_ids:
            raise ValueError('completed batch sample IDs/count mismatch')
        query_id = entry.get('query_id')
        if query_id not in job['query_ids']:
            raise ValueError('completed batch sample query identity mismatch')
        seen_query_ids.add(query_id)
        sample_path = (run / entry['result']['path']).resolve()
        if not sample_path.is_relative_to(run.resolve()):
            raise ValueError('completed sample path escapes run')
        if not sample_path.is_file():
            raise ValueError('completed sample result is missing')
        sample = json.loads(sample_path.read_text())
        if entry['result']['sha256'] != sha256_json(sample):
            raise ValueError('completed sample result hash mismatch')
        validate_sample_result(sample)
        if sample.get('run_id') != manifest.get('run_id') or sample.get('task') != manifest.get('task'):
            raise ValueError('completed sample run/task identity mismatch')
        if sample.get('query_id') != query_id:
            raise ValueError('completed sample query identity mismatch')
        if entry.get('status') != sample.get('status') or entry.get('valid_output') != sample.get('valid_output'):
            raise ValueError('completed sample index identity mismatch')
        for artifact in sample['artifacts'].values():
            verify_artifact_record(artifact, root=run)
        seal_path = sample_path.parent / '_COMPLETE'
        if seal_path.is_file():
            seal = json.loads(seal_path.read_text())
            # Document-pool v2 changes the required presence of this seal; its
            # wire schema remains the canonical phase-4 v1 marker.
            if seal.get('schema_version') != 'dci-query-result-complete-v1':
                raise ValueError('completed sample seal schema mismatch')
            if seal.get('query_id') != query_id:
                raise ValueError('completed sample seal query identity mismatch')
            if seal.get('run_id') is not None and seal.get('run_id') != manifest.get('run_id'):
                raise ValueError('completed sample seal run identity mismatch')
            if seal.get('task') is not None and seal.get('task') != manifest.get('task'):
                raise ValueError('completed sample seal task identity mismatch')
            if seal.get('result_sha256') != sha256_json(sample):
                raise ValueError('completed sample seal hash mismatch')
        elif is_v2:
            raise ValueError('completed v2 batch is missing per-query completion seal')
        timing_artifact = sample['artifacts'].get('runtime_timing')
        stage_timing = json.loads((run / timing_artifact['path']).read_text()) if timing_artifact else {}
        final_artifact = sample['artifacts'].get('final')
        if final_artifact:
            final = json.loads((run / final_artifact['path']).read_text())
            stage_timing.update(final.get('stage_timing', {}))
        samples.append({'task_id': job['task_id'], 'query_id': sample['query_id'], 'tier': job['tier'],
                        'group_id': groups.get(sample['query_id'], sample['query_id']), 'valid': sample['valid_output'],
                        'metrics': sample['metrics'], 'usage': sample['usage'],
                        'wall_seconds': sample['timing']['wall_time_seconds'], 'failure': sample['failure'],
                        'stage_timing': stage_timing,
                        'result_path': str(run / entry['result']['path'])})
    if (len(samples) != len(job['query_ids']) or
            {sample['query_id'] for sample in samples} != set(job['query_ids'])):
        raise ValueError('completed batch sample IDs/count mismatch')
    return samples


def run_child(command: list[str], *, env: dict, log: Path, timeout: float) -> int:
    with log.open('w') as handle:
        child = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def _report_accounted_cost(report: dict, *, label: str) -> float:
    measured = report.get('cost_cny')
    spending_guard = report.get('spending_guard', {})
    if spending_guard is None:
        spending_guard = {}
    if not isinstance(spending_guard, dict):
        raise ValueError(f'{label} calibration spending guard is invalid')
    guard = spending_guard.get('accounted_or_reserved_cny')
    for value in (measured, guard):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError(f'{label} calibration cost is invalid')
    values = [value for value in (measured, guard) if type(value) in (int, float)
              and math.isfinite(value) and value >= 0]
    if not values:
        raise ValueError(f'{label} calibration cost is unavailable')
    return max(values)


def _reservation_totals(ledger: dict) -> tuple[float, float]:
    """Return all reservations and known first-attempt reservations.

    Unknown reservation keys are deliberately included in the all-attempt total
    but cannot be attributed to a first attempt.  Existing ledgers therefore
    remain conservative across a resume without silently rewriting history.
    """
    reservations = ledger.get('reservations', {})
    if not isinstance(reservations, dict):
        raise ValueError('ledger reservations must be an object')
    all_reserved = 0.0
    for value in reservations.values():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('ledger reservations must be finite and nonnegative')
        all_reserved += value
    first_reserved = 0.0
    seen_batches = set()
    for attempt in ledger.get('attempts', []):
        if not isinstance(attempt, dict):
            raise ValueError('ledger attempts must contain objects')
        batch = attempt.get('batch')
        if not isinstance(batch, str) or not batch:
            continue
        if batch in seen_batches:
            continue
        seen_batches.add(batch)
        key = attempt.get('reservation_key', f"{batch}:{attempt.get('attempt')}")
        value = reservations.get(key, attempt.get('reserved_cny', 0))
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('ledger first-attempt reservation is invalid')
        first_reserved += value
    return all_reserved, first_reserved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    parser.add_argument('--split', choices=['smoke', 'dev', 'test'], default='smoke')
    parser.add_argument('--concurrency', type=int, choices=[4, 8, 16], default=4)
    parser.add_argument('--token-budget', type=int, choices=[32000, 64000, 96000], default=64000)
    parser.add_argument('--backend-manifest', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--budget-selection', type=Path)
    parser.add_argument('--prices', type=Path)
    parser.add_argument('--budget-cny', type=float, default=1000)
    parser.add_argument('--prior-cost-cny', type=float)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--bridge-port', type=int, default=28100)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight-only', action='store_true')
    action.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.budget_cny) or args.budget_cny <= 0:
        raise ValueError('--budget-cny must be finite and positive')
    if args.prior_cost_cny is not None and (
            not math.isfinite(args.prior_cost_cny) or args.prior_cost_cny < 0):
        raise ValueError('--prior-cost-cny must be finite and nonnegative')
    args.suite = args.suite.resolve()
    if args.backend_manifest:
        args.backend_manifest = args.backend_manifest.resolve()
    if args.output:
        args.output = args.output.resolve()
    plan = build_plan(args.suite, split=args.split, concurrency=args.concurrency, token_budget=args.token_budget,
                      backend_path=args.backend_manifest, allow_partial=args.allow_partial)
    if not args.execute and not args.preflight_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.output is None:
        raise ValueError('--output required for preflight/execute')
    if args.execute and args.backend_manifest is None:
        raise ValueError('--backend-manifest required for execute')
    price = PriceQuote.from_dict(json.loads(args.prices.read_text())) if args.prices else None
    if plan['mode'] == 'api' and args.execute and (price is None or price.model_key != plan['model_key']):
        raise ValueError('API execution requires a matching pinned --prices quote')
    if args.execute:
        from dci_bench.backends.manifest import load_backend_manifest
        backend = load_backend_manifest(args.backend_manifest)
        if backend['backend']['context_length'] != 65536:
            raise ValueError('v2 requires a 65536-token context window')
        max_seqs = backend['backend'].get('max_num_seqs')
        if max_seqs is not None and max_seqs < args.concurrency:
            raise ValueError('backend max_num_seqs must cover concurrency')
    if args.split == 'test' and args.execute:
        if args.calibration is None:
            raise ValueError('formal evaluation requires --calibration')
        calibration = json.loads(args.calibration.read_text())
        validate_complete_report(calibration, split='dev')
        for key in ('selection_sha256', 'backend_sha256', 'concurrency', 'token_budget', 'source_fingerprint'):
            if calibration.get(key) != plan[key]:
                raise ValueError('calibration does not match launch: ' + key)
        if calibration.get('split') != 'dev' or calibration.get('completed') != 400 or calibration.get('valid', 0) < 396:
            raise ValueError('formal launch requires 400 dev executions, >=99% valid')
        if args.budget_selection is None:
            raise ValueError('formal evaluation requires --budget-selection from all three dev arms')
        decision = json.loads(args.budget_selection.read_text())
        if decision.get('selected_budget') != args.token_budget:
            raise ValueError('token budget differs from the measured budget selection')
        if decision.get('sha256') != fingerprint({k: v for k, v in decision.items() if k != 'sha256'}):
            raise ValueError('budget selection hash mismatch')
        references = decision.get('reports')
        if not isinstance(references, dict) or set(references) != {'32000', '64000', '96000'}:
            raise ValueError('budget selection must reference all three calibration reports')
        budget_reports = {}
        for budget in (32000, 64000, 96000):
            reference = references[str(budget)]
            if not isinstance(reference, dict) or not isinstance(reference.get('path'), str):
                raise ValueError(f'budget selection report reference is invalid: {budget}')
            report_path = Path(reference['path']).resolve()
            if not report_path.is_file() or reference.get('sha256') != sha256_file(report_path):
                raise ValueError(f'budget selection report hash mismatch: {budget}')
            budget_reports[budget] = json.loads(report_path.read_text())
        independently_selected = select_budget(budget_reports)
        if decision.get('selected_budget') != independently_selected.get('selected_budget'):
            raise ValueError('budget selection decision differs from independently recomputed selection')
        if decision.get('candidates') != independently_selected.get('candidates'):
            raise ValueError('budget selection candidates differ from independently recomputed selection')
        selected_reference = references[str(args.token_budget)]
        if (Path(selected_reference['path']).resolve() != args.calibration.resolve()
                or selected_reference.get('sha256') != sha256_file(args.calibration)):
            raise ValueError('calibration differs from the selected budget-selection report')
        calibration = budget_reports[args.token_budget]
        if plan['mode'] == 'api':
            calibration_costs = {
                budget: _report_accounted_cost(budget_reports[budget], label=f'{budget}')
                for budget in (32000, 64000, 96000)
            }
            if args.prior_cost_cny is None:
                raise ValueError('formal API evaluation requires explicit --prior-cost-cny')
            prior_cost = args.prior_cost_cny
            prior_total = sum(calibration_costs.values()) + prior_cost
        else:
            calibration_costs = {}
            prior_cost = 0.0
            prior_total = 0.0
        plan['budget_selection_sha256'] = sha256_file(args.budget_selection)
        gate = cost_gate(execution_count=plan['execution_count'], calibration=calibration,
                         mode=plan['mode'], price=price, budget_cny=args.budget_cny)
        if not gate['passed']:
            raise ValueError('formal cost gate failed: ' + '; '.join(gate['failures']))
        formal_projection = gate['projections'].get('formal_cost_cny')
        if plan['mode'] == 'api' and (
                formal_projection is None or formal_projection + prior_total > args.budget_cny):
            raise ValueError('formal cost gate failed: calibration/projection plus prior cost exceeds budget')
        gate['calibration_costs_cny'] = calibration_costs
        gate['prior_cost_cny'] = prior_cost
        gate['prior_total_cny'] = prior_total
        plan['calibration_costs_cny'] = calibration_costs
        plan['prior_cost_cny'] = prior_cost
        plan['prior_total_cny'] = prior_total
        plan['cost_gate'] = gate
    plan['price_quote'] = json.loads(args.prices.read_text()) if args.prices else None
    plan['budget_cny'] = args.budget_cny
    plan['preflight_only'] = args.preflight_only
    plan['fingerprint'] = fingerprint(plan)
    if args.output.exists():
        if not args.resume:
            raise FileExistsError('output exists; use --resume for identical configuration')
        previous = json.loads((args.output / 'launch.json').read_text())
        if previous != plan:
            raise ValueError('resume launch fingerprint/configuration differs')
        if (args.output / '_COMPLETE').exists():
            marker = json.loads((args.output / '_COMPLETE').read_text())
            if marker['summary_sha256'] != sha256_file(args.output / 'summary.json'):
                raise ValueError('completed suite summary changed')
            if marker.get('launch_fingerprint') != plan['fingerprint']:
                raise ValueError('completed suite launch fingerprint changed')
            for job in plan['jobs']:
                run = args.output / 'evaluation' / plan['model_key'] / job['task_id'] / 'runs' / (args.split + '-' + job['tier'])
                completed_samples(run, job)
            print('Already complete; artifact hashes verified, no model calls.')
            return 0
    else:
        args.output.mkdir(parents=True)
        write_json_atomic(args.output / 'launch.json', plan)
    env = dict(os.environ, DCI_DOCUMENT_TOKEN_BUDGET=str(args.token_budget),
               DCI_WORKSPACE_PREFETCH='1', DCI_WORKSPACE_PREFETCH_BATCH_SIZE=str(args.concurrency))
    # Inspect's local HTTP bridge and local model endpoints must bypass proxies.
    no_proxy = ','.join(filter(None, (env.get('NO_PROXY'), env.get('no_proxy'), 'localhost,127.0.0.1,::1')))
    env['NO_PROXY'] = env['no_proxy'] = no_proxy
    env.pop('DCI_PROTOCOL_PROFILE', None)
    ledger_path = args.output / 'ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'reservations': {}, 'attempts': []}
    samples = []
    started = time.monotonic()
    prior_elapsed = ledger.get('elapsed_seconds', 0)
    if ledger.get('active_session_started_at') is not None:
        # A hard kill cannot run finally. Charge the unclosed interval, including
        # any downtime, conservatively instead of resetting the 24-hour guard.
        prior_elapsed += max(0, time.time() - ledger['active_session_started_at'])
        ledger['elapsed_recovery_basis'] = 'conservative_unclosed_interval_including_downtime'
    ledger['elapsed_seconds'] = prior_elapsed
    ledger['active_session_started_at'] = time.time()
    write_json_atomic(ledger_path, ledger)
    (args.output / 'logs').mkdir(exist_ok=True)
    if args.execute:
        from scripts.inspect_local_runtime import prepare
        if not (args.output / 'inspect-tools').exists():
            prepare(args.output / 'inspect-tools')
    telemetry = None
    telemetry_handle = None
    if args.execute and plan['mode'] == 'local':
        import shutil
        if shutil.which('nvidia-smi'):
            telemetry_handle = (args.output / 'gpu.csv').open('a')
            telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total',
                                           '--format=csv,noheader,nounits', '-l', '1'], stdout=telemetry_handle, stderr=subprocess.DEVNULL)
    try:
        for job in plan['jobs']:
            if args.execute and source_fingerprint() != plan['source_fingerprint']:
                raise ValueError('source changed after launch; stop before dispatching another batch')
            name = job['task_id'] + '-' + job['tier']
            env['DCI_PROTOCOL_PROFILE'] = 'document-pool-v2-' + job['tier']
            if args.preflight_only:
                code = run_child(command_for(job, args, None, preflight=True), env=env,
                                 log=args.output / 'logs' / (name + '.log'), timeout=600)
                if code:
                    raise RuntimeError('CPU data preflight failed: ' + name)
                continue
            run = args.output / 'evaluation' / plan['model_key'] / job['task_id'] / 'runs' / (args.split + '-' + job['tier'])
            if not (run / '_COMPLETE').exists():
                reserved = price.reservation(len(job['query_ids']), args.token_budget) if price else 0
                all_reserved, first_reserved = _reservation_totals(ledger)
                batch_attempts = [attempt_row for attempt_row in ledger['attempts']
                                  if isinstance(attempt_row, dict) and attempt_row.get('batch') == name]
                is_first_attempt = not batch_attempts
                prior_total = plan.get('prior_total_cny', 0) if plan['mode'] == 'api' and args.split == 'test' else 0
                all_limit = args.budget_cny - prior_total
                if all_limit < 0 or all_reserved + reserved > all_limit:
                    raise ValueError('spending guard: cannot reserve the next batch within budget')
                attempt = len(ledger['attempts'])
                reservation_key = f'{name}:{attempt}'
                if is_first_attempt and plan['mode'] == 'api' and args.split == 'test':
                    first_limit = min(args.budget_cny * 0.8, all_limit)
                    if first_reserved + reserved > first_limit:
                        raise ValueError('spending guard: formal first-attempt allocation exceeds 80% ceiling')
                ledger['reservations'][reservation_key] = reserved
                ledger['attempts'].append({'batch': name, 'attempt': attempt, 'reservation_key': reservation_key,
                                           'reserved_cny': reserved, 'status': 'started'})
                write_json_atomic(ledger_path, ledger)
                remaining = 24 * 3600 - prior_elapsed - (time.monotonic() - started)
                if remaining <= 0:
                    raise ValueError('run wall-time limit reached')
                batch_started = time.monotonic()
                code = run_child(command_for(job, args, plan['model_key']), env=env,
                                 log=args.output / 'logs' / f'{name}-{attempt}.log', timeout=remaining)
                ledger['attempts'][-1].update(exit_code=code, elapsed_seconds=time.monotonic() - batch_started,
                                            tier=job['tier'], status='complete' if (run / '_COMPLETE').exists() else 'interrupted')
                if source_fingerprint() != plan['source_fingerprint']:
                    raise ValueError('source changed during batch; artifacts retained without suite completion')
                if not (run / '_COMPLETE').exists():
                    write_json_atomic(ledger_path, ledger)
                    raise RuntimeError('batch interrupted; preserve artifacts and resume: ' + name)
                batch_samples = completed_samples(run, job)
                if price:
                    costs = [price.cost(row['usage'])['cost_cny'] for row in batch_samples]
                    if any(cost is None for cost in costs):
                        ledger['attempts'][-1]['status'] = 'billing_usage_unavailable'
                        write_json_atomic(ledger_path, ledger)
                        raise ValueError('API usage missing; spending remains reserved, execution stopped')
                    # A resumed attempt may have unobserved spend; retain its reservation.
                    if is_first_attempt:
                        ledger['reservations'][reservation_key] = sum(costs)
                write_json_atomic(ledger_path, ledger)
                if sum(ledger['reservations'].values()) > all_limit:
                    raise ValueError('recorded usage exceeded spending limit; execution stopped')
                if plan['mode'] == 'api' and args.split == 'test':
                    _, first_spent = _reservation_totals(ledger)
                    if first_spent > min(args.budget_cny * .8, all_limit):
                        raise ValueError('recorded formal first-attempt usage exceeded allocation; execution stopped')
            restored = completed_samples(run, job)
            if price and any(price.cost(row['usage'])['cost_cny'] is None for row in restored):
                raise ValueError('API usage missing in completed batch; execution stopped')
            samples.extend(restored)
            print(json.dumps({'batch': name, 'completed': len(samples), 'planned': plan['execution_count']}), flush=True)
        elapsed = prior_elapsed + time.monotonic() - started
        if args.preflight_only:
            write_json_atomic(args.output / 'preflight.json', {'status': 'data_preflight_passed', 'jobs': len(plan['jobs']),
                                                               'model_calls': 0, 'sandbox_verified': False})
            return 0
        summary = summarize(samples, expected_tasks=plan['expected_tasks'], elapsed_seconds=elapsed, price=price)
        summary.update({key: plan[key] for key in ('split', 'selection_sha256', 'source_fingerprint', 'backend_sha256',
                                                  'concurrency', 'token_budget', 'complete_scope', 'model_key', 'mode')})
        for tier in TIERS:
            attempts = [row for row in ledger['attempts'] if row.get('tier') == tier]
            summary['tiers'][tier]['recorded_batch_seconds'] = sum(row.get('elapsed_seconds', 0) for row in attempts)
        summary['gpu_telemetry'] = {'path': 'gpu.csv', 'scope': 'physical_gpu_driver_metric; MIG may be unavailable'} if telemetry else None
        if telemetry:
            telemetry_handle.flush()
            utilization = []
            for line in (args.output / 'gpu.csv').read_text().splitlines():
                columns = line.split(',')
                if len(columns) == 5 and columns[2].strip().replace('.', '', 1).isdigit():
                    utilization.append(float(columns[2]))
            summary['gpu_utilization'] = (sum(utilization) / len(utilization)) if utilization else None
            summary['gpu_telemetry']['numeric_samples'] = len(utilization)
        summary['spending_guard'] = {'accounted_or_reserved_cny': sum(ledger['reservations'].values()),
                                    'prior_cost_cny': plan.get('prior_total_cny', 0),
                                    'model_total_accounted_or_reserved_cny':
                                        plan.get('prior_total_cny', 0) + sum(ledger['reservations'].values()),
                                    'budget_cny': args.budget_cny,
                                    'attempts': len(ledger['attempts']), 'ledger': 'ledger.json'}
        write_json_atomic(args.output / 'summary.json', summary)
        write_json_atomic(args.output / '_COMPLETE', {'summary_sha256': sha256_file(args.output / 'summary.json'),
                                                      'launch_fingerprint': plan['fingerprint']})
        return 0
    finally:
        ledger['elapsed_seconds'] = prior_elapsed + time.monotonic() - started
        ledger.pop('active_session_started_at', None)
        write_json_atomic(ledger_path, ledger)
        if telemetry:
            telemetry.terminate()
            telemetry.wait(timeout=10)
            telemetry_handle.close()


if __name__ == '__main__':
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f'[benchmark-v2] {error}', file=sys.stderr)
        raise SystemExit(2)
