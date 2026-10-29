#!/usr/bin/env python3
"""Prepare and run a paired size-budget diagnostic from the immutable pilot.

The reviewed pilot startup is copied and adapted with checked, exact replacements.
The generated diff, all input hashes, and three independent budget runtimes are
sealed in a new bundle. No frozen source or existing result is edited.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import run_selective_expansion_pilot as pilot
import size_budget_experiment as support

E = pilot.expansion
ROOT = pilot.ROOT
BUNDLES = ROOT / 'releases/size-budget-experiments'
RUNS = ROOT / 'results/Qwen3.8-27B/size-budget-runs'
ACCEPTED_DRIVER_SHA256 = '8fad86f64b7a60735319f9ecb62c6ebc5230d66dde8666f913b837315db3af9f'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Reviewed source boundary changed: ' + old[:100])
    return text.replace(old, new, 1)


def make_runtime(source, target, size):
    target.mkdir()
    for sub in ['dci_bench', 'schemas', 'scripts']:
        shutil.copytree(source / sub, target / sub, copy_function=shutil.copyfile,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    pi = target / 'pi-dci'
    pi.mkdir()
    frozen_pi = (source / 'pi-dci').resolve()
    # Resolve package workspace symlinks into the immutable source, never the
    # live checkout. Pi itself stays byte-identical; only contracts differ.
    for name in ['node_modules', 'packages']:
        (pi / name).symlink_to(frozen_pi / name, target_is_directory=True)
    for name in ['package.json', 'package-lock.json', 'tsconfig.json', 'tsconfig.base.json']:
        shutil.copyfile(frozen_pi / name, pi / name)
    (pi / 'scripts').mkdir()
    before = (frozen_pi / 'scripts/dci-run.ts').read_text()
    # v1 removed the forecast and could spend the final request's budget while
    # still searching. Restore the accepted forecast, with the tier reserve as
    # a floor. Never patch an already sealed v1 runtime in place.
    after = before
    (pi / 'scripts/dci-run.ts').write_text(after)
    protocol = target / 'dci_bench/protocol'
    contract = E.read(protocol / 'contract.json')
    contract['max_total_model_tokens'] = size['total']
    contract['context_policy']['max_total_model_tokens'] = size['total']
    (protocol / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    profiles = E.read(protocol / 'pilot_profiles.json')
    profile = 'budgeted-selective-size-v2-' + size['name']
    profiles[profile] = deepcopy(profiles['budgeted-selective'])
    overrides = profiles[profile]['budget_control_overrides']
    overrides.update(version='selective-dynamic-reserve-size-v2-' + size['name'],
                     reserve_tokens=size['reserve'],
                     initial_prompt=profiles['_budget_policy']['policy']['initial_prompt'].replace(
                         '128000', str(size['total'])))
    (protocol / 'pilot_profiles.json').write_text(json.dumps(profiles, indent=2) + '\n')
    env = dict(E.environment(), PYTHONPATH=str(target), DCI_PROTOCOL_PROFILE=profile)
    env.pop('DCI_PROTOCOL_CONTRACT_SHA256', None)
    resolved = json.loads(subprocess.check_output([E.PYTHON, '-B', '-c',
        'import json; from dci_bench.protocol.contracts import protocol_contract; '
        'print(json.dumps(protocol_contract()))'], cwd=target, env=env, text=True))
    baseline = E.read(E.SOURCE / 'selective-contract.json')['contract']
    normalized = deepcopy(resolved)
    for key in ['max_total_model_tokens', 'protocol_profile', 'context_policy']:
        normalized[key] = baseline[key]
    for key in ['version', 'reserve_tokens', 'initial_prompt']:
        normalized['budget_control'][key] = baseline['budget_control'][key]
    if normalized != baseline:
        raise ValueError('Budget arm changed non-budget selective behavior')
    E.write(target / 'resolved-contract.json', resolved)
    (target / 'budget-runtime.patch').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='baseline/dci-run.ts', tofile='tiered/dci-run.ts')))
    return dict(runtime=str(target), profile=profile, contract_sha256=E.canonical(resolved),
                reserve_mode=support.RESERVE_MODE, **size)


def adapt_driver(source, bundle):
    if E.hashlib.sha256(source.encode()).hexdigest() != ACCEPTED_DRIVER_SHA256:
        raise ValueError('Pilot driver differs from the accepted 70-query run')
    text = replace_once(source, "ROOT = Path(__file__).resolve().parents[1]", f'ROOT = Path({str(ROOT)!r})')
    text = replace_once(text, "RESULTS = ROOT / 'results/Qwen3.8-27B/expansion-pilot-runs'", f'RESULTS = Path({str(RUNS)!r})')
    setup = f'''
import size_budget_experiment as experiment
EXPERIMENT_BUNDLE = Path({str(bundle)!r})
_original_summarize = summarize
_original_backend_check = check_backend_identity
def check_backend_identity(confirmation, expected, actual, concurrency=None):
    if actual['backend']['context_length'] != experiment.CONTEXT:
        raise ValueError('Backend context differs from the paired experiment')
    comparison = json.loads(json.dumps(actual))
    comparison['backend']['context_length'] = expected['backend']['context_length']
    _original_backend_check(confirmation, expected, comparison, concurrency)
def summarize(records, task_order):
    return experiment.summarize(records, task_order, _original_summarize)
def batches(plan, splits, concurrency=8):
    return experiment.batches(plan, splits, concurrency, EXPERIMENT_BUNDLE)
'''
    text = replace_once(text, '\ndef main():', setup + '\ndef main():')
    text = replace_once(text, '    args = parser.parse_args()',
                        '    args = parser.parse_args()\n    expansion.verify_directory(EXPERIMENT_BUNDLE)\n    experiment.verify_bindings(EXPERIMENT_BUNDLE)')
    text = replace_once(text, "MAX_MODEL_LEN='65536'", "MAX_MODEL_LEN='262144'")
    if text.count('context_length=65536') != 2:
        raise ValueError('Unexpected backend context declarations')
    text = text.replace('context_length=65536', 'context_length=262144')
    text = replace_once(text, 'context_window=65536', 'context_window=262144')
    text = replace_once(text, "'--context-length', '65536'", "'--context-length', '262144'")
    text = replace_once(text, 'bridge + len(work) * args.max_concurrency - 1',
                        "work[-1]['bridge_port'] + work[-1]['bridge_port_count'] - 1")
    text = replace_once(text, 'cumulative_token_budget=128000, sample_timeout_seconds=600, stages=work,',
        "budget_tiers=experiment.TIERS, budget_policy=experiment.RESERVE_MODE, "
        "sample_timeout_seconds=600, stages=work,")
    text = replace_once(text, "unique_queries=sum(len(b['query_ids']) for b in work), smoke_queries=7,",
                        'unique_queries=14, executions=28, smoke_queries=0,')
    text = replace_once(text, "query_ids={t: splits[t]['pilot'] for t in plan['task_order']}",
                        "query_ids={t: experiment.choose(t, splits[t]['pilot']) for t in plan['task_order']}")
    text = replace_once(text, "selection_seed=plan['selection_seed']", 'selection_seed=experiment.SEED')
    text = replace_once(text, "'parallel_smoke_tasks': min(7, args.max_concurrency)",
                        "'parallel_task_processes': max(1, args.max_concurrency // min(2, args.max_concurrency))")
    text = replace_once(text, "'Separate from the 70 benchmark queries and their usage totals'",
                        "'Separate from 28 paired diagnostic executions; no new held-out claims'")
    text = replace_once(text, "expansion.write(output / 'launch-plan.json', launch_plan)",
        "launch_plan.update(experiment_bundle=str(EXPERIMENT_BUNDLE), "
        "bundle_sha256=expansion.sha(EXPERIMENT_BUNDLE / '_COMPLETE.json'))\n    "
        "expansion.write(output / 'launch-plan.json', launch_plan)")
    text = replace_once(text, "            if args.inspect_preflight_only:",
        "            for label, info in expansion.read(EXPERIMENT_BUNDLE / 'runtimes.json').items():\n"
        "                if label == 'baseline':\n                    continue\n"
        "                step('budget-probe-' + label, [expansion.PYTHON, '-B', str(EXPERIMENT_BUNDLE / 'size_budget_experiment.py'),\n"
        "                     '--cpu-probe', info['runtime'], str(output / 'budget-probes' / label)],\n"
        "                     output=output, cwd=Path(info['runtime']), env=experiment.arm_environment(env, info), timeout=300)\n"
        "                step('inspect-' + label, [expansion.PYTHON, '-B', str(EXPERIMENT_BUNDLE / 'check_inspect_parallel.py'),\n"
        "                     '--runtime', info['runtime'], '--tools-directory', str(tools_directory),\n"
        "                     '--output', str(output / ('inspect-' + label)), '--max-concurrency', '2'],\n"
        "                     output=output, cwd=Path(info['runtime']), env=experiment.arm_environment(env, info), timeout=300)\n"
        "            if args.inspect_preflight_only:")
    text = replace_once(text, "        step('backend',", "        experiment.check_context(models)\n        step('backend',")
    text = text.replace("ROOT / 'scripts/check_inspect_parallel.py'", "EXPERIMENT_BUNDLE / 'check_inspect_parallel.py'")
    # Run arm-specific Inspect, solver, verification imports in separate processes.
    start = text.index('        def run_batch(batch):')
    end = text.index('        summary = summarize(records,', start)
    block = '''        def run_batch(batch):
            selected_runtime = Path(batch['runtime'])
            selected_env = experiment.arm_environment(env, batch)
            step(batch['stage'] + '-' + batch['task'],
                 command(batch, inventory[batch['task']], selected_runtime, output),
                 output=output, cwd=selected_runtime, env=selected_env,
                 timeout=1920, accepted=(0, 1))
            return experiment.verify_batch_in_child(batch, inventory[batch['task']], selected_runtime,
                output, selected_env, step, EXPERIMENT_BUNDLE / 'size_budget_experiment.py')
        experiment.execute_batches(work, run_batch, args.max_concurrency, records)
        if len(records) != 28 or len({(r['sample']['run_id'], r['sample']['task'], r['sample']['query_id']) for r in records}) != 28:
            raise ValueError('Paired execution set differs from expected 14 x 2')
'''
    text = text[:start] + block + text[end:]
    if text.count("summary['valid'] == 70") != 2:
        raise ValueError('Unexpected completion condition')
    text = text.replace("summary['valid'] == 70", "summary['valid'] == 28")
    text = text.replace("summary[\"valid\"] == 70", "summary[\"valid\"] == 28")
    text = text.replace('completed=70', 'completed=28').replace("invalid=70 - summary['valid']", "invalid=28 - summary['valid']")
    text = text.replace('valid={summary["valid"]}/70', 'valid={summary["valid"]}/28')
    text = replace_once(text, "        status = 'complete' if summary['valid'] == 28 else 'complete_with_sample_failures'",
        "        for failure in summary['failed_samples']:\n"
        "            print('[size-budget] sample failure: ' + json.dumps(failure, ensure_ascii=False), flush=True)\n"
        "        status = 'complete' if summary['valid'] == 28 else 'complete_with_sample_failures'")
    compile(text, str(bundle / 'driver.py'), 'exec')
    return text


def prepare(bundle):
    if {'results', 'eval_data', 'raw_data', 'eval_data_code'} & set(bundle.resolve().parts):
        raise ValueError('Executable bundles belong under the code repository releases/ directory')
    E.verify(pilot.PLAN)
    if E.sha(ROOT / 'scripts/run_selective_expansion_pilot.py') != ACCEPTED_DRIVER_SHA256:
        raise ValueError('Accepted pilot driver changed')
    if bundle.exists():
        raise FileExistsError('Refusing to overwrite experiment bundle')
    source = E.code_path(E.read(pilot.PLAN / 'plan.json')['execution_runtime'])
    bundle.mkdir(parents=True)
    (bundle / 'runtimes').mkdir()
    runtimes = {'baseline': dict(runtime=str(source), profile=E.PROFILE,
                                 contract_sha256=E.CONTRACT, total=128000, reserve=32000,
                                 reserve_mode='original dynamic forecast')}
    for _, label, total, reserve in support.TIERS:
        runtimes[label] = make_runtime(source, bundle / 'runtimes' / label,
                                      dict(name=label, total=total, reserve=reserve))
    E.write(bundle / 'runtimes.json', runtimes)
    E.write(bundle / 'bindings.json', dict(baseline_runtime=str(source.resolve()),
        frozen_pi=str((source / 'pi-dci').resolve()),
        anchors={str(p): E.sha(p) for p in [pilot.PLAN / '_COMPLETE.json',
            E.SOURCE / 'frozen.json']}))
    names = ['size_budget_experiment.py', 'run_selective_expansion_pilot.py',
             'prepare_selective_expansion.py', 'stage_expansion_corpora.py', 'inspect_local_runtime.py',
             'check_inspect_parallel.py', 'run_size_budget_experiment.py']
    for name in names:
        shutil.copyfile(ROOT / 'scripts' / name, bundle / name)
    checker = bundle / 'check_inspect_parallel.py'
    checker.write_text(replace_once(checker.read_text(), 'model_context_window=65536',
                                    'model_context_window=262144'))
    # These helpers are relocated, unlike the original repo-local launcher.
    for name in ['prepare_selective_expansion.py', 'run_selective_expansion_pilot.py']:
        path = bundle / name
        path.write_text(replace_once(path.read_text(), 'ROOT = Path(__file__).resolve().parents[1]',
                                    f'ROOT = Path({str(ROOT)!r})'))
    source_text = (ROOT / 'scripts/run_selective_expansion_pilot.py').read_text()
    derived = adapt_driver(source_text, bundle)
    (bundle / 'driver.py').write_text(derived)
    (bundle / 'driver.patch').write_text(''.join(difflib.unified_diff(source_text.splitlines(True),
        derived.splitlines(True), fromfile='pilot/driver.py', tofile='size-budget/driver.py')))
    E.write(bundle / 'experiment.json', dict(version=support.VERSION,
        reserve_mode=support.RESERVE_MODE, tiers=support.TIERS, seed=support.SEED,
        paired_queries=14, executions=28, baseline_contract=E.CONTRACT, fixed_context=support.CONTEXT,
        unchanged=['original Pi source and dynamic reserve forecast',
                   'selective prompts except total-budget numeral', 'read and output rules', 'scoring',
                   'generation', 'search tools and isolation', '600 second timeout',
                   'request-level and actual cumulative usage guards'],
        shared_backend_change=dict(previous_context=65536, experiment_context=262144),
        external_runtime_verification='Every launch verifies the original expansion plan and full frozen runtime '
            'inventory (including package dependencies) before probes or GPU. Tier packages/node_modules link to that verified snapshot.',
        limitation='Developmental paired diagnostic on previously evaluated pilot queries. '
                   'Both total budget and early-finalization policy change; no individual causal attribution.',
        source_files={str(ROOT / 'scripts' / name): E.sha(ROOT / 'scripts' / name) for name in names}))
    E.seal(bundle)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, help='Previously prepared, sealed experiment bundle')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'-{os.getpid()}')
    args, remaining = parser.parse_known_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', args.run_id):
        parser.error('Unsafe run ID')
    bundle = args.bundle.resolve() if args.bundle else BUNDLES / args.run_id
    if args.bundle:
        E.verify_directory(bundle)
    else:
        prepare(bundle)
    if args.prepare_only:
        print(bundle)
        return 0
    return subprocess.call([E.PYTHON, '-B', str(bundle / 'driver.py'), '--run-id', args.run_id, *remaining])


if __name__ == '__main__':
    raise SystemExit(main())
