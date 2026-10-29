#!/usr/bin/env python3
"""Freeze accepted selective document protocol and plan existing expansion data.

Only creates new release/plan directories. Never invokes a model or rewrites data.
Optional sandbox checks inspect file metadata and run the frozen CPU preflight.
"""
from __future__ import annotations

import argparse
import difflib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'releases/protocol-freezes/20261015-selective-full-v1'
RUNTIME = SOURCE / 'runtime'
SOURCE_RUNTIME = RUNTIME
FULL = ROOT / 'results/Qwen3.8-27B/full-selective-runs/20261015T071454Z-selective-full-1'
CONFIRM = ROOT / 'results/analysis/harness-confirmation-20261015T023746Z/analysis.json'
INDEX = Path('/mnt/afs2/202609/embedding_models/dci-bench/eval_data/dci_protocol_20261015_v2')
PYTHON = '/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python'
PROFILE = 'budgeted-selective'
CONTRACT = 'c6f4a9e46de4f0ebf9b21aca6f73c86a7b1dfd726783362bc3c476807be6419f'
SEED = '20261015-selective-expansion-v1'


def code_path(path):
    """Resolve archived code references without rewriting historical manifests."""
    path = Path(path)
    for group in ('protocol-freezes', 'expansion-runtimes', 'harness-confirmations',
                  'harness-continuations', 'harness-replays', 'size-budget-experiments'):
        old = ROOT / 'results' / group
        if path.is_relative_to(old):
            return ROOT / 'releases' / group / path.relative_to(old)
    if path.is_relative_to(ROOT / 'results') and not path.exists():
        relocated = ROOT / 'releases/result-code' / path.relative_to(ROOT / 'results')
        if relocated.exists():
            return relocated
    return path


def read(path):
    return json.loads(code_path(path).read_text())


def sha(path):
    return hashlib.sha256(code_path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def environment():
    env = dict(os.environ)
    env.update(DCI_PROTOCOL_PROFILE=PROFILE, DCI_EVALUATION_TRACK='document',
               DCI_PROTOCOL_CONTRACT_SHA256=CONTRACT, PYTHONPATH=str(RUNTIME),
               PYTHONDONTWRITEBYTECODE='1',
               PATH=str(Path(PYTHON).parent) + os.pathsep + env.get('PATH', ''),
               LD_LIBRARY_PATH=str(Path(PYTHON).parent.parent / 'lib') +
               (os.pathsep + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''))
    return env


def partition(task, ids, take=10):
    if len(ids) != len(set(ids)) or len(ids) < take:
        raise ValueError(f'Invalid or insufficient query IDs: {task}')
    key = lambda q: (hashlib.sha256(canonical([SEED, task, q]).encode()).hexdigest(), q)
    selected = set(sorted(ids, key=key)[:take])
    # Present each partition in source metadata order, independently of ID spelling.
    return [q for q in ids if q in selected], [q for q in ids if q not in selected]


def bounded_run(argv, cwd, timeout):
    start = time.monotonic()
    p = subprocess.Popen(argv, cwd=cwd, env=environment(), text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    limited = False
    try:
        output, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        limited = True
        os.killpg(p.pid, signal.SIGKILL)
        output, _ = p.communicate()
    return dict(argv=argv, cwd=str(cwd), elapsed_seconds=time.monotonic() - start,
                exit_code=p.returncode, preflight_time_cap_seconds=timeout,
                status='preflight_cap_exceeded' if limited else 'passed' if p.returncode == 0 else 'failed',
                output=output[-24000:])


def seal(directory):
    records = {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    write(directory / '_COMPLETE.json', {'schema_version': 'dci-selective-adoption-artifacts-v1',
                                        'artifact_sha256': records})


def verify_directory(directory):
    marker = read(directory / '_COMPLETE.json')
    for name, expected in marker['artifact_sha256'].items():
        if sha(directory / name) != expected:
            raise ValueError(f'Generated artifact changed: {directory / name}')


def validate_execution_runtime(runtime):
    """Reject runtime overrides that extend beyond the reviewed host adapter."""
    if runtime == SOURCE_RUNTIME:
        return
    adapter = read(runtime / 'adapter.json')
    allowed = {'dci_bench/results/protocol.py', 'dci_bench/results/phase4_runner.py',
               'dci_bench/tasks/mteb_llm_retrieval.py'}
    if adapter.get('adapter') != 'dci-query-artifact-paths-v1' or set(adapter['changes']) != allowed:
        raise ValueError('Execution runtime must declare exactly the reviewed host artifact adapter')
    pi = runtime / 'pi-dci'
    if not pi.is_symlink() or pi.resolve() != (SOURCE_RUNTIME / 'pi-dci').resolve():
        raise ValueError('Expansion must reuse the original immutable Pi runtime')
    expected = set()
    for sub in ['dci_bench', 'schemas']:
        expected.update(str(p.relative_to(SOURCE_RUNTIME)) for p in (SOURCE_RUNTIME / sub).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    expected.update(['scripts/run_task.py', 'scripts/run_expansion_task.py'])
    actual = {str(p.relative_to(runtime)) for sub in ['dci_bench', 'schemas', 'scripts']
              for p in (runtime / sub).rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != expected:
        raise ValueError('Expansion runtime source/schema closure differs from the original')
    patch = []
    for name in sorted(expected):
        before, after = sha(SOURCE_RUNTIME / name), sha(runtime / name)
        if name not in allowed and before != after:
            raise ValueError(f'Unreviewed execution runtime change: {name}')
        if name in allowed:
            if adapter['changes'][name] != {'before_sha256':before, 'after_sha256':after}:
                raise ValueError(f'Execution adapter fingerprint changed: {name}')
    # Original patch generation preserves this declared file order.
    for name in ['dci_bench/results/protocol.py', 'dci_bench/results/phase4_runner.py',
                 'dci_bench/tasks/mteb_llm_retrieval.py']:
        patch.extend(difflib.unified_diff((SOURCE_RUNTIME / name).read_text().splitlines(True),
                     (runtime / name).read_text().splitlines(True),
                     fromfile='original/' + name, tofile='expansion/' + name))
    if (runtime / 'host-artifact-paths.patch').read_text() != ''.join(patch):
        raise ValueError('Execution adapter patch does not match actual source changes')


def verify(output):
    global RUNTIME
    verify_directory(output)
    plan = read(output / 'plan.json')
    RUNTIME = code_path(plan['execution_runtime'])
    validate_execution_runtime(RUNTIME)
    release = code_path(plan['protocol_release'])
    verify_directory(release)
    for path, expected in read(release / 'release.json')['evidence_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Freeze evidence changed: {path}')
    for path, expected in plan['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Expansion input changed: {path}')
    splits = read(output / 'query-splits.json')
    totals = dict(smoke=0, pilot=0, remaining=0, full_new=0)
    for task, row in splits.items():
        ids = [q['_id'] for q in (json.loads(line) for line in
               (Path(row['metadata']) / 'queries.jsonl').read_text().splitlines())]
        pilot, rest = partition(task, ids)
        if row['pilot'] != pilot or row['remaining'] != rest or row['full_new'] != ids or row['smoke'] != pilot[:1]:
            raise ValueError(f'Query partition changed: {task}')
        if set(pilot) & set(rest) or set(pilot) | set(rest) != set(ids):
            raise ValueError(f'Invalid split closure: {task}')
        for stage in totals:
            totals[stage] += len(row[stage])
    if totals != plan['evaluation_counts']:
        raise ValueError('Evaluation count changed')
    result = bounded_run([PYTHON, '-B', '-c',
        'from pathlib import Path; import dci_bench; '
        f'assert Path(dci_bench.__file__).resolve().is_relative_to(Path({str(RUNTIME)!r})); '
        'from dci_bench.protocol.contracts import CONTRACT_SHA256; '
        f'assert CONTRACT_SHA256 == {CONTRACT!r}; print(CONTRACT_SHA256)'], RUNTIME, 30)
    if result['status'] != 'passed':
        raise ValueError(f'Frozen protocol import failed: {result}')
    sandbox_status = read(output / 'verification.json')['sandbox_status']
    return dict(status='verified' if sandbox_status == 'passed' else 'verified_with_preflight_pending',
                artifact_integrity='verified', evaluation_counts=totals, model_calls=0, gpu_runs=0,
                source_corpus_bytes_verified=False,
                sandbox_status=sandbox_status)


def prepare(output, release, *, sandboxes=False, timeout=60):
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing plan; use a new directory')
    reuse_release = release.exists()
    if reuse_release:
        verify_directory(release)
        if read(release / 'release.json')['base_contract_sha256'] != CONTRACT:
            raise ValueError('Existing protocol release does not match the accepted contract')
    validate_execution_runtime(RUNTIME)
    contract = read(SOURCE / 'selective-contract.json')
    if contract['sha256'] != CONTRACT or canonical(contract['contract']) != CONTRACT:
        raise ValueError('Accepted selective contract changed')
    confirmation = read(CONFIRM)
    if not confirmation['decision']['all_passed']:
        raise ValueError('Confirmation acceptance criteria did not pass')
    done = read(FULL / '_COMPLETE')
    if (done['status'], done['completed'], done['valid'], done['invalid']) != ('complete', 527, 527, 0):
        raise ValueError('Full run completion differs from accepted evidence')
    if done['bundle_manifest_sha256'] != sha(SOURCE / 'frozen.json') or done['completion_audit_sha256'] != sha(FULL / 'completion-audit.json'):
        raise ValueError('Full-run evidence chain changed')
    source_manifest = read(SOURCE / 'frozen.json')
    if sha(SOURCE / 'frozen.json') != (SOURCE / 'frozen.sha256').read_text().strip():
        raise ValueError('Frozen source manifest changed')
    for name, expected in source_manifest['files'].items():
        if sha(SOURCE / name) != expected:
            raise ValueError(f'Frozen source attachment changed: {name}')
    source_check = bounded_run([PYTHON, '-B', '-c',
        'import sys; from pathlib import Path; '
        f'sys.path.insert(0, {str(SOURCE_RUNTIME / "scripts")!r}); '
        'from prepare_harness_full import verify_bundle; '
        f'verify_bundle(Path({str(SOURCE)!r})); print("frozen runtime inventory verified")'], SOURCE_RUNTIME, 120)
    if source_check['status'] != 'passed':
        raise ValueError(f'Frozen runtime inventory verification failed: {source_check}')
    # Existing verifier checks all index/source metadata, not full corpus contents.
    check = bounded_run([PYTHON, '-B', str(ROOT / 'scripts/prepare_protocol_data.py'),
                         '--output', str(INDEX), '--verify-only'], ROOT, 120)
    if check['status'] != 'passed':
        raise ValueError(f'Protocol data verification failed: {check}')
    old = read(SOURCE / 'evaluation-plan.json')['task_counts']
    documents = read(INDEX / 'document_tasks.json')
    new = sorted([r for r in documents if r['task'] not in old], key=lambda r: (r['num_docs'], r['task']))
    if len(documents) != 13 or len(new) != 7 or sum(r['num_queries'] for r in new) != 1872:
        raise ValueError('Expanded document scope changed')
    input_hashes = {}
    def anchor(path):
        input_hashes[str(path)] = sha(path)
    for name in ['_COMPLETE.json', 'manifest.json', 'document_tasks.json', 'commands.json']:
        anchor(INDEX / name)
    for name in ['scripts/prepare_protocol_data.py', 'scripts/prepare_expansion_data.py']:
        anchor(ROOT / name)
    for p in (ROOT / 'dci_bench').rglob('*'):
        if p.is_file() and p.suffix in {'.py', '.json', '.conf'}:
            anchor(p)
    anchor(INDEX.with_name(INDEX.name + '_acceptance.json'))
    for row in documents:
        md = Path(row['metadata'])
        anchor(md / '_COMPLETE.json')
        marker = read(md / '_COMPLETE.json')
        for filename in marker['metadata_hashes']:
            anchor(md / filename)
        anchor(Path(row['suite']) / 'catalog.json')
        anchor(Path(row['suite']) / '_COMPLETE.json')
    # Reuse the verified source runtime; capture required Python and Pi sources.
    runtime_hashes = {}
    for sub in ['dci_bench', 'pi-dci/scripts', 'pi-dci/packages/ai/src',
                'pi-dci/packages/agent/src', 'pi-dci/packages/coding-agent/src']:
        for p in sorted((RUNTIME / sub).rglob('*')):
            if p.is_file() and p.suffix in {'.py', '.json', '.conf', '.ts', '.js'}:
                runtime_hashes[str(p)] = sha(p)
    for name in ['scripts/run_expansion_task.py', 'scripts/run_task.py', 'pi-dci/package.json',
                 'pi-dci/package-lock.json', 'pi-dci/node_modules/tsx/dist/cli.mjs']:
        p = RUNTIME / name
        runtime_hashes[str(p)] = sha(p)
    input_hashes.update(runtime_hashes)
    if RUNTIME != SOURCE_RUNTIME:
        for name in ['adapter.json', 'host-artifact-paths.patch', 'snapshot.json',
                     'snapshot.sha256', 'verify_snapshot.py']:
            anchor(RUNTIME / name)
        for p in (RUNTIME / 'schemas').rglob('*'):
            if p.is_file():
                anchor(p)
    splits, instructions, commands = {}, {}, {}
    for row in new:
        task = row['task']
        queries = [json.loads(s) for s in (Path(row['metadata']) / 'queries.jsonl').read_text().splitlines()]
        ids = [q['_id'] for q in queries]
        pilot, rest = partition(task, ids)
        splits[task] = dict(metadata=row['metadata'], smoke=pilot[:1], pilot=pilot,
                            remaining=rest, full_new=ids,
                            partition_sha256={k: canonical(v) for k, v in
                                              dict(smoke=pilot[:1], pilot=pilot, remaining=rest, full_new=ids).items()})
        instructions[task] = dict(instruction=row['task_instruction'],
                                   instruction_sha256=hashlib.sha256(row['task_instruction'].strip().encode()).hexdigest(),
                                   task_context_contract_sha256=canonical({**contract['contract'], 'task_context':
                                       {'task_name':task, 'instruction':row['task_instruction'].strip()}}))
        for stage in ['smoke', 'pilot', 'remaining', 'full_new']:
            ids_stage = splits[task][stage]
            argv = [PYTHON, '-B', str(RUNTIME / 'scripts/run_expansion_task.py'),
                    '--suite', row['suite'], '--task', task, '--query-ids', *ids_stage,
                    '--model-key', 'Qwen3.8-27B', '--backend-manifest', '<ACTUAL_BACKEND_MANIFEST>',
                    '--results-root', '<NEW_RESULTS_ROOT>', '--run-id', f'selective-expansion-v1-{stage}',
                    '--max-concurrency', '1' if stage == 'smoke' else '4',
                    '--bridge-port', '28131', '--metric-ks', '1', '3', '5', '10', '20']
            commands[f'{stage}/{task}'] = dict(cwd=str(RUNTIME), environment={
                'DCI_PROTOCOL_PROFILE':PROFILE, 'DCI_EVALUATION_TRACK':'document',
                'DCI_PROTOCOL_CONTRACT_SHA256':CONTRACT, 'PYTHONPATH':str(RUNTIME),
                'PYTHONDONTWRITEBYTECODE':'1', 'PATH_prefix':str(Path(PYTHON).parent),
                'LD_LIBRARY_PATH_prefix':str(Path(PYTHON).parent.parent / 'lib')}, argv=argv)
    output.mkdir(parents=True)
    if not reuse_release:
        release.mkdir(parents=True)
    evidence_paths = [SOURCE / 'frozen.json', SOURCE / 'frozen.sha256', SOURCE / 'selective-contract.json',
                      SOURCE / 'evaluation-plan.json', CONFIRM, FULL / '_COMPLETE',
                      FULL / 'completion-audit.json', FULL / 'evaluation/summary.json',
                      ROOT / 'results/analysis/selective-full-20261015T071454Z/metrics.json',
                      ROOT / 'results/analysis/selective-full-20261015T071454Z/delegated-audits.json']
    release_record = dict(schema_version='dci-selective-document-adoption-v1',
          release_name='selective-document-v1', adopted_at=datetime.now(timezone.utc).isoformat(),
          profile=PROFILE, base_contract_sha256=CONTRACT, frozen_runtime=str(RUNTIME),
          adoption='Accepted document protocol for subsequent benchmark evaluations; underlying contract unchanged.',
          evidence_sha256={str(p):sha(p) for p in evidence_paths},
          frozen_boundaries=['system/query/final/repair prompts', 'stable-rg search policy',
              'token/output/step/time budgets and budget controls', 'read evidence and exact corpus ID rules',
              'tool and sandbox policy', 'generation settings', 'scoring and metric aggregation'],
          task_context_policy='Record exact source task instruction separately; never tune on evaluated outcomes. New tasks currently have empty instruction.',
          benchmark_model='Qwen3.8-27B BF16; model identity is a separate backend manifest, not part of task-neutral protocol.',
          change_policy='Any behavioral change requires a new protocol version and separate results; preserve v1.',
          limitations=['Not a claim of optimality or unseen six-task generalization.',
              'Read gate requires at least one nonempty read, not a receipt per returned document.',
              'Document release does not approve selective prompts/schema for code.'])
    if not reuse_release:
        write(release / 'release.json', release_record)
        write(release / 'contract.json', contract)
        write(release / 'runtime-source-hashes.json', runtime_hashes)
        seal(release)
    counts = {stage:sum(len(row[stage]) for row in splits.values()) for stage in ['smoke', 'pilot', 'remaining', 'full_new']}
    plan = dict(schema_version='dci-selective-expansion-plan-v1', protocol_release=str(release),
          base_contract_sha256=CONTRACT, dataset_index=str(INDEX), input_sha256=input_hashes,
          execution_runtime=str(RUNTIME), original_runtime=str(SOURCE_RUNTIME),
          host_artifact_adapter='dci-query-artifact-paths-v1' if RUNTIME != SOURCE_RUNTIME else 'original',
          adapter_boundary='Only result directory encoding; preserve original query IDs, labels, model-visible inputs, scoring and selective base contract.',
          document_total_tasks=13, document_total_queries=2399, original_reference_queries=527,
          new_document_tasks=7, evaluation_counts=counts, selection_seed=SEED,
          selection_method='Rank SHA256(canonical-json-SHA256([seed, task, original_query_id])); take ten; emit in source metadata order. No qrels/scores used.',
          splits_not_additive='Smoke is inside pilot. full_new includes pilot and remaining. Prior 527 are a separate original-suite reference.',
          remainder_interpretation='Non-pilot portion, not asserted to be an independently unseen holdout; exact/semantic overlap and historical model exposure may exist.',
          candidate_policy='Every sample retains its entire frozen task corpus. Do not construct gold-only or reduced-candidate workspaces.',
          task_order=[r['task'] for r in new], task_inventory=new,
          primary_report='All-sample equal-task macro F1@10; invalid samples score zero. Report original six, new seven, and combined thirteen separately.',
          secondary_report=['Recall@10', 'actual-window Precision@10', 'nDCG@10', 'valid_only',
              'validity/failure kinds', 'tokens/latency/startup cost', 'repairs/read/finalization traces',
              'family-level results; WixQA tasks share corpus; task average is not family average'],
          gates=['CPU source/data/contract/sandbox preflight', '7-query endpoint/tool smoke before pilot',
              '70-query pilot: check failures, repair behavior and token/startup cost before full evaluation',
              'Then explicitly launch 1802 non-pilot queries, or a separately labeled 1872-query full rerun'],
          no_quality_gate='Do not drop difficult tasks or queries based on model scores, or tune v1 on pilot labels. Infrastructure fixes require recorded provenance; protocol changes require v2.',
          code_track='Separate code protocol/scorer; ARB 287 and nested 45-query untruncated-positive subset are outside this plan.',
          endpoint_commands='Templates require a fresh actual backend manifest and a new result root. They do not start a model server or establish endpoint acceptance.')
    write(output / 'plan.json', plan)
    write(output / 'query-splits.json', splits)
    write(output / 'task-contexts.json', instructions)
    write(output / 'commands.json', commands)
    (output / 'build_source.py').write_bytes(Path(__file__).read_bytes())
    host_script = '''import json
from pathlib import Path
from dci_bench.data.registry import TASKS, TaskSpec
from dci_bench.results.phase4_runner import data_audit
from dci_bench.tasks.mteb_llm_retrieval import _make_samples
plan=json.loads(Path(PLAN_PATH).read_text())
splits=json.loads(Path(SPLIT_PATH).read_text())
count=0
for row in plan['task_inventory']:
    task=row['task']
    TASKS[task]=TaskSpec(task,row['dataset'],row['revision'],task,row['task_instruction'])
    data_audit(Path(row['metadata']).parent,task)
    samples=_make_samples(task,splits[task]['pilot'],Path(row['metadata']).parent,Path(row['workspace']).parent)
    assert [s.id for s in samples]==splits[task]['pilot']
    assert all(not s.target and set(s.metadata)=={'task','query_id','query_index','workspace'} for s in samples)
    count+=len(samples)
assert count==70
print(json.dumps({'host_pilot_samples':count,'labels_host_only':True,'model_calls':0}))
'''.replace('PLAN_PATH', repr(str(output / 'plan.json'))).replace('SPLIT_PATH', repr(str(output / 'query-splits.json')))
    host_check = bounded_run([PYTHON, '-B', '-c', host_script], RUNTIME, 120)
    if host_check['status'] != 'passed':
        raise ValueError(f'Frozen host sample validation failed: {host_check}')
    cpu = {'frozen_runtime_inventory_check':source_check, 'data_index_check':check,
           'frozen_host_sample_check':host_check, 'model_calls':0, 'gpu_runs':0,
           'source_corpus_bytes_verified':False, 'sandboxes':{}}
    if sandboxes:
        for row in new:
            print(f"[preflight] {row['task']} ({row['num_docs']} files), cap={timeout}s", flush=True)
            argv = [str(Path(PYTHON).with_name('node')), str(RUNTIME / 'pi-dci/node_modules/tsx/dist/cli.mjs'),
                    'scripts/dci-run.ts', '--preflight-only', '--workspace', row['workspace']]
            cpu['sandboxes'][row['task']] = bounded_run(argv, RUNTIME / 'pi-dci', timeout)
            print(json.dumps({'task':row['task'], **{k:cpu['sandboxes'][row['task']][k]
                  for k in ['status','elapsed_seconds','exit_code']}}), flush=True)
    cpu['sandbox_status'] = ('passed' if len(cpu['sandboxes']) == 7 and
           all(x['status']=='passed' for x in cpu['sandboxes'].values()) else 'not_fully_verified')
    write(output / 'verification.json', cpu)
    seal(output)
    return verify(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--release-output', type=Path)
    p.add_argument('--execution-runtime', type=Path,
                   help='Separate source snapshot for host-only compatibility fixes; never edits the original runtime')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--check-sandboxes', action='store_true')
    p.add_argument('--sandbox-timeout', type=int, default=60)
    args = p.parse_args()
    global RUNTIME
    if args.execution_runtime:
        RUNTIME = args.execution_runtime.resolve(strict=True)
    if args.verify_only:
        result = verify(args.output.resolve())
    else:
        if not args.release_output or args.sandbox_timeout < 1:
            p.error('--release-output and a positive sandbox timeout are required for preparation')
        result = prepare(args.output.resolve(), args.release_output.resolve(),
                         sandboxes=args.check_sandboxes, timeout=args.sandbox_timeout)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
