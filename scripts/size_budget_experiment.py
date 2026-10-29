"""Historical full-corpus diagnostic; never changes the accepted selective runtime.

For the new 200/500/1000 per-query pools use dci_bench.protocol.document_pools
and scripts/run_document_pool.py. These historical tiers intentionally retain
their original meaning so sealed paired experiments remain interpretable.

The experiment uses fresh processes per contract and pairs the same two pilot
queries per task. A tier floor plus dynamic forecast controls finalization; the original
request, context, actual-usage, read, repetition and deadline guards remain.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

TIERS = ((1000, 'small', 128000, 32000),
         (100000, 'medium', 256000, 48000),
         (None, 'large', 512000, 64000))
SEED = 'selective-size-budget-diagnostic-v1-20261021'
CONTEXT = 262144
VERSION = 'dci-size-budget-diagnostic-v2'
RESERVE_MODE = 'original dynamic forecast with tier reserve floor; original hard guards'


def verify_bindings(bundle):
    bindings = json.loads((bundle / 'bindings.json').read_text())
    runtimes = json.loads((bundle / 'runtimes.json').read_text())
    if Path(runtimes['baseline']['runtime']).resolve() != Path(bindings['baseline_runtime']):
        raise ValueError('Baseline runtime binding changed')
    for _, name, _, _ in TIERS:
        root = Path(runtimes[name]['runtime'])
        if root.resolve() != (bundle / 'runtimes' / name).resolve():
            raise ValueError('Tier runtime binding changed')
        for sub in ['packages', 'node_modules']:
            if (root / 'pi-dci' / sub).resolve() != (Path(bindings['frozen_pi']) / sub).resolve():
                raise ValueError('Frozen Pi dependency binding changed')
    for path, expected in bindings['anchors'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Original freeze anchor changed: ' + path)


def tier(num_docs):
    if type(num_docs) is not int or num_docs <= 0:
        raise ValueError('Corpus size must be a positive integer')
    return next(dict(name=name, total=total, reserve=reserve)
                for upper, name, total, reserve in TIERS if upper is None or num_docs <= upper)


def choose(task, ids):
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError('Expected the original ten-query pilot per task')
    key = lambda q: (hashlib.sha256(json.dumps([SEED, task, q], ensure_ascii=False).encode()).hexdigest(), q)
    selected = set(sorted(ids, key=key)[:2])
    return [q for q in ids if q in selected]


def batches(plan, splits, concurrency, bundle):
    runtime_info = json.loads((bundle / 'runtimes.json').read_text())
    inventory = {r['task']: r for r in plan['task_inventory']}
    work = []
    for i, task in enumerate(plan['task_order']):
        size = tier(inventory[task]['num_docs'])
        for arm in (['baseline', 'tiered'] if i % 2 == 0 else ['tiered', 'baseline']):
            info = runtime_info['baseline' if arm == 'baseline' else size['name']]
            work.append(dict(stage=arm, task=task, query_ids=choose(task, splits[task]['pilot']),
                             max_concurrency=min(2, concurrency), tier=size['name'], **info))
    return work


def check_context(models, expected=CONTEXT):
    selected = [m for m in models.get('data', []) if m.get('id') == 'Qwen3.8-27B']
    if len(selected) != 1 or selected[0].get('max_model_len') != expected:
        raise ValueError(f'Live vLLM max_model_len must explicitly equal {expected}')


def arm_environment(env, batch):
    return dict(env, PYTHONPATH=batch['runtime'], DCI_PROTOCOL_PROFILE=batch['profile'],
                DCI_PROTOCOL_CONTRACT_SHA256=batch['contract_sha256'])


def execute_batches(work, run_batch, concurrency, records=None):
    # Counterbalance arm order across tasks. At most floor(cap/batch_size)
    # task processes run together, so two samples per task never exceed cap.
    if records is None:
        records = []
    width = max(b['max_concurrency'] for b in work)
    for wave in (work[::2], work[1::2]):
        with ThreadPoolExecutor(max_workers=max(1, concurrency // width)) as pool:
            for result in pool.map(run_batch, wave):
                records.extend(result)
    return records


def summarize(records, tasks, original):
    arms = {arm: original([r for r in records if r['sample']['run_id'] == arm], tasks)
            for arm in ['baseline', 'tiered']}
    keyed = {(r['sample']['run_id'], r['sample']['task'], r['sample']['query_id']): r
             for r in records}
    pairs = []
    for arm, task, qid in sorted(keyed):
        if arm != 'baseline' or ('tiered', task, qid) not in keyed:
            continue
        ra, rb = keyed[arm, task, qid], keyed['tiered', task, qid]
        a, b = ra['sample'], rb['sample']
        if a['query_sha256'] != b['query_sha256'] or ra['data_identity'] != rb['data_identity']:
            raise ValueError('Paired samples differ in query/corpus/labels')
        pairs.append(dict(task=task, query_id=qid,
            tier=rb['experiment']['tier'], budgets={arm:r['experiment'] for arm,r in [('baseline',ra),('tiered',rb)]},
            baseline_valid=a['valid_output'], tiered_valid=b['valid_output'],
            deltas={k: (b['metrics'][k] if b['valid_output'] else 0) -
                         (a['metrics'][k] if a['valid_output'] else 0)
                    for k in ['recall_at_10', 'f1_at_10', 'ndcg_at_10']},
            tokens={arm: s['usage'].get('total_tokens') for arm, s in [('baseline', a), ('tiered', b)]},
            seconds={arm: s['timing'].get('wall_time_seconds') for arm, s in [('baseline', a), ('tiered', b)]}))
        pairs[-1]['diagnostics'] = {arm:r.get('diagnostics', {}) for arm,r in [('baseline',ra),('tiered',rb)]}
    by_tier = {name: {arm: original([r for r in records if r['experiment']['tier'] == name
                                and r['sample']['run_id'] == arm], tasks)
                     for arm in ['baseline','tiered']} for _,name,_,_ in TIERS}
    for report in [*arms.values(), *(r for group in by_tier.values() for r in group.values())]:
        report['interpretation'] = 'Subset of paired size-budget developmental diagnostic; not a 70-query pilot.'
    return dict(schema_version=VERSION, completed=len(records), by_tier=by_tier,
                valid=sum(r['sample']['valid_output'] for r in records), arms=arms, paired_samples=pairs,
                failed_samples=[dict(arm=r['sample']['run_id'], task=r['sample']['task'],
                    query_id=r['sample']['query_id'], failure=r['sample']['failure'],
                    usage=r['sample']['usage'], timing=r['sample']['timing'], run=r['run'])
                    for r in records if not r['sample']['valid_output']],
                interpretation='Two previously evaluated pilot queries per task, selected by ID hash without scores. '
                'Paired developmental diagnostic, not a held-out benchmark. Total and reserve change together. '
                'Both arms use context 262144 and sample wall limit 600s; report their failures explicitly.')


def verify_batch_in_child(batch, row, runtime, output, env, step, support):
    request = output / 'logs' / (batch['stage'] + '-' + batch['task'] + '-verify-input.json')
    result = request.with_name(request.stem.replace('-input', '-output') + '.json')
    request.write_text(json.dumps(dict(batch=batch, row=row, runtime=str(runtime), output=str(output),
                                      result=str(result))) + '\n')
    step(batch['stage'] + '-' + batch['task'] + '-verify',
         [sys.executable, '-B', str(support), '--verify-batch', str(request)],
         output=output, cwd=runtime, env=env, timeout=120)
    return json.loads(result.read_text())


def verify_child(request):
    payload = json.loads(Path(request).read_text())
    sys.path.insert(0, payload['runtime'])
    # The helpers live beside this copied support file; task/contract imports
    # resolve from the selected immutable arm in this fresh process.
    import run_selective_expansion_pilot as pilot
    from dci_bench.protocol.contracts import CONTRACT_SHA256, MAX_TOTAL_MODEL_TOKENS
    batch = payload['batch']
    if CONTRACT_SHA256 != batch['contract_sha256'] or MAX_TOTAL_MODEL_TOKENS != batch['total']:
        raise ValueError('Verifier imported the wrong arm contract')
    pilot.expansion.CONTRACT = CONTRACT_SHA256
    records = pilot.verify_batch(batch, payload['row'], Path(payload['runtime']), Path(payload['output']))
    for record in records:
        sample = record['sample']
        if sample['valid_output'] and sample['usage']['total_tokens'] >= batch['total']:
            raise ValueError('Valid sample exceeded cumulative token limit')
        run = Path(payload['output']) / record['run']
        manifest = json.loads((run / 'run-manifest.json').read_text())
        record['data_identity'] = manifest['data']
        record['experiment'] = {key:batch[key] for key in ['tier','total','reserve','reserve_mode','profile','contract_sha256']}
        trace = json.loads((run / sample['artifacts']['trace']['path']).read_text())
        record['diagnostics'] = dict(
            peak_request_input_tokens=max((t.get('input_tokens') or 0 for t in trace.get('usage_by_turn',[])),default=0),
            control_events=trace.get('control_events', []))
    Path(payload['result']).write_text(json.dumps(records) + '\n')


def cpu_probe(runtime, output):
    """Real frozen Pi mock: long-history reserve, final repair and hard cap.

    Leave about 96k tokens: above every tier's fixed floor but below the
    original forecast for a 16k-input request plus tool growth. v1 fails to
    finalize here. Final/repair usage also pays for replaying that history.
    """
    sys.path.insert(0, str(runtime))
    from dci_bench.agents.pi_agent import run_pi_dci
    from dci_bench.protocol.contracts import CONTRACT_SHA256, MAX_TOTAL_MODEL_TOKENS, protocol_contract
    contract = protocol_contract()
    total = MAX_TOTAL_MODEL_TOKENS
    reserve = contract['budget_control']['reserve_tokens']
    output.mkdir(parents=True)
    workspace = output / 'workspace'
    (workspace / 'corpus').mkdir(parents=True)
    (workspace / 'corpus/probe.md').write_text('Independent synthetic document.\n')
    # Synthetic per-turn usage stays below the fixed context while its sum
    # reaches each cumulative boundary. No network/model calls.
    prefix = [dict(type='tool_call', name='read', arguments={'path': 'probe.md'},
                   usage={'input': 1, 'output': 1})]
    used = 2
    while total - used > 96000:
        amount = 16000
        prefix.append(dict(type='tool_call', name='bash', arguments={'command': f'echo probe-{used}'},
                           usage={'input': amount - 1, 'output': 1}))
        used += amount
    outcomes = {}
    for case, suffix in {
        'dynamic-reserve-repair': [dict(type='final', text='{}', usage={'input': 16500, 'output': 10}),
                                 dict(type='final', text='{"ranked_doc_ids":["probe"]}', usage={'input': 17000, 'output': 10})],
        'hard-cap': [dict(type='final', text='{"ranked_doc_ids":["probe"]}',
                          usage={'input': total, 'output': 1})],
    }.items():
        target = output / case
        target.mkdir()
        mock = target / 'mock.json'
        mock.write_text(json.dumps(prefix + suffix))
        result = run_pi_dci(query='Synthetic budget-boundary probe', workspace=workspace,
                           output_path=target / 'final.json', trace_path=target / 'trace.json',
                           mock_script=mock, model='mock', context_window=CONTEXT, timeout_seconds=120)
        final = json.loads((target / 'final.json').read_text())
        events = [e for e in final['control_events'] if e['action'] == 'finalize']
        if (len(events) != 1 or events[0]['reserve_tokens'] <= reserve
                or not reserve < events[0]['remaining_tokens'] <= events[0]['reserve_tokens']
                or final['base_contract_sha256'] != CONTRACT_SHA256):
            raise ValueError(f'Budget boundary/hash mismatch: {events}')
        if case == 'dynamic-reserve-repair':
            if not result.valid_output or result.repair_attempts != 1:
                raise ValueError(f'Final repair failed: {final}')
        else:
            if result.valid_output or result.failure_kind != 'token_limit':
                raise ValueError(f'Actual usage cap failed: {final}')
        outcomes[case] = dict(valid=result.valid_output, failure=result.failure_kind,
                              trigger_remaining=events[0]['remaining_tokens'],
                              forecast_reserve=events[0]['reserve_tokens'])
    (output / 'result.json').write_text(json.dumps(dict(status='passed', outcomes=outcomes,
        total=total, reserve=reserve, contract_sha256=CONTRACT_SHA256, external_model_calls=0), indent=2) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == '--verify-batch':
        verify_child(sys.argv[2])
    elif sys.argv[1] == '--cpu-probe':
        cpu_probe(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit('Unknown support operation')
