#!/usr/bin/env python3
"""Compare sealed pool contents and recorded trajectories; no model execution."""
import argparse
import hashlib
from collections import defaultdict, Counter
import json
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.run / 'summary.json')['samples']
    plan = read(args.run / 'launch-plan.json')
    work = {(x['task'], x['capacity']): x for x in plan['work']}
    tasks = defaultdict(dict)
    checks = []
    for row in rows:
        task, cap, query = row['task'], row['capacity'], row['query_id']
        entry = work[task, cap]
        md = Path(entry['suite']) / 'metadata' / task
        gold = {doc: grade for doc, grade in read(md / 'qrels.json')[query].items() if grade > 0}
        pool = read(md / 'candidate_pools.json')['pools'][query]
        candidates = pool['candidate_doc_ids']
        assert set(gold) <= set(candidates)
        inventory = read(md / 'file_inventory.json')
        gold_hashes = {v['alias']: v['sha256'] for v in inventory.values() if v['alias'] in gold}
        assert set(gold_hashes) == set(gold)
        corpus = Path(entry['suite']).parent.parent / read(Path(entry['suite']) / 'shared_corpus.json')['corpus']
        for name, record in inventory.items():
            if record['alias'] in gold:
                actual = hashlib.sha256((corpus / Path(name).name).read_bytes()).hexdigest()
                assert actual == record['sha256'], (task, cap, name)
        run = Path(row['run'])
        sample_ref = next(x for x in read(run / 'task-result.json')['samples'] if x['query_id'] == query)
        sample_dir = (run / sample_ref['result']['path']).parent
        result = read(sample_dir / 'result.json')
        final = read(sample_dir / 'final.json')
        trace = read(sample_dir / 'trace.json')
        seen = set()
        reads = {v['doc_id'] for v in final['body_evidence']['read_receipts'] if v['returned_codepoints'] > 0}
        tool_steps = []
        pending = {}
        for event in trace['events']:
            if event['type'] == 'tool_execution_start':
                pending[event['toolCallId']] = event
            elif event['type'] == 'tool_execution_end':
                start = pending[event['toolCallId']]
                text = '\n'.join(c.get('text', '') for c in event['result'].get('content', []))
                visible = set(re.findall(r'(?<![A-Za-z0-9_])d\d+(?![A-Za-z0-9_])', text)) & set(gold)
                if start['toolName'] == 'bash':
                    seen.update(visible)
                tool_steps.append({'step': len(tool_steps) + 1, 'tool': start['toolName'],
                                   'args': start['args'], 'is_error': event.get('isError', False),
                                   'gold_ids_in_output': sorted(visible)})
        selected = set(result['ranked_doc_ids'][:10])
        calculated_recall = len(selected & set(gold)) / len(gold)
        assert abs(calculated_recall - row['metrics']['recall_at_10']) < 1e-12
        tasks[task][cap] = {'query_id': query, 'recall_at_10': calculated_recall,
                           'gold': gold, 'gold_content_hashes': gold_hashes,
                           'candidate_ids': candidates, 'gold_candidate_positions': {doc: candidates.index(doc)+1 for doc in gold},
                           'read_ids': sorted(reads), 'gold_read_ids': sorted(reads & set(gold)),
                           'gold_seen_in_bash': sorted(seen), 'submitted_ids': result['ranked_doc_ids'],
                           'gold_submitted_ids': sorted(selected & set(gold)),
                           'usage': row['usage'], 'execution': result['execution'],
                           'control_events': final['control_events'], 'tool_steps': tool_steps,
                           'final_text': final['raw_final_text'], 'trace_path': str(sample_dir / 'trace.json')}
    changed = {}
    for task, tiers in tasks.items():
        assert set(tiers) == {200, 500, 1000}
        assert tiers[200]['gold'] == tiers[500]['gold'] == tiers[1000]['gold']
        assert tiers[200]['gold_content_hashes'] == tiers[500]['gold_content_hashes'] == tiers[1000]['gold_content_hashes']
        check = {'task': task, 'gold_identical_present_all_tiers': True,
                 'gold_content_hashes_identical': True, 'gold_content_bytes_verified': True,
                 'nested_200_500': set(tiers[200]['candidate_ids']) <= set(tiers[500]['candidate_ids']),
                 'nested_500_1000': set(tiers[500]['candidate_ids']) <= set(tiers[1000]['candidate_ids'])}
        checks.append(check)
        if tiers[200]['recall_at_10'] != tiers[1000]['recall_at_10']:
            changed[task] = tiers
    aggregate = {}
    for cap in (200, 500, 1000):
        all_rows = [v[cap] for v in tasks.values()]
        stops = [e for x in all_rows for e in x['control_events'] if e['action'] == 'finalize']
        aggregate[cap] = {'mean_unique_documents_read': statistics.mean(len(x['read_ids']) for x in all_rows),
                          'mean_model_calls': statistics.mean(x['execution']['model_calls'] for x in all_rows),
                          'gold_read_recall_mean': statistics.mean(len(x['gold_read_ids'])/len(x['gold']) for x in all_rows),
                          'samples_gold_read_but_none_submitted': sum(bool(x['gold_read_ids']) and not x['gold_submitted_ids'] for x in all_rows),
                          'mean_used_budget_fraction': statistics.mean(x['usage']['total_tokens']/plan['budgets'][str(cap)] for x in all_rows),
                          'finalize_reasons': dict(Counter(e['reason'] for e in stops)),
                          'mean_remaining_tokens_at_finalize': statistics.mean(e['remaining_tokens'] for e in stops)}
    output = {'run': str(args.run.resolve()), 'method': 'Read-only pool metadata and recorded result/final/trace comparison; no counterfactual inference.',
              'pool_checks': checks, 'aggregate': aggregate, 'changed_queries': changed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'sources_checked': len(checks), 'changed': len(changed), 'nested_200_500': sum(c['nested_200_500'] for c in checks), 'nested_500_1000': sum(c['nested_500_1000'] for c in checks), 'aggregate': aggregate}, ensure_ascii=False, indent=2))
    for task, tiers in changed.items():
        print(task, [(cap, x['recall_at_10'], len(x['read_ids']), x['gold_read_ids'], x['usage']['total_tokens'], x['gold_candidate_positions']) for cap,x in sorted(tiers.items())])


if __name__ == '__main__':
    main()
