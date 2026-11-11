#!/usr/bin/env python3
"""Summarize frozen query recall from saved pre-injection pool audits; CPU only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.data.benchmark_v2 import (
    _load_selected_candidate_payload, sha256_file, stable_json_sha256,
)


def summarize(rows: list[dict]) -> dict:
    return {'query_count': len(rows), 'positive_count': sum(r['positive_count'] for r in rows),
            **{f'recall_at_{k}': mean(r[f'hits_at_{k}'] / r['positive_count'] for r in rows)
               for k in (200, 1000)},
            **{f'hits_at_{k}': sum(r[f'hits_at_{k}'] for r in rows) for k in (200, 1000)}}


def build(evidence: Path) -> dict:
    prepared_path, catalog_path = evidence / 'prepared-manifest.json', evidence / 'catalog.json'
    prepared, catalog = (json.loads(p.read_text()) for p in (prepared_path, catalog_path))
    if stable_json_sha256({k: v for k, v in prepared.items() if k != 'overall_sha256'}) != prepared['overall_sha256']:
        raise ValueError('Prepared manifest identity mismatch')
    if stable_json_sha256(catalog) != prepared['config_identity']['catalog_sha256']:
        raise ValueError('Catalog differs from the frozen selection')
    sources = {row['source']: row for row in catalog['sources']}
    tasks, queries = [], []
    inputs = {str(p.resolve()): sha256_file(p) for p in (prepared_path, catalog_path, Path(__file__))}
    for domain in prepared['scope']['domains']:
        for language in prepared['scope']['languages']:
            task = f'{domain}_{language}'
            selection = prepared['tasks'][task]
            if selection['status'] != 'ready':
                raise ValueError(f'{task}: recall summary requires a prepared task')
            selected_ids = selection['query_ids']['dev'] + selection['query_ids']['test']
            if len(selected_ids) != 60 or len(set(selected_ids)) != 60:
                raise ValueError(f'{task}: expected 60 distinct frozen dev/test query IDs')
            source = sources[selection['source_id']]
            pool = next(p for p in source['pools'] if p['capacity'] == 1000)
            pool_root = Path(pool['path'])
            metadata = pool_root / 'metadata' / selection['source_task']
            identity = json.loads((metadata / 'source_identity.json').read_text())
            if identity != selection['source_identity']:
                raise ValueError(f'{task}: source identity differs from the frozen selection')
            source_ids = json.loads((metadata / 'source_doc_ids.json').read_text())
            qrels = json.loads((metadata / 'original_qrels.json').read_text())
            payload = _load_selected_candidate_payload(pool_root, selection['source_task'], selected_ids, source_ids)
            rows = []
            for split in ('dev', 'test'):
                for query_id in selection['query_ids'][split]:
                    audit = payload['pools'][query_id]['audit']
                    initial = audit['initial_top_ids']
                    positives = {doc for doc, score in qrels[query_id].items() if score > 0}
                    missing = positives - set(initial)
                    if (audit['capacity'] != 1000 or audit['ordering'] != 'scored'
                            or len(initial) != 1000 or len(set(initial)) != 1000
                            or not positives or len(positives) != audit['positive_count']
                            or missing != set(audit['injected_positive_ids'])
                            or len(missing) != audit['injection_count']):
                        raise ValueError(f'{task}/{query_id}: inconsistent pre-injection audit or qrels')
                    rows.append({'task': task, 'query_id': query_id, 'split': split,
                                 'positive_count': len(positives),
                                 'hits_at_200': len(positives & set(initial[:200])),
                                 'hits_at_1000': len(positives & set(initial))})
            files = [metadata / name for name in ('candidate_pools.json', 'original_qrels.json',
                                                  'source_doc_ids.json', 'source_identity.json', '_COMPLETE.json')]
            inputs.update({str(p): sha256_file(p) for p in files})
            tasks.append({'task': task, 'source': selection['source_id'], 'domain': domain, 'language': language,
                          'source_pool': str(pool_root),
                          'splits': {split: summarize([r for r in rows if r['split'] == split])
                                     for split in ('dev', 'test')}})
            queries.extend(rows)
            print(f'{task}: {len(rows)} frozen queries checked', flush=True)
    return {'schema_version': 'dci-kalm-pre-injection-recall-v1',
            'computed_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
            'prepared_manifest_sha256': prepared['overall_sha256'],
            'method': {'ranking': 'source pool audit.initial_top_ids, decoded to original document IDs; first 200 / 1000',
                       'positives': 'source pool original_qrels, relevance > 0',
                       'aggregation': 'mean of per-query positive recall; equal task weight because every task has dev10/test50',
                       'scope': 'frozen dev and test separately; smoke is a dev subset, not counted again',
                       'boundary': 'offline source-pool retrieval before positive injection, not DCI agent evaluation'},
            'inputs_sha256': inputs, 'tasks': tasks, 'queries': queries,
            'aggregate': {split: summarize([r for r in queries if r['split'] == split])
                          for split in ('dev', 'test')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = build(args.evidence)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'kalm-pre-injection-recall.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (args.output / 'kalm-pre-injection-recall.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['task', 'source', 'split', 'query_count', 'positive_count',
                  'recall_at_200', 'recall_at_1000', 'hits_at_200', 'hits_at_1000']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for task in result['tasks']:
            for split, stats in task['splits'].items():
                writer.writerow({'task': task['task'], 'source': task['source'], 'split': split, **stats})
    print(json.dumps(result['aggregate'], ensure_ascii=False, indent=2))
