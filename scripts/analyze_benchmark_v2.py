#!/usr/bin/env python3
"""Select a measured budget or compare models on the same frozen test queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dci_bench.results.benchmark_v2 import METRICS, TIERS, compare_concurrency, fingerprint, paired_interval, select_budget, validate_complete_report
from dci_bench.results.protocol import sha256_file, write_json_atomic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    budgets = sub.add_parser('select-budget')
    budgets.add_argument('--reports', type=Path, nargs=3, required=True, help='32k, 64k, 96k summary paths in that order')
    budgets.add_argument('--output', type=Path, required=True)
    concurrency = sub.add_parser('compare-concurrency')
    concurrency.add_argument('--reports', type=Path, nargs=3, required=True, help='4, 8, 16 concurrency summaries in that order')
    concurrency.add_argument('--output', type=Path, required=True)
    comparison = sub.add_parser('compare-models')
    comparison.add_argument('--left', type=Path, required=True)
    comparison.add_argument('--right', type=Path, required=True)
    comparison.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('analysis output exists; choose a new path')
    if args.command in ('select-budget', 'compare-concurrency'):
        keys = (32000, 64000, 96000) if args.command == 'select-budget' else (4, 8, 16)
        reports = {key: json.loads(path.read_text()) for key, path in zip(keys, args.reports)}
        result = select_budget(reports) if args.command == 'select-budget' else compare_concurrency(reports)
        result['schema_version'] = 'dci-benchmark-v2-' + args.command
        result['reports'] = {str(key): {'path': str(path.resolve()), 'sha256': sha256_file(path)}
                             for key, path in zip(keys, args.reports)}
    else:
        left, right = [json.loads(path.read_text()) for path in (args.left, args.right)]
        for key in ('split', 'selection_sha256', 'token_budget', 'source_fingerprint', 'concurrency'):
            if left.get(key) != right.get(key):
                raise ValueError('model comparison requires identical ' + key)
        if left['split'] != 'test' or not left['complete_scope'] or not right['complete_scope']:
            raise ValueError('formal model comparison requires complete test reports')
        validate_complete_report(left, split='test')
        validate_complete_report(right, split='test')
        if any(not report.get('model_key') or not report.get('backend_sha256') for report in (left, right)):
            raise ValueError('model comparison requires recorded model and backend identities')
        left_identity = {'model_key': left.get('model_key'), 'mode': left.get('mode'),
                         'backend_sha256': left.get('backend_sha256')}
        right_identity = {'model_key': right.get('model_key'), 'mode': right.get('mode'),
                          'backend_sha256': right.get('backend_sha256')}
        result = {'schema_version': 'dci-benchmark-v2-model-comparison', 'direction': 'left minus right',
                  'left': left_identity, 'right': right_identity,
                  'model_identities': {'left': left_identity['model_key'], 'right': right_identity['model_key']},
                  'backend_identities': {'left': left_identity['backend_sha256'],
                                         'right': right_identity['backend_sha256']},
                  'inputs': [
                      {'path': str(path.resolve()), 'sha256': sha256_file(path), **identity}
                      for path, identity in ((args.left, left_identity), (args.right, right_identity))
                  ],
                  'tiers': {tier: {metric: paired_interval([s for s in left['samples'] if s['tier'] == tier],
                                                          [s for s in right['samples'] if s['tier'] == tier], metric=metric)
                                    for metric in METRICS} for tier in TIERS}}
    result['sha256'] = fingerprint(result)
    write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
