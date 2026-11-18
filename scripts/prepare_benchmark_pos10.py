#!/usr/bin/env python3
"""Reuse source pools and freeze a new seed-42 benchmark after legal replacement."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dci_bench.data.benchmark_v2 import (
    _load_pool_metadata, _prepare_query_eligibility, audit_benchmark_v2,
    load_benchmark_config, prepare_benchmark_v2, sha256_file,
)

BASE = Path('/mnt/afs2/202611/embedding_models/dci-bench/eval_data')
OLD_SUITE = BASE / 'benchmark_v2_20261103_v1'
NEW_SUITE = BASE / 'benchmark_v2_20261116_pos10_v1'


def reused_sources(base_suite: Path) -> tuple[list[dict], list[dict]]:
    assessment = json.loads((base_suite / 'assessment.json').read_text())
    frozen = json.loads((base_suite / 'manifest.json').read_text())
    sources, inventory = [], []
    for task, item in sorted(assessment.items()):
        if task.startswith('legal_'):
            continue
        source = copy.deepcopy(item['source'])
        # The corpus migration changed absolute paths, not the source bytes.
        # Use the live suite reference to establish the current full-pool root.
        shared = json.loads((base_suite / 'tasks' / task / 'full/shared_corpus.json').read_text())
        full = Path(shared['source_pool'])
        old_full = next(Path(row['path']) for row in source['pools'] if row['capacity'] == 1000)
        old_parent, new_parent = old_full.parent, full.parent
        for pool in source['pools']:
            pool['path'] = str(new_parent / Path(pool['path']).relative_to(old_parent))
            marker = Path(pool['path']) / 'metadata' / source.get('task', source['source']) / '_COMPLETE.json'
            pool['previous_catalog_completion_sha256'] = pool['completion_sha256']
            pool['completion_sha256'] = sha256_file(marker)
            completion = json.loads(marker.read_text())
            for name in ['queries.jsonl', 'qrels.json', 'source_doc_ids.json', 'source_identity.json']:
                if sha256_file(marker.parent / name) != completion['metadata_hashes'][name]:
                    raise ValueError(f'{task}: reused metadata hash mismatch: {name}')
        metadata = _load_pool_metadata(full, source.get('task', source['source']), include_candidate_pools=False)
        if metadata['source_identity'] != frozen['tasks'][task]['source_identity']:
            raise ValueError(f'{task}: source identity differs from the previous frozen suite')
        eligible, excluded, _ = _prepare_query_eligibility(
            metadata['queries'], metadata['qrels'], max_positive_count=10,
        )
        if len(eligible) < 60:
            raise ValueError(f'{task}: only {len(eligible)} eligible queries')
        sources.append(source)
        inventory.append({'task': task, 'source': source['source'],
                          'source_queries': len(metadata['queries']),
                          'eligible_queries': len(eligible), 'excluded_queries': len(excluded),
                          'full_source_pool': str(full), 'requires_gpu_pooling': False})
    return sources, inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inventory', 'freeze'])
    parser.add_argument('--base-suite', type=Path, default=OLD_SUITE)
    parser.add_argument('--output', type=Path, default=NEW_SUITE)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/benchmark_v2.json')
    parser.add_argument('--legal-en-catalog', type=Path)
    parser.add_argument('--legal-zh-catalog', type=Path)
    args = parser.parse_args()
    sources, inventory = reused_sources(args.base_suite)
    if args.action == 'inventory':
        print(json.dumps(inventory, ensure_ascii=False, indent=2))
        return
    config = load_benchmark_config(args.config)
    if config.get('query_eligibility', {}).get('max_positive_count') != 10:
        raise ValueError('New suite requires the 1..10 positive policy')
    for task, path in [('legal_en', args.legal_en_catalog), ('legal_zh', args.legal_zh_catalog)]:
        if path is None:
            parser.error(f'{task} requires a completed replacement catalog row')
        row = json.loads(path.read_text())
        expected = config['sources'][task]['candidates'][0]
        old_id = json.loads((args.base_suite / 'assessment.json').read_text())[task]['source_id']
        if row['source'] != expected['source_id'] or row['source'] == old_id:
            raise ValueError(f'{task}: replacement identity differs from configured source')
        if expected['provenance'] != 'native':
            raise ValueError(f'{task}: replacement must have native provenance')
        for pool in row['pools']:
            marker = Path(pool['path']) / 'metadata' / row.get('task', row['source']) / '_COMPLETE.json'
            if sha256_file(marker) != pool['completion_sha256']:
                raise ValueError(f'{task}: pool completion hash mismatch')
        if sha256_file(Path(row['ranking'])) != row['ranking_sha256']:
            raise ValueError(f'{task}: saved ranking hash mismatch')
        sources.append(row)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite suite: {args.output}')
    evidence = args.output.with_name(args.output.name + '-inputs')
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence / 'catalog.json').write_text(json.dumps({'sources': sources}, ensure_ascii=False, indent=2) + '\n')
    (evidence / 'reuse-inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')
    # Freeze the reviewed config alongside its catalog before constructing pools.
    (evidence / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    manifest = prepare_benchmark_v2(output_dir=args.output, config_path=evidence / 'config.json',
                                    catalog_path=evidence / 'catalog.json', seed=42)
    audit = audit_benchmark_v2(args.output)
    (evidence / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    if not audit['ready_for_formal']:
        blocked = {task: row['reason'] for task, row in manifest['tasks'].items() if row['status'] != 'ready'}
        raise ValueError(f'New suite is incomplete; preserved for diagnosis: {blocked}')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
