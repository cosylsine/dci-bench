#!/usr/bin/env python3
"""CPU acceptance/export of the two native Chinese sources before GPU ranking."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dci_bench.data.pipeline.core import load_bundle, verify_release, export_dci, file_digest
from dci_bench.data.benchmark_v2 import (
    _prepare_query_eligibility, sample_global_query_splits, sample_query_splits,
)
from dci_bench.data.expansion import verify_document_task

DEFAULT_DATA = ROOT.parent / 'dci-bench-zh-data/20261102_v4'
SOURCES = {'software_zh': 'gapfill_t2ranking_software_zh',
           'enterprise_zh': 'gapfill_cfqa_enterprise_zh'}


def export_sources(data_root: Path) -> dict:
    rows = []
    eligible_by_task = {}
    for task, source in SOURCES.items():
        release = data_root / 'releases' / source
        manifest = verify_release(release)
        bundle = load_bundle(release)
        if manifest['state'] != 'labels_ready' or manifest['automatic_labels']:
            raise ValueError(f'{source}: requires source human labels and labels_ready')
        if manifest['num_documents'] < 1000 or manifest['num_queries'] < 60:
            raise ValueError(f'{source}: fewer than 1000 documents or 60 queries')
        if bundle['provenance'].get('translation_status') != 'native':
            raise ValueError(f'{source}: missing native source provenance')
        hashes = [hashlib.sha256(row['text'].encode()).hexdigest() for row in bundle['documents']]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f'{source}: exact duplicate bodies remain')
        records = []
        for query in bundle['queries']:
            gold = bundle['qrels'][query['_id']]
            if not 1 <= len(gold) <= 200:
                raise ValueError(f'{source}/{query["_id"]}: positive count outside 1..200')
            metadata = query['metadata']
            records.append({**metadata, 'query_id': query['_id'], 'text': query['text'],
                            'group_id': metadata['group_id']})
        eligible, exclusions, _ = _prepare_query_eligibility(records, bundle['qrels'])
        if len(eligible) < 60:
            raise ValueError(f'{source}: fewer than 60 queries after benchmark eligibility checks')
        eligible_by_task[task] = eligible
        splits = sample_query_splits(eligible, dev_size=10, test_size=50, smoke_size=1,
                                     seed=42, task_id=task, group_key='group_id')
        suite = data_root / 'run/source'
        md = suite / 'metadata' / source
        if md.exists():
            exported = json.loads((md / 'source_manifest.json').read_text())
            if exported['source_fingerprint'] != manifest['bundle_hash']:
                raise ValueError(f'{source}: existing export has another source identity')
            verify_document_task(suite, source, io_workers=8)
        else:
            export_dci(release, suite, source, io_workers=8)
        rows.append({'task': task, 'source': source, 'release': str(release),
                     'release_completion_sha256': file_digest(release / '_COMPLETE.json'),
                     'export_completion_sha256': file_digest(md / '_COMPLETE.json'),
                     'documents': manifest['num_documents'], 'queries': manifest['num_queries'],
                     'positive_pairs': manifest['num_positive_pairs'],
                     'positive_min': min(map(len, bundle['qrels'].values())),
                     'positive_max': max(map(len, bundle['qrels'].values())),
                     'benchmark_eligible_queries': len(eligible),
                     'benchmark_query_exclusions': exclusions,
                     'sampling_preview': splits})
    global_splits = sample_global_query_splits(
        eligible_by_task,
        source_context_by_task={task: {'source_id': source} for task, source in SOURCES.items()},
        dev_size=10, test_size=50, smoke_size=1, seed=42,
    )
    result = {'schema_version': 'dci-zh-gapfill-readiness-v1', 'sources': rows,
              'seed': 42, 'status': 'sources_ready_for_gpu_ranking',
              'global_sampling_preview': global_splits,
              'ranking_completed': False, 'benchmark_model_inference_completed': False,
              'boundary': 'Verified source releases and exports; preview is not the final frozen benchmark suite.'}
    target = data_root / 'source-readiness.json'
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    if target.exists() and target.read_text() != payload:
        raise ValueError('Readiness receipt differs; use a new data root')
    if not target.exists():
        target.write_text(payload)
    return result


def check_sources(data_root: Path) -> dict:
    receipt = json.loads((data_root / 'source-readiness.json').read_text())
    if {r['source'] for r in receipt['sources']} != set(SOURCES.values()):
        raise ValueError('Unexpected prepared source set')
    for row in receipt['sources']:
        release = Path(row['release'])
        marker = data_root / 'run/source/metadata' / row['source'] / '_COMPLETE.json'
        if file_digest(release / '_COMPLETE.json') != row['release_completion_sha256']:
            raise ValueError(f'Changed release: {release}')
        if file_digest(marker) != row['export_completion_sha256']:
            raise ValueError(f'Changed export: {marker}')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'check'])
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    result = export_sources(args.data_root) if args.action == 'export' else check_sources(args.data_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
