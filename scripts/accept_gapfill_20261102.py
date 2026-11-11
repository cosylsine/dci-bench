#!/usr/bin/env python3
"""Freeze the four completed gapfill sources into a new 17-task data suite.

No model is loaded and no historical suite, pool, or catalog is replaced.
The subsequent preflight and report publication consume the saved receipts.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_gapfill_20261031 import BASE, RUN, RELEASES
from dci_bench.data.pipeline.core import verify_release
from dci_bench.data.benchmark_v2 import prepare_benchmark_v2, audit_benchmark_v2

SOURCES = ('gapfill_arguana_en', 'gapfill_android_en', 'gapfill_acord_en_v3',
           'gapfill_cedar_zh_v2')
EVIDENCE = ROOT / 'results/implementation/20261102-gapfill-suite-v2'
CATALOG = BASE / 'benchmark_gapfill_20261102_v2/catalog.json'
SUITE = BASE / 'benchmark_v2_20261102_v2'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    if CATALOG.exists() or SUITE.exists():
        raise FileExistsError('This dated acceptance refuses overwrite; inspect existing receipts.')
    old_path = BASE / 'retrieval_pools/catalog.json'
    catalog = read(old_path)
    review = {row['source']: row for row in read(
        ROOT / 'results/implementation/20261031-gapfill/source-review.json')['sources']}
    receipts = []
    for source in SOURCES:
        print(f'Checking sealed source and pool metadata: {source}', flush=True)
        name = 'catalog-row-v2.json' if source in {'gapfill_arguana_en', 'gapfill_cedar_zh_v2'} else 'catalog-row.json'
        path = RUN / source / name
        row = read(path)
        assert row['content_hashes_verified'] is True
        assert digest(Path(row['ranking'])) == row['ranking_sha256']
        assert {pool['capacity'] for pool in row['pools']} == {200, 1000}
        md = RUN / 'source/metadata' / source
        assert digest(md / '_COMPLETE.json') == review[source]['export_completion_sha256']
        assert digest(md / 'provenance.json') == review[source]['provenance_sha256']
        release = RELEASES[source]
        verified = verify_release(release)
        assert verified['state'] == 'labels_ready'
        assert verified['bundle_hash'] == read(md / 'source_manifest.json')['source_fingerprint']
        for pool in row['pools']:
            pool_md = Path(pool['path']) / 'metadata' / source
            assert digest(pool_md / '_COMPLETE.json') == pool['completion_sha256']
            marker = read(pool_md / '_COMPLETE.json')
            for filename, expected in marker['metadata_hashes'].items():
                assert digest(pool_md / filename) == expected, (source, filename)
        row.update(release=str(release), release_completion_sha256=digest(release / '_COMPLETE.json'),
                   status='complete', source_files_present=True)
        assert source not in {entry['source'] for entry in catalog['sources']}
        catalog['sources'].append(row)
        receipts.append({'source': source, 'receipt_path': str(path), 'receipt_sha256': digest(path),
                         'ranking_hash_verified': True, 'pool_metadata_hashes_verified': True,
                         'release_verified': True, 'source_body_verification_recorded': True})
    catalog.update(source_count=len(catalog['sources']),
                   pool_count=sum(len(row['pools']) for row in catalog['sources']),
                   queries=sum(row['queries'] for row in catalog['sources']),
                   parent_catalog=str(old_path), parent_catalog_sha256=digest(old_path))
    save(CATALOG, catalog)
    save(EVIDENCE / 'source-closure.json', {'sources': receipts, 'benchmark_model_inference_completed': False})
    print('Preparing 17-task suite', flush=True)
    start = time.monotonic()
    prepared = prepare_benchmark_v2(output_dir=SUITE, catalog_path=CATALOG,
                                    config_path=ROOT / 'configs/benchmark_v2.json')
    save(EVIDENCE / 'prepared-manifest.json', prepared)
    audit = audit_benchmark_v2(SUITE)
    assert len(audit['ready_tasks']) == 17, audit
    assert set(audit['blocked_tasks']) == {'web_en', 'software_zh', 'enterprise_zh'}, audit
    audit.update(suite=str(SUITE), body_hashes_checked=False, elapsed_seconds=time.monotonic() - start)
    save(EVIDENCE / 'data-audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
