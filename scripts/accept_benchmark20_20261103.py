#!/usr/bin/env python3
"""Add the sealed BrowseComp+ pools to the accepted 19-task suite, on CPU."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.data.benchmark_v2 import (
    _load_pool_metadata, audit_benchmark_v2, prepare_benchmark_v2,
    sha256_file, stable_json_sha256,
)
from scripts.prepare_gapfill_20261031 import BASE, RUN
from scripts.publish_zh_gapfill_report_20261102 import bcp_completion_metadata
from scripts.accept_zh_gapfill_20261102 import _selected_full_hashes

SOURCE = 'gapfill_browsecomp_plus_en'
ZH_ROOT = Path('/mnt/afs/250010100/zxp/dci-bench-zh-data/20261102_v4')
BASE_SUITE = ZH_ROOT / 'suites/benchmark_v2_20261102_zh_v1'
BASE_EVIDENCE = ZH_ROOT / 'evidence/suite-acceptance'
DEFAULT_SUITE = BASE / 'benchmark_v2_20261103_v1'
DEFAULT_EVIDENCE = ROOT / 'results/implementation/20261103-benchmark20-v1'


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def source_closure() -> dict:
    receipt = bcp_completion_metadata()
    if receipt is None:
        raise ValueError('BrowseComp+ source pools have not completed')
    row = read(receipt)
    review_path = ROOT / 'results/implementation/20261031-gapfill/source-review.json'
    review = next(item for item in read(review_path)['sources'] if item['source'] == SOURCE)
    metadata = RUN / 'source/metadata' / SOURCE
    provenance = read(metadata / 'provenance.json')
    for name, expected in (('_COMPLETE.json', review['export_completion_sha256']),
                           ('provenance.json', review['provenance_sha256'])):
        if sha256_file(metadata / name) != expected:
            raise ValueError(f'BrowseComp+ reviewed source changed: {name}')
    for name, digest in read(metadata / '_COMPLETE.json')['metadata_hashes'].items():
        if Path(name).name != name or sha256_file(metadata / name) != digest:
            raise ValueError(f'BrowseComp+ export metadata changed: {name}')
    parent = Path(provenance['parent_suite'])
    parent_marker = parent / 'metadata/browsecomp_plus/_COMPLETE.json'
    if sha256_file(parent_marker) != provenance['parent_completion_sha256']:
        raise ValueError('BrowseComp+ parent completion changed')
    if (provenance['translation_status'], provenance['language'], provenance['domain']) != ('native', 'en', 'web'):
        raise ValueError('BrowseComp+ native English web provenance differs')
    export_manifest = read(metadata / 'manifest.json')
    if (row['documents'], row['queries'], export_manifest['num_qrels']) != (99930, 825, 5023):
        raise ValueError('BrowseComp+ frozen source counts differ')
    if export_manifest['resolved_revision'] != review['resolved_revision']:
        raise ValueError('BrowseComp+ source revision changed')
    qrels = read(metadata / 'qrels.json')
    original_qrels = read(metadata / 'original_qrels.json')
    mapping = read(metadata / 'source_doc_ids.json')
    pool_checks = []
    for pool in row['pools']:
        full = _load_pool_metadata(Path(pool['path']), SOURCE)
        if (full['qrels'] != qrels or full['original_qrels'] != original_qrels
                or full['source_doc_ids'] != mapping):
            raise ValueError('BrowseComp+ source pool labels or IDs differ from export')
        if full['source_identity']['source_manifest_sha256'] != sha256_file(metadata / 'manifest.json'):
            raise ValueError('BrowseComp+ pool/export identity differs')
        records = full['candidate_pools']['pools']
        if set(records) != set(qrels):
            raise ValueError('BrowseComp+ pool query coverage differs')
        for query_id, record in records.items():
            candidates = record['candidate_doc_ids']
            positives = {doc for doc, grade in qrels[query_id].items() if grade > 0}
            if len(candidates) != pool['capacity'] or len(set(candidates)) != pool['capacity']:
                raise ValueError(f'BrowseComp+ pool capacity mismatch: {query_id}')
            if not positives or not positives <= set(candidates) or not set(candidates) <= mapping.keys():
                raise ValueError(f'BrowseComp+ missing positive or unknown candidate: {query_id}')
        pool_checks.append({'capacity': pool['capacity'], 'queries': len(records),
                            'positive_pairs': sum(sum(g > 0 for g in labels.values()) for labels in qrels.values()),
                            'all_positives_preserved': True})
    ranking = read(Path(row['ranking']))['rankings']
    quality = {}
    for capacity in (200, 1000):
        recalls = []
        for query_id, labels in original_qrels.items():
            positives = {doc for doc, grade in labels.items() if grade > 0}
            recalls.append(len(positives & set(ranking[query_id][:capacity])) / len(positives))
        quality[str(capacity)] = {'queries': len(recalls), 'pre_injection_macro_recall': sum(recalls) / len(recalls)}
    row.update(release=str(parent), release_completion_sha256=sha256_file(parent_marker),
               status='complete', source_files_present=True)
    return {'catalog_row': row, 'receipt_path': str(receipt), 'receipt_sha256': sha256_file(receipt),
            'source_review_path': str(review_path), 'source_review_sha256': sha256_file(review_path),
            'export_completion_sha256': sha256_file(metadata / '_COMPLETE.json'),
            'provenance_sha256': sha256_file(metadata / 'provenance.json'),
            'pool_checks': pool_checks, 'retrieval_quality': quality,
            'source_metadata_hashes_verified': True, 'source_body_verification': 'inherited sealed source evidence',
            'new_selected_bodies': 'verified separately after suite selection',
            'benchmark_model_inference_completed': False}


def assert_selection_stable(base: dict, current: dict) -> dict:
    old = {task: row for task, row in base['tasks'].items() if row['status'] == 'ready'}
    ready = {task for task, row in current['tasks'].items() if row['status'] == 'ready'}
    if len(old) != 19 or 'web_en' in old or ready != set(old) | {'web_en'}:
        raise ValueError('Expected the accepted 19 tasks plus web_en')
    fields = ('source_id', 'source_task', 'query_ids', 'query_groups')
    changed = {task: [key for key in fields if row[key] != current['tasks'][task][key]]
               for task, row in old.items()}
    changed = {task: keys for task, keys in changed.items() if keys}
    if changed:
        raise ValueError(f'Existing source/query/group selection changed: {changed}')
    return {'stable_old_task_count': len(old), 'stable_old_tasks': sorted(old),
            'fields_checked': list(fields), 'changed': changed}


def validate_pools(suite: Path, manifest: dict, base_suite: Path) -> dict:
    dev_groups, test_groups = set(), set()
    pool_records = 0
    stable_metadata = 0
    task_checks = []
    for task, row in sorted(manifest['tasks'].items()):
        ids = row['query_ids']
        if {split: len(values) for split, values in ids.items()} != {'dev': 10, 'test': 50, 'smoke': 1}:
            raise ValueError(f'Wrong split counts: {task}')
        if set(ids['dev']) & set(ids['test']) or not set(ids['smoke']) <= set(ids['dev']):
            raise ValueError(f'Invalid query split isolation: {task}')
        dev_groups.update(row['query_groups'][qid] for qid in ids['dev'])
        test_groups.update(row['query_groups'][qid] for qid in ids['test'])
        selected = set(ids['dev']) | set(ids['test'])
        tiers = {}
        for tier, capacity in (('lite', 200), ('full', 1000)):
            pool = suite / row['pools'][tier]['suite']
            metadata = pool / 'metadata' / task
            values = _load_pool_metadata(pool, task)
            records = values['candidate_pools']['pools']
            if set(records) != selected:
                raise ValueError(f'Final pool selection differs: {task}/{tier}')
            if list((pool / 'workspaces' / task).iterdir()):
                raise ValueError(f'Final agent workspace is not empty: {task}/{tier}')
            for query in values['queries']:
                if {'answer', 'answers', 'qrels', 'gold', 'gold_doc_ids', 'host_labels', 'judgments'} & query.keys():
                    raise ValueError(f'Host labels exposed in query: {task}/{tier}')
            for qid, record in records.items():
                candidates = record['candidate_doc_ids']
                positives = {doc for doc, grade in values['qrels'][qid].items() if grade > 0}
                original = {values['source_doc_ids'][doc]: grade for doc, grade in values['qrels'][qid].items()}
                if original != values['original_qrels'][qid]:
                    raise ValueError(f'Original positive labels differ: {task}/{tier}/{qid}')
                if len(candidates) != capacity or len(set(candidates)) != capacity or not positives <= set(candidates):
                    raise ValueError(f'Final pool capacity/positive failure: {task}/{tier}/{qid}')
                pool_records += 1
            if task != 'web_en':
                old_metadata = base_suite / row['pools'][tier]['suite'] / 'metadata' / task
                for name in ('queries.jsonl', 'qrels.json', 'original_qrels.json', 'candidate_pools.json',
                             'source_doc_ids.json', 'doc_id_files.json', 'file_inventory.json',
                             'source_identity.json', '_COMPLETE.json'):
                    if sha256_file(metadata / name) != sha256_file(old_metadata / name):
                        raise ValueError(f'Existing task metadata changed: {task}/{tier}/{name}')
                    stable_metadata += 1
            tiers[tier] = records
        for qid in selected:
            if not set(tiers['lite'][qid]['candidate_doc_ids']) <= set(tiers['full'][qid]['candidate_doc_ids']):
                raise ValueError(f'Lite is not nested in Full: {task}/{qid}')
        task_checks.append({'task': task, 'queries': len(selected), 'pool_records': 120,
                            'lite_subset_of_full': True, 'all_positives_preserved': True,
                            'host_only_labels': True})
        print(f'Validated {task}: 60 queries, nested 200/1000 pools', flush=True)
    if dev_groups & test_groups:
        raise ValueError('Global dev/test query group leakage')
    return {'passed': True, 'tasks': task_checks, 'pool_records_checked': pool_records,
            'stable_old_metadata_files': stable_metadata, 'dev_groups': len(dev_groups),
            'test_groups': len(test_groups), 'global_group_leakage': False}


def accept(output_suite: Path, evidence: Path) -> dict:
    if output_suite.exists() or evidence.exists():
        raise FileExistsError('Use new suite and evidence paths; prior artifacts must be retained')
    baseline = read(BASE_SUITE / 'manifest.json')
    base_audit = audit_benchmark_v2(BASE_SUITE)
    if len(base_audit['ready_tasks']) != 19 or base_audit['blocked_tasks'] != ['web_en']:
        raise ValueError('Baseline is not the accepted 19-task suite')
    config = read(BASE_EVIDENCE / 'config.json')
    if stable_json_sha256(config) != stable_json_sha256(read(ROOT / 'configs/benchmark_v2.json')):
        raise ValueError('Current config changed since the 19-task acceptance')
    print('Checking sealed BrowseComp+ source, rankings and both pool metadata sets', flush=True)
    closure = source_closure()
    catalog = copy.deepcopy(read(BASE_EVIDENCE / 'catalog.json'))
    if SOURCE in {row['source'] for row in catalog['sources']}:
        raise ValueError('BrowseComp+ already exists in the baseline catalog')
    catalog['sources'].append(closure['catalog_row'])
    catalog.update(source_count=len(catalog['sources']),
                   pool_count=sum(len(row['pools']) for row in catalog['sources']),
                   queries=sum(row['queries'] for row in catalog['sources']),
                   parent_catalog=str(BASE_EVIDENCE / 'catalog.json'),
                   parent_catalog_sha256=sha256_file(BASE_EVIDENCE / 'catalog.json'))
    evidence.mkdir(parents=True)
    save(evidence / 'config.json', config)
    save(evidence / 'catalog.json', catalog)
    save(evidence / 'source-closure.json', closure)
    save(evidence / 'baseline.json', {
        'suite': str(BASE_SUITE), 'manifest_sha256': sha256_file(BASE_SUITE / 'manifest.json'),
        'audit': base_audit, 'body_evidence': [
            str(BASE_EVIDENCE / 'status-ready.json'),
            str(ROOT / 'results/implementation/20261102-gapfill-suite-v2/selected-body-validation.json')],
        'boundary': 'Old 19 tasks retain their existing body evidence; new web_en bodies are rehashed.'})
    print(f'Preparing immutable 20-task suite: {output_suite}', flush=True)
    prepared = prepare_benchmark_v2(output_dir=output_suite, config_path=evidence / 'config.json',
                                    catalog_path=evidence / 'catalog.json', seed=42)
    return complete_acceptance(output_suite, evidence, prepared, baseline)


def complete_acceptance(output_suite: Path, evidence: Path, prepared: dict, baseline: dict) -> dict:
    selection = assert_selection_stable(baseline, prepared)
    save(evidence / 'old-19-stability.json', selection)
    save(evidence / 'prepared-manifest.json', prepared)
    audit = audit_benchmark_v2(output_suite)
    if len(audit['ready_tasks']) != 20 or audit['blocked_tasks'] or not audit['ready_for_formal']:
        raise ValueError(f'Expected all 20 data tasks ready: {audit}')
    audit['body_hashes_checked'] = False
    save(evidence / 'data-audit.json', audit)
    checks = validate_pools(output_suite, prepared, BASE_SUITE)
    if checks['pool_records_checked'] != 2400:
        raise ValueError('Expected 2400 final query/tier pool records')
    save(evidence / 'suite-validation.json', checks)
    print('Hashing selected web_en Full bodies; Lite is a verified subset', flush=True)
    body_hashes = _selected_full_hashes(output_suite, prepared, 'web_en')
    save(evidence / 'selected-body-validation.json', body_hashes)
    result = {'status': 'data_accepted', 'suite': str(output_suite),
              'manifest_sha256': sha256_file(output_suite / 'manifest.json'),
              'ready_tasks': 20, 'blocked_tasks': [], 'data_ready_for_formal': True,
              'formal_model_execution_ready': False, 'benchmark_model_inference_completed': False,
              'old_task_count_unchanged': 19, 'pool_records_checked': 2400,
              'new_body_count_verified': body_hashes['verified_unique_body_count'],
              'body_verification_scope': 'Selected web_en Full body bytes; old 19 tasks use prior evidence and identical metadata.',
              'next_step': '40 data-only preflight jobs, then independent report publication'}
    save(evidence / 'status-ready.json', result)
    return result


def resume_validation(output_suite: Path, evidence: Path, previous: Path) -> dict:
    if evidence.exists():
        raise FileExistsError('Resume validation requires a new evidence directory')
    prepared = read(output_suite / 'manifest.json')
    if prepared != read(previous / 'prepared-manifest.json'):
        raise ValueError('The existing suite differs from the preserved preparation receipt')
    baseline = read(BASE_SUITE / 'manifest.json')
    if sha256_file(BASE_SUITE / 'manifest.json') != read(previous / 'baseline.json')['manifest_sha256']:
        raise ValueError('The 19-task baseline changed')
    bcp = bcp_completion_metadata()
    if bcp is None or sha256_file(bcp) != read(previous / 'source-closure.json')['receipt_sha256']:
        raise ValueError('The sealed BCP source changed')
    evidence.mkdir(parents=True)
    for name in ('config.json', 'catalog.json', 'source-closure.json', 'baseline.json'):
        save(evidence / name, read(previous / name))
    save(evidence / 'resume.json', {'previous_evidence': str(previous),
        'suite': str(output_suite), 'suite_manifest_sha256': sha256_file(output_suite / 'manifest.json'),
        'reason': 'Final tier metadata contains qrels and _COMPLETE, not host_labels.json; resume corrected verification.',
        'suite_rebuilt': False, 'previous_artifacts_preserved': True})
    return complete_acceptance(output_suite, evidence, prepared, baseline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-suite', type=Path, default=DEFAULT_SUITE)
    parser.add_argument('--evidence-dir', type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    result = (resume_validation(args.output_suite, args.evidence_dir, args.resume_from) if args.resume_from
              else accept(args.output_suite, args.evidence_dir))
    print(json.dumps(result, ensure_ascii=False), flush=True)
