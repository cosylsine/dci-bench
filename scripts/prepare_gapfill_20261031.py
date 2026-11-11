#!/usr/bin/env python3
"""Immutable data-only integration for the 2026-10-31 benchmark gap review."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dci_bench.data.pipeline.core import (
    file_digest, load_bundle, publish_bundle, read_json, verify_release, write_json,
)

BASE = Path('/mnt/afs2/202609/embedding_models/dci-bench/eval_data')
RUN = BASE / 'benchmark_gapfill_20261031_v1'
SOURCES = {
    'gapfill_arguana_en': BASE / 'pipeline_v1/domain_arguana_test/expansion-v1-fullqrels-noequiv',
    'gapfill_android_en': BASE / 'pipeline_v1/gapfill_beir_cqadupstack_android/v2-fulltext',
}
RELEASES = {
    'gapfill_arguana_en': BASE / 'pipeline_v1/gapfill_arguana_en/20261031_v2',
    'gapfill_android_en': BASE / 'pipeline_v1/gapfill_android_en/20261031_v2',
    'gapfill_acord_en_v3': BASE / 'pipeline_v1/gapfill_acord_en/20261031_v3',
    'gapfill_cedar_zh_v2': BASE / 'pipeline_v1/gapfill_cedar_zh/20261031_v2',
    'gapfill_browsecomp_plus_en': BASE / 'dci_expansion_20261014_v1',
}


def derive(source: str) -> dict:
    parent = SOURCES[source]
    output = BASE / 'pipeline_v1' / source / '20261031_v1'
    if output.exists():
        return verify_release(output)
    bundle = load_bundle(parent)
    is_arguana = source == 'gapfill_arguana_en'
    for row in bundle['queries']:
        row['metadata'].update(translation_status='native', original_split='test')
        if is_arguana:
            row['metadata']['group_id'] = 'arguana-topic:' + row['_id'].rsplit('-', 1)[0]
    # CQADupStack duplicate questions form connected components. Keep linked
    # questions on one side of the dev/test boundary, even across qrels rows.
    if not is_arguana:
        parent_id = {}

        def find(value):
            parent_id.setdefault(value, value)
            while parent_id[value] != value:
                value = parent_id[value]
            return value

        for qid, gold in bundle['qrels'].items():
            for did in gold:
                a, b = find(qid), find(did)
                parent_id[max(a, b)] = min(a, b)
        for row in bundle['queries']:
            row['metadata']['group_id'] = 'cqadupstack-android:' + find(row['_id'])
    bundle['provenance'].update({
        'source_id': source, 'translation_status': 'native',
        'license': 'CC-BY-SA-4.0 (BEIR dataset card; preserve original attribution)',
        'review_date': '2026-10-31', 'review_parent': str(parent),
        'review_policy': 'Exclude entire queries with unlabelled exact-text positive equivalents or >200 positives; never manufacture qrels.',
        'self_match_policy': 'Remove original query ID from each saved ranking before pool selection.' if is_arguana else 'Source duplicate-question qrels unchanged.',
        'task_instruction': ('Retrieve counterarguments to the query argument.' if is_arguana else
                             'Retrieve Android questions that are duplicates of the query question.'),
        'review_sources': ([
            'https://huggingface.co/datasets/BeIR/arguana',
            'https://aclanthology.org/P18-1023/',
            'https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py',
        ] if is_arguana else [
            'https://huggingface.co/datasets/BeIR/cqadupstack',
            'https://huggingface.co/datasets/BeIR/cqadupstack-qrels',
        ]),
    })
    return publish_bundle(bundle, output, parent_release=parent, policy={
        'max_positive_per_query': 200, 'exclude_review_required': True,
        'drop_near_duplicates': False,
    })


def approve_cleaned(source: str) -> dict:
    parent = BASE / 'pipeline_v1' / source / '20261031_v1'
    output = parent.with_name('20261031_v2')
    if output.exists():
        return verify_release(output)
    bundle = load_bundle(parent)
    audit = read_json(parent / 'audit.json')
    if audit['review_required'] or audit['errors']:
        raise ValueError('Unresolved retained-data issues')
    bundle['provenance']['reviewed_exclusions'] = audit['exclusions']
    bundle['provenance']['review_resolution'] = 'All ambiguous queries were excluded in the parent release; retained qrels are unchanged and re-audited.'
    return publish_bundle(bundle, output, parent_release=parent,
                          policy={'max_positive_per_query': 200, 'exclude_review_required': True})


def derive_acord() -> dict:
    parent = BASE / 'pipeline_v1/gapfill_acord_en/20261031_v1'
    output = parent.with_name('20261031_v2')
    if output.exists():
        return verify_release(output)
    bundle = load_bundle(parent)
    for row in bundle['queries']:
        # A contract category is a broad class, not an equivalent query.
        # Preserve it for auditing; isolate original/parent query families.
        row['metadata']['group_id'] = 'acord-query:' + (row['metadata'].get('parent_query_id') or row['_id'])
    bundle['provenance'].update({
        'source_id': 'gapfill_acord_en_v2', 'translation_status': 'native',
        'group_policy': 'Original/parent query identity; broad category retained as metadata. Supplemental rewrites excluded.',
        'split_policy': 'DCI seed42 dev10/test50 from all114 original queries; official train/valid/test retained in host metadata. Not an official ACORD test score.',
        'task_instruction': 'Retrieve contract clauses satisfying the legal clause-search request.',
    })
    return publish_bundle(bundle, output, parent_release=parent,
                          policy={'max_positive_per_query': 200, 'exclude_review_required': True})


def derive_cedar() -> dict:
    parent = BASE / 'pipeline_v1/gapfill_cedar_zh/20261031_v1_reviewed'
    output = parent.with_name('20261031_v2')
    if output.exists():
        return verify_release(output)
    bundle = load_bundle(parent)
    canonical_by_annotation = {
        alias['source_annotation_id']: document['_id']
        for document in bundle['documents'] for alias in document['metadata']['source_aliases']
    }
    for row in bundle['queries']:
        row['metadata']['self_match_document_ids'] = [canonical_by_annotation[row['metadata']['source_annotation_id']]]
        if set(row['metadata']['self_match_document_ids']) & set(bundle['qrels'][row['_id']]):
            raise ValueError('CEDAR self match is annotated positive; requires separate review')
    bundle['provenance'].update({
        'source_id': 'gapfill_cedar_zh_v2', 'translation_status': 'native',
        'task_instruction': '检索与查询论点存在明确支持或反对关系的论点和证据。不要返回查询论点自身；仅话题相同不代表相关。',
        'self_match_policy': 'Exclude the canonical document of the query claim after exact-text alias resolution.',
    })
    return publish_bundle(bundle, output, parent_release=parent,
                          policy={'max_positive_per_query': 200, 'exclude_review_required': True})


def derive_acord_heldout() -> dict:
    parent = BASE / 'pipeline_v1/gapfill_acord_en/20261031_v2'
    output = parent.with_name('20261031_v3')
    if output.exists():
        manifest = verify_release(output)
        if (manifest['parent_release'] != str(parent)
                or manifest['parent_completion_sha256'] != file_digest(parent / '_COMPLETE.json')
                or (manifest['num_documents'], manifest['num_queries'], manifest['num_positive_pairs']) != (3931, 63, 676)):
            raise ValueError('Existing ACORD held-out release differs from the frozen selection')
        return manifest
    bundle = load_bundle(parent)
    splits = Counter(row['metadata']['source_split'] for row in bundle['queries'])
    if splits != {'train': 51, 'valid': 6, 'test': 57} or any(
        row['metadata']['split'] != row['metadata']['source_split'] for row in bundle['queries']
    ):
        raise ValueError('ACORD official split metadata differs from the frozen source')
    excluded = [row['_id'] for row in bundle['queries'] if row['metadata']['source_split'] == 'train']
    bundle['queries'] = [row for row in bundle['queries'] if row['metadata']['source_split'] in {'valid', 'test'}]
    retained = {row['_id'] for row in bundle['queries']}
    bundle['qrels'] = {qid: gold for qid, gold in bundle['qrels'].items() if qid in retained}
    for field in ('evidence', 'judgments'):
        bundle[field] = [row for row in bundle[field] if row['query_id'] in retained]
    bundle['provenance'].update({
        'source_id': 'gapfill_acord_en_v3', 'excluded_official_train_query_ids': excluded,
        'split_policy': 'DCI seed42 dev10/test50 from official valid6 + test57 only. All51 official train queries excluded; no official ACORD score implied.',
    })
    return publish_bundle(bundle, output, parent_release=parent,
                          policy={'max_positive_per_query': 200, 'exclude_review_required': True})


def export_browse() -> dict:
    import pyarrow.parquet as pq
    from dci_bench.data.expansion import build_document_task, verify_document_task
    old = BASE / 'dci_expansion_20261014_v1'
    name = 'browsecomp_plus'
    md = old / 'metadata' / name
    # Reuse the sealed canonical corpus only after checking its complete
    # metadata, parquet and body inventory. Old artifacts are never rewritten.
    verified = verify_document_task(old, name, io_workers=16)
    provenance = read_json(md / 'provenance.json')
    provenance.update({
        'translation_status': 'native', 'language': 'en', 'domain': 'web',
        'source_id': 'gapfill_browsecomp_plus_en', 'parent_suite': str(old),
        'parent_completion_sha256': file_digest(md / '_COMPLETE.json'),
        'parent_audit': read_json(md / 'audit.json'),
        'candidate_scope': '99,930 original documents surviving the frozen 1MB compatibility limit',
        'review_date': '2026-10-31',
        'review_sources': ['https://github.com/texttron/BrowseComp-Plus',
                           'https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus'],
    })
    corpus = (row for batch in pq.ParquetFile(old / 'canonical' / name / 'corpus/corpus.parquet').iter_batches(batch_size=1024)
              for row in batch.to_pylist())
    task = {
        'name': 'gapfill_browsecomp_plus_en', 'dataset': verified['dataset'],
        'revision': verified['revision'], 'track': 'document',
        'task_instruction': 'Retrieve documents providing evidence needed to answer the web research query.',
        'corpus': corpus,
        'queries': [json.loads(line) for line in (md / 'queries.jsonl').read_text().splitlines()],
        'qrels': read_json(md / 'qrels.json'), 'host_labels': read_json(md / 'host_labels.json'),
        'provenance': provenance,
    }
    build_document_task(task, RUN / 'source', read_json(md / 'source_manifest.json'), io_workers=16)
    return verify_document_task(RUN / 'source', task['name'], io_workers=16)


def filter_query_self_matches(raw: Path, output: Path, mapping: Path,
                                *, self_documents: dict | None = None, qrels: dict | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite filtered ranking: {output}')
    payload = read_json(raw)
    original_to_alias = {original: alias for alias, original in read_json(mapping).items()}
    removed = {}
    for qid, ranking in payload['rankings'].items():
        original = self_documents[qid] if self_documents is not None else qid
        alias = original_to_alias.get(original)
        if qrels is not None and alias in qrels[qid]:
            raise ValueError(f'Cannot remove an annotated positive: {qid}/{alias}')
        # Saved retrieval uses original source IDs; pool metadata uses aliases.
        if original in ranking:
            removed[qid] = original
        payload['rankings'][qid] = [did for did in ranking if did != original][:1000]
        if len(payload['rankings'][qid]) < 1000:
            raise ValueError(f'Insufficient ranking after self exclusion: {qid}')
        payload['scores'][qid].pop(original, None)
    payload['ranking_filter'] = {
        'policy': 'Query claim self-match exclusion, then top1000',
        'raw_ranking': str(raw), 'raw_ranking_sha256': file_digest(raw),
        'removed': removed, 'removed_count': len(removed),
    }
    write_json(output, payload)
    return payload['ranking_filter']


def rank_and_pool(source: str, batch_size: int) -> dict:
    from dci_bench.data.pipeline.stages import rank_source
    from dci_bench.data.document_pools import prepare_document_pools, verify_document_pool_suite
    directory = RUN / source
    directory.mkdir(parents=True, exist_ok=True)
    exclude_self = source in {'gapfill_arguana_en', 'gapfill_cedar_zh_v2'}
    ranking = directory / ('ranking-no-self-v2.json' if exclude_self else 'ranking.json')
    receipt = directory / ('catalog-row-v2.json' if exclude_self else 'catalog-row.json')
    if receipt.exists():
        row = read_json(receipt)
        if file_digest(ranking) != row['ranking_sha256']:
            raise ValueError('Completed source ranking changed')
        for pool in row['pools']:
            verify_document_pool_suite(Path(pool['path']), task=source)
        return row
    raw = directory / 'ranking-raw.json' if exclude_self else ranking
    start = time.monotonic()
    if not raw.exists():
        print(f'{source}: encoding/ranking start', flush=True)
        rank_source(raw, suite=RUN / 'source', task=source, cache_dir=directory / 'embedding-cache',
                    device='cuda:0', batch_size=batch_size, max_length=8192,
                    top_k=1001 if exclude_self else 1000,
                    cache_shard_size=1024, search_device='cuda:0', search_query_batch_size=32)
    if exclude_self and not ranking.exists():
        md = RUN / 'source/metadata' / source
        self_documents = None
        if source == 'gapfill_cedar_zh_v2':
            metadata = read_json(md / 'host_labels.json')['query_metadata']
            self_documents = {qid: row['self_match_document_ids'][0] for qid, row in metadata.items()}
        filter_query_self_matches(raw, ranking, md / 'source_doc_ids.json',
                                    self_documents=self_documents, qrels=read_json(md / 'qrels.json'))
    pools = []
    for capacity in (200, 1000):
        print(f'{source}: pool {capacity}', flush=True)
        legacy = directory / 'pools'
        output = (directory / f'pools-{capacity}-v2' if exclude_self else
                  legacy if capacity == 200 and legacy.exists() else directory / f'pools-{capacity}')
        if not output.exists():
            prepare_document_pools(RUN / 'source', output, source,
                                   capacity=capacity, seed=42, rankings_path=ranking)
        verified = verify_document_pool_suite(output, task=source)
        if verified['capacity'] != capacity:
            raise ValueError('Existing pool capacity differs from requested capacity')
        pools.append({'capacity': capacity, 'path': str(output),
                      'completion_sha256': file_digest(output / 'metadata' / source / '_COMPLETE.json')})
    manifest = read_json(RUN / 'source/metadata' / source / 'manifest.json')
    row = {'source': source, 'task': source, 'documents': manifest['num_docs'],
           'queries': manifest['num_queries'], 'export': str(RUN / 'source'),
           'ranking': str(ranking), 'ranking_sha256': file_digest(ranking), 'pools': pools,
           'content_hashes_verified': True, 'elapsed_seconds': time.monotonic() - start}
    write_json(receipt, row)
    return row


def build_catalog() -> dict:
    if (RUN / 'catalog.json').exists():
        raise FileExistsError(f'Refusing to overwrite catalog: {RUN / "catalog.json"}')
    old = BASE / 'retrieval_pools/catalog.json'
    catalog = read_json(old)
    existing = {row['source'] for row in catalog['sources']}
    for source, release in RELEASES.items():
        receipt_name = 'catalog-row-v2.json' if source in {'gapfill_arguana_en', 'gapfill_cedar_zh_v2'} else 'catalog-row.json'
        row = read_json(RUN / source / receipt_name)
        if source in existing:
            raise ValueError(f'Source already present: {source}')
        md = RUN / 'source/metadata' / source
        if source == 'gapfill_browsecomp_plus_en':
            marker = release / 'metadata/browsecomp_plus/_COMPLETE.json'
            if file_digest(marker) != read_json(md / 'provenance.json')['parent_completion_sha256']:
                raise ValueError('BrowseComp+ parent completion differs from the verified export')
        else:
            verified = verify_release(release)
            if verified['state'] != 'labels_ready' or verified['bundle_hash'] != read_json(md / 'source_manifest.json')['source_fingerprint']:
                raise ValueError(f'Release identity differs from exported source: {source}')
            marker = release / '_COMPLETE.json'
        row.update(release=str(release), release_completion_sha256=file_digest(marker),
                   status='complete', source_files_present=True)
        for pool in row['pools']:
            marker = Path(pool['path']) / 'metadata' / source / '_COMPLETE.json'
            if file_digest(marker) != pool['completion_sha256']:
                raise ValueError(f'Pool completion changed: {source}')
        catalog['sources'].append(row)
    catalog.update(source_count=len(catalog['sources']),
                   pool_count=sum(len(row['pools']) for row in catalog['sources']),
                   queries=sum(row['queries'] for row in catalog['sources']),
                   parent_catalog=str(old), parent_catalog_sha256=file_digest(old))
    write_json(RUN / 'catalog.json', catalog)
    return {key: value for key, value in catalog.items() if key != 'sources'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['derive', 'approve-cleaned', 'derive-acord', 'derive-acord-heldout', 'derive-cedar', 'browse', 'filter-arguana', 'rank-pools', 'catalog'])
    parser.add_argument('--source')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.action == 'derive':
        result = derive(args.source)
    elif args.action == 'approve-cleaned':
        result = approve_cleaned(args.source)
    elif args.action == 'derive-acord':
        result = derive_acord()
    elif args.action == 'derive-cedar':
        result = derive_cedar()
    elif args.action == 'derive-acord-heldout':
        result = derive_acord_heldout()
    elif args.action == 'browse':
        result = export_browse()
    elif args.action == 'rank-pools':
        result = rank_and_pool(args.source, args.batch_size)
    elif args.action == 'catalog':
        result = build_catalog()
    else:
        source = 'gapfill_arguana_en'
        result = filter_query_self_matches(
            RUN / source / 'ranking-raw.json', RUN / source / 'ranking-no-self-v2.json',
            RUN / 'source/metadata' / source / 'source_doc_ids.json')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
