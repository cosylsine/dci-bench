#!/usr/bin/env python3
"""Build explicit retrieval adaptations from pinned human QA/reranking data.

Raw inputs remain immutable. Exact text aliases retain every source location;
missing human answers are audited, and synthetic HC3 answers are outside scope.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dci_bench.data.pipeline.core import build_source, load_bundle, verify_release
from dci_bench.data.pipeline.download import download_url

BASE = Path('/mnt/afs2/202609/embedding_models/dci-bench')
RESEARCH = BASE / 'eval_data/source_research_20261023_round4/raw_qa'
POLICY = {'max_positive_per_query': None, 'exclude_review_required': False,
          'drop_near_duplicates': False, 'max_document_bytes': 16777216,
          'max_query_bytes': 16777216}


class SourceLabelConflict(ValueError):
    def __init__(self, conflicts):
        self.conflicts = conflicts
        super().__init__(f'{len(conflicts)} queries have contradictory source positive/negative labels')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Immutable artifact differs: {path}')
    else:
        with path.open('xb') as handle:
            handle.write(data)


def json_once(path, value):
    once(path, (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + '\n').encode())


def convert(rows, source):
    documents, queries, qrels = {}, [], {}
    counts, exclusions, conflicts = Counter(), [], []
    kind, filename, language = source['kind'], source['filename'], source['language']

    def document(text, row_number, role, answer_index):
        if not isinstance(text, str) or not text.strip():
            exclusions.append({'row': row_number, 'role': role, 'answer_index': answer_index,
                               'reason': 'missing_or_empty_text'})
            counts['invalid_answer_slots'] += 1
            return None
        doc_id = 'text-' + digest((language + '\0' + text).encode())
        alias = {'source_file': filename, 'source_row_1based': row_number,
                 'source_field': role, 'source_answer_index_0based': answer_index}
        if doc_id not in documents:
            documents[doc_id] = {'_id': doc_id, 'title': '', 'text': text, 'language': language,
                                 'metadata': {'source_family': source['family'], 'source_aliases': []}}
        assert documents[doc_id]['text'] == text
        documents[doc_id]['metadata']['source_aliases'].append(alias)
        counts['valid_document_occurrences'] += 1
        return doc_id

    for number, row in enumerate(rows, 1):
        counts['source_rows'] += 1
        qid = f'{filename}#row{number}'
        query = row['question'] if kind == 'hc3_human' else row['query']
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f'Invalid query: {qid}')
        positives, negatives = set(), set()
        roles = ['human_answers'] if kind == 'hc3_human' else ['positive', 'negative']
        if kind == 'hc3_human':
            counts['chatgpt_answer_slots_outside_human_only_scope'] += len(row['chatgpt_answers'])
        for role in roles:
            if not isinstance(row[role], list):
                raise ValueError(f'Invalid answer list: {qid}/{role}')
            for index, text in enumerate(row[role]):
                counts[role + '_occurrences'] += 1
                doc_id = document(text, number, role, index)
                if doc_id is not None:
                    (negatives if role == 'negative' else positives).add(doc_id)
        if positives & negatives:
            conflicts.append({'query_id': qid, 'document_ids': sorted(positives & negatives)})
        if not positives:
            exclusions.append({'query_id': qid, 'row': number, 'reason': 'no_published_valid_human_answer'})
            counts['queries_without_positive'] += 1
            continue
        queries.append({'_id': qid, 'text': query, 'language': language,
                        'metadata': {'source_family': source['family'], 'source_file': filename,
                                     'source_row_1based': number,
                                     'source_split': source['split'],
                                     'translation_status': 'non_translated',
                                     'annotation_source': 'source',
                                     'label_basis': source['label_basis']}})
        qrels[qid] = {doc_id: 1 for doc_id in sorted(positives)}
    if conflicts:
        raise SourceLabelConflict(conflicts)
    counts.update(documents=len(documents), queries=len(queries),
                  positive_pairs=sum(map(len, qrels.values())),
                  exact_duplicate_occurrences_aliased=counts['valid_document_occurrences'] - len(documents))
    return list(documents.values()), queries, qrels, {
        'counts': dict(counts), 'exclusions': exclusions, 'positive_negative_conflicts': conflicts,
        'deduplication': 'Exact decoded text plus language only; no normalization or semantic merging. Every source location retained.',
        'query_ids': 'Source has no IDs; deterministic filename and 1-based row number identify each query.',
        'label_basis': source['label_basis'], 'limitations': source['limitations'],
    }


def acquire(source):
    raw = BASE / 'raw_data' / source['id']
    path = raw / 'source_inputs' / source['filename']
    download_url(source['url'], path, timeout=45, retries=1, max_bytes=30000000,
                 expected_sha256=source['sha256'], expected_bytes=source['bytes'], overwrite=False)
    assert digest(path.read_bytes()) == source['sha256']
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else path.open(encoding='utf-8')) as handle:
        rows = [json.loads(line) for line in handle]
    docs, queries, qrels, conversion = convert(rows, source)
    if len(docs) <= 1000:
        raise ValueError('Source does not exceed 1000 distinct documents')
    canonical = raw / 'canonical_input'
    for name, records in [('documents', docs), ('queries', queries)]:
        once(canonical / f'{name}.jsonl', ''.join(json.dumps(row, ensure_ascii=True, sort_keys=True) + '\n' for row in records).encode())
    json_once(canonical / 'qrels.json', qrels)
    json_once(raw / 'source_manifest.json', {**source, 'local_path': str(path)})
    json_once(raw / 'conversion_manifest.json', conversion)
    spec = {'id': source['id'], 'connector': 'generic_jsonl',
            'input_kind': 'evidence_qa' if source['kind'] == 'hc3_human' else 'retrieval',
            'domain': source['domain'], 'language': source['language'],
            'revision': source['revision'], 'license': source['license'],
            'metadata': {'source_family': source['family'], 'source_scope': source['scope'],
                         'translation_status': 'non_translated', 'query_split': source['split'],
                         'official_urls': source['official_urls'], 'document_granularity': source['granularity'],
                         'task_origin': source['task_origin'], 'label_basis': source['label_basis'],
                         'raw_conversion_manifest': str(raw / 'conversion_manifest.json')},
            'options': {'dataset': source['dataset'], 'files': {
                name: {'name': f'{name}.{extension}', 'path': str(canonical / f'{name}.{extension}')}
                for name, extension in [('documents','jsonl'), ('queries','jsonl'), ('qrels','json')]}}}
    release = BASE / 'eval_data/pipeline_v1' / source['id'] / 'v1'
    if not release.exists():
        build_source(spec, BASE / 'raw_data', release, policy=POLICY)
    manifest = verify_release(release)
    bundle = load_bundle(release, verify=False)
    assert {r['_id']: (r['title'], r['text']) for r in bundle['documents']} == {r['_id']: (r['title'], r['text']) for r in docs}
    assert {r['_id']: r['text'] for r in bundle['queries']} == {r['_id']: r['text'] for r in queries}
    assert bundle['qrels'] == qrels
    # Reconstruct every relation independently from raw source rows and published aliases.
    text_ids = {r['text']: r['_id'] for r in bundle['documents']}
    for query in queries:
        row = rows[query['metadata']['source_row_1based'] - 1]
        answers = row['human_answers'] if source['kind'] == 'hc3_human' else row['positive']
        expected = {text_ids[text]: 1 for text in answers if isinstance(text, str) and text.strip()}
        assert bundle['qrels'][query['_id']] == expected
    receipt = {'id': source['id'], 'domain': source['domain'], 'language': source['language'],
               'source_family': source['family'], 'release_path': str(release),
               'release_state': manifest['state'], 'counts': conversion['counts'],
               'revision': source['revision'], 'source_url': source['url'],
               'official_urls': source['official_urls'], 'source_manifest': str(raw / 'source_manifest.json'),
               'conversion_manifest': str(raw / 'conversion_manifest.json'),
               'translation_status': 'non_translated', 'limitations': source['limitations'],
               'checks': ['Source bytes and SHA256 match pinned mirror tree', 'Gzip CRC checked where applicable',
                          'All published IDs/text/qrels match the complete scoped source',
                          'Every qrel independently reconstructed from raw positive/human answer text',
                          'Release inventory and bundle hash verified']}
    json_once(RESEARCH / (source['id'] + '.research.json'), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/data_pipeline/rawqa_round4.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    receipts, failures = [], []
    for source in config['sources']:
        try:
            receipt = acquire(source)
        except SourceLabelConflict as exc:
            failure = {'id': source['id'], 'status': 'blocked_source_label_conflicts',
                       'conflicts': exc.conflicts, 'message': str(exc),
                       'raw_preserved': True, 'release_published': False,
                       'source_url': source['url'], 'revision': source['revision']}
            json_once(RESEARCH / (source['id'] + '.failure.json'), failure)
            failures.append(failure)
            print(json.dumps({'id': source['id'], 'status': failure['status'], 'conflicting_queries': len(exc.conflicts)}), flush=True)
        else:
            receipts.append(receipt)
            print(json.dumps({key: receipt[key] for key in ['id', 'release_state', 'counts']}, ensure_ascii=False), flush=True)
    json_once(RESEARCH / 'research.json', {'sources': receipts, 'failures': failures,
              'acquired_verified': [r['id'] for r in receipts]})


if __name__ == '__main__':
    main()
