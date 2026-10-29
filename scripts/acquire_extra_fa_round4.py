#!/usr/bin/env python3
"""Acquire pinned Persian BEIR translations and preserve source qrel grades."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.acquire_rawqa_round4 import BASE, POLICY, json_once, once
from dci_bench.data.pipeline.core import build_source, load_bundle, verify_release
from dci_bench.data.pipeline.download import download_url

RESEARCH = BASE / 'eval_data/source_research_20261023_round4/extra_fa'


def records(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def acquire(source):
    raw = BASE / 'raw_data' / source['id']
    inputs = raw / 'source_inputs'
    for item in source['files']:
        download_url(item['url'], inputs / item['path'], timeout=60, retries=1,
                     max_bytes=300000000, expected_bytes=item['size'],
                     expected_sha256=item['lfs']['oid'], opener=urllib.request.build_opener())
    corpus, all_queries, raw_qrels = (records(inputs / name) for name in ['corpus.jsonl', 'queries.jsonl', 'qrels/test.jsonl'])
    raw_scores, grade_counts = {}, Counter()
    duplicate_pairs = []
    for number, row in enumerate(raw_qrels, 1):
        qid, did = str(row['query-id']), str(row['corpus-id'])
        grade = float(row['score'])
        if not math.isfinite(grade):
            raise ValueError('Nonfinite source qrel grade')
        grade = int(grade) if grade.is_integer() else grade
        grade_counts[str(grade)] += 1
        if (qid, did) in raw_scores:
            if raw_scores[qid, did] != grade:
                raise ValueError(f'Conflicting original qrel: {qid}/{did}')
            duplicate_pairs.append({'row': number, 'query_id': qid, 'doc_id': did, 'grade': grade})
        raw_scores[qid, did] = grade
    judged_doc_ids = {did for _,did in raw_scores}
    docs, aliases, by_content, exclusions = {}, {}, {}, []
    for row in sorted(corpus, key=lambda r: str(r['_id'])):
        did, title, text = str(row['_id']), row.get('title') or '', row['text']
        if did in aliases:
            raise ValueError(f'Duplicate original document ID: {did}')
        if not isinstance(text, str) or not text.strip():
            if did in judged_doc_ids:
                raise ValueError(f'Judged source document lacks text: {did}')
            exclusions.append({'doc_id': did, 'reason': 'empty_unjudged_document'})
            continue
        if not isinstance(title, str):
            raise ValueError(f'Invalid source title: {did}')
        key = (title, text, 'fa')
        canonical = by_content.setdefault(key, did)
        aliases[did] = canonical
        if canonical not in docs:
            docs[canonical] = {'_id': canonical, 'title': title, 'text': text, 'language': 'fa',
                               'metadata': {'source_family': source['adapted_from'][0], 'source_document_ids': []}}
        docs[canonical]['metadata']['source_document_ids'].append(did)
    mapped_scores = {}
    for (qid, did), grade in raw_scores.items():
        if did not in aliases:
            raise ValueError(f'Qrel references missing document: {qid}/{did}')
        key = (qid, aliases[did])
        if key in mapped_scores and mapped_scores[key] != grade:
            raise ValueError(f'Exact alias would merge different qrel grades: {key}')
        mapped_scores[key] = grade
    qrels = {}
    for (qid,did), grade in mapped_scores.items():
        if grade > 0:
            qrels.setdefault(qid, {})[did] = grade
    qids = {qid for qid,_ in raw_scores}
    if set(qrels) != qids:
        raise ValueError('Test queries without positive qrels require explicit review')
    queries, all_query_ids = [], set()
    for row in all_queries:
        qid = str(row['_id'])
        if qid in all_query_ids:
            raise ValueError(f'Duplicate source query ID: {qid}')
        all_query_ids.add(qid)
        if qid not in qids:
            continue
        if not isinstance(row['text'], str) or not row['text'].strip():
            raise ValueError(f'Invalid test query: {qid}')
        queries.append({'_id': qid, 'text': row['text'], 'language': 'fa',
                        'metadata': {'translation_status': 'translated', 'source_split': 'test',
                                     'source_family': source['adapted_from'][0], 'annotation_source': 'source'}})
    assert {q['_id'] for q in queries} == qids
    canonical = raw / 'canonical_input'
    for name, rows in [('documents', list(docs.values())), ('queries', queries)]:
        once(canonical / f'{name}.jsonl', ''.join(json.dumps(r, ensure_ascii=True, sort_keys=True) + '\n' for r in rows).encode())
    json_once(canonical / 'qrels.json', qrels)
    json_once(raw / 'source_manifest.json', source)
    conversion = {'counts': {'original_documents': len(corpus), 'documents': len(docs),
                  'original_queries_all_splits': len(all_queries), 'queries': len(queries),
                  'raw_test_qrel_rows': len(raw_qrels), 'positive_pairs': sum(map(len,qrels.values()))},
                  'original_qrel_grades': dict(grade_counts), 'duplicate_same_grade_pairs': duplicate_pairs,
                  'document_aliases': aliases, 'empty_unjudged_document_exclusions': exclusions,
                  'query_selection': 'All and only original test qrel query IDs; other source queries stay in raw file.',
                  'nonpositive_qrels': [{'query_id': q, 'doc_id': d, 'grade': g} for (q,d),g in raw_scores.items() if g <= 0],
                  'deduplication': 'Exact title/text/language only; canonical is lexicographically smallest source ID; conflicting mapped grades fail closed.'}
    json_once(raw / 'conversion_manifest.json', conversion)
    scope = f'Complete published {source["name"]} test retrieval view. Exact title/text/language duplicates are represented by original-ID aliases; empty unjudged documents are explicitly audited. No local translation.'
    spec = {'id': source['id'], 'connector': 'generic_jsonl', 'input_kind': 'retrieval',
            'domain': source['domain'], 'language': 'fa', 'revision': source['dataset']['revision'],
            'license': source['license'], 'metadata': {'source_family': source['adapted_from'][0],
                'translation_status': 'translated', 'query_split': 'test', 'source_scope': scope,
                'official_urls': [source['reference'], 'https://arxiv.org/abs/2502.11571'],
                'raw_conversion_manifest': str(raw / 'conversion_manifest.json')},
            'options': {'dataset': source['dataset']['path'], 'files': {name: {'name': f'{name}.{ext}', 'path': str(canonical / f'{name}.{ext}')}
                 for name,ext in [('documents','jsonl'), ('queries','jsonl'), ('qrels','json')]}}}
    release = BASE / 'eval_data/pipeline_v1' / source['id'] / 'v1'
    if not release.exists():
        build_source(spec, BASE / 'raw_data', release, policy=POLICY)
    manifest = verify_release(release)
    bundle = load_bundle(release, verify=False)
    assert {r['_id']: (r['title'],r['text']) for r in bundle['documents']} == {k:(r['title'],r['text']) for k,r in docs.items()}
    assert {r['_id']: r['text'] for r in bundle['queries']} == {r['_id']:r['text'] for r in queries}
    assert bundle['qrels'] == qrels
    result = {'id': source['id'], 'domain': source['domain'], 'language': 'fa',
              'source_family': source['adapted_from'][0], 'translation_status': 'translated',
              'dataset': source['dataset'], 'release_path': str(release), 'release_state': manifest['state'],
              'counts': conversion['counts'], 'qrel_grades': dict(grade_counts),
              'aliased_document_ids': sum(k != v for k,v in aliases.items()),
              'empty_unjudged_documents': len(exclusions), 'official_urls': spec['metadata']['official_urls'],
              'checks': ['All raw bytes match pinned HF LFS SHA256 and size',
                         'All source qrel numeric grades retained through exact aliases; conflicting grades fail closed',
                         'All original test qrel queries retained; raw other-split queries preserved',
                         'Every canonical document/query text and qrel ID/grade compared to release',
                         'Release inventory and bundle hash verified']}
    json_once(RESEARCH / (source['id'] + '.research.json'), result)
    return result


def main():
    config = json.loads((ROOT / 'configs/data_pipeline/extra_fa_round4.json').read_text())
    results, failures = [], []
    for source in config['sources']:
        try:
            result = acquire(source)
        except Exception as exc:
            failure = {'id': source['id'], 'error_type': type(exc).__name__, 'error': str(exc), 'status': 'failed_preserved'}
            failures.append(failure)
            print(json.dumps(failure), flush=True)
        else:
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    json_once(RESEARCH / 'research.json', {'sources': results, 'failures': failures,
              'acquired_verified': [r['id'] for r in results], 'gpu_or_cascade': 'not executed'})


if __name__ == '__main__':
    main()
