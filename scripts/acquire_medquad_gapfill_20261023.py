#!/usr/bin/env python3
"""Build answer retrieval from the published, pinned MedQuAD XML collection.

The source has no official train/test split. This is a development retrieval
adaptation, not a reproduction of TREC LiveQA or an unseen-query test set.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dci_bench.data.pipeline.core import build_source, load_bundle, verify_release
from dci_bench.data.pipeline.download import download_url

BASE = Path('/mnt/afs2/202609/embedding_models/dci-bench')
SOURCE = 'gapfill_medquad_rawqa'
REVISION = '577bd37b96c02d1833b2c9eed2de9f96964e96cb'
RAW = BASE / 'raw_data' / SOURCE
RELEASE = BASE / 'eval_data/pipeline_v1' / SOURCE / 'v1'
RESEARCH = BASE / 'eval_data/source_research_20261023_round3/medquad'
ARCHIVE = RAW / 'source_inputs/MedQuAD-577bd37.zip'
ARCHIVE_SHA = '161d948f8ce8f82accca1f5769a512dff2a01af2b3004d48f2b3da2758a38a09'
ARCHIVE_BYTES = 16953181
URL = f'https://codeload.github.com/abachaa/MedQuAD/zip/{REVISION}'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Existing artifact differs: {path}')
    else:
        with path.open('xb') as stream:
            stream.write(data)


def json_once(path, value):
    once(path, (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + '\n').encode())


def get_tree():
    path = RAW / 'source_inputs/github_tree.json'
    if not path.exists():
        request = urllib.request.Request(
            f'https://api.github.com/repos/abachaa/MedQuAD/git/trees/{REVISION}?recursive=1',
            headers={'User-Agent': 'dci-bench-data-pipeline'})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=20) as response:
            data = response.read(6000000)
        once(path, data)
    tree = json.loads(path.read_bytes())
    if tree.get('truncated'):
        raise ValueError('Incomplete Git tree')
    return tree


def convert(tree):
    files = {r['path']: r for r in tree['tree'] if r['type'] == 'blob'}
    documents, queries, qrels, exclusions = {}, [], {}, []
    counts = Counter()
    source_counts = Counter()
    original_ids = set()
    verified_files = set()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in sorted(archive.namelist()):
            if member.endswith('/'):
                continue
            relative = member.split('/', 1)[1]
            content = archive.read(member)  # Also checks ZIP CRC.
            expected = files[relative]
            blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            if blob != expected['sha'] or len(content) != expected['size']:
                raise ValueError(f'Git blob mismatch: {relative}')
            verified_files.add(relative)
            if relative in {'readme.txt', 'LICENSE.txt'}:
                once(RESEARCH / relative, content)
            if not relative.endswith('.xml'):
                continue
            counts['xml_files'] += 1
            root = ET.fromstring(content)
            for pair in root.findall('.//QAPair'):
                counts['raw_qa_pairs'] += 1
                question = pair.find('Question')
                answer = pair.find('Answer')
                source_id = question.attrib.get('qid', '') if question is not None else ''
                qid = f'{relative}#{source_id or pair.attrib.get("pid", "")}'
                if qid in original_ids:
                    raise ValueError(f'Duplicate source query ID: {qid}')
                original_ids.add(qid)
                question_text = ''.join(question.itertext()) if question is not None else ''
                answer_text = ''.join(answer.itertext()) if answer is not None else ''
                if not question_text.strip() or not answer_text.strip():
                    exclusions.append({'query_id': qid, 'original_query_id': source_id,
                                       'source_file': relative, 'reason': 'missing_published_answer' if not answer_text.strip() else 'empty_question'})
                    continue
                counts['answerable_qa_pairs'] += 1
                source_counts[relative.split('/')[0]] += 1
                doc_id = 'medquad-answer-' + sha(answer_text.encode())
                alias = {'source_file': relative, 'source_document_id': root.attrib.get('id'),
                         'original_query_id': source_id, 'pair_id': pair.attrib.get('pid'),
                         'source_url': root.attrib.get('url'), 'source': root.attrib.get('source')}
                if doc_id not in documents:
                    documents[doc_id] = {'_id': doc_id, 'title': '', 'text': answer_text, 'language': 'en',
                                         'metadata': {'source_family': 'MedQuAD', 'source_aliases': []}}
                elif documents[doc_id]['text'] != answer_text:
                    raise ValueError('Answer hash collision')
                documents[doc_id]['metadata']['source_aliases'].append(alias)
                queries.append({'_id': qid, 'text': question_text, 'language': 'en',
                                'metadata': {**alias, 'source_family': 'MedQuAD',
                                             'translation_status': 'non_translated',
                                             'source_split': 'original_unsplit',
                                             'question_construction': 'published MedQuAD question; source collection uses question types/templates',
                                             'label_basis': 'published XML question-answer association'}})
                qrels[qid] = {doc_id: 1}
    if verified_files != set(files):
        raise ValueError('Archive does not contain the complete pinned repository')
    counts['unique_exact_answer_documents'] = len(documents)
    counts['duplicate_answer_rows_aliased'] = counts['answerable_qa_pairs'] - len(documents)
    counts['excluded_qa_pairs'] = len(exclusions)
    return list(documents.values()), queries, qrels, {
        'counts': dict(counts), 'source_answerable_counts': dict(source_counts), 'exclusions': exclusions,
        'verified_git_blobs': len(verified_files),
        'deduplication': 'Exact decoded answer text only; original XML IDs/URLs/positions retained in aliases; no normalization or semantic merging.',
        'readme_count_reconciliation': 'README reports 47457 QA pairs; pinned XML contains 47441. Use direct XML count.',
        'limitations': ['No official held-out split; this is a development retrieval adaptation.',
                       'Labels identify the published source answer, not exhaustive corpus relevance.',
                       'Source questions can be templated; this is not a natural-user-query-only corpus.',
                       'Missing published answers remain in raw XML and explicit exclusions; no recrawling or synthetic answers.'],
    }


def main():
    if not ARCHIVE.exists():
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        download_url(URL, ARCHIVE, timeout=20, retries=1, max_bytes=60000000,
                     expected_sha256=ARCHIVE_SHA, expected_bytes=ARCHIVE_BYTES, overwrite=False)
    data = ARCHIVE.read_bytes()
    if len(data) != ARCHIVE_BYTES or sha(data) != ARCHIVE_SHA:
        raise ValueError('Archive integrity mismatch')
    docs, queries, qrels, conversion = convert(get_tree())
    canonical = RAW / 'canonical_input'
    for role, rows in [('documents', docs), ('queries', queries)]:
        once(canonical / f'{role}.jsonl', ''.join(json.dumps(r, ensure_ascii=True, sort_keys=True) + '\n' for r in rows).encode())
    json_once(canonical / 'qrels.json', qrels)
    json_once(RAW / 'conversion_manifest.json', conversion)
    spec = {'id': SOURCE, 'connector': 'generic_jsonl', 'input_kind': 'evidence_qa',
            'domain': 'medical', 'language': 'en', 'revision': REVISION, 'license': 'CC-BY-4.0; source terms in LICENSE.txt',
            'metadata': {'source_family': 'MedQuAD', 'source_scope': 'Complete published nonempty answers from all 12 XML collections; exact-answer corpus; original unsplit QA queries; development adaptation, no held-out benchmark claim.',
                         'translation_status': 'non_translated', 'query_split': 'original_unsplit',
                         'official_urls': ['https://github.com/abachaa/MedQuAD'],
                         'document_granularity': 'published medical answer, title omitted'},
            'options': {'dataset': 'abachaa/MedQuAD', 'files': {
                role: {'name': f'{role}.{extension}', 'path': str(canonical / f'{role}.{extension}')}
                for role, extension in [('documents', 'jsonl'), ('queries', 'jsonl'), ('qrels', 'json')]}}}
    if not RELEASE.exists():
        build_source(spec, BASE / 'raw_data', RELEASE,
                     policy={'max_positive_per_query': None, 'exclude_review_required': False, 'drop_near_duplicates': False})
    manifest = verify_release(RELEASE)
    bundle = load_bundle(RELEASE, verify=False)
    assert {r['_id']: r['text'] for r in bundle['documents']} == {r['_id']: r['text'] for r in docs}
    assert {r['_id']: r['text'] for r in bundle['queries']} == {r['_id']: r['text'] for r in queries}
    assert bundle['qrels'] == qrels
    receipt = {'source': SOURCE, 'revision': REVISION, 'raw_archive': str(ARCHIVE),
               'archive_url': URL, 'archive_sha256': ARCHIVE_SHA, 'archive_bytes': ARCHIVE_BYTES,
               'conversion': str(RAW / 'conversion_manifest.json'),
               'counts': conversion['counts'], 'limitations': conversion['limitations'],
               'release_path': str(RELEASE), 'release_state': manifest['state'],
               'release_bundle_hash': manifest['bundle_hash'],
               'checks': ['All repository files match pinned Git blob IDs and ZIP CRC',
                          'All retained query texts and answer texts equal parsed original XML',
                          'All question-to-answer qrels equal original source associations',
                          'Release inventory and bundle hash verified'],
               'official_urls': ['https://github.com/abachaa/MedQuAD']}
    json_once(RESEARCH / 'research.json', {'sources': [receipt]})
    print(json.dumps({k: receipt[k] for k in ['source', 'counts', 'release_state', 'release_path']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
