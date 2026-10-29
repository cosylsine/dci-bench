#!/usr/bin/env python3
"""Adapt the complete official CSL title-summarization benchmark to retrieval.

The corpus contains train/dev/test abstracts; only original test titles become
queries. This title-to-abstract task is derived, not an existing CSL IR benchmark.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.acquire_rawqa_round4 import BASE, POLICY, RESEARCH, digest, json_once, once
from dci_bench.data.pipeline.core import build_source, load_bundle, verify_release
from dci_bench.data.pipeline.download import download_url

SOURCE = 'r4_rawqa_csl_zh'
REVISION = '0f1ac0626d584cd3c72f8be26d329eb4f967cfd0'
ARCHIVE_SHA = '5a68212df7183f2c8945dd71183fc4e1581e19d21718e563137b860c0758ffb7'
ARCHIVE_BYTES = 9116084
RAW = BASE / 'raw_data' / SOURCE
URL = f'https://codeload.github.com/ydli-ai/CSL/zip/{REVISION}'


def main():
    archive = RAW / 'source_inputs' / f'CSL-{REVISION}.zip'
    download_url(URL, archive, timeout=30, retries=1, max_bytes=30000000,
                 expected_sha256=ARCHIVE_SHA, expected_bytes=ARCHIVE_BYTES, overwrite=False)
    tree_path = RAW / 'source_inputs/github_tree.json'
    if not tree_path.exists():
        prior = RESEARCH / 'csl_github_tree.json'
        if prior.exists():
            once(tree_path, prior.read_bytes())
        else:
            url = f'https://api.github.com/repos/ydli-ai/CSL/git/trees/{REVISION}?recursive=1'
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=30) as response:
                once(tree_path, response.read())
    tree = json.loads(tree_path.read_text())
    assert tree['sha'] == REVISION and not tree['truncated']
    expected = {r['path']: r for r in tree['tree'] if r['type'] == 'blob'}
    verified, documents, queries, qrels = set(), {}, [], {}
    counts = Counter()
    with zipfile.ZipFile(archive) as z:
        for member in sorted(z.namelist()):
            if member.endswith('/'):
                continue
            relative = member.split('/', 1)[1]
            body = z.read(member)
            git_blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
            assert expected[relative]['sha'] == git_blob and expected[relative]['size'] == len(body)
            verified.add(relative)
            if not relative.startswith('benchmark/ts/') or not relative.endswith('.tsv'):
                continue
            once(RAW / 'source_inputs' / relative, body)
            split = Path(relative).stem
            for number, line in enumerate(body.decode('utf-8').split('\n'), 1):
                if not line:
                    # The final newline is not a source record.
                    assert number == len(body.decode('utf-8').split('\n'))
                    continue
                fields = line.split('\t')
                if len(fields) != 3 or fields[0] != 'to title':
                    raise ValueError(f'Unexpected CSL TSV record: {relative}:{number}')
                _, abstract, title = fields
                if not abstract.strip() or not title.strip():
                    raise ValueError(f'Empty CSL title/abstract: {relative}:{number}')
                doc_id = 'abstract-' + digest(abstract.encode())
                alias = {'source_file': relative, 'source_row_1based': number,
                         'original_title': title, 'source_split': split}
                if doc_id not in documents:
                    documents[doc_id] = {'_id': doc_id, 'title': '', 'text': abstract, 'language': 'zh',
                                         'metadata': {'source_family': 'CSL', 'source_aliases': []}}
                assert documents[doc_id]['text'] == abstract
                documents[doc_id]['metadata']['source_aliases'].append(alias)
                counts[split + '_records'] += 1
                if split == 'test':
                    qid = f'{relative}#row{number}'
                    queries.append({'_id': qid, 'text': title, 'language': 'zh',
                                    'metadata': {**alias, 'source_family': 'CSL',
                                                 'translation_status': 'non_translated',
                                                 'annotation_source': 'source',
                                                 'label_basis': 'Published title paired with its own abstract'}})
                    qrels[qid] = {doc_id: 1}
    assert verified == set(expected)
    assert dict(counts) == {'dev_records': 1000, 'test_records': 1000, 'train_records': 8000}
    counts['documents'] = len(documents)
    counts['queries'] = len(queries)
    counts['positive_pairs'] = sum(map(len, qrels.values()))
    counts['exact_abstract_duplicates_aliased'] = 10000 - len(documents)
    canonical = RAW / 'canonical_input'
    for name, rows in [('documents', list(documents.values())), ('queries', queries)]:
        once(canonical / f'{name}.jsonl', ''.join(json.dumps(r, ensure_ascii=True, sort_keys=True) + '\n' for r in rows).encode())
    json_once(canonical / 'qrels.json', qrels)
    limitations = ['Derived title-to-abstract retrieval; not the original CSL summarization metric or an official CSL IR benchmark.',
                   'Corpus is the complete 10000-record CSL benchmark, not the 396209-record full parent collection.',
                   'Queries are paper titles, not natural user questions. Title metadata is host-only; candidate documents contain abstract text only.',
                   'Only exact abstract duplicates are aliased; every source split/row/title is preserved.']
    json_once(RAW / 'conversion_manifest.json', {'counts': dict(counts), 'verified_git_blobs': len(verified),
               'exclusions': [], 'limitations': limitations})
    spec = {'id': SOURCE, 'connector': 'generic_jsonl', 'input_kind': 'retrieval',
            'domain': 'scientific', 'language': 'zh', 'revision': REVISION,
            'license': 'Not specified in the pinned CSL repository',
            'metadata': {'source_family': 'CSL', 'source_scope': 'Complete CSL 10k benchmark abstracts; original test titles as retrieval queries; derived title-to-abstract task.',
                         'translation_status': 'non_translated', 'query_split': 'test',
                         'task_origin': 'Title-to-abstract derived retrieval', 'document_granularity': 'scientific abstract',
                         'official_urls': ['https://github.com/ydli-ai/CSL', 'https://aclanthology.org/2022.coling-1.344/']},
            'options': {'dataset': 'ydli-ai/CSL', 'files': {name: {'name': f'{name}.{ext}', 'path': str(canonical / f'{name}.{ext}')}
                for name, ext in [('documents','jsonl'),('queries','jsonl'),('qrels','json')]}}}
    release = BASE / 'eval_data/pipeline_v1' / SOURCE / 'v1'
    if not release.exists():
        build_source(spec, BASE / 'raw_data', release, policy=POLICY)
    manifest = verify_release(release)
    bundle = load_bundle(release, verify=False)
    assert {r['_id']: r['text'] for r in bundle['documents']} == {k: r['text'] for k,r in documents.items()}
    assert {r['_id']: r['text'] for r in bundle['queries']} == {r['_id']: r['text'] for r in queries}
    assert bundle['qrels'] == qrels
    receipt = {'id': SOURCE, 'domain': 'scientific', 'language': 'zh', 'source_family': 'CSL',
               'source_url': URL, 'revision': REVISION, 'archive_sha256': ARCHIVE_SHA, 'archive_bytes': ARCHIVE_BYTES,
               'counts': dict(counts), 'release_path': str(release), 'release_state': manifest['state'],
               'limitations': limitations, 'checks': ['All archive files match pinned Git blob SHA1 and size; ZIP CRC verified',
                       'All 10000 benchmark title/abstract rows retained as corpus aliases',
                       'Only original 1000 test titles used as queries; qrels match source row associations',
                       'All document/query text and qrel IDs match parsed source', 'Release inventory and bundle hash verified']}
    json_once(RESEARCH / 'csl_research.json', {'sources': [receipt], 'acquired_verified': [SOURCE]})
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
