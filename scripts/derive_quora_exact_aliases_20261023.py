#!/usr/bin/env python3
"""Create an explicitly derived Quora view with exact duplicate documents aliased.

No question is removed. Each retained document is an original document with
unchanged title/text. Grades must agree when two judgments collapse to one
pair; conflicting grades fail closed. The official BEIR release is untouched.
"""
from __future__ import annotations

import copy
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dci_bench.data.pipeline.core import load_bundle, publish_bundle, verify_release

BASE = Path('/mnt/afs2/202609/embedding_models/dci-bench')
PARENT = BASE / 'eval_data/pipeline_v1/gapfill_beir_quora/v1'
SOURCE_ID = 'gapfill_beir_quora_exact_dedup'
OUTPUT = BASE / 'eval_data/pipeline_v1' / SOURCE_ID / 'v1'
RESEARCH = BASE / 'eval_data/source_research_20261023_round3/quora_aliases'


def exact_alias_view(documents, qrels):
    groups = defaultdict(list)
    for row in documents:
        groups[(row.get('title', ''), row['text'], row.get('language'))].append(row)
    aliases, kept, merged = {}, [], []
    for rows in groups.values():
        canonical = min(rows, key=lambda row: row['_id'])
        canonical_id = canonical['_id']
        for row in rows:
            aliases[row['_id']] = canonical_id
        result = dict(canonical)
        if len(rows) > 1:
            original_ids = sorted(row['_id'] for row in rows)
            merged.append({'canonical_id': canonical_id, 'original_document_ids': original_ids})
            result['metadata'] = {**canonical.get('metadata', {}), 'exact_document_aliases': original_ids}
        kept.append(result)
    remapped = {}
    for qid, judgments in qrels.items():
        values = {}
        for doc_id, grade in judgments.items():
            canonical_id = aliases[doc_id]
            if canonical_id in values and values[canonical_id] != grade:
                raise ValueError(f'Conflicting grades within exact aliases: {qid}, {canonical_id}')
            values[canonical_id] = grade
        remapped[qid] = values
    return kept, remapped, aliases, merged


def main():
    parent_manifest = verify_release(PARENT)
    source = load_bundle(PARENT, verify=False)
    documents, qrels, aliases, merged = exact_alias_view(source['documents'], source['qrels'])
    # Independently establish that every original positive survives with its grade.
    for qid, gold in source['qrels'].items():
        assert all(qrels[qid][aliases[doc_id]] == grade for doc_id, grade in gold.items())
    source_docs = {r['_id']: r for r in source['documents']}
    for doc in documents:
        assert (doc['title'], doc['text']) == (source_docs[doc['_id']]['title'], source_docs[doc['_id']]['text'])
    bundle = {**source, 'documents': documents, 'qrels': qrels}
    bundle['provenance'] = {**source.get('provenance', {}), 'source_id': SOURCE_ID,
                            'exact_alias_derivation': {'parent_release': str(PARENT),
                                'parent_bundle_hash': parent_manifest['bundle_hash'],
                                'key': ['title', 'text', 'language'], 'grade_policy': 'require equal grades, otherwise fail',
                                'merged_groups': merged}}
    spec = copy.deepcopy(parent_manifest['specification'])
    spec['id'] = SOURCE_ID
    spec['metadata'] = {**spec.get('metadata', {}), 'source_family': 'Quora',
                        'source_scope': 'Explicitly derived BEIR Quora view: exact title/text/language duplicates aliased; all official test queries retained. Not directly comparable to original BEIR metrics.',
                        'translation_status': 'native', 'parent_release': str(PARENT)}
    if not OUTPUT.exists():
        publish_bundle(bundle, OUTPUT, specification=spec, parent_release=PARENT,
                       policy={'max_positive_per_query': None, 'exclude_review_required': False,
                               'drop_near_duplicates': False})
    manifest = verify_release(OUTPUT)
    actual = load_bundle(OUTPUT, verify=False)
    assert actual['queries'] == source['queries']
    assert actual['qrels'] == qrels
    assert actual['documents'] == documents
    receipt = {'source': SOURCE_ID, 'parent_release': str(PARENT), 'release_path': str(OUTPUT),
               'parent_completion_sha256': hashlib.sha256((PARENT / '_COMPLETE.json').read_bytes()).hexdigest(),
               'state': manifest['state'], 'bundle_hash': manifest['bundle_hash'],
               'source_counts': {'documents': len(source['documents']), 'queries': len(source['queries']),
                                 'positive_pairs': sum(map(len, source['qrels'].values()))},
               'counts': {'documents': len(documents), 'queries': len(actual['queries']),
                          'positive_pairs': sum(map(len, qrels.values())), 'merged_document_groups': len(merged),
                          'aliased_duplicate_documents': len(source['documents']) - len(documents)},
               'merged_groups': merged,
               'checks': ['all original query records unchanged', 'every original positive survives under reversible ID aliases with its original grade',
                          'retained document titles/texts unchanged', 'no query exclusions or grade aggregation',
                          'parent and derived release inventories and bundle hashes verified'],
               'limitation': 'This changes the retrieval-unit identity/count; it is a derived DCI corpus, not official BEIR leaderboard reproduction.'}
    RESEARCH.mkdir(parents=True, exist_ok=True)
    path = RESEARCH / 'research.json'
    data = (json.dumps({'sources': [receipt]}, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == data
    else:
        with path.open('xb') as stream:
            stream.write(data)
    print(json.dumps({'source': SOURCE_ID, 'state': manifest['state'], 'counts': receipt['counts']}))


if __name__ == '__main__':
    main()
