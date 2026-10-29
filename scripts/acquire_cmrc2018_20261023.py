#!/usr/bin/env python3
"""Pin CMRC2018, retain dev questions, and use its full train+dev passage corpus."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dci_bench.data.pipeline.connectors import fetch_source, _base_bundle
from dci_bench.data.pipeline.core import publish_bundle, verify_release

BASE = Path('/mnt/afs2/202609/embedding_models/dci-bench')
REVISION = 'c0eb1b6ba219847457e6af3180da722bbeb656af'
FILES = {
    'cmrc2018_train.json': (7408757, 'cd56f7d38f6f4a6f92df2616479bbf134a37bebf'),
    'cmrc2018_dev.json': (3299139, '94c1f2df14a669b0b7b836665d6339a34df8aa97'),
}
SPEC = {
    'id': 'qa_cmrc2018_zh', 'connector': 'squad', 'input_kind': 'evidence_qa',
    'domain': 'encyclopedia', 'language': 'zh', 'revision': REVISION,
    'license': 'CC-BY-SA-4.0; see ymcui/cmrc2018 LICENCE',
    'options': {
        'dataset': 'ymcui/cmrc2018', 'translation_status': 'native',
        'genre': 'wikipedia_passage', 'max_bytes': 12000000, 'timeout': 20, 'retries': 1,
        'headers': {'Accept': 'application/vnd.github.raw+json', 'User-Agent': 'dci-bench-data-pipeline'},
        'files': [{'url': f'https://api.github.com/repos/ymcui/cmrc2018/contents/squad-style-data/{name}?ref={REVISION}',
                   'name': f'data/{name}', 'bytes': size} for name, (size, _) in FILES.items()],
    },
    'metadata': {'source_scope': 'Full train+dev corpus; dev questions only; QA-derived passage retrieval, not the original reading-comprehension leaderboard.',
                 'query_split': 'dev', 'corpus_splits': ['train', 'dev'],
                 'granularity': 'Chinese Wikipedia paragraph',
                 'official_url': 'https://github.com/ymcui/cmrc2018',
                 'expected_git_blobs': {name: blob for name, (_, blob) in FILES.items()}},
}


def main():
    out = BASE / 'eval_data/pipeline_v1/qa_cmrc2018_zh/expansion-v1'
    receipt = BASE / 'eval_data/source_research_20261023/cmrc2018/research.json'
    if out.exists() or receipt.exists():
        raise FileExistsError('Preserve prior release; verify it or choose a new version explicitly.')
    snapshot = fetch_source(SPEC, BASE / 'raw_data')
    for name, (size, blob) in FILES.items():
        content = (snapshot / 'data' / name).read_bytes()
        actual = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        if len(content) != size or actual != blob:
            raise ValueError(f'Pinned Git blob mismatch: {name}')
    bundle = _base_bundle(SPEC, json.loads((snapshot / 'source_manifest.json').read_text()))
    ignored_spans = []
    ids, document_ids = set(), set()
    for filename in FILES:
        split = 'dev' if '_dev.' in filename else 'train'
        original = json.loads((snapshot / 'data' / filename).read_text())
        for article in original['data']:
            for paragraph in article['paragraphs']:
                did, context = paragraph['id'], paragraph['context']
                if did in document_ids:
                    raise ValueError(f'Duplicate original paragraph ID: {did}')
                document_ids.add(did)
                bundle['documents'].append({'_id': did, 'title': article['title'], 'text': context, 'language': 'zh',
                                            'metadata': {'original_split': split, 'source_article_id': article['id'],
                                                         'source_file': filename, 'translation_status': 'native'}})
                if split != 'dev':
                    continue
                for q in paragraph['qas']:
                    qid = q['id']
                    if qid in ids:
                        raise ValueError(f'Duplicate original dev query ID: {qid}')
                    ids.add(qid)
                    bundle['queries'].append({'_id': qid, 'text': q['question'], 'language': 'zh',
                                               'metadata': {'annotation_source': 'source_human', 'original_split': 'dev',
                                                            'translation_status': 'native', 'source_paragraph_id': did}})
                    # Retrieval supervision is the original QA-to-paragraph link.
                    # An invalid optional answer offset must not erase that link.
                    bundle['qrels'][qid] = {did: 1}
                    for i, answer in enumerate(q.get('answers', [])):
                        start, quote = answer.get('answer_start'), answer.get('text')
                        if (not isinstance(start, int) or start < 0 or not isinstance(quote, str)
                                or not quote or context[start:start + len(quote)] != quote):
                            ignored_spans.append({'query_id': qid, 'answer_index': i,
                                                  'reason': 'invalid_original_optional_answer_offset', 'original_answer': answer})
                            continue
                        evidence = {'query_id': qid, 'doc_id': did, 'quote': quote, 'start': start,
                                    'end': start + len(quote), 'annotation_source': 'source_human', 'offset_unit': 'unicode_codepoints'}
                        if evidence not in bundle['evidence']:
                            bundle['evidence'].append(evidence)
    bundle['provenance'].update({'query_split': 'dev', 'corpus_splits': ['train', 'dev'],
                                 'query_selection': 'original dev IDs; no train queries', 'original_dev_queries': len(ids),
                                 'qrels_derivation': 'Original dev question-to-paragraph association; no generated labels.',
                                 'optional_evidence_policy': 'Keep exact original offsets only; invalid offsets are retained in audit provenance, never repaired.',
                                 'ignored_optional_spans': ignored_spans})
    manifest = publish_bundle(bundle, out, specification=SPEC, raw_snapshot=snapshot,
                              policy={'max_positive_per_query': None, 'exclude_review_required': False})
    verify_release(out)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = {'sources': [{'id': SPEC['id'], 'acquisition_status': 'published_verified', 'release_path': str(out),
                            'raw_snapshot': str(snapshot), 'revision': REVISION, 'domain': 'encyclopedia', 'language': 'zh',
                            'urls': ['https://github.com/ymcui/cmrc2018'], 'granularity': 'passage',
                            'counts': {k: manifest[k] for k in ['num_documents', 'num_queries', 'num_positive_pairs', 'state']},
                            'limitations': ['QA-derived seed positives, not exhaustive qrels.',
                                            'Full train+dev corpus with dev queries; not the original reading-comprehension evaluation.']}],
               'command': f'python {ROOT / "scripts/acquire_cmrc2018_20261023.py"}',
               'git_blob_verified': True}
    receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
