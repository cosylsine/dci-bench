import json
import base64
import hashlib
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from dci_bench.data.expansion_documents import (
    _BROWSE_CANARY, _browse_decrypt, _trec, browsecomp_tasks, enterprise_tasks, wixqa_tasks, worksurface_tasks,
)


class DocumentAdaptersTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def jsonl(self, name, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')

    def parquet(self, name, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)

    def test_wix_original_ids_hidden_labels_and_unknown_exclusion(self):
        docs = [dict(id='a.md', title='Source title', contents='public A', url='a', article_type='article'),
                dict(id='unused', contents='distractor', url='b', article_type='article')]
        self.jsonl('wix_kb_corpus/wix_kb_corpus.jsonl', docs)
        for config in ['expertwritten', 'simulated']:
            self.jsonl(f'wixqa_{config}/test.jsonl', [
                dict(question='question', answer='SECRET', article_ids=['a.md']),
                dict(question='invalid', answer='SECRET', article_ids=['a.md', 'missing'])])
        tasks = list(wixqa_tasks(self.root, 'fixed'))
        self.assertEqual(len(tasks), 2)
        task = tasks[0]
        self.assertEqual(task['queries'], [{'_id': 'expertwritten/0', 'text': 'question'}])
        self.assertEqual(task['qrels'], {'expertwritten/0': {'a.md': 1.0}})
        corpus = list(task['corpus'])
        self.assertEqual({r['_id'] for r in corpus}, {'a.md', 'unused'})
        self.assertNotIn('SECRET', json.dumps(corpus + task['queries']))
        self.assertEqual(task['host_labels']['expertwritten/0']['answer'], 'SECRET')
        self.assertEqual(task['provenance']['corpus_metadata_source']['fields'], ['url', 'article_type', 'html_content'])
        self.assertEqual(corpus[0]['title'], 'Source title')
        self.assertEqual(task['provenance']['exclusions'][0]['document_ids'], ['missing'])

    def test_browse_preserves_more_than_ten_positives_and_zero_padded_ids(self):
        self.parquet('data/train-00000-of-00001.parquet', [dict(docid=f'{i:03}', text='doc', url='url') for i in range(12)])
        (self.root / 'queries.tsv').write_text('007\tquery\n8\tbad gold\n', encoding='utf-8')
        (self.root / 'qrel_evidence.txt').write_text(''.join(f'007 Q0 {i:03} 1\n' for i in range(11)) + '8 Q0 absent 1\n9 Q0 001 1\n', encoding='utf-8')
        (self.root / 'qrel_golds.txt').write_text('007 Q0 000 1\n', encoding='utf-8')
        task = next(browsecomp_tasks(self.root, 'fixed', self.root))
        self.assertEqual(len(task['qrels']['007']), 11)
        self.assertEqual(len(list(task['corpus'])), 12)
        self.assertIn('000', task['qrels']['007'])
        self.assertEqual(task['host_labels']['007']['gold_qrels'], {'000': 1.0})
        self.assertEqual({e['reason'] for e in task['provenance']['exclusions']}, {'unknown_gold_ids', 'qrels_without_query'})

    def test_browse_official_decryption_known_vector(self):
        self.assertEqual(_browse_decrypt('1Oi/9uZsDQn7RoTUoCKkL4Nb8rvOrj8=', 'test'), 'Known plaintext: 你好')
        self.assertEqual(_browse_decrypt('', 'test'), '')
        with self.assertRaises(ValueError):
            _browse_decrypt('not base64!')

    def test_browse_native_hf_projection_preserves_ids_and_hides_answers(self):
        def encrypted(text):
            key = hashlib.sha256(_BROWSE_CANARY.encode()).digest()
            data = text.encode()
            return base64.b64encode(bytes(value ^ key[i % 32] for i, value in enumerate(data))).decode()
        self.parquet('corpus/data/train-00000-of-00001.parquet', [dict(docid='0007.md', text='public', url='url')])
        self.parquet('questions/data/test-00000-of-00001.parquet', [
            dict(query_id='009', query=encrypted('Question with\t原始 text'), answer=encrypted('SECRET'),
                 gold_docs=[dict(docid='wrong_nested_id', text='MUST NOT LOAD')])])
        (self.root / 'qrel_evidence.txt').write_text('009 Q0 0007.md 1\n', encoding='utf-8')
        (self.root / 'qrel_golds.txt').write_text('009 Q0 0007.md 1\n', encoding='utf-8')
        task = next(browsecomp_tasks(self.root/'corpus', 'corpus-pin', self.root,
                                    self.root/'questions', queries_revision='query-pin'))
        self.assertEqual(task['queries'], [{'_id': '009', 'text': 'Question with\t原始 text'}])
        self.assertEqual(task['qrels'], {'009': {'0007.md': 1.0}})
        self.assertEqual(task['host_labels']['009']['answer'], 'SECRET')
        self.assertNotIn('SECRET', json.dumps(list(task['corpus']) + task['queries']))
        self.assertEqual(task['provenance']['query_source']['revision'], 'query-pin')
        self.assertEqual(task['provenance']['query_source']['projected_columns'], ['query_id', 'query', 'answer'])

    def test_enterprise_empty_unknown_and_large_gold(self):
        docs = [dict(doc_id=f'dsid_{i}', title='title', content='body', source_type='slack') for i in range(12)]
        self.parquet('data/documents/test.parquet', docs)
        self.parquet('data/questions/test.parquet', [
            dict(question_id='q1', question='large', expected_doc_ids=[d['doc_id'] for d in docs], gold_answer='SECRET'),
            dict(question_id='q2', question='absent', expected_doc_ids=[], gold_answer='SECRET'),
            dict(question_id='q3', question='unknown', expected_doc_ids=['oops'], gold_answer='SECRET')])
        task = next(enterprise_tasks(self.root, 'fixed'))
        self.assertEqual(len(task['qrels']['q1']), 12)
        self.assertEqual(len(task['queries']), 1)
        self.assertEqual(len(task['provenance']['exclusions']), 2)
        self.assertNotIn('SECRET', json.dumps(list(task['corpus']) + task['queries']))

    def test_enterprise_duplicate_adaptation_never_selects_conflicting_payload(self):
        rows = [dict(doc_id='same', title='T', content='body', source_type='slack')]
        rows += [dict(rows[0]), dict(rows[0])]
        rows += [dict(doc_id='conflict', title='T', content='body', source_type='slack'),
                 dict(doc_id='conflict', title='T', content='body', source_type='gmail'),
                 dict(doc_id='clean', title='T', content='body', source_type='slack')]
        self.parquet('data/documents/test.parquet', rows)
        self.parquet('data/questions/test.parquet', [
            dict(question_id='keep', question='q', expected_doc_ids=['same', 'clean']),
            dict(question_id='exclude_whole', question='q', expected_doc_ids=['conflict', 'clean'])])
        task = next(enterprise_tasks(self.root, 'fixed'))
        self.assertEqual(task['qrels'], {'keep': {'same': 1.0, 'clean': 1.0}})
        self.assertEqual([row['_id'] for row in task['corpus']], ['same', 'clean'])
        self.assertEqual(task['provenance']['exclusions'], [dict(query_id='exclude_whole', reason='conflicting_corpus_ids', document_ids=['conflict'])])
        audit = task['provenance']['corpus_audit']
        self.assertEqual(audit['raw_rows'], 6)
        self.assertEqual(audit['retained_documents'], 2)
        self.assertEqual(audit['exact_duplicate_rows_collapsed'], 2)
        self.assertEqual(audit['conflicting_rows_excluded'], 2)
        self.assertEqual(audit['duplicate_id_row_counts'], {'conflict': 2, 'same': 3})
        self.assertEqual(len(audit['conflicting_payload_sha256']['conflict']), 2)

    def test_duplicate_qrel_pairs_fail_even_equal_or_nonpositive(self):
        path = self.root / 'qrels.txt'
        for scores in [(1, 1), (1, 2), (0, 1), (-1, -1)]:
            with self.subTest(scores=scores):
                path.write_text(''.join(f'q Q0 doc {score}\n' for score in scores), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Duplicate qrel pair'):
                    _trec(path)

    def test_duplicate_corpus_ids_fail(self):
        self.jsonl('wix_kb_corpus/wix_kb_corpus.jsonl', [dict(id='same', contents='a'), dict(id='same', contents='b')])
        with self.assertRaisesRegex(ValueError, 'Duplicate corpus ID'):
            list(wixqa_tasks(self.root, 'fixed'))

    def test_worksurface_personas_exact_filenames_and_pure_rag(self):
        rows = []
        for i, (persona, slug) in enumerate([('Backend Developer', 'backend_developer'), ('Researcher', 'researcher')]):
            directory = self.root / f'resources/profiles/{slug}/kb_docs'
            directory.mkdir(parents=True)
            (directory / 'original.pdf.md').write_text(persona, encoding='utf-8')
            (directory / 'distractor.md').write_text('noise', encoding='utf-8')
            rows.append(dict(id=f'q{i}', source={'persona': persona, 'task_id': str(i)}, question='q', task_type='rag_only', required_surfaces=['rag'],
                             gold_answer='SECRET', gold_evidence=[dict(surface='rag', file='original.pdf.md', span='text')]))
        rows.append(dict(rows[0], id='cross', task_type='cross_surface', required_surfaces=['rag', 'graph']))
        rows.append(dict(rows[0], id='unknown', gold_evidence=[dict(surface='rag', file='original.pdf')]))
        self.jsonl('data/tasks.jsonl', rows)
        tasks = list(worksurface_tasks(self.root, 'fixed'))
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]['qrels'], {'q0': {'original.pdf.md': 1.0}})
        self.assertEqual(tasks[1]['qrels'], {'q1': {'original.pdf.md': 1.0}})
        corpus0, corpus1 = list(tasks[0]['corpus']), list(tasks[1]['corpus'])
        self.assertEqual(len(corpus0), 2)
        self.assertEqual(corpus0[1]['text'], 'Backend Developer')
        self.assertEqual(corpus1[1]['text'], 'Researcher')
        self.assertNotIn('SECRET', json.dumps(corpus0 + tasks[0]['queries']))
        self.assertEqual({r['reason'] for r in tasks[0]['provenance']['exclusions']}, {'non_pure_rag_task', 'unknown_gold_ids'})

    def test_worksurface_requires_native_source_persona(self):
        self.jsonl('data/tasks.jsonl', [dict(id='q1', persona='Backend Developer', source={},
                                           question='q', task_type='rag_only', required_surfaces=['rag'])])
        with self.assertRaisesRegex(ValueError, 'source.persona'):
            list(worksurface_tasks(self.root, 'fixed'))

    def test_worksurface_skips_workspaces_without_pure_rag_queries(self):
        directory = self.root / 'resources/profiles/product_manager/kb_docs'
        directory.mkdir(parents=True)
        (directory / 'document.md').write_text('Unchanged source document', encoding='utf-8')
        self.jsonl('data/tasks.jsonl', [dict(id='q1', source={'persona': 'Product Manager'},
                                           question='q', task_type='cross_surface', required_surfaces=['rag', 'graph'])])
        self.assertEqual(list(worksurface_tasks(self.root, 'fixed')), [])
        self.assertTrue((directory / 'document.md').is_file())

    def test_worksurface_excludes_incomplete_equivalent_judgments_without_merging_corpus(self):
        directory = self.root / 'resources/profiles/backend_developer/kb_docs'
        directory.mkdir(parents=True)
        for number, body in [(1, 'identical body'), (2, 'identical body'), (3, 'distinct body')]:
            (directory / f't{number}__shared.md').write_text(
                f'<!-- source_task: {number} | source_file: shared.md | surface: rag -->\n\n{body}', encoding='utf-8')
        rows = []
        for qid, files in [('incomplete', ['t1__shared.md']),
                           ('complete', ['t1__shared.md', 't2__shared.md']),
                           ('distinct', ['t3__shared.md'])]:
            rows.append(dict(id=qid, source={'persona': 'Backend Developer'}, question='Find shared.md',
                             task_type='rag_only', required_surfaces=['rag'],
                             gold_evidence=[dict(surface='rag', file=name) for name in files]))
        self.jsonl('data/tasks.jsonl', rows)
        task = next(worksurface_tasks(self.root, 'fixed'))
        self.assertEqual(set(task['qrels']), {'complete', 'distinct'})
        self.assertEqual(set(task['qrels']['complete']), {'t1__shared.md', 't2__shared.md'})
        self.assertEqual(len(list(task['corpus'])), 3)
        self.assertEqual(task['provenance']['equivalent_document_groups'], [['t1__shared.md', 't2__shared.md']])
        self.assertEqual(task['provenance']['exclusions'], [dict(query_id='incomplete',
                         reason='unlabelled_equivalent_documents', original_gold_ids=['t1__shared.md'],
                         document_ids=['t2__shared.md'])])


if __name__ == '__main__':
    unittest.main()
