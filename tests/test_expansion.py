import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dci_bench.data.document_ids import (
    build_document_id_map,
    remap_qrels,
    validate_document_id_map,
)
from dci_bench.data.expansion import build_document_task, sha256_file, verify_document_task, write_json
from scripts.prepare_expansion_data import verify_catalog


class ExpansionArtifactTests(unittest.TestCase):
    def task(self):
        return {'name': 'toy_expansion', 'dataset': 'official/toy', 'revision': 'a' * 40,
                'track': 'document',
                'corpus': [{'_id': f'd/{i}.md', 'title': '', 'text': f'Document {i}'} for i in range(12)],
                'queries': [{'_id': 'q1', 'text': 'Find the supporting documents'}],
                'qrels': {'q1': {f'd/{i}.md': 1.0 for i in range(11)}},
                'host_labels': {'q1': {'answer': 'HOST_ONLY_CANARY'}},
                'provenance': {'exclusions': []}}

    def source(self):
        return {'raw_directory': '/raw/toy', 'source_fingerprint': 'b' * 64}

    def test_complete_artifact_hidden_labels_top10_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            result = build_document_task(self.task(), suite, self.source(), io_workers=2)
            self.assertEqual((result['num_docs'], result['num_qrels']), (12, 11))
            md = suite / 'metadata/toy_expansion'
            source_ids = json.loads((md / 'source_doc_ids.json').read_text())
            self.assertEqual(len(source_ids), 12)
            self.assertEqual(source_ids['d000001'], 'd/0.md')
            self.assertEqual(source_ids['d000003'], 'd/10.md')
            self.assertEqual(json.loads((md / 'original_qrels.json').read_text())['q1']['d/0.md'], 1.0)
            aliases = json.loads((md / 'qrels.json').read_text())['q1']
            self.assertEqual(
                set(aliases),
                {alias for alias, original in source_ids.items() if original != 'd/11.md'},
            )
            self.assertEqual(sorted((suite / 'workspaces/toy_expansion/corpus').iterdir())[0].name, 'd000001.md')
            self.assertEqual(verify_document_task(suite, 'toy_expansion', io_workers=1),
                             verify_document_task(suite, 'toy_expansion', io_workers=4))
            audit = json.loads((md / 'audit.json').read_text())
            self.assertEqual(audit['queries_over_top_k'], 1)
            self.assertAlmostEqual(audit['mean_recall_ceiling_at_10'], 10 / 11)
            for path in (suite / 'workspaces').rglob('*.md'):
                self.assertNotIn('HOST_ONLY_CANARY', path.read_text())
            corpus = suite / 'workspaces/toy_expansion/corpus/d000001.md'
            corpus.write_text('tampered')
            with self.assertRaises(ValueError):
                verify_document_task(suite, 'toy_expansion')
            with self.assertRaises(ValueError):
                build_document_task(self.task(), suite, self.source())

    def test_oversized_document_excludes_whole_query_not_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = self.task()
            task['corpus'][0]['text'] = 'x' * 1_000_001
            task['queries'].append({'_id': 'q2', 'text': 'another query'})
            task['qrels']['q2'] = {'d/1.md': 1.0}
            result = build_document_task(task, Path(tmp), self.source(), io_workers=2)
            self.assertEqual(result['num_queries'], 1)
            md = Path(tmp) / 'metadata/toy_expansion'
            self.assertEqual(json.loads((md / 'qrels.json').read_text()), {'q2': {'d000002': 1.0}})
            self.assertEqual(json.loads((md / 'audit.json').read_text())['query_exclusions'][0]['query_id'], 'q1')
            verify_document_task(Path(tmp), 'toy_expansion')

    def test_bulk_qrel_remap_is_constant_and_matches_per_query_semantics(self):
        corpus = [
            {'_id': f'source/{index:03d}.md', 'title': '', 'text': f'Document {index}'}
            for index in range(48)
        ]
        corpus[0]['text'] = 'x' * 1_000_001
        queries = []
        qrels = {}
        for index in range(1000):
            qid = f'q{index:04d}'
            first = 1 + (index % 47)
            second = 1 + ((index + 1) % 47)
            queries.append({'_id': qid, 'text': f'Find documents for query {index}'})
            qrels[qid] = {
                f'source/{first:03d}.md': 2.0,
                f'source/{second:03d}.md': 1.0,
            }
        queries.append({'_id': 'excluded', 'text': 'This query depends on an excluded document'})
        qrels['excluded'] = {'source/000.md': 1.0, 'source/001.md': 2.0}
        task = {
            'name': 'bulk_remap', 'dataset': 'official/toy', 'revision': 'a' * 40,
            'track': 'document', 'corpus': corpus, 'queries': queries, 'qrels': qrels,
            'host_labels': {}, 'provenance': {'exclusions': []},
        }
        mapping = build_document_id_map([row['_id'] for row in corpus])
        expected_original = {
            qid: dict(sorted(gold.items())) for qid, gold in qrels.items()
            if qid != 'excluded'
        }
        expected_qrels = {
            qid: dict(sorted((mapping[doc_id], score) for doc_id, score in gold.items()))
            for qid, gold in expected_original.items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch('dci_bench.data.expansion.remap_qrels', wraps=remap_qrels) as remap, \
                    patch('dci_bench.data.document_ids.validate_document_id_map',
                          wraps=validate_document_id_map) as validate:
                result = build_document_task(task, Path(tmp), self.source(), io_workers=2)
            self.assertEqual(remap.call_count, 1)
            self.assertEqual(validate.call_count, 2)
            self.assertEqual(result['num_queries'], 1000)
            md = Path(tmp) / 'metadata/bulk_remap'
            self.assertEqual(json.loads((md / 'original_qrels.json').read_text()), {
                **expected_original,
                'excluded': qrels['excluded'],
            })
            self.assertEqual(json.loads((md / 'qrels.json').read_text()), expected_qrels)
            audit = json.loads((md / 'audit.json').read_text())
            self.assertEqual(audit['query_exclusions'], [{
                'query_id': 'excluded',
                'reason': 'positive_document_outside_compatible_corpus',
                'document_ids': ['source/000.md'],
            }])
            verify_document_task(Path(tmp), 'bulk_remap', io_workers=2)

    def test_unknown_gold_and_duplicate_ids_fail_without_completion(self):
        for kind in ('unknown', 'duplicate'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                task = self.task()
                if kind == 'unknown':
                    task['qrels']['q1']['missing'] = 1.0
                else:
                    task['corpus'].append(copy.deepcopy(task['corpus'][0]))
                with self.assertRaises(ValueError):
                    build_document_task(task, Path(tmp), self.source(), io_workers=2)
                self.assertFalse(list((Path(tmp) / 'metadata').glob('*/_COMPLETE.json')))

    def test_complete_task_can_resume_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            first = build_document_task(self.task(), suite, self.source(), io_workers=2)
            path = suite / 'metadata/toy_expansion/_COMPLETE.json'
            original = path.stat().st_mtime_ns
            second = build_document_task(self.task(), suite, self.source(), io_workers=2)
            self.assertEqual(first, second)
            self.assertEqual(path.stat().st_mtime_ns, original)
            with self.assertRaises(ValueError):
                build_document_task(self.task(), suite, dict(self.source(), source_fingerprint='c' * 64))
            changed = self.task()
            changed['queries'][0]['text'] = 'a changed request'
            with self.assertRaises(ValueError):
                build_document_task(changed, suite, self.source())

    def test_build_refuses_to_reuse_verified_legacy_document_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            first = build_document_task(self.task(), suite, self.source(), io_workers=1)
            legacy = dict(first)
            legacy.pop('document_id_policy')
            with patch('dci_bench.data.expansion.verify_document_task', return_value=legacy):
                with self.assertRaisesRegex(ValueError, 'legacy document ID policy'):
                    build_document_task(self.task(), suite, self.source(), io_workers=1)

    def test_integrity_coverage_and_visible_readme(self):
        for kind in ('readme', 'unlisted_parquet', 'missing_hash'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                suite = Path(tmp)
                build_document_task(self.task(), suite, self.source(), io_workers=2)
                if kind == 'readme':
                    (suite / 'workspaces/toy_expansion/README.md').write_text('injected gold')
                elif kind == 'unlisted_parquet':
                    (suite / 'canonical/toy_expansion/corpus/extra.parquet').write_bytes(b'extra')
                else:
                    path = suite / 'metadata/toy_expansion/_COMPLETE.json'
                    value = json.loads(path.read_text())
                    value['canonical_hashes'] = {}
                    path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    verify_document_task(suite, 'toy_expansion')

    def test_suite_catalog_anchors_task_completion_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            manifest = build_document_task(self.task(), suite, self.source(), io_workers=2)
            completion = suite / 'metadata/toy_expansion/_COMPLETE.json'
            write_json(suite / 'catalog.json', {
                'tasks': [{k: manifest[k] for k in ('task', 'dataset', 'num_queries')}],
                'task_completion_hashes': {'toy_expansion': sha256_file(completion)}})
            write_json(suite / '_COMPLETE.json', {'catalog_sha256': sha256_file(suite / 'catalog.json'),
                                                'num_tasks': 1, 'num_queries': 1})
            verify_catalog(suite)
            completion.write_text(completion.read_text() + ' ')
            with self.assertRaises(ValueError):
                verify_catalog(suite)


if __name__ == '__main__':
    unittest.main()
