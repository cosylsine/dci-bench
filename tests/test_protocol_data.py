"""Protocol packaging must preserve source scope and reject stale/tampered inputs."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import zstandard

from dci_bench.data import expansion_code as arb
from dci_bench.data.code_online import load_code_samples
from dci_bench.data.expansion import EXPANSION_VERSION, build_document_task, sha256_file, write_json
from dci_bench.data.expansion_code_subset import build_subset
from dci_bench.data.registry import TASKS
from dci_bench.tasks.code_retrieval import code_retrieval
from scripts.prepare_protocol_data import build_release, verify_release


class ProtocolDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.suites = [self.root / 'documents', self.root / 'supplement']
        for index, suite in enumerate(self.suites):
            task = {'name': f'protocol_fixture_{index}', 'dataset': 'test/documents', 'revision': 'b' * 40,
                    'corpus': [{'_id': 'a/b.md', 'title': '', 'text': 'source whitespace\n'},
                               {'_id': 'negative', 'title': '', 'text': 'Another candidate'}],
                    'queries': [{'_id': 'q/1', 'text': 'Find the document'}],
                    'qrels': {'q/1': {'a/b.md': 1.0}}, 'host_labels': {'answer': 'HOST_ONLY'},
                    'provenance': {'exclusions': []}}
            manifest = build_document_task(task, suite, {'source_fingerprint': 'c' * 64,
                                                       'raw_directory': '/fixture/raw'}, io_workers=1)
            row = {k: manifest[k] for k in ('task', 'dataset', 'revision', 'task_instruction',
                                           'num_docs', 'num_queries', 'num_qrels')}
            write_json(suite / 'catalog.json', {'schema_version': EXPANSION_VERSION,
                'evaluation_track': 'document', 'tasks': [row], 'task_completion_hashes': {
                task['name']: sha256_file(suite / 'metadata' / task['name'] / '_COMPLETE.json')}})
            write_json(suite / '_COMPLETE.json', {'catalog_sha256': sha256_file(suite / 'catalog.json'),
                                                'num_tasks': 1, 'num_queries': 1, 'audit_hashes': {}})
        self.source, self.subset = self.root / 'code', self.root / 'subset'
        self.labels, self.output = self.root / 'labels.json', self.root / 'prepared'
        release, commit = 'v2_trace2code', 'a' * 40
        rows = [{'id': qid, 'repo': 'owner/repo', 'base_commit': commit, 'task_type': 'trace2code',
                 'query': {'failure': 'observed failure'}, 'gold': {'root_cause_files': [gold]}}
                for qid, gold in [('q/good', 'src/a.py'), ('q/cut', 'src/cut.py')]]
        chunks = [{'repo': 'owner/repo', 'base_commit': commit, 'path': path, 'kind': 'file',
                   'start_line': 1, 'end_line': 2, 'text': text}
                  for path, text in [('src/a.py', 'def f():\n    return 1\n'),
                                     ('src/cut.py', 'prefix...[truncated]'), ('README.md', 'negative candidate')]]
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as handle:
            for name, values in [(f'benchmark/{release}/samples.jsonl', rows),
                                 (f'corpus/{release}/owner__repo/{commit}.chunks.jsonl', chunks)]:
                data = ''.join(json.dumps(row) + '\n' for row in values).encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))
        raw = self.root / 'raw'
        archive = raw / 'releases' / release / f'agent_retrieval_bench_{release}.tar.zst'
        archive.parent.mkdir(parents=True)
        archive.write_bytes(zstandard.ZstdCompressor().compress(buffer.getvalue()))
        with patch.dict(arb.RELEASES, {release: (2, hashlib.sha256(archive.read_bytes()).hexdigest())}, clear=True):
            arb.build_arb(raw, self.source, arb.ARB_REVISION, (release,), io_workers=1)
        arb.export_scoring_labels(self.source, self.labels, io_workers=1)
        build_subset(self.source, self.subset)

    def build(self):
        return build_release(self.output, self.suites, self.source, self.subset, self.labels)

    def test_current_loaders_accept_views_without_losing_negative_candidates(self):
        registry = dict(TASKS)
        before = sha256_file(self.source / '_COMPLETE.json')
        self.build()
        verified = verify_release(self.output, verify_corpus=True, io_workers=1)
        self.assertTrue(verified['source_corpus_bytes_verified'])
        self.assertEqual(verified['document_tasks'], 2)
        self.assertEqual(verified['code_queries'], 2)
        self.assertEqual(verified['code_gold_untruncated_queries'], 1)
        self.assertEqual(TASKS, registry)
        self.assertEqual(sha256_file(self.source / '_COMPLETE.json'), before)
        for view, expected in [('file_view', 2), ('gold_untruncated', 1)]:
            path = self.output / 'code' / view / 'queries.json'
            rows = json.loads(path.read_text())
            self.assertTrue(all(set(row) == {'query_id', 'query', 'workspace'} for row in rows))
            labels = self.labels if view == 'file_view' else self.subset / 'labels.json'
            samples = load_code_samples(queries_path=path, labels_path=labels, all_queries=True)
            self.assertEqual(len(samples), expected)
            self.assertEqual(set(samples[0].reference['files']), {'src/a.py', 'src/cut.py', 'README.md'})
            task = code_retrieval(samples, output_dir=self.root / 'unused-results')
            self.assertTrue(all(not s.target for s in task.dataset))
            self.assertNotIn('gold_files', json.dumps([s.model_dump(mode='json') for s in task.dataset]))
        self.assertFalse((self.root / 'unused-results').exists())

    def test_refuse_overwrite_and_detect_packaged_or_source_tamper(self):
        self.build()
        with self.assertRaises(FileExistsError):
            self.build()
        query_file = self.output / 'code/gold_untruncated/queries.json'
        original = query_file.read_bytes()
        query_file.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            verify_release(self.output)
        query_file.write_bytes(original)
        self.labels.write_text(self.labels.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
            verify_release(self.output)

    def test_reject_wrong_expanded_labels_and_duplicate_document_tasks(self):
        labels = json.loads(self.labels.read_text())
        labels['queries']['q/good']['gold_files'] = ['README.md']
        self.labels.write_text(json.dumps(labels))
        with self.assertRaisesRegex(ValueError, 'reference mismatch'):
            self.build()
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, 'duplicate document task'):
            build_release(self.output, [self.suites[0], self.suites[0]], self.source, self.subset, self.labels)

    def test_refuse_output_inside_immutable_source(self):
        with self.assertRaises(FileExistsError):
            build_release(self.source / 'new', self.suites, self.source, self.subset, self.labels)

    def test_reject_reanchored_wrong_code_source_identity_or_catalog(self):
        path = self.source / 'manifest.json'
        original = json.loads(path.read_text())
        completion = self.source / '_COMPLETE.json'
        for field, value in [('source_complete', True), ('dataset', 'wrong/source'),
                             ('revision', '0' * 40), ('scope', 'full_source'),
                             ('source_line_correspondence_verified', True), ('snapshot_hashes', {})]:
            with self.subTest(field=field):
                path.write_text(json.dumps({**original, field: value}))
                done = json.loads(completion.read_text())
                done['manifest_sha256'] = sha256_file(path)
                completion.write_text(json.dumps(done))
                with self.assertRaises(ValueError):
                    self.build()
                self.assertFalse(self.output.exists())

    def test_reject_reanchored_wrong_document_track_or_schema(self):
        suite = self.suites[0]
        path = suite / 'catalog.json'
        original = json.loads(path.read_text())
        for field, value in [('evaluation_track', 'code'), ('schema_version', 'unknown')]:
            with self.subTest(field=field):
                path.write_text(json.dumps({**original, field: value}))
                done = json.loads((suite / '_COMPLETE.json').read_text())
                done['catalog_sha256'] = sha256_file(path)
                (suite / '_COMPLETE.json').write_text(json.dumps(done))
                with self.assertRaisesRegex(ValueError, 'document expansion catalog'):
                    self.build()
                self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
