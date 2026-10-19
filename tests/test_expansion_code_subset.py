import json
from pathlib import Path
import tempfile
import unittest

from dci_bench.data.expansion_code import _hash, _sha, _snapshot_key, _write
from dci_bench.data.expansion_code_subset import build_subset, ARTIFACTS
from dci_bench.scoring.code import file_metadata


class GoldCompleteSubsetTests(unittest.TestCase):
    def source(self, root):
        source = root / 'source'
        repo, commit = 'owner/repo', 'a' * 40
        key = _snapshot_key(repo, commit)
        texts = {'good.py': 'def f():\n    return 1', 'bad.py': 'prefix...[truncated]'}
        files = {name: file_metadata(text) for name, text in texts.items()}
        snapshot = {'snapshot_id': key, 'repository': repo, 'base_commit': commit,
                    'corpus_sha256': _hash(files), 'files': files,
                    'upstream_file_view': {name: {'characters': len(text), 'truncation_marker': name == 'bad.py'}
                                           for name, text in texts.items()}}
        for name, text in texts.items():
            path = source / 'snapshots' / key / 'corpus' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        _write(source / 'snapshots' / key / 'manifest.json', snapshot)
        labels = {qid: {'snapshot_id': key, 'corpus_sha256': _hash(files), 'task': 'code2test', 'gold_files': gold}
                  for qid, gold in [('q1', ['good.py']), ('q2', ['good.py', 'bad.py'])]}
        _write(source / 'host/labels.json', {'queries': labels})
        _write(source / 'queries.json', [{'query_id': qid, 'snapshot_id': key, 'task': 'code2test', 'query': qid} for qid in labels])
        _write(source / 'manifest.json', {'artifact_hashes': {name: _sha(source / name) for name in ('host/labels.json', 'queries.json')},
                                         'snapshot_manifest_sha256': {key: _sha(source / 'snapshots' / key / 'manifest.json')}})
        _write(source / '_COMPLETE.json', {'manifest_sha256': _sha(source / 'manifest.json')})
        return source, key

    def test_exclude_whole_query_preserve_candidate_scope_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, key = self.source(root)
            out = root / 'subset'
            result = build_subset(source, out)
            self.assertEqual(result['samples'], 1)
            labels = json.loads((out / 'labels.json').read_text())['queries']
            self.assertEqual(set(labels), {'q1'})
            self.assertEqual(set(labels['q1']['files']), {'good.py', 'bad.py'})
            excluded = json.loads((out / 'selection_audit.json').read_text())['excluded_queries']
            self.assertEqual(excluded, [{'query_id': 'q2', 'reason': 'at_least_one_positive_file_view_truncated', 'truncated_gold_files': ['bad.py']}])
            marker = json.loads((out / '_COMPLETE.json').read_text())['artifact_sha256']
            self.assertEqual(set(marker), ARTIFACTS)
            self.assertTrue(all(_sha(out / name) == digest for name, digest in marker.items()))
            with self.assertRaises(FileExistsError):
                build_subset(source, out)

    def test_reject_changed_positive_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, key = self.source(root)
            (source / 'snapshots' / key / 'corpus/good.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Positive corpus bytes'):
                build_subset(source, root / 'subset')


if __name__ == '__main__':
    unittest.main()
