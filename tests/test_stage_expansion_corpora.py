"""Full-candidate relocation preserves identity, isolation and frozen verification."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dci_bench.data.expansion import build_document_task, verify_document_task
from scripts import stage_expansion_corpora as S


class LocalCorpusTests(unittest.TestCase):
    def fixture(self, root):
        suite = root / 'source'
        task = dict(name='toy', dataset='fixture/toy', revision='a' * 40, track='document',
            corpus=[{'_id': f'd/{i}', 'title': '', 'text': f'Document {i}'} for i in range(3)],
            queries=[{'_id': 'q1', 'text': 'Find documents'}], qrels={'q1': {'d/0': 1.0}},
            host_labels={'q1': {'answer': 'HOST_ONLY_CANARY'}}, provenance={'exclusions': []})
        build_document_task(task, suite, {'raw_directory': '/raw/toy', 'source_fingerprint': 'b' * 64}, io_workers=1)
        (suite / 'catalog.json').write_text('{}\n')
        (suite / 'annotation_audit.json').write_text('{"source":"fixture"}\n')
        (suite / '_COMPLETE.json').write_text(json.dumps({'catalog_sha256': S.digest(suite / 'catalog.json'), 'audit_hashes': {
            'annotation_audit.json': S.digest(suite / 'annotation_audit.json')}}))
        # Unselected catalog tasks need metadata for the unchanged registrar.
        extra = suite / 'metadata/unselected'
        extra.mkdir()
        (extra / '_COMPLETE.json').write_text('{}\n')
        row = dict(task='toy', suite=str(suite), num_docs=3, metadata=str(suite / 'metadata/toy'),
                   workspace=str(suite / 'workspaces/toy'), canonical=str(suite / 'canonical/toy'))
        return {'toy': row}

    def test_complete_copy_passes_original_verifier_and_preserves_host_only_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.fixture(root)
            source = Path(rows['toy']['suite'])
            before = {str(p): S.digest(p) for p in source.rglob('*') if p.is_file()}
            new, record = S.stage(rows, root / 'local', workers=2)
            target = Path(new['toy']['suite'])
            self.assertEqual(verify_document_task(source, 'toy'), verify_document_task(target, 'toy'))
            self.assertEqual(S.digest(source / 'annotation_audit.json'), S.digest(target / 'annotation_audit.json'))
            self.assertEqual(record['tasks']['toy']['documents'], 3)
            self.assertTrue((target / 'metadata/unselected/_COMPLETE.json').is_file())
            for path, digest in before.items():
                self.assertEqual(S.digest(path), digest)
            for path in (target / 'workspaces/toy').rglob('*'):
                if path.is_file():
                    self.assertEqual(path.stat().st_nlink, 1)
                    self.assertFalse(path.is_symlink())
                    self.assertNotIn('HOST_ONLY_CANARY', path.read_text())
            with self.assertRaises(FileExistsError):
                S.stage(rows, root / 'local')

    def test_corruption_extra_file_and_links_fail_closed(self):
        for kind in ['bytes', 'extra', 'missing', 'symlink', 'hardlink', 'audit', 'catalog']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows = self.fixture(root)
                corpus = Path(rows['toy']['workspace']) / 'corpus'
                path = next(corpus.iterdir())
                if kind == 'bytes':
                    path.write_bytes(b'x' * path.stat().st_size)
                elif kind == 'extra':
                    (corpus / 'extra.md').write_text('extra')
                elif kind == 'missing':
                    path.unlink()
                elif kind == 'audit':
                    (Path(rows['toy']['suite']) / 'annotation_audit.json').write_text('tampered')
                elif kind == 'catalog':
                    (Path(rows['toy']['suite']) / 'catalog.json').write_text('tampered')
                elif kind == 'symlink':
                    payload = path.read_bytes()
                    path.unlink()
                    external = root / 'external'
                    external.write_bytes(payload)
                    path.symlink_to(external)
                else:
                    os.link(path, root / 'external')
                with self.assertRaises((ValueError, FileNotFoundError)):
                    S.stage(rows, root / 'local', workers=1)
                self.assertFalse((root / 'local/_STAGED.json').exists())

    def test_remote_mount_and_insufficient_storage_are_rejected(self):
        with patch.object(S.subprocess, 'check_output', return_value='nfs\n'):
            with self.assertRaisesRegex(ValueError, 'node-local'):
                S.local_filesystem(Path('/tmp'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.fixture(root)
            with patch.object(S.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 1})()):
                with self.assertRaisesRegex(ValueError, 'GiB free'):
                    S.stage(rows, root / 'local')
            self.assertFalse((root / 'local').exists())


if __name__ == '__main__':
    unittest.main()
