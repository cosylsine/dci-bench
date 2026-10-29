"""Regression guards for complete private tools publication and path-only bootstrap."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from scripts import inspect_local_runtime as R


class InspectRuntimeTests(unittest.TestCase):
    def archive(self, root, unsafe=None):
        path = root / 'bundle.tgz'
        with tarfile.open(path, 'w:gz') as handle:
            for name, data in [('inspect-sandbox-tools', b'launcher'), ('_internal/PYZ.pyz', b'archive')]:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                member.mode = 0o755
                handle.addfile(member, io.BytesIO(data))
            if unsafe:
                member = tarfile.TarInfo(unsafe)
                if unsafe == 'link':
                    member.type = tarfile.SYMTYPE
                    member.linkname = '/etc/passwd'
                handle.addfile(member)
        return path

    def test_bundle_identity_new_directory_and_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self.archive(root)
            directory = root / 'private'
            record = R.extract_bundle(archive, directory)
            self.assertEqual(record['archive_sha256'], R.sha(archive))
            self.assertEqual(set(record['files']), {'inspect-sandbox-tools', '_internal/PYZ.pyz'})
            with self.assertRaises(FileExistsError):
                R.extract_bundle(archive, directory)
            payload = directory / '_internal/PYZ.pyz'
            payload.chmod(0o644)
            payload.write_bytes(b'truncated')
            with self.assertRaisesRegex(ValueError, 'tools changed'):
                R.activate(directory)

    def test_partial_and_unsafe_bundle_are_not_accepted(self):
        for unsafe in ['../escape', '/absolute', 'link']:
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    R.extract_bundle(self.archive(root, unsafe), root / 'partial')
                self.assertFalse((root / 'partial/_READY.json').exists())
                with self.assertRaises(FileNotFoundError):
                    R.activate(root / 'partial')

    def test_eager_and_late_imports_all_use_private_cli_without_changing_server_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'private'
            R.extract_bundle(self.archive(root), directory)
            # Run in an independent interpreter, like the production bootstrap.
            code = '''
import importlib, json, pathlib, sys
import inspect_ai.agent
from scripts.inspect_local_runtime import activate
from inspect_ai.util._sandbox.local import LocalSandboxEnvironment
activate(pathlib.Path(sys.argv[1]))
for name in [
    'inspect_ai.util._sandbox._cli', 'inspect_ai.util._sandbox.local',
    'inspect_ai.util._sandbox.exec_remote', 'inspect_ai.util._sandbox._json_rpc_transport',
    'inspect_ai.agent._bridge.sandbox.bridge', 'inspect_ai.tool._sandbox_tools_utils.sandbox',
    'inspect_ai.tool._tools._text_editor']:
    module = importlib.import_module(name)
    assert module.SANDBOX_CLI == sys.argv[1] + '/inspect-sandbox-tools', name
assert LocalSandboxEnvironment._SANDBOX_TOOLS_DIR_ENV == 'INSPECT_SANDBOX_TOOLS_DIR'
from inspect_ai.util._sandbox._cli import SANDBOX_CLI
assert SANDBOX_CLI == sys.argv[1] + '/inspect-sandbox-tools'
print('all aliases use private tools; per-sample server-state selection preserved')
'''
            result = subprocess.run([sys.executable, '-B', '-c', code, str(directory)],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
