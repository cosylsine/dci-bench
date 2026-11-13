"""Run the actual bootstrap control flow against a disposable package manager."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def exercise(self, *, installed=False, apt_fails=False, namespace_fails=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / 'bin'
            binary.mkdir()
            trace = root / 'apt.log'
            bwrap, rg = binary / 'bwrap', binary / 'rg'
            # Only substitute destination paths, so no system package is changed.
            script = (ROOT / 'scripts/ensure_pool_smoke_tools.sh').read_text()
            script = script.replace('/usr/bin/bwrap', str(bwrap)).replace('/usr/bin/rg', str(rg))
            target = root / 'bootstrap.sh'
            target.write_text(script)
            tool = '#!/bin/bash\nif [[ "$1" == --version ]]; then echo fixture; exit 0; fi\nexit ' + ('1' if namespace_fails else '0') + '\n'
            template = root / 'tool'
            template.write_text(tool)
            template.chmod(0o755)
            apt = binary / 'apt-get'
            apt.write_text('#!/bin/bash\nset -eu\n'
                '[[ -z "${LD_LIBRARY_PATH:-}" && -z "${LD_PRELOAD:-}" ]] || exit 77\n'
                f'echo "$*" >> "{trace}"\n' + ('exit 42\n' if apt_fails else
                f'if [[ "$1" == install ]]; then cp "{template}" "{bwrap}"; cp "{template}" "{rg}"; fi\n'))
            apt.chmod(0o755)
            (binary / 'id').write_text('#!/bin/bash\necho 0\n')
            (binary / 'id').chmod(0o755)
            if installed:
                bwrap.write_text(tool)
                rg.write_text(tool)
                bwrap.chmod(0o755)
                rg.chmod(0o755)
            env = dict(os.environ, PATH=str(binary) + ':/usr/bin:/bin', LD_LIBRARY_PATH='/invalid/conda/lib')
            env.pop('LD_PRELOAD', None)
            env.pop('DCI_RUNTIME_CONFIG', None)
            result = subprocess.run(['bash', str(target)], env=env, text=True, capture_output=True, timeout=20)
            return result, trace.read_text() if trace.exists() else ''

    def test_missing_tools_are_installed_with_clean_system_library_environment(self):
        result, log = self.exercise()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('update', log)
        self.assertIn('install -y --no-install-recommends', log)
        self.assertIn('bubblewrap ripgrep', log)
        self.assertIn('Acquire::Retries=3', log)

    def test_existing_tools_skip_package_manager(self):
        result, log = self.exercise(installed=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(log, '')

    def test_install_failure_is_not_reported_as_success(self):
        result, _ = self.exercise(apt_fails=True)
        self.assertEqual(result.returncode, 42)

    def test_namespace_denial_stops_before_model(self):
        result, _ = self.exercise(namespace_fails=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('GPU/model loading has not started', result.stderr)

    def test_frozen_mode_installs_only_host_bwrap_and_skips_host_rg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root/'bin'
            binary.mkdir()
            bwrap = binary/'bwrap'
            template = root/'tool'
            template.write_text('#!/bin/sh\nif [[ "$1" == --version ]]; then echo fixture; fi\nexit 0\n')
            template.chmod(0o755)
            apt_trace = root/'apt.log'
            apt = binary/'apt-get'
            apt.write_text('#!/bin/bash\nset -eu\n'
                           f'echo "$*" >> "{apt_trace}"\n'
                           f'if [[ "$1" == install ]]; then cp "{template}" "{bwrap}"; fi\n')
            apt.chmod(0o755)
            script = (ROOT/'scripts/ensure_pool_smoke_tools.sh').read_text()
            script = script.replace('/usr/bin/bwrap', str(bwrap)).replace('/usr/bin/rg', str(binary/'missing-rg'))
            target = root/'bootstrap.sh'
            target.write_text(script)
            env = dict(os.environ, DCI_RUNTIME_CONFIG=str(root/'runtime.json'),
                       PATH=str(binary)+':/usr/bin:/bin')
            result = subprocess.run(['bash', str(target)], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('bubblewrap', apt_trace.read_text().splitlines()[-1])
            self.assertNotIn('ripgrep', apt_trace.read_text())

    def test_frozen_mode_namespace_failure_stops_before_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root/'bin'
            binary.mkdir()
            bwrap = binary/'bwrap'
            bwrap.write_text('#!/bin/bash\nif [[ "$1" == --version ]]; then echo fixture; exit 0; fi\nexit 23\n')
            bwrap.chmod(0o755)
            script = (ROOT/'scripts/ensure_pool_smoke_tools.sh').read_text().replace('/usr/bin/bwrap', str(bwrap))
            target = root/'bootstrap.sh'
            target.write_text(script)
            env = dict(os.environ, DCI_RUNTIME_CONFIG=str(root/'runtime.json'), PATH=str(binary)+':/usr/bin:/bin')
            result = subprocess.run(['bash', str(target)], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn('GPU/model loading has not started', result.stderr)

    def test_dry_run_skips_frozen_host_dependency_installation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root/'bin'
            binary.mkdir()
            apt_trace = root/'apt.log'
            apt = binary/'apt-get'
            apt.write_text(f'#!/bin/sh\ntouch "{apt_trace}"\nexit 99\n')
            apt.chmod(0o755)
            script = (ROOT/'scripts/ensure_pool_smoke_tools.sh').read_text().replace('/usr/bin/bwrap', str(binary/'missing-bwrap'))
            target = root/'bootstrap.sh'
            target.write_text(script)
            env = dict(os.environ, DCI_RUNTIME_CONFIG=str(root/'runtime.json'), DRY_RUN='1',
                       PATH=str(binary)+':/usr/bin:/bin')
            result = subprocess.run(['bash', str(target)], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(apt_trace.exists())


if __name__ == '__main__':
    unittest.main()
