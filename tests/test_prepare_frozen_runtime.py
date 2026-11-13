"""Exercise the Frozen runtime preparation shell control flow without a cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PrepareFrozenRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="dci-prepare-frozen-")
        self.root = Path(self.tempdir.name)
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        self.helper = self.scripts / "prepare_frozen_runtime.sh"
        shutil.copyfile(ROOT / "scripts/prepare_frozen_runtime.sh", self.helper)
        self.helper.chmod(0o755)
        (self.scripts / "check_frozen_runtime.py").write_text("# fake probe entrypoint\n", encoding="utf-8")
        self.config = self.root / "runtime.json"
        self.config.write_text("{}\n", encoding="utf-8")
        self.bundle = self.root / "release" / "rootfs.tar"
        self.bundle.parent.mkdir()
        self.bundle.write_bytes(b"fixture bundle")
        self.output = self.root / "frozen-probe.json"
        self.trace = self.root / "calls.jsonl"
        self.python = self.root / "bin" / "python"
        self.python.parent.mkdir()
        self.python.write_text(
            """#!%s
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if args[:3] == ['-m', 'dci_bench.runtime', 'prepare']:
    kind = 'prepare'
elif args and args[0].endswith('check_frozen_runtime.py'):
    kind = 'probe'
else:
    kind = 'unexpected'
with Path(os.environ['TEST_TRACE']).open('a', encoding='utf-8') as handle:
    handle.write(json.dumps({'kind': kind, 'args': args}) + '\\n')
if kind == 'prepare' and os.environ.get('TEST_PREPARE_FAIL'):
    raise SystemExit(int(os.environ['TEST_PREPARE_FAIL']))
""" % sys.executable,
            encoding="utf-8",
        )
        self.python.chmod(0o755)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_helper(self, *, bundle: Path | None = None, output: bool = False,
                   dry_run: bool = False, prepare_fail: int | None = None) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
        environment = {
            **os.environ,
            "DCI_RUNTIME_CONFIG": str(self.config),
            "DCI_BENCH_PYTHON": str(self.python),
            "TEST_TRACE": str(self.trace),
            "DRY_RUN": "1" if dry_run else "0",
        }
        environment.pop("DCI_RUNTIME_BUNDLE", None)
        environment.pop("DCI_FROZEN_RUNTIME_PROBE_OUTPUT", None)
        if bundle is not None:
            environment["DCI_RUNTIME_BUNDLE"] = str(bundle)
        if output:
            environment["DCI_FROZEN_RUNTIME_PROBE_OUTPUT"] = str(self.output)
        if prepare_fail is not None:
            environment["TEST_PREPARE_FAIL"] = str(prepare_fail)
        result = subprocess.run(
            ["bash", str(self.helper)],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = [json.loads(line) for line in self.trace.read_text().splitlines()] if self.trace.exists() else []
        return result, rows

    def test_cold_prepare_passes_offline_preflight_bundle_and_probe_output(self) -> None:
        result, rows = self.run_helper(bundle=self.bundle, output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row["kind"] for row in rows], ["prepare", "probe"])
        prepare = rows[0]["args"]
        self.assertEqual(prepare[:3], ["-m", "dci_bench.runtime", "prepare"])
        self.assertIn("--offline", prepare)
        self.assertIn("--preflight", prepare)
        self.assertEqual(prepare[prepare.index("--config") + 1], str(self.config.resolve()))
        self.assertEqual(prepare[prepare.index("--bundle") + 1], str(self.bundle))
        probe = rows[1]["args"]
        self.assertTrue(probe[0].endswith("scripts/check_frozen_runtime.py"))
        self.assertEqual(probe[probe.index("--runtime-config") + 1], str(self.config.resolve()))
        self.assertEqual(probe[probe.index("--output") + 1], str(self.output))

    def test_prepare_failure_stops_before_probe(self) -> None:
        result, rows = self.run_helper(bundle=self.bundle, prepare_fail=23)
        self.assertEqual(result.returncode, 23)
        self.assertEqual([row["kind"] for row in rows], ["prepare"])

    def test_cold_failure_without_bundle_requests_explicit_bundle(self) -> None:
        result, rows = self.run_helper(prepare_fail=17)
        self.assertEqual(result.returncode, 17)
        self.assertEqual([row["kind"] for row in rows], ["prepare"])
        self.assertIn("DCI_RUNTIME_BUNDLE", result.stderr)

    def test_warm_cache_prepare_and_probe_need_no_bundle(self) -> None:
        result, rows = self.run_helper(output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([row["kind"] for row in rows], ["prepare", "probe"])
        self.assertNotIn("--bundle", rows[0]["args"])

    def test_dry_run_makes_no_prepare_or_probe_call(self) -> None:
        result, rows = self.run_helper(bundle=self.bundle, output=True, dry_run=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows, [])
        self.assertIn("dry run", result.stdout)


if __name__ == "__main__":
    unittest.main()
