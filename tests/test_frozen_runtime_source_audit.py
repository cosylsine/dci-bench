import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dci_bench.results import phase4_runner


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "benchmark_runner_source_audit", ROOT / "scripts/run_benchmark_v2.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _write_source_tree(root: Path) -> Path:
    files = {
        "dci_bench/fixture.py": "fixture = 1\n",
        "dci_bench/protocol/fixture.json": "{}\n",
        "pi-dci/scripts/dci-run.ts": "export const run = 1;\n",
        "pi-dci/scripts/dci-runtime.ts": "export const runtime = 1;\n",
        "scripts/run_document_pool.py": "# fixture\n",
        "scripts/run_task.py": "# fixture\n",
        "scripts/inspect_local_runtime.py": "# fixture\n",
        "scripts/run_benchmark_v2.py": "# fixture\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root / "pi-dci/scripts/dci-runtime.ts"


class FrozenRuntimeSourceAuditTest(unittest.TestCase):
    def test_runtime_and_adapter_sources_change_both_fingerprints(self):
        for relative in ('pi-dci/scripts/dci-runtime.ts', 'pi-dci/scripts/adapters/fixture.ts',
                         'dci_bench/backends/adapters/fixture.py'):
            with self.subTest(source=relative):
                self.check_source_fingerprints(relative)

    def check_source_fingerprints(self, relative):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_source_tree(root)
            runtime_source = root / relative
            runtime_source.parent.mkdir(parents=True, exist_ok=True)
            runtime_source.write_text('fixture = 1\n')
            runner_file = root / "scripts/run_benchmark_v2.py"
            with patch.object(runner, "ROOT", root), patch.object(
                runner, "__file__", str(runner_file)
            ):
                source_before = runner.source_fingerprint()

                with patch.object(
                    phase4_runner, "_git_audit", return_value={"revision": "fixture"}
                ), patch.object(
                    phase4_runner, "environment_audit", return_value={}
                ), patch.object(
                    phase4_runner, "_package_version", return_value="fixture"
                ):
                    audit_before = phase4_runner.code_audit(root)[
                        "source_tree_fingerprint"
                    ]

                    runtime_source.write_text(
                        "export const runtime = 2;\n", encoding="utf-8"
                    )

                    source_after = runner.source_fingerprint()
                    audit_after = phase4_runner.code_audit(root)[
                        "source_tree_fingerprint"
                    ]

            self.assertNotEqual(source_before, source_after)
            self.assertNotEqual(audit_before, audit_after)


if __name__ == "__main__":
    unittest.main()
