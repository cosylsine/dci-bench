"""Small safeguards for immutable preparation and query partitioning."""
import importlib.util
import json
import shutil
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'selective_expansion_preparer', Path(__file__).resolve().parents[1] / 'scripts/prepare_selective_expansion.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SelectiveExpansionTests(unittest.TestCase):
    def test_partition_preserves_original_ids_and_candidate_independent_selection(self):
        ids = ['expertwritten/' + str(i) for i in range(45)]
        pilot, rest = MODULE.partition('wixqa_expertwritten', ids)
        reverse_pilot, _ = MODULE.partition('wixqa_expertwritten', list(reversed(ids)))
        self.assertEqual(len(pilot), 10)
        self.assertEqual(len(rest), 35)
        self.assertEqual(set(pilot), set(reverse_pilot))
        self.assertEqual(set(pilot) | set(rest), set(ids))
        self.assertFalse(set(pilot) & set(rest))
        self.assertEqual(pilot, [q for q in ids if q in set(pilot)])

    def test_invalid_query_inventory_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.partition('task', ['duplicate'] * 12)
        with self.assertRaises(ValueError):
            MODULE.partition('task', ['a', 'b'])

    def test_existing_directory_refused_before_reading_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'plan'
            output.mkdir()
            sentinel = output / 'user-file'
            sentinel.write_text('preserve')
            with self.assertRaises(FileExistsError):
                MODULE.prepare(output, Path(tmp) / 'release')
            self.assertEqual(sentinel.read_text(), 'preserve')
            self.assertFalse((Path(tmp) / 'release').exists())

    def test_generated_artifact_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / 'artifact.json').write_text(json.dumps({'scope': 'frozen'}))
            MODULE.seal(output)
            MODULE.verify_directory(output)
            (output / 'artifact.json').write_text(json.dumps({'scope': 'changed'}))
            with self.assertRaises(ValueError):
                MODULE.verify_directory(output)

    def test_runtime_override_cannot_change_the_frozen_contract(self):
        source = MODULE.ROOT / 'results/expansion-runtimes/20261015-selective-document-v1'
        if not source.exists():
            self.skipTest('Prepared expansion snapshot is not present')
        MODULE.validate_execution_runtime(source)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp) / 'runtime'
            shutil.copytree(source, runtime, symlinks=True)
            contract = runtime / 'dci_bench/protocol/contract.json'
            contract.chmod(0o644)
            contract.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Unreviewed execution runtime change'):
                MODULE.validate_execution_runtime(runtime)


if __name__ == '__main__':
    unittest.main()
