"""Executable releases must be rejected before touching data/output directories."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_pool_smoke
import prepare_selective_expansion
from dci_bench.data.pipeline import batch


class CodeLayoutTests(unittest.TestCase):
    def test_sealers_reject_data_and_results_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            for directory in ('results', 'eval_data', 'raw_data', 'eval_data_code'):
                target = parent / directory / 'new-bundle'
                with self.subTest(directory=directory):
                    with self.assertRaisesRegex(ValueError, 'releases/'):
                        run_pool_smoke.seal(parent / 'missing-data', target, parent / 'outputs')
                    with self.assertRaisesRegex(ValueError, 'releases/'):
                        batch.prepare(target, parent / 'missing-matrix')
                    self.assertFalse(target.exists())

    def test_historical_code_paths_resolve_without_rewriting_results(self):
        old = ROOT / 'results/protocol-freezes/20261015-selective-full-v1/runtime'
        new = ROOT / 'releases/protocol-freezes/20261015-selective-full-v1/runtime'
        self.assertEqual(prepare_selective_expansion.code_path(old), new)
        result = ROOT / 'results/Qwen3.8-27B/pool-smoke-runs/old/summary.json'
        self.assertEqual(prepare_selective_expansion.code_path(result), result)
