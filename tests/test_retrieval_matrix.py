"""Checks for benchmark eligibility, independent of optional evidence."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/render_retrieval_matrix.py'
spec = importlib.util.spec_from_file_location('retrieval_matrix', SCRIPT)
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class EligibilityTests(unittest.TestCase):
    def row(self, **kwargs):
        return dict(source='example', documents=1200, queries=100,
                    positive_pairs=100, release_state='labels_ready', **kwargs)

    def test_evidence_is_not_a_retrieval_gate(self):
        plain = matrix.classify(self.row(evidence=0))
        annotated = matrix.classify(self.row(evidence=100))
        self.assertEqual(plain['inventory_role'], 'three_tier_source')
        self.assertEqual(plain['inventory_role'], annotated['inventory_role'])
        self.assertEqual(plain['size_supported_tiers'], [200, 500, 1000])

    def test_small_corpus_retains_only_real_capacities(self):
        row = self.row()
        row['documents'] = 240
        result = matrix.classify(row)
        self.assertEqual(result['inventory_role'], 'small_corpus')
        self.assertEqual(result['size_supported_tiers'], [200])

    def test_queries_without_qrels_are_not_labelled_coverage(self):
        row = self.row()
        row['positive_pairs'] = 0
        result = matrix.classify(row)
        self.assertEqual(result['inventory_role'], 'corpus_only')
        self.assertEqual(result['retrieval_state'], 'C')

    def test_demo_does_not_enter_main_coverage(self):
        row = self.row()
        row.update(source='github_kubernetes_website_zh', documents=5, queries=0, positive_pairs=0)
        result = matrix.classify(row)
        self.assertEqual(result['inventory_role'], 'connector_demo')
        self.assertEqual(result['size_supported_tiers'], [])

    def test_review_release_does_not_enter_ready_coverage(self):
        row = self.row()
        row['release_state'] = 'review_required'
        self.assertEqual(matrix.classify(row)['inventory_role'], 'review_required')

    def test_new_source_in_same_cell_is_not_extra_coverage(self):
        old = dict(domain='medical', language='vi', inventory_role='three_tier_source')
        parallel = dict(old)
        new = dict(domain='legal', language='de', inventory_role='three_tier_source')
        pending = dict(domain='encyclopedia', language='ar', inventory_role='review_required')
        change = matrix.coverage_change([old], [old, parallel, new, pending])
        self.assertEqual(change['ready_before'], 1)
        self.assertEqual(change['ready_after'], 2)
        self.assertEqual(change['new_ready_cells'], [['legal', 'de']])
        self.assertEqual(change['pending_only_cells'], [['encyclopedia', 'ar']])

    def test_becoming_ready_is_distinct_from_new_size_coverage(self):
        old = dict(domain='legal', language='de', inventory_role='review_required')
        new = dict(old, inventory_role='three_tier_source')
        change = matrix.coverage_change([old], [old, new])
        self.assertEqual(change['new_ready_cells'], [['legal', 'de']])
        self.assertEqual(change['new_size_eligible_cells'], [])

    def test_source_domain_correction_preserves_original_classification(self):
        row = self.row(domain='web')
        row['source'] = 'gapfill_mteb_germandpr'
        result = matrix.classify(row)
        self.assertEqual(result['domain'], 'encyclopedia')
        self.assertEqual(result['source_domain'], 'web')
        self.assertIn('reference', result['domain_correction'])

    def test_translation_is_not_inferred_from_language(self):
        for value in ['native Chinese', 'non_translated', 'untranslated']:
            self.assertEqual(matrix.translation_kind(value), 'native')
        for value in ['human_translated', 'machine-translated from MS MARCO', 'translated']:
            self.assertEqual(matrix.translation_kind(value), 'translated')
        for value in [None, 'unknown', 'mixed']:
            self.assertEqual(matrix.translation_kind(value), 'unknown')

    def test_translation_coverage_counts_each_ready_cell_once(self):
        def row(domain, language, translation, role='three_tier_source'):
            return dict(domain=domain, language=language, translation=translation, inventory_role=role)
        result = matrix.translation_coverage([
            row('medical', 'zh', 'native'), row('medical', 'zh', 'translated'),
            row('web', 'fr', 'translated'), row('web', 'fr', 'translated'),
            row('enterprise', 'en', 'unknown'),
            row('legal', 'fr', 'native', 'review_required'),
        ])
        self.assertEqual(result, {'cells_with_native_source': 1, 'translated_only_cells': 1,
                                 'remaining_unknown_or_mixed_cells': 1})


if __name__ == '__main__':
    unittest.main()
