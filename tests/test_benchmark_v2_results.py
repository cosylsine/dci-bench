import copy
import unittest

from dci_bench.results.benchmark_v2 import PriceQuote, cost_gate, paired_interval, summarize, select_budget, compare_concurrency, validate_complete_report
from dci_bench.backends.manifest import build_api_manifest, inspect_generation_kwargs, validate_backend_manifest
from dci_bench.data.benchmark_v2 import TASK_IDS


def quote():
    return PriceQuote.from_dict({'currency': 'CNY', 'model_key': 'fixture', 'as_of': '2026-10-29',
                                 'source': 'fixture only, not a real market quote',
                                 'input_per_million': 2, 'output_per_million': 8,
                                 'cache_read_per_million': 1, 'cache_write_per_million': 3})


def row(task='scientific_en', q='1', tier='lite', score=1, valid=True):
    return {'task_id': task, 'query_id': q, 'tier': tier, 'group_id': task + ':' + q,
            'valid': valid, 'metrics': {'recall_at_10': score, 'ndcg_at_10': score},
            'wall_seconds': 120, 'usage': {'available': True, 'input_tokens': 10000,
             'output_tokens': 1000, 'cache_read_tokens': 5000, 'cache_write_tokens': 1000}}


class BenchmarkV2ResultsTest(unittest.TestCase):
    def test_cached_usage_has_no_double_charge(self):
        self.assertAlmostEqual(quote().cost(row()['usage'])['cost_cny'], .024)

    def test_unknown_usage_is_not_zero(self):
        self.assertIsNone(quote().cost({'available': False})['cost_cny'])
        usage = {**row()['usage'], 'cache_read_tokens': None}
        self.assertEqual(quote().cost(usage)['basis'], 'upper_bound_cache_breakdown_unavailable')
        self.assertAlmostEqual(quote().cost(usage)['cost_cny'], .038)

    def test_invalid_quote_and_cache_counts_rejected(self):
        with self.assertRaises(ValueError):
            PriceQuote.from_dict({'currency': 'USD'})
        with self.assertRaises(ValueError):
            quote().cost({**row()['usage'], 'cache_read_tokens': 10001})

    def test_cost_gate_requires_measurement_and_reserve(self):
        self.assertFalse(cost_gate(execution_count=2000, calibration=None, mode='local')['passed'])
        self.assertTrue(cost_gate(execution_count=2000, calibration={'measured': True, 'queries_per_hour': 200}, mode='local')['passed'])
        self.assertFalse(cost_gate(execution_count=2000, calibration={'measured': True, 'queries_per_hour': 100}, mode='local')['passed'])
        self.assertTrue(cost_gate(execution_count=2000, calibration={'measured': True, 'mean_usage': row()['usage']}, mode='api', price=quote())['passed'])

    def test_macro_equal_cells_not_equal_query_volume(self):
        rows = [row(q=str(i), score=1) for i in range(9)] + [row(task='legal_zh', score=0)]
        result = paired_interval(rows, None, metric='recall_at_10', replicates=100)
        self.assertEqual(result['estimate'], .5)

    def test_invalid_samples_are_zero_and_pairs_are_identical(self):
        samples = [row(valid=False), row(q='2', score=.5)]
        self.assertEqual(paired_interval(samples, None, metric='recall_at_10', replicates=100)['estimate'], .25)
        result = paired_interval(samples, samples, metric='recall_at_10', replicates=100)
        self.assertEqual(result['interval_95'], [0, 0])
        with self.assertRaises(ValueError):
            paired_interval(samples, samples[:1], metric='recall_at_10', replicates=100)

    def test_missing_task_does_not_publish_formal_macro(self):
        summary = summarize([row()], expected_tasks=['scientific_en', 'legal_zh'], elapsed_seconds=120)
        self.assertIsNone(summary['tiers']['lite']['macro'])
        self.assertIsNone(summary['gpu_utilization'])

    def test_duplicate_samples_and_partial_calibration_rejected(self):
        with self.assertRaises(ValueError):
            summarize([row(), row()], expected_tasks=['scientific_en'], elapsed_seconds=120)
        with self.assertRaises(ValueError):
            select_budget({32000: {'completed': 4}, 64000: {'completed': 4}, 96000: {'completed': 4}})

    def test_remote_identity_does_not_invent_weights(self):
        manifest = build_api_manifest(model_key='fixture', served_model_name='fixture-2026',
                                      provider='example', revision='2026-01-01', base_url='https://example.com/v1')
        validate_backend_manifest(manifest)
        self.assertNotIn('weights', manifest['model'])
        self.assertNotIn('extra_body', inspect_generation_kwargs(manifest['backend']))
        manifest['model']['revision'] = ''
        with self.assertRaises(ValueError):
            validate_backend_manifest(manifest)

    def test_budget_selection_checks_measured_counts_and_quality(self):
        reports = {}
        for budget, score in ((32000, .94), (64000, .99), (96000, 1)):
            samples = [row(task=task, q=str(q), tier=tier, score=score)
                       for task in TASK_IDS for tier in ('lite', 'full') for q in range(10)]
            reports[budget] = {'samples': samples, 'completed': 400, 'valid': 400, 'split': 'dev',
                               'token_budget': budget, 'complete_scope': True, 'measured': True, 'concurrency': 8,
                               'selection_sha256': 'same', 'backend_sha256': 'same', 'source_fingerprint': 'same'}
        self.assertEqual(select_budget(reports)['selected_budget'], 64000)
        for sample in reports[32000]['samples'][:5]:
            sample['valid'] = False
        reports[32000]['valid'] = 395
        self.assertEqual(select_budget(reports)['selected_budget'], 64000)
        reports[32000]['samples'] = reports[32000]['samples'][:-1]
        with self.assertRaisesRegex(ValueError, 'cover 20 tasks'):
            select_budget(reports)

    def test_budget_selection_requires_measured_arms(self):
        reports = {}
        for budget, score in ((32000, .94), (64000, .99), (96000, 1)):
            samples = [row(task=task, q=str(q), tier=tier, score=score)
                       for task in TASK_IDS for tier in ('lite', 'full') for q in range(10)]
            reports[budget] = {'samples': samples, 'completed': 400, 'valid': 400, 'split': 'dev',
                               'token_budget': budget, 'complete_scope': True, 'measured': True, 'concurrency': 8,
                               'selection_sha256': 'same', 'backend_sha256': 'same', 'source_fingerprint': 'same'}
        reports[32000]['measured'] = False
        with self.assertRaisesRegex(ValueError, 'measured'):
            select_budget(reports)

    def test_budget_selection_rejects_cross_tier_recall_cancellation(self):
        reports = {}
        for budget, score in ((32000, .7), (64000, .8), (96000, .8)):
            samples = [row(task=task, q=str(q), tier=tier, score=score)
                       for task in TASK_IDS for tier in ('lite', 'full') for q in range(10)]
            reports[budget] = {'samples': samples, 'completed': 400, 'valid': 400, 'split': 'dev',
                               'token_budget': budget, 'complete_scope': True, 'measured': True, 'concurrency': 8,
                               'selection_sha256': 'same', 'backend_sha256': 'same', 'source_fingerprint': 'same'}
        for sample in reports[64000]['samples']:
            sample['metrics']['recall_at_10'] = .6 if sample['tier'] == 'lite' else 1
        result = select_budget(reports)
        cancellation = result['candidates'][1]['recall_deltas']
        self.assertAlmostEqual(cancellation['lite']['estimate'] + cancellation['full']['estimate'], 0)
        self.assertFalse(result['candidates'][1]['passed'])
        self.assertEqual(result['selected_budget'], 96000)

    def test_domain_language_diagnostics_and_stage_time(self):
        samples = [row(task='general_qa_en', score=1), row(task='general_qa_zh', score=0)]
        samples[0]['stage_timing'] = {'workspace_prepare_seconds': 2.0}
        result = summarize(samples, expected_tasks=['general_qa_en', 'general_qa_zh'], elapsed_seconds=120)
        self.assertEqual(result['tiers']['lite']['domain_diagnostics']['general_qa']['recall_at_10'], .5)
        self.assertEqual(result['tiers']['lite']['language_diagnostics']['en']['recall_at_10'], 1)
        self.assertEqual(result['stage_seconds']['workspace_prepare_seconds']['total'], 2)

    def test_concurrency_selection_rejects_fast_error_prone_trial(self):
        base = {'samples': [row()], 'completed': 1, 'split': 'dev', 'measured': True,
                'token_budget': 64000, 'selection_sha256': 'same', 'backend_sha256': 'same',
                'source_fingerprint': 'same', 'p95_seconds': 200, 'gpu_utilization': None, 'complete_scope': False}
        reports = {c: {**copy.deepcopy(base), 'concurrency': c, 'queries_per_hour': c * 10,
                       'failure_rate': .02 if c == 16 else 0} for c in (4, 8, 16)}
        result = compare_concurrency(reports)
        self.assertEqual(result['selected_concurrency'], 8)
        self.assertFalse(result['complete_scope'])
        reports[16]['token_budget'] = 32000
        with self.assertRaisesRegex(ValueError, 'token_budget'):
            compare_concurrency(reports)

    def test_formal_coverage_cannot_be_claimed_with_one_cell(self):
        report = {'samples': [row(), row(tier='full')], 'completed': 2, 'valid': 2,
                  'split': 'test', 'complete_scope': True}
        with self.assertRaisesRegex(ValueError, '20 tasks'):
            validate_complete_report(report, split='test')

    def test_formal_coverage_requires_canonical_task_ids(self):
        samples = [row(task=task, q=str(q), tier=tier)
                   for task in TASK_IDS for tier in ('lite', 'full') for q in range(50)]
        samples[0]['task_id'] = 'domain0_en'
        report = {'samples': samples, 'completed': 1000, 'valid': 1000,
                  'split': 'test', 'complete_scope': True}
        with self.assertRaisesRegex(ValueError, '20 tasks'):
            validate_complete_report(report, split='test')


if __name__ == '__main__':
    unittest.main()
