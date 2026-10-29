"""Cost gates and paired statistics for the bilingual Lite/Full benchmark.

This module has no model/backend imports. Missing usage is never free usage.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import random
from statistics import mean

from dci_bench.data.benchmark_v2 import TASK_IDS

TIERS = {'lite': 200, 'full': 1000}
METRICS = ('recall_at_10', 'ndcg_at_10')


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PriceQuote:
    model_key: str
    as_of: str
    source: str
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float
    cache_write_per_million: float

    @classmethod
    def from_dict(cls, value: dict) -> 'PriceQuote':
        if value.get('currency') != 'CNY':
            raise ValueError('price quote must use CNY (record the FX conversion in source)')
        for key in ('model_key', 'as_of', 'source'):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f'price quote requires {key}')
        fields = ('input_per_million', 'output_per_million', 'cache_read_per_million', 'cache_write_per_million')
        for key in fields:
            if type(value.get(key)) not in (float, int) or not math.isfinite(value[key]) or value[key] < 0:
                raise ValueError(f'price quote requires a finite nonnegative {key}')
        return cls(**{key: value[key] for key in ('model_key', 'as_of', 'source', *fields)})

    def cost(self, usage: dict) -> dict:
        if usage.get('available') is not True:
            return {'cost_cny': None, 'basis': 'usage_unavailable'}
        for key in ('input_tokens', 'output_tokens'):
            if type(usage.get(key)) is not int or usage[key] < 0:
                raise ValueError(f'usage requires nonnegative integer {key}')
        cached = usage.get('cache_read_tokens')
        written = usage.get('cache_write_tokens')
        if cached is None or written is None:
            rate = max(self.input_per_million, self.cache_read_per_million, self.cache_write_per_million)
            value = usage['input_tokens'] * rate + usage['output_tokens'] * self.output_per_million
            return {'cost_cny': value / 1_000_000, 'basis': 'upper_bound_cache_breakdown_unavailable'}
        if any(type(v) is not int or v < 0 for v in (cached, written)) or cached + written > usage['input_tokens']:
            raise ValueError('cache usage must be a nonnegative subset of input tokens')
        value = ((usage['input_tokens'] - cached - written) * self.input_per_million +
                 cached * self.cache_read_per_million + written * self.cache_write_per_million +
                 usage['output_tokens'] * self.output_per_million)
        return {'cost_cny': value / 1_000_000, 'basis': 'provider_usage_times_pinned_quote'}

    def reservation(self, count: int, token_budget: int) -> float:
        """Conservative batch reservation; retries are disabled by the v2 runner."""
        return count * token_budget * max(self.input_per_million, self.output_per_million,
                                         self.cache_read_per_million, self.cache_write_per_million) / 1_000_000


def cost_gate(*, execution_count: int, calibration: dict | None, mode: str,
              price: PriceQuote | None = None, budget_cny: float = 1000,
              hours: float = 24, reserve_fraction: float = .2) -> dict:
    if mode not in ('local', 'api') or execution_count <= 0:
        raise ValueError('invalid cost gate mode or execution count')
    if (not 0 <= reserve_fraction < 1 or not math.isfinite(budget_cny)
            or not math.isfinite(hours) or budget_cny <= 0 or hours <= 0):
        raise ValueError('invalid cost limits')
    failures = []
    projections = {}
    if not calibration or calibration.get('measured') is not True:
        failures.append('measured calibration required')
    elif mode == 'local':
        throughput = calibration.get('queries_per_hour')
        if type(throughput) not in (int, float) or not math.isfinite(throughput) or throughput <= 0:
            failures.append('positive measured throughput required')
        else:
            projected = execution_count / throughput / (1 - reserve_fraction)
            projections['wall_hours_with_reserve'] = projected
            if projected > hours:
                failures.append('projected wall time exceeds limit')
    else:
        if price is None:
            failures.append('pinned price quote required')
        else:
            estimate = price.cost(calibration.get('mean_usage', {}))
            if estimate['cost_cny'] is None:
                failures.append('measured token usage required')
            else:
                projected = execution_count * estimate['cost_cny']
                projections.update(formal_cost_cny=projected, basis=estimate['basis'])
                if projected > budget_cny * (1 - reserve_fraction):
                    failures.append('projected cost exceeds formal allocation')
    return {'passed': not failures, 'failures': failures, 'projections': projections,
            'execution_count': execution_count, 'limits': {'hours': hours, 'budget_cny': budget_cny,
                                                         'reserve_fraction': reserve_fraction}}


def _metric(sample: dict, metric: str) -> float:
    if not sample['valid']:
        return 0.0
    value = sample['metrics'][metric]
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'invalid {metric}')
    return float(value)


def paired_interval(left: list[dict], right: list[dict] | None, *, metric: str,
                    replicates: int = 2000, seed: int = 42) -> dict:
    """Equal-task macro bootstrap; shared group weights preserve query pairing.

    Positive exponential cluster weights avoid empty cells when a family spans
    tasks/languages. Tiers are evaluated separately, never counted as new queries.
    """
    key = lambda row: (row['task_id'], row['query_id'], row['tier'])
    lhs = {key(row): row for row in left}
    rhs = {key(row): row for row in right} if right is not None else None
    if len(lhs) != len(left) or (rhs is not None and (len(rhs) != len(right) or set(lhs) != set(rhs))):
        raise ValueError('paired statistics require unique identical query/task/tier IDs')
    if not lhs or replicates < 100 or len({row['tier'] for row in left}) != 1:
        raise ValueError('bootstrap requires one nonempty tier and at least 100 replicates')
    cells = defaultdict(list)
    for identity, sample in lhs.items():
        group = sample['group_id']
        if rhs is not None and rhs[identity]['group_id'] != group:
            raise ValueError('paired query groups differ')
        difference = _metric(sample, metric) - (_metric(rhs[identity], metric) if rhs is not None else 0)
        cells[sample['task_id']].append((group, difference))
    groups = sorted({group for rows in cells.values() for group, _ in rows})
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        weights = {group: rng.expovariate(1) for group in groups}
        estimates.append(mean(sum(weights[g] * v for g, v in rows) / sum(weights[g] for g, _ in rows)
                              for rows in cells.values()))
    estimates.sort()
    return {'estimate': mean(mean(value for _, value in rows) for rows in cells.values()),
            'interval_95': [estimates[int(.025 * replicates)], estimates[min(replicates - 1, int(.975 * replicates))]],
            'method': 'paired_cluster_multiplier_bootstrap', 'replicates': replicates,
            'seed': seed, 'groups': len(groups), 'cells': len(cells)}


def summarize(samples: list[dict], *, expected_tasks: list[str], elapsed_seconds: float,
              price: PriceQuote | None = None) -> dict:
    identities = [(row['task_id'], row['tier'], row['query_id']) for row in samples]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate executed sample')
    tiers = {}
    for tier in TIERS:
        selected = [row for row in samples if row['tier'] == tier]
        present = {row['task_id'] for row in selected}
        complete = present == set(expected_tasks)
        tiers[tier] = {'samples': len(selected), 'represented_tasks': len(present), 'coverage_complete': complete,
                       'macro': {metric: paired_interval(selected, None, metric=metric) for metric in METRICS}
                       if complete and selected else None}
        tier_costs = [price.cost(row['usage'])['cost_cny'] for row in selected] if price else []
        durations = sorted(row['wall_seconds'] for row in selected if row.get('wall_seconds') is not None)
        tiers[tier].update(
            valid=sum(row['valid'] for row in selected),
            failure_rate=mean(not row['valid'] for row in selected) if selected else None,
            cost_cny=sum(tier_costs) if tier_costs and all(cost is not None for cost in tier_costs) else None,
            total_query_seconds=sum(durations), mean_query_seconds=mean(durations) if durations else None,
            p95_query_seconds=durations[min(len(durations) - 1, math.ceil(.95 * len(durations)) - 1)] if durations else None)
        task_scores = {task: {metric: mean(_metric(row, metric) for row in selected if row['task_id'] == task)
                              for metric in METRICS} for task in sorted(present)}
        tiers[tier]['task_diagnostics'] = task_scores
        for dimension, component in (('domain', 0), ('language', 1)):
            groups = defaultdict(list)
            for task, scores in task_scores.items():
                groups[task.rsplit('_', 1)[component]].append(scores)
            tiers[tier][dimension + '_diagnostics'] = {
                group: {metric: mean(scores[metric] for scores in values) for metric in METRICS}
                for group, values in sorted(groups.items())}
    costs = [price.cost(row['usage']) for row in samples] if price else []
    times = sorted(row['wall_seconds'] for row in samples if row.get('wall_seconds') is not None)
    available = bool(samples) and all(row['usage'].get('available') is True for row in samples)
    mean_usage = {'available': available}
    if available:
        # Preserve integer units for the pricing interface; round up projections.
        for name in ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens'):
            values = [row['usage'].get(name) for row in samples]
            mean_usage[name] = math.ceil(mean(values)) if all(value is not None for value in values) else None
        if mean_usage['cache_read_tokens'] is not None and mean_usage['cache_write_tokens'] is not None:
            mean_usage['input_tokens'] = max(mean_usage['input_tokens'],
                                            mean_usage['cache_read_tokens'] + mean_usage['cache_write_tokens'])
    stages = defaultdict(list)
    for row in samples:
        for name, value in row.get('stage_timing', {}).items():
            if name.endswith('_seconds') and type(value) in (int, float):
                stages[name].append(value)
    return {'schema_version': 'dci-benchmark-v2-summary', 'samples': samples, 'tiers': tiers,
            'completed': len(samples), 'valid': sum(row['valid'] for row in samples),
            'failure_rate': mean(not row['valid'] for row in samples) if samples else None,
            'measured': bool(samples) and elapsed_seconds > 0,
            'elapsed_seconds': elapsed_seconds,
            'queries_per_hour': len(samples) * 3600 / elapsed_seconds if elapsed_seconds > 0 else None,
            'p50_seconds': times[len(times) // 2] if times else None,
            'p95_seconds': times[min(len(times) - 1, math.ceil(.95 * len(times)) - 1)] if times else None,
            'mean_usage': mean_usage,
            'cost_cny': sum(row['cost_cny'] for row in costs) if costs and all(row['cost_cny'] is not None for row in costs) else None,
            'cost_basis': sorted({row['basis'] for row in costs}),
            'stage_seconds': {name: {'mean': mean(values), 'total': sum(values), 'samples': len(values)}
                              for name, values in sorted(stages.items())},
            'gpu_utilization': None}


def validate_complete_report(report: dict, *, split: str) -> None:
    """Validate the query matrix at the report ingestion boundary."""
    size = {'dev': 10, 'test': 50}[split]
    samples = report['samples']
    cells = defaultdict(set)
    for sample in samples:
        cells[(sample['task_id'], sample['tier'])].add(sample['query_id'])
    tasks = {task for task, _ in cells}
    if (report.get('split') != split or report.get('complete_scope') is not True
            or len(samples) != 40 * size or report.get('completed') != len(samples)
            or tasks != set(TASK_IDS) or len(cells) != 40
            or any(len(ids) != size for ids in cells.values())
            or {tier for _, tier in cells} != set(TIERS)
            or any(cells[(task, 'lite')] != cells[(task, 'full')] for task in tasks)
            or report.get('valid') != sum(sample['valid'] for sample in samples)):
        raise ValueError(f'{split} report must cover 20 tasks x 2 tiers x {size} paired queries')


def select_budget(reports: dict[int, dict]) -> dict:
    if set(reports) != {32000, 64000, 96000}:
        raise ValueError('calibration requires 32k, 64k and 96k reports')
    baseline = reports[96000]
    if any(report['completed'] != 400 or report.get('split') != 'dev'
           or report.get('measured') is not True for report in reports.values()):
        raise ValueError('each calibration arm requires the complete measured 400-execution dev set')
    for budget, report in reports.items():
        if report.get('token_budget') != budget or report.get('complete_scope') is not True:
            raise ValueError('calibration budget/scope mismatch')
        validate_complete_report(report, split='dev')
    for field in ('selection_sha256', 'backend_sha256', 'source_fingerprint', 'concurrency'):
        if not baseline.get(field) or any(report.get(field) != baseline[field] for report in reports.values()):
            raise ValueError(f'calibration {field} mismatch')
    candidates = []
    for budget in (32000, 64000, 96000):
        report = reports[budget]
        deltas = {}
        for tier in TIERS:
            left = [row for row in report['samples'] if row['tier'] == tier]
            right = [row for row in baseline['samples'] if row['tier'] == tier]
            deltas[tier] = paired_interval(left, right, metric='recall_at_10')
        passed = (report['valid'] / report['completed'] >= .99
                  and all(d['estimate'] >= -.02 for d in deltas.values()))
        candidates.append({'budget': budget, 'passed': passed, 'recall_deltas': deltas})
    chosen = next((row['budget'] for row in candidates if row['passed']), None)
    return {'selected_budget': chosen, 'candidates': candidates,
            'interpretation': 'Development selection only; intervals do not establish equivalence.'}


def compare_concurrency(reports: dict[int, dict]) -> dict:
    """Rank measured 4/8/16 trials; expose utilization instead of inventing a target."""
    if set(reports) != {4, 8, 16}:
        raise ValueError('concurrency comparison requires 4, 8 and 16 reports')
    baseline = reports[4]
    identities = {(s['task_id'], s['tier'], s['query_id']) for s in baseline['samples']}
    candidates = []
    for concurrency, report in sorted(reports.items()):
        for field in ('selection_sha256', 'backend_sha256', 'source_fingerprint', 'token_budget', 'split'):
            if not baseline.get(field) or report.get(field) != baseline[field]:
                raise ValueError('concurrency trials differ in ' + field)
        if (report.get('split') != 'dev' or report.get('concurrency') != concurrency
                or report.get('measured') is not True or report.get('completed') != len(identities)
                or {(s['task_id'], s['tier'], s['query_id']) for s in report['samples']} != identities):
            raise ValueError('concurrency trials require matching measured dev samples')
        throughput, p95 = report['queries_per_hour'], report['p95_seconds']
        eligible = (report['failure_rate'] <= .01 and throughput is not None and throughput > 0
                    and p95 is not None and p95 <= 300)
        candidates.append({'concurrency': concurrency, 'eligible': eligible,
                           'queries_per_hour': throughput, 'p95_seconds': p95,
                           'failure_rate': report['failure_rate'], 'gpu_utilization': report['gpu_utilization']})
    eligible = [row for row in candidates if row['eligible']]
    chosen = max(eligible, key=lambda row: (row['queries_per_hour'], -row['concurrency'])) if eligible else None
    return {'selected_concurrency': chosen['concurrency'] if chosen else None, 'candidates': candidates,
            'complete_scope': all(report.get('complete_scope') is True for report in reports.values()),
            'interpretation': 'Highest observed completion throughput among trials with <=1% failures and P95 <=300s. '
                              'GPU values describe the driver metric; inspect sharing/MIG before interpreting them.'}
