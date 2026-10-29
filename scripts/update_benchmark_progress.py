#!/usr/bin/env python3
"""Render the current benchmark report; preserve the dated inventory and snapshot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'results/implementation/20261029-lite-full'
DOMAINS = {'encyclopedia': '百科', 'general_qa': '通用问答', 'scientific': '科研',
           'medical': '医疗', 'finance': '金融', 'legal': '法律', 'enterprise': '企业知识',
           'web': '网页检索', 'software': '软件技术', 'argumentation': '论证检索'}


def json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build_coverage(config: dict, manifest: dict, audit: dict, status: dict) -> dict:
    """Project audited selections into the current domain/language grid.

    Counts come from selected IDs, never from inventory size or editorial text.
    This is a metadata projection, not a corpus integrity audit or inference run.
    """
    body = {k: v for k, v in manifest.items() if k != 'overall_sha256'}
    if json_hash(body) != manifest['overall_sha256'] or audit['overall_sha256'] != manifest['overall_sha256']:
        raise ValueError('Prepared manifest and audit identities do not match')
    if json_hash(config) != manifest['config_identity']['config_sha256']:
        raise ValueError('Current config differs from the audited prepared manifest')
    scope, sizes = config['scope'], config['sampling']['sizes']
    expected = {f'{d}_{lang}' for d in scope['domains'] for lang in scope['languages']}
    notes = {r['task']: r for r in status['data_tasks']}
    if (len(expected) != scope['task_count'] or set(manifest['tasks']) != expected
            or set(notes) != expected or len(notes) != len(status['data_tasks'])):
        raise ValueError('Coverage must contain exactly the configured domain/language tasks')
    tasks, supplementary = [], []
    for domain in scope['domains']:
        for language in scope['languages']:
            task = f'{domain}_{language}'
            row, note = manifest['tasks'][task], notes[task]
            ready = row['status'] == 'ready'
            if row['status'] not in ('ready', 'blocked') or note['status'] != ('已准备' if ready else '阻塞'):
                raise ValueError(f'Stale or unsupported task status: {task}')
            counts = {split: len(row['query_ids'][split]) for split in sizes}
            if counts != (sizes if ready else dict.fromkeys(sizes, 0)):
                raise ValueError(f'Unexpected selected query counts: {task}')
            pools = {}
            for tier in ('lite', 'full'):
                pool = row['pools'][tier]
                pools[tier] = pool['capacity'] if pool else None
                if pools[tier] != (config['tiers'][tier]['capacity'] if ready else None):
                    raise ValueError(f'Unexpected pool capacity: {task}/{tier}')
            if ready and note['source'] != row['source_id']:
                raise ValueError(f'Stale selected source: {task}')
            tasks.append({'task': task, 'domain': domain, 'domain_label': DOMAINS[domain],
                          'language': language, 'status': row['status'],
                          'source': row['source_id'], 'source_display': note['source'],
                          'counts': counts, 'pools': pools, 'detail': note['detail'],
                          'excluded_query_count': row['excluded_query_count'], 'reason': row['reason']})
            for candidate in config['sources'][task]['candidates']:
                if candidate['provenance'] == 'translated':
                    supplementary.append({'task': task, 'source': candidate['source_id'],
                                          'domain_label': DOMAINS[domain], 'language': language})
    ready_ids = {r['task'] for r in tasks if r['status'] == 'ready'}
    if ready_ids != set(audit['ready_tasks']) or expected - ready_ids != set(audit['blocked_tasks']):
        raise ValueError('Audit task readiness differs from prepared manifest')
    return {'schema_version': 'dci-benchmark-v2-coverage-v1', 'as_of': status['as_of'],
            'scope': scope, 'sampling': config['sampling'],
            'tiers': {k: config['tiers'][k]['capacity'] for k in ('lite', 'full')},
            'target': {k: v * len(expected) for k, v in sizes.items()},
            'prepared': {k: sum(r['counts'][k] for r in tasks) for k in sizes},
            'ready_tasks': len(ready_ids), 'blocked_tasks': len(expected - ready_ids),
            'ready_for_formal': audit['ready_for_formal'], 'tasks': tasks,
            'supplementary_translations': supplementary,
            'evidence': {'prepared_manifest_sha256': manifest['overall_sha256'],
                         'config_sha256': manifest['config_identity']['config_sha256'],
                         'body_hashes_checked': audit['body_hashes_checked'],
                         'boundary': 'Selected query IDs and pool metadata; not model execution.'}}


def update(report: Path, status: dict, *, config_path: Path = ROOT / 'configs/benchmark_v2.json',
           prepared_path: Path = EVIDENCE / 'prepared-manifest.json',
           audit_path: Path = EVIDENCE / 'data-audit.json') -> None:
    config, prepared, audit = (json.loads(p.read_text()) for p in (config_path, prepared_path, audit_path))
    coverage = build_coverage(config, prepared, audit, status)
    history = report / 'history' / '20261028'
    if not history.exists():
        history.mkdir(parents=True)
        for path in report.iterdir():
            if path.is_file():
                shutil.copy2(path, history / path.name)
    esc = html.escape
    stage_rows = ''.join('<tr><th scope="row">' + esc(r['name']) + '</th><td>' + esc(r['status']) +
                         '</td><td>' + esc(r['detail']) + '</td></tr>' for r in status['stages'])
    evidence = ''.join('<li><a href="' + esc(r['path'], quote=True) + '">' + esc(r['label']) +
                       '</a></li>' for r in status.get('evidence', []))
    gaps = ''.join('<li>' + esc(item) + '</li>' for item in status['limitations'])
    timing = ''.join('<tr><th scope="row">' + esc(r['phase']) + '</th><td>' + esc(str(r['lite'])) +
                     '</td><td>' + esc(str(r['full'])) + '</td></tr>' for r in status.get('timing_rows', []))
    timing_table = ('<h3>CPU链路阶段计时</h3><p class="muted">每档10题，均值秒；finance_en，4并发，'
                    '脚本化本机 provider，AFS 临时工作区。准备与推理有重叠，Pi执行含工具与HTTP时间，不能相加或外推GPU吞吐。</p>'
                    '<div class="table-wrap"><table><thead><tr><th>阶段</th><th>Lite</th><th>Full</th></tr>'
                    '</thead><tbody>' + timing + '</tbody></table></div>') if timing else ''
    scope_rows = ''.join(f'<tr><th scope="row">{label}</th><td>{config["sampling"]["sizes"][split]}</td>'
                         f'<td>{coverage["target"][split]:,}</td><td>{2 * coverage["target"][split]:,}</td>'
                         f'<td>{coverage["prepared"][split]:,} / {coverage["target"][split]:,}</td></tr>'
                         for split, label in [('smoke', 'Smoke（开发集子集）'), ('dev', '开发集 / 配置'), ('test', '正式集 / 模型')])
    # Full task table remains readable if the host disables scripts.
    task_rows = ''.join('<tr><th scope="row">' + esc(r['domain_label']) + '<small>' + esc(r['task']) +
                        '</small></th><td>' + r['language'] + '</td><td>' + ('已准备' if r['status'] == 'ready' else '阻塞') +
                        '</td><td class="source-name">' + esc(r['source_display']) + '</td><td>' +
                        ' / '.join(str(r['counts'][k]) for k in ('dev', 'test', 'smoke')) + '</td><td>' +
                        ('200 / 1000' if r['status'] == 'ready' else '— / —') + '</td><td>' + esc(r['detail']) +
                        '</td></tr>' for r in coverage['tasks'])
    supplements = ''.join('<li>' + esc(r['domain_label'] + ' / ' + r['language'] + '：' + r['source']) +
                           '（译本，仅补充候选，未计入20任务准备数）</li>' for r in coverage['supplementary_translations'])
    page = (ROOT / 'docs/templates/benchmark_progress.html').read_text(encoding='utf-8')
    substitutions = {'AS_OF': esc(status['as_of']), 'SUMMARY': esc(status['summary']),
                     'READY': str(coverage['ready_tasks']), 'BLOCKED': str(coverage['blocked_tasks']),
                     'DEV': str(coverage['prepared']['dev']), 'TEST': str(coverage['prepared']['test']),
                     'SCOPE_ROWS': scope_rows, 'STAGE_ROWS': stage_rows, 'TIMING': timing_table,
                     'TASK_ROWS': task_rows, 'SUPPLEMENTS': supplements or '<li>无译本补充候选。</li>',
                     'EVIDENCE': evidence, 'GAPS': gaps,
                     'COVERAGE_JSON': json.dumps(coverage, ensure_ascii=False).replace('<', '\\u003c')}
    for key, value in substitutions.items():
        page = page.replace('__' + key + '__', value)
    (report / 'index.html').write_text(page, encoding='utf-8')
    (report / 'benchmark-v2-coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + '\n')
    with (report / 'benchmark-v2-coverage.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task', 'domain', 'language', 'status', 'selected_source', 'source_or_candidate',
                         'dev_queries', 'test_queries', 'smoke_queries', 'lite', 'full', 'detail'])
        for row in coverage['tasks']:
            writer.writerow([row['task'], row['domain'], row['language'], row['status'], row['source'], row['source_display'],
                             *(row['counts'][k] for k in ('dev', 'test', 'smoke')),
                             row['pools']['lite'], row['pools']['full'], row['detail']])
    lines = ['# DCI-Bench 执行进展（' + status['as_of'] + '）', '', status['summary'], '',
             '- [详细进展](index.html#benchmark-v2) · [当前20任务覆盖](index.html#data) · [运行说明](../benchmark_v2.md) · [历史报告](history/20261028/index.html)',
             '- 目标：10领域 × 中英双语，共20任务；Lite=200、Full=1000。',
             '- 每任务正式50题、开发10题；Smoke取开发集1题。',
             '- 目标正式1,000 query / 2,000次执行；开发200 query / 400次执行；Smoke 40次。',
             f'- 实际准备：{coverage["ready_tasks"]}/20任务；开发{coverage["prepared"]["dev"]}/200、正式{coverage["prepared"]["test"]}/1,000、Smoke {coverage["prepared"]["smoke"]}/20题；不代表已执行。',
             '- 采样：random无放回、seed 42，保留策略接口；开发与正式集按同源组隔离。',
             '- 两档默认累计64k token；12轮搜索、24次工具调用、一次提交与一次修复。',
             '- 上下文65,536；输出1,024 token；工具8KiB；硬超时300秒。',
             '- 成本目标：本地≤24小时或API≤¥1000/模型，预留20%；正式集不自动缩样。', '', '## 实施状态', '']
    lines += ['- **' + r['name'] + '：' + r['status'] + '**。' + r['detail'] for r in status['stages']]
    lines += ['- **报告呈现：已对齐当前设计**。主矩阵固定10领域×中英，支持状态/领域/语言筛选及任务明细；旧来源矩阵与三档结果保留在历史报告。章节导航使用页内滚动。',
              '', '## 未完成与边界', ''] + ['- ' + item for item in status['limitations']]
    lines += ['', '- 历史159次真实smoke保留；不代表新版协议、双语正式集或成本目标已经验证。',
              '- [20任务覆盖JSON](benchmark-v2-coverage.json) · [CSV](benchmark-v2-coverage.csv) · [执行状态与证据](benchmark-v2-status.json) · [交互校验](browser-validation-v2.json) · [报告校验](manifest.json)', '']
    assert len(lines) <= 80
    (report / 'progress.md').write_text('\n'.join(lines), encoding='utf-8')
    (report / 'benchmark-v2-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in report.iterdir() if p.is_file() and p.name != 'manifest.json'}
    browser_path = report / 'browser-validation-v2.json'
    browser = json.loads(browser_path.read_text()) if browser_path.exists() else {}
    manifest = {'schema_version': 'dci-project-progress-manifest-v2', 'as_of': status['as_of'],
                'historical_manifest': 'history/20261028/manifest.json', 'outputs': outputs,
                'inputs': {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (config_path, prepared_path, audit_path, ROOT / 'docs/templates/benchmark_progress.html')},
                'checks': {'markdown_lines': len(lines), 'historical_matrix_preserved':
                           (report / 'data_matrix.json').read_bytes() == (history / 'data_matrix.json').read_bytes(),
                           'coverage_task_count': len(coverage['tasks']), 'coverage_ready_tasks': coverage['ready_tasks'],
                           'new_gpu_inference_completed': False,
                           'browser_validation_current': browser.get('passed') is True and browser.get('html_sha256') == outputs['index.html'],
                           'yuque_host_verified': False}}
    (report / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', required=True, type=Path)
    parser.add_argument('--report', type=Path, default=ROOT / 'docs/project-progress-20261028')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/benchmark_v2.json')
    parser.add_argument('--prepared', type=Path, default=EVIDENCE / 'prepared-manifest.json')
    parser.add_argument('--audit', type=Path, default=EVIDENCE / 'data-audit.json')
    args = parser.parse_args()
    update(args.report, json.loads(args.status.read_text(encoding='utf-8')),
           config_path=args.config, prepared_path=args.prepared, audit_path=args.audit)
