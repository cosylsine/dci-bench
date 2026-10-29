#!/usr/bin/env python3
"""Build the 2026-10-28 collaborator report from existing metadata, without inference.

This is a dated report, not a live experiment dashboard. The snapshot assertions
prevent its editorial text from silently describing a different dataset or run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/mnt/afs2/202609/embedding_models/dci-bench/eval_data')
DATE = '2026-10-28'


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New report directory; must not exist')
    parser.add_argument('--matrix', type=Path, default=BASE / 'pipeline_matrix_overview_20261023_v3/data_matrix.json')
    parser.add_argument('--catalog', type=Path, default=BASE / 'retrieval_pools/catalog.json')
    parser.add_argument('--quality', type=Path, default=BASE / 'retrieval_pools/audit/kalm_quality/kalm_pre_injection_quality.json')
    parser.add_argument('--run', type=Path, default=ROOT / 'results/Qwen3.8-27B/pool-smoke-runs/20261026T103919Z-1')
    args = parser.parse_args()
    matrix, catalog, quality = read(args.matrix), read(args.catalog), read(args.quality)
    summary_path, complete_path = args.run / 'summary.json', args.run / '_COMPLETE'
    summary, complete = read(summary_path), read(complete_path)
    launch_path = args.run / 'launch-plan.json'
    launch = read(launch_path)
    catalog_complete_path = args.catalog.parent / '_COMPLETE.json'
    catalog_complete = read(catalog_complete_path)
    assert catalog_complete['catalog_sha256'] == digest(args.catalog), 'Catalog completion hash mismatch'
    assert complete['summary_sha256'] == digest(summary_path), 'Smoke summary hash mismatch'
    assert complete['launch_plan_sha256'] == digest(launch_path), 'Launch plan hash mismatch'
    assert complete['status'] == 'complete' and summary['status'] == 'passed'
    assert (summary['planned'], summary['completed'], summary['valid'], summary['smoke_passed']) == (159, 159, 159, 159)
    assert matrix['counts']['source_views'] == len(matrix['sources']) == 99
    assert matrix['counts']['three_tier_source_views'] == 56
    assert matrix['counts']['three_tier_review_views'] == 17
    assert catalog['source_count'] == len(catalog['sources']) == 53
    assert catalog['pool_count'] == sum(len(r['pools']) for r in catalog['sources']) == 159
    assert catalog['queries'] == sum(r['queries'] for r in catalog['sources']) == 196088
    assert quality['totals'] == {'queries': 196088, 'positive_pairs': 234570}
    assert launch['bundle'].endswith('/20261026-pool-smoke-v4')
    assert launch['budgets'] == {'200': 96000, '500': 128000, '1000': 192000}
    assert launch['model'] == 'Qwen3.8-27B'

    pools = {r['source']: r for r in catalog['sources']}
    quality_by_source = {r['source_id']: r for r in quality['sources']}
    sources = []
    for original in matrix['sources']:
        row = {k: original[k] for k in ('source', 'domain', 'language', 'documents', 'queries',
                                       'positive_pairs', 'inventory_role', 'translation_kind', 'source_path')}
        row.update(source_family=original.get('source_family', original['source']),
                   task_origin=original.get('task_origin', '历史冻结任务'),
                   pool_built=original['source'] in pools, kalm_recall_1000=None)
        if row['pool_built']:
            pool, q = pools[row['source']], quality_by_source[row['source']]
            assert row['queries'] == pool['queries'] == q['query_count']
            assert row['documents'] == pool['documents']
            assert row['positive_pairs'] == q['positive_count']
            assert (row['domain'], row['language']) == (q['domain'], q['language'])
            assert sorted(p['capacity'] for p in pool['pools']) == [200, 500, 1000]
            for p in pool['pools']:
                assert (Path(p['path']) / 'metadata' / pool['task'] / '_COMPLETE.json').is_file(), p['path']
            row['kalm_recall_1000'] = q['metrics']['recall@1000']
        sources.append(row)
    built = [r for r in sources if r['pool_built']]
    assert len(built) == len(pools) == len(quality_by_source)
    current = {
        'source_count': len(built), 'pool_count': catalog['pool_count'], 'queries': catalog['queries'],
        'positive_pairs': quality['totals']['positive_pairs'],
        'domains': sorted({r['domain'] for r in built}), 'languages': sorted({r['language'] for r in built}),
        'cells': len({(r['domain'], r['language']) for r in built}),
        'excluded_ready': sorted(r['source'] for r in sources if r['inventory_role'] == 'three_tier_source' and not r['pool_built']),
        'injected_positives': {k: sum(p['injected_positives'] for r in catalog['sources'] for p in r['pools'] if p['capacity'] == k) for k in (200, 500, 1000)},
        'full_source_content_hashes_verified': catalog_complete['content_hashes_verified'],
    }
    samples = summary['samples']
    assert len(samples) == 159 and all(s['valid'] and s['real_model_and_tools'] and not s['failure'] for s in samples)
    paired_ids = {k: {(s['task'], s['query_id']) for s in samples if s['capacity'] == k} for k in (200, 500, 1000)}
    assert paired_ids[200] == paired_ids[500] == paired_ids[1000]
    assert len(paired_ids[200]) == 53
    tiers = []
    for capacity in (200, 500, 1000):
        selected = [s for s in samples if s['capacity'] == capacity]
        assert len(selected) == 53
        tiers.append({'capacity': capacity, 'n': len(selected),
                      'recall_at_10': mean(s['metrics']['recall_at_10'] for s in selected),
                      'ndcg_at_10': mean(s['metrics']['ndcg_at_10'] for s in selected),
                      'f1_at_10': mean(s['metrics']['f1_at_10'] for s in selected),
                      'zero_recall': sum(s['metrics']['recall_at_10'] == 0 for s in selected),
                      'mean_wall_seconds': mean(s['timing']['wall_time_seconds'] for s in selected)})

    svg_path = ROOT / 'docs/assets/project-progress-2026-10-28/workflow.svg'
    rules_svg_path = ROOT / 'docs/assets/project-progress-2026-10-28/harness-rules.svg'
    freeze = ROOT / 'releases/protocol-freezes/20261026-pool-smoke-v4'
    template_path = ROOT / 'docs/templates/project_progress.html'
    rules_template_path = ROOT / 'docs/templates/harness_rules.html'
    evidence = [
        ('来源收集矩阵（2026-10-23）', args.matrix),
        ('当前 53 来源 / 159 池 catalog', args.catalog),
        ('统一候选池完成记录', catalog_complete_path),
        ('KaLM 补正例前质量报告', args.quality),
        ('159 次真实 smoke 汇总', summary_path),
        ('smoke 完成标记', complete_path),
        ('执行计划与实际冻结包', launch_path),
        ('当前启动入口', ROOT / 'scripts/launch_pool_smoke.sh'),
        ('候选池协议和预算', ROOT / 'dci_bench/protocol/document_pools.json'),
        ('当前冻结三档契约', freeze / 'contracts.json'),
        ('当前冻结预算控制与输出修复源码', freeze / 'runtime/pi-dci/scripts/dci-run.ts'),
        ('当前冻结评分与汇总口径', freeze / 'runtime/dci_bench/results/protocol.py'),
        ('报告模板', template_path), ('流程图源文件', svg_path),
        ('Harness 规则图源文件', rules_svg_path),
        ('Harness 阅读流程模板', rules_template_path),
        ('报告生成器', Path(__file__).resolve()),
    ]
    report = {'schema_version': 'dci-project-progress-v1', 'as_of': DATE, 'timezone': 'Asia/Shanghai',
              'collection_counts': matrix['counts'], 'current': current, 'sources': sources,
              'kalm': quality['aggregate'], 'smoke': {'run_id': args.run.name, 'tiers': tiers,
                  'completed': 159, 'valid': 159, 'paired_queries': 53, 'full_benchmark_completed': False},
              'evidence': [{'label': label, 'path': str(path)} for label, path in evidence],
              'verification_scope': 'Lightweight metadata, catalog/summary/launch-plan completion hashes, pool completion file presence, paired smoke summaries; no corpus rescan, no inference.'}
    stats = [(99, '收集来源视图'), (53, '已构池来源'), (159, '候选池'),
             (196088, 'Query 实例'), ('159/159', 'Smoke 执行通过')]
    assert current['cells'] == 49 and len(current['domains']) == 8 and len(current['languages']) == 21
    stats_html = ''.join(f'<div class="stat"><strong>{value:,}</strong><span>{label}</span></div>'
                         if isinstance(value, int) else f'<div class="stat"><strong>{value}</strong><span>{label}</span></div>'
                         for value, label in stats)
    page = template_path.read_text(encoding='utf-8').replace('__DATE__', DATE).replace('__STATS__', stats_html)
    page = page.replace('__SVG__', svg_path.read_text(encoding='utf-8'))
    page = page.replace('__RULES_CONTENT__', rules_template_path.read_text(encoding='utf-8'))
    page = page.replace('__DATA__', json.dumps(report, ensure_ascii=False).replace('<', '\\u003c'))
    markdown = f'''# DCI-Bench 项目进展（{DATE}）

当前已完成 53 个来源的三档候选池构造，并用 Qwen3.8-27B 跑通 159 次真实 smoke；全量 benchmark 尚未完成。

- [交互式数据与进展展示](index.html) · [框架流程图](workflow.svg) · [Harness 预算与修复规则图](harness-rules.svg)
- 本文聚焦文档检索；代码评测保持独立契约，不计入以下统计。

## 数据与候选池

- 已整理 99 个来源视图：56 个具备三档规模与标签条件、17 个待复核、21 个小语料、5 个接口样例。
- 收集矩阵有 50 个就绪领域×语言格；16 格有原生来源、29 格仅有译本、5 格属性未知或混合。
- 当前构池覆盖 53 个来源、8 个领域、21 种语言、49 个领域×语言格。
- 共 196,088 个 query 实例、234,570 个正例关系，构建 200 / 500 / 1000 三档，共 159 个池。
- 这些是来源视图与来源内实例数；平行译本、共享语料和同源派生集未做全局独立性去重。
- `beir_scifact_test`、`browsecomp_plus`、`enterprise_rag` 在收集矩阵中就绪，但未纳入当前 53 来源池目录。
- 每任务保留一份共享正文，每题存候选 doc_id；原始排名无损压缩，运行时只物化当前题工作区。

## 当前 harness 与完整流程

- 三档 profile 为 `document-pool-v1-{{small,medium,large}}`，harness 为 `dci-mvp-v3.4-pilot`；冻结包 `20261026-pool-smoke-v4`。
- 三类输入（已有检索集、带证据 QA、原始语料）经审计统一成 corpus + queries + qrels；evidence 为可选辅助。
- KaLM 原始排名用于独立构造三档池；补回遗漏正例并等量移除非正例，三档不保证嵌套。
- 冻结数据、源码和配置 → 预检 → Inspect 编排 → HTTP bridge → Pi → vLLM / Qwen3.8-27B。
- `read` 在 Pi Node 进程内受控执行；`bash` 经 Bubblewrap 只读候选语料、无网络，qrels 仅由宿主持有。

Harness 是单题运行的控制程序；模型负责找相关文档，Harness 按下面的顺序管理过程。

1. **检索**：模型在指定候选文档中搜索、用 `read` 打开正文，目标是提交 1–10 个按相关性排序的文档 ID。评分标签不交给模型，避免答案泄漏。
2. **记账**：每次调用的实际输入和输出都累加，历史重传再次计入；这样可以控制整题开销。三档累计上限为 96k / 128k / 192k；单次上下文另限 65,536、输出限 1,024，两类上限分别约束整题与单次调用。
3. **预留**：为避免搜索花光额度，随历史增长重估“一轮检索 + 一次提交 + 一次修复”的成本。预留属于原预算；按下一次输入估计 N，当前公式为 max(24k/32k/48k, 3N+58,432)。
4. **停止搜索**：余额不高于预留，或相同参数工具调用连续 3 次，就干预。已读正文则关闭工具、整理结果；未读则最多 2 轮只读补证据。6 轮未读也触发补读；读到且余额充足可以恢复搜索。
5. **验收与修复**：检查 JSON、1–10 个唯一且存在的 ID，以及整题至少一次非空正文读取。失败时反馈错误，最多修复 3 次；收尾前可继续工具交互，收尾后只能用已有内容重交，防止借修复重新搜索。

- 预留只估算一次修复，不保证三次都有额度；每次请求仍受预算约束。每题 600 秒为独立硬超时，不会提前触发收尾；最多 4 题并发。
- 验收不逐 ID 强制匹配读取记录；格式合格也不代表检索正确。验收后用隐藏标签计算 Recall / F1 / nDCG，并保留排名、轨迹和成本。
- 运行硬失败直接终止；3 次修复后仍无效记 invalid_json。无效样本在主要平均指标中计零，有效样本指标另报。

## 已取得的结果

- 补正例前 KaLM 的全部 query 平均 Recall@200/500/1000 为 93.57% / 95.69% / 96.92%；这是离线检索覆盖。
- 真实 smoke 固定每来源一题、同题三档：159/159 完成、输出有效且通过运行验收。

| 候选池大小 | Query 数 | Recall@10 | nDCG@10 | 平均耗时 | 零召回题数 |
|---|---:|---:|---:|---:|---:|
'''
    for tier in tiers:
        markdown += f"| {tier['capacity']} | {tier['n']} | {tier['recall_at_10']:.2%} | {tier['ndcg_at_10']:.2%} | {tier['mean_wall_seconds']:.1f}s | {tier['zero_recall']} |\n"
    markdown += f'''
- 池内 DCI 指标与补正例前 KaLM 指标分开解释；smoke 通过不代表每题都检索正确。
- 三档同时改变候选集合与 token 预算；当前结果不能归因于单独扩大池大小，也不是完整 benchmark。
- 本次展示核对清单与已有结果，没有重跑推理；全库 `content_hashes_verified=false` 边界保留。
- smoke 的正文严格校验仅覆盖被选中的候选文档；全库正文未在本次重新扫描。

## 下一步与证据

- 扩大固定 query pilot，检查弱覆盖来源与零召回轨迹；再做同题模型对照与全量矩阵。
- 当前运行：`{args.run.name}`；依据为该目录的 `summary.json`、`_COMPLETE` 与 `launch-plan.json`。
- 数据依据：`retrieval_pools/catalog.json`、`audit/kalm_quality/kalm_pre_injection_quality.json` 与 2026-10-23 收集矩阵。
- [本报告输入/输出摘要](manifest.json) · [完整统计快照](data_matrix.json)；HTML 内有全部证据路径。
'''
    assert len(markdown.splitlines()) <= 80
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'index.html').write_text(page, encoding='utf-8')
    (args.output / 'workflow.svg').write_text(svg_path.read_text(encoding='utf-8'), encoding='utf-8')
    (args.output / 'harness-rules.svg').write_text(rules_svg_path.read_text(encoding='utf-8'), encoding='utf-8')
    (args.output / 'progress.md').write_text(markdown, encoding='utf-8')
    (args.output / 'data_matrix.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fields = ['source', 'domain', 'language', 'translation_kind', 'documents', 'queries', 'positive_pairs', 'inventory_role', 'pool_built', 'kalm_recall_1000', 'source_path']
    with (args.output / 'source_matrix.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sources)
    manifest = {'schema_version': 'dci-project-progress-manifest-v1', 'as_of': DATE,
                'inputs': [{'path': str(path), 'sha256': digest(path)} for _, path in evidence],
                'outputs': {p.name: digest(p) for p in sorted(args.output.iterdir())},
                'checks': {'catalog_completion_hash': True, 'smoke_summary_completion_hash': True,
                           'launch_plan_completion_hash': True, 'pool_completion_files_present': 159,
                           'paired_queries': 53, 'real_model_tool_samples': 159, 'markdown_lines': len(markdown.splitlines()),
                           'no_inference_started': True, 'full_source_hash_scan': False}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'checks': manifest['checks'], 'current': current}, ensure_ascii=False))


if __name__ == '__main__':
    main()
