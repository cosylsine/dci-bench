#!/usr/bin/env python3
"""Render the verified, acquired domain/language matrix (no inference)."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dci_bench.data.pipeline.core import verify_release

LANGUAGES = ['en', 'zh', 'ar', 'de', 'el', 'es', 'hi', 'ru', 'th', 'tr', 'vi', 'fr', 'ja']
DOMAINS = ['scientific', 'enterprise', 'software', 'encyclopedia']
DOMAIN_ZH = {'scientific': '科研', 'enterprise': '企业', 'software': '软件', 'encyclopedia': '百科'}
NAMES = {'beir_scifact_test': 'SciFact test', 'qasper_dev': 'QASPER dev',
         'wixqa_expertwritten': 'WixQA expert', 'wixqa_simulated': 'WixQA simulated',
         'worksurface_backend_developer': 'WorkSurface backend',
         'worksurface_logistics_manager': 'WorkSurface logistics',
         'worksurface_operations_manager': 'WorkSurface operations'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--override', action='append', default=[], metavar='SOURCE=RELEASE')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    rows = []
    overrides = dict(item.split('=', 1) for item in args.override)
    for release in sorted((args.eval_root / 'pipeline_v1').glob('*/v1')):
        if release.parent.name in overrides:
            name = overrides[release.parent.name]
            if Path(name).name != name:
                raise ValueError('Override must be a release basename')
            release = release.parent / name
        manifest = verify_release(release)
        spec = manifest['specification']
        audit = json.loads((release / 'audit.json').read_text())
        excluded_evidence = sum(e.get('kind') == 'evidence' and e.get('reason') == 'invalid_quote_or_offset'
                                for e in audit.get('exclusions', []))
        evidence_rows = [json.loads(line) for line in (release / 'evidence.jsonl').open() if line.strip()]
        evidence = len(evidence_rows)
        evidence_queries = len({item['query_id'] for item in evidence_rows})
        metadata = json.loads((release / 'bundle_metadata.json').read_text())
        review_queries = len({item['query_id'] for item in metadata.get('review_queue', []) if item.get('query_id')})
        translation = Counter()
        for line in (release / 'queries.jsonl').open():
            if line.strip():
                q = json.loads(line)
                translation[q.get('metadata', {}).get('translation_status', 'unknown')] += 1
        source = release.parent.name
        state = 'Q+R+E' if evidence and manifest['num_queries'] else 'Q+R' if manifest['num_queries'] else 'C'
        rows.append({'source': source, 'name': NAMES.get(source, 'XQuAD' if source.startswith('xquad_') else 'Kubernetes'),
                     'domain': spec['domain'], 'language': spec['language'],
                     'documents': manifest['num_documents'], 'queries': manifest['num_queries'],
                     'positive_pairs': manifest['num_positive_pairs'], 'evidence': evidence,
                     'evidence_queries': evidence_queries, 'review_queries': review_queries,
                     'evidence_complete': evidence_queries == manifest['num_queries'] and evidence_queries > 0,
                     'state': state, 'release_state': manifest['state'], 'translation': dict(translation),
                     'excluded_evidence': excluded_evidence,
                     'issue': '证据待复核' if review_queries else '证据转换待修复' if excluded_evidence else '',
                     'release': str(release), 'bundle_hash': manifest['bundle_hash']})
    assert all(r['domain'] in DOMAINS and r['language'] in LANGUAGES for r in rows)
    cells = {(d, lang): [r for r in rows if (r['domain'], r['language']) == (d, lang)]
             for d in DOMAINS for lang in LANGUAGES}
    ws = [r for r in rows if r['source'].startswith('worksurface_')]
    payload = {'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               'domains': DOMAINS, 'languages': LANGUAGES, 'sources': rows,
               'state_legend': {'C': '仅 corpus，尚无 query/qrels', 'Q+R': 'corpus、query、qrels 已齐全；未导出显式证据',
                                'Q+R+E': 'corpus、query、qrels、显式证据已齐全'},
               'pending': [{'source': 'TyDiQA GoldP dev', 'state': '用户要求暂缓；未下载成功',
                            'languages': ['en', 'ar', 'bn', 'fi', 'id', 'sw', 'ko', 'ru', 'te']}],
               'boundaries': ['XQuAD 非英语版本是平行译本，不是独立原生问题。',
                              'WixQA 两个 query 集共享 6,221 篇正文，不能重复累计独立文档。',
                              'E 仅表示清洗版本已导出 evidence，不断言原始来源完全没有证据。',
                              f'当前 WorkSurface 有 {sum(r["evidence"] for r in ws)} 条精确证据、{sum(r["review_queries"] for r in ws)} 条 query 待复核；历史版本保留。',
                              'labels_ready 不表示穷尽标注、人工复核完成或已完成模型评测。']}
    args.output.mkdir(parents=True)
    (args.output / 'matrix.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    with (args.output / 'matrix.csv').open('w', newline='') as stream:
        fields = ['domain', 'language', 'source', 'state', 'documents', 'queries', 'positive_pairs', 'evidence', 'evidence_queries', 'review_queries', 'release_state', 'excluded_evidence', 'issue']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    render_html(args.output, payload, cells)
    render_plot(args.output, payload, cells)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()}
    (args.output / 'manifest.json').write_text(json.dumps({'created_at': payload['created_at'], 'files': files,
                                                        'source_bundles': {r['source']: r['bundle_hash'] for r in rows}}, indent=2))
    print(json.dumps({'output': str(args.output), 'sources': len(rows), 'occupied_cells': sum(bool(v) for v in cells.values()),
                      'queries': sum(r['queries'] for r in rows)}, ensure_ascii=False), flush=True)


def render_html(out, payload, cells):
    table = '<thead><tr><th>领域 / 语言</th>' + ''.join(f'<th>{x}</th>' for x in LANGUAGES) + '</tr></thead><tbody>'
    for d in DOMAINS:
        table += f'<tr><th>{DOMAIN_ZH[d]}<small>{d}</small></th>'
        for lang in LANGUAGES:
            items = cells[d, lang]
            table += '<td>'
            if not items:
                table += '<div class="empty" data-state="empty">—<small>未接入</small></div>'
            for r in items:
                t = '译本' if r['translation'].get('translated') else '原生标记' if r['translation'].get('native') else '翻译状态未声明' if r['queries'] else '无 query'
                cls = {'C': 'corpus', 'Q+R': 'labels', 'Q+R+E': 'evidence'}[r['state']]
                table += (f'<article class="{cls}" data-state="{cls}"><strong>{html.escape(r["name"])}</strong>'
                          f'<span class="badge">{r["state"]}</span><p>{r["documents"]:,} 文档 · {r["queries"]:,} queries</p>'
                          f'<p>{r["positive_pairs"]:,} 正例关系 · {r["evidence"]:,} 证据</p><small>{t}</small>'
                          + (f'<p style="color:#a43c25;font-weight:600">⚠ {r["issue"]}：{r["excluded_evidence"]} 条被拒收</p>' if r['issue'] else '') +
                          f'<details><summary>来源与状态</summary><code>{r["source"]}</code><p>{r["release_state"]}</p>'
                          f'<code>{html.escape(r["release"])}</code></details></article>')
            table += '</td>'
        table += '</tr>'
    table += '</tbody>'
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DCI-Bench 数据矩阵 · 2026-10-22</title><style>
*{box-sizing:border-box}body{margin:0;padding:30px;background:#f4f7fb;color:#182943;font:15px system-ui,sans-serif}h1{margin:0;font-size:30px}p{line-height:1.6}.sub{color:#50647c}.stats{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}.stats span{padding:12px 18px;background:white;border-radius:10px;border:1px solid #dce3ed}.legend{display:flex;gap:20px;flex-wrap:wrap;margin:18px 0}.legend b{padding:3px 7px;border-radius:4px}.wrap{overflow:auto;border:1px solid #dce3ed;border-radius:12px;background:white;max-height:75vh}table{border-collapse:separate;border-spacing:0;min-width:2450px;width:100%}th,td{border-right:1px solid #e6ebf1;border-bottom:1px solid #e6ebf1;padding:12px;vertical-align:top;min-width:160px}thead th{background:#192d49;color:white;position:sticky;top:0;z-index:3;text-align:left}th:first-child{position:sticky;left:0;z-index:2;background:#edf2f8;min-width:150px;color:#203651}thead th:first-child{z-index:4;background:#192d49;color:white}td:nth-child(2){min-width:270px}small{display:block;color:#596f88;font-size:12px;margin-top:5px}article{border:1px solid;border-radius:8px;padding:10px;margin-bottom:8px;transition:opacity .15s}article p{font-size:12px;margin:7px 0}strong{display:block;font-size:14px;margin-bottom:8px}.corpus{background:#fff6df;border-color:#e7ca86}.labels{background:#e8f6ef;border-color:#a4d2b8}.evidence{background:#eaf2ff;border-color:#a5bee9}.badge{font:11px monospace;font-weight:700;background:#fff9;padding:3px 6px;border-radius:4px}.empty{text-align:center;color:#a3afc1;padding:22px 0}.empty small{color:#8c9aad}.muted{opacity:.16}.notes{max-width:1150px;background:white;padding:18px 24px;border-radius:12px;margin-top:24px}li{line-height:1.7;margin:5px 0}select{padding:8px;border:1px solid #bbc9db;border-radius:6px;background:white}details{margin-top:8px;font-size:11px}code{word-break:break-all}a{color:#275a96}footer{margin-top:16px;color:#5c708a;font-size:12px}
</style><h1>DCI-Bench · Domain × Language</h1><p class="sub">已落盘数据快照 · 2026-10-22（北京时间） · 状态按当前清洗版本计算</p>
<div class="stats"><span><b>4</b> 领域</span><span><b>13</b> 语料语言 / <b>11</b> query 语言</span><span><b>18 / 52</b> 已占矩阵格</span><span><b>23</b> 来源版本</span><span><b>14,433</b> query 实例</span></div>
<div class="legend"><span><b class="corpus">C</b> 仅 corpus</span><span><b class="labels">Q+R</b> query + qrels 齐全</span><span><b class="evidence">Q+R+E</b> 另有显式证据</span><span>— 尚未接入</span></div>
<p><label>突出显示：<select id="state"><option value="all">全部</option><option value="corpus">仅 corpus</option><option value="labels">有 query/qrels，未导出显式证据</option><option value="evidence">query/qrels/证据齐全</option><option value="empty">尚未接入的格子</option></select></label>　可横向滚动；点击单项“来源与状态”查看详细路径。</p>
<div class="wrap"><table>TABLE</table></div><div class="notes"><b>如何解读</b><ul>
<li>Q = query，R = qrels 正例关系，E = 清洗版本导出的显式证据。所有非空格都有 corpus。</li>
<li>除百科外，当前带标签数据集中于英语；软件已有 5 种语言的语料，尚未生成并复核 query。</li>
<li>XQuAD 每语言 240 文档、1,190 个问题；10 个非英语版本为平行译本，不能算作独立原生覆盖。单语言不足以构造严格 500 / 1000 文档池。</li>
<li>两个 WixQA query 集共用 6,221 篇文档。WorkSurface 的 3 类角色合计 390 文档、178 queries。</li>
<li><b>WorkSurface：</b>当前所选版本保留 WS_EVIDENCE 条精确证据，另有 WS_REVIEW 个 query 待复核。evidence-v2 修复了 span 映射，query/qrels 未改；旧 v1 保留。</li>
<li>Q+R 表示当前未导出显式 evidence，不意味着原始来源不存在证据。labels_ready 也不等于穷尽标注、人工复核或完成模型评测。</li>
<li>TyDiQA GoldP 暂缓：计划 en/ar/bn/fi/id/sw/ko/ru/te，当前下载未成功，因此不填入已接入格。</li>
<li>PMC JATS、EUR-Lex、EDGAR 目前只有 connector / 格式测试，未形成实际数据格；不计入领域覆盖。</li>
</ul><p><a href="matrix.png">下载 PNG</a> · <a href="matrix.svg">下载 SVG</a> · <a href="matrix.csv">下载 CSV</a> · <a href="matrix.json">完整 JSON</a></p></div>
<footer>来源：pipeline_v1/*/v1；生成前逐项验证 release 清单与哈希。旧版本、原始标签及历史报告保持不变。</footer>
<script>document.getElementById('state').addEventListener('change',e=>{document.querySelectorAll('[data-state]').forEach(n=>n.classList.toggle('muted',e.target.value!=='all'&&n.dataset.state!==e.target.value))});</script></html>'''
    ws = [r for r in payload['sources'] if r['source'].startswith('worksurface_')]
    page = page.replace('WS_EVIDENCE', str(sum(r['evidence'] for r in ws))).replace('WS_REVIEW', str(sum(r['review_queries'] for r in ws)))
    (out / 'index.html').write_text(page.replace('TABLE', table), encoding='utf-8')


def render_plot(out, payload, cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(25, 11), dpi=140)
    fig.patch.set_facecolor('#f5f7fb')
    ax.set_facecolor('#f5f7fb')
    ax.set_xlim(0, 25); ax.set_ylim(0, 11); ax.axis('off')
    ax.text(.35, 10.55, 'DCI-Bench | Domain x Language', fontsize=23, weight='bold', color='#182943')
    ax.text(.35, 10.14, 'Acquired data only | 2026-10-22 | 23 source versions | 18 occupied cells | 14,433 query instances', fontsize=12, color='#52677f')
    colors = {'C': '#fff0c9', 'Q+R': '#dff1e6', 'Q+R+E': '#e2edff', 'empty': '#edf0f5'}
    for x, state, label in [(0.35, 'C', 'Corpus only'), (4.1, 'Q+R', 'Queries + qrels'), (9.0, 'Q+R+E', 'Queries + qrels + evidence')]:
        ax.add_patch(FancyBboxPatch((x, 9.45), .95, .35, boxstyle='round,pad=0.03', fc=colors[state], ec='none'))
        ax.text(x + .06, 9.57, state, fontsize=11, weight='bold')
        ax.text(x + 1.12, 9.57, label, fontsize=11, color='#364c67')
    ax.text(17.6, 9.57, 'D = documents   Q = queries   T = translated', fontsize=11, color='#364c67')
    x0, label_w, en_w, col_w = .35, 2.3, 3.45, 1.49
    xs = [x0 + label_w, x0 + label_w + en_w]
    other_start = xs[-1]
    xs.extend(other_start + col_w * i for i in range(1, 12))
    y_header = 8.85
    for i, lang in enumerate(LANGUAGES):
        ax.text(xs[i] + .10, y_header, lang.upper(), fontsize=12, weight='bold', color='#203651')
    row_specs = [('scientific', 6.85, 1.68), ('enterprise', 4.70, 1.98), ('software', 3.15, 1.38), ('encyclopedia', 1.60, 1.38)]
    for domain, y, height in row_specs:
        ax.text(x0, y + height - .28, domain.capitalize(), fontsize=15, weight='bold', color='#203651')
        for i, lang in enumerate(LANGUAGES):
            items = cells[domain, lang]
            state = 'Q+R+E' if any(r['state'] == 'Q+R+E' for r in items) else items[0]['state'] if items else 'empty'
            w = en_w if i == 0 else col_w
            ax.add_patch(FancyBboxPatch((xs[i], y), w - .07, height, boxstyle='round,pad=0.015', fc=colors[state], ec='#d6dfeb', lw=.6))
            if not items:
                ax.text(xs[i] + (w - .07)/2, y + height/2, '-', ha='center', va='center', color='#a4b0c0', fontsize=18)
                continue
            if domain == 'scientific':
                lines = ['SciFact  [Q+R]', '5,183 D / 300 Q', '', 'QASPER  [Q+R+E]', '281 D / 465 Q']
            elif domain == 'enterprise':
                worksurface = [item for item in items if item['source'].startswith('worksurface_')]
                valid = sum(item['evidence'] for item in worksurface)
                review = sum(item['review_queries'] for item in worksurface)
                ws_state = 'Q+R+E' if valid else 'Q+R'
                lines = ['WixQA expert + simulated [Q+R]', '6,221 shared D / 200 + 200 Q', '', f'WorkSurface x3  [{ws_state}]*', '390 D / 64 + 48 + 66 Q', f'*{valid} evidence / {review} pending review']
            elif domain == 'software':
                lines = ['Kubernetes', '[C]', f'{items[0]["documents"]} D / 0 Q']
            else:
                lines = ['XQuAD' + (' (T)' if lang != 'en' else ''), '[Q+R+E]', '240 D', '1,190 Q']
            ax.text(xs[i] + .10, y + height - .17, '\n'.join(lines), va='top', fontsize=10 if i == 0 else 9, linespacing=1.45, color='#203651')
    ax.text(.35, 1.08, 'Coverage gap: labelled scientific / enterprise data is English-only; software cells still need generated and reviewed queries.', fontsize=11, color='#364c67')
    ax.text(.35, .70, 'XQuAD: 1,190 parallel question families, not 13,090 independent questions. Q+R means no normalized evidence exported.', fontsize=11, color='#364c67')
    ax.text(.35, .32, 'Deferred: TyDiQA (download unavailable). Parser-only: PMC JATS / EUR-Lex / EDGAR. No cell is counted from adapters alone.', fontsize=11, color='#364c67')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out / 'matrix.png', dpi=140)
    fig.savefig(out / 'matrix.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
