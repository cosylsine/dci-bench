#!/usr/bin/env python3
"""Render a retrieval-first inventory from sealed releases and research receipts.

Size eligibility is separate from labels, optional evidence, and executed pools.
Historical snapshots and unacquired research never become newly verified data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dci_bench.data.pipeline.core import verify_release

TIERS = (200, 500, 1000)
DOMAIN_ALIASES = {'financial': 'finance', 'health': 'medical', 'encyclopaedic': 'encyclopedia',
                  'Wikipedia': 'encyclopedia', 'wikipedia': 'encyclopedia', 'web_search': 'web'}
SOURCE_DOMAIN_OVERRIDES = {
    'gapfill_mteb_germandpr': {
        'domain': 'encyclopedia',
        'reason': 'GermanDPR contexts are German Wikipedia passages; web in the original intake specification was too broad.',
        'reference': 'https://huggingface.co/datasets/deepset/germandpr',
    },
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def translation_kind(value):
    value = str(value or '').strip().lower()
    if value in {'native', 'non_translated', 'non-translated', 'not translated', 'untranslated'} or value.startswith('native '):
        return 'native'
    if value == 'translated' or 'translated' in value or 'translation' in value:
        return 'translated'
    return 'unknown'


def translation_coverage(rows):
    cells = {}
    for row in rows:
        if row['inventory_role'] == 'three_tier_source':
            cells.setdefault((row['domain'], row['language']), set()).add(translation_kind(row.get('translation')))
    return {'cells_with_native_source': sum('native' in kinds for kinds in cells.values()),
            'translated_only_cells': sum(kinds == {'translated'} for kinds in cells.values()),
            'remaining_unknown_or_mixed_cells': sum('native' not in kinds and kinds != {'translated'} for kinds in cells.values())}


def classify(row):
    """Do not use optional evidence as a retrieval readiness requirement."""
    if 'domain' in row:
        row.setdefault('source_domain', row['domain'])
        row['domain'] = DOMAIN_ALIASES.get(row['domain'], row['domain'])
        if row['source'] in SOURCE_DOMAIN_OVERRIDES:
            row['domain_correction'] = SOURCE_DOMAIN_OVERRIDES[row['source']]
            row['domain'] = row['domain_correction']['domain']
    n = row['documents']
    row['size_supported_tiers'] = [k for k in TIERS if n >= k]
    row['preferred_corpus_gt1000'] = n > 1000
    row['has_query_qrels'] = row['queries'] > 0 and row['positive_pairs'] > 0
    row['retrieval_state'] = 'Q+R' if row['has_query_qrels'] else 'C'
    if row['source'].startswith('github_kubernetes_website_') and n <= 5 and not row['queries']:
        role = 'connector_demo'
    elif not row['has_query_qrels']:
        role = 'corpus_only'
    elif n < 1000:
        role = 'small_corpus'
    elif row['release_state'] not in {'labels_ready', 'legacy_frozen_only'}:
        role = 'review_required'
    else:
        role = 'three_tier_source'
    row['inventory_role'] = role
    row['translation_kind'] = translation_kind(row.get('translation'))
    row['candidate_pool_status'] = 'not_built_in_this_expansion'
    return row


def from_release(path):
    manifest = verify_release(path)
    spec = manifest['specification']
    source_meta = {key: spec[key] for key in ('source_family', 'source_scope', 'translation_status', 'translation_of',
                                             'official_card', 'official_task', 'mirror', 'mirror_revision') if key in spec}
    source_meta.update(spec.get('metadata', {}))
    evidence = [json.loads(line) for line in (path / 'evidence.jsonl').open() if line.strip()]
    queries = [json.loads(line) for line in (path / 'queries.jsonl').open() if line.strip()]
    qrels = read(path / 'qrels.json')
    meta = read(path / 'bundle_metadata.json')
    audit = read(path / 'audit.json')
    preparation = audit.get('preparation', {})
    quality = {
        'input_queries': preparation.get('input_counts', {}).get('queries'),
        'excluded_queries': len(preparation.get('excluded_query_ids', [])),
        'excluded_documents': len(preparation.get('excluded_document_ids', [])),
        'review_queries': len({item['query_id'] for item in audit.get('review_required', []) if item.get('query_id')}),
        'optional_spans_not_used': len(meta.get('provenance', {}).get('ignored_optional_spans', [])),
        'audit_path': str(path / 'audit.json'),
    }
    pos = Counter(sum(float(score) > 0 for score in rel.values()) for rel in qrels.values())
    translation = Counter(q.get('metadata', {}).get('translation_status', 'unknown') for q in queries)
    translation_value = next(iter(translation)) if len(translation) == 1 else 'mixed' if translation else 'unknown'
    translation_basis = 'query_metadata'
    if translation_value == 'unknown' and source_meta.get('translation_status'):
        translation_value = source_meta['translation_status']
        translation_basis = 'source_metadata_claim'
    return classify({
        'origin': 'pipeline', 'source_id': spec['id'], 'source': spec['id'],
        'source_path': str(path), 'domain': spec['domain'], 'language': spec['language'],
        'documents': manifest['num_documents'], 'queries': manifest['num_queries'],
        'positive_pairs': manifest['num_positive_pairs'], 'evidence': len(evidence),
        'evidence_queries': len({r['query_id'] for r in evidence}),
        'evidence_pending_review': len({r.get('query_id') for r in meta.get('review_queue', []) if r.get('query_id')}),
        'translation': translation_value, 'translation_basis': translation_basis,
        'release_state': manifest['state'], 'revision': spec.get('revision'),
        'license': spec.get('license', 'unknown'), 'dataset': manifest.get('source', {}).get('dataset'),
        'source_metadata': source_meta,
        'source_family': source_meta.get('source_family', spec['id']),
        'task_origin': 'QA 派生检索' if manifest.get('input_kind') == 'evidence_qa' else '已有检索任务' if manifest.get('input_kind') == 'retrieval' else '语料接入',
        'quality': quality,
        'verification': 'release_inventory_and_bundle_hash_verified',
        'completion_sha256': sha(path / '_COMPLETE.json'),
        'max_positives_per_query': max(pos, default=0),
        'positive_overflow_queries': {str(k): sum(count for size, count in pos.items() if size > k) for k in TIERS},
        'notes': source_meta.get('source_scope', '') + ' Corpus size eligibility only; per-query positive overflow and source quality must be checked before pools.',
    })


def coverage_change(before, after):
    """Count cells once even when multiple sources or parallel views share one."""
    def cells(rows, roles):
        return {(r['domain'], r['language']) for r in rows if r['inventory_role'] in roles}
    ready_before = cells(before, {'three_tier_source'})
    ready_after = cells(after, {'three_tier_source'})
    sized_before = cells(before, {'three_tier_source', 'review_required'})
    sized_after = cells(after, {'three_tier_source', 'review_required'})
    return {
        'ready_before': len(ready_before), 'ready_after': len(ready_after),
        'new_ready_cells': [list(cell) for cell in sorted(ready_after - ready_before)],
        'new_size_eligible_cells': [list(cell) for cell in sorted(sized_after - sized_before)],
        'pending_only_cells': [list(cell) for cell in sorted(sized_after - ready_after)],
    }


def render_html(out, data):
    escape = lambda x: html.escape(str(x), quote=True)
    roles = {'three_tier_source': '三档来源', 'small_corpus': '小语料', 'connector_demo': '接口样例',
             'corpus_only': '仅语料', 'review_required': '待复核'}
    rows = data['sources']
    cards = []
    for row in rows:
        tiers = ' / '.join(map(str, row['size_supported_tiers'])) or '无'
        values = [row['source'], row.get('source_family', row['source']), row.get('task_origin', '冻结任务'), row['translation'], row['documents'], row['queries'], row['positive_pairs'], tiers,
                  roles[row['inventory_role']], row['evidence'], row['release_state']]
        overflow = row.get('positive_overflow_queries', {})
        warning = f'<p>超过池容量的 query：{escape(overflow)}</p>' if any(overflow.values()) else ''
        quality = row.get('quality', {})
        if row.get('domain_correction'):
            correction = row['domain_correction']
            warning += f'<p>领域归类修正：{escape(correction["reason"])} <a href="{escape(correction["reference"])}">来源说明</a>；原记录保留为 {escape(row["source_domain"])}。</p>'
        if quality.get('excluded_queries') or quality.get('excluded_documents') or quality.get('review_queries'):
            warning += f'<p>清洗排除 {quality.get("excluded_documents", 0)} 篇文档、{quality.get("excluded_queries", 0)} 个 query；保留待复核 {quality.get("review_queries", 0)} 个 query。详见 audit.json。</p>'
        details = f"{warning}<details><summary>来源 / 校验</summary><code>{escape(row['source_path'])}</code><p>{escape(row.get('verification'))}</p><p>{escape(row.get('notes', ''))}</p></details>"
        cards.append(f"<tr data-role='{row['inventory_role']}' data-new='{str(row['added_in_this_expansion']).lower()}' data-domain='{escape(row['domain'])}' data-language='{escape(row['language'])}' data-translation='{row['translation_kind']}'><td>{escape(row['domain'])}</td><td>{escape(row['language'])}</td>" + ''.join(f'<td>{escape(v)}</td>' for v in values) + f'<td>{details}</td></tr>')
    matrix = []
    for domain in data['domains']:
        cells = []
        for language in data['languages']:
            items = [r for r in rows if r['domain'] == domain and r['language'] == language]
            content = []
            for r in items:
                tier = '/'.join(map(str, r['size_supported_tiers'])) or '—'
                content.append(f"<article data-role='{r['inventory_role']}' data-new='{str(r['added_in_this_expansion']).lower()}' data-domain='{escape(domain)}' data-language='{escape(language)}' data-translation='{r['translation_kind']}'><b>{escape(r['source'])}</b><p>{r['documents']:,} D · {r['queries']:,} Q</p><p>{roles[r['inventory_role']]} · 档位 {tier}</p><small>{escape(r['retrieval_state'])} · {escape(r.get('task_origin', '冻结任务'))} · {escape(r['translation'])} · evidence {r['evidence']:,}（可选）</small></article>")
            cells.append('<td>' + (''.join(content) or '<span class="empty">—</span>') + '</td>')
        matrix.append(f'<tr><th>{escape(domain)}</th>{"".join(cells)}</tr>')
    research = ''.join(f'<details><summary>{escape(r["receipt_path"])}（{len(r["payload"].get("sources", r["payload"].get("tasks", [])))} 条研究记录）</summary><pre>{escape(json.dumps(r["payload"], ensure_ascii=False, indent=2))}</pre></details>' for r in data['research'])
    counts = data['counts']
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DCI 检索数据矩阵 · 2026-10-23</title><style>
*{box-sizing:border-box}body{margin:24px;background:#f5f7fb;color:#18314a;font:14px system-ui,sans-serif}h1{margin-bottom:8px}p{line-height:1.6}.stats,.controls{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}.stats span,.controls,section{background:white;border:1px solid #dce4ed;padding:14px;border-radius:8px}.wrap{overflow:auto;max-height:75vh;background:white;border:1px solid #dce4ed;margin:16px 0}table{border-collapse:separate;border-spacing:0;width:100%}th,td{border-bottom:1px solid #dae4ed;border-right:1px solid #dae4ed;padding:10px;vertical-align:top}th{position:sticky;top:0;background:#193650;color:white;z-index:2}th:first-child{position:sticky;left:0;z-index:3}article{min-width:220px;background:#e9f6ef;border:1px solid #b4d7c4;border-radius:7px;padding:9px;margin-bottom:7px}article[data-role=small_corpus]{background:#fff5df;border-color:#e5d2a1}article[data-role=connector_demo],article[data-role=corpus_only]{background:#eff1f4;border-color:#cad2dc}article[data-role=review_required]{background:#fff0dc;border-color:#e2b875}article p{margin:5px 0}small,.empty{color:#63788d}code{word-break:break-all}pre{white-space:pre-wrap;max-height:400px;overflow:auto}select,input{padding:6px}.hidden{display:none!important}a{color:#235c9d}section{margin-top:18px;max-width:1400px}li{margin:8px 0}
</style><h1>DCI-Bench · 检索数据矩阵</h1>
<p>按 query + corpus + qrels 与 200 / 500 / 1000 池规模筛选。evidence 是可选辅助信息。D 指来源的检索单元，可能是段落、摘要、回答或完整文档。</p>
<div class="stats">STATS</div><p>TRANSLATIONCOUNTS。原生 / 翻译沿用来源声明；未知不自动视为原生。已有译本与原库属于同一数据集组。</p><p>新增就绪格：NEWCELLS。按领域×语言去重；同一格加入多个来源只计一次。</p>
<div class="controls"><label>范围 <select id="role"><option value="scale_candidates">三档语料（含待复核）</option><option value="three_tier_source">三档来源且已就绪</option><option value="new">本轮新增来源</option><option value="all">全部已落盘来源</option><option value="small_corpus">小语料</option><option value="connector_demo">接口样例</option><option value="corpus_only">仅语料</option><option value="review_required">待复核</option></select></label><label>领域 <select id="domain"><option value="">全部</option>DOMAINS</select></label><label>语言 <select id="language"><option value="">全部</option>LANGUAGES</select></label><label>来源 <input id="search" type="search"></label></div>
<div class="controls"><label>语言来源 <select id="translation"><option value="">全部</option><option value="native">原生 / 未翻译</option><option value="translated">已有译本</option><option value="unknown">未知 / 混合</option></select></label></div>
<div class="wrap"><table id="matrix"><thead><tr><th>domain / language</th>LANGHEAD</tr></thead><tbody>MATRIX</tbody></table></div>
<h2>来源明细</h2><div class="wrap"><table id="sources"><thead><tr><th>领域</th><th>语言</th><th>来源</th><th>数据集组</th><th>任务来源</th><th>原生 / 翻译</th><th>D</th><th>Q</th><th>正例关系</th><th>规模支持档位</th><th>用途</th><th>可选 evidence</th><th>发布状态</th><th>来源</th></tr></thead><tbody>ROWS</tbody></table></div>
<section><h2>口径</h2><ul><li>文档级 Recall / MRR / nDCG 只需 query、corpus、qrels；evidence 用于 QA 转换、生成标签和诊断，不是已有检索标签的准入要求。</li><li>三档来源表示 corpus 至少 1000 且已有 query/qrels；优先原库大于 1000。这里不声称 KaLM 排名、三档池或模型评测已完成。每题正例超过 K 时必须另作处理，详情见 JSON 的 positive_overflow_queries。</li><li>Kubernetes 4–5 篇是接口样例，不计入默认覆盖。小语料保留历史记录；不复制文档、不混语言或跨任务凑规模。</li><li>同源派生版本、WixQA 共享语料、平行译本不可累加成独立文档或独立问题。统计均为来源视图数，没有全局去重文档总数。</li><li>旧冻结任务沿用原索引计数；pipeline 版本重新核验清单与哈希。qrels 是已有标注，未标注文档不等于已确认负例。偏小的父任务若本身不足 1000，需换来源或另立扩展协议。</li><li>新接入语料保持来源的检索粒度；QA 段落与论文全文不应混称为同一种“长文档”。</li></ul><p><a href="data_matrix.json">完整 JSON</a> · <a href="source_matrix.csv">CSV</a> · <a href="matrix.svg">规模覆盖 SVG</a> · <a href="manifest.json">文件校验清单</a> · <a href="cascade_plan.json">Cascade 命令计划（未执行）</a></p></section>
<section><h2>来源研究与未完成项</h2><p>以下记录含已下载、候选及受阻来源。只有经验证的本地 release 才进入上面的新增覆盖。</p>RESEARCH</section>
<script>function apply(){const role=document.querySelector('#role').value,domain=document.querySelector('#domain').value,lang=document.querySelector('#language').value,translation=document.querySelector('#translation').value,q=document.querySelector('#search').value.toLowerCase();document.querySelectorAll('article[data-role],#sources tr[data-role]').forEach(x=>x.classList.toggle('hidden',(role==='scale_candidates'?!['three_tier_source','review_required'].includes(x.dataset.role):(role==='new'?x.dataset.new!=='true':(role!=='all'&&x.dataset.role!==role)))||(domain&&x.dataset.domain!==domain)||(lang&&x.dataset.language!==lang)||(translation&&x.dataset.translation!==translation)||(q&&!x.textContent.toLowerCase().includes(q))));}document.querySelectorAll('select,input').forEach(x=>x.addEventListener('input',apply));apply();</script></html>'''
    options = lambda items: ''.join(f'<option>{escape(x)}</option>' for x in items)
    stats = [f'{counts["source_views"]} 已落盘来源视图', f'{counts["three_tier_source_views"]} 三档来源已就绪 / {counts["three_tier_review_views"]} 待复核',
             f'{data["coverage_change"]["ready_before"]} → {counts["three_tier_cells"]} 已就绪的领域×语言格', f'{counts["new_source_views"]} 本次新增视图']
    trans = data['translation_coverage']
    for token, value in {'STATS': ''.join(f'<span>{x}</span>' for x in stats),
                         'TRANSLATIONCOUNTS': f"就绪格中：{trans['cells_with_native_source']} 格有原生来源，{trans['translated_only_cells']} 格仅有译本，{trans['remaining_unknown_or_mixed_cells']} 格来源未知或混合",
                         'DOMAINS': options(data['domains']),
                         'LANGUAGES': options(data['languages']), 'LANGHEAD': ''.join(f'<th>{escape(x)}</th>' for x in data['languages']),
                         'MATRIX': ''.join(matrix), 'ROWS': ''.join(cards), 'RESEARCH': research,
                         'NEWCELLS': escape('、'.join(f'{domain} / {language}' for domain, language in data['coverage_change']['new_ready_cells']) or '暂无')}.items():
        page = page.replace(token, value)
    (out / 'index.html').write_text(page)


def render_svg(out, data):
    """Small exportable coverage summary, without external rendering packages."""
    domains, languages = data['domains'], data['languages']
    width, height = 190 + 76 * len(languages), 125 + 48 * len(domains)
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#f5f7fb"/><g font-family="sans-serif" fill="#18314a"><text x="18" y="28" font-size="20">DCI retrieval: ready sources for 200 / 500 / 1000</text><text x="18" y="52" font-size="12">Cells count source views, not unique corpora. Size eligibility is not completed pools.</text>']
    for j, lang in enumerate(languages):
        pieces.append(f'<text x="{190+j*76}" y="85" font-size="13">{html.escape(lang)}</text>')
    for i, domain in enumerate(domains):
        y = 102 + i * 48
        pieces.append(f'<text x="18" y="{y+25}" font-size="12">{html.escape(domain)}</text>')
        for j, lang in enumerate(languages):
            n = sum(r['inventory_role'] == 'three_tier_source' and r['domain'] == domain and r['language'] == lang for r in data['sources'])
            x = 177 + j * 76
            color = '#d7eee2' if n else '#e9edf2'
            pieces.append(f'<rect x="{x}" y="{y}" width="69" height="40" rx="4" fill="{color}"/><text x="{x+34}" y="{y+25}" text-anchor="middle" font-size="15">{n or "—"}</text>')
    (out / 'matrix.svg').write_text(''.join(pieces) + '</g></svg>')


def cascade_plan(out, rows):
    """Record explicit commands without creating corpora or invoking any model."""
    base = out.parent / f'cascade_{out.name}'
    py = '/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python'
    embed_py = '/mnt/afs/250010100/miniconda/envs/retriever/bin/python'
    script = ROOT / 'scripts/data_pipeline.py'
    plans = []
    for row in rows:
        if not row['added_in_this_expansion'] or row['inventory_role'] != 'three_tier_source':
            continue
        target = base / row['source_id']
        overflow = row.get('positive_overflow_queries', {})
        available = [k for k in TIERS if not overflow.get(str(k), 0)]
        commands = [
            [py, str(script), 'export', '--release', row['source_path'], '--output-suite', str(target / 'source_suite'), '--task', row['source_id']],
            [embed_py, str(script), 'rank', '--source-suite', str(target / 'source_suite'), '--task', row['source_id'],
             '--device', 'cuda:0', '--batch-size', '8', '--max-length', '8192', '--top-k', '1000',
             '--cache-dir', str(target / 'cache'), '--output', str(target / 'rankings.json')],
        ]
        if available:
            commands.append([py, str(script), 'pools', '--source-suite', str(target / 'source_suite'), '--task', row['source_id'],
                             '--rankings', str(target / 'rankings.json'), '--output', str(target / 'pools'),
                             '--capacities', *map(str, available)])
        plans.append({'source': row['source_id'], 'release': row['source_path'],
                      'release_completion_sha256': row['completion_sha256'], 'documents': row['documents'],
                      'queries': row['queries'], 'positive_overflow_queries': overflow,
                      'all_queries_supported_tiers': available, 'status': 'prepared_not_executed',
                      'commands': [shlex.join(command) for command in commands],
                      'boundary': 'Full retained query set; unsupported tiers need an explicit query policy. Batch size is a prior SciFact setting, not a throughput guarantee for this source.'})
    (out / 'cascade_plan.json').write_text(json.dumps({'status': 'prepared_not_executed', 'sources': plans}, ensure_ascii=False, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--releases', type=Path, nargs='*', default=[])
    p.add_argument('--research', type=Path, nargs='*', default=[])
    p.add_argument('--annotations', type=Path, help='Report-only scope notes; sealed source records stay intact.')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = read(args.base)
    rows = []
    for original in baseline['sources']:
        if original['origin'] == 'pipeline':
            row = from_release(Path(original['source_path']))
        else:
            row = classify(dict(original, verification='inherited_frozen_index_counts_not_rehashed'))
        row['added_in_this_expansion'] = False
        rows.append(row)
    ids = {r['source_id'] for r in rows}
    baseline_rows = list(rows)
    for release in args.releases:
        row = from_release(release)
        if row['source_id'] in ids:
            raise ValueError(f'Duplicate source ID: {row["source_id"]}')
        ids.add(row['source_id'])
        row['added_in_this_expansion'] = True
        rows.append(row)
    annotations = {row['source_id']: row['report_annotation'] for row in baseline['sources'] if row.get('report_annotation')}
    for source_id, annotation in (read(args.annotations) if args.annotations else {}).items():
        annotations[source_id] = {**annotations.get(source_id, {}), **annotation}
    unknown = set(annotations) - ids
    if unknown:
        raise ValueError(f'Annotations reference unknown sources: {sorted(unknown)}')
    for row in rows:
        annotation = annotations.get(row['source_id'])
        if annotation:
            row['report_annotation'] = annotation
            if annotation.get('notes'):
                row['notes'] = row.get('notes', '') + ' ' + annotation['notes']
            if annotation.get('task_origin'):
                row['task_origin'] = annotation['task_origin']
    research = [{'receipt_path': str(path), 'sha256': sha(path), 'payload': read(path)} for path in args.research]
    eligible = [r for r in rows if r['inventory_role'] == 'three_tier_source']
    payload = {'schema_version': 'dci-retrieval-matrix-v1', 'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               'sources': rows, 'research': research, 'domain_aliases': DOMAIN_ALIASES,
               'source_domain_overrides': SOURCE_DOMAIN_OVERRIDES,
               'coverage_change': coverage_change(baseline_rows, rows),
               'translation_coverage': translation_coverage(rows),
               'domains': sorted({r['domain'] for r in rows}),
               'languages': sorted({r['language'] for r in rows}),
               'counts': {'source_views': len(rows), 'new_source_views': len(args.releases),
                          'three_tier_source_views': len(eligible),
                          'three_tier_review_views': sum(r['inventory_role'] == 'review_required' for r in rows),
                          'three_tier_cells': len({(r['domain'], r['language']) for r in eligible}),
                          'source_roles': dict(Counter(r['inventory_role'] for r in rows))}}
    args.output.mkdir(parents=True)
    (args.output / 'data_matrix.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    fields = ['source', 'source_family', 'task_origin', 'origin', 'domain', 'source_domain', 'language', 'documents', 'queries', 'positive_pairs',
              'size_supported_tiers', 'preferred_corpus_gt1000', 'inventory_role', 'evidence',
              'release_state', 'translation', 'translation_kind', 'translation_basis', 'source_path', 'verification', 'candidate_pool_status', 'quality']
    with (args.output / 'source_matrix.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    render_html(args.output, payload)
    render_svg(args.output, payload)
    cascade_plan(args.output, rows)
    (args.output / 'renderer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    manifest = {'schema_version': payload['schema_version'], 'base': {'path': str(args.base), 'sha256': sha(args.base)},
                'renderer': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__))},
                'annotations': {'path': str(args.annotations), 'sha256': sha(args.annotations)} if args.annotations else None,
                'research_inputs': {r['receipt_path']: r['sha256'] for r in research},
                'release_completions': {r['source_path']: r['completion_sha256'] for r in rows if r.get('completion_sha256')},
                'outputs': {f.name: sha(f) for f in args.output.iterdir() if f.is_file()}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), **payload['counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
