#!/usr/bin/env python3
"""Publish the two Chinese tasks only after source, pool and suite acceptance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dci_bench.data.benchmark_v2 import audit_benchmark_v2, sha256_file, stable_json_sha256
from scripts.update_benchmark_progress import build_coverage, update
from scripts.prepare_gapfill_20261031 import RUN


def bcp_completion_metadata() -> Path | None:
    """Check source closure for reporting without walking the 100K body files."""
    source = 'gapfill_browsecomp_plus_en'
    root = RUN / source
    receipt = root / 'catalog-row.json'
    if not receipt.exists():
        return None
    row = json.loads(receipt.read_text())
    if row['source'] != source or row['task'] != source:
        raise ValueError('BrowseComp+ receipt source identity mismatch')
    ranking = root / 'ranking.json'
    if Path(row['ranking']) != ranking or sha256_file(ranking) != row['ranking_sha256']:
        raise ValueError('BrowseComp+ ranking identity mismatch')
    if sorted(pool['capacity'] for pool in row['pools']) != [200, 1000]:
        raise ValueError('BrowseComp+ pool capacities are incomplete')
    for pool in row['pools']:
        suite = root / f"pools-{pool['capacity']}"
        metadata = suite / 'metadata' / source
        marker = metadata / '_COMPLETE.json'
        if Path(pool['path']) != suite or sha256_file(marker) != pool['completion_sha256']:
            raise ValueError('BrowseComp+ pool completion identity mismatch')
        complete = json.loads(marker.read_text())
        if complete['content_hashes_verified'] is not True:
            raise ValueError('BrowseComp+ source pool lacks content-verified completion')
        for name, digest in complete['metadata_hashes'].items():
            if Path(name).name != name or sha256_file(metadata / name) != digest:
                raise ValueError(f'BrowseComp+ pool metadata changed: {name}')
    return receipt


def publish(data_root: Path) -> None:
    evidence = data_root / 'evidence/suite-acceptance'
    suite = data_root / 'suites/benchmark_v2_20261102_zh_v1'
    report = ROOT / 'docs/project-progress-20261028'
    accepted = json.loads((evidence / 'status-ready.json').read_text())
    if accepted['status'] != 'ready' or accepted.get('check_only'):
        raise ValueError('A completed suite acceptance is required')
    prepared = json.loads((evidence / 'prepared-manifest.json').read_text())
    config = json.loads((evidence / 'config.json').read_text())
    audit = audit_benchmark_v2(suite)
    audit['body_hashes_checked'] = False
    if len(audit['ready_tasks']) != 19 or set(audit['blocked_tasks']) != {'web_en'}:
        raise ValueError('Expected the preserved 17 tasks plus two Chinese tasks')
    if audit['overall_sha256'] != prepared['overall_sha256']:
        raise ValueError('Accepted and live suite identities differ')
    preflight = json.loads((data_root / 'evidence/data-preflight/preflight.json').read_text())
    if (preflight['jobs'] != 38 or preflight['model_calls'] != 0
            or preflight['status'] != 'data_preflight_passed'):
        raise ValueError('Expected 38 data-only preflight jobs and zero model calls')
    prior = json.loads((report / 'benchmark-v2-status.json').read_text())
    readiness = json.loads((data_root / 'source-readiness.json').read_text())
    sources = {row['task']: row for row in readiness['sources']}
    notes = {row['task']: row for row in prior['data_tasks']}
    for task, row in sources.items():
        notes[task] = {'task': task, 'status': '已准备', 'source': row['source'],
                      'detail': f"原生中文；{row['documents']}文档/{row['queries']}查询/{row['positive_pairs']}正例关系；"
                                f"每题{row['positive_min']}–{row['positive_max']}正例。"
                                'dev10/test50/smoke1，seed42；原始人工标签全保留，Lite200⊂Full1000；新两任务所选Full正文hash通过。'}
    bcp = bcp_completion_metadata()
    web_detail = ('BrowseComp+ 排名与200/1000源池已完成；完成回执、排名及元数据哈希已核验，尚未合入当前19任务suite。'
                  if bcp else 'BrowseComp+仍待构池；尚未合入当前19任务suite。')
    notes['web_en'] = {**notes['web_en'], 'detail': web_detail,
                       'source': 'BrowseComp+（源池完成，待suite合入）' if bcp else 'BrowseComp+（源池待完成）',
                       'pending_reason': 'source pools complete; awaiting suite integration' if bcp
                       else 'source pools pending'}
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    status = {**prior, 'as_of': stamp,
              'summary': '数据就绪19/20：保留旧17任务，新增T2Ranking中文软件和CFQA中文企业业务。开发190、正式950、Smoke19；' + web_detail + 'KaLM仅用于数据准备，正式评测模型未执行。',
              'data_tasks': list(notes.values()),
              'stages': [
                  {'name': '中文来源与正文验收', 'status': '通过',
                   'detail': '固定原始版本、split、许可及SHA-256；T2Ranking人工grade≥2，CFQA原始人工证据物理页；领域过滤丢失正例时排除整题。'},
                  {'name': '四卡KaLM排名与构池', 'status': '完成',
                   'detail': '两源编码、排名与200/1000源池完成，完成回执和内容hash已核验。未标注候选不作为已确认负例。'},
                  {'name': '19任务suite', 'status': '通过',
                   'detail': '旧17任务source/query选择保持不变；新两任务seed42分组隔离，dev10/test50/smoke1；最终Lite⊂Full且包含全部已知正例。'},
                  {'name': '正文与元数据', 'status': '通过',
                   'detail': '全suite元数据审计及38个数据预检；新两任务所选Full正文实际SHA-256另行核验，覆盖Lite子集。旧17任务正文沿用已有证据。'},
                  {'name': '剩余网页与模型评测', 'status': '未完成',
                   'detail': web_detail + '全20任务正式门禁仍关闭。没有进行agent GPU/API标定或正式模型评测。'}],
              'limitations': [
                  web_detail + '全20任务正式启动门禁仍阻塞。',
                  'enterprise_zh是公司年报中的企业业务、研发、治理与经营风险知识，不代表内部制度或组织办事流程库。',
                  'CFQA标注仓库MIT；官方公司年报保留原来源与权利，不将标注许可扩展到PDF正文。',
                  'T2Ranking保留原train/dev信息后重新抽样；此处DCI结果不能称为上游官方测试成绩。',
                  'KaLM编码是数据准备；正例注入后的覆盖率不表示检索或agent模型得分。'],
              'data_preparation': {'embedding_gpu_inference_completed': True,
                                   'accepted_sources_embedding_completed': True,
                                   'accepted_source_count': 6, 'pending_source_count': 1,
                                   'benchmark_model_inference_completed': False}}
    status['evidence'] = [
        {'label': '中文数据接入与四卡启动说明', 'path': '../zh_gapfill_20261102.md'},
        *[{'label': label, 'path': os.path.relpath(path, report)} for label, path in (
            ('中文来源验收', data_root / 'source-readiness.json'),
            ('19任务准备清单', evidence / 'prepared-manifest.json'),
            ('中文源池闭合', evidence / 'source-closure.json'),
            ('19任务验收与正文哈希', evidence / 'data-audit.json'))],
        *prior['evidence']]
    if bcp:
        status['evidence'].insert(1, {'label': 'BrowseComp+源池完成回执',
                                      'path': os.path.relpath(bcp, report)})
    status['evidence'] = list({row['path']: row for row in status['evidence']}.values())
    # The renderer checks config/manifest/audit identities before publishing.
    build_coverage(config, prepared, audit, status)
    current = json.loads((ROOT / 'configs/benchmark_v2.json').read_text())
    for task in sources:
        current['sources'][task] = config['sources'][task]
    if stable_json_sha256(current) != stable_json_sha256(config):
        raise ValueError('Other benchmark configuration changed; preserve it and reconcile first')
    # A failed browser check must be resumable without replacing suite evidence.
    audit_path = evidence / 'report-audit.json'
    if audit_path.exists() and json.loads(audit_path.read_text()) != audit:
        raise ValueError('Existing report audit differs from the live suite; preserve and investigate')
    attempt_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    ui = ROOT / 'results/implementation' / ('zh-gapfill-report-' + attempt_id) / 'report-ui'
    ui.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / 'results/implementation/20261102-gapfill-suite-v2/report-ui/check_browser.py',
                 ui / 'check_browser.py')
    runtime = Path('/mnt/afs2/202609/embedding_models/dci-bench/eval_data/benchmark_gapfill_20261031_v1/report-runtime')
    browser_env = {**os.environ, 'PYTHONPATH': str(runtime / 'python'),
                   'PLAYWRIGHT_BROWSERS_PATH': str(runtime / 'browsers'),
                   'LD_LIBRARY_PATH': ':'.join(str(runtime / path) for path in (
                       'sysroot/usr/lib/x86_64-linux-gnu', 'sysroot/lib/x86_64-linux-gnu')),
                   'FONTCONFIG_FILE': str(runtime / 'sysroot/fonts.conf')}
    print(f'Chinese suite already accepted; CPU report logs: {ui}', flush=True)
    # Check the node's browser libraries before changing the published report.
    with (ui / 'browser-preflight.log').open('x') as log:
        subprocess.run([sys.executable, '-c',
                        'from playwright.sync_api import sync_playwright\n'
                        'with sync_playwright() as pw:\n'
                        '    browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])\n'
                        '    print(browser.version)\n'
                        '    browser.close()\n'], env=browser_env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    snapshot = report / 'history' / ('before-zh-' + attempt_id)
    snapshot.mkdir(parents=True)
    for path in report.iterdir():
        if path.is_file():
            shutil.copy2(path, snapshot / path.name)
    (ROOT / 'configs/benchmark_v2.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    if not audit_path.exists():
        with audit_path.open('x') as stream:
            json.dump(audit, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    update(report, status, config_path=evidence / 'config.json',
           prepared_path=evidence / 'prepared-manifest.json', audit_path=audit_path)
    with (ui / 'browser-check.log').open('x') as log:
        subprocess.run([sys.executable, str(ui / 'check_browser.py')], env=browser_env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    update(report, status, config_path=evidence / 'config.json',
           prepared_path=evidence / 'prepared-manifest.json', audit_path=audit_path)
    print(json.dumps({'status': 'published', 'ready_tasks': 19, 'report': str(report / 'index.html'),
                      'browser_logs': str(ui), 'bcp_pools_complete': bcp is not None}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    publish(args.data_root)
