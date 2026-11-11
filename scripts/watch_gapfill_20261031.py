#!/usr/bin/env python3
"""Finish this dated data integration after its already-running encoders close.

No evaluation model is launched. New data suites and evidence files are
exclusive writes; only the operational status and requested reports change.
"""
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_gapfill_20261031 import BASE, RELEASES, RUN, build_catalog
from dci_bench.data.benchmark_v2 import prepare_benchmark_v2, audit_benchmark_v2

EVIDENCE = ROOT / 'results/implementation/20261031-gapfill'
REPORT = ROOT / 'docs/project-progress-20261028'
REPORT_RUNTIME = RUN / 'report-runtime'


def save_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def status(phase, **details):
    value = {'phase': phase, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
             'benchmark_model_inference_completed': False, **details}
    (EVIDENCE / 'background-status.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run_step(command, output, *, env=None):
    with (EVIDENCE / output).open('x') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and 'href' in attrs:
            self.hrefs.append(attrs['href'])
        if 'id' in attrs:
            self.ids.add(attrs['id'])


def main():
    logs = [EVIDENCE / 'encoding-logs/browse-rank-b8.log',
            EVIDENCE / 'encoding-logs/cedar-rank-b2.log']
    receipt_paths = {
        source: RUN / source / ('catalog-row-v2.json' if source in {'gapfill_arguana_en', 'gapfill_cedar_zh_v2'} else 'catalog-row.json')
        for source in RELEASES
    }
    while not all(path.exists() for path in receipt_paths.values()):
        for log in logs:
            if log.exists() and 'Traceback' in log.read_text():
                status('failed', log=str(log), reason='Data encoder queue failed; report was not published.')
                raise RuntimeError(f'Inspect preserved data preparation log: {log}')
        status('encoding_and_pooling', completed_sources=[s for s, p in receipt_paths.items() if p.exists()],
               document_shards={source: len(list((RUN / source / 'embedding-cache').glob('**/document-*.npz')))
                                for source in RELEASES},
               expected_document_shards={'gapfill_arguana_en': 9, 'gapfill_android_en': 23,
                                         'gapfill_browsecomp_plus_en': 98, 'gapfill_acord_en_v3': 4,
                                         'gapfill_cedar_zh_v2': 26})
        time.sleep(30)

    status('catalog_and_suite')
    build_catalog()
    suite = BASE / 'benchmark_v2_20261031_v1'
    manifest = prepare_benchmark_v2(output_dir=suite, catalog_path=RUN / 'catalog.json',
                                    config_path=ROOT / 'configs/benchmark_v2.json')
    save_new(EVIDENCE / 'prepared-manifest.json', manifest)
    accepted = BASE / 'benchmark_v2_20261102_v2/manifest.json'
    if accepted.exists():
        previous = json.loads(accepted.read_text())
        stable = []
        for task, row in previous['tasks'].items():
            if row['status'] != 'ready':
                continue
            current = manifest['tasks'][task]
            assert current['status'] == 'ready' and current['source_id'] == row['source_id'], task
            assert current['query_ids'] == row['query_ids'], task
            stable.append(task)
        assert len(stable) == 17
        save_new(EVIDENCE / 'accepted-17-stability.json',
                 {'previous_suite': str(accepted.parent), 'stable_tasks': stable,
                  'query_selections_unchanged': True})
    audit_start = time.monotonic()
    audit = audit_benchmark_v2(suite)
    audit.update(suite=str(suite), body_hashes_checked=False,
                 elapsed_seconds=time.monotonic() - audit_start)
    save_new(EVIDENCE / 'data-audit.json', audit)
    status('data_validation')
    run_step([sys.executable, str(EVIDENCE / 'verify_suite.py')], 'suite-validation.json')
    run_step([sys.executable, str(EVIDENCE / 'summarize_pools.py')], 'retrieval-quality.json')
    run_step([sys.executable, 'scripts/run_benchmark_v2.py', '--suite', str(suite), '--split', 'smoke',
              '--allow-partial', '--preflight-only', '--output', str(EVIDENCE / 'data-preflight')],
             'preflight-driver.log')
    status('report_publication')
    run_step([sys.executable, 'scripts/publish_gapfill_report_20261031.py'], 'report-publication.log')
    browser_env = {**os.environ, 'PYTHONPATH': str(REPORT_RUNTIME / 'python'),
                   'PLAYWRIGHT_BROWSERS_PATH': str(REPORT_RUNTIME / 'browsers'),
                   'LD_LIBRARY_PATH': ':'.join(str(REPORT_RUNTIME / path) for path in
                                              ('sysroot/usr/lib/x86_64-linux-gnu', 'sysroot/lib/x86_64-linux-gnu')),
                   'FONTCONFIG_FILE': str(REPORT_RUNTIME / 'sysroot/fonts.conf')}
    run_step([sys.executable, str(EVIDENCE / 'report-ui/check_browser.py')], 'browser-check.log', env=browser_env)
    run_step([sys.executable, 'scripts/publish_gapfill_report_20261031.py'], 'report-browser-binding.log')

    quality = json.loads((EVIDENCE / 'retrieval-quality.json').read_text())
    doc = ROOT / 'docs/gapfill_20261031.md'
    text = doc.read_text()
    text = text.replace('本文记录五个已完成数据源适配的输入边界，以及它们进入最终 DCI suite 前仍需通过的 pool 审计。源 release、ranked candidate pool 和最终 suite 是三个阶段；源 release 的数量或 `labels_ready` 状态不等于 pool 或 suite 已完成。',
                        '五个新增来源已完成源审查、实际 KaLM 排名、候选池和最终 suite 审计，数据覆盖为 18/20；中文软件仍阻塞，中文企业暂缓。以下计数与限制来自冻结数据产物，不代表模型评测成绩。')
    text = text.replace('已审核源；待 ranked pool audit', '源与最终嵌套池已通过审计').replace('两档源池已验证；待最终 suite audit', '源与最终嵌套池已通过审计')
    text = text.replace('最终目标是数据池审计通过后的 18/20 tasks；该数字是目标边界，不是当前完成数。主线程仍需补齐：五源实际 ranking/pool 的 `_COMPLETE` 与 hash、Lite⊂Full 证据、query/group split audit、host-only labels 检查、最终 suite catalog 与动态路径快照。网页报告由主线程更新。',
                        '最终 suite 已通过 18/20 数据任务审计，开发 180 / 正式 900 query；旧 13 任务的查询清单保持不变。五源源池已验证正文 hash，最终 suite 的跨任务审计为元数据审计。36 个任务档位预检通过，评测模型调用数为 0。证据见 `results/implementation/20261031-gapfill/`；HTML 与 progress.md 已同步。')
    text = text.replace('`argumentation_zh` 已由 CEDAR 提供候选源，但仍须完成上述 ranked pool audit 后才可进入最终 suite。', '`argumentation_zh` 已由 CEDAR 完成 ranked pool 与最终 suite 接入。')
    text = text.replace('本文件描述的是数据接入和待验收边界，不包含模型调用、评测分数或网页报告结论。', '本文件描述的是已验证的数据接入边界，不包含正式评测模型的分数。')
    text += '\n## 已闭合的排名质量证据\n\n以下是全部保留源 query 的注入前宏平均 Recall；正例注入后的覆盖率均为 100%，两者不混用。\n\n| 来源 | Query | Recall@200 | Recall@1000 |\n|---|---:|---:|---:|\n'
    for row in quality['sources']:
        metrics = row['retrieval_quality']
        text += f"| `{row['source']}` | {row['queries']} | {metrics['200']['pre_injection_macro_recall']:.4f} | {metrics['1000']['pre_injection_macro_recall']:.4f} |\n"
    doc.write_text(text)
    cedar = ROOT / 'docs/cedar_20261031.md'
    cedar.write_text(cedar.read_text().replace('该 source release 仍需由主流程生成并审计实际 KaLM ranked pools，随后才能进入',
                                             '该 source release 已通过实际 KaLM ranked pools 与最终 suite 审计，产物位于'))

    page = (REPORT / 'index.html').read_text()
    parsed = Links()
    parsed.feed(page)
    checked = 0
    for href in parsed.hrefs:
        target = urlsplit(href)
        if target.scheme or target.netloc:
            continue
        if target.path:
            assert (REPORT / unquote(target.path)).exists(), href
            checked += 1
        elif target.fragment:
            assert unquote(target.fragment) in parsed.ids, href
    scripts = re.findall(r'<script>(.*?)</script>', page, flags=re.S)
    js = EVIDENCE / 'report-syntax-check.js'
    js.write_text('\n'.join(scripts))
    run_step([str(REPORT_RUNTIME / 'python/playwright/driver/node'), '--check', str(js)],
             'report-javascript-check.log')
    report_manifest = json.loads((REPORT / 'manifest.json').read_text())
    for name, expected in report_manifest['outputs'].items():
        assert hashlib.sha256((REPORT / name).read_bytes()).hexdigest() == expected, name
    for name, expected in report_manifest['inputs'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert report_manifest['checks']['coverage_ready_tasks'] == 18
    assert report_manifest['checks']['browser_validation_current'] is True
    assert report_manifest['checks']['data_embedding_gpu_inference_completed'] is True
    save_new(EVIDENCE / 'report-static-validation.json', {'passed': True, 'local_links_checked': checked,
             'manifest_hashes_valid': True, 'javascript_syntax_valid': True, 'browser_validation_current': True})
    completed = {'suite': str(suite), 'ready_tasks': 18, 'blocked_tasks': ['software_zh', 'enterprise_zh'],
                 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                 'benchmark_model_inference_completed': False,
                 'html_sha256': hashlib.sha256((REPORT / 'index.html').read_bytes()).hexdigest(),
                 'progress_sha256': hashlib.sha256((REPORT / 'progress.md').read_bytes()).hexdigest()}
    save_new(EVIDENCE / 'integration-complete.json', completed)
    status('complete', **completed)


if __name__ == '__main__':
    main()
