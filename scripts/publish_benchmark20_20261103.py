#!/usr/bin/env python3
"""Publish accepted 20-task data; browser validation is a separate command."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dci_bench.data.benchmark_v2 import sha256_file, stable_json_sha256
from scripts.accept_benchmark20_20261103 import SOURCE, read, save
from scripts.update_benchmark_progress import build_coverage, update

DEFAULT_EVIDENCE = ROOT / 'results/implementation/20261103-benchmark20-v2'


def accepted_data(evidence: Path) -> tuple[Path, dict, dict, dict]:
    accepted = read(evidence / 'status-ready.json')
    suite = Path(accepted['suite'])
    prepared = read(evidence / 'prepared-manifest.json')
    audit = read(evidence / 'data-audit.json')
    preflight = read(evidence / 'data-preflight/preflight.json')
    launch = read(evidence / 'data-preflight/launch.json')
    if (accepted['status'] != 'data_accepted' or accepted['ready_tasks'] != 20
            or accepted['blocked_tasks'] or len(audit['ready_tasks']) != 20 or audit['blocked_tasks']
            or not audit['ready_for_formal']):
        raise ValueError('All 20 data tasks must be accepted')
    if (sha256_file(suite / 'manifest.json') != accepted['manifest_sha256']
            or prepared != read(suite / 'manifest.json')
            or prepared['overall_sha256'] != audit['overall_sha256']):
        raise ValueError('Live and accepted suite identities differ')
    if (preflight['status'] != 'data_preflight_passed' or preflight['jobs'] != 40
            or preflight['model_calls'] != 0 or not launch['preflight_only']
            or launch['selection_sha256'] != accepted['manifest_sha256']
            or launch['execution_count'] != 40 or not launch['complete_scope'] or launch['blocked_tasks']):
        raise ValueError('Expected 40 data-only preflight jobs bound to the accepted suite')
    stability = read(evidence / 'old-19-stability.json')
    pools = read(evidence / 'suite-validation.json')
    bodies = read(evidence / 'selected-body-validation.json')
    if (stability['stable_old_task_count'] != 19 or stability['changed']
            or not pools['passed'] or pools['pool_records_checked'] != 2400
            or pools['global_group_leakage'] or bodies['query_count'] != 60
            or bodies['verified_unique_body_count'] <= 0):
        raise ValueError('Selection, nested pool or selected body acceptance is incomplete')
    return suite, prepared, audit, read(evidence / 'config.json')


def publish(evidence: Path, browser_receipt: Path | None = None) -> dict:
    suite, prepared, audit, config = accepted_data(evidence)
    report = ROOT / 'docs/project-progress-20261028'
    if stable_json_sha256(config) != stable_json_sha256(read(ROOT / 'configs/benchmark_v2.json')):
        raise ValueError('Current benchmark configuration differs from the accepted suite')
    if browser_receipt is not None:
        browser = read(browser_receipt)
        if browser['passed'] is not True or browser['html_sha256'] != sha256_file(report / 'index.html'):
            raise ValueError('Browser validation does not match the published HTML')
        status = read(report / 'benchmark-v2-status.json')
        if status['data_suite'] != str(suite):
            raise ValueError('Current report points to a different data suite')
        shutil.copy2(browser_receipt, report / 'browser-validation-v2.json')
        update(report, status, config_path=evidence / 'config.json',
               prepared_path=evidence / 'prepared-manifest.json', audit_path=evidence / 'data-audit.json')
        result = {'status': 'report_browser_validated', 'browser_checks': len(browser['checks']),
                  'html_sha256': sha256_file(report / 'index.html'), 'browser_receipt': str(browser_receipt),
                  'data_acceptance_independent_of_browser': True}
        save(evidence / 'report-browser-complete.json', result)
        return result

    receipt_names = ['config.json', 'catalog.json', 'source-closure.json', 'prepared-manifest.json',
                     'data-audit.json', 'old-19-stability.json', 'suite-validation.json',
                     'selected-body-validation.json', 'status-ready.json',
                     'data-preflight/preflight.json', 'data-preflight/launch.json']
    integration = {'status': 'data_complete', 'suite': str(suite), 'ready_tasks': 20, 'blocked_tasks': [],
                   'prepared_queries': {'dev': 200, 'test': 1000, 'smoke': 20},
                   'preflight_jobs': 40, 'model_calls': 0, 'data_ready_for_formal': True,
                   'formal_model_execution_ready': False, 'benchmark_model_inference_completed': False,
                   'artifact_sha256': {name: sha256_file(evidence / name) for name in receipt_names},
                   'browser_validation': 'separate report-only stage'}
    complete = evidence / 'integration-complete.json'
    if complete.exists():
        if read(complete) != integration:
            raise ValueError('Existing integration receipt differs; preserve and investigate')
    else:
        save(complete, integration)

    prior = read(report / 'benchmark-v2-status.json')
    notes = {row['task']: row for row in prior['data_tasks']}
    body_count = read(evidence / 'selected-body-validation.json')['verified_unique_body_count']
    notes['web_en'] = {'task': 'web_en', 'status': '已准备', 'source': SOURCE,
                       'detail': '原生英文网页；99,930文档/825源查询/5,023正例关系。'
                                 f'冻结dev10/test50/smoke1；Lite200⊂Full1000，{body_count:,}篇所选Full正文SHA-256通过。'}
    now = datetime.now(timezone(timedelta(hours=8)))
    snapshot = report / 'history' / ('before-20-' + now.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    status = {**prior, 'as_of': now.isoformat(timespec='seconds'), 'data_suite': str(suite),
              'summary': '数据就绪20/20：BrowseComp+已合入，原19任务查询选择和候选池保持不变。'
                         '开发200、正式1,000、Smoke20题；Lite200⊂Full1000。40个零模型调用数据预检通过；真实模型Smoke、开发标定与正式评测尚未执行。',
              'data_tasks': list(notes.values()),
              'data_ready_for_formal': True, 'formal_model_execution_ready': False,
              'body_validation_note': f'本轮实际核验BCP所选Full的{body_count:,}篇不同正文，覆盖Lite子集；'
                                      '原19任务沿用既有正文证据，并确认候选池、标签和正文清单一致。未重扫全部历史语料。',
              'validation_note': '本轮新增20任务suite的CPU数据验收与40个零模型调用预检；'
                                 '沙箱和真实模型需在评测节点另行Smoke。预算／并发标定与正式评测尚未运行；历史159次smoke属于旧协议。',
              'stages': [
                  {'name': '20任务数据接入', 'status': '通过',
                   'detail': 'BCP源池已合入；20个中英任务均已准备，旧19任务source/query/group选择保持不变。'},
                  {'name': '冻结抽样与嵌套池', 'status': '通过',
                   'detail': 'seed42；每任务dev10/test50/smoke1，跨任务dev/test分组隔离；2,400条池记录满足容量、全正例保留与Lite⊂Full。'},
                  {'name': '正文与标签隔离', 'status': '通过',
                   'detail': f'新BCP所选Full的{body_count:,}篇正文实际SHA-256通过；原19任务候选和标签元数据不变，沿用已有正文证据；qrels留在宿主元数据。'},
                  {'name': '数据预检', 'status': '通过',
                   'detail': '20任务×2档，共40个数据预检通过；model_calls=0，未进行当前评测节点沙箱或模型推理验收。'},
                  {'name': '真实模型Smoke与开发标定', 'status': '未执行',
                   'detail': '后续先验证40次真实Smoke，再比较4/8/16并发与32k/64k/96k预算、有效率、吞吐及成本。'},
                  {'name': '正式模型评测', 'status': '未执行',
                   'detail': '每模型1,000 query×两档，共2,000次；须先满足开发标定和费用/时间门禁。'}],
              'limitations': [
                  '20/20只表示数据就绪；正式模型启动仍需真实Smoke、完整开发标定、匹配运行指纹与成本门禁。',
                  '本轮40个预检没有调用模型，也不证明当前评测节点的沙箱已通过。',
                  '旧19任务正文沿用已有验收证据，本轮未重新扫描全部历史语料。',
                  *[item for item in prior['limitations'] if item.startswith(('enterprise_zh', 'CFQA', 'T2Ranking', 'KaLM'))]],
              'data_preparation': {'embedding_gpu_inference_completed': True,
                                   'accepted_sources_embedding_completed': True,
                                   'accepted_source_count': 7, 'pending_source_count': 0,
                                   'benchmark_model_inference_completed': False},
              'report_validation': {'data_acceptance_independent_of_browser': True,
                                    'browser_receipt': 'browser-validation-v2.json'}}
    evidence_rows = [('20任务数据闭合', complete), ('20任务冻结清单', evidence / 'prepared-manifest.json'),
                     ('原19任务选择保持不变', evidence / 'old-19-stability.json'),
                     ('2,400条嵌套池与分组验收', evidence / 'suite-validation.json'),
                     ('BCP所选Full正文哈希', evidence / 'selected-body-validation.json'),
                     ('40个零模型调用数据预检', evidence / 'data-preflight/preflight.json'),
                     ('BCP源闭合与注入前召回', evidence / 'source-closure.json'),
                     ('更新前19/20报告快照', snapshot / 'index.html')]
    status['evidence'] = [
        {'label': '20任务接入与验收说明', 'path': '../benchmark20_20261103.md'},
        *[{'label': label, 'path': os.path.relpath(path, report)} for label, path in evidence_rows],
        *[{**row, 'label': row['label'] if row['label'].startswith('历史') else '历史 / ' + row['label']}
          for row in prior['evidence']]]
    status['evidence'] = list({row['path']: row for row in status['evidence']}.values())
    build_coverage(config, prepared, audit, status)
    snapshot.mkdir(parents=True)
    for path in report.iterdir():
        if path.is_file():
            shutil.copy2(path, snapshot / path.name)
    update(report, status, config_path=evidence / 'config.json',
           prepared_path=evidence / 'prepared-manifest.json', audit_path=evidence / 'data-audit.json')
    result = {'status': 'report_published', 'ready_tasks': 20, 'html_sha256': sha256_file(report / 'index.html'),
              'snapshot': str(snapshot), 'browser_validation': 'pending independent report check'}
    save(evidence / 'report-publication.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument('--bind-browser', type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.evidence_dir, args.bind_browser), ensure_ascii=False), flush=True)
