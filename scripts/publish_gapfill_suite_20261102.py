#!/usr/bin/env python3
"""Publish the accepted four-source, 17-task data suite; no inference."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.update_benchmark_progress import update

EVIDENCE = ROOT / 'results/implementation/20261102-gapfill-suite-v2'


def read(path):
    return json.loads(path.read_text())


def main():
    prepared = read(EVIDENCE / 'prepared-manifest.json')
    audit = read(EVIDENCE / 'data-audit.json')
    validation = read(EVIDENCE / 'suite-validation.json')
    bodies = read(EVIDENCE / 'selected-body-validation.json')
    preflight = read(EVIDENCE / 'data-preflight/preflight.json')
    quality = read(EVIDENCE / 'retrieval-quality.json')
    assert len(audit['ready_tasks']) == 17
    assert set(audit['blocked_tasks']) == {'web_en', 'software_zh', 'enterprise_zh'}
    assert validation['new_status'] == {'ready': 17, 'blocked': 3}
    assert validation['stable_old_ready_tasks'] == 13
    assert validation['pool_records_checked'] == 2040
    assert validation['self_match_records_checked'] == 240
    assert validation['acord_train_excluded'] == 51
    assert bodies['passed'] and len(bodies['tasks']) == 4
    assert preflight['status'] == 'data_preflight_passed'
    assert preflight['jobs'] == 34 and preflight['model_calls'] == 0
    old = read(ROOT / 'results/implementation/20261029-lite-full/status.json')
    notes = {row['task']: row for row in old['data_tasks']}
    metrics = {row['source']: row for row in quality['sources']}
    details = {
        'argumentation_en': 'ArguAna：排除218个歧义query，topic整组隔离；排除query自身。',
        'software_en': 'Android：排除78472歧义题与19971的262正例溢出；其余官方qrels保留。',
        'legal_en': 'ACORD：只取官方valid6+test57，排除train51；3/4/5星→relevance1/2/3；DCI采样不等于官方评分。',
        'argumentation_zh': 'CEDAR：显式support/against关系、精确文本alias合并、每topic一题、排除自身；未标注不作负例。',
    }
    blocked = {
        'web_en': ('BrowseComp+（待四卡5090续跑）', '99930文档/825 query/5023正例关系；已保留44/98个文档分片，尚无完成源池。启动环境lychee_reranker_5090_kalm451，四卡分片并行；本轮未启动GPU任务。'),
        'software_zh': ('DQA-Bench（原生来源证据不足）', '7490 query/7481 answer/7490 QA样例；缺逐条原生中文来源证明，保持阻塞。'),
        'enterprise_zh': ('暂缓', '按本轮范围暂缓中文企业来源接入。'),
    }
    tasks = []
    for task, row in prepared['tasks'].items():
        if row['status'] == 'blocked':
            source, detail = blocked[task]
            tasks.append({'task': task, 'status': '阻塞', 'source': source, 'detail': detail})
            continue
        detail = notes[task]['detail']
        if task in details:
            metric = metrics[row['source_id']]
            recalls = metric['retrieval_quality']
            detail = (details[task] + f" 源规模{metric['documents']}/{metric['queries']}/{metric['positive_pairs']}；"
                      f"注入前Recall@200={recalls['200']['pre_injection_macro_recall']:.4f}、"
                      f"@1000={recalls['1000']['pre_injection_macro_recall']:.4f}；suite与选中正文hash验收通过。")
        tasks.append({'task': task, 'status': '已准备', 'source': row['source_id'], 'detail': detail})
    evidence = [
        ('17任务准备清单', 'prepared-manifest.json'), ('17任务元数据审计', 'data-audit.json'),
        ('抽样、嵌套、self-match与ACORD划分核验', 'suite-validation.json'),
        ('新增四任务选中Full正文hash与标签隔离', 'selected-body-validation.json'),
        ('四源排名质量（注入前召回）', 'retrieval-quality.json'),
        ('34个数据预检，0模型调用', 'data-preflight/preflight.json'),
        ('冻结源与池元数据闭合', 'source-closure.json'),
    ]
    status = {
        'as_of': '2026-11-02',
        'summary': '数据就绪17/20＝原13项＋原待suite验收的3项（ArguAna、Android、ACORD）＋已完成构池并通过本轮验收的CEDAR 1项。此前两项待构池现只剩BrowseComp+；中文软件仍阻塞，中文企业暂缓。开发170、正式850、Smoke17 query；正式模型评测未执行。',
        'data_tasks': tasks,
        'stages': [
            {'name': '就绪数变化：13＋3＋1＝17', 'status': '已核实', 'detail': 'CEDAR原列为待构池；本轮检查发现其200/1000两档源池均已完成，随后通过suite、分组隔离、self-match及选中文档hash验收。因此本轮新增4项，剩余待构池仅BrowseComp+。'},
            {'name': '四个新增来源suite验收', 'status': '通过', 'detail': '源版本、排名及池元数据闭合；四项均进入冻结的17任务suite。没有等待BrowseComp+完成，也未重建已有源池。'},
            {'name': '抽样与嵌套候选池', 'status': '通过', 'detail': 'seed42；每任务dev10/test50/smoke1。旧13任务清单不变；2040条任务/档位/query池记录通过200/1000容量、正例保留及Lite⊂Full核验，dev/test无分组泄漏。'},
            {'name': '正文与标签边界', 'status': '通过（新增四项）', 'detail': '新增四任务选中Full正文重新核验hash，覆盖其Lite子集；标签只在宿主元数据，agent工作区为空待运行时构造。旧13任务正文证据沿用历史记录。'},
            {'name': '数据预检', 'status': '通过', 'detail': '17任务×2档，共34个preflight jobs，模型调用0。ACORD官方train51全部排除；两个论证任务240条self-match记录通过。'},
            {'name': '源池布局兼容修复', 'status': '通过', 'detail': 'suite支持内联正文与既有外置共享正文两种源池布局。14项数据测试、10项runtime测试通过；1项真实Pi mock因当前环境缺Node/Bubblewrap跳过。失败v1产物保留，成功suite为20261102_v2。'},
            {'name': '剩余网页构池', 'status': '待用户在集群启动', 'detail': 'BrowseComp+保留44/98个文档分片；四卡5090启动器使用lychee_reranker_5090_kalm451，先验证旧缓存兼容性再并行处理缺失分片，已完成CEDAR自动跳过。'},
            {'name': 'CPU耗时与评测并发', 'status': '维持现状', 'detail': '本轮未优化CPU链路、未改变评测并发；下方CPU计时是2026-10-29历史记录。'},
            {'name': '真实评测模型标定与正式评测', 'status': '未执行', 'detail': 'KaLM编码属于数据准备；本轮suite验收只使用CPU，没有运行真实评测模型、吞吐或费用标定。'},
        ],
        'limitations': [
            '剩余3格：英文网页BrowseComp+待完成构池，中文软件缺原生来源证据，中文企业暂缓。全20任务正式启动门禁仍阻塞。',
            '四卡5090启动器完成本地CPU检查不等于集群GPU smoke已通过；用户启动后先运行实际编码及缓存兼容性检查。',
            'suite全局审计为metadata-only；新四项的选中Full正文hash另行实测，旧13项未重新扫描正文。',
            'ACORD使用官方valid/test重新采样；CEDAR仅显式支持/反对关系形成qrels，未标注关系不作为负例。',
            '排名召回在正例注入前统计；注入后100%正例覆盖不代表检索模型或agent评测成绩。',
        ],
        'evidence': [{'label': label, 'path': '../../' + str((EVIDENCE / name).relative_to(ROOT))} for label, name in evidence] + [
            {'label': '内联布局数据回归14项', 'path': '../../results/implementation/20261102-gapfill-suite/inline-layout-data-tests.log'},
            {'label': 'Runtime回归10通过/1跳过', 'path': '../../results/implementation/20261102-gapfill-suite/inline-layout-runtime-tests-2.log'},
            {'label': '四卡启动器与发布门禁11项CPU检查', 'path': '../../results/implementation/20261102-gapfill-suite-v2/launcher-and-publication-tests.log'},
            {'label': '分片截断审计保留回归7项', 'path': '../../results/implementation/20261102-gapfill-suite-v2/launcher-truncation-tests.log'},
            {'label': '四卡实际数据只读启动计划', 'path': '../../results/implementation/20261102-gapfill-suite-v2/launcher-check-only.json'},
            {'label': '数据接入记录与5090启动说明', 'path': '../gapfill_20261102.md'},
            {'label': '2026-10-31中间报告快照', 'path': 'history/20261031/index.html'},
        ],
        'timing_rows': old['timing_rows'],
        'data_preparation': {'embedding_gpu_inference_completed': False,
                             'accepted_sources_embedding_completed': True,
                             'accepted_source_count': 4, 'pending_source_count': 1,
                             'benchmark_model_inference_completed': False,
                             'note': 'Four previously encoded source pools are accepted; BrowseComp+ remains incomplete. No GPU work was launched in this acceptance turn.'},
    }
    update(ROOT / 'docs/project-progress-20261028', status,
           config_path=ROOT / 'configs/benchmark_v2.json',
           prepared_path=EVIDENCE / 'prepared-manifest.json', audit_path=EVIDENCE / 'data-audit.json')


if __name__ == '__main__':
    main()
