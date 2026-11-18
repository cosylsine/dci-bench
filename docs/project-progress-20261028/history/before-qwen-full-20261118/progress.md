# DCI-Bench 新版数据进展（2026-11-16）

20/20任务已完成数据验收；法律换源后，全部任务先过滤1–10正例，再按seed42统一重抽。旧数据与旧模型结果保留。

- 法律英文：PrivacyQA，4,657文档 / 167合格问题 / 990正例关系；法律中文：STARD，55,348文档 / 1,542合格问题 / 2,694正例关系。
- 两源60个文档向量分片及200/1000四个源级池完成；其余18项复用原源级池，重新生成选中题目的候选索引。
- dev200 / test1000 / smoke20（dev子集）；40个任务/档位池，2,400条候选记录。每题1–10正例，平均2.15，超限0题。
- Lite200严格包含于Full1000；两档gold一致且完整保留；dev/test无等价组泄漏。最终数据验收于15:40完成。
- 每个实际来源展示一个真实dev query和正例正文摘录，共20例；核对qrel、两档成员关系及正文SHA-256，未全扫18项历史正文。
- 新版实验由专用入口执行数据/沙箱预检、32k/64k/96k标定及正式评测；本文档不认定模型运行完成。

- 新版KaLM注入前召回，开发200题：R@200 **92.58%** / R@1000 **97.44%**；离线检索指标，不是Agent成绩。
- 新版KaLM注入前召回，正式1000题：R@200 **89.79%** / R@1000 **96.82%**；离线检索指标，不是Agent成绩。

```bash
PREFLIGHT_CONCURRENCY=16 CONCURRENCY=8 DRY_RUN=0 \
bash scripts/launch_benchmark_pos10_qwen38_5090.sh --full
```

- 本次同步范围是2026-11-16数据版本；目标节点预检和模型运行状态应查看对应实验目录，旧模型分数不能作为新版成绩。
- 18个复用源核对元数据、来源身份和选中候选池；没有重新编码或重新扫描其全部历史正文。样本正文核验只覆盖展示的20篇。
- 法律来源在源转换阶段已有整题过滤；表中抽样前排除数仅计suite阶段，不能当作上游总排除数。
- enterprise_zh为公开年报知识，不代表内部制度库；CFQA标注许可不延伸到原PDF正文。
- T2Ranking保留上游train/dev来源后重新抽样，DCI结果不等于上游官方测试成绩。
- KaLM是离线构池模型；补入正例后的100%覆盖不代表原始检索或Agent质量。

[详细页面](index.html) · [20来源样本](index.html#samples) · [样本JSON](source-examples.json) · [数据验收](pos10-data-audit.json) · [覆盖JSON](benchmark-v2-coverage.json) / [CSV](benchmark-v2-coverage.csv) · [阶段状态](benchmark-v2-status.json) · [新版说明](../benchmark_pos10_20261116.md) · [同步前记录](history/before-pos10-20261116/index.html)
