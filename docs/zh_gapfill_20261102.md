# 中文软件与企业数据接入（2026-11-02）

11月3日复核：中文两源的排名、200/1000源池、19任务suite验收及38个数据预检均已完成。原集群作业失败发生在最后的报告浏览器检查，原因是该节点缺少 `libdbus-1.so.3`；不需要重跑GPU。BrowseComp+也已完成源池，尚未合入当前19任务suite。恢复过程见文末。

本轮为 `software_zh` 接入 T2Ranking 软件技术子集，为 `enterprise_zh` 接入 CFQA 公司年报中的企业业务正文。源数据验收、KaLM 排名、候选池、冻结 suite 是不同阶段；只有最后一项完成后，当前覆盖才由 17/20 更新为 19/20。英文网页 BrowseComp+ 和正式评测模型不属于这条中文构池命令的执行范围。

最终来源与运行目录：`/mnt/afs/250010100/zxp/dci-bench-zh-data/20261102_v4`。T2Ranking 完整原始文件复用 `20261102_v1/raw/t2ranking`；CFQA 原始数据与 33 份官方 PDF 复用 `20261102_v2/raw/cfqa`。保留旧 suite、失败下载与中间版本。

## 来源与转换

| 任务 | 固定来源 | 标签及转换 |
|---|---|---|
| `software_zh` | [THUIR/T2Ranking](https://huggingface.co/datasets/THUIR/T2Ranking/tree/2a369a430a70979223f1b9a41b1919774d46b432)，revision `2a369a430a70979223f1b9a41b1919774d46b432`；数据卡 Apache-2.0 | 原始 train/dev 的人工四级标注；固定 grade≥2 为正例，输出 binary qrels，原等级另存。严格筛中文软件知识，排消费/硬件问题与纯代码生成；正文去重后合并 alias 和正例。 |
| `enterprise_zh` | [ygan/CFQA](https://github.com/ygan/CFQA/tree/61c9ec3c4335d0411a1735cd228af8b3ead114fc)，commit `61c9ec3c4335d0411a1735cd228af8b3ead114fc`；标注仓库 MIT | 固定 2022 年报，使用原始人工 `答案出自` 的全部 **1-based PDF 物理页**。从巨潮资讯定位唯一原版官方报告，按公司、年份、封面和 SHA-256 核实；不信任有误的原股票代码，仍保留原字段。 |

CFQA 共下载 **33 份官方 PDF，142,064,262 字节**；7,697 页中 7,670 页可解析中文正文（99.65%）。排除财务报表章节、目录、数字表格、短正文与非企业主题后，得到 **1,728 个不同正文页、63 条查询、67 条原始正例关系**；每题 1–4 个正例。报告与明确的改写模板合并成 **25 个分组**，seed42 本地与全局采样的 dev10/test50/smoke1 预检查通过。公开标注文件本轮实读 10,311 条，原 train/dev/test 保留在 host metadata。

内容复核共排除 8 条原始查询：6 条涉及答案与页码证据或问答范围不一致，另 2 条分别缺少问题要求的前景证据、以财务数字为主。均整题排除，没有替换页码或生成新 gold。这是 assistant 内容排除审核；剩余 gold 仍完全来自上游人工证据页。旧候选保留并标记 superseded。

[CFQA 原论文 §4.1–4.2](https://link.springer.com/article/10.1140/epjds/s13688-025-00601-6) 说明答案由人工编写并交叉核验，Answer Source 为报告证据页；ChatPDF 用于辅助核对，不是本轮新标签来源。

CFQA 是企业业务、研发、治理、人才和经营风险知识，不能代表内部制度/办事流程库。MIT 是 CFQA 标注仓库许可；公司年报仍保留官方来源与权利，没有把标注许可扩展到 PDF 正文。答案文本匹配只用于排除错误报告或证据页，不能生成新标签。

T2Ranking 完整 collection 固定为 3,659,243,528 字节，SHA-256：

```text
07b84e543e9ba696124a727c00629d5bce586631648c436c98dd6e9b146da212
```

最终软件子集为 **5,635 个不同正文、61 条查询、131 条正例关系**，每题 **1–6 个正例**；原 split 为 dev10/train51，形成 61 个分组。全量 collection 的 2,303,643 条正文行可解析，TSV 解析率 100%（原计数另含表头）。按正式 benchmark 的查询去重规则及 seed42 分组采样复核 dev10/test50/smoke1。原始等级与全部保留查询的 grade≥2 正例均保持不变。

正文二次筛选排除了父候选中的 9,944 篇正文，并因正例被排除而整题删除 30 条查询。50 篇确定性正文样本的内容复核记录已保存；领域筛选和抽查由 assistant 完成，不是全库逐篇人工审核，也不产生新 gold。失败/中间候选和原始全量扫描证据均保留；最终计数以 `source-readiness.json` 和 `evidence/t2ranking/` 为准。

T2Ranking 可追溯到固定数据集版本、原始 split、query ID 和 passage ID；上游 TSV 未逐篇给出原网页 URL，不能声称完成原网页级来源分组。已有共用正例、正文 alias、规范化问题和等价模板按组隔离。CFQA 采用具有独立企业语义的证据页作为文档单元，保留公司、报告年份、官方 PDF 地址与物理页码；数量不是整份年报数。

两源都发布全部合格查询，再由 benchmark-v2 按 seed42 冻结 dev10/test50；smoke 从 dev 取。领域过滤不得丢弃某条保留查询的任何已知正例；正例缺失、不可用、超过 200 时排除整题。正文 alias、原始等级/页码、原 split、转换与排除记录均保留。未标注候选不称作已确认负例。

## 四卡 RTX 5090

CPU 来源与导出、两源共同划分预检及真实四卡 dry-run 已完成。启动器与正文门禁的 25 项 CPU 测试、4 项子检查通过。执行：

```bash
cd /mnt/afs/250010100/zxp/dci-bench
DRY_RUN=0 bash scripts/launch_zh_gapfill_5090_4gpu.sh
```

只做真实输入、环境及四卡分片计划检查，不加载模型：

```bash
cd /mnt/afs/250010100/zxp/dci-bench
DRY_RUN=1 bash scripts/launch_zh_gapfill_5090_4gpu.sh
```

命令要求共享目录可访问，并使用四张 RTX 5090。保留调度器设置的 `CUDA_VISIBLE_DEVICES`，未设置时用 `0,1,2,3`。编码固定为 `lychee_reranker_5090_kalm451` 环境、KaLM multilingual mini instruct v2.5、float32、max_length8192、batch8；四个 worker 各写独立缓存，控制进程核验后合并。实际 GPU smoke 成功才继续编码。CPU 预检和报告使用现有 `dci-bench` 环境，相关依赖在 GPU 前检查。

成功路径依次为：源 release/export 门禁 → 实际 GPU smoke → 并行文档编码 → 排名与 200/1,000 源池 → 最终嵌套 Lite/Full → 旧 17 任务选择不变及新正文 hash → 38 个零模型调用数据预检 → 同步 HTML/Markdown/覆盖 JSON/CSV → 44 项浏览器检查。源池两档是中间产物，最终 benchmark-v2 从 Full 构造 Lite 并检查 Lite⊂Full；不能拿中间两档声称嵌套验收通过。

这只执行数据准备，不启动 agent 正式评测。

```text
运行日志/分片计划：run/.zh_5090_4gpu_state/
源完成回执：run/<source_id>/catalog-row.json
最终suite：suites/benchmark_v2_20261102_zh_v1/
最终验收：evidence/suite-acceptance/status-ready.json
数据预检：evidence/data-preflight/preflight.json
```

GPU 完成但 CPU 收尾失败时，保留日志和产物，先检查第一个错误；不要删除或覆盖已有 suite/验收目录再跑。

## CPU 重建

本轮完整 T2Ranking 已下载并通过固定 SHA 校验，无需重复下载。CPU 重建必须使用全新的 release/evidence 输出目录；命令和输入 hashes 记录在对应 evidence 中，不要覆盖现有 release。

```bash
cd /mnt/afs/250010100/zxp/dci-bench
python scripts/prepare_t2ranking_zh_20261102.py \
  --raw-root /mnt/afs/250010100/zxp/dci-bench-zh-data/20261102_v1/raw/t2ranking \
  --output /path/to/new/t2ranking-release --evidence-dir /path/to/new/t2ranking-evidence \
  --splits dev train --workers 8
python scripts/refine_t2ranking_zh_20261102.py \
  --parent-release /path/to/new/t2ranking-release \
  --output /path/to/new/t2ranking-final --evidence-dir /path/to/new/t2ranking-refinement-evidence
python scripts/prepare_cfqa_zh_20261102.py prepare \
  --raw-root /mnt/afs/250010100/zxp/dci-bench-zh-data/20261102_v2/raw/cfqa \
  --acquisition /mnt/afs/250010100/zxp/dci-bench-zh-data/20261102_v2/raw/cfqa/official-2022 \
  --output /path/to/new/cfqa-release --evidence /path/to/new/cfqa-evidence
```

## 未采用的快速替代

- FinRAGBench-V 的 855 条中文 QA 中，157 条标为文本类；本轮严格企业主题预筛仅 17 条，宽口径诊断也未达 60。未下载完 PDF、未完成页级验收，不能按 855 条直接补格；保留查询级排除记录和已下载片段。
- 本地 DuRetrieval 快照为 100,001 文档/2,000 查询；严格软件 query 与全部可用正例的交集仅 6 条，不能替代完整 DuReader-Retrieval 的领域验收，也没有拼接异源查询来凑数。
- DQA 保留为历史候选，未以缺少逐条原生中文证据的样例解除门禁。

证据见 [CPU 检查记录](../results/implementation/20261102-zh-gapfill/)、[当前详细报告](project-progress-20261028/index.html) 与 [简版进度](project-progress-20261028/progress.md)。

## 11月3日报告检查失败与CPU恢复

原中文控制进程日志 `results/gapfill-controller-logs/20261102T133038Z-bft8Dh.log` 记录退出码0。两源 `catalog-row.json`、19任务suite的 `status-ready.json` 和38个零模型调用数据预检均已闭合。首次报告发布已写出HTML与进度文件，但随后Chromium启动失败：

```text
error while loading shared libraries: libdbus-1.so.3:
cannot open shared object file: No such file or directory
```

原始失败日志保留在 `results/implementation/20261102-zh-gapfill-accepted/report-ui/browser-check.log`。这是报告运行环境错误，不是中文构池失败。当前CPU节点使用同一共享Playwright运行时，浏览器启动检查通过。

发布脚本现在会核对并复用一致的 `report-audit.json`，每次尝试写入新的 `results/implementation/zh-gapfill-report-*/report-ui/`，发布前先做浏览器启动检查。旧失败目录、源池和suite保留；不覆盖验收证据。单独重试报告时，在具备浏览器依赖的CPU节点执行：

```bash
cd /mnt/afs/250010100/zxp/dci-bench
LD_LIBRARY_PATH=/mnt/afs/250010100/miniconda/envs/dci-bench/lib \
  /mnt/afs/250010100/miniconda/envs/dci-bench/bin/python \
  scripts/publish_zh_gapfill_report_20261102.py \
  --data-root /mnt/afs/250010100/zxp/dci-bench-zh-data/20261102_v4
```

不要重新执行整条构池启动器来重试报告：它后面的suite验收采用独占写入，已有suite应保留。BrowseComp+源池已完成的状态会在报告中单独反映；它尚未合入当前suite，不能把19/20提前写成20/20。

本次CPU恢复已成功：11项发布与恢复测试、44项真实浏览器检查、70项报告哈希/链接/原产物保留检查通过，发布退出码0。HTML、Markdown和覆盖JSON/CSV已同步，旧失败日志、中文源池和19任务suite保持不变。证据：[故障与数据状态](../results/implementation/zh-gapfill-recovery-20261103T104517Z/diagnosis.json)、[恢复测试](../results/implementation/zh-gapfill-recovery-20261103T104517Z/tests-final.json)、[恢复验收](../results/implementation/zh-gapfill-recovery-20261103T104517Z/recovery-validation-final.json)、[浏览器日志](../results/implementation/zh-gapfill-report-20261103T105251518161Z/report-ui/browser-check.log)。本次没有GPU推理或正式模型评测。
