# DCI-Bench Lite / Full 执行说明

本版固定为10领域 × 中英双语，共20任务；每任务开发10题、正式50题，Smoke取开发集1题。Lite=200、Full=1000。正式集共1,000 query，两档2,000次执行；开发集每配置400次执行。旧500档、原始语料与历史结果保持独立。

当前进展见[详细报告](project-progress-20261028/index.html#benchmark-v2)和[简短报告](project-progress-20261028/progress.md)。代码与CPU验证不能替代模型质量、吞吐和费用标定；未补齐的原生双语任务会阻止正式启动。

## 数据与采样

`configs/benchmark_v2.json`固定领域、来源资格和采样数量。`dci_bench/data/benchmark_v2.py`提供采样策略接口，默认random无放回、seed 42；固定输入排序并记录算法版本、输入摘要、同源组及最终ID。新增策略在数据准备阶段注册，执行器始终读取冻结清单。

先做资格过滤、去重、同源与平行组隔离，再采样。Lite和Full使用相同query，候选池保留全部正例，按固定排名补齐且Lite嵌套在Full中。正例超过200的query在抽样前排除并计数。数量不足或来源不合格时显式阻塞，不能用译本替代原生缺格、复制文档或缩小正式集。

数据准备仅生成抽样后的元数据，引用既有共享语料；执行时复制本题候选正文并核对hash。当前已准备的13任务可直接审计和预检：

```bash
PY=/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python
export PATH=/mnt/afs/250010100/miniconda/envs/dci-bench/bin:$PATH
export LD_LIBRARY_PATH=/mnt/afs/250010100/miniconda/envs/dci-bench/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
SUITE=/mnt/afs2/202609/embedding_models/dci-bench/eval_data/benchmark_v2_20261029_v1
$PY scripts/prepare_benchmark_v2.py audit --output "$SUITE"
$PY scripts/run_benchmark_v2.py --suite "$SUITE" --split smoke --allow-partial
$PY scripts/run_benchmark_v2.py --suite "$SUITE" --split smoke --allow-partial \
  --preflight-only --output results/benchmark-v2/data-preflight
```

新增来源后，用`prepare_benchmark_v2.py prepare --catalog <catalog.json> --config configs/benchmark_v2.json --seed 42 --output <新目录>`准备新版本；已有目录不可覆盖。

默认命令仅输出计划。`--preflight-only`检查数据与运行清单，不调用模型，也不代表完成沙箱验收。实际沙箱CPU验收可用`check_benchmark_v2.py --suite "$SUITE" --task finance_en --output <新目录>`；它使用本机脚本化HTTP provider，执行开发集两档、真实Bubblewrap和结果发布，并检查恢复不重复请求。沙箱依赖先运行`bash scripts/ensure_pool_smoke_tools.sh`检查并补齐。

## 协议与运行

两个profile为`document-pool-v2-lite`、`document-pool-v2-full`。默认64,000累计输入＋输出token，缓存输入同样计入协议；上下文65,536、单次输出1,024、工具输出8KiB、硬超时300秒。最多12轮搜索／读取、24次工具执行，另留提交和一次修复；提交与修复均消耗累计预算。32k／64k／96k通过`--token-budget`选择并纳入运行指纹。

每任务、每档位一个batch，内部query并发且Pi会话独立。运行器复用常驻模型服务及一次准备的Inspect工具包；工作区预取默认由新版运行器启用，`DCI_WORKSPACE_PREFETCH_BATCH_SIZE`不超过并发数，可用`DCI_WORKSPACE_PREFETCH_MAX_BYTES`约束正文占用。逐bash Bubblewrap隔离、候选精确隔离与复制后的正文hash验证保留。逐query记录准备、等待、桥启动、Pi执行与清理时间，以及沙箱校验、预检和bash调用耗时；bash耗时包含命令执行，不能当作纯沙箱开销。

模型服务由既有部署入口启动一次，随后提供审计过的`backend.json`；上下文须为65,536，本地服务`max_num_seqs`须覆盖并发数。API后端使用`build_api_manifest`记录provider、固定模型版本及endpoint；密钥通过`DCI_API_KEY`环境变量传入。远程API必须支持本协议的工具调用、temperature/seed与usage回传；实际兼容性以Smoke验证为准。

```bash
# 服务常驻后先跑Smoke；缺格时仅Smoke/dev允许显式 --allow-partial。
$PY scripts/run_benchmark_v2.py --suite "$SUITE" --split smoke \
  --backend-manifest /path/to/backend.json --concurrency 4 \
  --execute --output results/benchmark-v2/smoke

# 比较4/8/16并发；每个输出目录唯一。
$PY scripts/run_benchmark_v2.py --suite "$SUITE" --split dev \
  --backend-manifest /path/to/backend.json --concurrency 8 --token-budget 64000 \
  --execute --output results/benchmark-v2/dev-c8-64k
$PY scripts/analyze_benchmark_v2.py compare-concurrency \
  --reports /path/to/c4/summary.json /path/to/c8/summary.json /path/to/c16/summary.json \
  --output results/benchmark-v2/concurrency-selection.json
```

并发比较输出完成query／小时、P95、失败率和GPU driver利用率；推荐有效率≥99%、P95≤300秒的最高实测吞吐配置。MIG或共享GPU的driver指标不等于本进程利用率，不可据此宣称GPU已充分利用。部分开发集只能用于工程诊断。

## 质量、费用和正式启动

固定并发后分别跑完整开发集32k／64k／96k，每档400次；用配对组bootstrap报告两档Recall差及95%区间。Lite和Full各自的20任务等权宏平均Recall下降均须≤2个百分点，并满足有效率≥99%，再选择最低预算；两档的提升与下降不能互相抵消。开发集选择结果不是等价性证明。

```bash
$PY scripts/analyze_benchmark_v2.py select-budget \
  --reports /path/to/dev32/summary.json /path/to/dev64/summary.json /path/to/dev96/summary.json \
  --output results/benchmark-v2/budget-selection.json
$PY scripts/run_benchmark_v2.py --suite "$SUITE" --split test \
  --backend-manifest /path/to/backend.json --concurrency 8 --token-budget 64000 \
  --calibration /path/to/dev64/summary.json \
  --budget-selection results/benchmark-v2/budget-selection.json \
  --execute --output results/benchmark-v2/formal
```

正式启动要求全20任务、完整开发标定、匹配的数据／代码／模型／并发／预算指纹、三组预算报告及其可复算的选择证据和成本门禁。本地按实测吞吐加20%预留估算≤24小时；API正常正式执行≤¥800，探测及重试计入单模型≤¥1000总额。正式50题固定，门禁失败就停止。

API执行额外传`--prices /path/to/prices.json`，字段为`currency: CNY`、`model_key`、`as_of`、`source`及每百万token的`input_per_million`、`output_per_million`、`cache_read_per_million`、`cache_write_per_million`。报价来源须含实际单价依据，换汇时记录汇率和日期；仓库不内置可能过期的市场价格。缓存拆分未知时保守估算；usage缺失时在当前batch结束后停止派发，保留该批预留额度，不能按免费处理。费用为provider usage乘固定报价，最终账单仍需核对供应商账单。

每个batch启动前按最大token单价预留费用；禁止隐式模型重试，失败／中断留痕。`--resume`仅允许同一配置；已发布query验证hash后跳过。中断的未知费用保留额度，重新执行会另计预留，避免低估花费。旧暂存轨迹归档，尝试日志与ledger保留。

API正式启动会自动计入三组开发预算报告中已记账或保留的费用；另须显式传`--prior-cost-cny <金额>`记录这三组之外的Smoke、并发探测等已发生费用，确无其他费用时传`0`。未知费用不能填零。启动前预测与运行中余额检查都包含这些前期费用；Inspect重试和底层SDK重试均关闭。

报告分别输出Lite、Full的20任务等权宏平均、配对区间、领域与语言诊断分数、费用、耗时、失败率和阶段计时。失败query按零计入主指标。`analyze_benchmark_v2.py compare-models --left ... --right ... --output ...`仅比较同一冻结正式集和协议下的两模型结果；置信区间按同源组配对，不能把两档重复query当独立样本。
