# DCI-Bench Lite / Full 执行说明

本版固定为10领域 × 中英双语，共20任务；每任务开发10题、正式50题，Smoke取开发集1题。Lite=200、Full=1000。正式集共1,000 query，两档2,000次执行；开发集每配置400次执行。旧500档、原始语料与历史结果保持独立。

当前进展见[详细报告](project-progress-20261028/index.html#benchmark-v2)和[简短报告](project-progress-20261028/progress.md)。代码与CPU验证不能替代模型质量、吞吐和费用标定；未补齐的原生双语任务会阻止正式启动。

## Qwen3.8-27B：4×RTX 5090 全流程

在 4 卡 5090 节点、仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CONCURRENCY=4 PREFLIGHT_CONCURRENCY=4 \
VLLM_GPU_MEMORY_UTILIZATION=0.80 \
DCI_RUNTIME_CONFIG="$PWD/configs/runtime/frozen-debian13.json" \
DCI_RUNTIME_BUNDLE="$PWD/results/frozen-runtime/20261111-xaDmpt/release-v2/rootfs.tar" \
DRY_RUN=0 bash scripts/launch_benchmark20_qwen38_5090.sh --full
```

入口使用 `benchmark_v2_20261103_v1`（`/mnt/afs2/202611/embedding_models/dci-bench/eval_data/`），固定 Qwen3.8-27B BF16、TP=4、context=65536、GDNWarmupWorker、max_num_batched_tokens=2048。它先检查主机 Bubblewrap、准备 Frozen 缓存并探测 Debian 内工具，然后分别预检 dev/test 数据及 Inspect 沙箱；全部通过才启动模型。指定的离线 Bundle 必须已在目标节点可用；已有有效缓存时可复用。

模型只加载一次：依次执行 dev 32k/64k/96k（每组 400 次，共 1,200 次），调用既有预算选择逻辑，再以选中预算执行 test（1,000 个独立 query × Lite/Full，共 2,000 次）。无预算通过、数据/代码/后端身份不一致或正式门槛失败时停止；不会绕过门槛继续 test。完整成功共 3,200 次 benchmark 执行，另有服务预热请求。现有 smoke 默认行为保持为 40 次。

`CONCURRENCY=4|8|16` 同步控制 vLLM `max_num_seqs`、预热、Inspect 桥接端口范围和题目并发；TP 始终为 4，不表示每卡一个独立模型。默认先用 4；8/16 的目标硬件吞吐及显存稳定性需实测。每个 task/tier 依次运行，内部 query 并发，不能把 40 个 task/tier 数量乘入并发。三组标定和正式执行使用同一并发数。

`PREFLIGHT_CONCURRENCY=1|2|4|8|16` 仅控制 CPU 数据预检，不改变模型并发；全流程 dev/test 各 40 个检查任务。建议先用 4，磁盘允许时可单独调至 8。

输出每次新建于 `results/Qwen3.8-27B/benchmark-v2-full-runs/<UTC>-<random>/`，保留 `launcher.log`、`server.log`、`backend.json`、`dev-{32000,64000,96000}/summary.json`、`budget-selection.json` 和 `test/summary.json`。失败时保留已有产物；重启此 shell 会创建新运行，不会自动恢复整条流程。

默认 `DRY_RUN=1` 仅输出 test 计划，其中 64k 是待标定的占位预算，不加载模型；在上述命令末尾加 `--preflight-only` 可只跑 CPU 数据/沙箱预检。运行期间保持代码与配置不变，以免标定与正式执行的源码指纹不一致。

## 数据与采样

`configs/benchmark_v2.json`固定领域、来源资格和采样数量。`dci_bench/data/benchmark_v2.py`提供采样策略接口，默认random无放回、seed 42；固定输入排序并记录算法版本、输入摘要、同源组及最终ID。新增策略在数据准备阶段注册，执行器始终读取冻结清单。

先做资格过滤、去重、同源与平行组隔离，再采样。Lite和Full使用相同query，候选池保留全部正例，按固定排名补齐且Lite嵌套在Full中。2026-11-16 起新版配置要求每题正例为 **1–10 篇**，超过10篇的整题在抽样前排除，不能截断其正例；20项任务统一重新按seed42抽样。旧冻结配置未声明此规则时仍按历史200篇上限验收，以保留旧实验的可复现性。数量不足或来源不合格时显式阻塞，不能用译本替代原生缺格、复制文档或缩小正式集。

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

数据预检默认同时运行 4 个独立 task/tier 子进程，可用 `--preflight-concurrency 1|2|4|8|16` 调整；它与模型评测的 `--concurrency` 独立。各任务保留 `logs/<task>-<tier>.log`，控制台逐项报告完成进度，全部通过后才写入 `preflight.json`（含并发数和本次耗时）。单项启动后最多 600 秒；失败、超时或中断会停止派发并清理在途进程。数据、协议和 Frozen runtime 校验仍逐项执行，runtime 缓存锁及共享磁盘可能限制加速比例。

## 协议与运行

两个profile为`document-pool-v2-lite`、`document-pool-v2-full`。默认64,000累计输入＋输出token，缓存输入同样计入协议；上下文65,536、单次输出1,024、工具输出8KiB、默认无单题时间上限；耗时仅作参考。最多12轮搜索／读取、24次工具执行，另留提交和一次修复；提交与修复均消耗累计预算。32k／64k／96k通过`--token-budget`选择并纳入运行指纹。

2026-11-14 修订为 `document-pool-v2-tokenizer-finalize-v3`。已知消息、工具定义和控制提示先经过与 Inspect 相同的转换，再调用当前 vLLM 的 CPU `/tokenize` 接口计数。预测只对尚未生成的模型输出和未来工具结果保留增长量，不再把已知文本按 UTF-8 字节计 token，也不再给每次未来请求叠加 4096 safety tokens。V2 的 ID 错误反馈使用固定短提示；最终回答和一次修复均保留，最终阶段始终禁用工具。

`control_events[].forecast` 记录立即收尾的 `finalReserve`、尚无正文证据时先读取再收尾的 `requiredReadReserve`，以及继续一次搜索所需的 `reserve`。未来工具返回按每次最多 8192 字节、最近一轮工具调用数量（至少一次）预测，必需读取按一次返回预测；未知输出沿用 1024 token 上限，并保留 32 token 的边界余量。这不是任意多工具数组都能成功收尾的保证。每次模型请求，包括自动连续的工具轮，都用准确输入计数加输出上限检查累计余额与上下文；实际 usage 继续记账。`token_checks` 记录预测输入、实际输入与差值，差值非零会停止，不能静默回退到字节估算。

早期版本会过晚收尾，随后动态预留版本又因叠加字节上界而过早停止；v3 同时回归这两类问题。有效率与 Recall 必须重新标定，不能把旧 dev 结果或只补跑的失败样本混入新版标定。`repair_attempts` 与 `repair_reasons` 只记录通过请求前检查并进入调用的修复；预算或截止时间拦截的修复不计作已尝试。99% 有效率门槛、评分及各档预算保持不变。

Qwen 启动器及 `run_benchmark_v2.py` 的 vLLM 路径自动从后端身份设置计数服务，无需修改原命令。启动器在预热后额外检查 search/read/final/repair 四种合成请求，保存 `tokenizer-preflight.json`，不一致则不进入 benchmark。直接调用底层 Pi 时须配置 `DCI_TOKEN_COUNT_BASE_URL`、`DCI_TOKEN_COUNT_MODEL`、`DCI_TOKEN_COUNT_PYTHON`。当前计数适配器针对 Inspect OpenAI-compatible → vLLM；其他服务需要相同模型的兼容计数接口，未验证的远程服务不能以此宣称 v3 可用。

v3 的本机离线回归使用封存的 `20261114T081056Z-0mysDb` smoke：40 题共 230 次请求的输入计数与原服务 usage 精确一致；40 个旧 forced-final 状态均允许继续一次搜索，最小剩余余量为 11,459 token。旧 `20261113T140604Z-Ap4MVU/dev-96000` 的 `legal_zh/dev-lite/00057` 收尾状态下，新 final＋repair 预留为 27,435，低于剩余 27,982；还会提前要求读取。测试位于 Pi `scripts/token-budget.test.ts`，并包含 32k/64k/96k 的真实 Bubblewrap＋脚本化 HTTP 状态机检查。这些检查不生成新的模型搜索结果，也不证明 Recall 或新一轮 GPU Smoke 已通过。离线模式直接使用冻结 `tokenizer.json`，避免不同 Transformers 版本重建预分词器造成漂移；正式执行只使用服务自身计数。

修复后的低成本验证包括 `tests.test_document_pool_v2_budgets`、Pi `scripts/dci-run.test.ts` 和真实 Frozen Bubblewrap 测试。CPU 脚本化回答能检查 XML 回答进入修复且不执行工具，但不能证明 Qwen 的 XML 输出率下降，也不能证明正式门槛通过。真实模型验收先将上方启动命令末尾的 `--full` 去掉，运行 40 次 Smoke；通过后以新目录重跑三档完整 dev 标定。相同历史的服务兼容性对照应保留原始请求与响应，比较当前省略工具字段和显式 `tools: [], tool_choice: "none"`；旧 `prepare_harness_final_replay.py` 针对 v1 profile，不能直接拿来回放此次 v2 失败。

本次另准备了 `results/harness-validation/v2-finalize-v2-20261114-wire-replay/`：从 `legal_zh` 的 `00057`、`00061` 封存轨迹重建最终请求，校验旧协议哈希，每题仅工具字段有两个变体，共 4 次请求。它们不是当时抓取的原始下游请求，直连 vLLM 也不经过 Inspect，只用于隔离字段形状的影响。原服务已启动时运行以下命令；默认 dry run 不发送请求，明确设置 `DRY_RUN=0` 才执行，并将响应写入新的随机目录：

```bash
bash results/harness-validation/v2-finalize-v2-20261114-wire-replay/run-replay.sh
# 在同配置 Qwen3.8-27B 服务上执行四次收尾请求：
DRY_RUN=0 bash results/harness-validation/v2-finalize-v2-20261114-wire-replay/run-replay.sh http://127.0.0.1:29099/v1
```

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

并发比较输出完成query／小时、P95、失败率和GPU driver利用率；推荐有效率≥99%的最高实测吞吐配置；P95仅作参考，不设统一时间阈值。MIG或共享GPU的driver指标不等于本进程利用率，不可据此宣称GPU已充分利用。部分开发集只能用于工程诊断。

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

正式启动要求全20任务、完整开发标定、匹配的数据／代码／模型／并发／预算指纹、三组预算报告及其可复算的选择证据和成本门禁。本地按实测吞吐加20%预留报告预计耗时，默认不以耗时阻止启动；只有显式指定`--wall-time-limit-hours`时才执行作业时间保护；API正常正式执行≤¥800，探测及重试计入单模型≤¥1000总额。正式50题固定，门禁失败就停止。

API执行额外传`--prices /path/to/prices.json`，字段为`currency: CNY`、`model_key`、`as_of`、`source`及每百万token的`input_per_million`、`output_per_million`、`cache_read_per_million`、`cache_write_per_million`。报价来源须含实际单价依据，换汇时记录汇率和日期；仓库不内置可能过期的市场价格。缓存拆分未知时保守估算；usage缺失时在当前batch结束后停止派发，保留该批预留额度，不能按免费处理。费用为provider usage乘固定报价，最终账单仍需核对供应商账单。

每个batch启动前按最大token单价预留费用；禁止隐式模型重试，失败／中断留痕。`--resume`仅允许同一配置；已发布query验证hash后跳过。中断的未知费用保留额度，重新执行会另计预留，避免低估花费。旧暂存轨迹归档，尝试日志与ledger保留。

API正式启动会自动计入三组开发预算报告中已记账或保留的费用；另须显式传`--prior-cost-cny <金额>`记录这三组之外的Smoke、并发探测等已发生费用，确无其他费用时传`0`。未知费用不能填零。启动前预测与运行中余额检查都包含这些前期费用；Inspect重试和底层SDK重试均关闭。

报告分别输出Lite、Full的20任务等权宏平均、配对区间、领域与语言诊断分数、费用、耗时、失败率和阶段计时。失败query按零计入主指标。`analyze_benchmark_v2.py compare-models --left ... --right ... --output ...`仅比较同一冻结正式集和协议下的两模型结果；置信区间按同源组配对，不能把两档重复query当独立样本。

## 硬件与执行时间参考

`summary.json`中的`hardware_sessions`记录每次启动／续跑的运行器硬件，以及backend清单提供的模型服务端硬件（GPU型号、数量、显存、CPU、内存及采集时间）；`task_executions`通过`hardware_session`索引关联每批任务及其`elapsed_seconds`。单题耗时保留在`samples[].wall_seconds`，整轮耗时为`elapsed_seconds`；并行单题耗时不能相加充当整轮耗时，异常退出后恢复的整轮计时可能包含停机间隔。

模型服务端记录来自`backend.runtime`；现有部署脚本通过`--runtime-info`传入，旧清单缺失的字段记为未知。远程API没有公开服务端配置时也标为未知。运行器本机的GPU清单不等于模型实际分配的GPU。硬件和耗时用于解释结果，不进入质量评分。`--wall-time-limit-hours <小时>`为可选作业保护，默认关闭；该参数记入启动指纹，中断结果保留供续跑。
