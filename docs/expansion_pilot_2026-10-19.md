# Qwen3.8-27B 新增 document 任务 pilot

只评测原六任务之外的七个 document 任务；使用已冻结的
`results/expansion-plans/20261015-selective-document-v2/query-splits.json`，不重新挑题。
每任务十条，共 **70 条不同查询**，保留每个任务的完整候选语料。

| 任务 | 查询 | 候选文档 |
|---|---:|---:|
| worksurface_backend_developer_rag | 10 | 64 |
| worksurface_logistics_manager_rag | 10 | 100 |
| worksurface_operations_manager_rag | 10 | 226 |
| wixqa_expertwritten | 10 | 6,221 |
| wixqa_simulated | 10 | 6,221 |
| browsecomp_plus | 10 | 99,930 |
| enterprise_rag | 10 | 511,954 |

使用原计划固定的 seed `20261015-selective-expansion-v1`，仅根据 task 和原始 query ID
的哈希选题，不使用 qrels、历史得分或模型输出。七条 smoke 每任务一条，随后执行各任务其余
九条；**7 + 63 = 70**，smoke 不重跑，也不在汇总中重复计数。
这是一轮粗略性能与成本诊断，不能解释为新增 1,872 条查询的完整 benchmark。

## 节点与启动

申请单机 **4 × H100**，建议至少 32 个 CPU 核、64 GB 主机内存，以及至少 **20 GiB
空闲节点本地盘和 65 万空闲 inode**。默认查询并发为 8；可预留 4 小时作为启动、
复制和评测的调度余量，这不是实测用时预测。

在集群分配的节点上执行：

```bash
MODEL_PATH=/mnt/afs2/202610/Qwen3.8-27B \
CUDA_VISIBLE_DEVICES=0,1,2,3 MAX_CONCURRENCY=8 \
LOCAL_CORPUS_ROOT=/tmp/dci-expansion-corpora DRY_RUN=0 bash \
  /mnt/afs/250010100/zxp/dci-bench/scripts/run_selective_expansion_pilot.sh
```

无需另启模型服务。脚本使用：

- 默认权重目录 `/mnt/afs/share/Qwen3.8-27B`。若同一份权重挂载在其他目录，支持
  `MODEL_PATH=/实际权重目录`，或追加 `--model-path /实际权重目录`；CLI 优先于环境变量。
  必须匹配冻结的配置和全部 18 个分片的名称、大小及 SHA256，允许变化的只有挂载位置。
- vLLM 环境 `/mnt/afs/250010100/miniconda/envs/dci-vllm-qwen38`；benchmark 环境
  `/mnt/afs/250010100/miniconda/envs/dci-bench`。
- BF16、TP=4、单请求上下文 65,536、累计模型 token 预算 128,000、单题 600 秒。
- `budgeted-selective` / `selective-document-v1`；契约 SHA256 为
  `c6f4a9e46de4f0ebf9b21aca6f73c86a7b1dfd726783362bc3c476807be6419f`。
- temperature=0、seed=0、max_tokens=1024、enable_thinking=false。
- 服务最多八条序列；七个 smoke 任务并行；后续每任务最多八条查询并行，包含 EnterpriseRAG。
  `MAX_CONCURRENCY` 或 `--max-concurrency` 支持 1–8，CLI 优先。
- 每次运行先在 `LOCAL_CORPUS_ROOT/<run-id>` 创建全新本地语料副本。正文共约 4.95 GiB，
  加上小文件磁盘分配、canonical 数据和 metadata 后需要更多空间。逐文件校验 SHA256 和
  完整文件集合；不使用硬链接、不裁剪候选集，保留原 metadata 和协议字节。
  拒绝远程文件系统和已有目录；退出后不自动删除副本，也不覆盖旧结果。

默认端口为 vLLM `28099` 和 bridge `28131` 至 `28242`。14 个批次各自保留独立的八端口段，
第 i 批（从零开始）使用 `BRIDGE_PORT + MAX_CONCURRENCY*i` 作为基址；单题批次也保留完整端口段，批次之间
不复用端口。首次读取权重前检查全部 112 个 bridge 端口与 vLLM 端口，CPU 预检完成后、
CUDA 探测前再次检查。各批次基址和整体范围写入 `launch-plan.json`。
端口冲突时可在命令前指定 `VLLM_PORT` 和 `BRIDGE_PORT`（整个范围的起点），它们不会改变
任务/模型/协议。每次启动都创建全新的结果目录，不自动恢复或覆盖旧任务。

仅检查计划、不安装依赖、不启动模型：

```bash
DRY_RUN=1 bash /mnt/afs/250010100/zxp/dci-bench/scripts/run_selective_expansion_pilot.sh
```

在节点进行完整 CPU 数据/沙箱检查，保持零外部模型调用：

```bash
bash /mnt/afs/250010100/zxp/dci-bench/scripts/run_selective_expansion_pilot.sh --preflight-only
```

只验证 Inspect 并发启动和完整 bridge/Pi 工具链，不读取权重、不复制候选语料：

```bash
MAX_CONCURRENCY=8 bash \
  /mnt/afs/250010100/zxp/dci-bench/scripts/run_selective_expansion_pilot.sh --inspect-preflight-only
```

该检查使用合成文档与本地 mock 模型响应，真实执行 Inspect、HTTP bridge、Bubblewrap、
`bash`、`read` 和 final 校验。正式启动也会自动执行此检查，无需先单独运行。

提交正式任务前，只检查实际权重挂载与身份，无需 GPU、不扫描候选语料：

```bash
MODEL_PATH=/实际权重目录 bash \
  /mnt/afs/250010100/zxp/dci-bench/scripts/run_selective_expansion_pilot.sh --model-preflight-only
```

该步骤会读取约 55.56 GB 权重计算摘要，耗时取决于共享盘带宽。通过后可使用同一
`MODEL_PATH` 启动正式任务；正式启动仍会重新验证。模型检查记录和实际挂载路径保存在
新运行目录的 `model-preflight.json`、`model-location.json` 中，冻结 manifest 保持原样。

## 自动执行顺序

1. 创建 driver 日志，检查 `/usr/bin/bwrap` 和 `/usr/bin/rg`。缺少时，root 环境先刷新 apt
   索引再安装 `bubblewrap` / `ripgrep`；APT 清除 Conda 的 `LD_LIBRARY_PATH`/`LD_PRELOAD`，
   避免系统动态库冲突。无安装权限或安装失败时提前退出。
2. 验证冻结源码/依赖、数据元数据与查询划分；正式运行先检查全部计划端口，再为本次运行准备
   独立、完整且逐文件校验的 Inspect 工具包。先验收七个独立 Inspect 进程并发，再验收单个
   Inspect 内八样本并发（按实际并发设置缩放），失败则在权重哈希、语料复制和 GPU 前退出。
   然后检查实际权重配置和分片完整性、完整模型身份。随后创建完整节点本地副本，校验正文哈希和候选范围，
   并使用冻结数据验证器再次验证本地 suite。实际路径及身份保存在 `local-corpora.json`
   与 `execution-inventory.json`。
3. 七任务分别执行真实 Bubblewrap sandbox 预检。独立 CPU preflight cap 为 1,800 秒；
   这不会改变实际查询的 600 秒 deadline。
4. EnterpriseRAG 按实际查询并发执行 CPU mock：经过同一 Python adapter 和完整逐题初始化，
   实际执行沙箱隔离检查、一次查询无关的全库 `rg` 扫描与 host `read`。每条必须在 **120 秒**
   内有效完成，且两条 bash 命令 exit_code=0；为真实 600 秒 deadline 留出推理余量。无外部模型调用。
   mock 的已知文档来自候选 ID 映射，不使用查询或 qrels，也不计算相关性成绩。
5. CUDA 探测、确认四张 H100 和软件版本，启动 vLLM，检查模型 ID、权重与 backend manifest；
   执行独立的合成工具调用/最终输出预热。
6. 并行执行七条真实 smoke；只检查输出有效及实际模型/工具调用，不按检索分数设门槛。
   如有失败，保留七条记录并停止，不执行后续 63 条。
7. smoke 通过后执行其余 63 条，汇总全部 70 条唯一查询；退出时关闭本次模型服务进程组。

## 输出与判读

所有运行产物位于：

```text
/mnt/afs/250010100/zxp/dci-bench/results/Qwen3.8-27B/
  job-logs/expansion-pilot-*.log
  expansion-pilot-runs/<UTC时间>-<进程ID>/
    launch-plan.json
    driver-source.py
    model-preflight.json
    model-location.json
    logs/
    cpu-mock/enterprise_rag/
    preflight.json
    runtime.json
    backend.json
    models.json
    warmup.json
    server.log
    evaluation/Qwen3.8-27B/<task>/runs/{smoke,pilot_tail}/
    summary.json
    _COMPLETE
```

`summary.json` 按任务报告 all-sample Recall/F1/nDCG/实际返回窗口 Precision@10、
valid_only、有效/失败数量、token、延迟、repair 次数与原因；另给七任务等权平均。
无效样本在 all-sample 检索指标中计零；缺失 token/延迟单独记录可用数量。
启动、语料校验和合成预热成本保存在各自日志中，不混入 70 条查询的模型 token 总量。
每题原始 final/trace/result 和 Inspect 日志继续保留。

`_COMPLETE` 会区分 `complete` 和 `complete_with_sample_failures`，后者退出码为 1。
启动/校验失败保留 `_FAILED.json`；smoke 失败还保留 `smoke-summary.json`，不写完整 70 条的完成标记。
`--preflight-only` 只产生 CPU 检查记录，不产生模型评测完成标记。
`--model-preflight-only` 只检查冻结源码/依赖与模型身份，同样不产生评测完成标记。

## 2026-10-19 集群启动失败与修复

运行 `20261019T094142Z-1` 在模型挂载检查处失败，未启动 CUDA、模型服务或真实查询。
日志开头的 APT 找不到包随后经刷新索引重试已恢复，两个依赖均成功安装；导致退出的是
`/mnt/afs/share/Qwen3.8-27B/config.json` 不存在。

启动器此前强制覆盖 `MODEL_PATH`，且冻结检查把 `source_path` 也视为模型身份字段。
本次修复支持显式挂载路径，比较时只允许 `source_path` 改变，实际路径仍写入新 manifest；
配置、权重及其他模型/backend 字段继续使用原冻结检查，未修改冻结包。

首次诊断时可见的 `Qwen3.5-27B`、`Qwen3.6-27B` 与 `Qwen3.8-27B-NVFP4` 均与冻结 BF16
模型的配置或分片不符，下载状态目录中的零字节校验标记也不是权重。后续集群运行已通过
`MODEL_PATH=/mnt/afs2/202610/Qwen3.8-27B` 找到并验证完整 BF16 权重。

修复后 18 项针对性测试和 shell 语法检查通过；dry-run 保持 70 条唯一查询及原固定参数。
实际执行模型预检确认：缺失默认挂载时提前退出，显式指定现有 Qwen3.5 目录时因配置不符
提前退出，两种情况均未启动 GPU、模型或语料扫描。

## 2026-10-19 批次切换端口冲突与修复

运行 `20261019T130607Z-1` 已通过完整 BF16 身份校验、七任务 CPU/沙箱检查、四卡运行时
检查、模型服务与预热。第一条 smoke 查询 `pool_r_0003` 有效完成，第二个任务启动前失败：
`bridge port 28131 is unavailable: [Errno 98] Address already in use`。

旧 pilot 调度器为全部批次传入同一个基础端口；冻结 runner 的端口预检使用普通 `bind()`，
上一批关闭后的 TCP `TIME_WAIT` 也会导致该检查报错。本次只修改外层调度：为全部 14 个
批次分配互不重叠的四端口段，并在耗时工作前检查完整范围；冻结 runtime、协议和数据不变。

22 项针对性测试通过，包括真实 loopback TCP 回归：每批关闭后确认旧端口仍被原冻结预检
拒绝，再立即用下一批的新端口通过同一预检，连续覆盖全部 14 批；另验证最后一个预留端口
冲突会在权重哈希或 GPU 工作前退出。此次修复只做 CPU 验证，未重新启动模型评测。
旧失败目录和已完成的一条查询继续保留；重新提交会创建新的运行目录，执行固定 70 条查询。

## 2026-10-20 EnterpriseRAG 超时修复

运行 `20261019T151345Z-1` 的前六条 smoke 有效，EnterpriseRAG `qst_0033` 被 Inspect
600 秒 deadline 终止，剩余 63 条未启动。不是 APT、模型加载或端点故障：该题四次
模型请求均 HTTP 200，Inspect 用量为 11,125 token，未触及 128,000 token 上限。

trace/Inspect 时间线显示，首次模型调用前已耗去约 446 秒。冻结 Pi 每题先对
511,954 个文件逐一 `lstat`，再进行完整 `rg --files` 沙箱检查；这些操作在共享盘上
很慢，而且属于既定单题 wall time。接下来模型的两条默认 10 秒命令、一次 60 秒
全库检索相继超时，最后一次检索在样本 deadline 被中止，没有形成 read receipt 或 final。

本次修复保留冻结 runtime、全部候选、SHA256、安全检查和 600 秒期限，在加载模型前
创建完整节点本地副本，让逐题检查与搜索走本地 I/O。所有实际执行路径统一使用本地 suite；
原始 catalog、metadata、canonical、query ID 和协议保持原样。增加 8 并发 CPU mock
（隔离检查 + 全库扫描 + read），每条 120 秒验收，失败则不启动 GPU。

服务容量从 4 提至 8，EnterpriseRAG 从串行提至 8 并发；七个独立 smoke 任务同时调度。
实际容量和存储路径记录在新运行 manifest/sidecar，backend 检查仅额外允许声明的
`max_num_seqs` 变化，其余模型与服务身份继续严格验证。旧结果目录保留，新运行仍为
固定 70 条查询。存储和并发变化意味着新旧 wall latency 不应当作为同环境性能直接对比。

修复验证：27 项定向 CPU 回归测试、shell 语法和 70 条唯一查询 dry-run 通过；两个本地
suite 均通过原冻结 runner 的 catalog 注册。真实 CPU 运行
`results/Qwen3.8-27B/expansion-pilot-runs/cpu-local-validation-20261020-1/` 完成七任务
全部候选字节校验和真实 Bubblewrap 预检。EnterpriseRAG 单次 sandbox 为 27.57 秒；
8 路同时执行完整逐题启动、全库扫描与 read，每路 41.08–45.64 秒，8/8 有效，
各有一个 read receipt，两条 bash 均 exit 0。记录见该目录
`cpu-mock/enterprise_rag/measurement.json`。本地副本占约 8.2 GiB。

这次只做 CPU/沙箱/mock 验证，GPU 和真实模型调用均为零；未把 CPU mock 作为真实
smoke 或 70 查询 pilot 的验收结果。集群正式启动会重新检查本地副本和并发 I/O。

## 2026-10-20 Inspect 并发解压竞争修复

运行 `20261020T103537Z-1` 中，EnterpriseRAG 已通过真实模型 smoke，样本耗时约
158.21 秒；本地语料修复有效。此次四个 smoke 的失败发生于模型调用之前：
`SandboxInjectionError` 内分别出现 PyInstaller 的 `zlib.error: incomplete or truncated
stream`、`EOFError` / `Failed to setup PYZ archive reader`，不是同一个语料超时问题。

Inspect 0.3.260 的 local sandbox 在多个 Python 进程中使用同一个安装路径
`/var/tmp/.da7be258e003d428`，直接写入同名归档并覆盖解压；其初始化锁仅在单进程内生效。
七个 smoke 同时启动会相互覆盖正在执行的工具包。旧 CPU mock 直接调用 `run_pi_dci`，
绕过了 Inspect 注入与 bridge 初始化，因此未覆盖此次竞争。

修复在启动任何样本前，从已安装、版本固定的 Inspect 二进制包创建本次运行独立的节点本地
工具目录，核对所有文件 SHA256，设为只读，再启动并发进程。bootstrap 仅在进程内重定向
Inspect 工具安装路径及其已导入的别名；每个 local sandbox 的 server state 仍由原 Inspect
创建独立临时目录。未改动已安装依赖、冻结 solver、Pi、查询、预算或评分规则。
工具包来源哈希和文件清单写入 `inspect-tools.json`；实际执行的 bootstrap 与检查器副本
保存在 `launcher-runtime/`，并纳入运行源码哈希。

新增 `inspect-parallel` 前置检查经真实 Inspect → HTTP bridge → 冻结 Pi → Bubblewrap，
每样本完成一次隔离检查、一次 `read`、三次 mock 模型请求及最终 doc_id 校验。分别覆盖
七个独立进程同时启动和一个进程内八样本并发；检查使用独立合成文档，外部模型调用及 GPU
启动均为零。失败会在读取权重、复制语料、加载模型之前停止。

初轮验证 `results/Qwen3.8-27B/inspect-concurrency-validation-20261020-1/`
通过 15/15 样本；通过正式 shell/driver 的全新目录验证记录在
`results/Qwen3.8-27B/expansion-pilot-runs/inspect-cold-validation-20261020-2/`
和 `inspect-cold-validation-20261020-3/`，详细验收见各目录的
`inspect-preflight/result.json`。三轮共 45/45 样本通过；最后一轮耗时约 102 秒。
31 项定向单元测试、shell 语法检查、70 查询 dry-run 通过，另验证了工具包 bootstrap
到原冻结 `run_expansion_task.py` 的 CLI 转发。测试覆盖工具包损坏、非法归档路径、
导入别名重定向、预检失败时不进入权重/语料/GPU，以及原端口和冻结身份约束。
这些是基础设施并发验收，不代表真实模型 70 查询已完成。

## 准备阶段 CPU 验证

2026-10-19 已通过 15 项针对性测试、shell 语法检查和 70 条唯一查询 dry-run。
七个任务的全部候选正文/元数据校验与真实 Bubblewrap sandbox 均通过。
BrowseComp-Plus sandbox 约 85.79 秒；EnterpriseRAG 候选正文校验约 108.51 秒、
sandbox 约 425.40 秒，完整逐题启动 CPU mock 约 **494.04 秒**，得到一个非空 read 凭证。
这些 CPU mock 不使用真实 benchmark 查询，也没有外部模型调用。

494 秒占当前 600 秒单题期限约 82%，只剩约 106 秒供实际搜索和推理。
因此 EnterpriseRAG 在本轮承担大库压力诊断；若发生超时，必须结合初始化开销判读，
不能直接归因为模型无法检索。集群节点会重新执行这些检查，其文件系统性能可能不同。
该 CPU 验收阶段未启动 GPU、真实模型端点或正式 smoke/pilot；后续集群验收见上节。

CPU 阶段日志和 mock 原始 final/trace 保留在
`results/Qwen3.8-27B/expansion-pilot-runs/cpu-validation-20261019-1/`。
期间外层交互进程结束，独立 CPU 子检查继续完成；其最终状态通过单独验收记录核对，
不将该目录标记为完整模型运行。最终验收记录为
`results/Qwen3.8-27B/expansion-pilot-preparation-20261019.json`。

## bash 与 read

当前协议只有 `bash` 和 `read` 两个 agent 工具。`bash` 没有逐条命令白名单，
在只读系统目录和当前任务语料挂载内运行已安装命令，如 `rg`、`grep`、`find`、`ls`、
`cat`、`head`、`tail`、`sed`、`awk`、`sort`、`uniq`、`wc`、`cut`；可写目录为 scratch，网络隔离。
`stable-rg-v1` 为 `rg` 设置默认 `--sort=path`，模型可以显式覆盖该默认参数。

`read` 是 DCI 在 `pi-dci/scripts/dci-run.ts` 的 `createReadTool()` 中定制的 host 侧工具。
它验证语料路径、链接和文件大小，按字符偏移/长度返回正文及原始 `doc_id`，
并记录非空正文的读取凭证（范围、字节数、内容哈希）。`cat` 可以查看相同文本，
但不会生成该凭证。提交前至少一次非空 `read` 是硬条件；它不代表读全文件、
不要求每篇提交文档都有凭证，也不证明相关性。

## 2026-10-21 三档预算配对诊断

本节是独立实验，原冻结 selective、70 题结果及以上历史启动方式保持原样。
新入口为 `scripts/run_size_budget_experiment.sh`，封存配置位于
`releases/size-budget-experiments/paired-v1-20261021/`。

| 完整候选文档数 | 实验组累计 token 上限 | 固定收尾预留 | 单请求上下文 | 配对查询数 |
|---|---:|---:|---:|---:|
| 1–1,000 | 128,000 | 32,000 | 262,144 | 6：三个 WorkSurface 任务各 2 题 |
| 1,001–100,000 | 256,000 | 48,000 | 262,144 | 6：两个 WixQA 和 BrowseComp-Plus 各 2 题 |
| >100,000 | 512,000 | 64,000 | 262,144 | 2：EnterpriseRAG |

这里预算的 k 按 1,000 计，精确值以上表为准。累计预算包括每轮重复输入的历史。
收尾预留包含在总预算内，是剩余预算触发收尾的阈值，不是额外额度，也不保证足以完成任意长历史的最终请求和修复。
单请求上下文检查、请求前预算检查、实际累计用量硬上限、强制 read、重复调用控制、
最终修复、Top-10 输出和评分均沿用 selective；仅替换收尾 reserve 计算并调整预算提示中的总额。

对照组仍用 selective 的 128,000 累计上限及原动态收尾预估（32,000 是下限）；
两组统一在同一个 `max_model_len=262144` 服务上重新执行同样的查询。
旧结果的服务窗口是 65,536，不直接当作本次对照。小档总预算不变，可观察固定收尾预留的影响；
中、大档同时改变总预算和预留，不能把收益单独归因于其中一个参数。

从原 pilot 每任务十题中，用 seed `selective-size-budget-diagnostic-v1-20261021`
与 task/query ID 的 SHA256 选两题，不读取历史得分。共 14 道不同查询，**28 次执行**。
这是已见任务、已评测 query 池上的开发诊断，不作独立留出集结论。
每任务内两组使用相同查询顺序，任务之间交替两组先后；每进程最多 2 题并行，
最多 4 个任务进程同时运行，总并发上限 8。候选语料始终完整，单题 600 秒。

结果写入 `results/Qwen3.8-27B/size-budget-runs/<run-id>/`，包含两组及各档汇总、
逐题配对 Recall/F1/nDCG@10 差值、token 和耗时、峰值单请求输入、收尾事件、失败类别，
以及独立的契约、源码和数据哈希校验。失败样本仍计入全样本检索指标，不通过删除失败提高得分。
重点查看检索改善是否伴随有效输出率下降、token_limit、context_overflow 或 600 秒超时。
大档仅两题，结果用于决定是否扩大验证，不能证明大库总体提升。

在 **4 × H100** 节点启动（自动起停服务，无需另起 vLLM）：

```bash
MODEL_PATH=/mnt/afs2/202610/Qwen3.8-27B \
CUDA_VISIBLE_DEVICES=0,1,2,3 MAX_CONCURRENCY=8 DRY_RUN=0 \
bash /mnt/afs/250010100/zxp/dci-bench/scripts/run_size_budget_experiment.sh \
  --bundle /mnt/afs/250010100/zxp/dci-bench/releases/size-budget-experiments/paired-v1-20261021
```

默认使用新 run ID；不要复用已有目录。服务端口为 28099、bridge 端口段为 28131–28158。
启动先验证原冻结依赖、新包完整性及绑定，执行真实 Inspect/bridge/Bubblewrap CPU 探针、
三档收尾/修复/硬预算探针、全候选语料校验和模型身份校验，然后才启动 GPU 服务。
实际 `/v1/models` 必须明确报告 `max_model_len=262144`，不符则在评测前退出。
KV 容量和 262,144 上下文下的模型运行仍须在该 GPU 节点验收；8 是调度上限，
不保证 8 个最大长度请求同时常驻显存。

命令追加 `--dry-run` 仅打印计划；追加 `--inspect-preflight-only` 验证合成响应的
并发工具链和三档预算边界，不读取权重、不复制完整语料、不启动 GPU 或外部模型调用。

准备验收已通过：8 项针对性单元测试、shell 语法与实际 dry-run；最终封存包完成
27 个真实 Inspect/bridge/Bubblewrap 合成样例，以及 6 个三档预算边界用例
（固定预留触发并完成一次最终修复、超出实际累计预算被拒绝）。
证据：`results/Qwen3.8-27B/size-budget-preparation-20261021/verification.json`；
CPU 原始输出：`results/Qwen3.8-27B/size-budget-runs/paired-v1-cpu-20261021/`。
本次准备未启动 GPU、模型服务或正式检索实验，未重扫全部语料或重算权重哈希；
这些步骤会在实际节点启动时执行。

## 2026-10-21 配对诊断失败审计与 v2 收尾修复

运行 `20261021T075832Z-1` 已完整结束，根 `_COMPLETE` 记录 28 次执行、
25 次有效、3 次失败。所有失败都在 tiered 组；baseline 为 14/14。
各任务 `exit=1` 表示存在无效样本；随后的 `verify exit=0` 表示结果及其
哈希完整性通过，不能解释为样本成功。未发现模型服务 OOM 或启动失败。

| task / query ID | 实际失败 | 关键证据 |
|---|---|---|
| worksurface_logistics_manager_rag / ws_lite_207_aq003 | token_limit | 已用 110,784，剩余 17,216；下一请求预估 25,521，固定预留 32,000，19 turns |
| wixqa_expertwritten / expertwritten/23 | token_limit | 已用 233,262，剩余 22,738；下一请求预估 33,285，固定预留 48,000，15 turns |
| wixqa_simulated / simulated/110 | timeout | bash 命令不完整并重复出现语法错误，最终达到 600 秒期限；失败 trace 的完整用量不可用 |

前两例揭示 v1 固定预留策略的缺陷：历史随工具调用增长，跨过固定阈值的一轮
已花掉最终请求所需预算。硬预算检查正确地拒绝了提交，因此不能通过放松检查
或把失败输出标为有效来处理。

新封存包 `releases/size-budget-experiments/paired-v2-20261021/` 恢复原始 Pi
动态预估，32k/48k/64k 改为各档预留下限。预估考虑下一次 search/read、
最终提交、一次修复及工具输出增长；并不保证任意多工具输出或多次修复都能完成。
三档总预算、查询选择、完整候选、提示词、单轮输出上限、600 秒硬超时、
read 规则及评分保持原样。Pi 源码与原 selective 冻结源码逐字节相同，
新 profile/contract/schema 明确标记 v2；v1 包及历史结果没有修改。
小档恢复了与 baseline 相同的预算控制行为，不能再用它衡量固定预留的效果。

第三例有 5 次 bash 命令输出正好达到 1,024 tokens，命令均为未闭合的 `for`
循环，在 17 个完整的长文件名之后中途结束，工具返回 `exit_code=2`。
前三次重复后触发强制 read 并取得正文凭证，随后又重复两次同类长命令。
输出上限导致命令被截断是高置信度解释，但服务端以 `tool_calls` 而非 `length`
记录停止原因，不能把它描述为已收到明确的截断标记。
此修复不增加时间或输出额度、不引入按时间强制
收尾，也不声称已经消除该样本的超时。必须在新 GPU 运行中检查该轨迹；
CPU 通过不能保证 28/28。新 driver 会将失败样本 ID 和原因打印到主日志，
同时保存在 `summary.json.failed_samples`。

验证：32 项针对性单元测试通过；三档各 2 个真实 Pi/Bubblewrap CPU mock
通过，覆盖固定下限以上提前收尾、一次最终修复，以及实际累计超限仍被拒绝。
六个探针的证据位于
`results/Qwen3.8-27B/size-budget-preparation-v2-20261021/verification.json`。
新旧实验包完整性校验、shell 语法、v2 实际 dry-run 通过。
未启动 GPU、外部模型调用、全语料评估或权重哈希重算。

在 4 × H100 节点重新提交如下命令，自动分配新 run ID；不要继续使用 v1 bundle：

```bash
MODEL_PATH=/mnt/afs2/202610/Qwen3.8-27B \
CUDA_VISIBLE_DEVICES=0,1,2,3 MAX_CONCURRENCY=8 DRY_RUN=0 \
bash /mnt/afs/250010100/zxp/dci-bench/scripts/run_size_budget_experiment.sh \
  --bundle /mnt/afs/250010100/zxp/dci-bench/releases/size-budget-experiments/paired-v2-20261021
```
