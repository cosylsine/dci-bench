# DCI-Bench 项目进展（2026-11-13 文档同步）

Frozen RootFS 已接入 Lite / Full 运行链，当前为显式启用的开发交付；Host Tools 仍为默认，Code Track 保持独立。Runtime 验证依据为 2026-11-11，数据准备沿用 2026-11-03 快照；11月13日另完成开放 Bash 提示词修订与真实沙箱合成回归，未启动模型。

- [详细进展](index.html) · [Frozen Runtime](index.html#runtime) · [架构图 SVG](workflow.svg) / [PNG](workflow.png) · [规则图 SVG](harness-rules.svg) / [PNG](harness-rules.png)
- [完整方案](../../DCI-Bench_Frozen_RootFS_Runtime_Design_and_Implementation.md) · [Runtime 使用说明](../frozen_runtime.md) · [当前20任务覆盖](index.html#data)

## Frozen RootFS 设计与实施

- 动机：固定 Agent 的 Bash 用户态工具及版本，避免宿主发行版、安装环境或升级改变被测能力；仍共享宿主内核。
- 可信构建：固定 Debian 13 base、签名 APT snapshot 与 rg 14.1.1-1+b4，输出 OCI、RootFS tar、摘要和工具清单；不额外安装 Python。
- 准备：离线校验 Bundle / 文件树、安全解包、文件锁和原子 READY；多任务共享只读 RootFS，样本内不下载、不安装、不重新展开。
- 部署：AFS 并发 flock 在实施节点实测失败；持久 Bundle 留在 AFS，缓存明确为 `/tmp/dci-bench-runtime`。重启若丢失本地缓存，从同一 Bundle 离线重建。
- 隔离：每次 Bash 经 Bubblewrap，RootFS 与 corpus 只读，Scratch 独立；`/tmp` 映射 `scratch/tmp` 并计入同一配额。禁网，不挂宿主工具、qrels、密钥、proc 或 GPU。
- Host 边界：Inspect、Pi、HTTP Bridge、自定义 read 和评分留在宿主侧；Bash 读正文不会生成 read 凭据。候选池与评分协议不变；11月13日另修订中性 Prompt，两档共用。
- 身份：镜像摘要、平台、文件树及策略摘要贯穿 final / trace / manifest / fingerprint；恢复、完成批次复用和校准拒绝跨 Runtime 混用。部署绝对路径不进入逻辑身份。
- 启用：当前 task / document-pool / v2 入口传 `--runtime-config`；5090 launcher 用 `DCI_RUNTIME_CONFIG`，加载模型前预检。历史 sealed-bundle 入口明确拒绝 Frozen。
- 产物：`results/frozen-runtime/20261111-xaDmpt/release-v2/`；离线本地交付，未推送 Registry。完整摘要见 [Runtime 状态 JSON](frozen-runtime-status.json)。

## 开发验证（证据日期 2026-11-11）

- 105 项不同 Python 定向用例最终通过（包含初次夹具失败修正后的复测），非全仓回归；Pi 31/31 通过、0 skip，完整 `npm run check` 通过。
- v1 Small / v2 Lite 真实沙箱预检、OCI / Bundle 摘要与平台检查、两个新进程的并发离线缓存复用通过。
- 资源回归覆盖独立 Scratch、`/tmp` 配额、输出超限、超时/取消后的子进程清理及宿主路径/网络隔离。
- Inspect：先 7 独立进程，再单进程 8 样本并发，共 15/15；真实 HTTP/Bash/read/final，但使用 generic v3、合成 corpus 与 mockllm。
- 本次 Runtime 验收外部模型调用 0、未启动 GPU；这些证据不代表正式候选数据验收、模型质量或吞吐结论。
- [实施验收记录（本地）](../../results/frozen-runtime/20261111-xaDmpt/acceptance.json) · [本页状态摘要](frozen-runtime-status.json)

## 开放 Bash 提示词（2026-11-13）

- 移除 rg-first 与固定检索顺序；允许已安装命令、管道、循环和脚本。stable-rg 只描述可选工具行为，不规定策略。
- 保留至少一次非空 read、预算与调用上限；read 后按剩余额度恢复工具，V2 最终修复不重开检索。
- 新合同 SHA 单独记录，历史结果不改写。已准备 [Qwen3.8-27B 4×5090 smoke 计划与命令](../frozen_qwen38_smoke_20261113.md)：20 个 query × 两档 = 40 次，尚未启动模型。

## Lite / Full 与数据快照（2026-11-03）

- 20/20任务：10领域 × 原生中英双语；每任务开发10、正式50、Smoke1（开发子集），seed42，同源组隔离。
- 已准备开发200 / 正式1,000 / Smoke20题；计划两档执行400 / 2,000 / 40次。准备量不代表执行量。
- Lite200⊂Full1000，保留全部正例；新BCP所选Full的31,822篇正文哈希通过，原19任务沿用既有正文证据；40个零模型调用数据预检通过。
- 两档同查询、同Prompt、同预算：默认累计64k、上下文65,536、单次输出1,024、工具8KiB、12轮搜索、24次工具、一次提交和最多一次修复。
- 11月10日时间策略：默认无单题/整轮时间上限；每次Bash仍有默认10/最大60秒限制。记录硬件和耗时，仅供参考；API目标≤¥1000/模型、预留20%。
- KaLM补正例前召回：开发200题 R@200 / @1000 = 89.65% / 96.96%；正式1,000题 = 88.57% / 96.43%。属于离线检索，不是 Agent 成绩。
- [数据覆盖 JSON](benchmark-v2-coverage.json) / [CSV](benchmark-v2-coverage.csv) · [KaLM口径与逐题来源](kalm-pre-injection-recall.json) · [11月3日阶段状态](benchmark-v2-status.json)

## 待验收与解释边界

- P6：Frozen 真实模型 smoke、同查询/模型/Prompt/预算的 Host–Frozen 消融、开发标定与官方默认启用决策；不合并旧 Runtime 结果。
- 外层容器真实重启、目标节点兼容性、部分中断/磁盘耗尽故障场景，以及默认联网构建与可选Docker流程仍待端到端验收。
- 11月11日实施现场未挂载 launcher 默认正式数据集；合成桥接不能替代正式 pooled-data 验收。
- 未重扫全部历史语料；enterprise_zh 是公开年报知识，非内部制度库；CFQA 标注许可不延伸到 PDF 正文。
- T2Ranking 保留上游 train/dev 信息后重新抽样，DCI 结果不等于上游官方测试成绩。正例注入后100%覆盖不代表检索效果。
- 旧159次真实smoke属于旧三档协议，仅保留历史参考；本页不更新 Host 模型评测进度。
- [同步前页面](history/before-frozen-rootfs-20261113/index.html) · [原始历史报告](history/20261028/index.html) · [本次浏览器验证](browser-validation-frozen.json) · [报告摘要清单](manifest.json)
