# 2026-10-22 数据流水线首批接入记录

本批完成三种输入的适配、真实来源下载与清洗，以及本机 KaLM GPU 候选构造。它是数据构建验收，不是 retrieval agent 的正式 benchmark，也不是全领域、多语言平衡覆盖已经完成。

## 已落盘数据

- 原始快照：`/mnt/afs2/202609/embedding_models/dci-bench/raw_data/`
- 清洗版本：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/<source>/v1/`
- 当前看板：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_report_v2/index.html`
- 验证材料：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_validation/20261022_first_build/`

| 来源 | 清洗版本数 | 文档条目数 | Query 实例数 | 状态 |
|---|---:|---:|---:|---|
| BEIR SciFact test | 1 | 5,183 | 300 | labels_ready |
| QASPER dev | 1 | 281 | 465 | labels_ready |
| WixQA expertwritten / simulated | 2 | 各 6,221，共用同一语料 | 各 200 | labels_ready |
| WorkSurface backend / logistics / operations | 3 | 64 / 100 / 226 | 64 / 48 / 66 | labels_ready |
| XQuAD en/ar/de/el/es/hi/ru/th/tr/vi/zh | 11 | 每语言 240 | 每语言 1,190 | labels_ready |
| Kubernetes en/zh/fr/ja/ru | 5 | 5 / 5 / 5 / 5 / 4 | 0 | corpus_ready |

合计 **23 个来源版本、18 个带标签版本、5 个纯语料版本、14,433 条 query 实例**。20,960 是各版本文档条目之和，不能作为去重后的独立文档数。语料有 13 种语言；有标签 query 有 11 种语言。当前领域为 scientific、enterprise、software、encyclopedia。

XQuAD 的 11 个语言版本是 1,190 个问题的平行版本；10 个非英语版本不能算作独立采集的原生问题。来源未显式声明 native/translated 的记录计为 unknown，不因为语言为英语就推断 native。每语言只有 240 篇文档，因此可做严格 200 档，单语言不能直接构造严格 500/1000 档。WorkSurface 的小语料也适用实际规模规则。

QASPER 原始 dev 的 1,005 个问题保留 465 个，其余 540 个带理由保存在排除记录中：annotator disagreement 388、unanswerable 60、unmapped evidence 39、no text 30、figure only 23。采用保守的整题排除策略；281 篇论文全部保留。证据去重后 694 条。WorkSurface operations 保留原始排除规则：35 个只标注同文档等价组中一员的问题明确排除，不补造 qrels。

## 下载与来源固定

- XQuAD 固定 Git commit `7d30520c717524000f0d9d2f9c10a069acd9d285`。原始 raw URL 慢时，使用同一官方仓库的 GitHub Contents API 与 raw Accept header；旧 partial 保留。
- Kubernetes 固定 `bc7f3479bc8b0af48097b155c22c8d58e8e1afe4`，只下载明确的文件 allowlist。
- WixQA 固定 `d662dc42479c14e202eccd832f8c4b66a035c4cc`；WorkSurface 固定 `3b3aef45f1e8f69416d6112c46e6e873cf522315`。从已有原始缓存复制导入新 raw 根目录，旧目录保留。
- SciFact 官方 ZIP、QASPER 官方 v0.3 train/dev 发布包均保存下载文件、哈希和来源清单。没有固定 commit 的官方归档依靠实际下载字节 hash 固定本次输入。
- 全程未使用本机 27897 代理。TyDiQA GoldP 官方对象返回 403；尝试别名和镜像后仍无法取得，catalog 显式标为 blocked，没有虚构下载成功。
- PMC JATS、EUR-Lex、EDGAR 完成适配器与格式测试；本批没有下载这三类的真实全量语料。

## 质量与模型验证

自动检查覆盖三种输入、坏引用、证据 offset、重复文档、缺失正例、模型输出 schema、token/call 预算、缓存、下载中断、归档路径逃逸、symlink、来源错配、不可覆盖发布，以及导出到既定候选池契约。

最终相关检查 **122 passed，9 subtests passed（18.47 秒）**，完整命令、退出码与输出见验证目录的 `test_execution.json`、`tests.log`。`pipeline_report_v1` 保留作历史结果，但其 native/translated 计数错误；使用 v2。v2 明确统计为 native 1,190、translated 11,900、unknown 1,343。另逐组验证了 XQuAD 的 1,190 个问题在 11 个语言版本中始终落在同一 split。

跨来源切分将共用正文、文章族、显式翻译关系和正例关联放在同一连通组，保留每个来源的原始 ID。此切分是合并数据时的辅助防泄漏方案，不改变各来源记录的原始 train/dev/test 标识，也不保证小语料具有精确的切分比例。

KaLM 使用已有 `retriever` 环境：Transformers 4.48.0、sentence-transformers 5.1.0、PyTorch 2.4.0。实际权重身份为 `59c9624a274740f8cf9d7aa195d6d599b4e3fabf8e47106be112ebca1ed6d2e0`。已有 `dci-bench` 环境中的 Transformers 5 无法加载这份旧 custom Qwen2 code，没有修改原权重或该环境。

真实 GPU 排名覆盖 SciFact 的全部 **5,183 篇文档和 300 个 query**，batch size 8、max length 8,192 tokens、topK 1,000，耗时 **409.48 秒**，PyTorch 峰值分配 **4,909,585,920 bytes（约 4.57 GiB）**。本批文档和 query 都没有发生 token 截断。结果保存为 `eval_data/pipeline_scifact_rankings_v1.json`，包含实际模型分数、权重/tokenizer/config 身份、来源文件清单身份及编码审计。

三档候选池已生成到 `eval_data/pipeline_pools_v1/{200,500,1000}/`，每档均通过 `verify_document_pool_suite`，并独立核对每个候选的保存分数等于原始 KaLM 输出。所有 339 个正例关系都保留；正例等级不参与相似度打分。

| 档位 | Query 数 | 每 query 候选数 | 核对分数数 | 补入漏召回正例数 | 共享正文文件数 |
|---|---:|---:|---:|---:|---:|
| 200 | 300 | 200 | 60,000 | 3 | 5,106 |
| 500 | 300 | 500 | 150,000 | 1 | 5,158 |
| 1000 | 300 | 1000 | 300,000 | 0 | 5,181 |

使用 seed `20261021`，每次补正例等量随机移除非正例，分数按真实模型结果排序。每档共享正文保存一次，没有为 300 个 query 分别复制正文。各档构建与完整验证分别约 105、100、98 秒，主要涉及共享文件系统文件读写。

本机另用 MiniCPM5-2B 做了真实 LLM **诊断**：vLLM 0.19.1、TP1、8,192 上下文、单并发。它不构成未修改生产流程的端到端成功：模型输出出现重复 JSON key、schema 不符、引用换行不匹配，judge 返回空列表；尝试 wire JSON schema 时还触发 xgrammar 对 `minItems` 的错误。诊断脚本曾在引用校验时变换原文，因此其“通过”产物未被接受为生产验证，也没有发布其自动标签。

最终使用原始 section 与保存的真实响应进行无网络、无 GPU 重放，生产 `generate_queries` 正确拒绝不匹配的引用；生产 `judge_queries` 将两个漏判候选保留为 `review_required/candidate_omitted`，没有产生默认负例。客户端已补充顶层、嵌套重复 JSON key 的拒绝，并通过回归检查。生产流程仍使用 JSON object 加本地 schema 验证，没有把上述 wire schema 临时补丁合入。

诊断共 8 个 chat POST（7 个 HTTP 成功、1 个失败），超过预定最多 6 个请求 2 次；这是本次执行偏差，日志保留在 `llm_diagnostic/failure_and_boundary_summary.json`，不能用最终成功响应的 usage 代表总成本。所有本次创建的模型服务已停止。正式 27B/24B 生成和跨模型复核尚未执行。

正式生成/复核配置为 Qwen3.8-27B + Mistral Small 3.2 24B，KaLM 负责正式池，BGE-M3 + BM25 可参与查漏。Qwen 权重本地已有；Mistral 权重尚未下载。当前可见一个 CUDA 分区，总显存约 39.5 GiB；4 × H100、两个 TP=2 服务是部署方案，未做四卡吞吐验证。

具体命令和接口见 [数据流水线说明](data_pipeline.md)。所有机器生成内容保留 automatic-label provenance；通过 schema/证据检查不等于人工金标。
