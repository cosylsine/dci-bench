# DCI 数据构建流水线

入口是 `scripts/data_pipeline.py`，素材目录与用户指定路径一致：

- 原始快照：`/mnt/afs2/202609/embedding_models/dci-bench/raw_data/<source>/<spec_sha256>/`
- 清洗版本：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/<source>/<release>/`
- 来源配置：`configs/data_pipeline/sources.json`
- 本地模型配置：`configs/data_pipeline/models.json`

原始快照、清洗版本、正式 DCI suite、模型缓存是独立目录。已有产物不覆盖；版本变更使用新的 release 名称。`_COMPLETE.json` 表示本阶段文件和哈希闭合，不代表完成模型评测。

## 三种输入与责任边界

| 输入 | 已有内容 | pipeline 做什么 | LLM 职责 |
|---|---|---|---|
| `retrieval` | corpus、queries、qrels | 保留原始 ID 和标签，验证引用，选 split，审计重复及遗漏正例 | 可选复核与查漏标 |
| `evidence_qa` | 问题、来源文档、人工证据或答案 span | 按来源关系确定正例，严格核对证据原文与 offset | 可选改写、复杂化和查漏标 |
| `corpus` | 原始文档 | 清洗、分组、语言/领域记录；停在 `corpus_ready` | 在明确选定的文档上生成问题和证据，再独立复核 |

connector 就是**来源适配器**：`fetch_source(spec, raw_root)` 固化下载，`load_source(spec, snapshot)` 离线解析到统一 bundle。它不决定评测分数，也不把检索结果变成真值。

统一 bundle 保存 `source/documents/queries/qrels/evidence/judgments/provenance`。每篇文档保存原始 `_id`、`title`、`text`、`language`、`metadata`；证据和标签只留在 host 数据中。DCI 导出复用现有 `build_document_task`，只向 workspace 传正文。

## 当前能力

- `generic_jsonl`、BEIR、WixQA、WorkSurface、QASPER、SQuAD/TyDiQA、GitHub Markdown、PMC JATS、EUR-Lex、EDGAR 适配器。
- 有界下载、直连、重试、断点续传、大小/hash 校验；ZIP/TAR 路径逃逸、重复成员、symlink、压缩炸弹防护。使用独立 urllib opener，不改全局代理，不使用本机 27897。
- 不可覆盖的清洗版本、原始来源关联、父版本关联、完整文件清单与哈希；失败中间目录保留。
- 空文档、坏 ID、重复内容、近重复候选、无效 qrels、证据 offset 和歧义引用审计。近重复是有界的候选审计，不宣称全量语义去重。
- 按共享正例、文章、显式翻译关系和重复内容分组切分，减少同源泄漏；小型语料可能只有少量大连通组，不保证切分比例精确。
- LLM 输出 schema 验证、重复 JSON key 拒绝、精确证据引用、请求缓存、调用/token 预算、重试、长度截断响应拒绝，以及意见不一致 review queue。
- BM25 + 多路 dense 候选合并和检索分歧记录；未知候选保持 unknown。候选召回来源和判断来源分开记录。
- 独立语言视图、覆盖统计及可筛选的离线 HTML 看板。声明语言、脚本提示和人工确认分开；原生与翻译样本不混为同一种覆盖。

PMC、EUR-Lex、EDGAR 接受显式 URL/文件清单，完成解析接口和格式测试；没有隐式全站爬取。GitHub 需要固定 commit 和路径 allowlist。大语料也不隐式全量拉取。

## 日常命令

使用已有 `dci-bench` Python 环境；模型功能额外需要 `numpy`、`sentence-transformers`。本次测试使用独立 `/tmp/dci-pipeline-venv`，没有向用户的 conda 环境安装 pytest。

KaLM 的旧 custom model code 与当前 `dci-bench` 环境中的 Transformers 5 不兼容，真实加载会报 `Qwen2Config.rope_theta` 缺失。**embedding 命令使用已验证的 `retriever` 环境**（Transformers 4.48.0、sentence-transformers 5.1.0）；不要修改原模型代码或覆盖其他任务的环境。可选依赖 `pipeline-models` 限定 Transformers 4 系列。

```bash
PY=/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python
export LD_LIBRARY_PATH=/mnt/afs/250010100/miniconda/envs/dci-bench/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
$PY scripts/data_pipeline.py doctor
$PY scripts/data_pipeline.py catalog
$PY scripts/data_pipeline.py fetch --source beir_scifact_test qasper_dev
$PY scripts/data_pipeline.py build --source beir_scifact_test qasper_dev --release-name v1
```

`fetch`/`build` 要求显式 source IDs。下载不会启动模型。原始 WixQA/WorkSurface 可以从现有缓存复制导入新的 raw 根目录，旧缓存不移动、不删除。

```bash
BASE=/mnt/afs2/202609/embedding_models/dci-bench/eval_data
$PY scripts/data_pipeline.py verify "$BASE/pipeline_v1/beir_scifact_test/v1"
$PY scripts/data_pipeline.py audit "$BASE/pipeline_v1/beir_scifact_test/v1"
$PY scripts/data_pipeline.py export \
  --release "$BASE/pipeline_v1/beir_scifact_test/v1" \
  --output-suite "$BASE/pipeline_dci_v1" --task pipeline_scifact_test
```

export 仅接受 `labels_ready`；自动标签必须经过复核且显式添加 `--allow-automatic-labels`。此开关不把机器标签称作人工标签，也不能跳过 `review_required`。

## 候选池：200 / 500 / 1000 篇文档

53 个来源的四卡 KaLM 批量构建、指令与 Doc ID 契约、冻结启动命令和恢复方式见 [2026-10-24 批次说明](kalm_pool_batch_20261024.md)。

集群实跑后发现的映射复杂度与 GPU 等待 CPU 建池问题已在 [v2 修复与恢复说明](kalm_pool_batch_20261024_v2.md) 中处理；[v3 启动与恢复说明](kalm_pool_batch_20261024_v3.md) 增加跳过内容哈希扫描和 batch size 128，后续切换使用 v3 启动入口。

2026-10-23 起，扩充来源优先选择**原始语料大于 1000 个检索单元**、已有 query/qrels 的任务。矩阵将原库规模、标签状态、候选池生成状态分开记录。小语料仍保留为历史或专项任务，但不进入三档主覆盖；4–5 篇 Kubernetes 白名单仅为 connector 样例，不代表软件领域的正式覆盖。不得复制文档、混入无关来源或混语言来凑档位。

`evidence` 是可选辅助层：文档检索 Recall/MRR/nDCG 使用 query、corpus、qrels，已有检索标注不需要补齐 span evidence。QA 转换与机器生成标签仍须遵守各自的来源映射、精确引用和复核要求。不要把数据证据字段与 agent 的工具读取轨迹混为一谈。

本轮实际落盘来源、父源审计、查询排除和执行边界见 [2026-10-23 扩充记录](data_pipeline_expansion_20261023.md)。检索优先看板由 `scripts/render_retrieval_matrix.py` 生成；既有看板不覆盖。

后续 BEIR、MTEB 多语言和原始 QA 补齐见 [2026-10-23 矩阵补齐记录](data_pipeline_gapfill_20261023.md)。继续扩充的 [50 格记录](data_pipeline_gapfill50_20261023.md) 对应历史收集快照 `eval_data/pipeline_matrix_overview_20261023_v3/`；v2 与 v3 保留。

当前展示入口为 [2026-10-28 数据与评测进展看板](project-progress-20261028/index.html)，附 [harness 流程图](project-progress-20261028/workflow.svg) 和 [简短进展文档](project-progress-20261028/progress.md)。它沿用 99 来源收集矩阵，叠加当前 53 来源 / 159 池、196,088 query 的 catalog，以及 159 次真实 smoke 的结果；收集就绪 50 格与当前构池 49 格分别展示。全量 benchmark 尚未完成。

仅用已有清单和结果重建该日期报告（输出需为新目录）：`python scripts/render_project_progress.py --output /path/to/new-report`。生成器不加载模型、不扫描全库正文；输入摘要与输出校验写入 `manifest.json`。

来源 corpus 可能是段落、摘要、回答或完整文档，报告必须保留这一粒度。回溯 MTEB(LLM) 的父任务时逐项核实真实原库规模与旧 ID/文本映射；父任务本身不足 1000 时不能声称可通过恢复原库解决。扩库产生的新派生任务不替代原始榜单结果。

全正例保留还要求每题正例数不超过 K。超过 200 的 query 即使拥有大型 corpus，也不能直接进入严格 200 池；保留标签并单独报告，不自动删正例。默认矩阵的“三档来源”仅表示语料规模和标签齐备，实际 query 可用性见 `positive_overflow_queries`。

正式候选池保持既定 KaLM 契约。先导出完整 DCI 文档任务，再对该任务实际可见的正文编码，输出绑定来源身份的保存排名：

```bash
EMBED_PY=/mnt/afs/250010100/miniconda/envs/retriever/bin/python
$EMBED_PY scripts/data_pipeline.py rank \
  --source-suite "$BASE/pipeline_dci_v1" --task pipeline_scifact_test \
  --device cuda:0 --batch-size 8 --max-length 8192 --top-k 1000 \
  --cache-dir "$BASE/pipeline_caches/kalm" --output "$BASE/pipeline_scifact_rankings_v1.json"
$PY scripts/data_pipeline.py pools \
  --source-suite "$BASE/pipeline_dci_v1" --task pipeline_scifact_test \
  --rankings "$BASE/pipeline_scifact_rankings_v1.json" --output "$BASE/pipeline_pools_v1"
```

每档先取 topK，补入遗漏正例，用固定 seed 随机移除相同数量非正例，按真实同模型分数排序。所有正例包含真实分数；不能用常数给补入正例伪造分数。分数相同保留 backend 顺序。语料不足 K 时保留实际规模，不复制文档凑数，不把它宣称为严格 K 规模任务。

共享正文按任务保存一次，单 query 保存候选 ID 列表；既定 runtime 负责临时 workspace。原始 ID 与短别名映射、qrels、分数和注入记录留在 host。

## 本地模型选择与 4 × H100 调度

| 职责 | 模型 | 理由和调度 |
|---|---|---|
| 生成 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 已有本地权重；多语言结构化输出；BF16 TP=2，先用 32K 上下文、小并发 |
| 复核 | [Mistral Small 3.2 24B](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) | 暂定跨模型复核器；BF16 TP=2；ModelScope 权重已下载并验证，judge 质量尚待校准 |
| 正式召回 | [KaLM multilingual mini v2.5](https://huggingface.co/KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5) | 保持既定本地路径与 `trust_remote_code=True`，使用模型 query/document prompt |
| 查漏召回 | [BGE-M3](https://huggingface.co/BAAI/bge-m3) + BM25 | 提供另一组候选；不替代正式池的 KaLM |

先独立做 embedding，再分阶段生成和复核。测过显存后可以让两个 TP=2 服务各占两卡。显存预算还包括 KV cache 和运行时开销，不能只按参数量声称吞吐或并发已验证。当前会话只发现一张可见 H100，CUDA 实际可用分区总显存为 39.5 GiB；4 卡方案是部署配置，未完成四卡实测。

2026-10-23：Mistral 权重已按用户授权通过 ModelScope 直连下载到 `/mnt/afs2/202609/embedding_models/dci-bench/pretrained_models/Mistral-Small-3.2-24B-Instruct-2506`，未使用代理。采用原生 `consolidated.safetensors`、`params.json`、`tekken.json` 等 8 个文件，总计 48,042,222,084 bytes；585 个张量全部为 BF16。完整权重 SHA-256 为 `2cc4cc283a4bf3820f77ddb73bef3128b7159057c240b94a72b663d97af86f89`，与 ModelScope 清单和 Hugging Face LFS 元数据一致。下载进程曾在哈希阶段以 143 退出，随后从完整暂存文件恢复离线校验，最终成功。模型目录内 `download_complete.json` 指向带哈希的收据；旧 1,227,718,424-byte partial 保存在 `.modelscope_download/20261022T1534Z/previous_target_files/`，下载分片也保留。此记录证明文件完整性，尚未执行该模型的 GPU 加载或 judge 推理。

本机 GPU 命令前使用用户提供的环境变量：

```bash
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
```

默认仅打印可检查的启动命令；添加 `--execute` 才会启动服务。judge 缺少本地权重时直接报错，不隐式下载几十 GB。

```bash
$PY scripts/serve_data_model.py --role generator --gpus 0,1
$PY scripts/serve_data_model.py --role judge --gpus 2,3
```

生成与复核示例（需先启动对应本地服务）：

```bash
$PY scripts/data_pipeline.py generate \
  --release "$BASE/pipeline_v1/github_kubernetes_website_zh/v1" \
  --output "$BASE/pipeline_v1/github_kubernetes_website_zh/generated-v1" \
  --stage-config configs/data_pipeline/generate_smoke.json \
  --cache-dir "$BASE/pipeline_caches/generator" --max-calls 2 --max-total-tokens 20000
$PY scripts/data_pipeline.py mine-review \
  --release "$BASE/pipeline_v1/github_kubernetes_website_zh/generated-v1" \
  --output "$BASE/kubernetes_zh_review_v1.json" --top-k 5
$PY scripts/data_pipeline.py judge \
  --release "$BASE/pipeline_v1/github_kubernetes_website_zh/generated-v1" \
  --output "$BASE/pipeline_v1/github_kubernetes_website_zh/judged-v1" \
  --stage-config configs/data_pipeline/judge_smoke.json --candidates "$BASE/kubernetes_zh_review_v1.json" \
  --cache-dir "$BASE/pipeline_caches/judge" --max-calls 20 --max-total-tokens 80000
```

`rank --release ...` 输出查漏用的 dense view，可通过 `mine-review --dense-files ...` 合并。与 `rank --source-suite ...` 的正式池排名用途不同，不能互换。BGE 需要显式 `--audit-model --model-path /mnt/afs/share/embedding_models/bge-m3`。

## 数据质量与报告边界

保留原有 query/qrels 的输入不必重新生成问题。QA 改造中的 qrels 是来源支持的 seed positives，不等于对整库做完穷尽标注；对其他候选不赋予“已确认负例”的含义。机器扩充标签另记 provenance，冲突进入 review queue，不覆盖原始快照。

原生多语与翻译语料分开计数；TyDiQA-GoldP 改造为 passage retrieval，不能宣称复现其原始阅读理解榜单。按语言拆分是明确的派生视图，不应与原始混合语言版本同时累计规模。

```bash
$PY scripts/data_pipeline.py report --releases "$BASE/pipeline_v1/"*/v1 \
  --output "$BASE/pipeline_report_v2"
```

看板输出 `index.html`、`coverage.json` 和 `collection_splits.json`。合并来源时应使用 collection 级切分，让 WixQA 共用正文、XQuAD 平行译本等关系跨 source 生效；单个 release 内的 `splits.json` 只保护该来源内部。真实数据接入结果、测试数量和未完成模型阶段见 [2026-10-22 运行记录](data_pipeline_run_20261022.md)；下载完成、数据可评测、LLM smoke、正式 benchmark 是不同的验证边界。重复生成报告请使用新的输出目录。

### 领域 × 语言状态矩阵

`scripts/render_data_matrix.py` 读取已发布版本，逐项校验文件清单和哈希，输出 HTML、PNG、SVG、CSV、JSON。每格区分仅语料（C）、query/qrels（Q+R）、另有原文证据（Q+R+E），并报告证据覆盖的 query 数及待复核数。当前快照是 `eval_data/pipeline_matrix_20261022_v3/`。

包含旧冻结 document 任务的总览为 `eval_data/pipeline_matrix_overview_20261022_v2/`：31 个来源视图（23 个 pipeline + 8 个旧任务，不重复累计 WixQA/WorkSurface）、16,254 个 query 实例。P/L 区分当前 pipeline 与旧冻结协议，T 标记译本；星号表示只有部分 query 有显式证据。旧数据的领域与语言依据保留在 JSON/HTML；部分英语标签来自任务说明和样本推断，BrowseComp+ 的 corpus 语言未声明。旧数据只复核索引和元数据，没有为画图重新扫描全部 51 万篇企业文档。图中的 D 是来源语料量，不是每 query 的候选档位。

```bash
EMBED_PY=/mnt/afs/250010100/miniconda/envs/retriever/bin/python
$EMBED_PY scripts/render_data_matrix.py --eval-root "$BASE" \
  --output "$BASE/pipeline_matrix_next" \
  --override worksurface_backend_developer=evidence-v2 \
  --override worksurface_logistics_manager=evidence-v2 \
  --override worksurface_operations_manager=evidence-v2
$EMBED_PY scripts/render_data_overview.py \
  --pipeline-json "$BASE/pipeline_matrix_20261022_v3/matrix.json" \
  --legacy-index "$BASE/dci_protocol_20261015_v2/document_tasks.json" \
  --output "$BASE/pipeline_matrix_overview_next"
```

2026-10-22 矩阵核对发现 WorkSurface connector 未将原生 `span` 映射到 `quote`，旧 v1 的 178 条证据因此全部被剔除。现已修复，并发布三个独立 `evidence-v2`，正文、queries、qrels 与 v1 逐字节相同：backend 64 条、operations 66 条、logistics 46 条证据可以精确定位；logistics 另 2 条因重复出现且没有消歧上下文而进入 `review_queue`，该 release 为 `review_required`。清洗器会保留被拒收的证据及其原始标签供复核，重新加载后审计也保持待复核状态。旧 v1 和首批验证归档不修改；修复验证记录位于 `eval_data/pipeline_validation/20261022_evidence_fix/`，重新审计回归记录位于 `20261023_review_replay/`，后者 88 项检查全部通过。采用修复版本的覆盖看板为 `eval_data/pipeline_report_v3/`。
