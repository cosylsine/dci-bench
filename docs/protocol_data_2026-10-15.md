# 当前可评测数据：2026-10-15

统一入口位于：

```text
/mnt/afs2/202609/embedding_models/dci-bench/eval_data/dci_protocol_20261015_v2
```

这是 host 侧数据索引和查询适配包，引用已有冻结语料，不复制大语料，也不改变原始查询、文档 ID、标签和候选范围。Document 与 code 继续使用各自的协议和评分模块。原始下载目录、2026-10-14 冻结包与历史发布索引保持原样。

## 数据范围

| Document 任务 | 候选文档 | 查询 | 正例配对 |
| --- | ---: | ---: | ---: |
| LLMAILAStatutes | 82 | 50 | 217 |
| LLMFQuADRetrieval | 269 | 100 | 100 |
| LLMHC3FinanceRetrieval | 415 | 100 | 100 |
| LLMLegalBenchConsumerContractsQA | 154 | 100 | 100 |
| LLMPublicHealthQA | 172 | 100 | 100 |
| LLMTwitterHjerneRetrieval | 262 | 77 | 262 |
| browsecomp_plus | 99,930 | 825 | 5,023 |
| enterprise_rag | 511,954 | 469 | 740 |
| wixqa_expertwritten | 6,221 | 200 | 258 |
| wixqa_simulated | 6,221 | 200 | 230 |
| worksurface_backend_developer_rag | 64 | 64 | 64 |
| worksurface_logistics_manager_rag | 100 | 48 | 48 |
| worksurface_operations_manager_rag | 226 | 66 | 66 |
| 合计 | 626,070 | 2,399 | 7,308 |

文档数量按任务求和；WixQA 两个任务使用相同候选集合，不能把合计理解成全局去重文档数。

| Code 文件视图 | code2test | comment2context | trace2code | 总查询 | 正例配对 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全量 file_view | 106 | 80 | 101 | 287 | 415 |
| gold_untruncated 子集 | 22 | 6 | 17 | 45 | 53 |

45 条属于 287 条之内。代码数据来自上游发布的文件视图，部分内容被上游截断；45 条子集仅保证正例文件视图没有截断标记，候选 snapshot 仍完整继承。行、函数、功能覆盖标注不可用，评分保持 `null`，不能解释成源码执行覆盖率。

## 清洗与排除规则

以下是对现有冻结数据清洗结果的整合，本次没有重复清洗原始文件：

- BrowseComp：排除 265 篇超过 1 MB 的文档，并整题排除依赖它们的 5 条查询；不裁剪文档，也不删减单条查询中的部分正例。
- EnterpriseRAG：排除 4 个冲突 ID 对应的 8 行文档、30 条无正例查询及依赖冲突文档的 `qst_0413`。
- WorkSurface：只纳入纯 RAG 查询。213 条中排除 35 条存在等价文档副本但标注不完整的查询，保留 178 条和各 persona 的完整候选集合。Product Manager 与 Researcher 无纯 RAG 查询，原始语料保留，不生成空评测任务。
- ARB：保留全量 287 条；另将 242 条含截断正例文件的查询整题排除，得到 45 条子集。所有负例候选继续保留。

`audit.json` 汇总 document 的逐项排除记录，并指向代码子集的 `selection_audit.json`。重复正文不是无条件删除的理由；不合并原始 ID，也不根据答案补造相关性标签。

## 协议消费格式

`document_tasks.json` 为每个任务记录 `suite`、`workspace`、`metadata`、`canonical`、源 revision 与数量。每个源 suite 中：

- `canonical/<task>/corpus/corpus.parquet`：`_id,title,text`。
- `canonical/<task>/queries/queries.parquet`：`_id,text`。
- `canonical/<task>/data/test.parquet`：`query-id,corpus-id,score`。
- `workspaces/<task>/corpus/*.md`：agent 可见文档；文件名可逆映射保存在 host 侧 `doc_id_files.json`。
- `metadata/<task>/`：host 侧 queries、qrels、原始标签、清洗审计、哈希及完成标记。

前 10 个任务引用 `eval_data/dci_expansion_20261014_v1`，WorkSurface 三个任务引用 `eval_data/worksurface/dci_v1_20261014`。旧仓库 `data/metadata` 中的基线副本不是本次索引的数据入口。

Code 的新查询文件分别为 `code/file_view/queries.json` 与 `code/gold_untruncated/queries.json`，每行严格只含现有加载器接受的三个字段：

```json
{"query_id": "原始查询ID", "query": "原始查询文本", "workspace": "/绝对路径/原始代码包/snapshots/<snapshot_id>"}
```

`workspace` 指向含 `corpus/` 的 snapshot 根目录。host 标签的位置保存在 `code_views.json`；使用已有 `dci-code-labels-v1` 文件，冻结 repository、base commit、候选文件哈希与 gold。单题只向 agent 暴露该 snapshot 的 corpus，标签、查询清单、其他 snapshot 及整个准备包都留在 host。

## 使用

在仓库根目录设置本机已有环境：

```bash
DCI_PY=/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python
DCI_DATA=/mnt/afs2/202609/embedding_models/dci-bench
DCI_READY="$DCI_DATA/eval_data/dci_protocol_20261015_v2"
export LD_LIBRARY_PATH=/mnt/afs/250010100/miniconda/envs/dci-bench/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

当前容器需要上面的动态库路径让 Inspect 的 SQLite/ICU 依赖加载环境内的 `libstdc++`；未安装或替换系统库。

快速复核准备包及所有源元数据哈希：

```bash
"$DCI_PY" scripts/prepare_protocol_data.py --output "$DCI_READY" --verify-only
```

需要重新读取全部候选内容与 canonical 文件时：

```bash
"$DCI_PY" scripts/prepare_protocol_data.py --output "$DCI_READY" \
  --verify-only --verify-corpus --io-workers 16
```

`commands.json` 保存每个任务的现有 runner 参数数组。打印一条命令供接入模型服务时使用：

```bash
"$DCI_PY" scripts/prepare_protocol_data.py --output "$DCI_READY" \
  --print-command document/worksurface_backend_developer_rag
"$DCI_PY" scripts/prepare_protocol_data.py --output "$DCI_READY" \
  --print-command code/gold_untruncated/trace2code
```

打印命令不会执行评测。实际评测需追加对应轨道的 `--backend-manifest`，或者使用现有 runner 的模型端点参数；code 必须使用符合 `code-v1` 的生成配置。

45 条代码子集可直接做现有入口的 preflight：

```bash
"$DCI_PY" scripts/run_code_task.py \
  --queries "$DCI_READY/code/gold_untruncated/queries.json" \
  --labels "$DCI_DATA/eval_data_code/agent_retrieval_bench/dci_gold_untruncated_v1_20261014/labels.json" \
  --all-queries --preflight-only
```

此命令还要求运行节点具备现有 Pi/Bubblewrap sandbox 依赖；数据格式验证与运行节点的 sandbox 验证是两个步骤。评测输出使用默认 `results/` 或新的独立结果目录。

重新生成索引时指定新目录，脚本拒绝覆盖既有输出：

```bash
"$DCI_PY" scripts/prepare_protocol_data.py \
  --output "$DCI_DATA/eval_data/dci_protocol_$(date -u +%Y%m%dT%H%M%SZ)"
```

## 验证记录

构建脚本检查源元数据哈希、查询/标签/候选 ID 闭合、清洗子集与原始查询一致性、host 样本构造及 scorer oracle。完成标记固定所有生成文件的 SHA256，`manifest.json` 固定引用的源元数据哈希。

普通构建及 `--verify-only` 不重读全部 corpus，输出明确记录 `source_corpus_bytes_verified: false`。全量源数据复核、现有 loader 验收及测试的本次结果记录在同级 `dci_protocol_20261015_v2_acceptance.json`，并绑定准备包完成标记哈希。Scorer oracle 使用人工理想排序验证接口，不是模型得分。本次不运行模型或 GPU benchmark。

最终索引为 `v2`，其构建还通过原 ARB verifier 的源身份、snapshot 清单与文件树检查，并严格校验 document 的轨道及 schema。早期 `v1` 索引保留为准备过程记录，不作为本次交付入口；两版均未修改被引用的冻结语料。
