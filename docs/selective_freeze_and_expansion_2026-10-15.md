# selective 冻结与数据扩展：2026-10-15

现有 `budgeted-selective` 已作为 **selective-document-v1** 采用。它保持原契约 SHA256 `c6f4a9e46de4f0ebf9b21aca6f73c86a7b1dfd726783362bc3c476807be6419f`，没有修改旧全量冻结包、原始数据或历史结果。

## 冻结依据和边界

- 120 对 confirmation 的预注册标准全部通过：F1 增加 3.44 个百分点，95% paired bootstrap 区间为 +0.65～+6.48 个百分点；Recall 区间满足非劣界限；tokens 比为 1.01616；两臂各 120/120 有效。
- 随后的六任务全量 selective 运行 527/527 有效；task-macro Recall/F1/nDCG@10 为 71.85/55.04/71.07%。完整性和 7,905 个重算指标均已复核。
- 这是采用下一轮文档 benchmark 协议的证据，不表示最优，也不能将端到端差值全部归因于 final prompt。confirmation 的多条搜索轨迹在 final 前已分叉；原六任务全量包含既有开发查询。

冻结完整 prompts/controller、128k 累计 token 预算及 reserve、输出/steps/time 上限、stable-rg、read/ID/sandbox 规则、生成设置和评分定义。至少一个非空 read 的规则保持原样；它不保证所有返回文档分别有 read receipt。ordinary repair 与 tools-disabled final repair 保留各自行为。

协议采用记录：`releases/protocol-freezes/20261015-selective-document-v1/release.json`。底层原始执行快照仍是 `releases/protocol-freezes/20261015-selective-full-v1/runtime`。将来任何影响模型行为或评分的调整都另开版本，保留 v1 和既有结果。模型/权重与实际服务环境仍通过每次运行的 backend manifest 单独固定。

## 数据已准备好，无需重新下载或清洗

统一数据入口：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/dci_protocol_20261015_v2`。

| 新增文档任务 | 全部查询 | Pilot | 非 pilot 查询 | 候选文档 |
|---|---:|---:|---:|---:|
| browsecomp_plus | 825 | 10 | 815 | 99,930 |
| enterprise_rag | 469 | 10 | 459 | 511,954 |
| wixqa_expertwritten | 200 | 10 | 190 | 6,221 |
| wixqa_simulated | 200 | 10 | 190 | 6,221 |
| worksurface_backend_developer_rag | 64 | 10 | 54 | 64 |
| worksurface_logistics_manager_rag | 48 | 10 | 38 | 100 |
| worksurface_operations_manager_rag | 66 | 10 | 56 | 226 |
| **合计** | **1,872** | **70** | **1,802** | — |

加上已有六任务的 527 条，共 **13 个文档任务、2,399 条查询**。候选数量为逐任务计数，两个 WixQA 任务共用语料。Code 的 ARB 287 条和其中 45 条子集另走 code 协议，不加进本次文档统计。

## 固定评测顺序

1. 完成源代码、数据、contract 与真实 sandbox 的 CPU preflight。大语料的逐文件元数据扫描成本单列，不能把 CPU cap 超时当作 600 秒样本协议失败。
2. 运行 **7 条 endpoint/tool smoke**，每个新任务一条，先小语料再大语料。没有真实模型/工具验收前，不声称评测入口已端到端通过。
3. 运行 **70 条 pilot**，每个新任务十条。查询只按固定 seed、task、原始 query ID 的哈希选择，不使用 qrels、历史分数或模型输出筛选；保留完整候选语料。7 条 smoke 属于这 70 条，不相加。
4. 核对有效率、错误分类、repair/read/finalization 行为、每查询 token 与启动/检索耗时，再由用户启动 **1,802 条非 pilot 查询**。若选择重跑全部新增 1,872 条，单独记录为 full_new，明确其包含 pilot，不能重复计数。

非 pilot 部分只表示未纳入本轮 pilot，不宣称它在历史训练或其他评测中未被使用，也未据此断言跨任务没有近重复。冻结后不根据 pilot 的低分筛掉任务或改标签；若需要改 prompts/controller，则另开 v2 并保留旧结果。

正式报告分别给出原六任务、新七任务、合并十三任务的 all-sample 等任务平均和 query-weighted 平均，并附 valid_only、Recall/F1/actual-window Precision/nDCG@10、tokens/latency、有效率和 trace。WixQA 共用语料、WorkSurface 有文件名提示等任务特点应单列，避免只给一个总分。

## 扩展运行兼容性

WixQA 的原始 query ID 如 `expertwritten/0` 含斜杠。旧 runner 能加载它，却把原始 ID 直接当结果目录，发布/完成校验时会失败。新增 `sample_directory_name()` 只编码落盘目录；原始 query ID、qrels、Inspect metadata、查询文本和排序文档 ID 保持不变。普通 ID 保留旧路径，保留前缀也会被转义以避免碰撞。

独立扩展执行快照位于 `releases/expansion-runtimes/20261015-selective-document-v1`，其 `adapter.json` 和 `host-artifact-paths.patch` 固定三个 Python 文件的变更。Pi 和原 selective 契约继续引用不可变的原始快照。此项为 host 结果路径适配，不重新定义 selective 的检索行为。

49 项相关单元/回归测试通过，包括四类特殊 ID 的完整结果发布及恢复读取、旧目录兼容、数据分区闭合、已有输出拒绝覆盖及禁止替换 runtime 后改动冻结契约。验证没有调用模型。另核对全部 2,399 个查询的目录映射无冲突，原六任务 527 个目录名完全保持原样。

## 运行准备文件和检查

采用当前计划 **`results/expansion-plans/20261015-selective-document-v2`**。其中 `plan.json` 记录范围和 gates，`query-splits.json` 固定查询，`task-contexts.json` 固定新任务原有的空 instruction，`commands.json` 保存逐任务 argv 模板，`verification.json` 保存实际 CPU 检查结果。

同名 v1 计划保留为首次准备记录：当时系统缺少 bubblewrap/ripgrep，小任务失败；两个大任务触及 60 秒 preflight cap。不得将该记录解释为模型能力或样本协议失败。当前容器后来仅补齐了 `bubblewrap=0.4.0-1ubuntu4.1` 和 `ripgrep=11.0.2-1ubuntu0.1`，合计下载约 1.2 MB，随后重查。

**当前不是 preflight-ready。** 重查中六项真实 sandbox 通过：三个 WorkSurface 和两个 WixQA 约 11～12 秒，BrowseComp+ 约 96.17 秒；EnterpriseRAG 超过 180 秒 CPU preflight cap，尚未证明其 sandbox 完整通过。返回状态为 `verified_with_preflight_pending`，只表示计划和已记录锚点通过验证。不得据此直接启动七任务 GPU pilot。

下一项应先测清 EnterpriseRAG 的逐样本元数据遍历，并评估在保持隔离校验与完整候选语料的前提下能否减少重复开销。新语料远大于原六任务，不能沿用原任务平均 token/latency 来估算全量成本。正式运行前还需要当前 corpus 内容哈希验收或可复核的冻结语料完整性证据；本轮没有重读全部候选正文。

只读核对计划与数据/源码锚点：

```bash
/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python -B \
  scripts/prepare_selective_expansion.py \
  --output results/expansion-plans/20261015-selective-document-v2 --verify-only
```

打印 smoke 命令，检查后在已有服务的运行节点执行：

```bash
/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python -B \
  scripts/print_selective_expansion_commands.py \
  --plan results/expansion-plans/20261015-selective-document-v2 \
  --stage smoke --backend-manifest "$BACKEND" --results-root "$OUT"
```

`BACKEND` 必须是当前实际服务的 manifest；`OUT` 必须是新的结果根目录。该命令只打印，既不启动服务器，也不执行推理。实际运行前，应核对其模型/权重、生成配置、上下文长度与参考后端；本轮尚未做 endpoint/tool smoke 或 GPU pilot。

普通数据校验不重读全部语料正文。此前全量语料校验记录在数据索引旁的 `dci_protocol_20261015_v2_acceptance.json`，本轮只复核其锚点并检查 CPU 接入，不能把这些记录混称为新的模型评测。

扩展执行副本的额外只读核对命令：

```bash
/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python -B \
  releases/expansion-runtimes/20261015-selective-document-v1/verify_snapshot.py \
  --data-index /mnt/afs2/202609/embedding_models/dci-bench/eval_data/dci_protocol_20261015_v2
```

它检查 45 份源码/结果 schema 文件、Pi symlink 目标、相同 selective 契约及 2,399 条原始查询的目录映射；不读取 corpus 正文，也不调用模型。
