# 2026-10-23 检索来源扩充

本次把来源矩阵的主口径调整为 `query + corpus + qrels`，优先原始 corpus 大于 1000，分别记录 200/500/1000 的规模条件、标签状态和实际执行状态。evidence 为可选辅助信息，不再因缺少 span 而把原生检索任务视为不完整。agent 工具读取轨迹与数据集 evidence 是两个不同层面。

新看板：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_matrix_overview_20261023_v1/index.html`。
旧 `pipeline_matrix_overview_20261022_v2` 保持原样。新目录同时保存 JSON、CSV、SVG、输入/输出 SHA-256 清单及尚未执行的 cascade 命令计划。

最终统计为 40 个已落盘来源视图，本次增加/扩展 9 个；11 个来源视图满足三档规模且标签就绪，3 个满足规模但待复核，覆盖 7 个已就绪的领域×语言格。另保留 21 个小语料视图和 5 个接口样例。23 个既有 pipeline release 与 9 个选用新 release 均重新核验，8 个 legacy view 沿用冻结索引计数。最终命令与验证收据位于 `source_research_20261023/integration/`。

## 已落盘来源

下表的 D 是来源定义的检索单元，并不全是完整长文档。版本均位于 `eval_data/pipeline_v1/<source>/<version>`。

| 来源 | 语言 / 领域 | D | 保留 Q | 正例关系 | 状态 / 粒度 |
|---|---|---:|---:|---:|---|
| `domain_qasper_dev_fullcorpus` | en / scientific | 1,169 | 465 | 465 | labels_ready；论文全文，888 train + 281 dev |
| `qa_cmrc2018_zh` | zh / encyclopedia | 3,251 | 3,219 | 3,219 | labels_ready；原生中文段落，train+dev corpus / dev queries |
| `qa_jsquad_ja` | ja / encyclopedia | 16,797 | 4,442 | 4,442 | labels_ready；原生日语段落，train+valid corpus / valid queries |
| `multilingual_vihealthqa_vi` | vi / medical | 9,896 | 1,930 | 4,806 | labels_ready；原生医疗回答段落 / test qrels |
| `domain_nfcorpus_test` | en / medical | 3,633 | 323 | 12,334 | review_required；医学文章检索单元 / test qrels |
| `domain_arguana_test` | en / argumentation | 8,674 | 1,401 | 1,401 | review_required；论点文本 / test qrels |
| `mteb_parent_hc3_finance` | en / finance | 8,395 | 100 | 100 | labels_ready；完整父源回答 corpus，沿用旧查询与 qrels |
| `domain_fiqa_test` | en / finance | 57,600 | 647 | 1,700 | labels_ready；金融回答 / test qrels |
| `multilingual_korquad_ko` | ko / encyclopedia | 10,645 | 5,774 | 5,774 | review_required；原生韩语段落 / 全部 public-dev queries |

QASPER、CMRC、JSQuAD、ViHealthQA 版本为 `expansion-v1`。NFCorpus、ArguAna 使用 `expansion-v1-fullqrels-noequiv`；名称中的 `noequiv` 表示不按等价未标注正文自动删题，并非已消除等价正文问题。

HC3 使用 `expansion-v1`，FiQA 使用 `expansion-v3-jsonl`，KorQuAD 使用 `expansion-v2-fullqrels`。看板将领域别名 `financial→finance`、`health→medical`、`encyclopaedic→encyclopedia`、`web_search→web` 统一展示，原值保留在 `source_domain`。

边界与校验：

- QASPER 原有 dev qrels 和 694 条 evidence 与旧版本逐字节相同；465 个 query 和 281 篇论文的内容、ID 不变，`metadata.source_file` 指向新的派生输入。新增 888 篇 train 论文只作 corpus，未增加训练集标签。派生输入仅去除 train 的 `qas` 字段；原始文件不变。
- CMRC 的 200 个无效原始 answer offset 记入 provenance，不猜测修正，也不删除其 query/qrel。检索标签来自原始 question-to-paragraph 关联。两份输入均按固定 Git commit、blob SHA-1 和字节数核验。
- JSQuAD 原文段落无 ID 时使用 split、article/paragraph 位置及正文哈希派生稳定 ID，保留来源位置和原始 query IDs。两份输入按固定 Git blob 核验；test 未使用。
- ViHealthQA 原始 test 有 2,013 个 query、5,019 条正例关系。83 个 query 涉及缺失文档，按已有清洗规则整体排除并写入 audit；raw 保留全部 ID/标签。最终版本并非完整 test。许可尚未明确，不能把它描述为已确认可重新分发。
- NFCorpus 完整保留 323 个 query 和原始 qrel grades；最大每题 475 正例，13 题超过 200，因此完整 query 集不能直接形成保留全部正例的 200 池。112 个 query 仍有未标注等价正文问题。
- ArguAna 原始 1,406 条 qrels 中 5 条指向官方 corpus 不存在的 ID，保留 raw 并明确排除；另有 218 个 query 涉及未标注等价正文，故保留待复核状态。该来源属于 argumentation，不计为 legal 覆盖。
- HC3 完整上游文件通过固定 ModelScope revision、9,889,186 字节和 SHA-256 校验。8,436 个回答记录精确文本去重为 8,395 个检索单元，所有来源行/answer side/index 均写入 crosswalk。旧 415 个文档全部精确匹配，旧 100 个 query IDs、文本及 qrel grades 不变；新增回答不生成标签。详见 [HC3 扩库记录](hc3_parent_20261023.md)。
- FiQA 官方归档 MD5 匹配且 ZIP CRC 完整。原始 57,638 篇文档中 38 篇空/无效，648 个 test query 中 1 个依赖无效文档；全部排除明细保留在 audit。修复的是 `load_bundle` 错将合法 U+2028 分隔符当作 JSONL 换行的问题，原始正文没有改写。先前 4 MiB partial、完整归档及失败 staging 均保留。两项回归测试验证 Unicode 分隔符原样往返，损坏 JSON 仍被拒绝。
- KorQuAD 使用全部 9,681 train + 964 dev 段落，全部 5,774 个 public-dev query/qrel 保留；55 个等价未标注正文问题仍待复核。先前自动排除后的 5,719-query release 保留，未选入最终矩阵。全部原始 answer offset 校验有效；hidden test 未取得也未使用。

这些状态是数据封存与审计状态，不代表检索模型成绩。QA 派生标签是原始来源的种子正例，未标注段落不视为已确认负例。来源视图数也不等于独立语料数量：同源扩展、共享 corpus 和平行译本不能累加为独立覆盖。

## 旧来源与 MTEB 父源

Kubernetes 4–5 文档的白名单只证明 connector 路径可用，归为接口样例并退出默认覆盖。XQuAD 每语言 240 个段落、WorkSurface 各角色小语料仍保留；不混语言、混角色或复制文档凑三档。WixQA 多个 query 视图共享同一 6,221 文档 corpus，不能重复计数。SciFact 原有 5,183 文档来源继续沿用。

六个 MTEB(LLM) 任务并非都能通过恢复上游直接达到三档，详见 [逐项父源审计](mteb_parent_audit_20261023.md)：

- FQuAD 1.0 train+dev 有 5,689 个段落，但 MTEB 使用 `fquad2_test`；必须实际获取后按 `(title, context)` 验证旧文档映射，不能直接当作同一个版本。
- HC3 Finance 上游已实际物化，精确去重后 8,395 个文档，旧 ID crosswalk 已验证；human/ChatGPT 是来源属性，不能替代相关性 grade。父源审计里的待获取状态属于先前阶段，以本次扩库收据为准。
- AILA 同任务 statutes 共 197；Consumer Contracts 全 split unique contracts 至多 158；PublicHealthQA 八语言原始记录合计 888；TwitterHjerne 262 个文档。四者无法自然满足三档，不能混入不同任务扩库。

原有六份 `llm-eval-*` 子集的重新封装不算扩库，未纳入本次新增覆盖。重复 CMRC 派生版本也未重复计入。

## 文件、复现与执行边界

数据位置沿用既定分工：

- 原始来源：`/mnt/afs2/202609/embedding_models/dci-bench/raw_data/`。
- 清洗发布：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/`。
- 来源、URL、revision、许可、下载失败与校验收据：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023/`。

接入脚本：`scripts/acquire_domains_20261023.py`、`scripts/acquire_cmrc2018_20261023.py`、`scripts/acquire_jsquad_20261023.py`、`scripts/acquire_hc3_parent_20261023.py`、`scripts/acquire_multilingual_20261023.py`。新脚本拒绝覆盖既有 release，部分脚本可核验后复用相同版本；重试须使用新版本和新收据目录。FiQA 完整快照的重建配置为 `configs/data_pipeline/fiqa_repaired_20261023.json`。CMRC/JSQuAD 使用各自明确的 split 和可选 evidence 策略，不应把规格直接交给默认 SQuAD build 路径替代专用脚本。

验证既有 release：

```bash
python scripts/data_pipeline.py verify /absolute/path/to/release
python -m unittest discover -s tests -p test_retrieval_matrix.py -v
```

`scripts/render_retrieval_matrix.py` 接受 `--base`、`--releases`、`--research`、`--output`，拒绝覆盖既有看板。发布输入清单见看板 `manifest.json`；来源与收据路径见 `data_matrix.json`。

`cascade_plan.json` 只为本次新增、标签就绪的三档来源列出 export → rank → pools 命令，并封存对应 release 完成文件哈希。每来源独立输出/cache；按全部保留 query 的正例溢出情况选择容量。命令参数已通过实际 CLI parser 核对；batch size 8 来自既有设置，尚未验证这些来源的 GPU 吞吐。

本次没有运行 GPU 排名、构造实际三档池或执行 DCI 模型评测。FQuAD 上游下载入口仍不可用，旧文档精确映射待完成；GermanQuAD、DuRetrieval、MLDR、MIRACL、Mr.TyDi、SciDocs 仅记录候选及网络/大小限制，不能将候选记录计作已落盘数据。
