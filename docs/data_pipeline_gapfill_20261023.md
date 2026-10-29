# 2026-10-23 矩阵补齐：BEIR、MTEB 与原始 QA 构建

新版看板：
`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_matrix_overview_20261023_v2/index.html`

以 `20261023_v1` 的 40 个来源视图为基线，新增 19 个视图，来自 12 个数据集组。
语料规模满足 200/500/1000 且标签就绪的领域×语言格由 **7 增至 22**。
合计 59 个来源视图，其中 28 个三档来源就绪、5 个待复核。
这些计数不是独立文档总数，也不表示三档候选池或模型评测已经完成。

新增就绪格包括：百科 ar/de/en/es/hi/ru/th/tr/vi，医学 en/zh，法律 de/zh，
web zh，通用问答 en。软件 en 已获取完整正文，但因 1 个 query 的等文档关系
待复核，不计入就绪格。Kubernetes 4–5 篇接口样例继续排除在默认覆盖之外。

## 本轮实际来源

| 来源 | 领域 / 语言 | D | Q | 正例关系 | 状态 |
|---|---|---:|---:|---:|---|
| MLQA，7 个同语言视图 | 百科 en/ar/de/es/hi/vi/zh | 每语种 4,509–10,894 | 每语种 4,517–11,590 | 每题 1 条 | 就绪 |
| CmedqaRetrieval | 医学 zh | 100,001 | 3,999 | 7,449 | 就绪 |
| DuRetrieval | web zh | 100,001 | 2,000 | 9,839 | 就绪 |
| GermanDPR | 百科 de | 2,876 | 1,025 | 1,025 | 就绪 |
| LeCaRDv2 | 法律 zh | 3,795 | 159 | 3,896 | 就绪 |
| GerDaLIRSmall | 法律 de | 9,969 | 12,234 | 14,320 | 就绪 |
| SberQuAD | 百科 ru | 9,078 | 5,036 | 5,036 | 就绪 |
| ThaiQA | 百科 th | 2,326 | 74 | 74 | 就绪 |
| TQuAD | 百科 tr | 2,182 | 892 | 892 | 就绪 |
| MedQuAD 原始 XML 回答检索 | 医学 en | 15,817 | 16,407 | 16,407 | 就绪，开发任务 |
| Quora 官方原始视图 | 通用问答 en | 522,931 | 10,000 | 15,675 | 待复核 |
| Quora 完全相同文档别名视图 | 通用问答 en | 522,363 | 10,000 | 15,675 | 就绪，显式派生 |
| CQADupStack Android 完整正文 | 软件 en | 22,998 | 699 | 1,696 | 1 个 query 待复核 |

D 是来源的检索单位：法条/判例、段落、回答、论坛问题等，不能一概称作长文档。
所有新增视图的 corpus 均超过 1,000。Android 有一题保留 262 个原始正例，
200 档不能同时装下这些正例；候选池构建前需要明确该题的处理方式，不能偷偷截断。
ThaiQA 的 74 个 dev query 足够接入，但不足以支持大样本性能结论。

## 自行构建与来源边界

- MLQA、SberQuAD、ThaiQA、TQuAD：用完整同语言 source contexts 构造语料，
  query 仅使用官方 test/dev。qrel 来自题目与来源段落的已知关联。
  MLQA 的非英语问题经过人工翻译，七种语言属于同一数据集组，不能视为七组独立题。
- MedQuAD：从固定 Git revision 的 XML 自建回答检索。原始 47,441 条 QA 中
  16,407 条有已发布回答；31,034 条缺失回答的记录逐条保留在转换审计中。
  没有官方 held-out split，题目含来源模板；这是开发检索任务。
- Quora：原版完整保留。新派生版只合并 title/text/language 逐字相同的文档，
  全部 query 和正例等级不变，原始 ID 通过别名追溯。指标不能直接视为官方 BEIR 指标。
- Android：最初 MTEB 镜像只有标题。最终矩阵采用官方 ZIP 的完整 Android
  corpus/query/qrel 成员，按 Range 获取并校验 CRC/哈希；标题版仅保留在历史研究中。
  完整版中 query 78472 对应两篇正文相同、标题不同的文档，保留待复核状态。
- GermanDPR：封存 spec 的 `web` 归类过宽。官方数据说明确认是维基百科上下文，
  看板归到 `encyclopedia`，同时保留 `source_domain=web` 和修正依据。
- MTEB 镜像数据固定 ModelScope revision，并核验每个文件大小和 SHA-256。
  官方 HF/MTEB 卡与代码用于确定任务来源；没有宣称两个托管站点的字节等价。
  GerDaLIRSmall 实际 14,320 条 qrel 与当前官方 viewer 的 14.3k 一致，
  早先研究的 22,203 不作为 qrel 基准（该数恰好等于 D+Q）。

## 数据、复现与验证

数据继续沿用约定目录：

- 原始数据：`/mnt/afs2/202609/embedding_models/dci-bench/raw_data/<source>/`
- 封存版本：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/<source>/<version>/`
- 本轮研究：`eval_data/source_research_20261023_round2/` 与 `source_research_20261023_round3/`
- 完整集成选择与渲染命令：`source_research_20261023_round3/integration/selection.json`
  和 `render_command.sh`。再次渲染必须改用新的 output 目录。

脚本与分源说明：
[MLQA](mlqa_gapfill_20261023.md)、[MTEB/法律](mteb_gapfill_20261023.md)、
[BEIR](beir_gapfill_20261023.md)、[原生多语言 QA](native_qa_gapfill_20261023.md)、
[MedQuAD](medquad_gapfill_20261023.md)、[Quora 别名](quora_exact_aliases_20261023.md)。

所有新增 release 核验清单与 bundle hash；转换逐题核对原 ID、文本和来源关系，
已有检索任务逐条保留 qrel grade。主线程额外全量核验五个 MTEB 来源的原始文本和标签，
独立 worker 复查 MedQuAD 和 Quora 的派生规则与实际别名。13 个针对性单元检查通过，
覆盖来源分类、覆盖格去重、等文档等级冲突和 Unicode JSONL 往返。
旧看板输出和封存 release 均保留；本轮没有运行 GPU、cascade 或模型评测。

## 保留为空或待处理的来源

- [LLeQA 法语法律](lleqa_gapfill_20261023.md)：官方语料和题目文件返回 401，
  需官方数据使用协议/访问授权；GitHub 只有代码，所查 ModelScope 命名空间返回 404。
  只有元数据落盘，没有伪造 corpus/query/qrels 或计入覆盖。
- FQuAD 法语：旧官方下载链接返回 403；GermanQuAD 独立原库链接返回 404。
  GermanDPR 已能提供德语百科检索来源，但并不表示 GermanQuAD 全库已获取。
- MedQuAD LiveQA judged-answer 包：已保留原始文件，168 个 query–answer 对有不同
  等级的重复判断，未自动裁决后发布。等级 1 表示 Incorrect，不能当作正例。
- 金融、企业、科研等领域的非英语格仍有较多空白，需要继续寻找可追溯的原始资料及
  query–document 关联；仅找到文档的来源应停在 corpus 状态，完成标注/复核后再计入就绪覆盖。
