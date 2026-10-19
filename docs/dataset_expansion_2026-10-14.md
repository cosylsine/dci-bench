# DCI-Bench 数据扩展：2026-10-14 核验与首批接入范围

本次已重新核验截图中的 14 个 test-time compute 候选，完成原始数据下载、标注审计、协议适配与数据冻结。通用 document 版包含 **10 个任务、2,221 条查询**；code 版包含 **ARB 三个子集、287 条查询**，另提供 **45 条所有正例文件均未截断的子集**。本次交付评测数据与 host 评分输入，没有运行模型。

通用版在原有 6 个任务、527 题基础上，新增 WixQA 两组、BrowseComp-Plus 与 EnterpriseRAG-Bench。代码版明确命名为上游发布的文件视图派生版：部分文本已被上游截断，不能称为完整源码仓库评测；文件级标注可用，行号、函数与功能覆盖标注均保持不可用。

WorkSurface 因 mirror TLS 与重定向后的连接反复失败延期，没有计入本次发布。其他候选的接入或延期依据见下文，2026-10-12 的调查报告保持为历史记录。

## 1. 接入约束

- 文档轨道输出 `ranked_doc_ids`，1–10 个唯一原始文档 ID，不补无关文档。代码轨道沿用现有 `ranked_files` / `abstain` 协议，最多 20 个文件；两者的输出数量、空结果语义与评分契约不能混用。
- queries、answers、qrels、判分标签、生成过程、gold 路径列表均只存 host。agent 只能读取该题合法候选语料。原始包中语料与标注混放时，不能直接将 raw 根目录挂载给 agent。
- 保留原始文档 ID、任务 ID、split、revision 和候选范围。没有原始 query ID 的数据，使用固定原始文件与行序号建立可逆定位，明确其为派生 ID。
- 文档转换为 Markdown，每篇不超过 1 MB。超限不得静默截断、丢弃其正例或擅自把文件标签扩成任意 chunk 标签；必须记录处理策略和影响。
- 有超过输出上限的正例时，保留完整 qrels，另报正例数分布与 recall 上限。只取前 10/20 个正例会改变基准。
- query-specific scope 必须保留：WorkSurface 按 persona；LoCoMo 按 conversation；LongMemEval-V2 按 haystack。不可把不同场景合成对每题全部可见的候选库。
- 静态文件获取、结构检查、转换审计、模型 smoke 和完整评测是不同验证边界。本次不以下载成功推断模型可用，更不报告尚未执行的 benchmark 成绩。

## 2. 全部 14 个候选的决定

### 通用 document 候选（12 个）

| 截图名称 | 本次决定 | 证据与适配依据 | 主要限制 / host-only 内容 | 官方来源 |
|---|---|---|---|---|
| LongMemEval-V2 | 延期 | 冻结 trajectories 与 small/medium haystacks 可获取 | 发布版明确移除 answer-bearing labels；haystack 是候选域，不是正例；另有截图依赖 | [官方 HF](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2) |
| BrowseComp-Plus | **首批** | 完整下载 100,195 篇原始文档；转换保留 99,930 篇；官方 evidence qrels | 超 1 MB 的文档及依赖它们的整题有记录地排除；答案、evidence/golds 与 query 元数据 host-only；保留 >10 positives | [HF corpus](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus) |
| MADOQA（截图拼写） | 延期，身份待确认 | 最接近已核验数据为 MADQA：页面级证据 | 不能直接断言二者同一数据；MADQA gated，PDF 在外部 DocumentCloud，test evidence 隐藏 | [MADQA HF](https://huggingface.co/datasets/OxRML/MADQA) |
| EnterpriseRAG-Bench | **首批：非空 gold 子集** | `expected_doc_ids` → corpus `doc_id` | 无 gold 的 High Level / Info Not Found 不可伪造正例；过滤需有逐题清单 | [官方 HF](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench) |
| Bamboogle | 延期 | 官方 QA 发布入口与 MIT 可核验 | 未确认官方冻结证据 corpus/qrels；Google Sheet 本次读取失败，第三方 QA 镜像不能补齐证据 | [官方入口](https://github.com/ofirpress/self-ask/blob/559e56ec1ba8dc93cf63c4148ec1f4717caf4b60/datasets/bamboogle.md) |
| OfficeQA Pro | 延期 | `source_files` 可映射 Treasury 文件 basename | 官方 gated；Markdown 约 460 MB，697 文档，单篇 >1 MB 与访问状态待审计；不能只用支持文件构库 | [官方 HF](https://huggingface.co/datasets/databricks/officeqa) |
| Context-Bench（Letta） | 可适配，首批延期 | 100 QA 有 `required_files`，10 个冻结文本文件 | 文件级 qrels 可用，但原难点是文件内部多跳推理；只返回文件 ID 会显著改变难度 | [官方文件](https://github.com/letta-ai/letta-evals/tree/688e099a70295a281ea3667398fb53b4d7af89d4/letta-leaderboard/filesystem-agent) |
| τ-Knowledge（terminal） | 可适配派生版，首批延期 | 97 tasks 有 `required_documents`，698 篇冻结 JSON 文档 | 查询是多轮用户模拟场景；静态化需要独立命名；39 题 >10 positives；DB、gold 动作 host-only | [官方任务](https://github.com/sierra-research/tau2-bench/blob/dd8757d4c8791dad7b3cdaf02a105a3d5f9d86bd/data/tau2/domains/banking_knowledge/tasks.json) |
| HippoCamp | 可适配 gold-text 派生版，首批延期 | `file_path` 与局部 `evidence`；公开 parsed Gold | 多模态原始设定不等于 gold-text；Gold `QAID/QANum` 会泄漏；许可限制公开镜像；1 MB/映射覆盖待审计 | [官方 HF](https://huggingface.co/datasets/MMMem-org/HippoCamp) |
| LoCoMo | 可适配，首批延期 | turn `dia_id` 与 QA `evidence` 可精确连接 | 9 题坏/未解析证据、4 题空证据；adversarial 不等于常规正例；必须 conversation scope | [官方数据](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json) |
| WixQA | **首批：expertwritten + simulated** | `article_ids` → KB `id`；使用完整冻结 KB | 无原始 qid；各 QA split 清楚分离，不能将问答/链接标注混进 KB | [官方 HF](https://huggingface.co/datasets/Wix/WixQA) |
| WorkSurface-Bench | **V1 延期：下载阻塞** | 后续可用 `gold_evidence.file` → persona `kb_docs/*.md` | mirror TLS/重定向连接反复失败；未计入本次 V1。后续仍只考虑 `rag_only`、五个 persona 候选域 | [官方 HF](https://huggingface.co/datasets/lhpku20010120/WorkSurface-Bench) |

### Code 候选（2 个）

| 截图名称 | 本次决定 | 证据与适配依据 | 主要限制 / host-only 内容 | 官方来源 |
|---|---|---|---|---|
| LongBench v2（CodeQA） | 延期 | 问答和长代码上下文可公开核验 | 没有可直接还原的文件/函数/行级 qrels；不能把整段 context 或答案字符串命中伪装成文件证据 | [官方 HF](https://huggingface.co/datasets/zai-org/LongBench-v2) |
| Agent Retrieval Bench | **首批：三个 v2 子集** | 287 条问题及上游文件/片段证据；保留 frozen file-chunk view | chunk 可能截断或空白归一化；没有原仓库源文件时不能编造行号、函数标签或声称完整仓库检索 | [官方 HF](https://huggingface.co/datasets/eyuansu71/agent_retrieval_bench) |

## 3. 固定来源、许可与已核验规模

下表区分本次已发布来源与后续候选的上游规模。未获取的数据不计入发布数量；未核验的 revision 不以 `main` 冒充冻结版本。

| 候选 / 组件 | 仓库与 revision | 许可 | 上游规模 / 核验边界 |
|---|---|---|---|
| WixQA | `Wix/WixQA` @ `d662dc42479c14e202eccd832f8c4b66a035c4cc` | MIT | 转换完成：每任务 6,221 篇；expertwritten 200 题 / 258 qrels，simulated 200 题 / 230 qrels；最大 Markdown 53,032 bytes |
| BrowseComp-Plus corpus | `Tevatron/browsecomp-plus-corpus` @ `b27b02bc3e45511b8b82a13e6f90ce761df726f6` | MIT | 原始 100,195 篇，约 1.76 GB；转换完成 99,930 篇，排除 265 篇超限文档；最大 Markdown 998,460 bytes |
| BrowseComp-Plus queries | `Tevatron/browsecomp-plus` @ `144cff8e35b5eaef7e526346aa60774a9deb941f`；[官方 GitHub](https://github.com/texttron/BrowseComp-Plus) @ `046949032b0328319cc9a02663a759ec601d9402` | MIT（上游标注） | HF query 六个 shards 全部下载；投影并解码 `query_id/query/answer`。已审计 830 题、5,064 evidence、2,407 answer-gold；Git tree 无 `queries.tsv` |
| EnterpriseRAG-Bench | `onyx-dot-app/EnterpriseRAG-Bench` @ `69916e31c68aa5963c00248fd7f0bc12d04fd235` | MIT | 已审计 511,962 行/511,958 唯一 ID；4 个冲突 ID 涉及 8 行、0 个完全重复行；保留 511,954 文档，排除 31 题后 469 题/740 qrels；已完成物化；最大 Markdown 42,377 bytes |
| WorkSurface-Bench | `lhpku20010120/WorkSurface-Bench` @ `3b3aef45f1e8f69416d6112c46e6e873cf522315` | CC BY 4.0 | 1,151 总题，213 `rag_only`；五个 persona 的 resources 约 21.7 MB；待逐题核对 |
| ARB 三个 v2 子集 | `eyuansu71/agent_retrieval_bench` @ `5901e1ee3aff048290db72edf9c63bc498b79ea3` | metadata/evaluator MIT；corpus 依各上游源代码许可 | 三个 archive 均已下载，106/80/101 题，共 287；已冻结 218 个唯一 snapshots、312,749 个原生文件视图；另有 45 题正例文件未截断子集 |
| LongBench v2（CodeQA） | `zai-org/LongBench-v2` @ `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9` | Apache-2.0 | `data.json` 约 465 MB；Code Repository Understanding / Code repo QA 为 50 题；无文件/commit/evidence 标签 |
| MADQA | `OxRML/MADQA`，revision 待授权获取时冻结 | CC BY-NC 4.0 | train 1,550、dev 200；test 500 的 evidence 隐藏；原始 PDF 外部托管 |
| OfficeQA Pro | `databricks/officeqa`，revision 待授权获取时冻结 | CC BY-SA 4.0 | `officeqa_pro.csv` 133 QA；697 Treasury 文档，文本约 460 MB |
| LongMemEval-V2 | `xiaowu0162/longmemeval-v2` @ `f152293e235517d504809563c833d7190b8c713b` | Apache-2.0 | metadata 实查：`questions.jsonl` 286,186 bytes；`trajectories.jsonl` 1,195,604,539 bytes；451 QA / 1,870 trajectories 为官方规模 |
| Bamboogle | `ofirpress/self-ask` @ `559e56ec1ba8dc93cf63c4148ec1f4717caf4b60` | MIT | 官方 `datasets/bamboogle.md` 195 bytes，指向 Google Sheet；本次未读取完整表，不报告导出行数/大小 |
| Context-Bench | `letta-ai/letta-evals` @ `688e099a70295a281ea3667398fb53b4d7af89d4` | Apache-2.0 | 小文件实查：100 QA；10 文本共 1,304,644 bytes；单文件最大 302,138 bytes |
| τ-Knowledge | `sierra-research/tau2-bench` @ `dd8757d4c8791dad7b3cdaf02a105a3d5f9d86bd` | MIT | 实查 tasks + 文件树：97 tasks；698 文档共 932,122 bytes，最大 8,148 bytes |
| HippoCamp | `MMMem-org/HippoCamp` @ `ff212ff7ec1a60d28023e65f7e7c4df5a87f552c` | HippoCamp Research License v1.0 | fullsets 581 QA；HF Gold 目录约 528 MB，完整 repo 80.3 GB；没有下载或逐文件审计 |
| LoCoMo | `snap-research/locomo` @ `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` | CC BY-NC 4.0 | 源 JSON 实查：2,805,274 bytes，10 对话、5,882 turns、1,986 QA |

## 4. 首批原始文件与映射

### WixQA

选择 `wixqa_expertwritten/test.jsonl` 与 `wixqa_simulated/test.jsonl`，完整 KB 为 `wix_kb_corpus/wix_kb_corpus.jsonl`。按原文件名记录 split，不依赖 HF 自动推断的 `train`。qrels 只由源 `article_ids` 精确映射到 corpus `id`；不能从答案文本重新猜相关性。无原始 query ID 时，保留 source file、split、原始行号和派生 query ID 的映射。转换已完成：每任务 6,221 篇，expertwritten 200 题 / 258 qrels，simulated 200 题 / 230 qrels；最大 Markdown 53,032 bytes。精确文本审计未发现重复内容组。已通过全量文件校验并纳入完成的通用 suite。[数据卡](https://huggingface.co/datasets/Wix/WixQA)

### BrowseComp-Plus

原始语料完整获取 100,195 篇，字段包括 `docid,text,url`。**官方 Git tree 没有 `queries.tsv`；它是官方解密脚本的生成文件，不能作为已存在的下载入口。** 当前 adapter 从固定 HF query revision 的六个 parquet shards 投影 `query_id/query/answer`，使用官方 GitHub revision `046949032b0328319cc9a02663a759ec601d9402` 对应的 XOR + SHA-256 canary 解码方法，不将嵌套答案文档放进 agent 语料。[官方 GitHub](https://github.com/texttron/BrowseComp-Plus/tree/046949032b0328319cc9a02663a759ec601d9402)

`topics-qrels/qrel_evidence.txt` 为主 qrels，`qrel_golds.txt` 为单独的 answer-gold 诊断；不混合两者。实际审计 830 题、5,064 条 evidence、2,407 条 answer-gold，17 题有 >10 positives、最大 13，ID 失配为 0。小型官方 qrels 的 SHA-256 已保存在 `topics/source_manifest.json`：evidence `a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1`（76,162 bytes）；golds `b875af4a745712bee7a94f464ed989232f8c77977c31824428470e11dcb28c73`（36,336 bytes）。全部六个 HF query shards 已下载，原始数据完整保留。[Corpus](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus)

转换已完成，保留 **99,930 篇文档、825 题、5,023 条 qrels**。265 篇文档全部因 `document_exceeds_1MB` 排除；依赖这些文档的 `815`、`216`、`324`、`1079`、`442` 五题整题排除，其中 `442` 涉及两篇超限正例。没有只删正例再保留问题。Markdown 总计 2,781,947,112 bytes，单篇最大 998,460 bytes；精确文本审计未发现重复内容组。保留的题中仍有 17 题超过 10 个正例，最大 13 个，完整 qrels 保留；输出上限为 10 时，逐题平均 recall 理论上限为 0.9975898848626121，F1 理论上限为 0.9987084851906985。这些是标签与输出约束推导的上限，不是模型成绩。

### EnterpriseRAG-Bench

来源 `data/documents/test.parquet` 与 `data/questions/test.parquet`；`expected_doc_ids` 精确连接 `doc_id`。实际审计 511,962 个源行、511,958 个唯一 ID；4 个 ID 对应冲突文本，共 8 行，完全重复行数为 0。为保留 ID 语义，排除这 4 个 ID 的全部 8 行，不挑任意版本，保留 511,954 篇。整题排除依赖冲突 ID 的 `qst_0413`，另排除 30 个空 gold 问题，得到 469 题/740 qrels；不得只剪掉该题冲突正例。所有排除记录保留原 ID 与原因，原始 raw 不改写。另按精确文本核对不同 ID 的内容：共有 1,300 个重复内容组，涉及 2,612 篇文档，每组最多 3 篇；740 个保留正例对均没有未标注的同内容替代文档。这与前述 4 个同 ID 冲突是不同问题。Markdown 物化已完成，未再出现超限文档；最终 corpus 为 511,954 篇、2,502,872,327 bytes，最大文档 42,377 bytes，正例数最大 10。[官方 HF](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)

### WorkSurface-Bench（延期方案，不计入本次 V1）

mirror TLS 与重定向到 Hugging Face 后的连接反复失败，因此未纳入本次 V1，不计入 2,221 题或构建完成项。后续若恢复获取，仅考虑 `rag_only`，优先读保留原嵌套结构的 `data/tasks.jsonl`（约 1.76 MB）；`data/tasks.parquet`（约 161 KB）部分字段为 JSON 字符串。候选仅来自对应 persona 的 `resources/profiles/{persona}/kb_docs/*.md`；`gold_evidence.file` 按原路径解析，不跨 persona 合并。213 题是上游 RAG 子集规模，不是本次已转换数量。[官方 HF](https://huggingface.co/datasets/lhpku20010120/WorkSurface-Bench)

### Agent Retrieval Bench

首批只处理三个 v2 子集、共 287 条问题的上游 frozen file-chunk view；不扩大到当前完整 release 的 427 题。已冻结 218 个唯一 snapshots、312,749 个原生文件视图，完成逐文件校验。精确路径分别为 `releases/v2_code2test/agent_retrieval_bench_v2_code2test.tar.zst`、`releases/v2_comment2context/agent_retrieval_bench_v2_comment2context.tar.zst`、`releases/v2_trace2code/agent_retrieval_bench_v2_trace2code.tar.zst`。使用 `sample.query` 的确定性序列化作为 query；实际全部 287 条样本都没有 `gold.files` 字段：分别使用 `gold.related_tests`、`gold.must_context_files`、`gold.root_cause_files` 作为三个任务的目标文件，287/287 映射一致；不能把其他角色的字段并入正例。保留原仓库、commit、文件路径与来源 revision；每个 repository/commit 候选域包含全部发布的 `kind=file` 记录，其他 chunk 仍保存在原始 archive。[上游许可说明](https://raw.githubusercontent.com/eyuansu62/agent-retrieval-bench/main/DATA_LICENSE.md)

上游 `corpus.py` 的 `kind=file` 视图经过 `io.truncate_text`：可能归一化 CRLF、`strip()`，并截到 8,000 字符后添加 `[truncated]`。因此“文件视图”不等于完整源码文件，也未必能逐字定位到原仓库。可选源代码行信息没有获取时，不输出伪造行号，不靠近似字符串匹配自动生成函数标签，也不宣称已经构建完整仓库环境。

实际 312,749 个文件视图中，85,724 个带截断标记。287 题中有 242 题至少一个正例被截断，共涉及 324 个正例出现次数；173 题的上游 span 超出发布视图行数，其余 114 题也未建立原始源码行对应关系。故这版的行/函数/功能覆盖指标均为 `null`，不是 0。截断可能使证据不可见，287 题版只能解释为该文件视图设定下的结果。

额外的 `dci_gold_untruncated_v1_20261014` 严格整题筛选：仅保留所有正例文件都没有截断标记的 45 题（code2test 22、comment2context 6、trace2code 17），共 53 个 query-file 正例对。候选文件集合与原题 snapshot 完全相同，没有只保留 gold，也没有剪去部分正例。这个子集适合优先做文件级验证，但其负例候选仍可能截断，也未建立源码行号对应，不能称为完整源码版本。

代码轨道沿用 `ranked_files` / `abstain` 与最多 20 文件的既有协议。实际共 415 个正例文件：code2test 148、comment2context 140、trace2code 127；每题最大 5 个，空 gold 与超过 20 个正例的查询均为 0。全部 query JSON 无完全重复；给定仓库与 commit 后亦无重复 query。8 条 trace2code 的原始堆栈包含目标文件路径，保留并在独立 annotation audit 中列出。没有 query 包含 fix_commit 或补丁文件头；review hunk 和 assertion diff 是原始工作流输入，不能单凭 `@@` 判断答案泄漏。

## 5. 后续候选的关键审计证据

- **LongMemEval-V2**：schema 中问题含 `id,domain,environment,question_type,question,image,answer,eval_function`；trajectory 含 `id,goal,outcome,states`，states 有文本 accessibility tree 与截图。small 每题 100 个候选，medium 通常 500；公开版明确移除了 answer-bearing labels。不能将所有 haystack 成员当 qrels。[Schema](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/blob/main/SCHEMA.md)
- **Context-Bench**：`datasets/filesystem_cloud.jsonl` 为 100 题，61,603 bytes；`filesystem_code.jsonl` 为同样问题添加指令，152,403 bytes。证据在 `agent_args.extra.required_files`；每题 3/4/5/6/7 篇的数量为 1/31/47/18/3。文件名本身就是原文件 ID。可以改编成文件选择任务，但分数不能等同原问答推理成绩。
- **τ-Knowledge**：`tasks.json` 1,138,204 bytes；`documents/*.json` 为 `id,title,content`。所有 97 题有 `required_documents`，39 题超过 10 个唯一正例、最大 30。`user_scenario.instructions` 含模拟用户行为与逐步披露信息，若整体作为检索 query 必须标成“静态场景检索派生版”。该 revision 包含当前任务修复，不能混用旧任务与新文档。[官方检索说明](https://github.com/sierra-research/tau2-bench/blob/dd8757d4c8791dad7b3cdaf02a105a3d5f9d86bd/src/tau2/knowledge/README.md)
- **HippoCamp**：三个 fullsets 的 files/QA 分别 Adam 344/123、Bei 875/235、Victoria 711/223；subset 不是额外独立评测样本。QA 的 `file_path` 为权威支持文件列表；Gold 的 `file_info.file_path` 可连接原文件，内容在 `segments[].content`。Gold 的 `QAID/QANum`、viewer 中按题拼接的 gold_text 及 evidence 不得进入 agent 语料。需要完整 Gold 文件集，不能仅抽取被标支持文件。非商业研究许可同时限制公开镜像；应保留许可并单独标明 gold-text 设定。[许可](https://huggingface.co/datasets/MMMem-org/HippoCamp/blob/main/LICENSE)
- **LoCoMo**：`evidence` 指向同一 `sample_id` 内的 `dia_id`。类别 1/2/3/4/5 共 282/321/96/841/446 题；9 题有未解析或不存在的证据，4 题为空，4 题有 >10 个唯一证据、最大 19。严格可解析的类别 1/2/4 共有 1,438 题；若纳入类别 3 则再加 89 题。类别 5 的 446 题虽有 evidence 也不能直接视为普通可回答正例。坏标签如 `D8:6; D9:17`、`D`、`D:11:26`、`D30:05`，不可静默纠正。图片本体未发布，只保留 URL/caption；现有文本 scope 不能声称覆盖原多模态任务。[README](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)
- **MADOQA/MADQA**：先确认截图名称。MADQA 的 train/dev `evidence[{document,page}]` 可以做页面证据审计，但 test labels 隐藏、PDF 外部托管与视觉信息保留是现实阻碍；不能将答案匹配当隐藏 test 证据。
- **OfficeQA Pro**：`source_files` 映射 basename 的确定性较好，但仍需合法访问、完整 697 文档集合、超 1MB 文件处理、源文件名冲突与证据粒度审核。
- **Bamboogle / LongBench v2 CodeQA**：公开 QA 或上下文本身不等于检索标注。当前没有可证明的原始文档/源码位置 qrels，不自行制造。

## 6. 已交付数据与验收

所有 HF 原始数据通过 `/mnt/afs2/202609/embedding_models/dci-bench/hfd.sh` 下载，设置 `HF_ENDPOINT=https://hf-mirror.com`，revision 固定到第 3 节所列 commit。BrowseComp-Plus 的两个小型官方 qrels 文件不在 HF 仓库，另从固定 GitHub commit 获取并保存 SHA-256。原始下载保留在各来源的 `raw` 目录。

### 通用 document

产物根目录：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/dci_expansion_20261014_v1`。

| 任务 | 文档数 | 查询数 | 正例对 |
|---|---:|---:|---:|
| LLMAILAStatutes | 82 | 50 | 217 |
| LLMFQuADRetrieval | 269 | 100 | 100 |
| LLMHC3FinanceRetrieval | 415 | 100 | 100 |
| LLMLegalBenchConsumerContractsQA | 154 | 100 | 100 |
| LLMPublicHealthQA | 172 | 100 | 100 |
| LLMTwitterHjerneRetrieval | 262 | 77 | 262 |
| wixqa_expertwritten | 6,221 | 200 | 258 |
| wixqa_simulated | 6,221 | 200 | 230 |
| browsecomp_plus | 99,930 | 825 | 5,023 |
| enterprise_rag | 511,954 | 469 | 740 |
| 合计 | 625,680 个任务内文档条目 | 2,221 | 7,130 |

两组 WixQA 使用同一完整 KB，所以文档合计不是跨任务去重数量。Markdown 总计 5,315,456,600 bytes，最大单文件 998,460 bytes。所有正例均能精确连接到各自候选语料。

- `workspaces/<task>/corpus/`：agent 可见的 Markdown 文档；保留原 ID 的可逆文件名映射。
- `metadata/<task>/`：host-only queries、qrels、原始标签、排除清单、文件 SHA-256、source manifest 与任务 `_COMPLETE.json`。
- `canonical/<task>/`：可重用的 corpus / queries / qrels Parquet。
- `catalog.json` 与根 `_COMPLETE.json`：固定 10 个任务及其完成标记哈希。
- `annotation_audit.json`：精确重复内容与正例替代检查，明确这是大小过滤前的源审计，不代表穷尽语义相关性。
- `build_sources/`、`build_provenance.json`：转换与验证代码快照、依赖和构建哈希。各任务 manifest 另记录其实际构建时的代码哈希。

根 catalog SHA-256：`7cf25856f56f04dbc8b98d598551dd38b83577e33ccc27377a9ab9f83e00457c`。

### Code

根目录：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data_code/agent_retrieval_bench`。

| 版本 / 子任务 | 查询数 | 正例对 | 用途 |
|---|---:|---:|---|
| 全量文件视图：code2test | 106 | 148 | 保留原发布样本 |
| 全量文件视图：comment2context | 80 | 140 | 保留原发布样本 |
| 全量文件视图：trace2code | 101 | 127 | 保留原发布样本 |
| 全量文件视图合计 | 287 | 415 | 包含上游截断影响 |
| 正例文件未截断子集 | 45 | 53 | 上述 287 题的子集，不额外相加 |

`dci_v1_20261014/` 是全量文件视图产物，包含 `queries.json`、host 标签、协议、source/archive hashes、每个 snapshot 的 manifest 和独立完成标记。**对单题只允许挂载 `snapshots/<snapshot_id>/corpus/`**；其余 snapshot、`host/`、query 列表及 raw archive 不应整体暴露给 agent。

`dci_gold_untruncated_v1_20261014/` 包含 45 题的 queries、可直接交给现有 code scorer 的 `labels.json`、逐题筛选清单及完成标记。该目录是 host 元数据包，候选语料引用全量文件视图版中的原 snapshot，不复制或缩减候选集。

`host_labels_v1_20261014.json` 是全量 287 题的独立 `dci-code-labels-v1` 评分输入，已逐题与 frozen snapshot reference 比较一致。45 题版本的评分输入为子集目录中的 `labels.json`。

`annotation_audit_v1_20261014.json` 保存原始角色字段、重复查询和 8 条路径线索审计；`acceptance_v1_20261014.json` 保存 287 题的 host scorer 人工完美排序验收。完美排序仅用于检查评分接口，F1=1 不代表模型成绩。

代码全量文件视图 manifest SHA-256：`b88d6a186a6a82bafdbbb39dfac26be18000078f9d48eebc962f342d77c21895`。

### 验证边界

32 项扩展转换测试、8 项 workspace builder 测试和 10 项 retrieval scoring 测试通过。通用版所有 2,221 题已通过现有 host data audit、Inspect sample 构造和 scorer oracle 检查，sample 中未携带答案或 gold。原有六任务已逐项比较 raw 与新 canonical 的 corpus、queries、qrels、revision 和 task instructions，内容一致。全量 Markdown 文件及 canonical Parquet 的哈希、独立普通文件属性、大小限制和标签闭合检查均通过。

代码版逐文件内容/哈希/行数/token 统计、snapshot 候选范围、标签闭合和 287 题现有 scorer 接口验收通过。45 题子集也已通过现有 code scoring CLI 验收，45/45 输出有效；结果保存在 `acceptance_gold_untruncated_v1_20261014.json`。行号、函数、功能覆盖指标均按缺失标注返回 `null`。代码轨道当前交付的是数据与 host 评分输入；仓库现有 document runner 没有被改成 code agent runtime。本次没有模型调用、端点 smoke、GPU 运行或 benchmark 模型成绩。

### 使用与复现

复核已完成的不可变版本：

```bash
DCI_DATA_ROOT=/mnt/afs2/202609/embedding_models/dci-bench
python scripts/prepare_expansion_data.py --verify-only --io-workers 32
python -m dci_bench.data.expansion_code verify \
  --output "$DCI_DATA_ROOT/eval_data_code/agent_retrieval_bench/dci_v1_20261014" \
  --io-workers 32
```

现有 document runner 通过 `scripts/run_expansion_task.py --suite <上述通用根目录>` 使用新增任务，其余参数沿用 `scripts/run_task.py`。已用现有 `dci-bench` 环境验证 `--help` 可加载全部 10 个任务；实际模型评测仍需要正常的 backend 和 sandbox preflight。

已有模型输出时，可按现有 code 协议评分；`predictions.json` 必须包含各题的 repository / base_commit / corpus_sha256 与模型 output：

```bash
python -m dci_bench.scoring --track code \
  --labels "$DCI_DATA_ROOT/eval_data_code/agent_retrieval_bench/dci_gold_untruncated_v1_20261014/labels.json" \
  --predictions predictions.json --output code_scores.json
```

需要重建时使用新输出目录，避免覆盖已冻结版本：

```bash
export HF_ENDPOINT=https://hf-mirror.com
DCI_DATA_ROOT=/mnt/afs2/202609/embedding_models/dci-bench
DCI_BUILD_TAG=$(date -u +%Y%m%dT%H%M%SZ)
DOC_OUT="$DCI_DATA_ROOT/eval_data/dci_expansion_$DCI_BUILD_TAG"
CODE_OUT="$DCI_DATA_ROOT/eval_data_code/agent_retrieval_bench/dci_$DCI_BUILD_TAG"
bash scripts/download_expansion_data.sh
python scripts/prepare_expansion_data.py --suite "$DOC_OUT" --baseline-only
python scripts/prepare_expansion_data.py --suite "$DOC_OUT" \
  --sources wixqa browsecomp_plus enterprise_rag
python scripts/prepare_expansion_data.py --suite "$DOC_OUT" --finalize
python -m dci_bench.data.expansion_code build-arb \
  --raw "$DCI_DATA_ROOT/eval_data_code/agent_retrieval_bench/raw" \
  --output "$CODE_OUT" --io-workers 32
python -m dci_bench.data.expansion_code_subset \
  --source "$CODE_OUT" --output "${CODE_OUT}_gold_untruncated"
```

默认下载与构建来源排除 WorkSurface。文档筛选不补标签、不从答案猜相关性、不只构建 gold corpus；代码子集整题筛选，保留原 snapshot 全部候选。

跨轨道交付清单：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/dci_expansion_20261014_v1_release.json`。其中固定 document、code 两个版本、代码子集与独立评分标签的路径、数量和 SHA-256。
