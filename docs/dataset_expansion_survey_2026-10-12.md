# DCI-Bench 扩展数据集调研与推荐排序

> 2026-10-14 实现补充：评价层已拆成 Document / Code 两轨。本文 2026-10-12
> 调研中的 Top-10、Markdown 折行、`ranked_doc_ids` 等现行约束继续指 Document。
> Code 使用独立的文件/行区间输出、显式弃答、文件召回/MRR/BCY 及可选函数与功能证据覆盖；
> 详见 [Code 评价协议与可运行样例](code_evaluation.md)。现已接入独立在线入口
> `scripts/run_code_task.py`：支持已完成的 ARB 导出、Pi 源码行号读取与逐题宿主端评分。
> 在线链路已通过 CPU / 本地 HTTP mock 验证，尚未运行真实模型代码基准。

调研基准日：2026-10-12。目标是扩展模型在**给定、冻结、离线语料库中自主搜索、阅读、核验证据并停止检索**的能力评测。本文面向当前 DCI-Bench 的任务选择，同时列出值得单独建立的代码、结构化数据、视觉文档和答案生成轨道。

推荐指数衡量的是**加入当前项目的价值与可行性**，不是数据集知名度、模型在其上的成绩或实验实测结论。来源事实与本文提出的 DCI 改造方案分别表述；建议抽样规模均为待实施设计。本次仅核查仓库、公开论文、官方项目与数据卡，没有下载完整候选语料、执行 OCR 或运行模型评测。

## 1. 当前项目基线与真正需要补足的能力

### 1.1 已有任务

以下规模来自本地 `data/metadata/<task>/manifest.json`，每题正例数量来自对应 `qrels.json`，均在本次调研中只读核对。

| 已有任务 | 文档数 | 查询数 | 每题正例数 | 已覆盖的主要能力 |
|---|---:|---:|---:|---|
| LLMAILAStatutes | 82 | 50 | 2–5 | 长案情转法律概念，寻找相关法条 |
| LLMFQuADRetrieval | 269 | 100 | 1 | 法语问答段落检索 |
| LLMHC3FinanceRetrieval | 415 | 100 | 1 | 财经问题与候选回答匹配 |
| LLMLegalBenchConsumerContractsQA | 154 | 100 | 1 | 消费合同条款检索 |
| LLMPublicHealthQA | 172 | 100 | 1 | 公共卫生问答段落检索 |
| LLMTwitterHjerneRetrieval | 262 | 77 | 1–6 | 丹麦语社交文本，拼写与形态变化，多正例 |

共 527 道查询；四项任务每题只有一个正例，全部六项的已标注正例数均不超过 10。这个组合适合验证工具与检索流程，但不足以单独代表大语料、多跳取证、长文档精确导航和跨文件约束推理。文档数量也不能代替总字节量或 token 量；后续分层应同时记录这些规模。

**这六项已覆盖 MTEB(LLM) 官方任务表的全部 retrieval 子集。** 官方共 37 项任务，其余为分类、聚类、STS 和成对分类；不能把它们直接算作尚未接入的同类检索任务。扩展应主要从下文的新数据源选取，而不是继续寻找不存在的第七个 `llm-eval-*` retrieval 任务。[MTEB(LLM) 官方任务表](https://github.com/embeddings-benchmark/embedders-dilemma/blob/main/DATASETS.md)

### 1.2 影响候选适配性的现行协议

当前权威实现是 [`registry.py`](../dci_bench/data/registry.py)、[`contract.json`](../dci_bench/protocol/contract.json)、[`workspace_builder.py`](../dci_bench/data/workspace_builder.py) 和 [`retrieval.py`](../dci_bench/scoring/retrieval.py)，不能直接套用旧运行 manifest 内的历史协议副本。

| 约束 | 当前状态 | 对扩展的影响 |
|---|---|---|
| 交互方式 | `dci-mvp-v3`；离线 `bash` 和 `read`；无网络、外部 retriever 或 embedding | 候选必须能冻结为本地可查阅语料；在线网站或搜索 API 的成绩不能直接迁移 |
| 输出 | `ranked_doc_ids`，1–10 个唯一原始 ID | 无答案/弃答不兼容；大于 10 个正例时完整召回存在结构性上限 |
| 证据门槛 | 至少一次非空 `read`；`bash` 命中不算完成证据读取 | 只证明发生了阅读，尚不能证明读过每个返回文档或全部必需证据 |
| 预算 | 累计输入输出 128,000 tokens；每题 600 秒；工具输出上限 20,000 UTF-8 bytes | 语料大小、文件布局、工具输出量和反复读取会共同影响结果 |
| 上下文 | 启动器通常使用单请求 65,536 上限；与累计预算分开 | 长语料测试不等于把全部语料塞入一个长上下文 |
| 文件 | 当前为 Markdown；单篇上限 1,000,000 bytes；正文自动折行 | 代码行号、表格结构、字符级证据跨度需要专门保真映射 |
| 标签 | query/qrels 留在 host，语料保留原始 ID | answer、supporting facts、gold context、裁判解释等不能写入模型可见语料 |
| 指标 | Recall、F1、nDCG；Precision 分母为实际返回数量；nDCG 使用线性相关性增益 | 与固定 k 分母或指数增益的上游排行榜不可直接比较；`@20` 不会突破输出最多 10 篇的约束 |

据此，最有价值的新增维度是：**推理后改写检索词、多跳证据闭合、长文档内部定位、指令限定与反事实条件、时间/版本消歧、多语言与跨语言、跨文件关系，以及有限预算下的检索效率**。仅扩大单跳问答题量的收益相对较小。

## 2. 推荐指数与接入等级

### 2.1 评分方法

每个维度取 1–5 分，分值越高越适合当前扩展。总分为：

`推荐指数 = 6D + 5G + 4N + 3I + 2R`，满分 100。

| 维度 | 权重 | 5 分意味着什么 | 降分的主要原因 |
|---|---:|---|---|
| D：直接语料交互测量价值 | 30% | 需要搜索、分步阅读、核验或跨文件追踪 | 主要是给定短上下文阅读、答案生成或纯合成字符串定位 |
| G：标签与评分可靠性 | 25% | 公开、有明确粒度、能表达必需证据或相关性 | 只有答案、弱代理标签、稀疏池化判定、测试 gold 不开放 |
| N：相对已有六项的增量 | 20% | 补足当前明显空白，覆盖独立能力 | 重复现有短问答/合同条款类型，或与其他候选高度重叠 |
| I：接入与运行可行性 | 15% | 小规模导入即可沿用输出、工具和评分主干 | 大规模资源、数据清洗、复杂跨度映射、需要新工具或新评分轨道 |
| R：可复现与发布清晰度 | 10% | 公开冻结数据、来源 ID、许可边界清楚 | 动态网页、下载限制、授权不明、代码许可与数据许可混淆 |

这些分值是基于来源与工程约束作出的审慎判断，不是统计估计。相差 1–3 分宜视为同一梯队；同分时优先考虑标签更直接、对现有任务更互补的具体子集。**排序对象是表中所推荐的子集/使用方式，不是整个数据集品牌。** 同一数据集的视觉版、生成版或完整规模版可能属于另一接入等级。

### 2.2 接入等级

| 等级 | 含义 | 可以保留什么 | 仍需做什么 |
|---|---|---|---|
| A | 原生文档/段落/实体检索标签 | 当前返回 ID 及检索指标主干 | 数据 adapter、ID/格式校验、固定 revision；A 不表示零代码工作 |
| B | 有证据、跨度、会话或代码位置，可确定性派生 qrels | 当前返回 ID 接口通常可保留 | 定义检索单位，映射证据，审计 gold 完整性；结果必须标为 DCI 派生任务 |
| C | 原始能力需要新输出、工具或评分语义 | 可复用语料与运行基础设施 | 新增答案/弃答/图像/区域/状态更新轨道，不能并入现有六项同口径总分 |

接入成本在后文用低/中/高表示，是相对工程成本；不暗示已测得下载时长或 GPU 费用。

## 3. 按推荐指数从高到低的总表

优先考虑 **BRIGHT / Bright-Pro、BrowseComp-Plus、FreshStack、MuSiQue-Ans**；它们分别强化推理及互补证据覆盖、固定大语料搜索、技术文档多要点覆盖和多跳证据闭合。若纳入代码能力，**RepoQA 与 ContextBench** 价值很高；记忆检索优先 **LMEB / LongMemEval**。中文和语言覆盖应从 **LeCaRDv2、MIRACL、C-MTEB** 中主动选取，不能仅按总分取前若干名而使组合过度集中于英文和代码。

下表共 **50 项候选方案**，部分行包含同源家族，ViDoRe 文本/视觉版则因工具边界不同分列；**不是 50 个互不重叠的数据集**。A/B 是当前检索主线的候选；C 即使分数较高，也必须先完成独立轨道设计。维度次序为 D/G/N/I/R，计算规则见第 2 节；同分排列不代表统计显著差异。

| 排名 | 候选方案 | 推荐指数 | D/G/N/I/R | 接入等级与成本 | 推荐起点 |
|---:|---|---:|---|---|---|
| 1 | BRIGHT | 93/100 | 5/4/5/5/4 | A；低至中 | biology/sustainable_living 的 short；biology-long 单列 |
| 2 | Bright-Pro | 92/100 | 5/5/5/3/4 | A；aspect评分需扩展；中 | biology、sustainable_living；官方 175-query agentic 子集可作固定抽样入口 |
| 3 | BrowseComp-Plus | 92/100 | 5/5/5/3/4 | A；中至高 | 官方 830-query 固定离线语料版；evidence qrels 为主 |
| 4 | FreshStack | 90/100 | 5/4/5/4/4 | A；中 | oct-2024 的 langchain、godot、yolo |
| 5 | MuSiQue-Ans | 90/100 | 5/4/5/4/4 | B；中 | Ans dev 的 2/3/4-hop；保留完整支持链 |
| 6 | RepoQA | 89/100 | 5/5/4/4/3 | B；中 | Searching Needle Function，按五种编程语言分层 |
| 7 | ContextBench | 88/100 | 5/5/5/3/2 | B；行级为C；中至高 | contextbench_verified，随后按语言扩展 |
| 8 | LMEB | 88/100 | 5/4/5/4/3 | A，需保留scene候选域；中 | Dialogue 的 temporal/multi-session；Procedural、Episodic 各取代表 |
| 9 | LeCaRDv2 | 87/100 | 5/4/5/3/4 | A；中至高 | 官方 test_query.json；必要时另建 ≤10 正例诊断子集 |
| 10 | QUEST | 87/100 | 5/4/5/3/4 | A；完整集合需扩展；中至高 | validation 中答案集合 ≤10 的分层诊断；全 test 另报截断版 |
| 11 | LongMemEval V1 | 87/100 | 5/4/5/3/4 | B；中 | cleaned release 的 S；answerable temporal/update/multi-session |
| 12 | EnterpriseRAG-Bench | 84/100 | 5/3/5/3/5 | A；无gold类别为C；中至高 | Semantic、Project Related、Constrained、Conflicting Info、Completeness |
| 13 | STaRK | 84/100 | 5/4/5/2/4 | A，需关系序列化；高 | MAG 的 human_generated_eval；再考虑 Prime 与 Amazon |
| 14 | MultiDoc2Dial | 84/100 | 4/4/5/4/4 | B；中 | 公开 validation 中跨文档切换的对话轮次 |
| 15 | LegalBench-RAG | 84/100 | 5/4/4/4/3 | B；跨度为C；中 | LegalBench-RAG-mini；CUAD、ContractNLI、MAUD、PrivacyQA 分项 |
| 16 | MIRACL | 83/100 | 4/5/5/2/4 | A；中至高 | 公开 dev：zh，加 ar/ja/sw/th 中两种 |
| 17 | FollowIR | 83/100 | 5/4/5/3/2 | A；配对评分需扩展；中至高 | original/modified 指令成对；core17 或 robust04 |
| 18 | InstructIR | 83/100 | 5/3/5/4/3 | A；配对评分需扩展；低至中 | 官方 qrels/test；同一原始查询的不同 instruction 成组保留 |
| 19 | CoIR | 81/100 | 4/4/4/5/3 | A；低至中 | CosQA、CodeSearchNet、CodeSearchNet-CCR；少量 CodeFeedback-MT |
| 20 | ViDoRe V3 文本派生版 | 81/100 | 4/4/5/3/4 | B；中至高 | 公开 HR、Finance-EN 中正文/OCR 足以支撑的页面查询 |
| 21 | ViDoRe V3 视觉原版 | 81/100 | 5/4/5/1/4 | C；高 | 八个公开领域；优先 HR、Finance-EN，再工业图示 |
| 22 | SciFact / SciFact-Open | 81/100 | 4/4/4/5/3 | A或B，取决于封装；低；Open为高 | 原始 dev 有证据的claim；或固定 BEIR scifact test |
| 23 | MLDR | 79/100 | 4/3/5/4/4 | A；中 | zh/en test；补一个非拉丁文字语种 |
| 24 | FinQA / ConvFinQA / TAT-QA | 79/100 | 4/4/5/3/3 | B；计算为C；中至高 | FinQA public test；ConvFinQA test_turn；TAT-QA released test gold |
| 25 | LitSearch | 78/100 | 4/4/4/4/3 | A；中 | author-written 查询优先；按 specificity 分层 |
| 26 | QMSum | 78/100 | 4/4/4/4/3 | B；总结为C；中 | specific_query_list 中有 relevant_text_span 的问题 |
| 27 | C-MTEB 中文检索 | 78/100 | 4/4/4/4/3 | A；低至中 | DuRetrieval、T2Retrieval；MMarcoRetrieval 作翻译对照 |
| 28 | QASPER | 78/100 | 4/4/4/4/3 | B；视觉/答案为C；中 | validation 的 answerable、text-evidence 问题 |
| 29 | LoCoMo | 78/100 | 4/4/4/4/3 | B；中 | locomo10 的 evidence 可对齐 single/multi-hop/temporal 问题 |
| 30 | FinanceBench | 77/100 | 4/4/5/3/2 | B；数值答案为C；中 | 官方 open-source 150 题 |
| 31 | HotpotQA | 77/100 | 4/4/4/3/4 | B；中至高 | fullwiki dev；distractor 仅作低成本对照 |
| 32 | FEVEROUS | 76/100 | 4/4/5/2/3 | B；表格/判定为C；高 | 公开 dev 的表文结合与多跳证据问题 |
| 33 | SWE-Explore-Bench | 75/100 | 5/3/5/2/2 | B；行区域为C；高 | 固定 repository snapshot 的 core-context 检索子集 |
| 34 | TREC iKAT | 75/100 | 5/4/5/1/1 | C；固定turn可派生A/B；很高 | 先考虑 2023 已标注turn；完整交互版另轨 |
| 35 | M3DocVQA | 74/100 | 5/3/5/1/3 | C；高 | M3DocVQA 的公开多文档、多页面问题 |
| 36 | BEIR 补充子集 | 73/100 | 3/4/3/5/4 | A；低至中 | ArguAna；NFCorpus；CQADupStack 的少量社区 |
| 37 | AIR-Bench | 73/100 | 4/3/4/4/3 | A，仅公开dev；中 | 24.05 Finance-English-dev；Finance-Chinese-dev 条件扩展 |
| 38 | LongEmbed / LoCo | 72/100 | 3/4/4/4/3 | A或B，按所用封装；中 | LongEmbed 真实长文任务；LoCoV1 按领域选取 |
| 39 | LongMemEval-V2 | 71/100 | 5/2/5/1/4 | C；高 | 公开 small/medium tiers；未来长期agent记忆轨道 |
| 40 | KILT | 70/100 | 4/4/3/2/4 | B；高 | 公开 dev 的 provenance 任务；避免重复计入 HotpotQA/FEVER |
| 41 | Mr.TyDi / mMARCO | 69/100 | 3/4/4/3/3 | A，选带qrels版本；中 | Mr.TyDi v1.1 的 ar/sw/th；mMARCO 使用明确评测封装 |
| 42 | 2WikiMultiHopQA | 69/100 | 4/4/3/3/2 | B；中 | 公开 dev 12,576，按四种推理类型分层 |
| 43 | CUAD / ContractNLI 原始数据 | 67/100 | 3/4/3/3/4 | B；NLI为C；中 | CUAD 有答案的条款；ContractNLI 有证据的 entailment/contradiction |
| 44 | FEVER | 67/100 | 3/4/3/3/4 | B；verdict/NEI为C；中至高 | 公开 labelled dev 中 SUPPORTS/REFUTES 的证据任务 |
| 45 | MemoryBench | 63/100 | 4/2/5/1/3 | C；高 | 完整memory交互轨道；已有LoCoMo子集不重复接入 |
| 46 | RULER / BABILong | 63/100 | 3/3/3/4/3 | B/C，诊断而非主榜；低至中 | 多needle/多跳支持事实；固定随机种子与干扰长度 |
| 47 | FinDER | 62/100 | 4/2/4/2/3 | B，需先完成出处对齐；高 | 公开 5,703-triplet 发布中另行预注册的 held-out 切分 |
| 48 | LongDA / LongProc | 61/100 | 4/2/5/1/2 | C；高 | 长期规划与跨文档计算的未来拓展 |
| 49 | LongBench / LongBench v2 / InfiniteBench | 56/100 | 3/2/4/2/3 | C；少量检索子题可派生；中至高 | 只作长上下文控制；有稳定证据的子任务另审 |
| 50 | MultiFin | 40/100 | 1/2/3/2/3 | C，暂不建议；高 | 只适合另设多语财经分类轨道 |

## 4. 逐项依据、适用子集和接入风险

### 4.1 BRIGHT

**推荐指数 93/100；A；成本低至中。** 主要增量：推理检索、领域术语迁移、短段与完整网页对照。

**数据事实：**官方 examples 配置提供 gold_ids、gold_ids_long、excluded_ids。当前 raw 数据卡中 biology 有 103 题；short corpus 57,359 文档，long corpus 524 文档。sustainable_living 有 108 题；对应 short/long 是不同粒度，不能共享一套 doc IDs。 [作者仓库](https://github.com/xlang-ai/BRIGHT)；[官方数据卡](https://huggingface.co/datasets/xlangai/BRIGHT)；[论文](https://arxiv.org/abs/2407.12883)

**接入建议：**首先做 biology-short、sustainable_living-short 和 biology-long，既有规模变化又有长文读取。检索主任务只用原始 query 与相应 gold IDs；按候选 corpus 真实字节大小检查长文件，20k 工具输出可分页读取，不应因此截断文档。

**主要限制：**reasoning、gold_answer 属于隐藏监督。数学/代码任务的 excluded_ids 必须按上游规则处理；不能把自匹配命中算成绩。原始 short/long 指标与 DCI 的截断返回、预算、scorer 不同，不能直接对比排行榜。涉及超过 1MB 的文件应另建明确版本。

**许可与获取：**官方卡标示 CC-BY-4.0；保留原网站内容来源及权利信息。

### 4.2 Bright-Pro

**推荐指数 92/100；A；aspect评分需扩展；成本中。** 主要增量：专家分解的互补推理要点、证据覆盖、检索停止时机。

**数据事实：**2026 年发布的 BRIGHT 扩展覆盖 7 个 StackExchange 领域、739 个查询、2,763 个推理要点；官方卡统计 5,272 个 gold passages、526,319 个 corpus documents。examples 提供 query/gold_ids/reference_answer，aspects 提供 content、1–3 的重要性权重及 supporting_docs；原正例经复核并补充新证据，由同领域第二位标注者复查。作者还发布每领域 25 题、共 175 题、seed=42 的固定 agentic 子集。 [官方数据卡与逐领域统计](https://huggingface.co/datasets/yale-nlp/Bright-Pro)；[作者仓库与固定题目列表](https://github.com/yale-nlp/Bright-Pro)；[论文](https://arxiv.org/abs/2605.04018)

**接入建议：**沿用原 query、冻结 corpus 和 gold_ids 做当前检索主分，同时保留 query→aspect→doc 映射，在 host 端增加加权 Aspect-Recall 和 α-nDCG 诊断。先选 biology（103 题、59,513 docs）及 sustainable_living（106 题、63,142 docs），并与 BRIGHT 同源题配对比较。官方 agentic 子集只作为固定题目列表，不复用其外部 retriever 或运行时搜索工具。

**主要限制：**reference_answer、aspect 描述、权重和支持文档映射全部属于隐藏监督。新增证据 ID 常形如 task-qid/extraction_k.txt，可能泄露题目归属；须审计模型可见 query ID、路径和文件顺序，原 ID 主版若存在可利用捷径则暂缓正式纳入，另设计可逆中性 ID 的独立消融版，保留原 ID/qrels。与 BRIGHT 的 query 和网页来源重叠，主总分不能重复加权；专家正例仍不保证穷尽所有可行证据。新评分的 α、权重归一化、Top-10 截断均须固定。

**许可与获取：**官方代码及数据卡标示 MIT；上游 StackExchange query、BRIGHT corpus 和新收集网页仍保留各自原始许可，不能用 MIT 概括全部来源。

### 4.3 BrowseComp-Plus

**推荐指数 92/100；A；成本中至高。** 主要增量：困难开放域搜索、长网页阅读、中间线索与最终答案出处。

**数据事实：**官方固定 corpus 有 100,195 个网页文档、830 道题。evidence documents 覆盖支持线索的证据，gold documents 是其中同时承载最终答案的子集。本次只读统计公开 qrels：evidence 共 5,064 对，单题 1–13 篇，17/830 题（2.05%）超过 10；gold 共 2,407 对，单题 1–9 篇。 [官方仓库](https://github.com/texttron/BrowseComp-Plus)；[查询与标签数据卡](https://huggingface.co/datasets/Tevatron/browsecomp-plus)；[语料卡](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus)；[官方 evidence qrels](https://raw.githubusercontent.com/texttron/BrowseComp-Plus/main/topics-qrels/qrel_evidence.txt)；[官方 gold qrels](https://raw.githubusercontent.com/texttron/BrowseComp-Plus/main/topics-qrels/qrel_golds.txt)

**接入建议：**主分使用 evidence qrels，另报 gold 文档命中/召回；只选 gold 会漏掉中间证据。完整 100k 语料是主目标，可先用固定 5k/20k 派生池验证成本。保留官方网页文本和原始 ID，使用分页 read；查询解码按官方脚本，answer 与解码监督文件留在 host。

**主要限制：**qrels 是固定 corpus 内已核验的证据池，并不穷尽所有可能相关网页。17 道超上限题不能通过删 gold 修正分母。不同于原版在线 BrowseComp，也不同于上游带专用 retriever 的 agent；最终答案正确率仍须另轨评测。

**许可与获取：**仓库/HF 卡标 MIT；原网页内容的来源与权利信息另行保留。

### 4.4 FreshStack

**推荐指数 90/100；A；成本中。** 主要增量：真实技术问题、文档与代码混合、多个答案要点覆盖。

**数据事实：**官方冻结版提供五个主题的查询、语料、query-level 与 nugget-level qrels。数据卡列出 LangChain 203 题/49,514 chunks，Godot 99 题/25,482 chunks，YOLO 57 题/27,207 chunks；这里的单位是 chunk，不是仓库文件。查询来自社区问题，nugget 与相关性判定包含 LLM 生成/判断，因此不是全人工穷尽标注。 [官方实现](https://github.com/fresh-stack/freshstack)；[查询数据卡](https://huggingface.co/datasets/freshstack/queries-oct-2024)；[语料数据卡](https://huggingface.co/datasets/freshstack/corpus-oct-2024)；[论文](https://arxiv.org/abs/2504.13128)

**接入建议：**优先选 LangChain 与 Godot，必要时补 YOLO；沿用原 chunk ID 和冻结版本。当前主分可用 query-level qrels，另加 nugget coverage@10 与 α-nDCG@10 观察是否只返回重复表达同一要点的文档。若做 1k/5k/完整主题三级语料，应公开派生池并保留全部已知正例。

**主要限制：**不能把参考答案、nuggets、query-to-nugget 映射暴露给代理，也不能让上游的 oracle retrieval 参与运行时搜索。库版本与问题时间必须匹配；原论文较大的召回 cutoff 不能直接当成 DCI@10。

**许可与获取：**数据卡/README 标示 CC-BY-SA-4.0，代码 Apache-2.0；底层 GitHub 文件来源与许可仍随 manifest 保留。

### 4.5 MuSiQue-Ans

**推荐指数 90/100；B；成本中。** 主要增量：真正的连续多跳、找全中间证据、避免单跳捷径。

**数据事实：**MuSiQue-Ans dev 有 2,417 题，包含 2、3、4 跳问题并标注 supporting paragraphs；MuSiQue-Full 另包含不可回答对照。原版为带干扰段落的 QA 任务。 [官方仓库和格式](https://github.com/stonybrooknlp/musique)；[论文](https://arxiv.org/abs/2108.00573)

**接入建议：**从 Ans dev 按 hop 数分层。最低成本版本保留原每题候选集；更具 DCI 价值的版本以同 split 段落去重并构建固定大池，全部支持段落保留，并明确命名为 pooled-corpus 派生版。必须维护原来源/段落 ID 与去重映射，额外报告 All-support@10。

**主要限制：**把各题 distractors 合并会引入未标注的替代证据，需要抽样核验。不得只用最终答案所在段落作正例，也不能将问题分解及答案链暴露给模型。Full 无答案题与当前非空输出冲突，需另轨。

**许可与获取：**官方仓库标示 CC-BY-4.0，底层来源映射随数据保留。

### 4.6 RepoQA

**推荐指数 89/100；B；成本中。** 主要增量：自然语言描述到真实函数、仓库内导航、代码语义阅读。

**数据事实：**官方 SNF 含 500 项：5 种编程语言 × 10 个仓库 × 10 个目标函数。查询描述避免直接透露函数名，原任务要求在长代码上下文中找出函数。 [官方仓库](https://github.com/evalplus/repoqa)；[论文](https://arxiv.org/abs/2406.06025)

**接入建议：**把对应固定仓库代码暴露为本地文件，以函数或固定源码区段建立可返回 ID；保留 file/commit/start_line/end_line 映射。以函数 ID 为主评分单位，文件命中率只作诊断。原版拼接长上下文与交互式文件搜索分别命名报告。

**主要限制：**只按父文件打分会把同一文件中选错函数也算正确。不能按 gold 边界只切出目标函数而省略其他函数；检索单位必须对整个仓库按统一规则构造。已有文本折行器不能原样用于代码行号评分。

**许可与获取：**代码仓库 Apache-2.0；汇入语料的各原始仓库需保留各自许可。

### 4.7 ContextBench

**推荐指数 88/100；B；行级为C；成本中至高。** 主要增量：真实 issue 驱动的上下文搜索、跨文件依赖与精确取证。

**数据事实：**项目报告 1,136 个任务、66 个仓库、8 种编程语言，并提供人工标注的 gold context。HF 另有 500 项 contextbench_verified；公开存储 split 名为 train，不代表应拿这些评测题训练模型。字段包含 base_commit、gold_context 的文件/起止行及参考补丁。 [项目与规模](https://contextbench.github.io/)；[官方数据卡](https://huggingface.co/datasets/Contextbench/ContextBench)；[代码与许可](https://github.com/EuniAI/ContextBench)

**接入建议：**先冻结 500 子集中的固定样本，在 base_commit 的原始代码快照上做只读探索。低成本版本把 gold context 投影为文件 qrels；更符合原目标的版本评价 read 轨迹的行级覆盖、精度与预算效率，需单独评分轨道。

**主要限制：**gold_context、patch、test_patch 和修复后源码全部留在 host。不能把 patch 修改文件当作完整上下文集合。文件 qrels 版丢失精确行定位能力，应在任务名中显式标记；各来源 issue 还要和 SWE-bench/RepoQA 去重。

**许可与获取：**官方代码仓库 Apache-2.0；本次读取的 HF 数据卡未见单独 license 声明，原始仓库与来源任务许可须在正式分发前逐项确认，因此 R 分保守。

### 4.8 LMEB

**推荐指数 88/100；A，需保留scene候选域；成本中。** 主要增量：长期记忆证据、经验/流程检索、时间和场景限定。

**数据事实：**正式名称是 Long-horizon Memory Embedding Benchmark。官方包含 22 个数据集、193 个检索任务，分 Dialogue、Episodic、Semantic、Procedural 四类；提供 corpus、queries、qrels 及候选域。此次检查 LongMemEval 的 candidates.jsonl，可见 scene_id→candidate_doc_ids，不能默认全根目录 corpus 对每个 query 都可见。 [官方实现](https://github.com/KaLM-Embedding/LMEB)；[数据卡、结构与许可](https://huggingface.co/datasets/KaLM-Embedding/LMEB)；[scene 候选文件](https://huggingface.co/datasets/KaLM-Embedding/LMEB/blob/main/Dialogue/LongMemEval/candidates.jsonl)

**接入建议：**优先选 Dialogue/LongMemEval 的 temporal_reasoning、multi_session、knowledge_update，以及一个许可清晰的 Procedural/Episodic 任务。沿用原 ID、qrels 和 scene 候选域，为每个 scene 构建只读视图，或分组为子任务。Semantic 中 QASPER/MLDR/SciFact 若已单独接入，只作同源实现对照，不重复加权。

**主要限制：**原指标 R_cap@10 与 DCI Recall/F1 不同；候选集合限制也是任务定义。将所有人的记忆或所有 scene 直接混池会改变任务。它评的是记忆检索，不能代表记忆写入、更新策略或回答正确性；Procedural 检索也不等于执行工具流程。

**许可与获取：**顶层 MIT 不覆盖所有组件；官方列出混合 CC-BY-NC、CC-BY-SA 等，部分组件许可未说明，NovelQA/ESG-Reports 还涉及版权内容。按实际选中组件逐项筛选。

### 4.9 LeCaRDv2

**推荐指数 87/100；A；成本中至高。** 主要增量：中文长案情、法律关系而非字面相似、分级相关性。

**数据事实：**官方有 800 个查询案例、55,192 个候选案例，train/test 为 640/160。由多名法律专家按定性、量刑、程序等维度判断相关性。relevence.trec 提供原分级标签，relevence_gold.trec 保留相关性 ≥2 的条目；官方平均相关案例数为 20.89，须明确 Top-10 的能力上限。 [官方数据结构和许可](https://github.com/THUIR/LeCaRDv2)；[MTEB 派生卡](https://huggingface.co/datasets/mteb/LeCaRDv2)；[LeCaRDv1](https://github.com/myx666/LeCaRD)

**接入建议：**首选官方 160 题 test 和完整候选集，先审计文件体积与评分等级。检索主分明确相关阈值，nDCG 保留可解释的分级增益；≤10 正例子集仅作完整集合诊断，不替代全 test。按罪名和长案情长度分层。

**主要限制：**MTEB 的 LeCaRDv2 repack 与官方完整候选集不同，不能把较小 repack 的结果称为官方全库结果。池化人工标注仍可能漏掉未评判相关案件。LeCaRDv1 的 107-query 小池可作廉价对照，但不必与 v2 同时纳入主总分。

**许可与获取：**官方仓库 MIT；来源案件、匿名化处理和所用发布版本一并记录。

### 4.10 QUEST

**推荐指数 87/100；A；完整集合需扩展；成本中至高。** 主要增量：隐含交集/并集/差集、严格条件过滤、集合完整性。

**数据事实：**QUEST 有 3,357 queries、325,505 个 Wikipedia 实体文档；validation 323 题，test 1,727 题。官方答案集合大小 2–20，平均 10.5；不是普通只找一个证据页面的问答任务。 [Google 官方实现](https://github.com/google-research/language/tree/master/language/quest)；[官方论文与集合统计](https://aclanthology.org/2023.acl-long.784/)

**接入建议：**优先完整保留 query 和 entity answer set。当前协议下，可先建 ≤10 个答案的分层子集；同时保留完整 test 的 Recall@10 与可达上限。若要评价官方式完整集合返回，应单独允许最多 20 或可变更多结果，新增集合完整性指标。

**主要限制：**不能只截掉 qrels 的第 11–20 个答案，让 Recall 看上去更高。≤10 过滤会改变难度和领域分布，必须披露；plants/animals 官方平均集合较大，不能只挑这两个域却忽略上限。知识类别与页面中可见证据也不总能完美对应。

**许可与获取：**代码与底层 Wikipedia 文本许可分别保留。

### 4.11 LongMemEval V1

**推荐指数 87/100；B；成本中。** 主要增量：时间推理、知识更新、跨会话证据找回。

**数据事实：**原基准有 500 道问题，S 每题历史约 115k tokens，M 约 500 sessions；提供 answer_session_ids 和 turn-level has_answer。官方 README 指向 2025-09 cleaned release，并说明检索评测跳过 30 道 abstention 题。 [官方仓库与字段](https://github.com/xiaowu0162/LongMemEval)；[cleaned 发布](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)；[原数据卡](https://huggingface.co/datasets/xiaowu0162/longmemeval)

**接入建议：**优先 cleaned S 的可回答部分，保留每题专属 haystack、时间戳及问题日期；用 session qrels，必要时再做 turn 粒度消融。现仓库的 task 共享 workspace 需要扩展成 scene/样本视图，或使用严格等价的任务分组，不能简单合并所有样本历史。

**主要限制：**oracle 文件只保留证据 sessions，不能用它宣称完成长历史搜索。answer、answer_session_ids、has_answer 都是 host-only。找回新旧记录不等于选择正确最新事实；真正更新/拒答能力需答案轨道。

**许可与获取：**原仓库 MIT；保留 cleaned revision 与原始生成/填充会话来源。

### 4.12 EnterpriseRAG-Bench

**推荐指数 84/100；A；无gold类别为C；成本中至高。** 主要增量：企业异构文档、别名、条件过滤、冲突与完整集合检索。

**数据事实：**Onyx 的官方发布含 500,000+ 文档、500 题，模拟同一家公司中的 Slack、邮件、工单、Drive、会议与 wiki 等九类来源；这是合成企业语料，不是真实公司内部资料。数据提供 expected_doc_ids。Completeness 类有 20 题，要求的完整文档集合不超过 10；High Level 10 题与 Info Not Found 20 题没有 gold documents。 [官方仓库与类别](https://github.com/onyx-dot-app/EnterpriseRAG-Bench)；[数据卡与字段](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)；[生成方法](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/main/methodology.md)

**接入建议：**首选上述五类有证据 ID 的问题，将 expected_doc_ids 转为 qrels；保留 source_type、时间与文档内部元数据。采用冻结的 5k/20k/全库规模阶梯，前两档明确属于派生 DCI 子集。另将 100 道 metadata-dependent extra questions 作为独立扩展，先核查所需元数据是否实际包含在发布语料。

**主要限制：**合成一致性和 gold 漏标需要抽查。全库 50 万文件不能直接假定在 600 秒内可公平扫描；不要在每个模型运行时动态修正 gold。冲突题返回对立证据可测取证，但只有新增答案评分才能判断模型如何解决冲突。30 道无 gold 类别不能硬塞进非空 ID 输出任务。

**许可与获取：**官方 HF 数据卡标示 MIT；保留上游 canary 与版本信息，避免将评测题混入训练。

### 4.13 STaRK

**推荐指数 84/100；A，需关系序列化；成本高。** 主要增量：文本条件加图关系、实体约束、多跳关系连接。

**数据事实：**STaRK 包括 Amazon、MAG、Prime。MAG 官方统计有 84 道人工题、13,323 道合成题，约 187 万实体；人工题平均答案数 3.26。Amazon 的人工题平均答案数 19.50，明显超过当前 10-ID 上限，因此不宜把它作为首个完整召回任务。实体数量不等于可返回目标文档数量。 [官方总览](https://stark.stanford.edu/datasets.html)；[MAG 数据与统计](https://stark.stanford.edu/dataset_mag.html)；[Amazon 数据与统计](https://stark.stanford.edu/dataset_amazon.html)；[官方实现](https://github.com/snap-stanford/stark)

**接入建议：**优先 MAG 人工题，目标论文作为可返回实体文档，作者/机构/引用边以统一、query-independent 的文本记录供查阅。若使用闭合子图，需保留支持路径和同类型干扰实体；仅保留 gold 论文会消除真正的关系搜索。先报告人工题与合成题分项，再扩域。

**主要限制：**当前输入仅复制 title/abstract 会丢掉该基准最重要的关系条件。不同关系路径可能都合法；实体类型、边方向和时间快照必须冻结。≤10 答案的过滤属于本文建议子集，不能代表完整 Amazon。

**许可与获取：**官方任务页面标示 CC-BY-4.0；KG 组成数据的来源许可仍分别追踪。

### 4.14 MultiDoc2Dial

**推荐指数 84/100；B；成本中。** 主要增量：对话省略、指代消解、持续上下文下切换证据文档。

**数据事实：**官方数据提供文档、对话，以及 turn 的 doc_id/id_sp 引用；document_domain 包含 488 个文档。完整历史对话与下一回复的监督是不同字段，可据此提取本轮证据。 [官方仓库](https://github.com/IBM/multidoc2dial)；[官方数据卡](https://huggingface.co/datasets/IBM/multidoc2dial)；[论文](https://arxiv.org/abs/2109.12595)

**接入建议：**保留截至当前用户轮次的对话作为 query，隐藏本轮及之后的系统答案和 references；将被引用证据 span 映射到统一片段或父文档 qrels。首选真正发生跨文档切换的轮次，并保留普通轮次作对照。

**主要限制：**把各轮随机拆分会泄漏同一对话；跨 split 应按 dialogue/document family 分组。只检索父文档可能过粗；语料总量仍较小，适合作为对话维度而非大规模检索压力测试。HF 卡不同配置的题数单位不同，应以选定配置锁定后的 turn 数为准。

**许可与获取：**官方 HF 卡标示 Apache-2.0；源文档来源一并保留。

### 4.15 LegalBench-RAG

**推荐指数 84/100；B；跨度为C；成本中。** 主要增量：完整法律文件中的精确条款定位与易混淆合同检索。

**数据事实：**官方提供四源整合的精确文件路径与字符跨度监督。mini 在论文表中为 776 queries、72 docs，四源各 194 题。完整集摘要和统计表的查询总数存在 6,858/6,889 差异，正式接入应以锁定发布文件实际计数为准，不混用论文总数。 [官方仓库](https://github.com/zeroentropy-ai/legalbenchrag)；[论文与统计表](https://arxiv.org/html/2408.10343)

**接入建议：**先用 mini 验证 span→统一段落/页/条款 ID 的映射，随后扩大语料。切分器按文档自身结构统一运行，host 端再将 gold spans 映射到检索单元。父文档 ID 版可低成本接入，但必须同时列出粒度退化；若直接评字符覆盖，需新增 scorer。

**主要限制：**原任务以单个来源文档内的精确片段为目标，并不天然提供跨文档推理。长法律文件需检查 1MB 上限，不能以开头截断代替完整文本。不要只把有 gold 的片段放入语料，也不要把文档模板识别当条款理解。

**许可与获取：**代码 MIT；CUAD、ContractNLI、MAUD、PrivacyQA 各自数据政策分别保留。

### 4.16 MIRACL

**推荐指数 83/100；A；成本中至高。** 主要增量：非英语检索、不同文字系统和形态、母语人工相关性。

**数据事实：**MIRACL 为 18 种语言的多语言检索基准；query 与检索语料通常同语种，不能称为原生跨语言查询。官方中文 dev 有 393 queries、3,928 judgments，中文全语料约 493 万 passages。 [官方仓库与 split](https://github.com/project-miracl/miracl)；[论文](https://aclanthology.org/2023.tacl-1.63/)

**接入建议：**优先公开 dev 的中文与两种互补语言；选择 Arabic/Thai 等检验分词和形态差异，Swahili 检验较低资源语言。固定每语言 corpus 及 qrels；若为预算裁剪到中等库规模，应标明 DCI 派生池，并附全库候选分布与采样规则。

**主要限制：**judgments 不是 positive 数，也不是对全部语料的穷尽判断。各语言题目和语料规模不同，不能仅按 query 混合平均，也不能把多语同语检索冒充跨语检索。

**许可与获取：**代码与 Wikipedia 文本许可分开；保留 Wikimedia 来源与相关署名/共享规则。

### 4.17 FollowIR

**推荐指数 83/100；A；配对评分需扩展；成本中至高。** 主要增量：检索指令遵循、排除条件、同query不同相关性标准。

**数据事实：**FollowIR 由 TREC Robust04、Core17、News21 派生，修改指令查询分别为 52、20、32。原版关注指令变化导致的相关性/排名变化，而非只对一个固定 qrels 集算常规检索分数。 [作者仓库](https://github.com/orionw/FollowIR)；[论文](https://aclanthology.org/2025.naacl-long.597/)；[原始 Robust04 配置](https://huggingface.co/datasets/jhu-clsp/robust04-instructions)；[MTEB 封装](https://huggingface.co/datasets/jhu-clsp/robust04-instructions-mteb)

**接入建议：**把原始和修改后的指令各作为一个样本，保留 pair ID；相同语料、相同预算、相同查询主干下评价约束生效。现有 Recall/F1/nDCG 可保留，另按原定义或明确的 Top-10 适配定义增加 paired 指标。

**主要限制：**只采用 BEIR 风格 query/corpus/qrels 会丢掉 qrel_diff 与原任务目标。原 p-MRR 若需要当前 Top-10 之外的名次，不能在列表外擅自补排名。TREC 来源版权是实质接入条件；公开 HF packaging 不等于所有原始新闻语料均可自由再分发。

**许可与获取：**HF packaging 标示 MIT，不覆盖 NYT、Washington Post、TREC Disks 等原始内容的全部权利；训练合成集也不能替代评测集。

### 4.18 InstructIR

**推荐指数 83/100；A；配对评分需扩展；成本低至中。** 主要增量：同一搜索问题随用户意图改变正确文档、约束敏感性。

**数据事实：**作者用 GPT-4 为同一查询生成多个实例级用户指令，并相应改变目标文本。官方 HF 卡列出 9,906 条 queries、16,072 条 corpus 和 9,906 条 test qrels；公开 query text 将用户情境/指令与原始查询用 [SEP] 连接。它基于 MS MARCO，任务重点是用户需求改变，而非仅改写通用 retrieval task prompt。 [作者仓库](https://github.com/kaistAI/InstructIR)；[官方数据卡](https://huggingface.co/datasets/kaist-ai/InstructIR)；[论文](https://arxiv.org/abs/2402.14334)

**接入建议：**以完整 instruction + query 作为每题输入，保留原 query family 和 instruction variant 的 host 元数据；同组变体使用同一 corpus/预算。除 Recall/F1/nDCG 外，增加“原问题相同但意图变化时，返回是否对应改变”的配对成功率，并核查官方鲁棒性指标在 Top-10 输出下能否忠实计算。

**主要限制：**查询与目标文本有合成成分，风格匹配可能成为捷径；官方 qrels 行数与 query 数相等也不等于已穷尽所有相关文档。公开样例 ID 含变体后缀，应同样审计 query/corpus ID 的关联。按原 query family 切分和重采样，不能把多个变体当独立问题；与 FollowIR 相比，真实人工判定的约束变化可信度较弱，因此 G 分更低。

**许可与获取：**官方 HF 卡与代码标示 MIT，但来源 MS MARCO 内容和原网页的使用范围仍应分别核对；冻结发布版，不重生成合成指令。

### 4.19 CoIR

**推荐指数 81/100；A；成本低至中。** 主要增量：代码/自然语言跨表示检索、多轮开发上下文。

**数据事实：**CoIR 汇集 10 个代码检索数据集，采用与 BEIR/MTEB 相近的 corpus/queries/qrels 结构；总量约 200 万文档。官方列出 cosqa、codesearchnet、codesearchnet-ccr、codefeedback-mt 等任务。 [官方仓库与任务清单](https://github.com/CoIR-team/coir)；[官方数据组织](https://huggingface.co/CoIR-Retrieval)；[CosQA 数据结构](https://huggingface.co/datasets/CoIR-Retrieval/cosqa)

**接入建议：**先导入 CosQA 或单一编程语言的 CodeSearchNet，再用 CCR 测代码到代码匹配。按原 qrels 固定检索单位，保留源码排版，不执行被检索代码。与 RepoQA 的仓库内导航互补，不必第一批把所有 CoIR 任务铺开。

**主要限制：**CodeSearchNet 文档字符串与自然语言查询可能高度相似；人工或合成代码检索不是完整仓库依赖推理。多轮数据必须隐藏目标答案和未来轮次。不同子任务的原始许可不统一。

**许可与获取：**CoIR 代码 Apache-2.0；各原始数据集与代码片段许可单独审计，不能套用框架许可。

### 4.20 ViDoRe V3 文本派生版

**推荐指数 81/100；B；成本中至高。** 主要增量：企业长文档页面导航、表格文本、跨页证据与跨语言查询。

**数据事实：**V3 官方有 10 个领域，其中 8 个公开、2 个私有；总计约 26,000 页、3,099 道原始查询并有六语版本，不能将总量当成公开子集规模。HR 数据卡提供 1,110 页、markdown、page_number_in_doc 和 page-level qrels；318 道原始问题展开为 1,908 条六语查询。 [官方 V3 说明与公开规模](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3)；[HR 数据卡与许可](https://huggingface.co/datasets/vidore/vidore_v3_hr)

**接入建议：**可以复用上游已给出的 markdown，先审计 text-sufficient 问题，再建立 page ID 检索版。依据证据实际可见性筛选，而非仅用 query_type=extractive 猜测。保留文档/页面映射，分开报告单语和跨语条件。

**主要限制：**原始标签来自视觉页面：OCR 不含图表/示意图时任务可能根本不可解。文本派生结果不能冒充原 ViDoRe 视觉结果；不同语言翻译题高度相关，应按原始 query 分组采样。当前 qrels 的 1/2 分级与完整答案、必需局部信息有关，需保留其语义。

**许可与获取：**标注 CC-BY-4.0；原文档及 markdown 继承出版方许可，逐文档见 documents_metadata.license。

### 4.21 ViDoRe V3 视觉原版

**推荐指数 81/100；C；成本高。** 主要增量：搜索后查看页面图像、表格与图示取证、跨页多模态阅读。

**数据事实：**官方 V3 提供 page-level qrels、参考答案与 bounding boxes；公开与私有领域分开。HR/财报等子集具有结构化页面元数据，适合冻结成本地视觉语料。 [官方 V3 说明](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3)；[公开数据集合](https://huggingface.co/collections/vidore/vidore-benchmark-v3)

**接入建议：**增加有界的本地页面图像查看工具及视觉输入预算，保留按页返回 ID；若评估 bounding-box 取证或最终答案，则另定义输出与指标。视觉轨道应单独排行榜，不能要求纯文本模型遵循不可能完成的同一工具契约。

**主要限制：**当前 read 只返回 UTF-8 文本，不能测图像能力；仅提供 OCR 会改变题目可解性。必须控制渲染版本、分辨率、页面编号和视觉 token 预算。

**许可与获取：**标注 CC-BY-4.0，源文档分项许可；两个私有域不在本地可接入范围。

### 4.22 SciFact / SciFact-Open

**推荐指数 81/100；A或B，取决于封装；成本低；Open为高。** 主要增量：专业科学表述、支持/反驳证据、摘要级精确阅读。

**数据事实：**原始 SciFact corpus 有 5,183 篇科学摘要，提供 claim、证据文档和证据句。公开 HF 中一条 claim 可对应展开的证据记录，不能把 viewer rows 直接当独立 claim 数。SciFact-Open 扩到约 50 万摘要，并提供较小的候选语料版本。 [SciFact 官方仓库](https://github.com/allenai/scifact)；[数据卡](https://huggingface.co/datasets/allenai/scifact)；[SciFact-Open 数据说明](https://github.com/dwadden/scifact-open/blob/main/doc/data.md)

**接入建议：**先用原始小库；使用 BEIR qrels 封装为 A，自行从 evidence_doc_id 聚合为 B。支持与反驳文献都可作为取证相关文档；若要判断 claim 真伪则新增 verdict scorer。NEI/无证据记录不纳入当前非空检索主分。

**主要限制：**cited_doc_ids 不等于全部存在可验证 evidence 的文档。多个 rationale/evidence set 可能是替代关系，不能一律要求召回其并集。Open 的池化标注更稀疏，不能视为原始小库简单等比例放大。

**许可与获取：**原始官方 HF 数据卡 CC-BY-NC-2.0；Open 的 S2ORC/论文文本来源需单独追踪。

### 4.23 MLDR

**推荐指数 79/100；A；成本中。** 主要增量：长文档与多语言组合、文档内部搜索。

**数据事实：**官方覆盖 13 种语言；中文与英文 test 各 800 题、各 200,000 篇候选文档；各语言整体发布规模不相同。查询通过 LLM 从长文档构造，提供相关文档 ID。 [官方数据卡与规模表](https://huggingface.co/datasets/Shitao/MLDR)；[官方任务说明](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/C_MTEB/MLDR/README.md)

**接入建议：**保留长文原貌，按语言、文档长度和 lexical overlap 分层。可对比固定 read 预算下的长文与段落语料，前者作为主任务，后者显式标记为粒度消融。

**主要限制：**不能把 LLM 生成问题的正例自动看作完整相关集合；从单篇长文生成的问题可能具有可利用的实体/表述捷径。与 LMEB Semantic/MLDR 属于同源，主总分只计一次。

**许可与获取：**HF 卡标示 MIT；Wikipedia、mC4、Wudao 等原始来源仍逐项保留。

### 4.24 FinQA / ConvFinQA / TAT-QA

**推荐指数 79/100；B；计算为C；成本中至高。** 主要增量：表格与正文联合取证、多轮财务约束与算术证据。

**数据事实：**FinQA 有 8,281 个专家 QA；ConvFinQA 有 3,892 段对话/14,115 个问题；TAT-QA 有 16,552 个问题/2,757 个表文混合 contexts。它们原本通常提供相关 report/context，目标包含程序、数值、单位或多轮答案，而非全库搜索。 [FinQA](https://github.com/czyssrs/FinQA)；[ConvFinQA](https://github.com/czyssrs/ConvFinQA)；[TAT-QA](https://github.com/NExTplusplus/TAT-QA)

**接入建议：**FinQA 用 gold_inds 对齐文本/表格行；ConvFinQA 只保留当前轮之前的对话；TAT-QA 保留表头、单位和跨行结构。分别建立固定页面/表文单元集合，用证据转换得到 qrels；需将“检索版本”与原版答案/执行指标分开。

**主要限制：**直接返回父报告会掩盖表格理解；自动折行不能破坏列对齐。table cell、paragraph 和 page 是不同粒度，不应混成同一 doc_id 类型。原始公式、program、exe_ans 不能流入语料。

**许可与获取：**各代码仓库许可与源报表分别核查；TAT-QA 数据声明 CC-BY-4.0，不能据此概括三者全部数据。

### 4.25 LitSearch

**推荐指数 78/100；A；成本中。** 主要增量：模糊研究需求、术语改写、文献引用与方法定位。

**数据事实：**官方数据包含 597 道 ML/NLP 文献查询和 64,183 篇论文记录；查询有 gold paper IDs、specificity/quality 与来源元数据。查询一部分来自引用上下文生成，另一部分由论文作者撰写，并经专家检查。corpus_clean 提供标题、摘要、出边引用，corpus_s2orc 提供更丰富元数据。 [官方数据卡](https://huggingface.co/datasets/princeton-nlp/LitSearch)；[官方仓库](https://github.com/princeton-nlp/LitSearch)

**接入建议：**优先作者撰写且不直接暴露标题的查询；分别评价仅标题摘要和保留引用元数据的语料版本。如要测全文导航，应先逐条核实 corpus_s2orc 的可用全文覆盖，而非默认每篇都有完整正文。

**主要限制：**引用目标或作者提名论文不等于对该研究需求的穷尽相关集合；把其他相关论文记成无关会低估精度。若与 BRIGHT/SciFact 共用论文，需要做源文档去重与能力分项。

**许可与获取：**本次读取的数据卡未明确统一数据许可；需核查 S2ORC 来源与论文文本的分发范围。

### 4.26 QMSum

**推荐指数 78/100；B；总结为C；成本中。** 主要增量：长会议记录定位、多人发言、离散证据聚合。

**数据事实：**官方数据有 232 场会议、1,808 个 query-summary 对。specific queries 带 relevant_text_span；general query 对应整场会议，没有同样的局部证据标签。 [官方仓库与字段示例](https://github.com/Yale-LILY/QMSum)；[论文](https://arxiv.org/abs/2104.05938)

**接入建议：**将 transcript 按统一轮次/连续时间窗口切成带稳定 ID 的片段，将证据区间映射为 qrels。保留 speaker 和轮次信息，优先跨远距离 span 的特定问题。summary 只留 host；同一会议的全部问题进入同一 split。

**主要限制：**整场会议作为一个文档会退化为找会议而非找证据；按标注边界切块会泄漏答案位置。泛化总结题需答案轨道，不宜伪造局部 qrels；证据覆盖超过 10 片段时需要单列。

**许可与获取：**仓库与所汇集会议来源分别核查；本文不把公开可下载等同于统一再分发授权。

### 4.27 C-MTEB 中文检索

**推荐指数 78/100；A；成本低至中。** 主要增量：中文真实搜索与不同查询表达，建立稳定中文基线。

**数据事实：**官方任务表列 T2Retrieval test 24,832 queries、DuRetrieval test 4,000、MMarcoRetrieval test 7,437。三者采用 passage 检索形式；MMarcoRetrieval 继承 MS MARCO 的机器翻译来源，与中文原生搜索数据不同。 [官方任务说明](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/C_MTEB/README.md)；[DuRetrieval](https://huggingface.co/datasets/C-MTEB/DuRetrieval)；[T2Retrieval](https://huggingface.co/datasets/C-MTEB/T2Retrieval)；[MMarcoRetrieval](https://huggingface.co/datasets/C-MTEB/MMarcoRetrieval)

**接入建议：**先选 DuRetrieval 与 T2Retrieval 的固定 50–100 题诊断样本，保持相应原生 corpus/qrels；再按规模扩展。样本数是本文建议，不是官方 split。MMarco 只作为翻译来源对照，避免中文任务全部来自同一英文母集。

**主要限制：**稀疏正例不支持“其他文档都无关”的强解释；网页噪声、重复段落和 query overlap 需查。MTEB task 版本与 HF config/split 必须一起锁定，不能依赖 viewer 总行数推断 corpus 规模。

**许可与获取：**分别核查三个数据源；C-MTEB 框架许可不是全部源数据许可。

### 4.28 QASPER

**推荐指数 78/100；B；视觉/答案为C；成本中。** 主要增量：完整科学论文内部导航、远距离段落证据。

**数据事实：**QASPER 提供 5,049 道问题、1,585 篇 NLP 论文；query 作者主要先看标题和摘要，另由回答者标注正文证据。evidence 文本对应完整段落，highlighted_evidence 对应选中的句子；图表证据有 FLOAT SELECTED 标记。 [官方数据卡与字段](https://huggingface.co/datasets/allenai/qasper)；[论文](https://arxiv.org/abs/2105.03011)；[LMEB 许可汇总](https://huggingface.co/datasets/KaLM-Embedding/LMEB)

**接入建议：**按全论文统一结构建立段落/节片段 ID，先选择可由正文证据回答的 validation 问题；保留多位回答者提供的替代证据集。对比单篇论文内导航与多论文池两个明确命名的条件。

**主要限制：**单篇论文作为唯一可返回 doc 会让任务失去段落定位意义；把多篇论文合并又会引入新的相关性漏标。表图证据不能靠删除标记假装已被 OCR 完整保留。答案与高亮证据全部留在 host。

**许可与获取：**当前 allenai/qasper 官方 HF 卡标 CC-BY-4.0；LMEB 汇总表对 QASPER 写为 CC-BY-NC-4.0，存在声明不一致。实际选用哪一份发布包就锁定其声明并核查源论文，不混用许可推断。

### 4.29 LoCoMo

**推荐指数 78/100；B；成本中。** 主要增量：长会话时间线、人物与事件关联、细粒度对话证据。

**数据事实：**官方 locomo10 发布 10 组长期对话，QA 中有证据 dialogue IDs，conversation 带 speaker、dia_id、文本和时间信息。其多模态设定与可公开获得的图片资产并不完全相同。 [官方仓库与数据](https://github.com/snap-research/locomo)

**接入建议：**按原对话划分场景，以 turn 或 session 为固定 ID 单元，映射 evidence 列表；选择有明确可对齐文本证据的问题。保留未涉及当前答案的历史和合法干扰，对话级分组隔离评测。

**主要限制：**只有 10 组对话，query 数很多也不等于独立场景很多；报告置信区间时按对话聚类。开放域/对抗题若无本地充分证据，不得自动补 qrels。与 LMEB Dialogue/LoCoMo 不重复计入总分。

**许可与获取：**官方发布 CC-BY-NC-4.0。

### 4.30 FinanceBench

**推荐指数 77/100；B；数值答案为C；成本中。** 主要增量：真实公司财报、年份/公司消歧、证据页定位。

**数据事实：**公开仓库提供 150 道带答案、证据字符串和文档信息的样本；完整 benchmark 所称 10,231 题不等于本地可下载的公开评测集。文档有财报类型和年度信息，适合检查同公司不同年份的干扰。 [官方仓库与公开范围](https://github.com/patronus-ai/financebench)；[论文](https://arxiv.org/abs/2311.11944)

**接入建议：**建议以页面或稳定段落为 doc 单位，利用 evidence_doc_name、证据页和证据文本做映射；保留同公司不同年度和相似指标的文档作为干扰。答案、justification 和 gold evidence 留在 host。

**主要限制：**检索出证据页只能证明取证，不证明财务计算正确。若只按整份年报打分，容易退化为公司/年份匹配。个别 PDF 的页码、抽取顺序与文档体积需要核对；公开题量较小，不宜过度调 prompt。

**许可与获取：**本次核查未确认统一、明确的数据再分发许可；公开样本、仓库代码与各财报的权利边界分开处理。

### 4.31 HotpotQA

**推荐指数 77/100；B；成本中至高。** 主要增量：桥接与比较、多页面支持证据、开放候选库搜索。

**数据事实：**官方 dev 有 7,405 题，supporting_facts 用页面标题和句子位置标记证据。distractor 设置每题给定 10 个候选段落；fullwiki 检索面向完整知识语料。 [官方项目与下载](https://hotpotqa.github.io/)；[作者仓库](https://github.com/hotpotqa/hotpot)

**接入建议：**用匹配版本的 Wikipedia 源库与支持页面映射，按桥接/比较和文本长度分层。若采用 BEIR HotpotQA，应直接声明是该固定检索封装；如果合并原 dev contexts 成池，则明确是自建派生语料。

**主要限制：**fullwiki dev 文件中已有的 top-10 contexts 不是完整语料，gold 不一定都在其中。distractor 的 2 个支持段落加 8 个干扰段落不代表开放检索。原始 test 不提供同等公开 gold，主机本地评分优先 dev。

**许可与获取：**数据 CC-BY-SA-4.0，代码与源 Wikipedia 许可分别保留。

### 4.32 FEVEROUS

**推荐指数 76/100；B；表格/判定为C；成本高。** 主要增量：表格单元与正文联合证据、数值与实体消歧。

**数据事实：**FEVEROUS 有约 87,026 个 claims，证据覆盖 Wikipedia 句子、表格单元、表头、caption 等元素；公开标签包含支持、反驳和证据不足。 [官方数据页](https://fever.ai/dataset/feverous.html)；[官方实现](https://github.com/Raldir/FEVEROUS)

**接入建议：**保留页面结构和表格坐标，定义固定 text block/table block 检索单位；可先做有证据的页面级 B 版，但表格证据覆盖需另加结构化评分。不要只抽出标注单元供搜索。

**主要限制：**整页命中无法证明找到正确表格/行列，普通 Markdown 扁平化会改变可解性。NEI 不可按常规正例处理，多个证据集合的替代关系必须保留。

**许可与获取：**按官方数据页及 Wikipedia/来源页面条款处理，代码许可不替代内容许可。

### 4.33 SWE-Explore-Bench

**推荐指数 75/100；B；行区域为C；成本高。** 主要增量：修复前探索、必需与可选代码上下文、行预算下的排名。

**数据事实：**官方报告 848 个 issue、203 个仓库、10 种编程语言。gold 从多条成功修复轨迹提炼，区分 core/optional 文件与行区域；它不是单纯以补丁涉及文件为标签。 [官方仓库](https://github.com/Qiushao-E/SWE-Explore-Bench)；[数据卡与许可](https://huggingface.co/datasets/SWE-Explore-Bench/SWE-Explore-Bench)

**接入建议：**先做只读源码探索，不启动修复/测试容器；可投影 core 文件为 qrels，并单独报告 optional 命中。若保留其核心贡献，应新增返回 regions 或读取轨迹的行级评分，不能只做粗文件命中。

**主要限制：**成功轨迹标签带有产生轨迹的模型/策略偏差，未访问但同样有用的区域可能未标注；不能与人工 gold 的 ContextBench 混同。主 benchmark 仍有许可约束，因此优先级低于 ContextBench/RepoQA。

**许可与获取：**代码 MIT；HF 数据卡标示 CC-BY-NC-ND-4.0。派生标注或改造包的公开分发不能按代码 MIT 处理，必须先明确授权范围。

### 4.34 TREC iKAT

**推荐指数 75/100；C；固定turn可派生A/B；成本很高。** 主要增量：用户画像、会话目标演化、个性化检索与引用。

**数据事实：**NIST 发布 passage qrels；原任务还包含 PTKB/用户信息、对话历史和回答。原语料基于 ClueWeb22-B，规模与授权要求显著高于本项目当前任务。 [NIST iKAT 数据](https://trec.nist.gov/data/ikat.html)；[2023 数据](https://trec.nist.gov/data/ikat2023.html)；[官方资源](https://www.trecikat.com/data/)

**接入建议：**只有取得匹配语料授权并冻结可用 passage 集之后，才能派生“给定截至本轮历史→相关 passage IDs”的检索任务。原始 persona、历史截止点、响应与引用评分应单独定义；这不是仅加一份 qrels 就能忠实复现的完整基准。

**主要限制：**ClueWeb 许可与大语料是实际阻碍；2023、后续交互式 track 不能混称同一协议。过度简化会只剩普通对话检索，失去个性化核心。

**许可与获取：**ClueWeb22-B 需相应许可；未获语料访问权限之前不列为首批可直接落地项。

### 4.35 M3DocVQA

**推荐指数 74/100；C；成本高。** 主要增量：开放文档库中的跨页/跨文档视觉证据。

**数据事实：**M3DocRAG 官方工作同时发布 M3DocVQA，目标是 3,000+ PDF、40,000+ 页上的开放域 DocVQA。应区分数据集 M3DocVQA 与方法 M3DocRAG。 [官方仓库](https://github.com/bloomberg/m3docrag)；[论文](https://arxiv.org/abs/2411.04952)

**接入建议：**冻结 PDF 与 page ID 映射，使用自有 DCI 搜索/页面阅读工具；原方法 ColPali 的检索输出不能预先提供给 DCI 模型。先审计所选 split 可用的证据粒度，再决定页面检索与答案指标。

**主要限制：**需要视觉工具、PDF 处理和更高预算，数据级证据完整性尚需验证；若只做 OCR，其分数只能标为文字转换派生版。与 ViDoRe 同时首批接入的边际收益不高。

**许可与获取：**官方代码和 PDF 来源的授权分别核查；本次未确认可一概覆盖全部 PDF 的统一再分发许可。

### 4.36 BEIR 补充子集

**推荐指数 73/100；A；成本低至中。** 主要增量：反论点、专业词汇、多样表述；成熟低成本对照。

**数据事实：**BEIR 提供统一检索数据格式。ArguAna 是反论点检索，NFCorpus 是医学/营养信息检索，CQADupStack 是社区重复问题检索。官方表中 ArguAna 约 8.67k 文档，NFCorpus 约 3.6k 文档，适合作为中小语料基线。 [BEIR 官方数据表与实现](https://github.com/beir-cellar/beir)；[官方数据卡](https://huggingface.co/datasets/BeIR/webis-touche2020)

**接入建议：**优先 ArguAna 测“同主题但立场不符”的困难负例；NFCorpus 用于领域词汇对照。TREC-COVID、Touché-2020 可作大正例集合压力测试，但不建议取代前面的新能力任务。

**主要限制：**多为已有成熟检索题型，对交互式搜索的新增价值有限。NFCorpus、Touché、TREC-COVID 存在较多正例，必须报告 Top-10 召回上限；不要把不同 CQADupStack 社区直接合并为无标注跨社区负例。

**许可与获取：**BEIR 框架许可不代表所有数据集；按各任务官方来源和原语料许可执行。

### 4.37 AIR-Bench

**推荐指数 73/100；A，仅公开dev；成本中。** 主要增量：可固定的行业/语言组合和较大语料压力。

**数据事实：**官方公开 dev labels，test labels 隐藏。任务表中 Finance-English-dev 有 26,266 corpus entries/317 queries；Finance-Chinese-dev 有 1,014,974 entries/361 queries。问题与相关性主要由自动化 LLM 管线生成。 [官方仓库与许可](https://github.com/AIR-Bench/AIR-Bench)；[任务规模表](https://github.com/AIR-Bench/AIR-Bench/blob/main/docs/available_tasks.md)

**接入建议：**先选英文金融公开 dev，若需中文大库再引入中文金融。保持官方任务名/版本；只对有公开 qrels 的部分做 host-side 评分。test 可作为外部排行榜对照来源，不作为本地完整可评分数据集。

**主要限制：**自动相关性标签的覆盖、难负例来源和标注模型偏差都需要样本审核；Finance 域本身并不保证具备财务推理。百万级中文库的成本需在抽样审计后决定。

**许可与获取：**代码 MIT，测试数据有 CC-BY-NC-SA-4.0 及用途条款；dev 的所用文件也需核对相应适用声明，不能默认只受代码许可约束。

### 4.38 LongEmbed / LoCo

**推荐指数 72/100；A或B，按所用封装；成本中。** 主要增量：长文档长度控制、多领域语料、长输入处理对照。

**数据事实：**LongEmbed 和 LoCo 是长文本检索资源；LoCo 后续版本扩大法律、医学、科学、金融等覆盖。这里应区分原生长文检索、从 QA 派生的 retrieval，以及合成 needle 测试，不把它们混成一个指标。 [LongEmbed 官方仓库](https://github.com/dwzhu-pku/longembed)；[LoCo 官方发布说明](https://hazyresearch.stanford.edu/blog/2024-05-20-m2-bert-retrieval)

**接入建议：**选有公开 corpus/queries/qrels 的真实长文任务，保留完整文档并设置固定文件布局；作为相同语料 token 量下的长短文对照。与 QASPER、MLDR、LMEB Semantic 重叠的源数据按 family 合并权重。

**主要限制：**长文本 embedding 难度不自动等于交互搜索难度；只需整篇相关性判断的任务可能无法评价阅读过程。必须锁定 LoCoV0/V1 和具体 task，不能使用模糊的 LoCo 总分。

**许可与获取：**框架和底层任务分别核查；不宣称所有长文档来源有统一许可。

### 4.39 LongMemEval-V2

**推荐指数 71/100；C；成本高。** 主要增量：环境状态、工作流程知识、任务经验与时间变化。

**数据事实：**V2 于 2026-05 首次公开，面向 agent 轨迹长期记忆；公开数据含文本 observations、轨迹映射及 screenshots，覆盖状态、流程等问题。发布时移除了构造过程中的 answer-bearing 标注和部分 provenance，不能假设具有可直接使用的检索 qrels。 [官方仓库](https://github.com/xiaowu0162/LongMemEval-V2)；[官方数据卡](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)；[论文与首次日期](https://arxiv.org/abs/2605.12493)

**接入建议：**优先作为新轨道设计参考：定义历史摄入、查询时的可见 memory、答案评分、状态时点与图像预算。若只能做文本轨迹检索，需补证据标注，单独命名为派生版。

**主要限制：**给出检索 ID 无法评估状态更新是否正确；移除图像或 construction labels 后能否确定充分证据，须数据级审计。它不能替代更易接入的 V1/LMEB 检索版。

**许可与获取：**官方发布 Apache-2.0；源轨迹/截图条款随 revision 记录。

### 4.40 KILT

**推荐指数 70/100；B；成本高。** 主要增量：统一知识快照、可核验出处、多种知识密集任务。

**数据事实：**KILT 将多个任务映射到统一 Wikipedia 知识源，官方示例列 5,903,530 个页面；output 可包含 answer 与 provenance 的 wikipedia_id、段落和字符位置。test 答案不公开，不能直接使用本地主机 scorer 完成标准 test。 [官方仓库与 provenance 说明](https://github.com/facebookresearch/KILT)；[论文](https://arxiv.org/abs/2009.02252)

**接入建议：**选公开 dev 中 provenance 完整的任务，严格使用匹配知识快照。可做文档级或段落级 qrels，但必须固定映射与不完整 evidence 的处理；适合作为统一来源层，而不必把已选 HotpotQA、FEVER 再计一次任务。

**主要限制：**旧任务到统一快照的映射会发生漂移或不精确；将整个知识源变成数百万小文件工程成本较高。只有 answer 的记录不能用字符串包含关系伪造 gold。

**许可与获取：**各来源任务与 Wikipedia 的许可分别追踪；不把 KILT 代码许可概括为全体数据许可。

### 4.41 Mr.TyDi / mMARCO

**推荐指数 69/100；A，选带qrels版本；成本中。** 主要增量：低资源语种基线与机器翻译来源消融。

**数据事实：**Mr.TyDi 提供 11 种语言的 Wikipedia passage retrieval；它不含中文，且通常是同语种检索。mMARCO 是 MS MARCO 的多语言机器翻译版本，原始训练 triples 与正式评测 qrels 不是同一对象。 [Mr.TyDi 官方仓库](https://github.com/castorini/mr.tydi)；[mMARCO 官方仓库](https://github.com/unicamp-dl/mMARCO)；[mMARCO 数据卡](https://huggingface.co/datasets/unicamp-dl/mmarco)

**接入建议：**MIRACL 已覆盖相近维度，故本家族是候补。若增加 Mr.TyDi，优先与 MIRACL 不重复的语言/查询来源；若增加中文 mMARCO，可直接选上一项已核对的 C-MTEB MMarcoRetrieval 封装。

**主要限制：**不能把正负训练对当成完整 test 相关性集合；跨语言翻译题、原英文题要按 query family 去重。二者的“多语言”都不自动等于跨语言搜索。

**许可与获取：**保留 Wikipedia/原 MS MARCO 与翻译派生条款；框架 Apache/MIT 不替代数据许可。

### 4.42 2WikiMultiHopQA

**推荐指数 69/100；B；成本中。** 主要增量：结构化关系链、桥接比较、文本与实体映射。

**数据事实：**官方 dev 有 12,576 个问题，类型包括 comparison、inference、compositional 和 bridge-comparison；supporting_facts 与 Wikidata evidences 提供不同层次的监督。 [官方仓库](https://github.com/Alab-NII/2wikimultihop)；[论文](https://arxiv.org/abs/2011.01060)

**接入建议：**先映射支持页面/段落为 qrels，保留关系路径作 host-only 辅助分析。原候选池与合并大池分别报告；与 MuSiQue 共存时按来源和问题类型去重，不让同类 Wikipedia 多跳任务占据大部分总分。

**主要限制：**模板化与 KG 构造可能带来捷径，文本页面命中不代表关系推理正确。原始 10 段 distractor context 不能叫完整开放检索；test 的隐藏监督不能按 dev 同等处理。

**许可与获取：**本次核查未确认覆盖整个派生数据集的统一许可；Wikipedia/Wikidata 与作者数据发布条款分别处理。

### 4.43 CUAD / ContractNLI 原始数据

**推荐指数 67/100；B；NLI为C；成本中。** 主要增量：律师条款标签、法律蕴含与例外关系的专项诊断。

**数据事实：**CUAD 有 510 份合同、41 类条款及 13,000+ 标签。ContractNLI 有 607 份 NDA，围绕 17 个固定 hypotheses 提供蕴含、矛盾、未提及标签及证据；二者原任务均已给定合同。 [CUAD 官方项目](https://www.atticusprojectai.org/cuad/)；[ContractNLI 官方项目](https://stanfordnlp.github.io/contract-nli/)

**接入建议：**若采用 LegalBench-RAG，通常无需再独立纳入本家族。独立改造时必须把问题限定到具体事实或合同语境，构建语义明确的检索请求；不能把同一个通用 hypothesis 到处复制，却只把当前合同设为正例。

**主要限制：**跨合同使用通用条款问题会形成大量额外相关文档，原 QA/NLI 标签不足以定义新 corpus-wide qrels。NotMentioned 不能强制配一个正例；doc-level 检索也不能代替 entailment/contradiction 分类。

**许可与获取：**CUAD/ContractNLI 数据声明 CC-BY-4.0；保留各下载页面附加条件和源合同信息。

### 4.44 FEVER

**推荐指数 67/100；B；verdict/NEI为C；成本中至高。** 主要增量：断言改写检索、反证寻找、替代证据集合。

**数据事实：**官方数据包含 185,445 个 claims，标签为 SUPPORTS、REFUTES、NOT ENOUGH INFO。证据以 Wikipedia 页面和句子组成，一个 claim 可以有多个可替代的充分证据集合。 [官方数据页](https://fever.ai/dataset/fever.html)；[官方实现](https://github.com/awslabs/fever)

**接入建议：**只将有可用证据的 claims 纳入当前检索轨道，按页面/句子映射 qrels；保留 evidence-set 结构做“命中任一充分集合”诊断。与 SciFact 相比，适合更大规模百科取证对照。

**主要限制：**把所有替代证据集合做并集后要求全召回会误罚正确路径。NEI 不意味着世界上不存在相关文档，只表示给定任务中证据不足。claim 的生成式陈述与真实用户问题分布不同。

**许可与获取：**遵循官方数据页对 Wikipedia 及来源内容的要求，不能从代码许可推定全语料再分发权。

### 4.45 MemoryBench

**推荐指数 63/100；C；成本高。** 主要增量：持续记忆写入/读取/更新、跨任务记忆行为。

**数据事实：**官方 MemoryBench 汇集多领域记忆任务，支持不同输入输出长度与 replay/on-policy 等设置；其目标包含交互过程中的 memory 状态和答案评估，而不是单次文档列表。 [官方仓库](https://github.com/THUIR/MemoryBench)

**接入建议：**作为后续 stateful-memory 轨道的协议参考；固定 episode、memory 初始状态、写入上限与用户历史可见性，再接入其具体任务。若只选其中 LoCoMo 证据，直接复用 LoCoMo/LMEB 的检索封装即可。

**主要限制：**把完整 MemoryBench 压成 ranked_doc_ids 会丢掉任务核心；不同记忆初始化/累积规则会产生不可比成绩。家族打包不意味着所有组件都有原生证据 qrels。

**许可与获取：**代码 MIT，组件许可分别核查。

### 4.46 RULER / BABILong

**推荐指数 63/100；B/C，诊断而非主榜；成本低至中。** 主要增量：可控长度、证据位置、噪声和搜索停止的因果诊断。

**数据事实：**RULER 提供可配置的合成长上下文任务；BABILong 将 bAbI 推理事实埋入长背景文本。它们可控制支持事实和干扰，但不等同于真实用户面对自然文档库的搜索。 [RULER 官方仓库](https://github.com/NVIDIA/RULER)；[BABILong 官方仓库](https://github.com/booydar/babilong)

**接入建议：**利用生成器保留 host-only 的 planted/support fact 位置，再按与 gold 无关的固定切块形成检索单元。改变证据位置、干扰量和文件分布，检验模型是否被工具输出/上下文预算限制。

**主要限制：**随机 key、合成事实可能让 rg 字面检索过强；若关键词已暴露，成绩主要反映命令构造。必须与真实数据主分分开，明确由生成器派生 qrels，并保留背景语料许可。

**许可与获取：**代码、bAbI 和背景文本分别核查；不将所有 PG-19 等背景文本概括为统一全球公版。

### 4.47 FinDER

**推荐指数 62/100；B，需先完成出处对齐；成本高。** 主要增量：较新的金融问法、缩写与定性/定量需求。

**数据事实：**官方 HF 提供约 5.7k query/answer/reference 条目，围绕 2024 年 SEC 10-K；目前数据配置为 train，没有现成官方 test。references 主要是证据段落文本，并非可直接套入 DCI 的稳定 corpus doc ID。 [官方数据卡](https://huggingface.co/datasets/Linq-AI-Research/FinDER)；[论文](https://arxiv.org/abs/2504.15800)

**接入建议：**只有在 reference→具体报告/页面/段落完成可靠对齐，并构建完整候选语料后，才转换为 B 类检索任务。需要公开自建 split、报告级隔离、原始文本匹配失败率；不能把 HF train 重命名就称为官方 test。

**主要限制：**出处不稳定、只有参考段落、切分泄漏与非穷尽标签增加接入成本；目前优先级低于 FinanceBench 和已有检索 qrels 的领域集合。

**许可与获取：**官方 HF 数据卡标示 CC-BY-NC-4.0；原始报告来源另行记录。

### 4.48 LongDA / LongProc

**推荐指数 61/100；C；成本高。** 主要增量：检索后数据整合、执行计算、程序化长任务。

**数据事实：**LongDA 面向长文档分析与跨材料计算；LongProc 面向长程序化任务。它们的目标输出包含分析结果、结构化产物或长流程结果，而非标准 doc-level relevance 集合。 [LongDA 官方仓库](https://github.com/Yiyang-Ian-Li/LongDA)；[LongDA 论文](https://arxiv.org/abs/2601.02598)；[LongProc 官方项目](https://princeton-pli.github.io/LongProc/)

**接入建议：**如果项目未来把“直接语料交互”扩到分析与执行，可先选择输入为可冻结文件、答案可确定性检查的任务，新增受控计算工具与产物评分；原 retrieval 主榜保持独立。

**主要限制：**没有原生完整 evidence qrels 时不能仅用相关文档命中率代替任务成功；程序执行、格式转换和规划难度会混入模型搜索能力。

**许可与获取：**需要按所选发布包与底层文档核查；现阶段不承诺统一可分发许可。

### 4.49 LongBench / LongBench v2 / InfiniteBench

**推荐指数 56/100；C；少量检索子题可派生；成本中至高。** 主要增量：单次长上下文能力与交互搜索的对照。

**数据事实：**LongBench v2 有 503 道长上下文多选题；InfiniteBench 包含超长上下文 QA、代码、数学和检索等任务。许多样本直接提供 context 与 answer/options，没有完整文档级 qrels。 [LongBench 官方仓库](https://github.com/THUDM/LongBench)；[InfiniteBench 官方仓库](https://github.com/OpenBMB/InfiniteBench)

**接入建议：**可为同一模型建立“全部上下文直接输入”的能力对照。若将 context 切为文件供搜索，须新增统一检索单位和证据标注，结果标为 DCI 派生任务，不能直接继承原多选准确率的解释。

**主要限制：**只有正确选项并不能定位全部必需文档；把 answer 字符串所在块当 gold 会遗漏推理证据并放入假正例。自然语言长文 QA 与合成 retrieval 子题应分项。

**许可与获取：**各子数据集来源许可独立，不能从基准代码许可概括所有长文档。

### 4.50 MultiFin

**推荐指数 40/100；C，暂不建议；成本高。** 主要增量：金融标题语言覆盖；对当前DCI新增价值有限。

**数据事实：**MultiFin 是约 10,048 个金融标题、15 种语言的多标签/多类分类数据，主要标签是金融主题，不是原始长文档的检索相关性。 [官方仓库](https://github.com/RasmusKaer/MultiFin)；[论文](https://aclanthology.org/2023.findings-eacl.66/)

**接入建议：**可用于分类或语言能力对照；如果人为改造成“找同主题标题”，必须自行建立检索语义与全库 gold，不能沿用分类标签后声称测到了直接语料取证。

**主要限制：**名称有 Fin 和多语言不代表它是金融文档检索；缺少天然 query→证据语料关系，因此不推荐进入 DCI 主轨。

**许可与获取：**作者仓库标示 CC-BY-NC-4.0。

## 5. 建议首批组合与后续扩展

总分排序用于比较单项；实际组合还要控制语言、来源与任务类型的覆盖。**首批建议选择下列 8 个家族，按任务分项报告，不直接把所有查询混为一个总分。** 每项先做 20 题左右的纯数据审计；下表的 50–100 题是后续预算申请时的建议评测规模，不是本次已执行的实验。

| 首批家族 | 建议起点 | 建议查询规模 | 首先解决的问题 |
|---|---|---:|---|
| BRIGHT / Bright-Pro | biology-short/long；Bright-Pro 同源题作 aspect 对照，再补 sustainable_living | 每子项 50，或该子项全量 | short/long 映射、长文件上限、Bright-Pro 的题目关联 ID 与 aspect gold 隔离 |
| BrowseComp-Plus | evidence 主标签 + gold 诊断，原始 100k corpus 为目标 | 分层 100 | 17 道超 10 证据题单列；小池与全库不能混报 |
| FreshStack | LangChain + Godot | 每主题 50 | 版本一致性、nugget 覆盖、合成判定的样本审核 |
| MuSiQue-Ans | dev 的 2/3/4-hop 平衡抽样 | 合计 90 | 全支持链、原候选池与 pooled corpus 区分 |
| LMEB | 一个 Dialogue 时间/多会话任务，加一个 Procedural 或 Episodic 任务 | 每项 50 | scene 候选域、上游许可、R_cap 与当前指标区别 |
| RepoQA | 五语言分层，固定仓库版本 | 合计 100 | 全仓库统一函数/区域 ID，代码格式与行号保真 |
| LeCaRDv2 | 官方 test 中按罪名、长度、正例数分层 | 80，后扩至 160 | 相关性阈值和 graded nDCG、Top-10 可达上限 |
| MIRACL | zh + 两种互补语言的 dev | 每语言 50 | 每语言候选规模、人工 judgments 的覆盖、多语言宏平均 |

首批结束后的补充顺序建议：

1. **指令与关系：**授权清楚的 FollowIR 子集、InstructIR 的同 query 多指令组、QUEST 的 ≤10 答案诊断及全量截断版、STaRK-MAG。它们更能揭示“查到了同主题内容但没有满足约束”的错误。
2. **企业与真实专业文档：**EnterpriseRAG-Bench、LegalBench-RAG、FinanceBench、FinQA/TAT-QA。前者补跨系统资料，后者补长文件和表格，但需要更仔细的证据粒度审计。
3. **更精确的代码上下文：**ContextBench；待数据许可与行级评分确认后，作为 RepoQA 的真实 issue 补充。SWE-Explore 的轨迹标签与许可限制决定其后置。
4. **视觉与有状态记忆：**ViDoRe V3、M3DocVQA、LongMemEval-V2/MemoryBench，分别建立新轨道。不能把视觉、生成答案、记忆更新能力强行压成当前文档 ID 分数。

如果只允许最快新增三项，选择 **BRIGHT 的两个小 long 子集/一个 short 对照、SciFact 的固定检索封装、C-MTEB DuRetrieval 的固定样本**，可先覆盖推理、专业证据与中文。如果更看重研究区分度而能接受接入成本，选择 **BrowseComp-Plus、MuSiQue-Ans、LMEB 的时间/多会话任务**。这两种组合是不同成本与目标下的建议，不改变总表单项分数。

## 6. 把候选变成可信 DCI 任务的共同要求

### 6.1 先确定检索对象与相关性含义

每个任务应在 manifest 中声明返回的是完整文档、网页、页面、固定段落、会话、函数还是实体。不能用“document”一个词掩盖粒度差异，也不能把查询涉及的唯一父文档作为全部正确答案后继续声称在测细粒度理解。

对于从 QA 派生的任务，需要分别保留：

- 原始回答和原始 evidence，仅 host 可见；
- evidence 到固定检索单位的映射，以及映射失败/一对多/多对一数量；
- 原 qrels、转换后的 qrels 和完整转换配置；
- 单个充分证据集、多个替代证据集，以及普通“相关文档集合”的区别。

如果按原文结构、固定 token/字符窗口或 AST 切分，切分必须对整个 corpus 使用同一规则，**不能先看 gold spans 再选择切块边界**。不同 query 若共享同一源文档，应复用同一切分结果。

### 6.2 不把稀疏标注误解成穷尽相关性

多数 IR qrels 是池化标注；QA supporting facts 更常是已知充分证据。模型找出未标注但确实有用的文档时，当前 scorer 会按非正例处理。因此：

- 主榜保留固定 qrels，称为“对已标注相关文档的 Recall/F1/nDCG”；
- 针对模型间分歧大的未标注返回结果，另抽样盲审并报告 judged coverage/漏标分析；
- 如确需扩充 gold，采用所有对比系统共享的池，统一盲审、冻结新数据版本，并重算全部系统；
- 不为某个模型单独修补 qrels，也不在同一排行榜运行期间动态改变答案集合。

合并 QA contexts 成新全局 corpus 尤其需要这一步。只保留每题 gold 与少量随机负例虽然容易得高分，但不能代表在自然语料分布中自主搜索。

### 6.3 Top-10 上限与完整集合必须分开解释

对一个含 `m>0` 个已知正例的 query，当前输出最多 10 篇，其理论最大召回是：

`RecallCeiling@10 = min(10, m) / m`

若只返回相关文档，当前变长列表 F1 的对应上限是：

`F1Ceiling@10 = 2 × min(10, m) / (m + min(10, m))`

应报告 `m` 的分布、`m>10` 占比和平均 RecallCeiling，保留现行原始 Recall/F1；可以额外提供 `hits/min(10,m)` 的容量归一化诊断，但**不能替换指标后仍沿用旧名称**。nDCG@10 的理想排序本来就截在 10，因此可以达到 1，并不意味着已召回全部相关文档。

这一点对 QUEST、LeCaRDv2、STaRK-Amazon、FreshStack/Bright-Pro 的多要点文档池、NFCorpus/TREC 类任务尤其重要。不能通过删除第 11 篇以后的 qrels 制造满召回，也不宜仅保留少正例问题而不披露分布变化。

本次对 BrowseComp-Plus 做了额外的低成本可复查检查：

| 公开文件 | 查询数 | 正例 query-doc 对 | 每题正例范围 | 超过 10 的查询 |
|---|---:|---:|---:|---:|
| qrel_evidence.txt | 830 | 5,064 | 1–13 | 17，约 2.05% |
| qrel_golds.txt | 830 | 2,407 | 1–9 | 0 |

读取到的原始文件 SHA-256：

- evidence：`a6f594975be57339de9e4e9f67f13c044f647feda77c0b84c45a1581e3041bd1`
- gold：`b875af4a745712bee7a94f464ed989232f8c77977c31824428470e11dcb28c73`

这是对公开小型标签文件的只读统计，没有下载对应网页语料。文件名与内容以该校验值快照为准；上游 README 中 `qrel_gold` 与实际 `qrel_golds` 的命名有差异，接入时应固定具体 commit。[官方 evidence 文件](https://raw.githubusercontent.com/texttron/BrowseComp-Plus/main/topics-qrels/qrel_evidence.txt)、[官方 gold 文件](https://raw.githubusercontent.com/texttron/BrowseComp-Plus/main/topics-qrels/qrel_golds.txt)

### 6.4 多跳任务增加“证据是否闭合”，保留替代路径

对单一必需支持集合，增加 **All-support@10**：返回集合是否覆盖全部必需支持文档。对 FEVER/QASPER 等有多个合法充分集合的情况，应定义为“至少覆盖一个完整合法证据集”，而非覆盖所有集合的并集。

文档相关性主分仍有用，但以下情况必须区分：

- 找到最终答案所在页，但漏掉中间实体或限制条件证据；
- 返回多篇重复描述同一要点的文档，但缺少另一必要要点；
- 找到一条有效替代推理链，却与某一标注者选择的链不同；
- 找到对立/过时证据，但无法作出正确结论。

FreshStack 可补 nugget coverage，Bright-Pro 可补加权 aspect coverage；BrowseComp-Plus 分别报告 evidence 与 gold；原始 MuSiQue 保留 hop 支持；SciFact/FEVER 保留证据集合。这些诊断都不能只从一张扁平二元 qrels 表事后猜测。

### 6.5 正确处理候选域、长度与文件布局

当前 DCI 的 workspace 主要按 task 共用，而 LMEB、LongMemEval、LoCoMo、RepoQA/ContextBench 等往往按 scene、问题或仓库限定候选域。接入时必须支持**与原任务相同的只读候选视图**，或通过等价任务分组实现；不能因为 loader 容易就把所有人的会话、不同版本代码或不同场景合成一个 corpus。

建议同时记录三个相互独立的规模维度：

| 维度 | 建议分层 | 用途 |
|---|---|---|
| 候选数量 | 原始小库、约 1k、5k/20k、原始完整库 | 测候选发现与全库扫描成本；规模仅为设计参考 |
| 文档长度 | 按冻结 tokenizer 的长度分位数，同时记录 UTF-8 bytes | 区分文档多和单篇长，观察分段读取与导航 |
| 证据复杂度 | 单证据、多证据、2/3/4-hop、时间冲突、条件过滤 | 避免大语料规模掩盖任务语义差异 |

工具一次只能返回 20,000 bytes，**不表示文档只能有 20,000 bytes**。`read(offset, limit)` 可以分段读取，单篇硬上限另为 1MB。优先保留原文档，通过分页阅读处理长文；只有确实要改变检索粒度或处理超硬上限文件时才做新的切分版本。

其他需要控制的因素：

- 统一保留标题、时间、来源和合法链接；不要只为正例添加更丰富元数据；
- 代码保留缩进和原始行号，表格保留行列/表头/单位，PDF 保留页码和读取顺序；
- 文件名保留源 ID；若原 ID 自带语义，应披露其影响。Bright-Pro 的 task-qid/extraction 命名尤其需要检查题目归属泄漏；若主版存在可利用捷径，应暂缓纳入，匿名重命名只能另做有双向映射的消融，不能覆盖原版；
- 子池固定 seed、选择规则、语料和 query ID 列表，保留所有已知必要证据；
- hard negatives 只能离线预构建并冻结，不能将 BM25/embedding 的预选排名悄悄作为模型运行时工具结果；
- 若引入 relation/graph 文本视图，序列化须独立于 query，子图必须保留必要连接与合理干扰。

### 6.6 无答案、答案生成和多模态分别建立协议

当前 schema 要求至少返回一个 ID，不能正确表达没有答案。LongMemEval abstention、MuSiQue-Full unanswerable、ContractNLI NotMentioned、FEVER NEI、EnterpriseRAG Info Not Found 不是同一种语义，也不能简单全部改成空 qrels。

建议今后独立定义：

| 轨道 | 输出/工具增量 | 适合数据 | 必须额外评价 |
|---|---|---|---|
| Retrieval 主轨 | 继续返回 1–10 IDs | 大多数 A/B 候选 | 已标注相关性、证据闭合、预算效率 |
| Answer + Evidence | 答案、引用及可选结构化推导 | FinanceBench、FinQA、QASPER、BrowseComp-Plus | 答案正确性、引用支持度、数值/单位正确性 |
| Abstention | 显式 abstain 与相应评分 | 原始基准中定义清楚的无答案题 | 可回答题误拒绝、不可回答题误作答、覆盖与风险 |
| Visual Corpus | 本地图像/页查看与视觉预算 | ViDoRe、M3DocVQA | 页面/区域证据，视觉 token 和渲染条件 |
| Stateful Memory | 历史摄入、写入/更新/读取规则 | LongMemEval-V2、MemoryBench | 更新正确性、旧事实干扰、跨回合成本 |
| Code/Analysis | 区域引用，必要时受控执行 | ContextBench、LongDA、LongProc | 行级定位、分析产物正确性、执行资源 |

这些轨道的汇总分不能和现有纯检索任务直接平均。先保留 Retrieval 作为稳定主榜，其余分别报告。

### 6.7 避免重复计权与数据泄漏

建议为每个任务维护 `source_family`、`original_dataset`、`source_doc_id`、`source_query_id`、`scene_id`、`snapshot_revision`。至少处理以下重叠：

| 同源/重叠家族 | 应如何计分 |
|---|---|
| BRIGHT 与 Bright-Pro | 同源 query 配对报告；corpus 和 gold 版本分开，主组合按一个源家族计权 |
| LMEB 与 LongMemEval/LoCoMo/QASPER/MLDR/SciFact | 可比较封装或候选域，主总分按源家族去重/等权 |
| LegalBench-RAG 与 CUAD/ContractNLI/MAUD/PrivacyQA | 主组合选择一个封装；原版只作专项消融 |
| HotpotQA/FEVER 与 KILT、BEIR 对应版本 | 显式标出知识快照和 qrels 映射；不重复扩大权重 |
| CoIR CodeSearchNet 与独立 CodeSearchNet 任务 | 同一 query/document family 去重 |
| C-MTEB MMarcoRetrieval 与 mMARCO/MS MARCO、InstructIR | 翻译题和指令变体按原 query family 分组，检查 query/corpus 重叠 |
| ViDoRe 六语 query 与原始 query | 按原始 query 分组；分开单语、跨语指标 |
| ContextBench/SWE-Explore 与原始 issue benchmarks | repo/issue/commit 级对齐；同源 issue 不跨开发/测试切分 |
| 长上下文套件中的 QASPER、HotpotQA、QMSum 等 | 来源家族相同，只改变呈现形式时视为同源条件 |

基准已长期公开，训练污染风险无法仅靠数据发布时间排除。应优先冻结版本、记录模型训练数据披露情况、保留 source-aware split，并增加必须依赖当前文件证据的诊断。不能因为数据较新就宣称模型从未见过，也不能仅靠改 ID 宣称完成去污染。

### 6.8 除检索分数外需要记录的过程指标

当前“至少一次非空 read”门槛不足以证明模型读过所有返回结果。建议在不改变主分的前提下记录：

- **返回文档的读取覆盖率**：返回的 IDs 中有多少对应非空 read receipt；按页/片段映射时明确覆盖定义；
- **必要证据读取覆盖率**：host 端对照 evidence 判断，严禁向模型反馈；
- **检索效率**：工具调用数、不同文档读取数、读取 bytes、重复读取比例、总输入/输出 tokens、P50/P95 wall time；
- **稳定性**：有效输出率、timeout/token_limit/context_overflow 等失败分布；
- **质量与成本关系**：在同一任务/语料/协议下报告 Recall 或 All-support 相对于 token/时间的曲线。

主结果以 **all-sample macro** 为准，失败按协议计零；`valid_only` 只作为诊断。语言、来源家族和能力维度另外宏平均，避免大题库或某个同源家族统治总分。若给置信区间，按源文档/会话/issue/query-pair 聚类重采样，不能把高度相关的问题视为完全独立样本。

## 7. 调研过但不宜直接充当当前主轨的资源

这些资源并非没有价值，问题在于它们缺少与当前输出和工具相匹配的独立检索定义。

| 资源 | 不直接纳入的原因 | 更合适的用法 |
|---|---|---|
| RAGBench | 约 100k RAG examples 主要给定检索上下文与模型回答，并标注支持/相关/利用等；给定上下文包不是一个具有完整 qrels 的开放候选 corpus | 做答案忠实性/裁判评估或证据判定诊断。[官方数据卡](https://huggingface.co/datasets/rungalileo/ragbench) |
| CRAG | 提供 QA、预先抓取的搜索结果和 mock KG/API；这些页面不保证相关，没有可直接把全部搜索结果当正例的依据 | 冻结工具环境后做 Answer + Evidence；不要把输入的 5/50 个页面误当全库检索。[官方数据说明](https://github.com/facebookresearch/CRAG/blob/main/docs/dataset.md) |
| RAR-b | 把常识/数学等推理题的答案作为检索对象，模型可能先凭参数知识解题再找答案字符串 | 用作“推理能力与语料搜索分离”的控制组，不作为真实证据搜索的主要来源。[官方仓库](https://github.com/gowitheflow-1998/RAR-b) |
| 原始 SWE-bench 的 patch 文件集合 | 修改文件不等于完成修复所需的全部上下文，直接当相关性 gold 会遗漏依赖与测试 | 优先本文的 ContextBench/SWE-Explore；若仅做补丁文件定位，应明确这是弱代理任务。[官方数据说明](https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/datasets.md) |
| 隐藏标签测试集 | AIR-Bench test、KILT test、部分原始 QA test 无法在当前 host 本地完整评分 | 用公开 dev/validation 做固定 DCI 集，或另设官方远端评测流程 |
| 只有在线网站/动态 API 的任务 | 搜索结果、可访问页面、答案随时间改变，无法保证纯离线同语料对比 | 先完成授权清楚的快照与证据标注；BrowseComp-Plus 已提供更合适的固定版本 |

## 8. 接入前的数据验收清单与核查边界

下一步最有价值的工作是对首批候选做小样本数据审计，而不是直接启动全量模型评测。

1. **固定来源。** 记录官方 dataset ID、commit/revision、具体 config/split、文件 hash、许可及原始来源；正文中的 main 链接只用于检索资料，不作为正式评测唯一版本标识。
2. **检查完整性。** query/文档 ID 唯一，全部 qrels 可定位，所有必需证据存在于该 query 的候选视图；区分 zero-positive、missing-evidence、no-answer 和解析失败。
3. **检查粒度。** 字符、行、页、段落、会话映射可逆；显示文本与原证据一致；保留长度/byte 分布、1MB 超限统计。
4. **检查标签口径。** 正例阈值、graded gain、替代证据集、候选排除规则、Top-10 上限都写入 manifest；不要只保存最终二元 qrels。
5. **检查隔离。** corpus 中没有导入 query、answer、gold rationale、reference patch、标注字段和未来对话；合法文档自身内容不因恰好包含答案而删除。
6. **冻结样本。** 预注册 seed、query 列表、场景/语言/难度分层和排除原因；原始版与派生子集分开命名，不覆盖已有数据。
7. **做最低成本验证。** 先解析/映射和人工核验少量题；再跑无模型的 corpus 搜索/读取检查。需要模型时再采用小规模、固定预算验收，之后才考虑全量任务。
8. **记录可比条件。** 模型、后端、并发、单请求 context、累计 tokens、文件布局、工具输出 cap、超时和 scorer 版本一并固定。

本文件已完成的核查范围为：当前仓库任务/协议/本地小型 manifests 与 qrels；候选官方论文/README/数据卡；少量公开 schema/候选域前缀及 InstructIR 样例；BrowseComp-Plus 两份公开 qrels 的统计与 hash。**尚未执行候选全库的下载、授权核准、完整 ID/span 对齐、全语料去重或任何模型评测。** 因此排序足以指导下一轮接入选择，候选的实际可用题数、运行成本和最终可靠性仍以冻结版本的数据验收为准。



