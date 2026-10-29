# MTEB 六项父任务/上游语料审计（2026-10-23）

本审计只回答一个问题：六个 DCI MTEB(LLM) 任务的非 `llm-eval-*` 父任务或上游原始语料，是否足以在保留旧 query/qrel 映射的前提下建立 200/500/1000 文档池。`llm-eval-*` 快照是保留的评测子集，本文件不把它们计为扩库。

后续完成项：HC3 Finance 已经从官方项目指向的 ModelScope 完整物化为 8,395 个文档，并精确映射旧 415/415 文档；旧 100 个 query/qrel 保留。见 [HC3 扩库记录](hc3_parent_20261023.md)。下表和只读审计执行边界保留当时状态；FQuAD 仍未完成获取与映射。

本次只读核查了 MTEB 官方 task source（commit `3e7f34a7167c1c760e9e7f1ec06eda2cc89f2b`）、官方 Hugging Face 数据卡/固定 revision、上游项目页面和论文统计，以及本地旧 canonical 计数。没有下载原始数据，没有修改 raw/release，也没有运行 GPU 或 `verify_release`。

## 结论表

“评测 corpus”是官方 MTEB 父任务实际用于评测的文档数；“上游全 split”优先报告同任务上游所有 split 的原始记录或文档数。若上游记录可能重复，记录数不等于去重后的可用文档数，状态会保留为 `待验证`。

| 任务 | 官方 MTEB 父任务（revision；评测 corpus / queries / qrels） | 上游全 split / 原始来源规模 | 200 / 500 / 1000 池 | 旧 ID 映射策略 | 当前获取结论 |
|---|---|---|---|---|---|
| `LLMAILAStatutes` | `mteb/AILA_statutes` (`ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e`；test **82 / 50 / 217**) | AILA 2019 同任务 statute corpus **197**；另有约 2,914 个 precedent/case 文档，但不属于 statutes 任务，不能混入 | **不支持**三档；197 也不足 200 | 旧 82 个 MTEB IDs 先按 exact ID/text hash 对回 Zenodo statute 文本；扩展只加入同一 statute corpus 的其他 115 条并保留原 qrels | 可追溯但不足 200；不能用 2,914 case 文档冒充 statute 扩库 |
| `LLMFQuADRetrieval` | `manu/fquad2_test` (`5384ce827bbc2156d46e6fcba83d75f8e6e1b4a6`；test **269 / 400 / 400**，validation **97 / 100 / 100**；合计 366 docs / 500 queries) | FQuAD 1.0：train **4,921** paragraphs、dev **768**、test **532**，全量 **6,221**；train+dev 已有 **5,689** | **支持**三档（按段落数）；需确认可取得的版本和去重后 materialization | 以 exact `(title, context)` 对回；保留 MTEB `title` doc ID 和旧 query ID/qrel，不能按数组位置猜测 | 最明确的扩库候选；官方测试集/许可获取仍待实际物化 |
| `LLMHC3FinanceRetrieval` | `mteb/HC3FinanceRetrieval` (`f7674417d3552fd20e071762bff049137f17b075`；test **415 / 415 / 415**) | HC3 Finance 原始 **3,933 questions**、3,933 human answers、4,503 ChatGPT answers，即 **8,436 answer records**；unique answer text 数未在只读审计中物化 | **待验证**；原始 answer records 明显超过三档，但去重后 doc 数和 exact crosswalk 尚未确认 | question text 与 answer text exact match；以 `human`/`chatgpt` side + source row index 保留文档身份；AI/human 标签不改成 qrel grade | 可行候选；需一次受限原始数据物化才能把“记录数”变成确定 doc 数 |
| `LLMLegalBenchConsumerContractsQA` | `mteb/legalbench_consumer_contracts_qa` (`b23590301ec94e8087e2850b21d43d4956b1cca9`；test **154 / 396 / 396**) | `nguha/legalbench` `consumer_contracts_qa` 全 split **400 QA rows**（train 4 + test 396）；MTEB test 仅 154 unique contracts，full unique contracts 最多 158 | **不支持**；full split 即使 4 行全是新合同也小于 200 | contract text + question exact crosswalk；保留 MTEB doc/query IDs，source row `index` 只作 provenance | 无法形成 200/500/1000 扩库 |
| `LLMPublicHealthQA` | `xhluca/publichealth-qa` (`main`；8 个 language views；English 原始 172 rows，旧 snapshot 为 172 docs/100 queries；全语言 MTEB union 的精确 dedup 数未取得) | 官方上游 8 语言合计 **888 rows**；每语言由 question/answer 去重，合计 unique answer docs 不超过 888 | **不支持 1000**；200/500 需决定跨语言合并并再验证去重计数；English 单语不够 200 | 按 language namespace，以 pinned source 重算 MTEB `Q{index}`/`D{index}`；旧 English IDs 只在 exact text + language 命中后保留 | 不能证明三档；跨语言合并也必须保留语言命名空间 |
| `LLMTwitterHjerneRetrieval` | `mteb/TwitterHjerneRetrieval` (`97ad55673cf9746f8e4b3aaa92b1bb92d82e52db`；train **262 / 78 / 262**) | 官方 raw train **340 QA pairs**，去重后 262 docs、78 queries | 200 **支持**；500/1000 **不支持**，因此不支持三档 | exact question/answer text crosswalk；旧 LLM 只有 77 queries，必须保留 parent 的第 78 个 query，不能假定位置对齐 | 小于 500，不能作为三档扩库 |

### AILA Statutes

官方 MTEB task source 为 [`aila_statutes_retrieval.py`](https://github.com/embeddings-benchmark/mteb/blob/3e7f34a7167c1c760e9e7f1ec06eda2cc89f2b/mteb/tasks/retrieval/eng/aila_statutes_retrieval.py)，数据卡是 [`mteb/AILA_statutes`](https://huggingface.co/datasets/mteb/AILA_statutes)，固定 revision 为 `ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e`，评测 split 是 `test`。数据卡统计为 82 documents、50 queries、217 qrels。上游官方来源是 [Zenodo AILA 2019 dataset](https://zenodo.org/records/4063986)；[AILA 2019 track paper](https://ceur-ws.org/Vol-2517/T1-8.pdf) 给出 197 statutes、约 2,914 case documents 和 50 queries。`AILAStatutes` 只使用 statutes；把 case documents 混入会改变任务语义，因此 197 是本任务可用上限的事实依据。

旧 canonical 的 82 个文档和 query 先用 MTEB parent ID 与完整文本 hash 对回 Zenodo。若上游文件的 ID 变更，以规范化空白后的完整 statute text 和标题做一对一 crosswalk；qrels 只沿用已有 parent qrels，不从 case 文档推导新正例。

### FQuAD Retrieval

官方 MTEB task source 为 [`f_qu_ad_retrieval.py`](https://github.com/embeddings-benchmark/mteb/blob/3e7f34a7167c1c760e9e7f1ec06eda2cc89f2b/mteb/tasks/retrieval/fra/f_qu_ad_retrieval.py)，数据卡/父源是 [`manu/fquad2_test`](https://huggingface.co/datasets/manu/fquad2_test)，固定 revision `5384ce827bbc2156d46e6fcba83d75f8e6e1b4a6`。代码将 `valid_hasAns`/`test_hasAns` 作为 validation/test，按 `title` 作为 corpus key，并为每条 question 建一个 grade 1 qrel。官方 MTEB release card [`mteb/FQuADRetrieval`](https://huggingface.co/datasets/mteb/FQuADRetrieval) 报告 test 269 docs/400 queries 和 validation 97 docs/100 queries。

上游 FQuAD 1.0 的官方论文 [FQuAD: French Question Answering Dataset](https://aclanthology.org/2020.findings-emnlp.107/) 表 1 给出 train 4,921 paragraphs、development 768、test 532，共 6,221 paragraphs（145 articles）。官方项目页是 [FQuAD explorer](https://illuin-tech.github.io/FQuAD-explorer/)；项目所列 train/dev 下载入口为 [`train.json.zip`](https://storage.googleapis.com/illuin/fquad/train.json.zip) 和 [`valid.json.zip`](https://storage.googleapis.com/illuin/fquad/valid.json.zip)。因此即使只取 train+dev，也超过 1,000；本次没有下载，且 FQuAD1.0 全量与 MTEB 使用的 `fquad2_test` 版本不是可直接替换的同一快照。

映射必须用 exact `(title, context)`，query 用 exact question text；若 `title` 在上游重复，使用 `(title, context hash)` 的稳定派生键并保留原 MTEB title。旧 269+97 文档与 qrels 只有在 exact 命中后才可继承。

复核备注：该论文正文 §3.1 写的是 train 4,951 / test 523，与第 4 页 Table 1 的 4,921 / 532 不同。这里明确引用 Table 1，train+dev 5,689 是论文表格统计，不能替代实际文件计数；两个口径都超过三档规模要求。

### HC3 Finance Retrieval

官方 MTEB task source 为 [`hc3_finance_retrieval.py`](https://github.com/embeddings-benchmark/mteb/blob/3e7f34a7167c1c760e9e7f1ec06eda2cc89f2b/mteb/tasks/retrieval/eng/hc3_finance_retrieval.py)，父数据卡为 [`mteb/HC3FinanceRetrieval`](https://huggingface.co/datasets/mteb/HC3FinanceRetrieval)，固定 revision `f7674417d3552fd20e071762bff049137f17b075`，test 为 415 docs/415 queries/415 qrels。上游官方项目 [Hello-SimpleAI/chatgpt-comparison-detection](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection/blob/main/HC3/README.md) 指向 [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)；Finance 子集数据卡/查看器显示 3,933 questions。原论文 [How Close is ChatGPT to Human Experts?](https://arxiv.org/abs/2301.07597) 的 finance 统计为 3,933 human answers 和 4,503 ChatGPT answers，即 8,436 answer records。

8,436 是原始 answer records，不是已经核实的 unique document 数；本审计不把它直接冒充去重后 corpus。实际物化时应保留 answer side（human/ChatGPT）和 source row index，使用 question/answer exact text 对回 415 parent docs/queries；`human`/`ChatGPT` 只作为来源字段，不得替换 qrel grade。只要去重后仍有至少 1,000 条 answer documents，即可建立三档，否则应按验证结果收敛。

### LegalBench Consumer Contracts QA

官方 MTEB task source 为 [`legal_bench_consumer_contracts_qa_retrieval.py`](https://github.com/embeddings-benchmark/mteb/blob/3e7f34a7167c1c760e9e7f1ec06eda2cc89f2b/mteb/tasks/retrieval/eng/legal_bench_consumer_contracts_qa_retrieval.py)，父数据卡为 [`mteb/legalbench_consumer_contracts_qa`](https://huggingface.co/datasets/mteb/legalbench_consumer_contracts_qa)，固定 revision `b23590301ec94e8087e2850b21d43d4956b1cca9`，test 为 154 docs/396 queries/396 qrels。上游 [LegalBench Consumer Contracts QA task page](https://hazyresearch.stanford.edu/legalbench/tasks/consumer_contracts_qa.html) 说明任务有 400 个 yes/no questions；固定的 [nguha/legalbench dataset info](https://huggingface.co/datasets/nguha/legalbench/blob/2895c24ebffea62f2a29b6b862147aa61024fd4c/dataset_infos.json) 显示该 config 为 train 4 rows + test 396 rows。

即便 train 的 4 行都引用新合同，full split unique contracts 也至多 158（MTEB test 已有 154），所以不能到 200，更不能到 500/1000。旧 qrels 只可按 exact contract text 和 question crosswalk；不把 yes/no 标签转成文档相关性等级。

### PublicHealth QA

官方 MTEB task source 为 [`public_health_qa_retrieval.py`](https://github.com/embeddings-benchmark/mteb/blob/3e7f34a7167c1c760e9e7f1ec06eda2cc89f2b/mteb/tasks/retrieval/multilingual/public_health_qa_retrieval.py)，任务使用的父数据卡为 [`xhluca/publichealth-qa`](https://huggingface.co/datasets/xhluca/publichealth-qa)，对应的 MTEB release card 是 [`mteb/PublicHealthQA`](https://huggingface.co/datasets/mteb/PublicHealthQA)。task metadata 使用 revision `main`，包含 Arabic、Chinese、English、French、Korean、Russian、Spanish、Vietnamese 八个语言 view。代码按语言分别去重 question/answer，生成 `Q{index}`、`D{index}`，并跳过 None；旧 English snapshot 有 172 docs/100 queries。上游数据卡的 [data tree](https://huggingface.co/datasets/xhluca/publichealth-qa/tree/main/data) 列出八个 CSV，官方页面统计合计 888 rows。

因此所有上游 raw rows 的数量上界是 888，无法证明 1,000 个不同 answer documents。English 单语言只有 172 rows，不足 200；跨语言合并后 200/500 的可用性仍取决于是否把语言作为 doc namespace 以及 answer 去重结果，必须重新物化核验。映射时固定 source revision、保留语言 namespace，并用 exact question/answer text 重新计算 MTEB 的 Q/D IDs；不能仅凭旧 ID 的数字顺序跨语言复用。

### TwitterHjerne Retrieval

官方 MTEB task source 为 [`twitter_hjerne_retrieval.py`](https://github.com/embeddings-benchmark/mteb/blob/3e7f34a7167c1c760e9e7f1ec06eda2cc89f2b/mteb/tasks/retrieval/dan/twitter_hjerne_retrieval.py)，父数据卡为 [`mteb/TwitterHjerneRetrieval`](https://huggingface.co/datasets/mteb/TwitterHjerneRetrieval)，固定 revision `97ad55673cf9746f8e4b3aaa92b1bb92d82e52db`，评测 split 是 `train`，统计为 262 docs、78 queries、262 qrels。上游 source 是 [`sorenmulli/da-hashtag-twitterhjerne`](https://huggingface.co/datasets/sorenmulli/da-hashtag-twitterhjerne/tree/main/data)，官方 Danish registry 见 [Sprogteknologi.dk Twitterhjerne](https://sprogteknologi.dk/dataset/twitterhjerne)；父卡记录 raw train 340 QA pairs，去重后 262 docs/78 queries。

262 docs 足以做 200 pool，但不足 500/1000。旧 LLM snapshot 只有 77 queries；映射必须按 exact Danish question/answer text，保留 parent 的第 78 个 query，不能按旧 1..N 位置假定完整。

## 本地旧快照与验证记录

只读检查了 `eval_data/dci_expansion_20261014_v1/canonical/<task>`：旧 corpus 数依次为 AILA 82、FQuAD 269、HC3 415、Legal 154、PublicHealth 172、Twitter 262；这些计数与各自 `llm-eval-*` 子集/旧 canonical 有关，不能作为父源扩库结果。保留原始 query/doc IDs 与 qrel grade 的原则是：先 exact text/ID crosswalk，再继承旧 qrels；不按相似度、不按标签、不按数组顺序制造正例。

执行的核查边界：

* 阅读 MTEB 官方 source tree 的六个 task files（commit `3e7f34a7167c1c760e9e7f1ec06eda2cc89f2b`）和对应 HF card/revision。
* 查阅 Zenodo、ACL Anthology、官方项目页及上游 HF dataset tree/card；文件内列出的 URL 是可复核入口。
* 只读统计本地旧 canonical；没有打印全量 ID 集。
* 没有下载任何原始数据，没有写 raw/release，没有运行 `verify_release`、GPU 或 cascade。下一步若要把 FQuAD/HC3 变成 release，仍需在受限大小内取得上游文件、生成 crosswalk，并对新 release 运行完整 `verify_release`。
