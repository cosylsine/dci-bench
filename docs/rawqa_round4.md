# 第四轮：原始 QA 与论文元信息派生检索

这批来源补充原生中文金融、通用问答和学术检索。只使用原始记录中已有的关联，
不生成新问题或用模型分数替代标签。每个来源都保留原始文件、固定版本、SHA-256、
转换记录和完整的来源位置别名。

| 来源 | 检索单元 | D | Q | 正例关系 | 状态 |
|---|---|---:|---:|---:|---|
| HC3 中文金融 | 人工回答 | 1,191 | 679 | 1,560 | labels_ready，开发任务 |
| HC3 中文开放问答 | 人工回答 | 7,361 | 3,293 | 7,362 | labels_ready，开发任务 |
| CSL 中文论文 | 摘要 | 9,999 | 1,000 | 1,000 | labels_ready，标题检索派生 |

`labels_ready` 表示这一转换的检索标签可以解析到实际语料，不表示回答经过事实核查、
标注穷尽全部相关文档，或已执行模型评测。D 不是全文长文档数量。

## HC3

原始来源：[作者仓库](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)、
[HC3-Chinese](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)。
作者仓库也列出了本次使用的 ModelScope `simpleai/HC3-Chinese` 发布地址。
文件固定 revision 为 `af38814d5f7ff4b20efa810fb6e0741ee1cc9513`。

完整处理 `finance.jsonl` 的 689 条记录和 `open_qa.jsonl` 的 3,293 条记录。
候选语料只含非空的 `human_answers`，query 使用对应原问题，qrel 为记录中的问答关联。
ChatGPT 回答完整保留在原始文件中，处于本次人工回答语料的范围之外，不充当正例或负例。

金融数据有 12 个无效人工回答位置，其中 10 个问题完全没有有效人工回答；这些问题
逐条记录在 `conversion_manifest.json`，没有生成替代回答。开放问答有 15 个无效回答位置，
但全部 3,293 个问题都有有效人工回答。两个任务均为原始未划分数据的开发检索适配，
没有宣称复现 HC3 检测任务的官方 train/test 划分。

所有有效人工回答按逐字相同文本合并，每个原始行号和回答下标都保留；不做空白、
Unicode 或语义归一化。来源文件没有原生问题 ID，使用文件名和 1-based 行号作为稳定定位。

## CSL

原始来源：[CSL 作者仓库](https://github.com/ydli-ai/CSL)、
[论文](https://aclanthology.org/2022.coling-1.344/)。固定 Git revision：
`0f1ac0626d584cd3c72f8be26d329eb4f967cfd0`。完整 ZIP 中所有文件核验 Git blob、大小与 ZIP CRC。

使用 `benchmark/ts/{train,dev,test}.tsv` 的全部 10,000 条摘要建立语料；仅用原始 test 的
1,000 个论文标题作 query，qrel 来自该标题与摘要的原始配对。一个完全重复摘要通过别名
合并为同一文档，其原行、标题和 split 均保留。候选文档只含摘要，title 字段为空；
来源标题元信息不会进入 DCI 的 corpus 导出字段。

这是从标题摘要任务构建的检索适配，检索问题是论文标题；不是官方 CSL 检索基准，也不是
396,209 条父库的全量发布。当前固定仓库未明确给出许可证，因此记录为未注明。

## 暂未发布的候选

MTEB `stackoverflowdupquestions-reranking` 的完整 test 文件已下载并核验，包含
2,992 个问题、3,467 次正例和 86,003 次负例出现。按题目文本建立候选并集时，有 12 个问题
的相同文本同时出现在正例和负例列表。原始数据及完整冲突记录保留，转换拒绝发布，
没有自动采用正例优先、删除冲突问题或计入新增就绪覆盖。

## 位置与复现

- 原始数据：`/mnt/afs2/202609/embedding_models/dci-bench/raw_data/r4_rawqa_*/`
- 封存版本：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/r4_rawqa_*/v1/`
- 来源记录：`eval_data/source_research_20261023_round4/raw_qa/`
- 独立逐条核验：`eval_data/source_research_20261023_round4/integration/root_source_validation.json`

```bash
cd /mnt/afs/250010100/zxp/dci-bench
python scripts/acquire_rawqa_round4.py
python scripts/acquire_csl_round4.py
python -m unittest discover -s tests -p test_rawqa_round4.py -v
```

重跑会核验已有文件并拒绝内容不同的覆盖。采集脚本会保留 Stack Overflow 标签冲突状态。
本轮没有运行 GPU、cascade 或模型评测。
