# 2026-10-23：50 格数据扩充

本轮将已就绪的 `(domain, language)` 覆盖从 22 格增加到 50 格。新增 28 个就绪来源和 12 个待复核来源视图；所有新增视图都超过 1,000 个检索单元。已有版本、原始下载和失败记录均保留。

新看板：[pipeline_matrix_overview_20261023_v3/index.html](/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_matrix_overview_20261023_v3/index.html)。数据继续存放在用户指定的 `/mnt/afs2/202609/embedding_models/dci-bench/raw_data/` 和 `eval_data/pipeline_v1/` 下，使用独立的 `r4_*` 来源目录。

“就绪”表示 corpus、queries、qrels 已落盘且通过数据发布检查；三档候选池、KaLM 排名和模型评测尚未执行。50 格不是 50 个独立数据集，也不表示 50 格都有原生语言题集。页面新增原生／翻译／未知筛选，按格去重统计，并展示实际题量。

最终按格统计：16 格至少有一个原生来源、29 格只有现成译本、5 格来源属性仍未知或混合。矩阵共列出 99 个来源视图；12 个本轮待复核视图不计入 50 格。新增 28 个就绪来源的每题正例数均未超过 200，因此从已知语料量与正例容量看支持 200／500／1000 三档，仍需实际运行检索与候选池生成。

## 新增就绪格

| 领域 | 新增语言 | 数量 | 来源与规模 |
| --- | --- | ---: | --- |
| web | ar、de、es、fr、hi、id、it、ja、nl、pt、ru、vi | 12 | mMARCO 现有机器译本；每种语言原始 106,813 D、6,980 Q、7,437 条正例关系 |
| scientific | de、es、fa、fr、it、ja、ko、nl、no、pl、sv、vi、zh | 13 | SciFact 多语译本、NanoSciFact、CSL 原始标题摘要配对 |
| finance | fa、zh | 2 | 波斯语 FiQA 与 HC3 中文人工回答 |
| general_qa | zh | 1 | HC3 中文 open_qa 人工回答 |

SciFact 荷兰语、波兰语、波斯语各有 5,183 D / 300 Q；越南语发布视图只有 134 Q，未声称拥有英文父任务的全部 300 Q。八种 NanoSciFact 语言各有 2,919 D / 50 Q，满足语料容量，但题量较小，适合先做开发验证。所有 SciFact 版本同源，跨语言汇总时必须按数据集组处理。

mMARCO 使用官方 MTEB 有界 dev 视图，不是完整的 800 万级父库。de/es/hi/id/ja 分别将 28/34/27/23/44 个完全相同的 `(title,text,language)` 文档 ID 映射到显式别名；全部查询和 qrel 等级保留，父版本继续保存。冲突等级会阻断发布，近重复不合并。

## 从 raw data 构建

HC3 中文金融：689 个原始问题中，10 个没有有效人工回答，明确排除并记录审计；发布 1,191 D / 679 Q / 1,560 条正例关系。中文 open_qa：7,361 D / 3,293 Q / 7,362 条正例关系，全部原始问题保留。只使用 `human_answers`；ChatGPT 回答留在 raw 中。来源没有官方 test split，因此标为派生开发集；问答配对不等于回答事实正确或相关性标注穷尽。

CSL：完整保留 10,000 条 benchmark 的摘要及来源位置，精确别名后为 9,999 D。仅使用原始 test split 的 1,000 个论文标题作为查询，按原始配对建立 1,000 条正例关系。候选文档 `title` 为空，标题仅留查询与 host 元数据。这个任务是标题→摘要检索，不是 CSL 原有摘要生成指标，也不是 396,209 条完整父库。

## 待复核和未发布来源

9 个 MIRACL HardNegatives.v2 原生语言视图（bn/fa/fi/fr/id/ko/sw/te/yo）全部保持 `review_required`。它们显式记录精确别名，保留所有 query 和正例 ID；正文相同但标题不同的候选尚未解决。原始 qrels 中，部分候选明确为零分，另一些未标注，两者不能混同；部分零分 ID 不在官方有界 corpus 内。完整原始 Parquet、零分记录与既有 YO 完整版本均保留。

荷兰语／波兰语 NFCorpus、荷兰语 LegalQA 也保留为待复核视图。波斯语 NFCorpus 的别名映射会合并不同相关性等级，未发布。NL/PL FiQA 有正例指向空文档，NL/PL/VN ArguAna 有正例指向缺失 ID，VN FiQA 有重复 query ID；这些来源停在原始快照与失败记录。StackOverflow 原始问答有 12 个精确文本正负例冲突，同样未发布。没有为达到目标而删除冲突问题或擅自选择最高等级。

## 复现与验证

收集脚本、固定 revision 与来源说明分别见 [web](web_round4.md)、[domains](domains_round4.md)、[MIRACL](encyclopedia_round4.md)、[raw QA / CSL](rawqa_round4.md)、[Persian](extra_fa_round4.md)。

本轮集成记录位于 [source_research_20261023_round4/integration](/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023_round4/integration)。其中 `selection.json` 固定来源选择，`annotations.json` 记录报告说明，`render_command.sh` 保存完整生成命令。目标目录已存在时，生成器拒绝覆盖；复现看板应选择新的输出目录。

验证覆盖原始下载哈希、release 清单与 bundle 哈希、全部选中 query 的 qrel 引用、原始等级映射、独立原始行核对、HTML 筛选执行、CSV/SVG、旧看板文件不变及 cascade 命令解析。中文构建脚本和德语网页精确别名另由独立 worker 复核；MIRACL 原始零分／未标注区别由主线程单独核对。矩阵与 raw QA 的 13 项定向单元检查通过。完整结果见 [final_validation.json](/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023_round4/integration/final_validation.json)。

`cascade_plan.json` 仅为后续命令计划。本轮没有运行 GPU、排名、候选池生成或模型评测。
