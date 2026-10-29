# Quora 完全相同文档的别名派生视图

原始 BEIR Quora release 有 522,931 篇文档、10,000 个 test query 和 15,675 条正例。
其中 42 个 query 产生了 46 条“正例存在文本相同、ID 不同而未标注的副本”记录，因此保留为
`review_required`。原始 release、下载 ZIP 和全部判断保持不变。

本轮新建 `gapfill_beir_quora_exact_dedup/v1`：只在 **title、text、language
逐字相同** 时合并文档，以字典序最小的原 ID 为代表，其余 ID 保存为别名。
每条原始正例按该映射转到代表 ID，等级不变；若同一题合并后的等级不一致，
脚本直接报错，不取最大值或平均值。没有删除问题。

结果为 522,363 篇文档、全部 10,000 个原始 query、15,675 条正例；524 个重复组
中的 568 篇完全相同文档转为别名。所有原始 query 记录逐字段不变，全部原始正例
都可通过别名追溯，保留文档的正文和标题不变。新 release 的状态为 `labels_ready`。

这是显式派生的 DCI 检索单位定义；不能将其指标直接当作官方 BEIR Quora 指标。
两个视图归于同一 Quora 数据集组，不累加为独立语料。

- 脚本：[derive_quora_exact_aliases_20261023.py](../scripts/derive_quora_exact_aliases_20261023.py)
- 父 release：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_beir_quora/v1/`
- 派生 release：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_beir_quora_exact_dedup/v1/`
- 全部重复组映射及校验：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023_round3/quora_aliases/research.json`

在仓库根目录运行 `python scripts/derive_quora_exact_aliases_20261023.py` 可核验已有
版本。三个单元检查覆盖等级冲突、标题差异和空白字符差异；实际全量转换额外核对
全部 query、正例、别名映射和文件哈希。未运行 cascade 或模型评测。
