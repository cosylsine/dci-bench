# MedQuAD 原始 XML → 回答检索

来源为 [abachaa/MedQuAD](https://github.com/abachaa/MedQuAD)，固定 revision
`577bd37b96c02d1833b2c9eed2de9f96964e96cb`。下载完整 16,953,181-byte
仓库 ZIP；所有 11,277 个文件逐一核对 Git blob ID、大小和 ZIP CRC。
许可证及原始 readme 随研究记录保留。

这是自行构建的 QA 派生开发任务：问题来自原始 XML，正例来自其对应的已发布回答。
检索文档只包含回答正文，title 为空，不把题目复制进文档。
来源没有官方 held-out split，部分问题按类型/模板构造，因此不能把它当作
未见自然用户问题的 TREC LiveQA 结果。关联标签也不是全库穷尽相关性标注。

| 项目 | 数量 |
|---|---:|
| XML 文件 | 11,274 |
| 原始 QA 记录 | 47,441 |
| 有已发布回答的 QA / 保留 query | 16,407 |
| 去重后的回答文档 | 15,817 |
| 正例关系 | 16,407 |
| 以别名合并的完全相同回答记录 | 590 |
| 缺少已发布回答的排除记录 | 31,034 |

README 写 47,457 条 QA；本次固定快照直接解析得到 47,441，采用后者。
原作者说明三个 MedlinePlus 子集移除了回答。缺少回答的记录保留在原始 XML，
逐条写入 `conversion_manifest.json`，没有自行补答案或重爬相关页面。
只合并完全相同的回答字符串，每个原始 XML 文件、QA ID、文档 ID 和 URL
均保存在别名中。问题文本不改写，原始局部 ID 以文件路径作为命名空间。

数据位置：

- raw：`/mnt/afs2/202609/embedding_models/dci-bench/raw_data/gapfill_medquad_rawqa/`
- release：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_v1/gapfill_medquad_rawqa/v1/`
- 研究及校验：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/source_research_20261023_round3/medquad/`

复现：在仓库根目录运行 `python scripts/acquire_medquad_gapfill_20261023.py`。
已有文件只核验，不覆盖。所有保留题目、回答和 qrel 均与原始 XML 对照；
release 状态为 `labels_ready`，文件清单和 bundle hash 校验通过。
没有运行模型、cascade 或 GPU 评测。

另外检查了作者发布的 LiveQA 2,479-answer ZIP。它包含 1,935 个唯一 answer ID，
其中 168 个 query–answer 对有不同等级的重复判断。该原始 ZIP 和官方题目 XML
保存在 `raw_data/gapfill_medquad_liveqa/source_inputs/`；本轮未自动裁决这些冲突，
也未把它发布为就绪检索任务。原始等级 1 表示 Incorrect，不能直接当成正例。
