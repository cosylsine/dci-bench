# 第四轮：波斯语 BEIR 派生来源

采用 MTEB Git revision `2ad77827480c853ee9518eb207a244f13f8a0966` 中
`mteb/tasks/retrieval/fas/beir_fa.py` 指定的固定数据版本，直接下载 HF 原始文件，
核验 LFS SHA-256 和字节数。这些是已有波斯语译本，本次没有进行翻译。

| 来源 | 数据版本 | 原始 D → 发布 D | 测试 Q | 正例关系 | 状态 |
|---|---|---:|---:|---:|---|
| MCINext/fiqa-fa-v2 | `3443316360bb14c1ef68f609d6fe8f89b117977d` | 57,638 → 57,595 | 648 | 1,706 | labels_ready |
| MCINext/scifact-fa-v2 | `75937d5e71adf8040e511fab41e06bc88825a52b` | 5,183 → 5,183 | 300 | 339 | labels_ready |
| MCINext/nfcorpus-fa | `70aa71825a791e87210c0355a01f538aa611feae` | 原文件保留 | 原文件保留 | 原文件保留 | 原始等级冲突，未发布 |

来源：[FiQA](https://huggingface.co/datasets/MCINext/fiqa-fa-v2)、
[SciFact](https://huggingface.co/datasets/MCINext/scifact-fa-v2)、
[NFCorpus](https://huggingface.co/datasets/MCINext/nfcorpus-fa)、
[FaMTEB 论文](https://arxiv.org/abs/2502.11571)。MTEB 对这些固定版本的许可证标记为未注明，
采集记录沿用该状态。

全部原始 corpus、queries 和 test qrels 留存。发布查询严格取原始 test qrels 的全部 query ID；
queries 文件中的其他 split 查询仍保存在原文件中。字符串型分数解析为对应数值，保持等级。
原始 qrels、等级统计及任何同等级重复关系均有记录。

FiQA 中 43 个文档 ID 对应完全相同的 title/text/language，显式映射到每组字典序最小的
原始 ID，所有 ID 均可追溯。查询、等级和测试正例关系保持不变；这是具有明确别名映射的
派生视图，不能将其指标直接当作未经改动的官方数据指标。

NFCorpus 的同内容文档存在不同原始相关性等级。例如问题 `PLAIN-2720` 的一个别名组
会合并等级 1 和 2。转换拒绝选择最大等级或自动覆盖，原始文件保留，没有发布或计入就绪。
完整冲突列表见 `extra_fa/nfcorpus_alias_conflicts.json`。

数据位置：

- `raw_data/r4_extra_fa_{fiqa,scifact,nfcorpus}/`
- `eval_data/pipeline_v1/r4_extra_fa_{fiqa,scifact}/v1/`
- `eval_data/source_research_20261023_round4/extra_fa/`
- 独立验证：`eval_data/source_research_20261023_round4/integration/extra_fa_validation.json`

以上相对位置基于 `/mnt/afs2/202609/embedding_models/dci-bench`。

```bash
cd /mnt/afs/250010100/zxp/dci-bench
python scripts/acquire_extra_fa_round4.py
```

重跑核验原始文件与已有 release；NFCorpus 仍会记录相同的拒绝发布原因。
本轮只完成数据准备和验证，没有运行 GPU、cascade 或模型评测。
