# Document 短 ID 与候选池协议

这是新的 document 数据准备与评测入口。数据集名单尚未确定；此版本只实现数据结构、
离线候选池准备和预算配置，不包含 embedding 召回实验或新的模型成绩。
原始数据、旧冻结包和历史实验目录保持原样；code 轨道继续使用文件路径。

## 文档 ID

每个任务独立使用 `d000001`、`d000002` 等短 ID，文件名为 `d000001.md`。
数字至少六位，超过六位时自然扩展，不截断、不取哈希前缀。
在完整源语料上按原始 ID 的确定性顺序建立映射，然后再选择各 query 的候选池。
映射不使用 qrels、召回分数或召回排名；同一任务、同一源版本的同一文档在不同 query
和不同档位保持同一短 ID。不同任务可以复用短 ID，完整身份为 task + source version + docid。

模型只看到短 ID 文件名、原正文和 `read` 返回的短 `doc_id`，最终返回
`{"ranked_doc_ids":["d000001"]}`。原始 ID 映射 `source_doc_ids.json` 与
`original_qrels.json` 保留在 host metadata 内；`qrels.json` 使用短 ID，保留全部相关性等级。
一对一重命名不改变 Recall、F1、nDCG 等评分。旧格式数据仍可验证，迁移需要另建输出目录。

## 逐查询候选池

每次准备显式选择 small、medium 或 large，容量分别为 200、500、1000。
档位表示交给 DCI agent 的候选数上限，不再用原始全库文档数的 1000/100000 分界。

1. 源语料不超过所选容量时，保留整个语料，不重复文档、不补齐到固定数量。
2. 超过容量时，读取该 query 的 embedding top-K 排序，K 为所选容量。
   同分时保留召回后端的原始顺序。
3. 找出该 query 的所有缺失正例（qrel 分数大于 0），全部加入。
4. 从原 top-K 中随机移除同样数量的非正例。随机过程固定 seed、task 和原 query ID，
   不依赖调用顺序或进程全局随机状态；不移除正例、不改相关性等级。
   最终池的 docid 按 embedding 分数降序保存，另存一一对应的 `candidate_scores`。
   同分保留后端原排序；未出现在原排序中的补入正例，在同分已有候选之后按原 ID 排序。
   补入正例必须使用同一模型/打分设置补算相似度，缺分数时拒绝发布，不能使用 qrel 等级代替。
5. 每个 query 只持久化短 docid 列表；所有 query 的入选文档取并集，正文在任务级共享 corpus
   中仅存一份。执行某个 query 时，按列表复制所选文档到临时工作区，结束或异常后自动清理。
   临时存储量随并发执行的 query 数增长，而不是随总 query 数增长。
   Pi 仅挂载当前临时 corpus，可见目录按短 ID 组织，不暴露共享全库、召回排名、补入顺序、
   被移除文档、分数或标签。这些信息仅保存在 host 审计产物中。

正例数量超过容量、存在未知文档 ID、候选排序重复或不足时显式失败。
这套规则使用原始人工标签补入正例，评测含义是**带正例保证的候选池内 DCI 检索能力**；
embedding 原始召回覆盖率和补入数量应另行记录，不能把补入后的覆盖率当作召回模型成绩。
检索池也不等同于历史全语料 benchmark，不能直接据此归因性能变化。

预定离线召回模型：

```text
/mnt/afs/share/embedding_models/KaLM-embedding-multilingual-mini-instruct-v2_5
trust_remote_code = true
```

当前接口消费已经保存的排序和来源身份，准备过程不加载模型。之后实现或运行 embedding
召回时，必须固定模型版本/权重身份、query instruction、pooling、归一化、截断长度和排序规则，
并与原始 corpus/query 身份绑定；本次不预设未经检查的推理细节。

## 初始预算

| 档位 | 候选上限 | 累计 input + output | 收尾预留量下限 | 单请求上下文 | 单次输出 | 单题时限 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 200 | 96,000 | 24,000 | 65,536 | 1,024 | 600 秒 |
| medium | 500 | 128,000 | 32,000 | 65,536 | 1,024 | 600 秒 |
| large | 1,000 | 192,000 | 48,000 | 65,536 | 1,024 | 600 秒 |

这些是**待实测校准的起始配置**，不是模型已完成测试的能力保证。累计 token 包括每轮重放
历史的输入和所有输出。预留量是下限，动态预测较大时使用预测值；它不等于可自由支配的搜索预算。
保留已有单请求、累计实际用量、非空 read、重复调用、final 修复及硬超时保护。
不会因剩余时间自动提前 final。短 ID 缩短命令，但无法保证模型不会再次生成不完整命令。

新 profile 为 `document-pool-v1-small`、`document-pool-v1-medium`、
`document-pool-v1-large`。旧 `budgeted-selective` 和历史 size-budget 实验保持原定义，
不能将旧 full-corpus launcher 当作新候选池入口。使用候选池时，档位、profile 和实际 corpus
必须一致，并纳入运行指纹。

## 准备与检查入口

先用新的 workspace builder 或 expansion builder，在新目录构建短 ID 数据。
`scripts/prepare_data.sh` 默认只允许复用匹配的新格式；不再自动刷新旧目录。
下面的路径是待选数据的占位符，本次未执行这些数据准备命令。

```bash
python scripts/prepare_document_pools.py \
  --source-suite /path/to/short-id-source \
  --output-suite /path/to/new-small-pools \
  --task TASK --pool-size small --seed 20261021 \
  --rankings /path/to/saved-original-id-rankings.json
```

若源语料不超过容量，可以省略 `--rankings`。排序文件必须绑定原始查询文本、语料正文和
ID 映射的身份，使用原始 ID，并记录指定的 KaLM 路径及 `trust_remote_code: true`。
无需保存全库排序，提供至少 K 条唯一、已知 ID 的有序 top-K，以及 top-K 和全部正例的
embedding 分数即可。源语料不超过容量且没有 embedding 结果时，保存确定性 ID 序，
标记 `ordering: "unscored"` 和 `candidate_scores: null`，不会伪造分数。

保存排序的 JSON 接口如下（示意字段，不能直接用于真实数据）：

```json
{
  "schema_version": "dci-saved-retrieval-v1",
  "task": "TASK",
  "model": {
    "path": "/mnt/afs/share/embedding_models/KaLM-embedding-multilingual-mini-instruct-v2_5",
    "trust_remote_code": true
  },
  "source_identity": {
    "queries_hash": "<actual-query-records-hash>",
    "corpus_files_sha256": "<actual-rendered-corpus-file-hashes-digest>",
    "source_doc_ids_hash": "<full-alias-map-hash>"
  },
  "rankings": {"query-1": ["original-id-A", "original-id-B"]},
  "scores": {"query-1": {"original-id-A": 0.81, "original-id-B": 0.72, "missing-positive": 0.31}}
}
```

`build_source_identity()` 生成上述三个身份字段。`rankings` 保留检索器原次序，必须与分数
非增顺序一致；`scores` 可以包含未进入 top-K、需要补入的正例。qrels 始终单独读取，
原始召回结果与补正例后的池分别留档。候选池 metadata 保留分数顺序，临时目录按短 ID
构造，分数、排名和补入标记不作为模型的可见输入。

准备后进行 CPU 数据与逐题 docid 列表检查（不生成每题工作区）：

```bash
python scripts/run_document_pool.py \
  --task TASK --metadata-root /path/to/new-small-pools/metadata \
  --workspace-root /path/to/new-small-pools/workspaces \
  --all-queries --preflight-only
```

该入口从候选池元数据选择 profile，再导入运行时，避免 Python 与 Pi 预算不一致。
这一步没有模型调用，也不等同于 Bubblewrap 验收或模型 smoke。
之后需要真实评测时，去掉 `--preflight-only`，追加 `run_task.py` 接受的 backend/model/run
参数；backend 必须使用本档位约定的上下文设置。每次评测使用独立结果目录。临时工作区采用系统临时目录（可通过 `TMPDIR` 指定本地盘）；
正常结束与异常退出均清理，进程被强制 kill 时可能需要人工清理遗留 `dci-query-*` 目录。
现有沙箱拒绝符号链接与硬链接，因此运行时使用临时副本；数据准备阶段不按 query 复制正文。


## MTEB 参考与边界

已核对本地 MTEB 2.20.4 的实现，并查阅官方源码/数据集说明：

- [MIRACL Hard Negatives task](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/retrieval/multilingual/miracl_retrieval.py)
  与 [MS MARCO Hard Negatives task](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/retrieval/eng/msmarco_retrieval.py)
  描述了 BM25、multilingual-e5-large、e5-mistral-instruct 的候选 pooling，使用 250 文档 cutoff。
  数据加载后是一个共享 corpus 和按 query 组织的 qrels；不同 query 可引用同一份正文。
  普通 retrieval 在共享 corpus 上搜索，qrels 不会自动限制该 query 的搜索范围。
- [二阶段 reranking](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/retrieval.py)
  会按每个 query 的预测分数降序取 top-K，生成 `query_id -> list[doc_id]`。
  DCI 借鉴这一候选表示，另外保存 similarity 分数便于审核与排序复验。
- [MS MARCO hard-negative 数据](https://huggingface.co/datasets/mteb/MSMARCO_test_top_250_only_w_correct-v2)
  中 qrels 的 `score` 是相关性等级（包括 0 标签），不是 embedding 相似度。
  当前公开 task/card 没有充分展示多检索器候选如何去重、最终截断，以及补正例时怎样驱逐负例；
  因此不能声称 MTEB 采用了本项目的“补正例并随机等量移除非正例”规则。

本项目保留“共享正文 + 每 query 有序候选列表”，并由临时工作区强制落实 per-query 搜索边界。
