# 53 个来源的 KaLM 三档检索池

本批次使用 `pipeline_matrix_overview_20261023_v3/data_matrix.json` 中除已有 SciFact 外的 53 个 `pipeline / three_tier_source / labels_ready` 来源视图。共 2,320,250 条文档记录、196,088 条 query，目标为 159 个池套件，容量分别为 200、500、1000。文档数是各视图之和，不是跨来源去重后的数量。全部 query 至少有一个正例，且不超过 200 个正例。

SciFact 历史排名 `pipeline_scifact_rankings_v1.json` 的 model、query encoding audit、document encoding audit 均确认已开启 `trust_remote_code=True` 并使用下面的指令，因此本批次保留旧产物，不重复运行它。冻结计划保存了该历史文件的 SHA-256。

## 模型与标签契约

- 模型：`/mnt/afs/share/embedding_models/KaLM-embedding-multilingual-mini-instruct-v2_5`；离线加载，固定权重、配置、tokenizer 和代码身份。
- `trust_remote_code=True`。
- Query 前缀精确为 `"Instruct: Given a query, retrieve documents that answer the query \n Query: "`；只添加一次。Document 前缀为空，与本地模型的配置一致。
- 最大长度 8192 token；初始编码 batch 32，CUDA OOM 时减半直至 1，不降低最大长度或改变正文。记录实际 batch、重试及截断数。
- Embedding 缓存每 2048 条一个原子分片，按模型、指令、文本等身份绑定；中断后复用完整分片。
- 归一化向量做 FP32 点积检索，关闭 TF32；GPU query batch 32，分块维护 top 1000，平分时保留 corpus 顺序。只承诺同后端的确定性，不承诺 CPU/GPU 浮点位级一致。
- 保留原始 qrels 和等级；每档注入缺失正例并等量剔除非正例，seed 为 20261021。候选始终按真实检索分数排序，qrel 等级不充当相似度分数。全正例保留的候选池不是自然 top-K 召回结果。

## Doc ID 已统一

`short-document-id-v1` 从完整来源 ID 的 UTF-8 字节排序生成任务内 `d000001`、`d000002` 等短 ID；正文文件为 `d000001.md`。同一来源的三个容量共用映射，不按 query 或排名重新编号。不同任务的短 ID 可以重复，跨任务引用须携带 task。

Host metadata 保留 `source_doc_ids.json`（短 ID → 原始 ID）、`doc_id_files.json`（短 ID → 文件名）、`original_qrels.json`；`qrels.json` 使用短 ID 并保留等级。候选记录同时保存短 ID、原始 ID 和分数。每个池套件只保存一次候选文档并集，query 保存 ID 列表；运行检索 agent 时再构造该 query 的临时工作区。不同容量是独立套件，会各自保存正文并集，不使用硬链接。

## 集群启动与恢复

冻结包：

```text
/mnt/afs2/202609/embedding_models/dci-bench/eval_data/kalm_pool_batch_20261024_v1
```

申请单节点 4×H100，在任务的启动命令中执行一次：

```bash
bash /mnt/afs/250010100/zxp/dci-bench/releases/data-pipeline/kalm_pool_batch_20261024_v1/launch.sh
```

不需要 `torchrun`，也不要启动四份 launcher。脚本内部动态分配四个 worker，每卡一个来源，来源完成后补下一个；继承调度器的 `CUDA_VISIBLE_DEVICES` 映射。默认使用现有 `retriever` Python 环境（Transformers 4.48.0、sentence-transformers 5.1.0），优先加载该环境的 C++ 运行库。节点须可访问相同共享路径。可用 `EMBED_PY=/absolute/path/to/python` 指定具有相同依赖的环境。

建议申请至少 32 个 CPU 核和 128 GiB 主机内存，优先 256 GiB；预留至少 200 GiB 输出空间及足够文件配额。这是为四进程、大型 JSON 排名和多档候选元数据预留的容量建议，尚无全量峰值实测。初始共享数据的 documents.jsonl 合计约 2.34 GiB，但候选槽总数为 333,349,600，输出与主机内存不能按正文大小估算。每个 worker 默认 8 个数值计算线程。

输出目录：

```text
/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_pool_runs/kalm_pool_batch_20261024_v1
```

- `logs/<source>-<attempt>.log`：各次 worker 日志。
- `sources/<source>/status.json`：当前阶段；`failures/` 保留失败原因和 traceback。
- `sources/<source>/phases/`：通过校验的 export/rank 阶段凭证；失败 attempt 留在 `attempts/`。
- `sources/<source>/pools/{200,500,1000}/`：通过校验的三档池。
- `summary.json`：本次调度完成/失败情况；只有所有来源成功才生成根目录 `_COMPLETE.json`。

发生中断后执行相同启动命令。脚本会重新验证已有阶段并复用缓存；已损坏或身份不符的完成产物直接报错，不覆盖。单源失败不会取消其余来源，批次最终返回非零状态并列出失败来源。两份 launcher 无法同时写同一运行目录。

冻结包中的 `_COMPLETE.json` 仅封存计划、代码与 launcher；运行输出中的根 `_COMPLETE.json` 才表示本批次 53 源、159 池全部完成。该任务不运行 DCI agent，也不构成一次检索 agent benchmark。

## 验证记录

2026-10-24 已通过 96 项相关回归测试；动态库启动修复后又复跑 11 项批次测试，全部通过。真实 KaLM 在 H100 MIG 3g.40gb 上完成 1001 文档、3 query、三档池及重复启动验证，用时 97.66 秒；没有编码 OOM 或 token 截断。三档的 1001 条短 ID 映射完全一致，文件名均对应短 ID。另以 513 个整数向量、7 条 query 对照 CPU/CUDA 和完整排序参考，包含跨块平分、末尾块及 top-K 外正例，排名一致，分数最大绝对差为 0。该整数向量检查不代表一般浮点输入位级一致。

封存代码已独立复核并复用 smoke 产物。封存 preflight 核对了全部 53 源元数据、模型身份、实际依赖导入及一个可见 MIG 设备的 CUDA kernel。第一次 smoke 因系统旧 `libstdc++` 导致 Pillow 导入失败，失败目录及日志保留；launcher 已优先使用 `retriever/lib`，第二次 smoke 成功。53 源正式运行目录尚未创建。

验证产物保存在 `eval_data/pipeline_validation/kalm_pool_batch_20261024_v1/`。其中测试、GPU smoke 和冻结计划 preflight 是不同边界；当前工作机只有一块 H100 的 MIG 分区，四卡并行实机吞吐须在集群执行后确认。完整来源文件哈希会在每个来源导出前校验，封存时先核对全部 53 个来源的元数据、query、qrels 和完成凭证。

开发入口为 `scripts/run_kalm_pool_batch.py`；`scripts/smoke_kalm_pool_batch.py` 用 1001 篇合成文档、3 条 query 跑实际模型、全部三档及恢复验证，不使用真实评测数据重新造池。测试环境隔离在 `/tmp/dci-kalm-pools-test-20261024`，未向用户 conda 环境安装测试依赖。

## 2026-10-26 代码迁移

冻结启动器及 `code/` 已迁至代码仓库 `releases/data-pipeline/`。原数据目录仅保留数据控制凭证。上述旧版本命令作为历史配置记录；v2/v3 计划的跨版本绝对路径仍保留原始封存值，恢复旧批次前须生成迁移后的续跑计划，不能直接按旧命令启动。此次仅验证文件迁移及原封存摘要，没有运行 KaLM 推理或重建数据池。
