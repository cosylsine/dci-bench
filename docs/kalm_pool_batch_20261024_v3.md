# KaLM 三档池 v3：跳过内容哈希扫描，batch size 128

本版本用于原来的 53 个来源和 200/500/1000 三档池。新执行的 embedding 请求 batch size 为 128，保留 OOM 自动减半；最大长度 8192、模型路径、query 指令、`trust_remote_code=True`、排序规则、标签和 seed 均不变。

## 哈希和验证边界

冻结计划设置 `validation.verify_content_hashes=false`。运行时不再为了校验而扫描原始 release 内容、导出的文档/Parquet、模型权重、已完成 ranking、embedding 缓存复制内容和池中文档内容。检索直接使用源 inventory 中已记录的文档摘要和 v2 冻结计划中的模型身份；读取文档正文仍是编码必需的。

保留文件类型/存在性/大小检查、JSON/JSONL 解析、短 ID 反查、query/qrels 闭包、标签等级、候选容量、分数及顺序检查。内容与已记录摘要是否一致不再验证，因此不能发现所有同长度内容变化。计划、preflight、ranking、池和完成凭证均记录此验证边界。

代码/启动器/计划封存摘要、少量控制凭证摘要、缓存索引键，以及首次生成产物所需的 metadata 摘要仍保留；这些不是逐文档或模型权重的重复扫描。单独调用通用验证 API 时，默认仍启用严格校验。

## 启动和恢复

先停止正在运行的 v2 集群任务，保留 v1、v2 的全部产物，然后在单节点 4×H100 任务中启动一次：

```bash
bash /mnt/afs/250010100/zxp/dci-bench/releases/data-pipeline/kalm_pool_batch_20261024_v3/launch.sh
```

无需 `torchrun`。新运行目录为：

```text
/mnt/afs2/202609/embedding_models/dci-bench/eval_data/pipeline_pool_runs/kalm_pool_batch_20261024_v3
```

v3 从 v2 固定可复用产物清单，保持旧目录只读：

- 已完成的 export、ranking、三档池继续使用原路径；允许 v2 的池继续引用 v1。修改 batch size 不会触发已完成 ranking 重算，旧结果保留实际使用 batch size 的审计记录。
- 未完成 export/pool staging 保留在旧目录，v3 创建新 attempt。
- v2 的 batch-32 embedding 分片与 batch-128 的缓存身份不同，因此不复制这些无法命中的部分缓存。尚未完成 ranking 的来源会按 128 重新编码；这是缓存兼容性边界。
- v3 自身中断后使用同一启动命令恢复，其 batch-128 缓存可继续使用。

完成凭证内 `pools[].path` 是结果路径的权威来源；不要删除 v1/v2 来腾空间。

## CPU 资源

默认导出并发 2、GPU 排名并发 4、建池并发 2、导出预取 8，CPU 数值线程上限 4、导出/验证 I/O workers 为 4。

申请更多 CPU 后，可在启动命令后追加 `--export-workers`、`--pool-workers`、`--export-prefetch`、`--cpu-threads`、`--io-workers`。例如已有足够主机内存及 CPU 时可用：

```bash
bash /mnt/afs/250010100/zxp/dci-bench/releases/data-pipeline/kalm_pool_batch_20261024_v3/launch.sh \
  --export-workers 4 --pool-workers 4 --export-prefetch 12 \
  --cpu-threads 8 --io-workers 16
```

数值线程数也受传入环境变量限制。并发增加会提高主机内存和共享盘压力，不能只按 GPU 显存判断资源余量。本版本尚未优化 dense search 的 CPU/GPU 往返或大 JSON 序列化；增加 batch size 只影响 embedding 阶段。

2026-10-24 22:01（北京时间）的封存后预检显示 `/mnt/afs2` 约剩 15.3 GiB，低于调试开始时的约 49 GiB；提交前需要处理输出容量并重新确认空间。不要删除被复用的 v1/v2 产物。

## 验证

验证记录保存在 `eval_data/pipeline_validation/kalm_pool_batch_20261024_v3/`。CPU 回归、单设备合成 smoke 与集群 53 源全量完成分别记录；本次不自动启动四卡全量任务。

- 136 项相关回归测试通过，覆盖严格默认行为、跳过内容哈希、batch 变更、旧 ranking 复用、v1→v2→v3 池路径复用和配置/标签错误拒绝。
- 单 H100 MIG、1001 合成文档、3 条 query 的实际 KaLM 分阶段 smoke 通过：文档/query 请求与有效 batch size 均为 128，OOM 重试为 0；三档池完成后又以 CPU 进程验证恢复，没有重复编码。五个阶段进程均成功，合计约 75.21 秒。此计时不代表四卡全量加速比例。
- 生产冻结包的 59 个代码文件与 smoke 快照完全一致；53 个来源条目与 v2 完全一致，模型/池参数只有 batch size 从 32 改为 128。
- 对真实 v2 的法语、意大利语和 Quora 运行目录做了只读凭证复用探针，成功解析旧 export/ranking 与跨版本池路径；未执行这些来源的全量校验或推理。
- 封存后单 GPU preflight 通过，并明确记录 `content_hashes_verified=false`。未启动 v3 全量任务。
- 补充下游运行时检查：在项目环境中 7 项通过，1 项真实 Pi mock 集成检查被当前环境的 bubblewrap 不可用阻塞。该检查不属于三档池构建 smoke；相关失败日志也保留。初次使用 retriever 环境缺少 `inspect_ai`，随后切换项目环境并修正动态库路径。

## 2026-10-26 代码迁移

冻结启动器及 `code/` 已迁至代码仓库 `releases/data-pipeline/`。原数据目录仅保留数据控制凭证。上述旧版本命令作为历史配置记录；v2/v3 计划的跨版本绝对路径仍保留原始封存值，恢复旧批次前须生成迁移后的续跑计划，不能直接按旧命令启动。此次仅验证文件迁移及原封存摘要，没有运行 KaLM 推理或重建数据池。
