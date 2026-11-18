# 正例上限 10 的新版数据与实验

新版已发布路径为
`/mnt/afs2/202611/embedding_models/dci-bench/eval_data/benchmark_v2_20261116_pos10_v1`。
旧版 `benchmark_v2_20261103_v1` 与已有 Qwen 结果保留不变。
新版只能在两个法律源构池完成、20 项统一重抽并验收后使用；源数据准备完成不等于候选池或模型实验完成。

2026-11-16 已完成两项新法律来源的不可变 release 和 DCI 源工作区导出，均通过人工标签数据合同：

| 来源 | 语料文档 | 合格问题 | 正例关系 | 每题正例数 |
| --- | ---: | ---: | ---: | --- |
| PrivacyQA / legal_en | 4,657 | 167 | 990 | 1–10 |
| STARD / legal_zh | 55,348 | 1,542 | 2,694 | 1–10 |

2026-11-16 15:40（北京时间），两源 GPU 检索、200/1000 四个源级池，以及新版最终 suite 均已完成。
最终验收为 20/20 ready、0 blocked：dev200/test1000，1,200 题正例均为 1–10（平均 2.15），
40 个任务/档位池共 2,400 条记录满足容量、Lite⊂Full、完整且一致的 gold 和 dev/test 等价组隔离。
冻结摘要为 `23e24edd8ad51f71040e437ae442c938e16363e2abea4e1dd8bc22567d315761`。
[验收副本](project-progress-20261028/pos10-data-audit.json)和
[20 来源真实样本](project-progress-20261028/index.html#samples)已同步到进展页。
数据验收不代表新版 Qwen 实验完成；模型执行状态需查看相应运行目录。
此前 CPU 抽样预演和只读构池计划保留在来源目录的 `evidence/sampling-preview.json` 与
`evidence/legal-pool-plan.json`，最终验收以 `benchmark_v2_20261116_pos10_v1-inputs/audit.json` 为准。

## 数据规则

- 以 canonical qrels 中 score > 0 的文档数计数；只接受 1–10 篇正例的 query。
- 超限整题排除，不截断正例，不修改相关性阈值，不生成补充问句。
- 资格过滤、文本去重和全局等价组隔离后，20 项统一用 seed 42 重抽。
  继续使用稳定排序、既有 group-constrained 无放回算法、dev10/test50、smoke⊂dev。
- Lite200⊂Full1000，同题使用完整且一致的 gold；新配置与 manifest 都冻结该约束。
  运行器的 suite audit 再次检查上限。历史配置没有此字段时按原 200 篇上限验收。
- `legal_en` 替换为 [PrivacyQA](privacyqa_pos10.md)：官方 test 问题、人工句子相关性标注，
  train/test 全部真实句子作语料；反复出现在多个 query 行的同一 policy 句子只算一个检索文档。
- `legal_zh` 替换为 [STARD](stard_pos10.md)：原生法律咨询与人工法条相关性标注。

其余 18 项复用已有源级 Full 候选池与语料，CPU 生成新选中题目的嵌套 Lite/Full 元数据。
源身份与 query/qrels/文档 ID 哈希已核查；最终抽样后继续检查每题的候选数、正例完整性和嵌套关系。
旧 catalog 的部分绝对路径和 completion 文件摘要在存储迁移后过期；新 catalog 从旧冻结 suite 的
`shared_corpus.json` 定位当前路径，验证内部元数据摘要和冻结 source identity，并保留原 catalog 摘要供追溯。

## 在原 4×RTX 5090 节点执行

2026-11-17：新版三档 dev 标定已完成，但没有预算通过自动选择门槛。
按用户指定，完整 test 可使用固定 64,000 累计输入＋输出 token 预算：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CONCURRENCY=16 PREFLIGHT_CONCURRENCY=16 DRY_RUN=0 \
  bash scripts/launch_benchmark_pos10_qwen38_5090.sh --full --token-budget 64000 \
  --run-dir results/Qwen3.8-27B/benchmark-v2-full-runs/20261117-pos10-64k-c16
```

此入口保留数据、Frozen 沙箱、GPU 服务和 tokenizer 预检，直接执行 20 项 × 50 题 × Lite/Full
共 2,000 次 test，不重复 dev 标定。上下文窗口仍为 65,536；Lite/Full 共用 64k 累计预算。
首次启动创建指定目录，若目录已存在则拒绝覆盖；不指定 `--run-dir` 时分配新时间戳目录。
`launch.json` 和 `summary.json` 的 `budget_policy`
记录 `user-fixed`、选择原因及未应用开发集选档/标定投影，不能描述为自动标定通过。
该模式仅支持本地模型；不指定 `--token-budget` 的 `--full` 保持原三档标定流程。

如果作业被 kill，在原作业已停止后，用下面命令恢复同一个运行（后续每次恢复也用这一条）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 CONCURRENCY=16 PREFLIGHT_CONCURRENCY=16 DRY_RUN=0 \
  bash scripts/launch_benchmark_pos10_qwen38_5090.sh --full --token-budget 64000 \
  --resume-run results/Qwen3.8-27B/benchmark-v2-full-runs/20261117-pos10-64k-c16
```

恢复入口仅支持固定预算的 `--full`。如果先前使用自动命名目录，把 `--resume-run` 换成
原日志中 `output=` 对应的目录。它复用原 `backend.json`，在加载模型前核对数据、代码、
预算、并发、Frozen 身份及后端设置；原目录下的 `resume-attempts/` 保存本次预检与服务日志。
未进入 test 即中断时重新准备；已经创建 test 时使用底层 `--resume`，跳过完整且校验通过的结果，
重跑尚未封存的执行。保留代码、数据、权重、原运行目录和相同启动设置。
若旧服务仍存活并占用端口，预检会退出，不会自动杀掉其他进程。
28 项离线检查通过，包括模拟评测子进程 SIGKILL 后恢复和原文件保留；未进行真实 GPU kill/resume 测试。

从仓库根目录运行。两个新法律源使用既有 KaLM embedding 模型、8192 最大长度、float32、
query prompt、seed42 和四卡文档分片控制器。query 编码、检索和候选池发布仍由单控制器完成。

```bash
# 只检查已准备来源并打印构池计划；不启动模型或修改数据。
bash scripts/launch_legal_pos10_5090_4gpu.sh --check-only

# 两个新法律源编码/检索/构池，然后冻结和验收全部20项新版数据。
DRY_RUN=0 bash scripts/launch_legal_pos10_5090_4gpu.sh

# 数据与真实 Frozen 沙箱预检；不启动GPU模型。
bash scripts/launch_benchmark_pos10_qwen38_5090.sh --full --preflight-only

# 重新标定，再正式评测。
DRY_RUN=0 bash scripts/launch_benchmark_pos10_qwen38_5090.sh --full
```

新版 Qwen 入口默认并发 **8**，TP=4、BF16、context=65536、显存比例0.80，
沿用 `configs/runtime/frozen-debian13.json` 和原离线 RootFS bundle。
主机 Bubblewrap 准备与 Debian13 内 rg 等工具探测在模型加载前完成。
它依次执行 dev32k/64k/96k 各400次，通过预算选择后执行 test2000次；不会直接沿用旧96k结果。
原 `launch_benchmark20_qwen38_5090.sh` 默认仍指向旧版；支持 `BENCHMARK_SUITE` 显式覆盖。

构池和 Full 评测分开启动。已有正确构池产物可以复用；所有新源、池、suite 均不覆盖旧版本。
Full shell 入口每次创建新运行目录，重新执行标定，不是断点恢复命令。
新旧数据的 query 与法律来源不同，旧分数仅作历史参考。

## CPU 准备与验收入口

```bash
PY=/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python
"$PY" scripts/prepare_privacyqa_pos10.py
"$PY" scripts/prepare_stard_pos10.py
"$PY" scripts/prepare_benchmark_pos10.py inventory
```

固定来源、原始文件 SHA-256、转换规则、排除清单和 release/export 回执保存在
`/mnt/afs2/202611/embedding_models/dci-bench/eval_data/benchmark_legal_pos10_20261116_v1/`。
构池完成后，`prepare_benchmark_pos10.py freeze` 读取两个 `catalog-row.json`，
生成 `benchmark_v2_20261116_pos10_v1-inputs/` 的 catalog/config/audit 与新版 suite。
源文件获取与 CPU 转换不会加载 embedding 或 Qwen；GPU 构池由上方独立入口完成。
