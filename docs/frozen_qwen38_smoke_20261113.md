# Qwen3.8-27B Frozen / Open Bash smoke（2026-11-13）

状态：已设计，未启动模型。目标是验证中性提示词、冻结 Debian Bash、read 凭据、Lite/Full 数据与结果身份的完整链路。不是正式 benchmark，也不是仅改变 Runtime 的消融。

## 实验矩阵与固定条件

- 10 领域 × 原生中英 2 语言 = 20 个任务；每任务沿用冻结 smoke split 的 1 个 query，分别运行 Lite200 和 Full1000，共 **20 个独立 query、40 次执行**。
- 使用 `benchmark_v2_20261103_v1`，seed 42，smoke 是 dev 子集，不重新抽样、不生成或改写候选池。两档同 query、同 Prompt、同预算，Lite ⊂ Full。
- 每题累计输入+输出 64,000 token；context 65,536、单次输出 1,024、工具返回 8 KiB、最多 12 轮搜索/24 次工具调用、最多一次 final 修复。默认无单题/整轮时间上限；Bash 每次默认 10 秒、最多 60 秒。
- 使用现有 BF16 模型 `/mnt/afs2/202610/Qwen3.8-27B`，4×RTX5090、TP=4、temperature=0、seed=0、thinking=false；显存利用率 0.80、max_num_seqs=4、max_num_batched_tokens=2048、eager，保留 GDNWarmupWorker。
- Runtime 为 `configs/runtime/frozen-debian13.json` 指定的 Debian 13 离线 Bundle；模型可自由选择其中已安装的命令、内建命令、管道和脚本。至少一次非空 read 仍是提交门槛。
- 当前 runner 按 task/tier 顺序分批，每批只有 1 题。`concurrency=4` 是服务与 runner 容量配置，**不代表这 40 次执行会同时并发 4 题**，本实验不用于吞吐验收。

[机器可读实验计划](../configs/experiments/qwen38_frozen_open_bash_smoke_20261113.json)记录 40 行 task/tier/query、数据与 pool 摘要、两档新协议 SHA 和 Runtime Identity。它是审阅记录，现有 launcher 直接读取冻结 suite 和 runtime config，不读取该计划 JSON。

2026-11-13 路径迁移：本次 suite 使用 `/mnt/afs2/202611/embedding_models/dci-bench/eval_data/`。36 个 Lite/Full 共享语料引用由 `202609` 更新为 `202611`，并同步引用、pool 和 suite 的校验摘要；query、候选池、qrels 和语料内容未改。原始引用及清单备份、迁移前后摘要见 [迁移记录](../results/maintenance/data-path-20261113-j0jywxbt/migration.json)。

| 领域 | 英文 query ID | 中文 query ID | 执行数（两档） |
|---|---|---|---:|
| encyclopedia | `c3549b20b2bbb3d669fe99c5b813dd3c2674fbc2` | `DEV_1628_QUERY_2` | 4 |
| general_qa | `457704` | `open_qa.jsonl#row2741` | 4 |
| scientific | `0a5ffe4697913a57fda1fd5a188cd5ed59bdc5c7` | `benchmark/ts/test.tsv#row478` | 4 |
| medical | `8_NHLBI_QA_XML/0000057.xml#0000057-3` | `72863768dadef62e738712d9d8c6012e` | 4 |
| finance | `7734` | `finance.jsonl#row311` | 4 |
| legal | `personal or bodily injury exception to liability cap` | `00050` | 4 |
| enterprise | `expertwritten/110` | `cfqa-test-2837` | 4 |
| web | `1020` | `78d57ba0f520349e94fb8b40bfa03501` | 4 |
| software | `23404` | `183629` | 4 |
| argumentation | `test-international-ghbunhf-pro02a` | `query-claim-0211-00000` | 4 |

## 目标节点启动命令

在已挂载上述模型与 `/mnt/afs2/202611/embedding_models/dci-bench/eval_data/benchmark_v2_20261103_v1` 及其源 corpus、具备项目 Python/Node 环境的 4 卡 5090 节点执行。Launcher 先准备并校验 host `/usr/bin/bwrap`（Frozen 不要求 host ripgrep），再从显式离线 Bundle 导入或复用节点缓存；Frozen Runtime 不下载、不回退到 host runtime。默认 dry-run 不安装依赖、不解包 Bundle。

导出冻结配置和离线 Bundle；launcher 会先准备 host 依赖，再按内容复用热缓存或从 Bundle 导入冷缓存。容器重启清空 `/tmp` 后可再次执行同一命令。仅预检、不启动 GPU（建议先单独执行）：

```bash
cd /mnt/afs/250010100/zxp/dci-bench
export DCI_RUNTIME_CONFIG="$PWD/configs/runtime/frozen-debian13.json"
export DCI_RUNTIME_BUNDLE="$PWD/results/frozen-runtime/20261111-xaDmpt/release-v2/rootfs.tar"
bash scripts/launch_benchmark20_qwen38_5090.sh --preflight-only
```

启动 40 次真实模型 smoke：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_GPU_MEMORY_UTILIZATION=0.80 \
DCI_RUNTIME_CONFIG="$PWD/configs/runtime/frozen-debian13.json" \
DCI_RUNTIME_BUNDLE="$PWD/results/frozen-runtime/20261111-xaDmpt/release-v2/rootfs.tar" \
DRY_RUN=0 bash scripts/launch_benchmark20_qwen38_5090.sh
```

执行入口本身会按 host 依赖、Frozen offline prepare/实际 probe、40 个数据预检与合成 Inspect 桥接、GPU/端口检查、GDN kernel probe、模型服务预热的顺序执行，最后执行 smoke。任一前置检查失败则停止，保留日志。冷 cache 必须设置 `DCI_RUNTIME_BUNDLE`；热 cache 可复用同一配置而省略它。不要用历史 `run_pool_smoke.sh`，它执行 sealed bundle，拒绝 Frozen 参数。

40 个数据预检默认并发 4 个，可在上述命令前设置 `PREFLIGHT_CONCURRENCY=8`；设为 `1` 可串行排查。该参数只影响 CPU 数据预检，不改变模型评测并发、样本顺序或预算。每项独立日志保留于 `data-preflight/logs/`，完成进度打印到 `launcher.log`，全通过后的 `data-preflight/preflight.json` 记录实际配置和耗时。Frozen runtime 的完整性及 bwrap/Bash/rg 探测继续执行。

默认模型端口 29099、桥接端口 29131–29134；冲突时可显式设置 `VLLM_PORT` 和 `BRIDGE_PORT`。每次启动创建新的 `results/Qwen3.8-27B/benchmark-v2-smoke-runs/<UTC>-<random>/`，不覆盖旧运行。本 shell launcher 不提供整轮恢复参数；不要手工将旧 Host 目录作为本轮输出。

## 验收与结果解读

1. 数据预检必须是 20 任务、40 次执行，无 blocked task；Lite/Full 使用上表同一 query。根目录 `smoke/_COMPLETE` 和哈希验证闭合。
2. 目标为 completed=40、valid=40；失败项保留具体阶段、错误、trace，不以重抽题或放松 read/预算门槛补足。
3. 每个 final/trace/run-manifest/summary 的 Runtime Identity 与配置一致；合同 SHA 对应新中性提示词。模型服务身份保持 Qwen3.8-27B BF16。
4. 分语言/领域/档位报告 Recall@10、F1@10、nDCG@10、无效项与零召回项；记录 token 用量、Bash 命令、read 凭据与修复次数。核查实际命令选择，但不把“是否使用 rg”或“命令种类数量”作为通过条件。
5. 硬件、耗时和 GPU 利用率仅作参考，不设 P95/吞吐质量门槛。40 次有效执行验证链路，不代表正式排名或 Frozen 优于 Host。

本次修改同时影响提示词与 Runtime；历史 11 月 5 日 Host/rg-first smoke 只能作诊断参考。若要隔离 Runtime 影响，另跑一组相同新提示词、query、预算与模型的 Host 对照，共另增 40 次，本轮未安排启动。

保留 `launcher.log`、`runtime.json`、`backend.json`、`gdn-kernel-preflight.json`、`server.log`、`warmup.json`、`smoke/launch.json`、`smoke/summary.json`、`smoke/gpu.csv` 及各样本产物。

## 当前现场验证边界

2026-11-13 已用现有 Bundle 完成离线缓存准备及真实 Frozen V2 Lite 沙箱预检，外部模型调用为 0。当前开发容器未挂载默认 benchmark suite；上表与计划来自保存的准备清单和预检 launch 记录，不能替代目标节点的数据/正文复核。本次不启动模型或 GPU smoke。

本轮定向验证全部通过（不含真实模型推理）：

- 72 项 Python unittest：46 项合同 / Inspect 控制 / launcher 回归，23 项任务指令与 Code Track 回归，3 项实际 Pi HTTP / Bubblewrap 测试（其中 Frozen 同时覆盖 Lite 与 Full；未跳过）。
- `pi-dci` 完整 `npm run check` 通过；启动脚本 `bash -n` 和父仓库 / 子模块 `git diff --check` 通过。
- 进度页与原生 SVG / PNG 同步，23 项桌面、移动端与 srcdoc 浏览器检查通过；40 行实验清单、20 个 query 配对、数据摘要、源码指纹及两档新合同 SHA 一致。

可复跑核心定向检查：

```bash
DCI_PY=/mnt/afs/250010100/miniconda/envs/dci-bench/bin/python
"$DCI_PY" -m unittest tests.test_document_pool_budgets tests.test_document_pool_v2_budgets tests.test_harness_pilot tests.test_benchmark20_launcher -v
"$DCI_PY" -m unittest tests.test_task_instructions tests.test_code_online -v
DCI_TEST_FROZEN_RUNTIME_CONFIG="$PWD/configs/runtime/frozen-debian13.json" \
  "$DCI_PY" -m unittest tests.test_open_bash_wire -v
```
