# 53 来源三档 smoke

固定每来源一条原始 query，同题运行 200/500/1000 三档，共 53 条独立 task/query、159 次执行。
选择键为 SHA-256(`20261026:task:original_query_id`)，不依据标签、召回分数或模型成绩选题。
这是运行诊断，不是全量 benchmark，也不用于调参后的独立 holdout 结论。

## 集群命令

在 **4 × H100 80GB** 节点上运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 DRY_RUN=0 \
bash /mnt/afs/250010100/zxp/dci-bench/scripts/launch_pool_smoke.sh
```

入口自动检查 `/usr/bin/bwrap`、`/usr/bin/rg`；缺失时使用 root 或免密 sudo，通过
APT（或 dnf/yum）安装 bubblewrap/ripgrep。包管理器不继承 Conda 的
`LD_LIBRARY_PATH`/`LD_PRELOAD`；APT 有下载重试和安装锁等待。
需节点能访问系统软件源。已装依赖则跳过安装。

安装后先测试 namespace，再运行真实 Inspect/bridge/Pi 合成响应预检和逐池临时工作区
检查。缺依赖、namespace 权限不足、数据或源码身份不匹配都会在模型加载前停止，
不绕过沙箱。预检使用与 Pi 相同的 user namespace 参数；不使用要求宿主网络
namespace 特权的简化探针。每个集群节点仍须独立通过真实工具链预检。

模型为 `/mnt/afs2/202610/Qwen3.8-27B`，BF16、TP=4、64k 上下文、最多 4 题并发；
档位累计 token 上限为 96k/128k/192k，每题时限 600 秒。启动时先核对接受过的
模型 config 与完整权重身份，再探测 CUDA 并启动 vLLM；不自动替换其它模型。
默认服务端口 29099，159 个 bridge 端口为 29131–29289，每次执行独立使用一个端口。

结果自动写入新的 `results/Qwen3.8-27B/pool-smoke-runs/<UTC timestamp>-<pid>/`，
启动和安装日志在 `results/Qwen3.8-27B/job-logs/`。结果目录排他创建；失败证据保留，
不覆盖旧结果。`summary.json` 的 `smoke_passed=159` 且根 `_COMPLETE.status=complete`
才表示 smoke 全部通过；完成标记不代表模型达到质量门槛。

可选检查命令：

```bash
bash scripts/launch_pool_smoke.sh --dry-run
bash scripts/launch_pool_smoke.sh --data-preflight-only
bash scripts/launch_pool_smoke.sh --preflight-only
```

最后一条同样自动安装依赖、运行真实沙箱检查，但不加载模型。所有 CPU 检查都不启动 GPU。

## 数据与可复现边界

源入口：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/retrieval_pools/catalog.json`。
smoke 数据：同级 `retrieval_pool_smoke_20261026_v2/`。

准备器原样保留所选题目的候选 doc IDs、分数、顺序、补回/移除审计与 qrels，
不从 1000 档截取较小档。只复制三档实际所需正文的任务内并集，逐文件核验 SHA-256；
全库编号映射仍保留在 host metadata。运行时仅当前题目的候选正文进入临时沙箱。

源 catalog 的 `completion_sha256` 是迁移前值。准备器用保留的
`provenance/pool-K-controls.tar.gz` 验证该历史 seal，只接受已有迁移记录中的
`candidate_pools.json` 压缩编码、manifest 档位/排名路径和共享语料引用变化；
其余 metadata/正文清单声明必须保持相同，当前 metadata 的全部文件 hash 仍需匹配。
新 plan/audit 同时记录历史凭证和当前 completion hash，原 catalog 不覆盖。
本次不重新计算全部 query 的迁移等价摘要或全来源正文 hash；严格正文核验范围为
smoke 选中候选。`retrieval_pool_smoke_20261026_v1/_FAILED.json` 保留了初次发现
历史 hash 差异的记录。

源码、Pi 依赖、三档 contract 与所选池完成文件身份在独立 bundle 中封存。
验证记录及准备日志在 `results/maintenance/pool_smoke_prepare_20261026/`。

首次封存入口的 CPU 预检发现整数标签 `1` 与浮点标准化 `1.0` 的 manifest
哈希约定差异。v2 运行器验证源 JSON 或旧式浮点标准化 JSON 的完整哈希，结果中的
qrels 哈希仍采用原有标准化约定；标签文件不改写，数值/文档变化仍会拒绝。
旧 bundle 和失败日志保留。当前入口固定到 `releases/protocol-freezes/20261026-pool-smoke-v4`；v3 修复 Python 导入路径，v4 修复原始 qrels 摘要校验并增加真实发布链路的 CPU 合成响应预检。

## 清理记录

本轮仅删除无引用、未封存的旧 confirmation v1 依赖副本，以及一处可再生 Python
缓存。删除前保存文件清单、占用统计、引用检查和本机 `/proc` 打开文件检查；
正式结果、冻结协议、续跑产物、失败诊断保留。

`results/maintenance/cleanup_20261026_smoke/summary.json` 记录移除
298,364,928 allocated bytes（约 284.5 MiB），受保护文件哈希未变。
此数字不包含新 smoke 必需产物的占用，也不由共享文件系统的 `df` 差值推算。
