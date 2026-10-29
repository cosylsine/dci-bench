# 冻结代码发布目录

维护代码在仓库 `scripts/`、`dci_bench/`、`pi-dci/`；这里保存不可变运行包、依赖及配置。

当前 159 次评测入口：`bash scripts/launch_pool_smoke.sh`，固定到 `protocol-freezes/20261026-pool-smoke-v4/`。结果仍输出到 `results/Qwen3.8-27B/pool-smoke-runs/`。

2026-10-26 将此前误放在结果/数据目录中的运行代码迁入这里。迁移映射及校验在 `results/maintenance/code_layout_20261026/`。旧冻结源码和历史凭证中的路径保持原始记录；跨包依赖软链接单独记录迁移。历史包保留用于审计，不等同于已在新路径重新验证的运行入口。当前 pool smoke v4 的新路径单独验证。

`result-code/` 保存历史独立脚本；`run-launchers/` 保存每次运行使用的 launcher 代码；`data-pipeline/` 保存数据构建运行包；`archives/` 保存历史代码压缩包。数据正文、标签、模型权重和运行结果继续保留在各自的数据/结果目录。

新增冻结包必须写入本目录，不得写入 `results/` 或 AFS2 数据目录。依赖副本不提交到 Git；源码和入口仍由仓库维护。
