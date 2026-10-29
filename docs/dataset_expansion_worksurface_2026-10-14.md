# WorkSurface 下载重试与 DCI 补充包（2026-10-14）

WorkSurface 的 DCI 原始子集已下载、逐文件校验，并生成独立的通用 document 补充包。

- 原始数据：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/worksurface/raw`
- DCI 补充包：`/mnt/afs2/202609/embedding_models/dci-bench/eval_data/worksurface/dci_v1_20261014`
- 上游：`lhpku20010120/WorkSurface-Bench`
- 固定 revision：`3b3aef45f1e8f69416d6112c46e6e873cf522315`
- 下载工具仍为 `/mnt/afs/250010100/zxp/dci-bench/scripts/hfd.sh`，镜像为 `https://hf-mirror.com`。
- 501 个选中文件，共 **9,836,909 字节**；全部长度与上游 Git blob 哈希一致。目录包含完整的 1,151 条任务 JSONL、493 篇 Markdown、5 份文档 registry 和 2 份 README。
- 本次下载范围是 DCI 所需的文档检索子集，不包括表格、图和模型运行轨迹。

## 下载慢的实测原因与修复

旧下载包装脚本默认清除了代理环境变量。这是沿用之前代理连接失败时的设置；本次重新测试发现代理已经可用。

| 同次排查项目 | 观测 |
| --- | --- |
| AFS 写入 1 MiB，包含 fsync | 约 45 MB/s |
| 不使用代理、镜像 resolve、HTTP/1.1，12 秒短测 | 约 6 KB/s |
| 不使用代理、镜像 resolve、HTTP/2，12 秒短测 | 约 46 KB/s；后续测试仍显著波动 |
| 不使用代理、镜像 raw，12 秒短测 | 约 0.7 KB/s |
| 保留代理、镜像 raw，完整标注文件 | 1,755,473 字节，约 4.9 秒 |
| 不使用代理、HF 直连 | 连接超时 |

这些是排障时的单次观测，不是稳定带宽基准。镜像的 resolve 路径还出现连接超时，Range 请求返回 HTTP 200 的完整内容，不能据此依赖断点续传。

修复后包装脚本默认保留现有代理。`DCI_KEEP_PROXY=0` 仅作为明确需要直连时的覆盖选项。新增可选 HTTP/2 传输助手，读取 hfd 生成的文件清单，并校验上游哈希；`DCI_HFD_USE_RAW=1` 只对非 LFS Git 文件使用同一固定 revision 的 raw 地址。hfd.sh 本身未修改。此次中断的下载片段及损坏的旧元数据保留在原始目录的 `.hfd/` 下。

实际成功的重试命令：

```bash
cd /mnt/afs/250010100/zxp/dci-bench
DCI_KEEP_PROXY=1 DCI_HFD_HTTP2=1 DCI_HFD_USE_RAW=1 \
  HF_ENDPOINT=https://hf-mirror.com \
  bash scripts/download_expansion_data.sh worksurface
```

## 标注检查与适配结果

上游 1,151 条任务中，213 条为 `rag_only` 且 `required_surfaces == ["rag"]`。实际 persona 字段是 `source.persona`。Product Manager 与 Researcher 没有纯 RAG 查询，因此不生成空的 DCI 任务；其原始语料仍完整下载。

Operations Manager 内发现 60 组同名、相同正文的文档副本，区别仅为自动添加的来源任务头。例如 `t139__event_plan_11.md` 与 `t159__event_plan_11.md` 正文相同，但部分原始 qrels 只标注其中一份。DCI 按文档 ID 评分时会把未标注的等价副本判错。因此排除受影响的 **35 条完整查询**，保留全部候选文档，不补造标签。逐条排除原因与文档 ID 写入各任务的 `provenance.json`、`audit.json` 和补充包的 `annotation_audit.json`。

| DCI 任务 | 查询 | 文档 | 正例配对 |
| --- | ---: | ---: | ---: |
| `worksurface_backend_developer_rag` | 64 | 64 | 64 |
| `worksurface_logistics_manager_rag` | 48 | 100 | 48 |
| `worksurface_operations_manager_rag` | 66 | 226 | 66 |
| 合计 | **178** | **390** | **178** |

保留查询的原始 ID、问题文本、文档 ID 与 qrels 不变。使用各 persona 的完整候选文档范围，输出 canonical Parquet、DCI Markdown workspace、host 侧标签和完成标记。原始答案、证据、路由及其他标签不注入 agent 样本。

验证结果：

- 213/213 条原始 RAG 证据 span 均精确出现在所标文档中。
- 202/202 条原始填空题，将原始答案填回后，完整句子均精确出现在所标文档中。
- 保留的 178 条查询全部通过 qrels 闭合、原始标注一致性、scorer oracle、host data audit 和 Inspect 样本构造检查。
- 390 篇输出文档均通过内容哈希与协议检查，无超长文档或文档裁剪。
- 文档适配器 12 项测试通过；传输助手验证了 Git/LFS 两类哈希、部分文件保留与损坏内容拒绝。
- 验证边界为 CPU 数据及协议检查，未运行模型推理。

数据特点：保留集合中有 119 条问题明确包含所标文档的原始文件名，且多数是填空式事实查询。保持原题，不把它们解释为高难度开放检索。WorkSurface 原生的答案、路由与效率指标也不等同于 DCI 检索指标。

## 冻结与复核

补充包的 `catalog.json` SHA256：

```text
7062169b4e984235f589b3ef8b1823ca2a3b599c26fee93cd8ccbb8d44d4f205
```

`_COMPLETE.json` SHA256：

```text
c826ad6e7ca414b99caa45370468c8bb6f04981ac4fb25d332d78882dbf836cc
```

适配代码快照、实际构建命令、下载参数、包版本和审计均保存在补充包内。早先尝试构建无查询 persona 留下的 staging 目录未发布，保留用于排障。

```bash
python scripts/prepare_expansion_data.py \
  --suite /mnt/afs2/202609/embedding_models/dci-bench/eval_data/worksurface/dci_v1_20261014 \
  --verify-only
```

此补充包与已有通用版组合后为 **13 个任务、2,399 条查询**，使用两个独立 catalog。已有通用版、code 版及其 release index 均保持原样；原 release index SHA256 仍为 `3dcaa3edf6f855f49f66409156fa45c2184c359d80c9bcc19c51c7a3bbf7264d`。
