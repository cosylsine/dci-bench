# Document 与 Code 双轨评价

实现日期：2026-10-14。两条轨道共享“冻结语料、host-only 标签、固定查询集合、失败样本保留、结果可审计”的原则，使用独立的输出契约、指标和报告。两轨分数不合并平均。

| 项目 | Document | Code |
|---|---|---|
| Python 入口 | `dci_bench.scoring.document` | `dci_bench.scoring.code` |
| 兼容入口 | 原 `scoring.retrieval` 和包级函数保持不变 | 显式选择 code |
| 契约 | 原 `protocol/contract.json`、`dci-retrieval-scoring-v2` | `protocol/code.py`、`dci-code-retrieval-scoring-v1` |
| 返回单元 | 原始 `ranked_doc_ids`，1–10 项 | 仓库相对路径 `ranked_files`，默认最多 20 项；可选 `spans` |
| 弃答 | 沿用现有非空输出规则 | 显式 `abstain=true` 且空排名 |
| 标签 | 文档 qrels | 冻结仓库文件清单、文件 gold、可选行/函数/功能证据 |
| 基础指标 | 原 Recall/F1/nDCG，原默认 k 不变 | 文件 P/R/F1/nDCG、Hit、全 gold 召回、MRR、选择性成功 |
| 额外指标 | 原协议不变 | 行覆盖、函数召回、功能证据召回、BCY |
| 执行范围 | 现有 Inspect/Pi、六任务启动器继续工作 | 独立在线入口 `scripts/run_code_task.py`，同时保留离线评分 |

`document.py` 是原评分函数的显式导出，不复制或改动旧算法。此次没有修改现有 document 的任务表、提示词、Top-10、结果 schema、协议指纹或续跑逻辑。

## 论文依据与 DCI 扩展

[Agent Retrieval Bench 的图 1 与 §3](https://arxiv.org/html/2607.24882v1) 将编码工作流中的下一步上下文获取独立出来：query 与冻结的 `repo@base_commit` 输入检索流程，输出相关文件或弃答。四类正例任务是 `code2test`、`comment2context`、`trace2code` 和 `edit2ripple`。相关性取决于下一步工作需要什么；已给出的文件、堆栈中的显眼路径不自动成为正例。

本实现采用其文件召回、MRR、显式 no-gold 和预算上下文收益思路。函数召回、功能证据集合召回是 **DCI 的新增定义**。文件 P/F1、binary nDCG 和全 gold 召回作为辅助诊断。论文的行级诊断与 canonical token BCY 是两套不同实验；本实现行覆盖直接评价显式提交的 source spans，不声称复现其旧版 8,000 字符 chunk packing。

这些指标衡量检索到的上下文，不表示代码已通过测试、分支已执行或补丁已修复问题。最终选中的 spans 也不是工具实际读取凭据。需要评价真实探索轨迹时，采集端必须从可信 read receipts 构建选中区域，并单独声明这一输入语义。

## Host 标签与快照身份

`labels.json` 是 host-only 数据，不放入 agent 的仓库工作区：

```json
{
  "schema_version": "dci-code-labels-v1",
  "queries": {
    "sample-id": {
      "repository": "owner/repository",
      "base_commit": "完整的40或64位小写Git提交哈希",
      "task": "trace2code",
      "files": {
        "src/parser.py": {
          "sha256": "canonical UTF-8文本的64位SHA256",
          "line_count": 120,
          "content_tokens": 400
        }
      },
      "tokenizer": "regex_code_tokenizer_v1",
      "gold_files": ["src/parser.py"],
      "gold_spans": [
        {"path": "src/parser.py", "start_line": 20, "end_line": 30}
      ],
      "gold_functions": [
        {"function_id": "parser.parse", "spans": [
          {"path": "src/parser.py", "start_line": 20, "end_line": 30}
        ]}
      ],
      "gold_functionalities": [
        {"functionality_id": "parse-request", "evidence_sets": [
          [{"path": "src/parser.py", "start_line": 20, "end_line": 30}]
        ]}
      ]
    }
  }
}
```

上面展示结构，哈希占位符不能直接用于评分；可执行数据见 [四查询合成样例](../examples/code_retrieval/README.md)。`task` 接受非空任务名，支持自有代码任务。每题有自己的候选清单和 base commit，不将不同版本仓库合为一个候选库。

- `files` 列出完整候选域，路径保持原样、区分大小写，禁止绝对路径、`..`、`.`、反斜杠和空路径段。不会移除扩展名、URL 解码或自动改写路径。
- `line_count` 按 LF 分隔的源代码行计算，末尾 LF 不额外产生一行；行区间为 **1-based、首尾均包含**。不使用会把换页符等也当作换行的 `str.splitlines()`。保留原缩进和字节，不经过 document Markdown 折行器。
- `file_metadata(text)` 计算文本 SHA256、LF 行数和官方 regex token 数。读取时使用 `path.read_bytes().decode("utf-8")` 保留 CRLF，避免文本读取自动转换换行。
- 所有 token 统计均由 host 从冻结文本生成。仅做文件/行评分时，可以同时省略 `tokenizer` 和所有文件的 `content_tokens`，BCY 则为 `null`；不接受混用 tokenizer 或只给部分文件成本。
- `gold_files` 只能包含候选路径；细粒度标签也必须落在这些文件的有效行内。不能从补丁路径、函数名匹配或“文件被找到了”自动推导完整覆盖标签。
- 缺少 `gold_spans` / `gold_functions` / `gold_functionalities` 表示该层标注不可用；应省略字段，不填空列表。代码不把缺失标注当零分或满分。
- 空 `gold_files` 必须另有 `no_gold_kind: "natural"` 或 `"counterfactual"`，且不能带正例细粒度标签。未标注的空 gold 会中止评价。两类 no-gold 分别汇总。

纯评分函数验证清单结构和预测的快照身份，不重新读取仓库。在线入口会先验证实际候选文件的字节 SHA256、LF 行数、token 成本和完整文件树；ARB 模式还调用 `verify_arb()` 验证导出完成标记与全部产物。数据适配器仍须负责正确冻结源版本、排除修复后文件及标签泄漏；合成样例不是 ARB 基准成绩。

## 模型输出与预测封装

模型只生成如下 JSON：

```json
{
  "ranked_files": ["src/parser.py"],
  "abstain": false,
  "spans": [{"path": "src/parser.py", "start_line": 20, "end_line": 30}]
}
```

`spans` 可以省略；其路径必须在 `ranked_files` 内。重叠、重复区域按文件求并集，不重复计分。重复文件路径、未知文件、越界行、格式错误使该预测无效。默认 cap=20，可以显式调整；这不改变 document 的 cap=10。

弃答仅接受 `{"ranked_files": [], "abstain": true}`（可有空 `spans`）。正例弃答属于有效输出，但检索分为零；格式有效性与检索质量分别记录。

采集端将模型输出封装到查询 ID 下，并写入源语料身份：

```python
from dci_bench.scoring.code import snapshot_identity, score_query

# reference 来自上述 host labels；output 为模型返回的 dict 或 JSON 字符串。
record = {**snapshot_identity(reference), "output": output}
result = score_query(record, reference)
```

`snapshot_identity` 返回 `repository`、`base_commit` 和 `corpus_sha256`；后者是候选 `files` 清单的 canonical JSON SHA256，**不包含 gold**。这些字段应在采集时绑定到实际使用的快照，不能在评价时给未知来源的历史预测补填匹配值。

`predictions.json` 为 `{"sample-id": record, ...}`。计划内缺失 ID 自动计为 `missing_prediction`；计划外 ID、重复 JSON key 和损坏标签导致 CLI exit=2，不发布分数。运行失败用 `{**identity, "failure_reason": "token_limit"}` 表达。若采集器同时保留失败原因与部分输出，失败原因优先，不能凭部分输出或空排名把失败变成成功弃答。

## 指标定义

令 `G` 为非空 gold 文件集合，`W_k` 为排名前 k 项，`P` 为显式选择的源代码行并集，`L` 为 gold 行并集。code 默认 k 为 5、10、20。

| 字段 | 定义及适用范围 |
|---|---|
| `file_recall_at_k` | `|G ∩ W_k| / |G|`，仅正例 |
| `file_precision_at_k` | `|G ∩ W_k| / |W_k|`；空排名为 0 |
| `file_f1_at_k` | 上述 P/R 的调和平均 |
| `file_ndcg_at_k` | 二元文件 gold、log2 折扣，理想排名含 `min(k, |G|)` 个正例 |
| `file_hit_at_k` | 至少命中一个 gold 文件 |
| `all_gold_files_at_k` | 命中全部 gold 文件 |
| `reciprocal_rank` | 首个命中位置的倒数；未命中为 0，按完整提交排名计算 |
| `line_coverage` | `|P ∩ L| / |L|`；仅有行标签的题 |
| `line_precision` / `line_f1` | 分母分别为实际选择行并集、P/R 调和平均；多读噪声行降低精度 |
| `function_recall` | 已**完整覆盖全部标注 spans** 的函数数 / 标注函数数；只碰到函数的一行不算完整召回 |
| `functionality_recall` | 已满足功能数 / 标注功能数；一个功能可有多个替代 `evidence_sets`，完整覆盖任意一套即可，不要求覆盖替代方案的并集 |
| `selective_success_at_k` | 正例 top-k 有命中，或显式 no-gold 有效弃答 |
| `no_gold_abstention` | no-gold 中有效弃答比例；正例不参与分母 |
| `accepted` / `abstention` | 在计划查询中的有效非弃答 / 有效弃答比例；无效输出两者均为 0 |

行、函数、功能指标评价所有显式提交 spans，不随文件 k 改变。文件 ID 命中不自动提供 source spans；有行标签但未提交 spans 时，覆盖为 0。函数与功能指标是证据覆盖代理量，不能替代真实行为测试。

### 预算上下文收益 BCY

默认预算为 **4,000 / 8,000 / 16,000 / 32,000 regex tokens**，与模型 tokenizer 的推理预算分开。按官方 [bcy_curve.py](https://github.com/eyuansu62/agent-retrieval-bench/blob/main/src/agent_retrieval_bench/bcy_curve.py) 的 canonical τ=1 规则：

1. 按排名装入完整 canonical 文件文本，header 为 `### {path}\n`。
2. regex 为 `[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^A-Za-z0-9_\s]`。header 计入预算，非空文本额外占 1 个 separator token。
3. 在预算边界做前缀截断；只有 header、没有正文的文件不算命中。
4. 某 gold 文件至少有 1 个正文 token 被装入才记为覆盖；空正文 gold 不获得信用。每题 BCY 为已覆盖 gold 文件数 / gold 文件数，再在正例中宏平均。

`bcy_at_B_tokens` 是**重建的上下文收益**，不是实际工具读取覆盖。它独立于提交的 `spans`。若排名用尽仍未填满预算，`context_packing[B].ranking_exhausted_before_budget=true`；排名截断时不能把该值当作完整 retriever 的无截断 BCY。canonical corpus text 如果与 ARB 发布语料不同，也不能声称官方 leaderboard 复现。

### 汇总与失败

每个指标保存 `mean`、`eligible_queries`、`valid_only_mean`、`valid_queries`。全样本均值在该指标有标签的所有计划查询上计算，缺失/失败预测按零计；无标签题不进入该指标分母。没有任何适用题时均值为 `null`。另有 `by_task`、`by_stratum`、失败数量和逐题结果。

MRR 是汇总 `metrics.reciprocal_rank.mean`。文件/行分数、有效输出率、no-gold 表现需要同时读取，不能用 valid-only 代替总体结果。标签错误是数据/preflight 错误，不能转成模型零分。

## 在线任务

两轨评分的执行时机和位置现在一致：**每条查询的 Pi solver 结束后，由 Inspect 调用宿主端 scorer**。Code 使用 `tasks/code_retrieval.py`，直接调用已有 `scoring.code.score_query()`；Document 继续使用原任务和评分器。汇总阶段校验逐题评分，并以 Inspect 的终止错误、超时和 token 限额为准；已保存的有效部分输出不能覆盖失败状态。

Inspect 数值预览也按各指标的适用查询汇总；完全无标注的指标不显示数值。权威 `code-result.json` 保留完整的 `null`、适用分母和终止状态修正。

Code 使用独立运行契约 `protocol/code_runtime.py`（`dci-code-online-v1` / `code-v1`），模型返回 `ranked_files/abstain/spans`。共享 Pi 进程生命周期、Inspect bridge、Bubblewrap 沙箱和预算控制基础设施。默认仍是每题 600 秒、累计输入输出 128,000 tokens、单次 context 65,536、输出 1,024、temperature=0、seed=0、thinking=false；document 的实验 profile 不注入 code 提示词或最终输出协议。

源码保留嵌套目录和原始字节。Code 的 `read` 接收 `path/start_line/end_line`，行号从 1 开始、首尾包含；返回实际可见区间，20,000 UTF-8 bytes 截断时标记最后一行是否完整。每次读取校验源文件哈希。最终提交（包括弃答）都需要至少一次非空 `read`；这不表示已读过每个返回文件，也不表示所有提交 spans 都有对应读取凭据。Gold 仅保留在宿主端闭包，`Sample.target` 为空，候选文件清单不含 gold。

先用合成样例做预检，不发模型请求、不创建结果：

```bash
python scripts/run_code_task.py \
  --queries examples/code_retrieval/online_queries.json \
  --labels examples/code_retrieval/labels.json \
  --query-ids demo-trace --preflight-only
```

需要现有 Python/Inspect、Node/Pi 依赖，以及可信的 `/usr/bin/bwrap` 和 `/usr/bin/rg`。预检检查实际沙箱、候选文件和桥接端口；依赖缺失会在评测前失败。

CPU mock 跑通 Inspect → Pi → 源码读取 → 在线评分与结果发布：

```bash
python scripts/run_code_task.py \
  --queries examples/code_retrieval/online_queries.json \
  --labels examples/code_retrieval/labels.json \
  --query-ids demo-trace \
  --mock-script examples/code_retrieval/mock_trace.json
```

连接已经启动的模型服务时，将 `--mock-script` 换成 `--backend-manifest /path/to/backend-manifest.json`，或 `--model openai-api/vllm/served-model-name --base-url http://127.0.0.1:17892/v1`。使用 manifest 时校验其模型、服务地址、context 和 code 生成配置，拒绝不一致的覆盖值；脚本不启动模型服务。

已完成的 ARB 导出可直接传给 `--code-root`，无需手动展开 compact labels：

```bash
python scripts/run_code_task.py \
  --code-root /path/to/completed-arb-export \
  --task trace2code --query-ids sample-id \
  --backend-manifest /path/to/backend-manifest.json \
  --max-concurrency 1 --bridge-port 13131
```

`--query-ids` 保留指定顺序；显式 `--all-queries` 才跑所选任务的全部查询。只接受带有效 `_COMPLETE.json` 的导出，不接受 `.building-*`。ARB 发布语料缺少可靠的源行/函数/功能标注时，这些指标保持 `null`，不会从文件 gold 推导。

自定义 `--queries` 是列表，条目必须恰有 `query_id`、`query`、`workspace`；workspace 相对于 queries 文件所在目录解析，其下仅 `corpus/` 是候选域。`--labels` 使用本文的完整 `dci-code-labels-v1`，查询 ID 集必须一致。一个样例：

```json
[{"query_id":"sample-id","query":"Find relevant repository context.","workspace":"snapshots/repo-a"}]
```

结果默认保存为新的 `results/code/<timestamp>-<suffix>/`，也可指定全新的 `--output-dir`。拒绝覆盖旧运行、写入候选 workspace 或 ARB 导出根目录；目前不支持在线续跑。

- `run-manifest.json`：查询顺序、快照、模型/服务、生成配置、独立运行及评分契约、标签与源码哈希。
- `samples/<query-id-sha256>/`：Pi `final.json/trace.json`、即时 `scorer-prediction.json/scorer-score.json`、核对终止状态后的 `prediction.json/score.json`。原始即时结果保留。
- `predictions.json` / `code-result.json`：所有计划查询的最终预测、汇总、适用分母、valid-only 和失败类别。
- 独立 code `_COMPLETE.json`：产物哈希和有效/失败数量。完成标记本身不代表模型输出全部有效。

在线 CLI 退出码：0 为全部输出有效（检索分可为零）；1 为完成但有样本失败；2 为配置、数据、基础设施或完整性错误。与离线 CLI 的退出码语义区分读取。`run_task.py`、`run_all.py` 和 `run_expansion_task.py` 继续服务 document，不自动混入 code 任务。

## 离线评分与验证边界

在项目根目录运行，无需模型服务、GPU、下载或执行目标代码：

```bash
python -m dci_bench.scoring --track code \
  --labels examples/code_retrieval/labels.json \
  --predictions examples/code_retrieval/predictions.json
```

可用 `--metric-ks 5 10 20 --max-files 20 --token-budgets 4000 8000 16000 32000` 显式冻结配置。默认输出到 stdout；`--output /existing/directory/new-report.json` 原子发布新文件，拒绝覆盖现有文件。报告记录输入文件 SHA256、有效评分配置及其哈希。存在样本失败仍返回 0，表示报告生成成功，应读取 `invalid_outputs`；输入/配置/发布错误返回 2。

Document 的原线上入口继续使用；新增离线入口也可调用：

```bash
python -m dci_bench.scoring --track document \
  --labels /path/to/qrels.json --predictions /path/to/document-predictions.json
```

Code 报告与完成标记使用独立 schema，不生成 document `task-result.json`。验证覆盖 CPU 单元测试、document 回归、合成样例，以及实际 Inspect bridge → Pi/Bubblewrap → 本地模拟 HTTP 服务 → 在线 scorer 的完整链路；未运行真实模型/GPU或完整代码基准。这里的覆盖率仍是检索证据覆盖，不是代码测试执行覆盖率或补丁正确率。
