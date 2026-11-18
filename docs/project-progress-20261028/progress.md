# DCI-Bench 全量结果与分析（2026-11-18）

Qwen3.8-27B：20任务 × 50道测试题 × Lite/Full，完成2,000次执行，有效1,981次（99.05%）。

| 指标 | Lite200 | Full1000 | Full − Lite |
|---|---:|---:|---:|
| Recall@10 | 64.96% | 62.22% | -2.74 pp |
| nDCG@10 | 61.96% | 58.75% | -3.22 pp |
| F1@10 | 42.87% | 41.82% | -1.05 pp |

失败记零，任务等权；F1按实际返回长度计算。完整20任务指标见[详细页面](index.html#results)与[CSV](qwen-full-results.csv)。

- 中文百科Recall/nDCG为100%，中文科学94%；英文网页Full Recall为22.43%，扩池后nDCG/F1分别下降12.68/12.00 pp。
- 1,879次执行因预算预留触发收尾（93.95%）；1,997条完整usage平均34,149 token，使用64k预算的53.4%。Lite/Full平均搜索轮数7.83/7.92。
- 网页题422：Lite通过Olympic/Murashige找到线索，Recall90%；Full停留于variety/abstract和截取结果，在剩余45.5k时收尾，Recall0。
- 19次失败：14未知ID、2 JSON schema、1 API超时、2 API连接错误；409次修复尝试。
- 4×RTX5090、TP4、BF16、并发16、Frozen Debian13；正式测试10h51m，184.23次/小时，P50/P95 234/408秒，GPU均值63.3%。
- 2,000样本指标独立重算一致；样本、批次和摘要完成哈希一致。新版池正例1–10、Lite⊂Full、gold一致与完整覆盖核验通过。
- 优先复盘困难题预留收尾、宽泛关键词与截断行为，以及最终ID准确性。

[完整分析](index.html#analysis) · [结果JSON](qwen-full-results.json) · [当前状态](benchmark-v2-status.json) · [来源样本](index.html#samples) · [更新前记录](history/before-qwen-full-20261118/index.html)

比较与验证边界：本次为user-fixed 64k，不能据此认定64k通过自动标定或为最优预算；与旧数据96k结果不能直接作预算因果比较。中英文来源不同，任务各50题，语言均值不能单独归因于语言能力。F1差值95%区间为[−2.31, +0.10] pp；单题轨迹说明失分机制，尚未量化其整体贡献。3个API异常末请求缺少实际usage；结果与完成标记已全量核验，原始trace附件未逐一重哈希。耗时和GPU利用率仅作当前硬件参考。正文样本展示核验20篇，18个复用源未重新全扫历史正文；法律源转换阶段的整题过滤与suite抽样排除分别统计。CFQA为公开年报知识且标注许可不延伸到原PDF；T2Ranking重新抽样结果不等于上游官方测试成绩。KaLM为补正例前的离线构池指标，补正例后的100%覆盖不代表Agent成绩。
