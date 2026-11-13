# 模型适配边界

文档和代码评测共用 provider 适配入口，各自保留独立的协议、预算与评分规则。当前内置实现是 `openai-chat-completions`；抽出接口不等于已经验收其他商业 provider。

## Python：provider 与服务协议

`dci_bench/backends/adapters/` 负责模型路由、认证、provider 参数和预热通信。`phase4_runner`、`run_task` 和 `code_runner` 通过 `inspect_model_kwargs` 获取调用接线，不再拼接 `openai-api/...` 或设置 `responses_api`。

`backend.adapter` 选择适配实现；省略时使用现有 Chat Completions 实现。模型身份、服务端版本、生成配置仍由 backend manifest 记录并校验摘要。`build_api_manifest` 支持 `adapter` 和 `transport_version`；本地 manifest 构造器也支持 `adapter` 和显式 `thinking_policy`，非推理模型可声明 `not_supported` 而不发送 chat-template 参数。

新增模型按以下顺序处理：

1. 若通信及参数语义相同，复用现有适配器，只新增 manifest 和模型启动配置。
2. 若通信不同，在 `adapters/` 添加实现，并在同目录的 `ADAPTERS` 中注册名称。manifest 只能选择已注册实现。
3. 实现 `inspect_kwargs`：选择 Inspect provider，映射认证与 SDK 重试设置。需要新 Inspect `ModelAPI` 时，将其实现和注册也放在适配目录中；由它将原始响应转换成 Inspect 的标准消息、工具调用、终止状态及 usage。
4. 实现 `thinking_body` 和 `generation_kwargs`：翻译冻结生成配置。不要静默丢弃 provider 不支持的 seed、工具控制或禁用 thinking 语义。
5. 实现 `warmup_request` 和 `normalize_warmup_response`：转换 URL、认证、请求和响应。预热使用内部统一的 chat-shaped 探针表示；适配器可以转换成其他 wire 格式，最终仍必须通过相同的工具、终止状态及 usage 检查。

`BackendAdapter` 定义上述接口和 `validate` 版本检查。现有 OpenAI 实现可供仅有少量参数差异的适配器继承。协议生成值在 manifest 层先校验，provider 的字段转换由适配代码负责；适配代码本身仍需要审阅和真实端点验收。

## Pi：内部 bridge 与规范结果

`pi-dci/scripts/adapters/` 集中处理 Inspect bridge 的 wire 格式、usage 归一化和文本兼容处理。Pi 的执行循环通过规范消息/流接口使用结果；预算、工具授权、read 凭据、最终 schema 校验与评分要求留在 harness 中。

实际模型不直接连接 Pi。新 provider 应在 Python 适配层转换为 Inspect 标准输出，再经现有 bridge 进入 Pi。不要在主循环中新增模型名分支，也不要让适配器代替模型补造 read 凭据、有效最终答案或 token 用量。

## 验证与来源记录

- 先验证适配单测和本地 HTTP wire：auto/named/disabled tools、JSON 参数、截断、thinking、缓存 token 和累计 usage。
- 再运行真实端点预热与 smoke；CPU fixture 只验证扩展接口，不证明新 provider 已兼容。
- 所有 Python 适配源码及 `pi-dci/scripts/adapters/**/*.ts` 纳入执行源码指纹；pilot 源码归档和 code track 源码副本也包含适配文件。改动适配实现会改变运行身份。
- 评测协议约束无法满足时，报告不兼容；若研究目标确需改变协议，另建明确 profile，不能以适配名义更改现有对照条件。
