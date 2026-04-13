---
title: NVIDIA NIM 响应网关
sdk: docker
app_port: 7860
pinned: false
---

# NVIDIA NIM 响应网关

这是一个面向公开使用的 NVIDIA NIM 兼容网关，同时支持 OpenAI `/v1/responses` 和 Anthropic `/v1/messages` 风格请求。

它不在本地保存任何用户的 NIM API Key。用户调用本项目时，需要自己通过请求头携带 NIM Key，网关只负责协议转换、性能优化、聚合统计和官方模型目录展示。

## 主要能力

- 将 NVIDIA 官方 `POST /v1/chat/completions` 转换为 OpenAI 风格的 `POST /v1/responses`
- 将 NVIDIA 官方 `POST /v1/chat/completions` 转换为 Anthropic 风格的 `POST /v1/messages`
- 支持 tool calling / function calling
- 支持 Anthropic `tools` / `tool_choice` / `tool_result` 以及 Claude Code 常见的客户端工具调用形态
- 支持 `function_call_output` 回灌
- 支持 `previous_response_id` 对话续写
- 对 `/v1/responses` 和 `/v1/responses/{response_id}` 使用用户自带的 NIM Key 做鉴权与上游转发
- 对 `/v1/messages` 使用用户自带的 NIM Key 做鉴权与上游转发，并支持 Anthropic SSE 风格流式事件
- `/v1/models` 直接返回来自 NVIDIA 官方 `/v1/models` 的同步结果，保持 OpenAI 风格结构
- `/` 为白色主题的模型健康度页面，按 10 分钟成功率矩阵展示 MODEL_LIST 中的模型
- `/model_list` 为独立的白色主题官方模型列表页面，支持按提供商筛选模型
- 模型提供商卡片为固定高度，避免模型较多时卡片过长
- 使用共享 HTTP 连接池、SQLite WAL 和异步线程化落库来增强高并发场景下的转发性能

## 用户如何调用

对于 `POST /v1/responses` 和 `POST /v1/messages`，请通过下面任意一种方式传入你自己的 NVIDIA NIM Key：

- `Authorization: Bearer <你的 NIM Key>`
- `X-API-Key: <你的 NIM Key>`

网关不会把原始 Key 持久化到数据库中，只会在内存中用于当前请求，并对响应链路使用 Key 哈希做隔离。

## 官方模型目录同步

项目会定时从官方接口拉取模型列表：




`https://integrate.api.nvidia.com/v1/models`

同步后的模型目录同时用于：


- `GET /v1/models`
- `GET /models`
- `GET /api/catalog`

## 页面与接口












页面：

- `GET /`：模型健康度页面
- `GET /model_list`：官方模型列表页面

前端数据接口：

- `GET /api/dashboard`
- `GET /api/catalog`

兼容接口：

- `POST /v1/responses`
- `POST /v1/messages`
- `GET /v1/responses/{response_id}`
- `GET /v1/models`
- `GET /models`

## 环境变量

- `NVIDIA_API_BASE`：默认 `https://integrate.api.nvidia.com/v1`
- `MODEL_LIST`：健康度页面监控模型列表，逗号分隔
- `MODEL_SYNC_INTERVAL_MINUTES`：官方模型目录同步周期，默认 `30`
- `PUBLIC_HISTORY_BUCKETS`：健康页展示最近多少个 10 分钟时间片，默认 `6`
- `REQUEST_TIMEOUT_SECONDS`：上游请求超时，默认 `90`
- `MAX_UPSTREAM_CONNECTIONS`：共享连接池最大连接数，默认 `512`
- `MAX_KEEPALIVE_CONNECTIONS`：共享连接池最大 keep-alive 连接数，默认 `128`
- `DATABASE_PATH`：默认 `./data.sqlite3`

## 本地验证

我已经完成两层本地联调：









1. Mock 联调：
   - 通过 `scripts/local_smoke_test.py` 验证了协议转换、官方模型同步、用户 Key 鉴权、`previous_response_id`、tool call、健康页数据接口、模型页数据接口和两个独立页面路由。

2. 真实上游联调：
   - 通过 `scripts/live_e2e_validation.py` 使用提供的测试 NIM Key，真实调用了 NVIDIA 官方模型目录和实际模型响应。
   - 实测结果：`live_gateway_ok`，并成功通过 `z-ai/glm5` 得到 `OK`。

## 部署到 Hugging Face Space


1. 新建 Hugging Face Space，SDK 选择 `Docker`
2. 将 `hf_space` 目录内的内容作为 Space 根目录上传
3. 按需配置 `MODEL_LIST` 等环境变量
4. 启动后即可直接公开使用

## 参考资料

- OpenAI Responses API: https://platform.openai.com/docs/guides/responses-vs-chat-completions
- OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
- Anthropic Messages API: https://docs.anthropic.com/en/api/messages
- Anthropic Tool Use: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
- NVIDIA Build: https://build.nvidia.com/
- NVIDIA NIM API 文档: https://docs.api.nvidia.com/
- NVIDIA NIM + Claude Code 集成: https://docs.nvidia.com/nim/large-language-models/latest/ai-assistant-integrations/claude-code.html
