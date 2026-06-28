---
title: Nim4cc
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app/main.py
pinned: false
---

# NIM4CC（让NIM的模型在Codex和Claude Code里面物尽其用）

NIM4CC 是一个面向公开使用的 NVIDIA NIM 兼容网关，目标是把 NIM 的 `chat/completions` 能力转换成更易接入的上层协议，并补上模型目录缓存、调用统计和健康度看板。

并且可以快速便捷的部署在免费的huggingface space上面使用

当前项目同时支持（可能支持不完全，遇到问题请带着相关日志提issue）：

- OpenAI 风格的 `POST /v1/responses`
- Anthropic 风格的 `POST /v1/messages`
- Claude Code 常见客户端工具调用形态的兼容转发

仓库地址：<https://github.com/Geek66666/nim4cc>

网关不会持久化保存用户原始 NIM Key。调用时由用户自己通过请求头携带 Key，服务端只在当前请求链路中使用，并通过哈希隔离本地响应记录。

## 主要能力

- 将 NVIDIA 官方 `POST /v1/chat/completions` 转换为 OpenAI 风格的 `POST /v1/responses`
- 将 NVIDIA 官方 `POST /v1/chat/completions` 转换为 Anthropic 风格的 `POST /v1/messages`
- 支持 OpenAI 风格的 tool calling / function calling
- 支持 Anthropic 风格的 `tools`、`tool_choice`、`tool_result`
- 尽量兼容 Claude Code 常见客户端工具调用形态
- 支持 `previous_response_id` 对话续写
- 支持 Anthropic SSE 风格流式事件
- 支持官方模型目录同步、本地缓存和公开展示
- 支持调用成功率聚合统计和模型健康度看板

## 当前接口

### 页面

- `GET /` 模型健康度页面
- `GET /model_list` 官方模型列表页面

### 页面数据接口

- `GET /api/dashboard`
- `GET /api/catalog`

### OpenAI 兼容接口

- `POST /v1/responses`
- `POST /responses`
- `GET /v1/responses/{response_id}`
- `GET /responses/{response_id}`
- `GET /v1/models`
- `GET /models`

### Anthropic 兼容接口

- `POST /v1/messages`

## Claude Code / Anthropic 工具支持说明

项目已经尽量适配 Claude Code 常见的客户端工具调用形态，并会把它们映射到 NIM 可理解的 function calling 结构。

当前兼容重点包括：

- 自定义 `tools`
- `tool_choice`
- `tool_result`
- 常见客户端工具类型，如 `bash_*`、`text_editor_*`、`computer_*`、`memory_*`

当前不会伪装支持 Anthropic 官方依赖服务端执行环境的能力。遇到这些能力时，网关会明确返回不支持，而不是静默吞掉：

- `mcp_servers`
- Anthropic server-side tools
- 需要 Anthropic 原生运行时的 `thinking` 扩展能力

这意味着项目优先支持“客户端自己执行工具，再把结果回灌”的 Claude Code / Anthropic 工作流。

## 项目结构

```text
.
├─ app/
│  ├─ __init__.py
│  └─ main.py          # 核心后端逻辑，路由/协议转换/统计都在这里
├─ static/
│  ├─ index.html       # 健康度页面
│  ├─ models.html      # 官方模型列表页面
│  ├─ public.js
│  ├─ models.js
│  └─ style.css
├─ Dockerfile
├─ requirements.txt
└─ README.md
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 配置环境变量

可以先复制 [.env.example](./.env.example) 作为参考。

常用环境变量如下：

| 变量名 | 说明 | 默认值 |
| --- | --- | --- |
| `NVIDIA_API_BASE` | NVIDIA NIM API 基础地址 | `https://integrate.api.nvidia.com/v1` |
| `MODEL_LIST` | 健康度页面关注的模型列表，逗号分隔 | 内置一组常用模型 |
| `APP_TIMEZONE` | 页面和统计使用的时区 | `Asia/Shanghai` |
| `MODEL_SYNC_INTERVAL_MINUTES` | 官方模型目录同步周期 | `30` |
| `PUBLIC_HISTORY_BUCKETS` | 健康页展示最近多少个 10 分钟时间片 | `22` |
| `REQUEST_TIMEOUT_SECONDS` | 上游请求超时秒数 | `180` |
| `MAX_UPSTREAM_CONNECTIONS` | 共享连接池最大连接数 | `512` |
| `MAX_KEEPALIVE_CONNECTIONS` | keep-alive 连接数 | `128` |
| `DATABASE_PATH` | SQLite 数据库文件路径 | `./data.sqlite3` |

### 3. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

启动后可访问：

- `http://127.0.0.1:7860/`
- `http://127.0.0.1:7860/model_list`
- `http://127.0.0.1:7860/docs`

## Docker 启动

```bash
docker build -t nim4cc .
docker run --rm -p 7860:7860 \
  -e NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1 \
  nim4cc
```

## 调用方式

请求时请通过下面任意一种方式传入你自己的 NIM Key：

- `Authorization: Bearer <你的 NIM Key>`
- `X-API-Key: <你的 NIM Key>`

### OpenAI Responses 示例

```bash
curl http://127.0.0.1:7860/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_NIM_KEY>" \
  -d '{
    "model": "z-ai/glm5",
    "input": "你好，请简单介绍一下自己。"
  }'
```

### Anthropic Messages 示例

```bash
curl http://127.0.0.1:7860/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -H "Authorization: Bearer <YOUR_NIM_KEY>" \
  -d '{
    "model": "z-ai/glm5",
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": "你好，请简单介绍一下自己。"
      }
    ]
  }'
```

### Claude Code 风格工具调用示例

```json
{
  "model": "z-ai/glm5",
  "max_tokens": 2048,
  "messages": [
    {
      "role": "user",
      "content": "请先查看当前目录文件"
    }
  ],
  "tools": [
    {
      "type": "bash_20250124",
      "name": "bash"
    }
  ]
}
```

## 健康度与统计

项目会把调用结果按模型聚合到 SQLite 中，用于生成健康度页面。

- 成功请求会计入成功率
- 上游返回错误状态码时会计入失败率
- 上游超时会自动重试 1 次，重试后仍失败同样计入失败率
- 健康页按 `MODEL_LIST` 展示最近若干个 10 分钟时间桶的成功率矩阵

## 模型目录同步

项目会定时从 NVIDIA 官方模型目录拉取数据：

`https://integrate.api.nvidia.com/v1/models`

同步结果会同时用于：

- `GET /v1/models`
- `GET /models`
- `GET /api/catalog`
- `/model_list` 页面展示

## 部署

### Hugging Face Spaces部署

0. 进入 Huggingface.co，并拥有一个账号
1. 新建一个 Space
2. SDK 选择 `Docker`
3. 上传当前仓库内容
4. 配置需要的环境变量
5. 启动服务即可使用

## 已知限制

- Anthropic server-side tools 和 `mcp_servers` 目前不支持
- README 中如果提到的外部脚本或部署目录与你本地实际内容不一致，请以仓库当前文件结构为准

---

Finally，Thanks to everyone on LinuxDo for their support! Welcome to join https://linux.do/ for all kinds of technical exchanges, cutting-edge AI information, and AI experience sharing, all on Linuxdo!

---

## 参考文档

- OpenAI Responses API: <https://platform.openai.com/docs/guides/responses-vs-chat-completions>
- OpenAI Function Calling: <https://platform.openai.com/docs/guides/function-calling>
- Anthropic Messages API: <https://docs.anthropic.com/en/api/messages>
- Anthropic Tool Use Overview: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview>
- NVIDIA Build: <https://build.nvidia.com/>
- NVIDIA NIM API 文档: <https://docs.api.nvidia.com/>
- NVIDIA NIM + Claude Code 集成: <https://docs.nvidia.com/nim/large-language-models/latest/ai-assistant-integrations/claude-code.html>