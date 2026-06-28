from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "data.sqlite3"))
RAW_NVIDIA_API_BASE = os.getenv("NVIDIA_API_BASE", os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")).rstrip("/")
NVIDIA_API_BASE = RAW_NVIDIA_API_BASE if RAW_NVIDIA_API_BASE.endswith("/v1") else f"{RAW_NVIDIA_API_BASE}/v1"
CHAT_COMPLETIONS_URL = f"{NVIDIA_API_BASE}/chat/completions"
MODELS_URL = f"{NVIDIA_API_BASE}/models"
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "180"))
MAX_UPSTREAM_CONNECTIONS = int(os.getenv("MAX_UPSTREAM_CONNECTIONS", "512"))
MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("MAX_KEEPALIVE_CONNECTIONS", "128"))
MODEL_SYNC_INTERVAL_MINUTES = int(os.getenv("MODEL_SYNC_INTERVAL_MINUTES", "30"))
PUBLIC_HISTORY_BUCKETS = int(os.getenv("PUBLIC_HISTORY_BUCKETS", "22"))
HEALTH_SUMMARY_WINDOW_MINUTES = 120
UPSTREAM_TIMEOUT_RETRIES = 1
BUCKET_MINUTES = 10
DEFAULT_MONITORED_MODELS = "z-ai/glm5,z-ai/glm4.7,minimaxai/minimax-m2.5,minimaxai/minimax-m2.7,moonshotai/kimi-k2.5,deepseek-ai/deepseek-v3.2,google/gemma-4-31b-it,qwen/qwen3.5-397b-a17b"
MODEL_LIST = [item.strip() for item in os.getenv("MODEL_LIST", DEFAULT_MONITORED_MODELS).split(",") if item.strip()]
APP_TIMEZONE = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Shanghai"))
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"
ANTHROPIC_MIN_THINKING_BUDGET_TOKENS = 1024
ANTHROPIC_SERVER_TOOL_MAX_ITERATIONS = 8
ANTHROPIC_SERVER_TOOL_PREFIXES = (
    "web_search_",
    "web_fetch_",
    "code_execution_",
    "advisor_",
    "tool_search_tool_",
    "mcp_toolset",
)
WEB_SEARCH_RSS_URL = os.getenv("WEB_SEARCH_RSS_URL", "https://www.bing.com/search")
WEB_SEARCH_DEFAULT_MAX_RESULTS = int(os.getenv("WEB_SEARCH_DEFAULT_MAX_RESULTS", "5"))
WEB_SEARCH_MAX_QUERY_LENGTH = int(os.getenv("WEB_SEARCH_MAX_QUERY_LENGTH", "512"))

http_client: httpx.AsyncClient | None = None
model_cache: list[dict[str, Any]] = []
model_cache_synced_at: str | None = None
model_cache_lock: asyncio.Lock | None = None
model_sync_task: asyncio.Task[None] | None = None
THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def utcnow() -> datetime:
    return datetime.now(APP_TIMEZONE)


def utcnow_iso() -> str:
    return utcnow().isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def normalize_provider(model_id: str, owned_by: str | None = None) -> str:
    if owned_by:
        return owned_by
    if "/" in model_id:
        return model_id.split("/", 1)[0]
    return "unknown"


def bucket_start(dt: datetime | None = None) -> datetime:
    dt = dt or utcnow()
    minute = dt.minute - (dt.minute % BUCKET_MINUTES)
    return dt.replace(minute=minute, second=0, microsecond=0)


def bucket_label(value: str) -> str:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    return dt.strftime("%H:%M")


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    conn = get_db_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS response_records (
                response_id TEXT PRIMARY KEY,
                api_key_hash TEXT NOT NULL,
                parent_response_id TEXT,
                model_id TEXT NOT NULL,
                request_json TEXT NOT NULL,
                input_items_json TEXT NOT NULL,
                output_json TEXT NOT NULL,
                output_items_json TEXT NOT NULL,
                status TEXT NOT NULL,
                success INTEGER NOT NULL,
                latency_ms REAL,
                error_message TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_response_api_hash ON response_records(api_key_hash);
            CREATE INDEX IF NOT EXISTS idx_response_parent ON response_records(parent_response_id);
            CREATE INDEX IF NOT EXISTS idx_response_model_created ON response_records(model_id, created_at);

            CREATE TABLE IF NOT EXISTS metric_buckets (
                bucket_start TEXT NOT NULL,
                model_id TEXT NOT NULL,
                total_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                total_latency_ms REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(bucket_start, model_id)
            );

            CREATE TABLE IF NOT EXISTS gateway_totals (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                total_requests INTEGER NOT NULL DEFAULT 0,
                total_success INTEGER NOT NULL DEFAULT 0,
                total_latency_ms REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS official_models_cache (
                id TEXT PRIMARY KEY,
                object TEXT NOT NULL,
                created INTEGER,
                owned_by TEXT,
                synced_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO gateway_totals (id, total_requests, total_success, total_latency_ms, updated_at)
            VALUES (1, 0, 0, 0, ?)
            """,
            (utcnow_iso(),),
        )
        conn.commit()
    finally:
        conn.close()


async def run_db(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


async def get_http_client() -> httpx.AsyncClient:
    global http_client
    if http_client is None or http_client.is_closed:
        limits = httpx.Limits(
            max_connections=MAX_UPSTREAM_CONNECTIONS,
            max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
        )
        http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, limits=limits)
    return http_client


async def get_model_cache_lock() -> asyncio.Lock:
    global model_cache_lock
    if model_cache_lock is None:
        model_cache_lock = asyncio.Lock()
    return model_cache_lock


def load_cached_models_from_db() -> tuple[list[dict[str, Any]], str | None]:
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT id, object, created, owned_by, synced_at FROM official_models_cache ORDER BY id ASC"
        ).fetchall()
        if not rows:
            return [], None
        synced_at = rows[0]["synced_at"]
        models = [
            {
                "id": row["id"],
                "object": row["object"],
                "created": row["created"],
                "owned_by": row["owned_by"],
            }
            for row in rows
        ]
        return models, synced_at
    finally:
        conn.close()


def save_models_to_db(models: list[dict[str, Any]], synced_at: str) -> None:
    unique_models: dict[str, dict[str, Any]] = {}
    for model in models:
        model_id = model.get("id")
        if model_id:
            unique_models[model_id] = model

    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM official_models_cache")
        conn.executemany(
            """
            INSERT INTO official_models_cache (id, object, created, owned_by, synced_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    model_id,
                    model.get("object", "model"),
                    model.get("created"),
                    model.get("owned_by") or normalize_provider(model_id),
                    synced_at,
                )
                for model_id, model in sorted(unique_models.items(), key=lambda item: item[0])
            ],
        )
        conn.commit()
    finally:
        conn.close()


async def refresh_official_models(force: bool = False) -> list[dict[str, Any]]:
    global model_cache, model_cache_synced_at
    if model_cache and not force:
        return model_cache
    lock = await get_model_cache_lock()
    async with lock:
        if model_cache and not force:
            return model_cache
        client = await get_http_client()
        response = await client.get(MODELS_URL, headers={"Accept": "application/json"})
        response.raise_for_status()
        payload = response.json()
        models = payload.get("data") or payload.get("models") or []
        normalized = [
            {
                "id": item.get("id"),
                "object": item.get("object", "model"),
                "created": item.get("created"),
                "owned_by": item.get("owned_by") or normalize_provider(item.get("id", "")),
            }
            for item in models
            if isinstance(item, dict) and item.get("id")
        ]
        synced_at = utcnow_iso()
        await run_db(save_models_to_db, normalized, synced_at)
        model_cache = normalized
        model_cache_synced_at = synced_at
        return normalized


async def model_sync_loop() -> None:
    while True:
        try:
            await refresh_official_models(force=True)
        except Exception:
            pass
        await asyncio.sleep(max(300, MODEL_SYNC_INTERVAL_MINUTES * 60))


def extract_user_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_nvidia_api_key: str | None = Header(default=None),
) -> str:
    token: str | None = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
    elif x_api_key:
        token = x_api_key.strip()
    elif x_nvidia_api_key:
        token = x_nvidia_api_key.strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请通过 Authorization Bearer 或 X-API-Key 提供你的 NIM Key。")
    return token

def normalize_content(content: Any, role: str) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "output_text" if role == "assistant" else "input_text", "text": content}]
    if isinstance(content, list):
        normalized: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                normalized.append({"type": "output_text" if role == "assistant" else "input_text", "text": part})
                continue
            if not isinstance(part, dict):
                normalized.append({"type": "input_text", "text": str(part)})
                continue
            if part.get("type") in {"input_text", "output_text", "text", "tool_call", "function_call"}:
                normalized.append(part)
                continue
            if "text" in part:
                normalized.append({"type": part.get("type", "input_text"), "text": part.get("text", "")})
        return normalized
    if isinstance(content, dict):
        if "text" in content:
            return [{"type": content.get("type", "input_text"), "text": content.get("text", "")}]
        return [{"type": "input_text", "text": json_dumps(content)}]
    return [{"type": "input_text", "text": str(content)}]


def normalize_input_items(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": value}]}]
    if isinstance(value, dict):
        value = [value]

    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": item}]})
            continue
        if not isinstance(item, dict):
            items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": str(item)}]})
            continue
        item_type = item.get("type")
        if item_type == "message" or item.get("role"):
            role = item.get("role", "user")
            items.append({"type": "message", "role": role, "content": normalize_content(item.get("content"), role)})
            continue
        if item_type == "function_call_output":
            output = item.get("output")
            if not isinstance(output, str):
                output = json_dumps(output) if output is not None else ""
            items.append({"type": "function_call_output", "call_id": item.get("call_id"), "output": output})
            continue
        if item_type == "function_call":
            arguments = item.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json_dumps(arguments)
            items.append({
                "type": "function_call",
                "call_id": item.get("call_id") or f"call_{uuid.uuid4().hex[:12]}",
                "name": item.get("name"),
                "arguments": arguments,
            })
            continue
        if item_type in {"input_text", "output_text", "text"}:
            items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": item.get("text", "")}]})
            continue
        items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": json_dumps(item)}]})
    return items


def extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", ""))
        return json_dumps(content)
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict) and part.get("type") in {"input_text", "output_text", "text"}:
                chunks.append(str(part.get("text", "")))
        return "\n".join(filter(None, chunks))
    return str(content)


def items_to_chat_messages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []

    def flush_pending_tool_calls() -> None:
        nonlocal pending_tool_calls
        if pending_tool_calls:
            messages.append({"role": "assistant", "content": "", "tool_calls": pending_tool_calls})
            pending_tool_calls = []

    for item in items:
        item_type = item.get("type")
        if item_type == "function_call":
            pending_tool_calls.append(
                {
                    "id": item.get("call_id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments", "{}")},
                }
            )
            continue
        if item_type == "function_call_output":
            flush_pending_tool_calls()
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": item.get("output", "")})
            continue
        if item_type != "message":
            continue
        flush_pending_tool_calls()
        role = item.get("role", "user")
        text_value = extract_text_from_content(item.get("content"))
        if role in {"system", "developer"}:
            messages.append({"role": "system", "content": text_value})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": text_value})
        else:
            messages.append({"role": role, "content": text_value})

    flush_pending_tool_calls()
    return [message for message in messages if message.get("content") is not None or message.get("tool_calls")]


def response_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function_payload = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function_payload.get("name")
        if not name:
            continue
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": function_payload.get("description"),
                    "parameters": function_payload.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return normalized


def normalize_tool_choice(tool_choice: Any, tools: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]]]:
    if tool_choice is None:
        return None, tools
    if isinstance(tool_choice, str):
        return tool_choice, tools
    if not isinstance(tool_choice, dict):
        return None, tools
    if tool_choice.get("type") == "function":
        function_name = tool_choice.get("name") or (tool_choice.get("function") or {}).get("name")
        if function_name:
            return {"type": "function", "function": {"name": function_name}}, tools
    if tool_choice.get("type") == "allowed_tools":
        allowed = tool_choice.get("tools") or []
        allowed_names = {
            entry if isinstance(entry, str) else entry.get("name")
            for entry in allowed
            if entry is not None
        }
        filtered_tools = [tool for tool in tools if tool["function"]["name"] in allowed_names]
        mode = tool_choice.get("mode", "auto")
        return mode if isinstance(mode, str) else "auto", filtered_tools
    return None, tools


def build_chat_payload(body: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    tools = response_tools_to_chat_tools(body.get("tools"))
    tool_choice, tools = normalize_tool_choice(body.get("tool_choice"), tools)
    payload: dict[str, Any] = {"model": body.get("model"), "messages": items_to_chat_messages(items)}
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if body.get("temperature") is not None:
        payload["temperature"] = body.get("temperature")
    if body.get("top_p") is not None:
        payload["top_p"] = body.get("top_p")
    if body.get("parallel_tool_calls") is not None:
        payload["parallel_tool_calls"] = body.get("parallel_tool_calls")
    if body.get("max_output_tokens") is not None:
        payload["max_tokens"] = body.get("max_output_tokens")
    if body.get("instructions"):
        payload["messages"] = [{"role": "system", "content": body["instructions"]}] + payload["messages"]
    text_config = body.get("text") or {}
    text_format = text_config.get("format") if isinstance(text_config, dict) else None
    if isinstance(text_format, dict):
        if text_format.get("type") == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif text_format.get("type") == "json_schema":
            payload["response_format"] = {"type": "json_schema", "json_schema": text_format.get("json_schema") or {}}
    return payload


def anthropic_content_to_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        return [content]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]

    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
        elif isinstance(part, dict):
            blocks.append(part)
        else:
            blocks.append({"type": "text", "text": str(part)})
    return blocks


def extract_anthropic_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        value_type = value.get("type")
        if value_type in {"text", "input_text", "output_text"}:
            return str(value.get("text", ""))
        if value_type == "thinking":
            return str(value.get("thinking", ""))
        if value_type == "redacted_thinking":
            return ""
        if value_type in {"image", "document"}:
            return f"[{value_type} content omitted]"
        if "content" in value:
            return extract_anthropic_text(value.get("content"))
        return json_dumps(value)
    if isinstance(value, list):
        chunks: list[str] = []
        for part in value:
            text_value = extract_anthropic_text(part)
            if text_value:
                chunks.append(text_value)
        return "\n".join(chunks)
    return str(value)


def anthropic_result_block_to_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if block.get("type") == "tool_result" and not block.get("is_error"):
        if isinstance(content, str):
            return content
        plain_text = extract_anthropic_text(content)
        if plain_text and plain_text != json_dumps(content):
            return plain_text

    payload: dict[str, Any] = {}
    if block.get("is_error") is not None:
        payload["is_error"] = bool(block.get("is_error"))
    payload["content"] = content
    return json_dumps(payload)


def is_anthropic_tool_result_block(block: dict[str, Any]) -> bool:
    block_type = block.get("type")
    return isinstance(block_type, str) and (block_type == "tool_result" or block_type.endswith("_tool_result"))


def parse_anthropic_beta_header(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def has_anthropic_message_prefill(messages: Any) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    last_message = messages[-1]
    if not isinstance(last_message, dict):
        return False
    if last_message.get("role") != "assistant":
        return False
    blocks = anthropic_content_to_blocks(last_message.get("content"))
    return any(block.get("type") != "redacted_thinking" for block in blocks)


def is_forced_anthropic_tool_choice(tool_choice: Any) -> bool:
    if isinstance(tool_choice, str):
        return tool_choice in {"any", "required"}
    if not isinstance(tool_choice, dict):
        return False
    return tool_choice.get("type") in {"any", "tool"}


def build_anthropic_thinking_config(body: dict[str, Any], anthropic_beta: str | None) -> dict[str, Any]:
    thinking = body.get("thinking")
    enabled = False
    budget_tokens = None

    if thinking is None:
        return {"enabled": False, "budget_tokens": None, "synthetic_signature": True}
    if isinstance(thinking, bool):
        thinking = {"type": "enabled" if thinking else "disabled"}
    if not isinstance(thinking, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="thinking 字段必须是对象或布尔值。")

    raw_thinking_type = thinking.get("type")
    if isinstance(raw_thinking_type, bool):
        thinking_type = "enabled" if raw_thinking_type else "disabled"
    elif isinstance(raw_thinking_type, str):
        lowered_type = raw_thinking_type.strip().lower()
        if lowered_type in {"enabled", "enable", "on", "true"}:
            thinking_type = "enabled"
        elif lowered_type in {"disabled", "disable", "off", "false"}:
            thinking_type = "disabled"
        else:
            thinking_type = lowered_type
    elif "enabled" in thinking:
        thinking_type = "enabled" if bool(thinking.get("enabled")) else "disabled"
    elif any(key in thinking for key in ("budget_tokens", "budgetTokens")):
        thinking_type = "enabled"
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="thinking.type 仅支持 enabled 或 disabled。")

    if thinking_type == "disabled":
        return {"enabled": False, "budget_tokens": None, "synthetic_signature": True}
    if thinking_type != "enabled":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="thinking.type 仅支持 enabled 或 disabled。")

    budget_tokens = thinking.get("budget_tokens", thinking.get("budgetTokens"))
    if isinstance(budget_tokens, str):
        try:
            budget_tokens = int(budget_tokens.strip())
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="thinking.enabled 时必须提供整数类型的 budget_tokens。",
            ) from exc
    if not isinstance(budget_tokens, int):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="thinking.enabled 时必须提供整数类型的 budget_tokens。")
    if budget_tokens < ANTHROPIC_MIN_THINKING_BUDGET_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"thinking.budget_tokens 不能小于 {ANTHROPIC_MIN_THINKING_BUDGET_TOKENS}。",
        )

    max_tokens = body.get("max_tokens")
    beta_flags = parse_anthropic_beta_header(anthropic_beta)
    interleaved_thinking = (
        ANTHROPIC_INTERLEAVED_THINKING_BETA in beta_flags
        and bool(body.get("tools"))
    )
    if isinstance(max_tokens, int) and budget_tokens >= max_tokens and not interleaved_thinking:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="thinking.budget_tokens 必须小于 max_tokens；只有启用 interleaved thinking beta 且使用工具时可以例外。",
        )

    if body.get("temperature") is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Anthropic thinking 模式不支持自定义 temperature。",
        )

    top_p = body.get("top_p")
    if top_p is not None:
        try:
            top_p_value = float(top_p)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="top_p 必须是数值。",
            ) from exc
        if not (0.95 <= top_p_value <= 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Anthropic thinking 模式下 top_p 只能在 0.95 到 1.0 之间。",
            )

    if is_forced_anthropic_tool_choice(body.get("tool_choice")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Anthropic thinking 模式不支持 forced tool choice。",
        )

    if has_anthropic_message_prefill(body.get("messages")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Anthropic thinking 模式不支持 assistant prefill。",
        )

    enabled = True
    return {
        "enabled": enabled,
        "budget_tokens": budget_tokens,
        "interleaved": interleaved_thinking,
        "synthetic_signature": True,
    }


def build_synthetic_thinking_signature(model_id: str | None, thinking_text: str) -> str:
    digest = hashlib.sha256(f"{model_id or 'unknown'}\n{thinking_text}".encode("utf-8")).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"nimthinking_{encoded}"


def split_anthropic_thinking_blocks(text: str, model_id: str | None) -> list[dict[str, Any]]:
    if not text:
        return []

    blocks: list[dict[str, Any]] = []
    cursor = 0
    for match in THINK_TAG_PATTERN.finditer(text):
        before = text[cursor:match.start()]
        if before.strip():
            blocks.append({"type": "text", "text": before.strip()})

        thinking_text = match.group(1).strip()
        if thinking_text:
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": thinking_text,
                    "signature": build_synthetic_thinking_signature(model_id, thinking_text),
                }
            )
        cursor = match.end()

    if cursor == 0:
        return [{"type": "text", "text": text.strip()}] if text.strip() else []

    after = text[cursor:]
    if after.strip():
        blocks.append({"type": "text", "text": after.strip()})
    return blocks


def build_bash_tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute in the persistent bash session.",
            },
            "restart": {
                "type": "boolean",
                "description": "Restart the persistent bash session before running the next command.",
            },
        },
    }


def build_text_editor_tool_schema(tool_type: str | None) -> dict[str, Any]:
    commands = ["view", "create", "str_replace", "insert"]
    if tool_type and (tool_type.endswith("20241022") or tool_type.endswith("20250124")):
        commands.append("undo_edit")
    return {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": commands,
                "description": "The editor operation to perform.",
            },
            "path": {"type": "string", "description": "Absolute or relative path to the target file."},
            "view_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Inclusive start/end line numbers for view operations.",
            },
            "file_text": {"type": "string", "description": "Full file contents when creating a file."},
            "old_str": {"type": "string", "description": "Existing text to replace."},
            "new_str": {"type": "string", "description": "Replacement text for str_replace."},
            "insert_line": {"type": "integer", "description": "Line number to insert text before."},
            "insert_text": {"type": "string", "description": "Text to insert."},
        },
    }


def build_memory_tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["view", "create", "str_replace", "insert", "delete", "rename"],
                "description": "The memory operation to perform under the memory directory.",
            },
            "path": {"type": "string", "description": "Path to the memory file."},
            "new_path": {"type": "string", "description": "New path when renaming a memory file."},
            "view_range": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Inclusive start/end line numbers for view operations.",
            },
            "file_text": {"type": "string", "description": "Full file contents when creating a memory file."},
            "old_str": {"type": "string", "description": "Existing text to replace."},
            "new_str": {"type": "string", "description": "Replacement text for str_replace."},
            "insert_line": {"type": "integer", "description": "Line number to insert text before."},
            "insert_text": {"type": "string", "description": "Text to insert."},
        },
    }


def build_computer_tool_schema(tool_type: str | None) -> dict[str, Any]:
    actions = [
        "screenshot",
        "left_click",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "mouse_move",
        "left_click_drag",
        "left_mouse_down",
        "left_mouse_up",
        "scroll",
        "type",
        "key",
        "hold_key",
        "wait",
    ]
    if tool_type and tool_type.endswith("20251124"):
        actions.append("zoom")
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": actions,
                "description": "The computer action to perform.",
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "X/Y coordinate for click and move actions.",
            },
            "start_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Start coordinate for drag actions.",
            },
            "end_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": "End coordinate for drag actions.",
            },
            "text": {"type": "string", "description": "Text to type or zoom target text."},
            "key": {"type": "string", "description": "Keyboard key or key chord to press."},
            "duration": {"type": "number", "description": "Optional wait duration in seconds."},
            "scroll_direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
                "description": "Scroll direction.",
            },
            "scroll_amount": {"type": "integer", "description": "Scroll distance in pixels or wheel units."},
            "region": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Optional region [left, top, width, height] for screenshots.",
            },
            "modifiers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Modifier keys to hold during the action.",
            },
        },
    }


def build_web_search_tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The web search query to execute.",
            },
        },
        "required": ["query"],
    }


def normalize_domain_name(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    host = (parsed.netloc or parsed.path or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host.split("/", 1)[0]


def normalize_domain_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized = [normalize_domain_name(str(item)) for item in values if str(item).strip()]
    return [item for item in normalized if item]


def domain_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def filter_web_search_url(url: str, allowed_domains: list[str], blocked_domains: list[str]) -> bool:
    try:
        host = normalize_domain_name(url)
    except Exception:
        return False
    if not host:
        return False
    if allowed_domains and not any(domain_matches(host, domain) for domain in allowed_domains):
        return False
    if blocked_domains and any(domain_matches(host, domain) for domain in blocked_domains):
        return False
    return True


def build_web_search_tool_description(raw_tool: dict[str, Any]) -> str:
    description = raw_tool.get("description") or "Search the public web and return concise result metadata."
    allowed_domains = normalize_domain_list(raw_tool.get("allowed_domains"))
    blocked_domains = normalize_domain_list(raw_tool.get("blocked_domains"))
    if allowed_domains:
        description = f"{description}\n\nOnly search and return results from these domains: {', '.join(allowed_domains)}."
    if blocked_domains:
        description = f"{description}\n\nDo not return results from these domains: {', '.join(blocked_domains)}."
    return description


def build_server_tool_result_error_block(
    result_type: str,
    tool_use_id: str,
    error_code: str,
    message: str | None = None,
) -> dict[str, Any]:
    error_item: dict[str, Any] = {"type": f"{result_type}_error", "error_code": error_code}
    if message:
        error_item["message"] = message
    return {
        "type": result_type,
        "tool_use_id": tool_use_id,
        "content": error_item,
        "is_error": True,
    }


def build_web_search_encrypted_content(payload: dict[str, Any]) -> str:
    raw = json_dumps(payload).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"nimsearch_{encoded}"


def parse_rss_results(xml_text: str) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    results: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = (item.findtext("description") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        if not title or not link:
            continue
        results.append(
            {
                "title": title,
                "url": link,
                "snippet": description,
                "page_age": pub_date or None,
            }
        )
    return results


def append_anthropic_tool_examples(description: str | None, examples: Any) -> str | None:
    if not isinstance(examples, list) or not examples:
        return description
    snippet = json_dumps(examples[:2])
    if description:
        return f"{description}\n\nInput examples: {snippet}"
    return f"Input examples: {snippet}"


def normalize_anthropic_tool_name(tool_type: str | None, fallback_name: str | None) -> str | None:
    if fallback_name:
        return fallback_name
    if not tool_type:
        return None
    if tool_type.startswith("bash_"):
        return "bash"
    if tool_type.startswith("text_editor_"):
        return "str_replace_based_edit_tool"
    if tool_type.startswith("computer_"):
        return "computer"
    if tool_type.startswith("memory_"):
        return "memory"
    return None


def anthropic_tools_to_chat_tools(tools: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    metadata_by_name: dict[str, dict[str, Any]] = {}

    for raw_tool in tools or []:
        if not isinstance(raw_tool, dict):
            continue
        tool_type = raw_tool.get("type")
        tool_name = normalize_anthropic_tool_name(tool_type, raw_tool.get("name"))
        allowed_callers = raw_tool.get("allowed_callers") or ["direct"]
        if isinstance(allowed_callers, (list, tuple, set)):
            allowed_callers_set = {str(item) for item in allowed_callers if item}
        else:
            allowed_callers_set = {str(allowed_callers)}

        if "direct" not in allowed_callers_set:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"当前网关暂不支持仅允许 programmatic caller 的工具：{tool_name or tool_type or 'unknown'}。",
            )

        if tool_type == "mcp_toolset":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"当前网关暂不支持 Anthropic 服务端工具 '{tool_type}'；请改用客户端工具或自定义 tools。",
            )

        if isinstance(tool_type, str) and tool_type.startswith("web_search_"):
            tool_name = tool_name or "web_search"
            if normalize_domain_list(raw_tool.get("allowed_domains")) and normalize_domain_list(raw_tool.get("blocked_domains")):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="web_search 工具不能同时设置 allowed_domains 和 blocked_domains。",
                )
            description = build_web_search_tool_description(raw_tool)
            parameters = build_web_search_tool_schema()
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
            metadata_by_name[tool_name] = {
                "anthropic_type": tool_type,
                "allowed_callers": sorted(allowed_callers_set) or ["direct"],
                "server_execution": "web_search",
                "allowed_domains": normalize_domain_list(raw_tool.get("allowed_domains")),
                "blocked_domains": normalize_domain_list(raw_tool.get("blocked_domains")),
                "user_location": raw_tool.get("user_location") if isinstance(raw_tool.get("user_location"), dict) else None,
                "max_uses": raw_tool.get("max_uses") if isinstance(raw_tool.get("max_uses"), int) and raw_tool.get("max_uses") > 0 else None,
                "uses": 0,
            }
            continue

        if isinstance(tool_type, str) and tool_type.startswith(ANTHROPIC_SERVER_TOOL_PREFIXES):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"当前网关暂不支持 Anthropic 服务端工具 '{tool_type}'；目前已兼容 web_search 工具，其他服务端工具仍需单独适配。",
            )

        if isinstance(tool_type, str) and tool_type.startswith("bash_"):
            description = raw_tool.get("description") or "Execute shell commands in a persistent bash session."
            parameters = build_bash_tool_schema()
        elif isinstance(tool_type, str) and tool_type.startswith("text_editor_"):
            description = raw_tool.get("description") or "View and edit text files with command-based operations."
            parameters = build_text_editor_tool_schema(tool_type)
        elif isinstance(tool_type, str) and tool_type.startswith("computer_"):
            description = raw_tool.get("description") or "Interact with a computer UI using screenshots, clicks, typing, keys, scrolling, and drag actions."
            parameters = build_computer_tool_schema(tool_type)
        elif isinstance(tool_type, str) and tool_type.startswith("memory_"):
            description = raw_tool.get("description") or "Read and edit persistent memory files with command-based operations."
            parameters = build_memory_tool_schema()
        else:
            if not tool_name:
                continue
            if tool_type and raw_tool.get("input_schema") is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"当前网关暂不支持 Anthropic 工具类型 '{tool_type}'。",
                )
            description = raw_tool.get("description")
            parameters = raw_tool.get("input_schema") or {"type": "object", "properties": {}}

        description = append_anthropic_tool_examples(description, raw_tool.get("input_examples"))
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
        metadata_by_name[tool_name] = {
            "anthropic_type": tool_type or "custom",
            "allowed_callers": sorted(allowed_callers_set) or ["direct"],
        }

    return normalized, metadata_by_name


def normalize_anthropic_tool_choice(tool_choice: Any) -> tuple[Any, bool | None]:
    if tool_choice is None:
        return None, None
    if isinstance(tool_choice, str):
        if tool_choice == "any":
            return "required", None
        return tool_choice, None
    if not isinstance(tool_choice, dict):
        return None, None

    parallel_tool_calls = None
    if tool_choice.get("disable_parallel_tool_use") is not None:
        parallel_tool_calls = not bool(tool_choice.get("disable_parallel_tool_use"))

    choice_type = tool_choice.get("type")
    if choice_type in {"auto", "none"}:
        return choice_type, parallel_tool_calls
    if choice_type == "any":
        return "required", parallel_tool_calls
    if choice_type == "tool":
        tool_name = tool_choice.get("name")
        if tool_name:
            return {"type": "function", "function": {"name": tool_name}}, parallel_tool_calls
    return None, parallel_tool_calls


def anthropic_messages_to_chat_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    chat_messages: list[dict[str, Any]] = []
    system_text = extract_anthropic_text(body.get("system"))
    if system_text:
        chat_messages.append({"role": "system", "content": system_text})

    for raw_message in body.get("messages") or []:
        if isinstance(raw_message, str):
            chat_messages.append({"role": "user", "content": raw_message})
            continue
        if not isinstance(raw_message, dict):
            chat_messages.append({"role": "user", "content": str(raw_message)})
            continue

        role = raw_message.get("role", "user")
        blocks = anthropic_content_to_blocks(raw_message.get("content"))

        if role == "assistant":
            text_chunks: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in blocks:
                block_type = block.get("type")
                if block_type == "text":
                    text_chunks.append(str(block.get("text", "")))
                    continue
                if block_type == "thinking":
                    thinking_text = str(block.get("thinking", "")).strip()
                    if thinking_text:
                        text_chunks.append(f"<think>\n{thinking_text}\n</think>")
                    continue
                if block_type == "redacted_thinking":
                    continue
                if block_type in {"tool_use", "server_tool_use"}:
                    arguments = block.get("input")
                    if not isinstance(arguments, str):
                        arguments = json_dumps(arguments or {})
                    tool_calls.append(
                        {
                            "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": arguments,
                            },
                        }
                    )
                    continue
                block_text = extract_anthropic_text(block)
                if block_text:
                    text_chunks.append(block_text)

            if text_chunks or tool_calls:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(filter(None, text_chunks)),
                }
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls
                chat_messages.append(assistant_message)
            continue

        pending_text: list[str] = []

        def flush_pending_text() -> None:
            nonlocal pending_text
            text_value = "\n".join(filter(None, pending_text))
            if text_value:
                target_role = "system" if role in {"system", "developer"} else "user"
                chat_messages.append({"role": target_role, "content": text_value})
            pending_text = []

        for block in blocks:
            if is_anthropic_tool_result_block(block):
                flush_pending_text()
                tool_use_id = block.get("tool_use_id") or block.get("id")
                result_text = anthropic_result_block_to_text(block)
                if tool_use_id:
                    chat_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": result_text,
                        }
                    )
                elif result_text:
                    pending_text.append(result_text)
                continue

            block_text = extract_anthropic_text(block)
            if block_text:
                pending_text.append(block_text)

        flush_pending_text()

    return [message for message in chat_messages if message.get("content") is not None or message.get("tool_calls")]


def build_anthropic_chat_payload(
    body: dict[str, Any],
    anthropic_beta: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    thinking_config = build_anthropic_thinking_config(body, anthropic_beta)
    if body.get("mcp_servers"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前网关暂不支持 Anthropic mcp_servers 直连能力。",
        )

    messages = anthropic_messages_to_chat_messages(body)
    tools, tool_metadata = anthropic_tools_to_chat_tools(body.get("tools"))
    tool_choice, parallel_tool_calls = normalize_anthropic_tool_choice(body.get("tool_choice"))
    payload: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "max_tokens": body.get("max_tokens"),
    }
    if thinking_config["enabled"]:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        payload["nvext"] = {"max_thinking_tokens": thinking_config["budget_tokens"]}
    elif body.get("thinking") is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    if parallel_tool_calls is not None and tools:
        payload["parallel_tool_calls"] = parallel_tool_calls
    if body.get("temperature") is not None:
        payload["temperature"] = body.get("temperature")
    if body.get("top_p") is not None:
        payload["top_p"] = body.get("top_p")
    if body.get("stop_sequences"):
        payload["stop"] = body.get("stop_sequences")
    return payload, messages, tool_metadata, thinking_config


def parse_anthropic_tool_input(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if arguments is None:
        return {}
    if not isinstance(arguments, str):
        return {"value": arguments}
    try:
        parsed = json.loads(arguments)
    except Exception:
        return {"raw_input": arguments}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def normalize_anthropic_message_id(message_id: Any) -> str:
    if isinstance(message_id, str) and message_id.startswith("msg_"):
        return message_id
    return f"msg_{uuid.uuid4().hex[:24]}"


def normalize_anthropic_tool_use_id(tool_use_id: Any) -> str:
    if isinstance(tool_use_id, str) and tool_use_id.startswith(("toolu_", "srvtoolu_")):
        return tool_use_id
    return f"toolu_{uuid.uuid4().hex[:24]}"


def anthropic_block_to_chat_tool_call(block: dict[str, Any]) -> dict[str, Any]:
    arguments = block.get("input") or {}
    if not isinstance(arguments, str):
        arguments = json_dumps(arguments)
    return {
        "id": block.get("id") or normalize_anthropic_tool_use_id(None),
        "type": "function",
        "function": {
            "name": block.get("name"),
            "arguments": arguments,
        },
    }


def anthropic_blocks_to_chat_assistant_message(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text_value = str(block.get("text", "")).strip()
            if text_value:
                text_chunks.append(text_value)
            continue
        if block_type == "thinking":
            thinking_text = str(block.get("thinking", "")).strip()
            if thinking_text:
                text_chunks.append(f"<think>\n{thinking_text}\n</think>")
            continue
        if block_type in {"tool_use", "server_tool_use"}:
            tool_calls.append(anthropic_block_to_chat_tool_call(block))

    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(filter(None, text_chunks))}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def is_server_tool_block(block: dict[str, Any], tool_metadata: dict[str, dict[str, Any]]) -> bool:
    if block.get("type") != "server_tool_use":
        return False
    tool_name = block.get("name")
    return bool(tool_name and tool_metadata.get(tool_name, {}).get("server_execution"))


def count_server_tool_blocks(content_blocks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in content_blocks:
        if block.get("type") != "server_tool_use":
            continue
        name = str(block.get("name") or "unknown")
        counts[name] = counts.get(name, 0) + 1
    return counts


def merge_anthropic_usage(base_usage: dict[str, Any], extra_usage: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "input_tokens": (base_usage or {}).get("input_tokens") or 0,
        "output_tokens": (base_usage or {}).get("output_tokens") or 0,
    }
    merged["input_tokens"] += (extra_usage or {}).get("input_tokens") or 0
    merged["output_tokens"] += (extra_usage or {}).get("output_tokens") or 0

    base_server = dict((base_usage or {}).get("server_tool_use") or {})
    extra_server = (extra_usage or {}).get("server_tool_use") or {}
    for key, value in extra_server.items():
        base_server[key] = (base_server.get(key) or 0) + (value or 0)
    if base_server:
        merged["server_tool_use"] = base_server
    return merged


async def execute_web_search_tool(block: dict[str, Any], metadata: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, int]]:
    tool_use_id = block.get("id") or normalize_anthropic_tool_use_id(None)
    tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
    query = str(
        tool_input.get("query")
        or tool_input.get("q")
        or tool_input.get("search_query")
        or ""
    ).strip()

    if not query:
        result_block = build_server_tool_result_error_block(
            "web_search_tool_result",
            tool_use_id,
            "invalid_input",
            "web_search 需要 query 字段。",
        )
        return result_block, json_dumps({"error": "missing_query"}), {"web_search_requests": 1}

    if len(query) > WEB_SEARCH_MAX_QUERY_LENGTH:
        result_block = build_server_tool_result_error_block(
            "web_search_tool_result",
            tool_use_id,
            "query_too_long",
            f"query 长度不能超过 {WEB_SEARCH_MAX_QUERY_LENGTH} 个字符。",
        )
        return result_block, json_dumps({"error": "query_too_long", "query": query}), {"web_search_requests": 1}

    max_uses = metadata.get("max_uses")
    if isinstance(max_uses, int) and metadata.get("uses", 0) >= max_uses:
        result_block = build_server_tool_result_error_block(
            "web_search_tool_result",
            tool_use_id,
            "max_uses_exceeded",
            "web_search 已达到当前请求允许的最大调用次数。",
        )
        return result_block, json_dumps({"error": "max_uses_exceeded", "query": query}), {"web_search_requests": 0}

    metadata["uses"] = metadata.get("uses", 0) + 1

    search_query = query
    user_location = metadata.get("user_location") if isinstance(metadata.get("user_location"), dict) else None
    if user_location:
        location_parts = [
            str(user_location.get(key)).strip()
            for key in ("city", "region", "country")
            if user_location.get(key)
        ]
        if location_parts:
            search_query = f"{query} {' '.join(location_parts)}"

    client = await get_http_client()
    try:
        response = await client.get(
            WEB_SEARCH_RSS_URL,
            params={"format": "rss", "q": search_query},
            headers={
                "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8",
                "User-Agent": "nim4cc/1.0 (+https://github.com/Geek66666/nim4cc)",
            },
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        result_block = build_server_tool_result_error_block(
            "web_search_tool_result",
            tool_use_id,
            "search_unavailable",
            f"web_search 上游请求失败：HTTP {exc.response.status_code}",
        )
        return result_block, json_dumps({"error": "search_unavailable", "query": query}), {"web_search_requests": 1}
    except httpx.RequestError as exc:
        result_block = build_server_tool_result_error_block(
            "web_search_tool_result",
            tool_use_id,
            "search_unavailable",
            f"web_search 请求异常：{exc}",
        )
        return result_block, json_dumps({"error": "search_unavailable", "query": query}), {"web_search_requests": 1}

    allowed_domains = metadata.get("allowed_domains") or []
    blocked_domains = metadata.get("blocked_domains") or []
    parsed_results = [
        item
        for item in parse_rss_results(response.text)
        if filter_web_search_url(item.get("url", ""), allowed_domains, blocked_domains)
    ][:max(1, WEB_SEARCH_DEFAULT_MAX_RESULTS)]

    outward_results: list[dict[str, Any]] = []
    model_results: list[dict[str, Any]] = []
    for result in parsed_results:
        encrypted_content = build_web_search_encrypted_content(
            {
                "query": query,
                "url": result.get("url"),
                "title": result.get("title"),
                "snippet": result.get("snippet"),
                "page_age": result.get("page_age"),
                "retrieved_at": utcnow_iso(),
            }
        )
        outward_item = {
            "type": "web_search_result",
            "url": result.get("url"),
            "title": result.get("title"),
            "encrypted_content": encrypted_content,
        }
        if result.get("page_age"):
            outward_item["page_age"] = result.get("page_age")
        outward_results.append(outward_item)
        model_results.append(
            {
                "url": result.get("url"),
                "title": result.get("title"),
                "snippet": result.get("snippet"),
                "page_age": result.get("page_age"),
                "encrypted_content": encrypted_content,
            }
        )

    result_block = {
        "type": "web_search_tool_result",
        "tool_use_id": tool_use_id,
        "content": outward_results,
    }
    model_payload = json_dumps({"query": query, "results": model_results})
    return result_block, model_payload, {"web_search_requests": 1}


async def execute_anthropic_server_tool_block(
    block: dict[str, Any],
    tool_metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[str, int]]:
    tool_name = block.get("name")
    metadata = tool_metadata.get(tool_name or "")
    if not metadata:
        result_block = build_server_tool_result_error_block(
            "tool_result",
            block.get("id") or normalize_anthropic_tool_use_id(None),
            "unknown_tool",
            f"未找到服务端工具 {tool_name} 的元数据。",
        )
        return result_block, json_dumps({"error": "unknown_tool"}), {}

    server_execution = metadata.get("server_execution")
    if server_execution == "web_search":
        return await execute_web_search_tool(block, metadata)

    result_block = build_server_tool_result_error_block(
        "tool_result",
        block.get("id") or normalize_anthropic_tool_use_id(None),
        "unsupported_tool",
        f"当前网关尚未实现服务端工具 {tool_name}。",
    )
    return result_block, json_dumps({"error": "unsupported_tool"}), {}


def anthropic_stop_reason(finish_reason: str | None, content_blocks: list[dict[str, Any]]) -> str:
    if any(block.get("type") == "tool_use" for block in content_blocks):
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    return "end_turn"


def chat_completion_to_anthropic_message(
    body: dict[str, Any],
    upstream_json: dict[str, Any],
    tool_metadata: dict[str, dict[str, Any]],
    thinking_config: dict[str, Any],
) -> dict[str, Any]:
    upstream_message, finish_reason = extract_upstream_message(upstream_json)
    assistant_text, tool_calls = extract_text_and_tool_calls(upstream_message)
    content_blocks: list[dict[str, Any]] = []
    if assistant_text:
        if thinking_config.get("enabled"):
            content_blocks.extend(split_anthropic_thinking_blocks(assistant_text, body.get("model")))
        else:
            content_blocks.append({"type": "text", "text": assistant_text})
    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        tool_info = tool_metadata.get(tool_name or "", {})
        is_server_tool = bool(tool_info.get("server_execution"))
        content_blocks.append(
            {
                "type": "server_tool_use" if is_server_tool else "tool_use",
                "id": (
                    tool_call.get("id")
                    if isinstance(tool_call.get("id"), str) and tool_call.get("id").startswith("srvtoolu_")
                    else (f"srvtoolu_{uuid.uuid4().hex[:24]}" if is_server_tool else normalize_anthropic_tool_use_id(tool_call.get("id")))
                ),
                "name": tool_name,
                "input": parse_anthropic_tool_input(tool_call.get("arguments")),
                **({} if is_server_tool else {"caller": {"type": "direct"}}),
            }
        )

    usage = upstream_json.get("usage") or {}
    return {
        "id": normalize_anthropic_message_id(upstream_json.get("id")),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": body.get("model"),
        "stop_reason": anthropic_stop_reason(finish_reason, content_blocks),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        },
    }


def build_anthropic_storage_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if body.get("system") is not None:
        items.append({"role": "system", "content": body.get("system")})
    for message in body.get("messages") or []:
        if isinstance(message, dict):
            items.append(message)
        else:
            items.append({"role": "user", "content": str(message)})
    return items


async def create_anthropic_message_with_server_tools(
    api_key: str,
    body: dict[str, Any],
    chat_payload: dict[str, Any],
    tool_metadata: dict[str, dict[str, Any]],
    thinking_config: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    request_messages = list(chat_payload.get("messages") or [])
    base_payload = {key: value for key, value in chat_payload.items() if key != "messages"}
    accumulated_content: list[dict[str, Any]] = []
    accumulated_usage: dict[str, Any] = {}
    last_message_payload: dict[str, Any] | None = None
    started = time.perf_counter()

    for _ in range(ANTHROPIC_SERVER_TOOL_MAX_ITERATIONS):
        loop_payload = {**base_payload, "messages": request_messages}
        upstream_json, _latency_ms = await post_nvidia_chat_completion(api_key, loop_payload)
        loop_message = chat_completion_to_anthropic_message(body, upstream_json, tool_metadata, thinking_config)
        last_message_payload = loop_message
        accumulated_usage = merge_anthropic_usage(accumulated_usage, loop_message.get("usage") or {})

        loop_content = list(loop_message.get("content") or [])
        accumulated_content.extend(loop_content)

        client_tool_blocks = [block for block in loop_content if block.get("type") == "tool_use"]
        server_tool_blocks = [block for block in loop_content if is_server_tool_block(block, tool_metadata)]

        if client_tool_blocks:
            break
        if not server_tool_blocks:
            break

        request_messages.append(anthropic_blocks_to_chat_assistant_message(loop_content))
        executed_usage: dict[str, Any] = {}
        for block in server_tool_blocks:
            result_block, model_tool_content, usage_increment = await execute_anthropic_server_tool_block(block, tool_metadata)
            accumulated_content.append(result_block)
            executed_usage = merge_anthropic_usage(executed_usage, {"server_tool_use": usage_increment})
            request_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("id"),
                    "content": model_tool_content,
                }
            )
        accumulated_usage = merge_anthropic_usage(accumulated_usage, executed_usage)
    else:
        accumulated_content.append(
            build_server_tool_result_error_block(
                "tool_result",
                f"srvtoolu_{uuid.uuid4().hex[:24]}",
                "max_iterations_exceeded",
                "服务端工具循环次数过多，已停止继续执行。",
            )
        )

    if last_message_payload is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="上游未返回有效的 Anthropic 兼容消息。",
        )

    message_payload = {
        **last_message_payload,
        "content": accumulated_content,
        "stop_reason": anthropic_stop_reason(None, accumulated_content),
        "usage": accumulated_usage,
    }
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return message_payload, latency_ms


def build_anthropic_streaming_response(message_payload: dict[str, Any], anthropic_version: str | None) -> StreamingResponse:
    async def event_stream() -> Any:
        opening_message = {
            **message_payload,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
        }
        yield f"event: message_start\ndata: {json_dumps({'type': 'message_start', 'message': opening_message})}\n\n"

        for index, block in enumerate(message_payload.get("content") or []):
            block_type = block.get("type")
            if block_type == "text":
                yield f"event: content_block_start\ndata: {json_dumps({'type': 'content_block_start', 'index': index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                text_value = str(block.get("text", ""))
                if text_value:
                    yield f"event: content_block_delta\ndata: {json_dumps({'type': 'content_block_delta', 'index': index, 'delta': {'type': 'text_delta', 'text': text_value}})}\n\n"
                yield f"event: content_block_stop\ndata: {json_dumps({'type': 'content_block_stop', 'index': index})}\n\n"
                continue

            if block_type == "thinking":
                content_block = {"type": "thinking", "thinking": ""}
                yield f"event: content_block_start\ndata: {json_dumps({'type': 'content_block_start', 'index': index, 'content_block': content_block})}\n\n"
                thinking_text = str(block.get("thinking", ""))
                if thinking_text:
                    yield f"event: content_block_delta\ndata: {json_dumps({'type': 'content_block_delta', 'index': index, 'delta': {'type': 'thinking_delta', 'thinking': thinking_text}})}\n\n"
                signature = block.get("signature")
                if signature:
                    yield f"event: content_block_delta\ndata: {json_dumps({'type': 'content_block_delta', 'index': index, 'delta': {'type': 'signature_delta', 'signature': signature}})}\n\n"
                yield f"event: content_block_stop\ndata: {json_dumps({'type': 'content_block_stop', 'index': index})}\n\n"
                continue

            if block_type == "tool_use":
                content_block = {**block, "input": {}}
                yield f"event: content_block_start\ndata: {json_dumps({'type': 'content_block_start', 'index': index, 'content_block': content_block})}\n\n"
                input_json = json_dumps(block.get("input") or {})
                if input_json:
                    yield f"event: content_block_delta\ndata: {json_dumps({'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': input_json}})}\n\n"
                yield f"event: content_block_stop\ndata: {json_dumps({'type': 'content_block_stop', 'index': index})}\n\n"
                continue

            if block_type == "server_tool_use":
                content_block = {**block, "input": {}}
                yield f"event: content_block_start\ndata: {json_dumps({'type': 'content_block_start', 'index': index, 'content_block': content_block})}\n\n"
                input_json = json_dumps(block.get("input") or {})
                if input_json:
                    yield f"event: content_block_delta\ndata: {json_dumps({'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': input_json}})}\n\n"
                yield f"event: content_block_stop\ndata: {json_dumps({'type': 'content_block_stop', 'index': index})}\n\n"
                continue

            if isinstance(block_type, str) and block_type.endswith("_tool_result"):
                yield f"event: content_block_start\ndata: {json_dumps({'type': 'content_block_start', 'index': index, 'content_block': block})}\n\n"
                yield f"event: content_block_stop\ndata: {json_dumps({'type': 'content_block_stop', 'index': index})}\n\n"
                continue

        yield f"event: message_delta\ndata: {json_dumps({'type': 'message_delta', 'delta': {'stop_reason': message_payload.get('stop_reason'), 'stop_sequence': message_payload.get('stop_sequence')}, 'usage': {'output_tokens': (message_payload.get('usage') or {}).get('output_tokens')}})}\n\n"
        yield "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n"

    headers = {
        "anthropic-version": anthropic_version or ANTHROPIC_API_VERSION,
        "cache-control": "no-cache",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


def extract_upstream_message(upstream_json: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    choices = upstream_json.get("choices") or []
    if not choices:
        return {}, None
    choice = choices[0] or {}
    return choice.get("message") or {}, choice.get("finish_reason")


def extract_text_and_tool_calls(message: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    content = message.get("content")
    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    if isinstance(content, str):
        text_chunks.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                text_chunks.append(part)
                continue
            if not isinstance(part, dict):
                text_chunks.append(str(part))
                continue
            if part.get("type") in {"input_text", "output_text", "text"}:
                text_chunks.append(str(part.get("text", "")))
                continue
            if part.get("type") in {"tool_call", "function_call"}:
                arguments = part.get("arguments") or "{}"
                if not isinstance(arguments, str):
                    arguments = json_dumps(arguments)
                tool_calls.append({
                    "id": part.get("id") or part.get("call_id") or f"call_{uuid.uuid4().hex[:12]}",
                    "name": part.get("name"),
                    "arguments": arguments,
                })

    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function_data = tool_call.get("function") or {}
        arguments = function_data.get("arguments") or tool_call.get("arguments") or "{}"
        if not isinstance(arguments, str):
            arguments = json_dumps(arguments)
        tool_calls.append(
            {
                "id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "name": function_data.get("name") or tool_call.get("name"),
                "arguments": arguments,
            }
        )

    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for tool_call in tool_calls:
        if tool_call["id"] in seen_ids:
            continue
        seen_ids.add(tool_call["id"])
        deduped.append(tool_call)
    return "\n".join(filter(None, text_chunks)).strip(), deduped

def build_choice_alias(output_items: list[dict[str, Any]], finish_reason: str | None) -> list[dict[str, Any]]:
    content_parts: list[dict[str, Any]] = []
    for item in output_items:
        if item.get("type") == "message":
            for part in item.get("content", []):
                content_parts.append({"type": part.get("type", "output_text"), "text": part.get("text", "")})
        elif item.get("type") == "function_call":
            arguments = item.get("arguments") or "{}"
            try:
                parsed_arguments = json.loads(arguments)
            except Exception:
                parsed_arguments = arguments
            content_parts.append({"type": "tool_call", "id": item.get("call_id"), "name": item.get("name"), "arguments": parsed_arguments})
    return [{"index": 0, "message": {"role": "assistant", "content": content_parts}, "finish_reason": finish_reason or "stop"}]


def chat_completion_to_response(body: dict[str, Any], upstream_json: dict[str, Any], previous_response_id: str | None) -> dict[str, Any]:
    upstream_message, finish_reason = extract_upstream_message(upstream_json)
    assistant_text, tool_calls = extract_text_and_tool_calls(upstream_message)
    response_id = upstream_json.get("id") or f"resp_{uuid.uuid4().hex}"
    output_items: list[dict[str, Any]] = []
    if assistant_text:
        output_items.append({
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": assistant_text, "annotations": []}],
        })
    for tool_call in tool_calls:
        output_items.append({
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "type": "function_call",
            "status": "completed",
            "call_id": tool_call["id"],
            "name": tool_call.get("name"),
            "arguments": tool_call.get("arguments", "{}"),
        })
    usage = upstream_json.get("usage") or {}
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": body.get("model"),
        "output": output_items,
        "output_text": assistant_text,
        "parallel_tool_calls": bool(body.get("parallel_tool_calls", True)),
        "previous_response_id": previous_response_id,
        "store": True,
        "text": body.get("text") or {"format": {"type": "text"}},
        "usage": {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "choices": build_choice_alias(output_items, finish_reason),
        "upstream": {
            "id": upstream_json.get("id"),
            "object": upstream_json.get("object", "chat.completion"),
            "finish_reason": finish_reason or "stop",
        },
    }


def store_success_record(api_key_hash: str, model_id: str, request_body: dict[str, Any], input_items: list[dict[str, Any]], response_payload: dict[str, Any], latency_ms: float) -> None:
    conn = get_db_connection()
    try:
        now = utcnow_iso()
        bucket = bucket_start().isoformat()
        output_items = response_payload.get("output")
        if not isinstance(output_items, list):
            output_items = response_payload.get("content")
        if not isinstance(output_items, list):
            output_items = []
        conn.execute(
            """
            INSERT OR REPLACE INTO response_records (
                response_id, api_key_hash, parent_response_id, model_id, request_json,
                input_items_json, output_json, output_items_json, status, success,
                latency_ms, error_message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                response_payload["id"],
                api_key_hash,
                request_body.get("previous_response_id"),
                model_id,
                json_dumps(request_body),
                json_dumps(input_items),
                json_dumps(response_payload),
                json_dumps(output_items),
                response_payload.get("status", "completed"),
                1,
                latency_ms,
                None,
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO metric_buckets (bucket_start, model_id, total_count, success_count, total_latency_ms)
            VALUES (?, ?, 1, 1, ?)
            ON CONFLICT(bucket_start, model_id) DO UPDATE SET
                total_count = total_count + 1,
                success_count = success_count + 1,
                total_latency_ms = total_latency_ms + excluded.total_latency_ms
            """,
            (bucket, model_id, latency_ms),
        )
        conn.execute(
            """
            UPDATE gateway_totals
            SET total_requests = total_requests + 1,
                total_success = total_success + 1,
                total_latency_ms = total_latency_ms + ?,
                updated_at = ?
            WHERE id = 1
            """,
            (latency_ms, now),
        )
        conn.commit()
    finally:
        conn.close()


def store_failure_metric(model_id: str, error_message: str) -> None:
    conn = get_db_connection()
    try:
        now = utcnow_iso()
        bucket = bucket_start().isoformat()
        conn.execute(
            """
            INSERT INTO metric_buckets (bucket_start, model_id, total_count, success_count, total_latency_ms)
            VALUES (?, ?, 1, 0, 0)
            ON CONFLICT(bucket_start, model_id) DO UPDATE SET
                total_count = total_count + 1
            """,
            (bucket, model_id),
        )
        conn.execute(
            """
            UPDATE gateway_totals
            SET total_requests = total_requests + 1,
                updated_at = ?
            WHERE id = 1
            """,
            (now,),
        )
        conn.commit()
    finally:
        conn.close()


def load_previous_conversation_items(api_key_hash: str, previous_response_id: str | None) -> list[dict[str, Any]]:
    if not previous_response_id:
        return []
    conn = get_db_connection()
    try:
        items: list[dict[str, Any]] = []
        current = previous_response_id
        chain: list[sqlite3.Row] = []
        while current:
            row = conn.execute(
                "SELECT * FROM response_records WHERE response_id = ? AND api_key_hash = ?",
                (current, api_key_hash),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"previous_response_id '{current}' 不存在，或不属于当前 Key。")
            chain.append(row)
            current = row["parent_response_id"]
        for row in reversed(chain):
            items.extend(json.loads(row["input_items_json"]))
            items.extend(json.loads(row["output_items_json"]))
        return items
    finally:
        conn.close()


def load_response_record(api_key_hash: str, response_id: str) -> dict[str, Any]:
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT output_json FROM response_records WHERE response_id = ? AND api_key_hash = ?",
            (response_id, api_key_hash),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="未找到对应响应，或当前 Key 无权访问。")
        return json.loads(row["output_json"])
    finally:
        conn.close()


def load_dashboard_data() -> dict[str, Any]:
    conn = get_db_connection()
    try:
        totals_row = conn.execute("SELECT * FROM gateway_totals WHERE id = 1").fetchone()
        total_requests = totals_row["total_requests"] if totals_row else 0
        now_bucket = bucket_start()
        bucket_points = [(now_bucket - timedelta(minutes=BUCKET_MINUTES * offset)).isoformat() for offset in reversed(range(PUBLIC_HISTORY_BUCKETS))]
        health_window_buckets = max(1, HEALTH_SUMMARY_WINDOW_MINUTES // BUCKET_MINUTES)
        health_window_points = [
            (now_bucket - timedelta(minutes=BUCKET_MINUTES * offset)).isoformat()
            for offset in reversed(range(health_window_buckets))
        ]
        placeholders = ",".join("?" for _ in MODEL_LIST) if MODEL_LIST else "''"
        totals_by_model = {
            row["model_id"]: row["total_count"]
            for row in conn.execute(
                f"SELECT model_id, COALESCE(SUM(total_count), 0) AS total_count FROM metric_buckets WHERE model_id IN ({placeholders}) GROUP BY model_id",
                MODEL_LIST,
            ).fetchall()
        } if MODEL_LIST else {}
        since_candidates = [value for value in [bucket_points[0] if bucket_points else None, health_window_points[0] if health_window_points else None] if value]
        since = min(since_candidates) if since_candidates else utcnow_iso()
        recent_rows = conn.execute(
            f"SELECT bucket_start, model_id, total_count, success_count FROM metric_buckets WHERE model_id IN ({placeholders}) AND bucket_start >= ? ORDER BY bucket_start ASC",
            [*MODEL_LIST, since],
        ).fetchall() if MODEL_LIST else []
        row_map: dict[str, dict[str, sqlite3.Row]] = {}
        for row in recent_rows:
            row_map.setdefault(row["model_id"], {})[row["bucket_start"]] = row
        models: list[dict[str, Any]] = []
        health_window_rates: list[float] = []
        for model_id in MODEL_LIST:
            points: list[dict[str, Any]] = []
            latest_bucket_rate: float | None = None
            for bucket_value in bucket_points:
                row = row_map.get(model_id, {}).get(bucket_value)
                total_count = row["total_count"] if row else 0
                success_count = row["success_count"] if row else 0
                success_rate = round((success_count / total_count) * 100, 1) if total_count else None
                points.append(
                    {
                        "bucket_start": bucket_value,
                        "label": bucket_label(bucket_value),
                        "total_count": total_count,
                        "success_count": success_count,
                        "success_rate": success_rate,
                    }
                )
                if bucket_value == bucket_points[-1] and total_count:
                    latest_bucket_rate = success_rate

            health_window_total = 0
            health_window_success = 0
            for bucket_value in health_window_points:
                row = row_map.get(model_id, {}).get(bucket_value)
                if not row:
                    continue
                health_window_total += row["total_count"]
                health_window_success += row["success_count"]

            health_window_rate = round((health_window_success / health_window_total) * 100, 1) if health_window_total else None
            if health_window_rate is not None:
                health_window_rates.append(health_window_rate)
            models.append(
                {
                    "model_id": model_id,
                    "provider": normalize_provider(model_id),
                    "total_calls": totals_by_model.get(model_id, 0),
                    "latest_success_rate": health_window_rate,
                    "average_success_rate": health_window_rate,
                    "health_window_success_rate": health_window_rate,
                    "health_window_total_count": health_window_total,
                    "health_window_success_count": health_window_success,
                    "health_window_minutes": HEALTH_SUMMARY_WINDOW_MINUTES,
                    "latest_bucket_success_rate": latest_bucket_rate,
                    "points": points,
                }
            )
        average_health = round(sum(health_window_rates) / len(health_window_rates), 1) if health_window_rates else None
        return {
            "generated_at": utcnow_iso(),
            "bucket_minutes": BUCKET_MINUTES,
            "health_window_minutes": HEALTH_SUMMARY_WINDOW_MINUTES,
            "total_requests": total_requests,
            "average_health": average_health,
            "models": models,
        }
    finally:
        conn.close()


def build_catalog_payload() -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for model in sorted(model_cache, key=lambda item: item.get("id", "")):
        provider = normalize_provider(model.get("id", ""), model.get("owned_by"))
        grouped.setdefault(provider, []).append(model)
    providers = [
        {
            "provider": provider,
            "count": len(items),
            "models": items,
        }
        for provider, items in sorted(grouped.items(), key=lambda entry: entry[0].lower())
    ]
    return {
        "generated_at": utcnow_iso(),
        "synced_at": model_cache_synced_at,
        "total_models": len(model_cache),
        "providers": providers,
    }


async def post_nvidia_chat_completion(api_key: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    client = await get_http_client()
    started = time.perf_counter()
    total_attempts = UPSTREAM_TIMEOUT_RETRIES + 1
    last_timeout: httpx.TimeoutException | None = None

    for attempt in range(1, total_attempts + 1):
        try:
            response = await client.post(
                CHAT_COMPLETIONS_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"},
                json=payload,
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            if response.status_code >= 400:
                try:
                    error_payload = response.json()
                    detail = error_payload.get("error", {}).get("message") or json_dumps(error_payload)
                except Exception:
                    detail = response.text
                raise HTTPException(status_code=response.status_code, detail=f"NVIDIA NIM 请求失败：{detail}")
            return response.json(), latency_ms
        except httpx.TimeoutException as exc:
            last_timeout = exc
            if attempt >= total_attempts:
                break
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"NVIDIA NIM 请求异常：{exc}",
            ) from exc

    detail = f"NVIDIA NIM 请求超时，已自动重试 {UPSTREAM_TIMEOUT_RETRIES} 次后仍未成功。"
    if last_timeout and str(last_timeout):
        detail = f"{detail} 最后错误：{last_timeout}"
    raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=detail)


def render_html(filename: str) -> HTMLResponse:
    content = (STATIC_DIR / filename).read_text(encoding="utf-8")
    return HTMLResponse(content=content, media_type="text/html; charset=utf-8")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model_cache, model_cache_synced_at, model_sync_task, http_client, model_cache_lock
    init_db()
    cached_models, cached_synced_at = await run_db(load_cached_models_from_db)
    model_cache = cached_models
    model_cache_synced_at = cached_synced_at
    model_cache_lock = asyncio.Lock()
    http_client = await get_http_client()
    try:
        await refresh_official_models(force=not bool(model_cache))
    except Exception:
        pass
    model_sync_task = asyncio.create_task(model_sync_loop())
    try:
        yield
    finally:
        if model_sync_task is not None:
            model_sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await model_sync_task
        if http_client is not None and not http_client.is_closed:
            await http_client.aclose()
        http_client = None
        model_sync_task = None
        model_cache_lock = None


app = FastAPI(title="NIM Responses Gateway", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def homepage() -> HTMLResponse:
    return render_html("index.html")


@app.get("/model_list", response_class=HTMLResponse)
async def models_page() -> HTMLResponse:
    return render_html("models.html")


@app.get("/api/dashboard")
async def dashboard_api() -> dict[str, Any]:
    return await run_db(load_dashboard_data)


@app.get("/api/catalog")
async def catalog_api() -> dict[str, Any]:
    if not model_cache:
        try:
            await refresh_official_models(force=True)
        except Exception:
            pass
    return build_catalog_payload()


async def build_models_response() -> dict[str, Any]:
    if not model_cache:
        await refresh_official_models(force=True)
    return {"object": "list", "data": model_cache}


@app.get("/v1")
async def v1_info() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "NIM4CC Gateway",
        "version": "1.0",
        "endpoints": {
            "models": "/v1/models",
            "responses": "/v1/responses",
            "messages": "/v1/messages"
        }
    }


@app.get("/v1/models")
async def list_models_v1() -> dict[str, Any]:
    return await build_models_response()


@app.get("/models")
async def list_models() -> dict[str, Any]:
    return await build_models_response()


async def fetch_response_record(response_id: str, api_key: str) -> dict[str, Any]:
    return await run_db(load_response_record, hash_api_key(api_key), response_id)


@app.post("/v1/messages")
async def create_anthropic_message(
    request: Request,
    api_key: str = Depends(extract_user_api_key),
    anthropic_version: str | None = Header(default=None),
    anthropic_beta: str | None = Header(default=None),
):
    return await create_anthropic_message_impl(request, api_key, anthropic_version, anthropic_beta)


@app.get("/v1/responses/{response_id}")
async def get_response_v1(response_id: str, api_key: str = Depends(extract_user_api_key)) -> dict[str, Any]:
    return await fetch_response_record(response_id, api_key)


@app.get("/responses/{response_id}")
async def get_response(response_id: str, api_key: str = Depends(extract_user_api_key)) -> dict[str, Any]:
    return await fetch_response_record(response_id, api_key)


@app.post("/v1/responses")
async def create_response_v1(request: Request, api_key: str = Depends(extract_user_api_key)):
    return await create_response_impl(request, api_key)


@app.post("/responses")
async def create_response(request: Request, api_key: str = Depends(extract_user_api_key)):
    return await create_response_impl(request, api_key)


@app.post("/v1/chat/completions")
async def chat_completions_v1(request: Request, api_key: str = Depends(extract_user_api_key)):
    """Standard OpenAI Chat Completions format passthrough"""
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请求体必须是 JSON 对象。")
    if not body.get("model"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="缺少 model 字段。")

    client = await get_http_client()
    response = await client.post(
        CHAT_COMPLETIONS_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"},
        json=body,
    )
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


@app.post("/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(extract_user_api_key)):
    """Standard OpenAI Chat Completions format passthrough"""
    return await chat_completions_v1(request, api_key)


async def create_anthropic_message_impl(
    request: Request,
    api_key: str,
    anthropic_version: str | None,
    anthropic_beta: str | None = None,
):
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请求体必须是 JSON 对象。")
    if not body.get("model"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="缺少 model 字段。")
    if body.get("messages") is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="缺少 messages 字段。")
    if not isinstance(body.get("messages"), list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="messages 字段必须是数组。")
    if body.get("max_tokens") is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="缺少 max_tokens 字段。")

    api_key_hash = hash_api_key(api_key)
    storage_items = build_anthropic_storage_items(body)
    chat_payload, _chat_messages, tool_metadata, thinking_config = build_anthropic_chat_payload(body, anthropic_beta)
    has_server_tools = any(meta.get("server_execution") for meta in tool_metadata.values())

    try:
        if has_server_tools:
            message_payload, latency_ms = await create_anthropic_message_with_server_tools(
                api_key,
                body,
                chat_payload,
                tool_metadata,
                thinking_config,
            )
        else:
            upstream_json, latency_ms = await post_nvidia_chat_completion(api_key, chat_payload)
            message_payload = chat_completion_to_anthropic_message(body, upstream_json, tool_metadata, thinking_config)
        await run_db(store_success_record, api_key_hash, body.get("model"), body, storage_items, message_payload, latency_ms)
    except HTTPException as exc:
        with contextlib.suppress(Exception):
            await run_db(store_failure_metric, body.get("model"), str(exc.detail))
        raise
    except Exception as exc:
        with contextlib.suppress(Exception):
            await run_db(store_failure_metric, body.get("model"), str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="网关处理 Anthropic Messages 请求时发生内部错误。",
        ) from exc

    resolved_version = anthropic_version or ANTHROPIC_API_VERSION
    if body.get("stream"):
        return build_anthropic_streaming_response(message_payload, resolved_version)

    return JSONResponse(content=message_payload, headers={"anthropic-version": resolved_version})


async def create_response_impl(request: Request, api_key: str):
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请求体必须是 JSON 对象。")
    if not body.get("model"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="缺少 model 字段。")
    if body.get("input") is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="缺少 input 字段。")

    api_key_hash = hash_api_key(api_key)
    input_items = normalize_input_items(body.get("input"))
    previous_items = await run_db(load_previous_conversation_items, api_key_hash, body.get("previous_response_id"))
    merged_items = previous_items + input_items
    chat_payload = build_chat_payload(body, merged_items)

    try:
        upstream_json, latency_ms = await post_nvidia_chat_completion(api_key, chat_payload)
        response_payload = chat_completion_to_response(body, upstream_json, body.get("previous_response_id"))
        await run_db(store_success_record, api_key_hash, body.get("model"), body, input_items, response_payload, latency_ms)
    except HTTPException as exc:
        with contextlib.suppress(Exception):
            await run_db(store_failure_metric, body.get("model"), str(exc.detail))
        raise
    except Exception as exc:
        with contextlib.suppress(Exception):
            await run_db(store_failure_metric, body.get("model"), str(exc))
        import traceback
        print(f"ERROR: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"网关处理请求时发生内部错误: {str(exc)}",
        ) from exc

    if body.get("stream"):
        async def event_stream() -> Any:
            yield f"event: response.created\ndata: {json_dumps({'type': 'response.created', 'response': {'id': response_payload['id'], 'model': response_payload['model'], 'status': 'in_progress'}})}\n\n"
            for index, item in enumerate(response_payload.get("output") or []):
                yield f"event: response.output_item.added\ndata: {json_dumps({'type': 'response.output_item.added', 'output_index': index, 'item': item})}\n\n"
                if item.get("type") == "message":
                    text_value = extract_text_from_content(item.get("content"))
                    if text_value:
                        yield f"event: response.output_text.delta\ndata: {json_dumps({'type': 'response.output_text.delta', 'output_index': index, 'delta': text_value})}\n\n"
                        yield f"event: response.output_text.done\ndata: {json_dumps({'type': 'response.output_text.done', 'output_index': index, 'text': text_value})}\n\n"
                if item.get("type") == "function_call":
                    yield f"event: response.function_call_arguments.done\ndata: {json_dumps({'type': 'response.function_call_arguments.done', 'output_index': index, 'arguments': item.get('arguments', '{}'), 'call_id': item.get('call_id')})}\n\n"
                yield f"event: response.output_item.done\ndata: {json_dumps({'type': 'response.output_item.done', 'output_index': index, 'item': item})}\n\n"
            yield f"event: response.completed\ndata: {json_dumps({'type': 'response.completed', 'response': response_payload})}\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return response_payload

