"""
LLM 服务：保留原框架 Provider / Cache / 降级 三层结构。

支持 OpenAI 兼容 API（含第三方 base_url），失败时降级到 FallbackProvider。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import Any, Protocol

import httpx

from ..config import LLMConfig
from .retry import get_retry_wait_seconds

logger = logging.getLogger(__name__)


# ── Provider 协议 ────────────────────────────────────────────

class LLMProvider(Protocol):
    name: str

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """发送 chat completion 请求，返回 assistant 消息文本。"""
        ...


# ── OpenAI 兼容 Provider ────────────────────────────────────

class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", self.config.temperature)
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")


# ── 降级 Provider（不调用外部 API）──────────────────────────

class FallbackProvider:
    """当 LLM 不可用时返回空占位，由各阶段自行做 heuristic 处理。"""
    name = "fallback"

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return ""


# ── JSON 提取工具 ────────────────────────────────────────────

def extract_json(text: str) -> dict | list:
    """
    从 LLM 返回的文本中提取 JSON。

    策略：
    1. 尝试直接解析整段文本
    2. 寻找 ```json ... ``` 代码块
    3. 寻找最外层 { ... } 或 [ ... ]
    """
    text = text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 寻找最外层 { ... } 或 [ ... ]
    for open_c, close_c in [("{", "}"), ("[", "]")]:
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

    raise ValueError("LLM 返回内容中未找到有效 JSON。")


# ── LLM 缓存（基于文件系统）─────────────────────────────────

class LLMCache:
    """简单的文件系统缓存，key 为 prompt hash。"""

    def __init__(self, cache_dir: str | None = None) -> None:
        self._cache_dir = cache_dir
        self._memory: dict[str, str] = {}

    @staticmethod
    def _hash(messages: list[dict[str, str]], model: str) -> str:
        payload = json.dumps({"model": model, "messages": messages}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, messages: list[dict[str, str]], model: str) -> str | None:
        key = self._hash(messages, model)
        return self._memory.get(key)

    def put(self, messages: list[dict[str, str]], model: str, response: str) -> None:
        key = self._hash(messages, model)
        self._memory[key] = response


# ── LLM Service（统一入口）───────────────────────────────────

class LLMService:
    """
    统一 LLM 调用入口，包含：
    - 自动重试
    - 缓存
    - 失败降级
    - JSON 提取
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._cache = LLMCache()
        self._provider: LLMProvider = self._build_provider(config)
        self._fallback = FallbackProvider()
        self.degraded = False
        self.degraded_reason: str | None = None

    @staticmethod
    def _build_provider(config: LLMConfig) -> LLMProvider:
        # 只要提供了 base_url、api_key、model，就视为 OpenAI 兼容 API
        # provider 字段仅用于标识（如 "zhipu"、"deepseek"、"openai_compatible" 等）
        if config.base_url and config.api_key and config.model:
            return OpenAICompatibleProvider(config)
        return FallbackProvider()

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """
        执行 chat completion，带缓存和重试。

        失败时降级到 FallbackProvider 并设置 self.degraded = True。
        """
        cached = self._cache.get(messages, self.config.model)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        total_attempts = max(1, self.config.max_retries)
        for attempt in range(1, total_attempts + 1):
            try:
                result = await self._provider.chat(messages, **kwargs)
                self._cache.put(messages, self.config.model, result)
                return result
            except Exception as e:
                last_error = e
                logger.warning("LLM 调用失败 (第 %d/%d 次): %s", attempt, total_attempts, e)
                if attempt >= total_attempts:
                    break

                wait_seconds = get_retry_wait_seconds(attempt)
                logger.info(
                    "LLM 将在 %d 秒后执行第 %d 次重试",
                    wait_seconds,
                    attempt,
                )
                await asyncio.sleep(wait_seconds)

        # 降级
        self.degraded = True
        self.degraded_reason = f"LLM 调用失败，已降级。错误: {last_error}"
        logger.error("LLM 全部重试失败，降级到 FallbackProvider: %s", last_error)
        return await self._fallback.chat(messages, **kwargs)

    async def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict | list:
        """调用 LLM 并提取 JSON，失败时抛出 ValueError。"""
        raw = await self.chat(messages, **kwargs)
        if not raw:
            raise ValueError("LLM 返回空内容，无法提取 JSON。")
        return extract_json(raw)

    @property
    def is_configured(self) -> bool:
        return isinstance(self._provider, OpenAICompatibleProvider)
