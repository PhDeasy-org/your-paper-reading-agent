"""OpenAI-compatible LLM client wrapper with structured output support."""

from __future__ import annotations

import base64
import time
import logging
import threading
from pathlib import Path
from typing import Any

import instructor
import openai
from pydantic import BaseModel

from ppagent.config import LLMConfig

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2  # seconds, doubles each retry


class LLMClient:
    """Thin wrapper around the OpenAI SDK for any compatible endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        )
        mode = self._resolve_instructor_mode()
        self._instructor = instructor.from_openai(self._client, mode=mode)
        self._thread_local = threading.local()

    def _get_local_usage(self) -> dict[str, int]:
        if not hasattr(self._thread_local, "usage"):
            self._thread_local.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return self._thread_local.usage

    def reset_usage(self) -> None:
        """Reset token usage counter for the current thread."""
        self._thread_local.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_usage(self) -> dict[str, int]:
        """Get accumulated token usage for the current thread."""
        return dict(self._get_local_usage())

    def _record_usage(self, usage: Any | None) -> None:
        if not usage:
            return
        local_usage = self._get_local_usage()
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        total = getattr(usage, "total_tokens", 0) or 0
        if total == 0:
            total = prompt + completion
        local_usage["prompt_tokens"] += prompt
        local_usage["completion_tokens"] += completion
        local_usage["total_tokens"] += total

    _REASONING_MODEL_HINTS = (
        "deepseek", "reasoner", "r1", "qwq", "qwq-plus",
        "magistral", "grok-4", "step-3", "mimo-v2",
        "glm-5", "glm-4.7", "doubao-seed",
    )

    def _resolve_instructor_mode(self) -> Any:
        mode_str = self.config.instructor_mode.lower()
        if mode_str == "auto":
            model_lower = self.config.model.lower()
            if any(hint in model_lower for hint in self._REASONING_MODEL_HINTS):
                logger.info("Auto-detected reasoning model '%s': using MD_JSON mode", self.config.model)
                return instructor.Mode.MD_JSON
            return instructor.Mode.TOOLS

        mapping = {
            "tools": instructor.Mode.TOOLS,
            "json": instructor.Mode.JSON,
            "md_json": instructor.Mode.MD_JSON,
            "json_schema": instructor.Mode.JSON_SCHEMA,
        }
        if mode_str not in mapping:
            logger.warning("Unknown instructor_mode '%s', falling back to TOOLS", self.config.instructor_mode)
            return instructor.Mode.TOOLS
        return mapping[mode_str]

    _PROVIDER_PATTERNS = (
        ("openai.com", "openai"),
        ("deepseek.com", "deepseek"),
        ("anthropic.com", "anthropic"),
        ("dashscope", "qwen"),
        ("aliyuncs.com", "qwen"),
        ("moonshot.cn", "kimi"),
        ("kimi.ai", "kimi"),
        ("googleapis.com", "gemini"),
        ("x.ai", "grok"),
        ("stepfun", "stepfun"),
        ("minimax", "minimax"),
        ("xiaomimimo.com", "mimo"),
        ("volces.com", "doubao"),
        ("volcengine.com", "doubao"),
        ("bigmodel.cn", "glm"),
        ("z.ai", "glm"),
        ("mistral.ai", "mistral"),
    )

    def _detect_provider(self) -> str | None:
        base = self.config.base_url.lower()
        for pattern, provider in self._PROVIDER_PATTERNS:
            if pattern in base:
                return provider
        return None

    def _thinking_kwargs(self) -> dict[str, Any]:
        """Return extra API kwargs to enable extended thinking/reasoning.

        Detects the provider from ``base_url`` and passes the exact parameters
        documented by each vendor so the model uses its default thinking budget.
        """
        if not self.config.enable_thinking:
            return {}
        provider = self._detect_provider()
        if provider == "openai":
            return {"extra_body": {"reasoning_effort": "medium"}}
        if provider == "deepseek":
            return {"extra_body": {"thinking": {"type": "enabled"}}}
        if provider == "anthropic":
            return {"extra_body": {"thinking": {"type": "enabled"}}}
        if provider == "qwen":
            return {"extra_body": {"enable_thinking": True}}
        if provider == "kimi":
            return {"extra_body": {"thinking": {"type": "enabled"}}}
        if provider == "gemini":
            return {"extra_body": {"reasoning_effort": "medium"}}
        if provider == "grok":
            return {"extra_body": {"reasoning_effort": "medium"}}
        if provider == "stepfun":
            return {"extra_body": {"reasoning_effort": "medium"}}
        if provider == "minimax":
            return {"extra_body": {"thinking": {"type": "enabled"}}}
        if provider == "mimo":
            return {"extra_body": {"thinking": {"type": "enabled"}}}
        if provider == "doubao":
            return {"extra_body": {"thinking": {"type": "enabled"}}}
        if provider == "glm":
            return {"extra_body": {"thinking": {"type": "enabled"}}}
        if provider == "mistral":
            return {"extra_body": {"reasoning_effort": "high"}}
        logger.warning("enable_thinking is set but provider for '%s' is not recognized; no thinking params sent", self.config.base_url)
        return {}

    def _clamp_max_tokens(self, max_tokens: int | None) -> int:
        val = max_tokens or self.config.max_tokens
        if val > 16384:
            logger.warning("max_tokens %d is abnormally high for output completion; clamping to 16384", val)
            return 16384
        return val

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> openai.types.chat.ChatCompletion:
        """Raw chat completion, optionally with tool definitions."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": self._clamp_max_tokens(max_tokens),
        }
        thinking = self._thinking_kwargs()
        if thinking:
            kwargs.update(thinking)
            extra = thinking.get("extra_body", {})
            if "reasoning_effort" in extra or extra.get("thinking"):
                kwargs.pop("temperature", None)
        if tools:
            kwargs["tools"] = tools
        resp = self._call_with_retry(kwargs)
        self._record_usage(resp.usage)
        return resp

    def chat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Chat completion that guarantees a structured Pydantic output via instructor."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "response_model": response_model,
            "temperature": self.config.temperature,
            "max_tokens": self._clamp_max_tokens(None),
        }
        thinking = self._thinking_kwargs()
        if thinking:
            kwargs.update(thinking)
            extra = thinking.get("extra_body", {})
            if "reasoning_effort" in extra or extra.get("thinking"):
                kwargs.pop("temperature", None)
        response, raw_completion = self._instructor.chat.completions.create_with_completion(**kwargs)
        self._record_usage(raw_completion.usage)
        return response

    def chat_vision(
        self,
        system: str,
        user_text: str,
        images: list[Path],
    ) -> str:
        """Multimodal chat completion: send images + text, return plain text.

        ``images`` are file paths to PNG/JPEGs. Each is embedded as a base64
        data URI so the call works with any OpenAI-compatible vision endpoint
        without exposing local files. Returns the assistant's text response.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img_path in images:
            data_uri = _image_to_data_uri(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": data_uri},
            })
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        resp = self.chat(messages)
        return resp.choices[0].message.content or ""

    def _call_with_retry(self, kwargs: dict[str, Any]) -> openai.types.chat.ChatCompletion:
        """Call the OpenAI API with exponential backoff on transient errors."""
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp
            except (
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
            ) as exc:
                last_err = exc
                wait = _RETRY_BACKOFF * (2**attempt)
                logger.warning("LLM API error (attempt %d/%d): %s — retrying in %ds",
                               attempt + 1, _MAX_RETRIES, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"LLM API call failed after {_MAX_RETRIES} retries: {last_err}")

    @staticmethod
    def build_messages(
        system: str,
        user: str,
        *,
        context: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Helper to build the messages array for a standard system+user call."""
        msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if context:
            msgs.extend(context)
        msgs.append({"role": "user", "content": user})
        return msgs


def _image_to_data_uri(img_path: Path) -> str:
    """Encode an image file as a base64 data URI for vision API calls."""
    ext = img_path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/{mime};base64,{b64}"
