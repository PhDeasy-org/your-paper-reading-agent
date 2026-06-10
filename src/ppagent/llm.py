"""OpenAI-compatible LLM client wrapper with structured output support."""

from __future__ import annotations

import time
import logging
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

    def _resolve_instructor_mode(self) -> Any:
        mode_str = self.config.instructor_mode.lower()
        if mode_str == "auto":
            model_lower = self.config.model.lower()
            if "deepseek" in model_lower or "reasoner" in model_lower or "r1" in model_lower:
                logger.info("Auto-detected DeepSeek/reasoner model '%s': using MD_JSON mode", self.config.model)
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
        if tools:
            kwargs["tools"] = tools
        return self._call_with_retry(kwargs)

    def chat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Chat completion that guarantees a structured Pydantic output via instructor."""
        return self._instructor.chat.completions.create(
            model=self.config.model,
            messages=messages,
            response_model=response_model,
            temperature=self.config.temperature,
            max_tokens=self._clamp_max_tokens(None),
        )

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
