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
from ppagent.providers import (
    detect_provider,
    is_reasoning_model,
    thinking_extra_body_for,
)

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

    def _resolve_instructor_mode(self) -> Any:
        mode_str = self.config.instructor_mode.lower()
        if mode_str == "auto":
            if is_reasoning_model(self.config.model):
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

    def _thinking_kwargs(self) -> dict[str, Any]:
        """Return extra API kwargs to enable extended thinking/reasoning.

        The per-provider ``extra_body`` payload is sourced from the central
        provider registry (see :mod:`ppagent.providers`). When thinking is
        enabled but the active provider has no known thinking parameter, an
        info note is logged and an empty dict is returned.
        """
        if not self.config.enable_thinking:
            return {}
        extra_body = thinking_extra_body_for(self.config.base_url)
        if extra_body is None:
            logger.info(
                "enable_thinking is set but provider for '%s' has no known thinking parameter; none sent",
                self.config.base_url,
            )
            return {}
        return {"extra_body": extra_body}

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

    def _describe_config(self) -> str:
        """Return a short human-readable description of the current LLM config."""
        masked_key = (self.config.api_key[:8] + "...") if len(self.config.api_key) > 8 else "<not set>"
        provider = detect_provider(self.config.base_url)
        return (
            f"model={self.config.model!r}, base_url={self.config.base_url!r}, "
            f"provider={provider!r}, api_key={masked_key}"
        )

    @staticmethod
    def _friendly_error(exc: Exception, config_desc: str) -> str:
        """Build an actionable error message from an OpenAI SDK exception."""
        if isinstance(exc, openai.AuthenticationError):
            return (
                f"Authentication failed (HTTP 401). The API key is invalid or expired.\n"
                f"  Config: {config_desc}\n"
                f"  → Regenerate your API key at the provider's console and update config/settings.toml."
            )
        if isinstance(exc, openai.PermissionDeniedError):
            return (
                f"Permission denied (HTTP 403). The API key lacks access to this resource.\n"
                f"  Config: {config_desc}\n"
                f"  → Check that your account/plan supports model {exc.body.get('model', '')!r} "
                f"or request access from the provider."
            )
        if isinstance(exc, openai.NotFoundError):
            return (
                f"Model not found (HTTP 404). The model name may be incorrect or unavailable.\n"
                f"  Config: {config_desc}\n"
                f"  → Verify the model name in config/settings.toml against the provider's "
                f"available models list."
            )
        if isinstance(exc, openai.BadRequestError):
            detail = ""
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                err = body.get("error", {})
                detail = err.get("message", "") if isinstance(err, dict) else str(err)
            return (
                f"Bad request (HTTP 400). The API rejected the request.\n"
                f"  Config: {config_desc}\n"
                f"  → Provider detail: {detail or exc.message}\n"
                f"  → Common causes: unsupported parameters, invalid model name, "
                f"or thinking/reasoning params not supported by this model."
            )
        if isinstance(exc, openai.RateLimitError):
            return (
                f"Rate limit exceeded (HTTP 429).\n"
                f"  Config: {config_desc}\n"
                f"  → Wait a moment and retry, or reduce request frequency. "
                f"Check your plan's rate limits at the provider's console."
            )
        if isinstance(exc, openai.APIConnectionError):
            return (
                f"Could not connect to the LLM API endpoint.\n"
                f"  Config: {config_desc}\n"
                f"  → Check your internet connection and that the base_url is reachable."
            )
        if isinstance(exc, openai.InternalServerError):
            return (
                f"Provider server error (HTTP 5xx).\n"
                f"  Config: {config_desc}\n"
                f"  → The provider is experiencing issues. Retry after a short wait."
            )
        # Fallback
        return f"LLM API error: {exc}\n  Config: {config_desc}"

    # Error types that should NOT be retried — they require user action.
    _NON_RETRYABLE = (
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.NotFoundError,
        openai.BadRequestError,
    )

    def _call_with_retry(self, kwargs: dict[str, Any]) -> openai.types.chat.ChatCompletion:
        """Call the OpenAI API with exponential backoff on transient errors.

        Non-transient errors (401, 403, 404, 400) are raised immediately with
        an actionable message so the user knows exactly what to fix.
        """
        config_desc = self._describe_config()
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp
            except self._NON_RETRYABLE as exc:
                # Auth, permission, not-found, bad-request: do not retry.
                msg = self._friendly_error(exc, config_desc)
                logger.error(msg)
                raise RuntimeError(msg) from exc
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
        # All retries exhausted on a transient error.
        msg = self._friendly_error(last_err, config_desc) if last_err else "LLM API call failed"
        logger.error(msg)
        raise RuntimeError(f"{msg}\n  (failed after {_MAX_RETRIES} retries)") from last_err

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
