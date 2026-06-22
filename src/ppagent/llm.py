"""OpenAI-compatible LLM client wrapper with structured output support."""

from __future__ import annotations

import base64
import time
import logging
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import instructor
import openai
from pydantic import BaseModel

from ppagent.config import LLMConfig
from ppagent.providers import (
    detect_provider,
    is_reasoning_model,
    supports_responses_api_for,
    thinking_extra_body_for,
    thinking_incompatible_with_tools_for,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2  # seconds, doubles each retry


@dataclass
class _CompatToolCall:
    id: str
    call_id: str
    name: str
    arguments: str
    type: str = "function_call"


@dataclass
class _StreamedMessage:
    """Minimal assistant-message shape reconstructed from a streamed response."""

    content: str | None = None
    tool_calls: list[Any] | None = None


@dataclass
class _StreamedChoice:
    """Minimal choice shape reconstructed from a streamed Chat Completion."""

    message: _StreamedMessage = field(default_factory=_StreamedMessage)


@dataclass
class _StreamedRaw:
    """Shim raw object that lets ``LLMResponse`` read a streamed completion.

    A streamed call produces only text deltas (no tool calls — those turns are
    routed through the non-streaming path). This object exposes the handful of
    attributes the ``LLMResponse`` wrapper and ``_run_with_tools`` inspect:
    ``output_text`` (Responses), ``choices[0].message`` (Chat Completions),
    ``output`` (Responses tool list, always empty here), and ``usage``.
    """

    text: str = ""
    usage: Any = None
    choices: list[_StreamedChoice] = field(
        default_factory=lambda: [_StreamedChoice()]
    )

    @property
    def output_text(self) -> str:
        return self.text

    @property
    def output(self) -> list[Any]:
        # Tool turns never stream, so a streamed response carries no tool calls.
        return []

    def __post_init__(self) -> None:
        # Keep the single choice's message content in sync with ``text``.
        self.choices[0].message.content = self.text


class LLMResponse:
    """Unified wrapper around OpenAI ChatCompletion or Response."""

    def __init__(self, raw: Any) -> None:
        self.raw = raw

    @property
    def output_text(self) -> str:
        is_mock = type(self.raw).__name__ in ("MagicMock", "Mock") or hasattr(self.raw, "_mock_self")
        if hasattr(self.raw, "output_text"):
            if not is_mock or "output_text" in self.raw.__dict__:
                return self.raw.output_text
        try:
            return self.raw.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return ""

    @property
    def output(self) -> list[Any]:
        is_mock = type(self.raw).__name__ in ("MagicMock", "Mock") or hasattr(self.raw, "_mock_self")
        if hasattr(self.raw, "output"):
            if not is_mock or "output" in self.raw.__dict__:
                return self.raw.output
        items = []
        try:
            message = self.raw.choices[0].message
            if message.tool_calls:
                for call in message.tool_calls:
                    items.append(
                        _CompatToolCall(
                            id=call.id,
                            call_id=call.id,
                            name=call.function.name,
                            arguments=call.function.arguments,
                            type="function_call",
                        )
                    )
        except (AttributeError, IndexError):
            pass
        return items

    @property
    def usage(self) -> Any:
        return self.raw.usage


class LLMClient:
    """Thin wrapper around the OpenAI SDK for any compatible endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        if not config.api_key:
            from ppagent.providers import detect_provider

            vendor = detect_provider(config.base_url)
            raise ValueError(
                f"Missing API key for LLM provider '{vendor}' "
                f"(base_url={config.base_url!r}). "
                f"Run `ppagent config` and enter your API key."
            )
        self._client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        )
        self.use_responses = supports_responses_api_for(config.base_url)
        self._mode = self._resolve_instructor_mode()
        self._instructor = instructor.from_openai(self._client, mode=self._mode)
        self._thread_local = threading.local()

    def _get_local_usage(self) -> dict[str, int]:
        if not hasattr(self._thread_local, "usage"):
            self._thread_local.usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        return self._thread_local.usage

    def reset_usage(self) -> None:
        """Reset token usage counter for the current thread."""
        self._thread_local.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def get_usage(self) -> dict[str, int]:
        """Get accumulated token usage for the current thread."""
        return dict(self._get_local_usage())

    def _record_usage(self, usage: Any | None) -> None:
        if not usage:
            return
        local_usage = self._get_local_usage()
        prompt = getattr(usage, "prompt_tokens", None)
        if prompt is None:
            prompt = getattr(usage, "input_tokens", 0) or 0

        completion = getattr(usage, "completion_tokens", None)
        if completion is None:
            completion = getattr(usage, "output_tokens", 0) or 0
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
                logger.info(
                    "Auto-detected reasoning model '%s': using MD_JSON mode",
                    self.config.model,
                )
                return instructor.Mode.MD_JSON
            # Providers whose thinking mode is incompatible with tool_choice
            # (e.g. Qwen, Kimi) must fall back to MD_JSON when thinking is on.
            if (
                self.config.enable_thinking
                and thinking_incompatible_with_tools_for(self.config.base_url)
            ):
                logger.info(
                    "Thinking enabled on provider incompatible with tool_choice; "
                    "using MD_JSON mode for '%s'",
                    self.config.model,
                )
                return instructor.Mode.MD_JSON
            if self.use_responses:
                return instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS
            return instructor.Mode.TOOLS

        mapping = {
            "tools": instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS
            if self.use_responses
            else instructor.Mode.TOOLS,
            "json": instructor.Mode.JSON,
            "md_json": instructor.Mode.MD_JSON,
            "json_schema": instructor.Mode.JSON_SCHEMA,
        }
        if mode_str not in mapping:
            logger.warning(
                "Unknown instructor_mode '%s', falling back to TOOLS/RESPONSES_TOOLS_WITH_INBUILT_TOOLS",
                self.config.instructor_mode,
            )
            return (
                instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS
                if self.use_responses
                else instructor.Mode.TOOLS
            )
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
            logger.warning(
                "max_tokens %d is abnormally high for output completion; clamping to 16384",
                val,
            )
            return 16384
        return val

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        on_text: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Raw chat completion, optionally with tool definitions.

        When ``stream`` is True **and** no tools are requested, the response is
        streamed token-by-token: each text delta is passed to ``on_text`` (if
        provided) and a reconstructed :class:`LLMResponse` is returned once the
        stream completes. Turns that include tools are never streamed — they
        fall back to the normal blocking path so tool-call parsing stays
        byte-for-byte identical.
        """
        if self.use_responses:
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "input": messages,
                "temperature": temperature
                if temperature is not None
                else self.config.temperature,
                "max_output_tokens": self._clamp_max_tokens(max_tokens),
            }
        else:
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature
                if temperature is not None
                else self.config.temperature,
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

        # Streaming only applies to plain-text turns (no tools). Tool turns
        # parse tool_calls from the full response, so they always use the
        # blocking retry path.
        if stream and on_text is not None and not tools:
            raw = self._stream_create(kwargs, on_text)
        else:
            raw = self._call_with_retry(kwargs)
        self._record_usage(getattr(raw, "usage", None))
        return LLMResponse(raw)

    def _stream_create(
        self,
        kwargs: dict[str, Any],
        on_text: Callable[[str], None],
    ) -> _StreamedRaw:
        """Call the LLM with ``stream=True`` and return a reconstructed raw.

        Connection / pre-stream errors (auth, rate-limit, 5xx, connect) are
        handled with the same retry + friendly-error logic as the blocking
        path. Once tokens start flowing, a stream error is surfaced as a
        :class:`RuntimeError` — we never partially retry a mid-stream failure.
        """
        kwargs = {**kwargs, "stream": True}
        if not self.use_responses:
            # Ask the Chat Completions endpoint to emit a final usage chunk.
            kwargs["stream_options"] = {"include_usage": True}

        config_desc = self._describe_config()
        last_err: Exception | None = None
        stream: Iterator[Any] | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                if self.use_responses:
                    stream = self._client.responses.create(**kwargs)
                else:
                    stream = self._client.chat.completions.create(**kwargs)
                break
            except self._NON_RETRYABLE as exc:
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
                logger.warning(
                    "LLM API error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
        if stream is None:
            # All pre-stream retries exhausted.
            msg = (
                self._friendly_error(last_err, config_desc)
                if last_err
                else "LLM API call failed"
            )
            logger.error(msg)
            raise RuntimeError(
                f"{msg}\n  (failed after {_MAX_RETRIES} retries)"
            ) from last_err

        accumulated: list[str] = []
        usage: Any = None
        try:
            for event in stream:
                delta_text, event_usage = self._extract_stream_delta(event)
                if delta_text:
                    accumulated.append(delta_text)
                    on_text(delta_text)
                if event_usage is not None:
                    usage = event_usage
        except Exception as exc:
            # Mid-stream failure: do not retry. Surface a clear error.
            logger.error("Stream interrupted: %s", exc)
            raise RuntimeError(
                self._friendly_error(exc, config_desc)
            ) from exc

        full_text = "".join(accumulated)
        return _StreamedRaw(text=full_text, usage=usage)

    def _extract_stream_delta(self, event: Any) -> tuple[str, Any]:
        """Return ``(text_delta, usage_or_none)`` for one stream event.

        Handles both the Chat Completions streaming shape (``choices[0].delta``
        plus a trailing usage chunk) and the Responses streaming shape
        (``response.output_text.delta`` / ``response.completed``).
        """
        usage = getattr(event, "usage", None)

        # Responses API streaming: text deltas arrive as
        # ``ResponseOutputTextDeltaEvent`` with a ``.delta`` attribute, and the
        # ``response.completed`` event carries final usage.
        delta = getattr(event, "delta", None)
        if isinstance(delta, str):
            return delta, usage
        # Some Responses events nest text under .text (ResponseTextDeltaEvent).
        text_attr = getattr(event, "text", None)
        if isinstance(text_attr, str):
            return text_attr, usage

        # Chat Completions streaming: choices[0].delta.content
        try:
            choices = event.choices
        except AttributeError:
            return "", usage
        if not choices:
            # Trailing usage-only chunk.
            return "", usage
        choice = choices[0]
        message_delta = getattr(choice, "delta", None)
        if message_delta is None:
            return "", usage
        content = getattr(message_delta, "content", None)
        return (content or "", usage)

    def chat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Chat completion that guarantees a structured Pydantic output via instructor."""
        is_responses_mode = self._mode in (
            instructor.Mode.RESPONSES_TOOLS,
            instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        )
        if is_responses_mode:
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "input": messages,
                "response_model": response_model,
                "temperature": self.config.temperature,
                "max_output_tokens": self._clamp_max_tokens(None),
            }
            thinking = self._thinking_kwargs()
            if thinking:
                kwargs.update(thinking)
                extra = thinking.get("extra_body", {})
                if "reasoning_effort" in extra or extra.get("thinking"):
                    kwargs.pop("temperature", None)
            response, raw_completion = (
                self._instructor.responses.create_with_completion(**kwargs)
            )
        else:
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
            response, raw_completion = (
                self._instructor.chat.completions.create_with_completion(**kwargs)
            )

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
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        resp = self.chat(messages)
        return resp.output_text

    def _describe_config(self) -> str:
        """Return a short human-readable description of the current LLM config."""
        masked_key = (
            (self.config.api_key[:8] + "...")
            if len(self.config.api_key) > 8
            else "<not set>"
        )
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
                f"  → Regenerate your API key at the provider's console and update ~/.config/ppagent/settings.toml."
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
                f"  → Verify the model name in ~/.config/ppagent/settings.toml against the provider's "
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

    def _call_with_retry(self, kwargs: dict[str, Any]) -> Any:
        """Call the LLM API with exponential backoff on transient errors.

        Non-transient errors (401, 403, 404, 400) are raised immediately with
        an actionable message so the user knows exactly what to fix.
        """
        config_desc = self._describe_config()
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                if self.use_responses:
                    resp = self._client.responses.create(**kwargs)
                else:
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
                logger.warning(
                    "LLM API error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
        # All retries exhausted on a transient error.
        msg = (
            self._friendly_error(last_err, config_desc)
            if last_err
            else "LLM API call failed"
        )
        logger.error(msg)
        raise RuntimeError(
            f"{msg}\n  (failed after {_MAX_RETRIES} retries)"
        ) from last_err

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
