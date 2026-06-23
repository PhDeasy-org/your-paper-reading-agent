"""Tests for the streaming chat path added to ``LLMClient.chat``.

Covers:
- text deltas are passed to the ``on_text`` callback in order;
- the returned :class:`LLMResponse` reconstructs ``output_text`` and ``usage``
  from the stream;
- tool turns are *not* streamed (they fall back to the blocking retry path);
- pre-stream transient errors are retried, but mid-stream errors surface.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from ppagent.config import LLMConfig
from ppagent.llm import LLMClient

# Ensure the OpenAI SDK doesn't complain about missing env credentials.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-unit-tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(**overrides: object) -> LLMClient:
    cfg = LLMConfig(
        base_url="https://api.moonshot.ai/v1",
        api_key="sk-test-streaming-key",
        model="kimi-k2.7-code",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return LLMClient(cfg)


def _chat_delta(content: str | None) -> SimpleNamespace:
    """A Chat Completions streaming chunk with a text delta."""
    delta = SimpleNamespace(content=content)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice], usage=None)


def _chat_usage_chunk(usage: object) -> SimpleNamespace:
    """The trailing Chat Completions chunk that carries usage (no choices)."""
    return SimpleNamespace(choices=[], usage=usage)


# ---------------------------------------------------------------------------
# chat(stream=True) — Chat Completions path
# ---------------------------------------------------------------------------


class TestChatStreamChatCompletions:
    def test_deltas_passed_to_callback_and_reassembled(self) -> None:
        client = _make_client()
        chunks = [
            _chat_delta("Hello"),
            _chat_delta(", "),
            _chat_delta("world!"),
        ]
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = iter(chunks)

        received: list[str] = []
        resp = client.chat(
            [{"role": "user", "content": "hi"}],
            stream=True,
            on_text=received.append,
        )

        assert received == ["Hello", ", ", "world!"]
        assert resp.output_text == "Hello, world!"

    def test_usage_captured_from_trailing_chunk(self) -> None:
        client = _make_client()
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        chunks = [_chat_delta("hi"), _chat_usage_chunk(usage)]
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = iter(chunks)

        client.reset_usage()
        client.chat(
            [{"role": "user", "content": "hi"}],
            stream=True,
            on_text=lambda _d: None,
        )

        assert client.get_usage()["total_tokens"] == 8

    def test_reconstructed_response_has_chat_completions_shape(self) -> None:
        client = _make_client()
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = iter([_chat_delta("abc")])

        resp = client.chat(
            [{"role": "user", "content": "hi"}],
            stream=True,
            on_text=lambda _d: None,
        )

        # The _run_with_tools loop reads .output (tool calls) — must be empty.
        assert resp.output == []
        # And .output_text via the fallback path reading choices[0].message.
        assert resp.raw.choices[0].message.content == "abc"
        assert resp.raw.choices[0].message.tool_calls is None

    def test_stream_options_requested_for_chat_completions(self) -> None:
        client = _make_client()
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = iter([_chat_delta("x")])

        client.chat(
            [{"role": "user", "content": "hi"}],
            stream=True,
            on_text=lambda _d: None,
        )

        _, kwargs = client._client.chat.completions.create.call_args
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# chat(stream=True) — tool turns are NOT streamed
# ---------------------------------------------------------------------------


class TestChatStreamToolFallback:
    def test_tools_bypass_streaming(self) -> None:
        """When tools are present, the blocking retry path is used even if
        ``stream`` and ``on_text`` are set."""
        client = _make_client()
        client._client = MagicMock()
        blocking_resp = MagicMock()
        client._client.chat.completions.create.return_value = blocking_resp

        on_text = MagicMock()
        resp = client.chat(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
            stream=True,
            on_text=on_text,
        )

        # Blocking path returns the raw object unchanged.
        assert resp.raw is blocking_resp
        # No streaming happened.
        on_text.assert_not_called()
        # And stream=True was NOT passed to the SDK.
        _, kwargs = client._client.chat.completions.create.call_args
        assert "stream" not in kwargs or kwargs["stream"] is False

    def test_stream_without_callback_does_not_stream(self) -> None:
        """``stream=True`` without an ``on_text`` callback is a no-op: the
        blocking path is used so callers that just pass the flag stay safe."""
        client = _make_client()
        client._client = MagicMock()
        blocking_resp = MagicMock()
        client._client.chat.completions.create.return_value = blocking_resp

        resp = client.chat(
            [{"role": "user", "content": "hi"}],
            stream=True,
            on_text=None,
        )

        assert resp.raw is blocking_resp


# ---------------------------------------------------------------------------
# chat(stream=True) — error handling
# ---------------------------------------------------------------------------


class TestChatStreamErrors:
    def test_pre_stream_auth_error_raises_immediately(self) -> None:
        client = _make_client()
        response = httpx.Response(
            status_code=401, request=httpx.Request("POST", "https://x")
        )
        auth_exc = openai.AuthenticationError(
            message="Invalid Authentication",
            response=response,
            body={"error": {"message": "Invalid Authentication"}},
        )
        client._client = MagicMock()
        client._client.chat.completions.create.side_effect = auth_exc

        with pytest.raises(RuntimeError, match="Authentication failed"):
            client.chat(
                [{"role": "user", "content": "hi"}],
                stream=True,
                on_text=lambda _d: None,
            )
        assert client._client.chat.completions.create.call_count == 1

    def test_pre_stream_transient_error_retries(self) -> None:
        client = _make_client()
        response = httpx.Response(
            status_code=500, request=httpx.Request("POST", "https://x")
        )
        ise_exc = openai.InternalServerError(
            message="Internal error",
            response=response,
            body={"error": {"message": "Internal error"}},
        )
        client._client = MagicMock()
        # Fail twice, then succeed on the third (streamed) attempt.
        client._client.chat.completions.create.side_effect = [
            ise_exc,
            ise_exc,
            iter([_chat_delta("ok")]),
        ]

        with patch("ppagent.llm.time.sleep"):
            resp = client.chat(
                [{"role": "user", "content": "hi"}],
                stream=True,
                on_text=lambda _d: None,
            )

        assert resp.output_text == "ok"
        assert client._client.chat.completions.create.call_count == 3

    def test_mid_stream_error_surfaces_as_runtime_error(self) -> None:
        client = _make_client()

        def _broken_stream():
            yield _chat_delta("partial")
            raise RuntimeError("connection reset mid-stream")

        client._client = MagicMock()
        client._client.chat.completions.create.return_value = _broken_stream()

        # No retry on mid-stream failures: sleep must never be called.
        with patch("ppagent.llm.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="connection reset"):
                client.chat(
                    [{"role": "user", "content": "hi"}],
                    stream=True,
                    on_text=lambda _d: None,
                )
        mock_sleep.assert_not_called()
