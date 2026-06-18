"""Tests for LLM client error handling — friendly messages and retry behaviour."""

from __future__ import annotations

import os
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
        api_key="sk-MUavNbciAN680Dd2KnRVMjTBPW09k6qr0L9fnklHeeR5O7rP",
        model="kimi-k2.7-code",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return LLMClient(cfg)


def _make_status_error(
    status_code: int,
    *,
    message: str = "boom",
    body: dict | None = None,
) -> openai.APIStatusError:
    """Build the correct OpenAI SDK status-error subclass for *status_code*."""
    response = httpx.Response(status_code=status_code, request=httpx.Request("POST", "https://x"))
    err_body: dict = body or {"error": {"message": message, "type": "invalid_request_error"}}
    return openai.APIStatusError(message=message, response=response, body=err_body)


# ---------------------------------------------------------------------------
# _describe_config
# ---------------------------------------------------------------------------

class TestDescribeConfig:
    def test_masks_long_api_key(self) -> None:
        client = _make_client(api_key="sk-1234567890abcdef")
        desc = client._describe_config()
        assert "sk-12345..." in desc
        assert "sk-1234567890abcdef" not in desc

    def test_shows_not_set_for_short_key(self) -> None:
        client = _make_client(api_key="short")
        desc = client._describe_config()
        assert "<not set>" in desc

    def test_includes_model_and_url(self) -> None:
        client = _make_client(model="gpt-4o", base_url="https://api.openai.com/v1")
        desc = client._describe_config()
        assert "gpt-4o" in desc
        assert "openai.com" in desc
        assert "provider='openai'" in desc


# ---------------------------------------------------------------------------
# _friendly_error
# ---------------------------------------------------------------------------

class TestFriendlyError:
    def test_authentication_error(self) -> None:
        exc = _make_status_error(401, message="Invalid Authentication")
        # Cast to the correct subclass
        auth_exc = openai.AuthenticationError(
            message="Invalid Authentication",
            response=exc.response,
            body=exc.body,
        )
        msg = LLMClient._friendly_error(auth_exc, "model='kimi-k2.7-code'")
        assert "Authentication failed" in msg
        assert "401" in msg
        assert "settings.toml" in msg

    def test_not_found_error(self) -> None:
        exc = _make_status_error(404, message="Model not found")
        nf_exc = openai.NotFoundError(
            message="Model not found",
            response=exc.response,
            body=exc.body,
        )
        msg = LLMClient._friendly_error(nf_exc, "model='fake-model'")
        assert "Model not found" in msg
        assert "404" in msg

    def test_bad_request_error(self) -> None:
        exc = _make_status_error(
            400,
            message="Bad request",
            body={"error": {"message": "thinking not supported"}},
        )
        br_exc = openai.BadRequestError(
            message="Bad request",
            response=exc.response,
            body=exc.body,
        )
        msg = LLMClient._friendly_error(br_exc, "model='x'")
        assert "Bad request" in msg
        assert "400" in msg
        assert "thinking not supported" in msg

    def test_rate_limit_error(self) -> None:
        exc = _make_status_error(429, message="Too many requests")
        rl_exc = openai.RateLimitError(
            message="Too many requests",
            response=exc.response,
            body=exc.body,
        )
        msg = LLMClient._friendly_error(rl_exc, "model='x'")
        assert "Rate limit" in msg
        assert "429" in msg


# ---------------------------------------------------------------------------
# _call_with_retry — retry vs. no-retry behaviour
# ---------------------------------------------------------------------------

class TestCallWithRetry:
    def test_auth_error_raises_immediately_without_retry(self) -> None:
        client = _make_client()
        exc = _make_status_error(401, message="Invalid Authentication")
        auth_exc = openai.AuthenticationError(
            message="Invalid Authentication",
            response=exc.response,
            body=exc.body,
        )
        client._client = MagicMock()
        client._client.chat.completions.create.side_effect = auth_exc

        with pytest.raises(RuntimeError, match="Authentication failed"):
            client._call_with_retry({"model": "m", "messages": []})

        # Should have been called exactly once — no retries for auth errors.
        assert client._client.chat.completions.create.call_count == 1

    def test_not_found_error_raises_immediately(self) -> None:
        client = _make_client()
        exc = _make_status_error(404, message="Model not found")
        nf_exc = openai.NotFoundError(
            message="Model not found",
            response=exc.response,
            body=exc.body,
        )
        client._client = MagicMock()
        client._client.chat.completions.create.side_effect = nf_exc

        with pytest.raises(RuntimeError, match="Model not found"):
            client._call_with_retry({"model": "m", "messages": []})
        assert client._client.chat.completions.create.call_count == 1

    def test_bad_request_raises_immediately(self) -> None:
        client = _make_client()
        exc = _make_status_error(400, message="bad params")
        br_exc = openai.BadRequestError(
            message="bad params",
            response=exc.response,
            body=exc.body,
        )
        client._client = MagicMock()
        client._client.chat.completions.create.side_effect = br_exc

        with pytest.raises(RuntimeError, match="Bad request"):
            client._call_with_retry({"model": "m", "messages": []})
        assert client._client.chat.completions.create.call_count == 1

    def test_transient_server_error_retries_then_raises(self) -> None:
        client = _make_client()
        exc = _make_status_error(500, message="Internal error")
        ise_exc = openai.InternalServerError(
            message="Internal error",
            response=exc.response,
            body=exc.body,
        )
        client._client = MagicMock()
        client._client.chat.completions.create.side_effect = ise_exc

        # Patch sleep so we don't wait
        with patch("ppagent.llm.time.sleep"):
            with pytest.raises(RuntimeError, match="server error"):
                client._call_with_retry({"model": "m", "messages": []})

        # Should have retried _MAX_RETRIES times (3)
        assert client._client.chat.completions.create.call_count == 3

    def test_transient_connection_error_retries_then_succeeds(self) -> None:
        client = _make_client()
        conn_exc = openai.APIConnectionError(request=httpx.Request("POST", "https://x"))
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]

        client._client = MagicMock()
        client._client.chat.completions.create.side_effect = [conn_exc, mock_resp]

        with patch("ppagent.llm.time.sleep"):
            result = client._call_with_retry({"model": "m", "messages": []})

        assert result is mock_resp
        assert client._client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Provider detection (moonshot.ai) — sourced from the central registry.
# ---------------------------------------------------------------------------

class TestProviderDetection:
    def test_moonshot_ai_detected_as_kimi_ai(self) -> None:
        # _describe_config embeds the detected provider key.
        client = _make_client(base_url="https://api.moonshot.ai/v1")
        assert "provider='kimi_ai'" in client._describe_config()

    def test_moonshot_cn_detected_as_kimi_cn(self) -> None:
        client = _make_client(base_url="https://api.moonshot.cn/v1")
        assert "provider='kimi_cn'" in client._describe_config()

    def test_unknown_provider_returns_custom(self) -> None:
        client = _make_client(base_url="https://custom.example.com/v1")
        assert "provider='custom'" in client._describe_config()
