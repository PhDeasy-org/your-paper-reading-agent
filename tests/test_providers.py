"""Tests for the central provider registry — the single source of truth.

Adding a new vendor is a one-place edit in ``ppagent.providers``. These tests
are the regression surface that catches drift if a vendor's endpoint,
default model, detection pattern, or thinking payload is misconfigured.
"""

from __future__ import annotations

import pytest

from ppagent.providers import (
    CUSTOM_KEY,
    PROVIDERS,
    REASONING_MODEL_HINTS,
    ProviderSpec,
    detect_provider,
    get_provider,
    is_reasoning_model,
    thinking_extra_body_for,
)


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


class TestRegistryIntegrity:
    def test_keys_are_unique(self) -> None:
        keys = [p.key for p in PROVIDERS]
        assert len(keys) == len(set(keys)), f"duplicate provider keys: {keys}"

    def test_each_provider_detects_from_its_own_base_url(self) -> None:
        """A provider's own default base_url must detect back to itself.

        ``custom`` is exempt: it has no base_url and always detects to itself.
        """
        for spec in PROVIDERS:
            assert detect_provider(spec.base_url) == spec.key, (
                f"{spec.key!r} base_url {spec.base_url!r} does not detect to itself"
            )

    def test_custom_is_last_and_present(self) -> None:
        assert PROVIDERS[-1].key == CUSTOM_KEY

    def test_get_provider_round_trip(self) -> None:
        for spec in PROVIDERS:
            assert get_provider(spec.key) is spec
        assert get_provider("does-not-exist") is None


# ---------------------------------------------------------------------------
# detect_provider
# ---------------------------------------------------------------------------


class TestDetectProvider:
    def test_none_and_empty_fall_back_to_custom(self) -> None:
        assert detect_provider(None) == CUSTOM_KEY
        assert detect_provider("") == CUSTOM_KEY

    def test_unknown_url_falls_back_to_custom(self) -> None:
        assert detect_provider("https://my-internal-proxy.corp/v1") == CUSTOM_KEY

    def test_case_insensitive(self) -> None:
        assert detect_provider("HTTPS://API.OPENAI.COM/V1") == "openai"
        assert detect_provider("https://Api.DeepSeek.Com") == "deepseek"

    @pytest.mark.parametrize(
        "url,key",
        [
            ("https://api.openai.com/v1", "openai"),
            ("https://api.deepseek.com", "deepseek"),
            ("https://api.deepseek.com/v1", "deepseek"),
            ("https://api.mistral.ai/v1", "mistral"),
            ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini"),
            ("https://api.anthropic.com/v1", "anthropic"),
            ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen"),
            # kimi_cn patterns (moonshot.cn, kimi.ai) are checked before kimi_ai.
            ("https://api.moonshot.cn/v1", "kimi_cn"),
            ("https://kimi.ai/v1", "kimi_cn"),
            ("https://api.moonshot.ai/v1", "kimi_ai"),
            ("https://open.bigmodel.cn/api/paas/v4", "glm"),
            ("https://api.x.ai/v1", "grok"),
            ("https://api.stepfun.ai/v1", "stepfun"),
            ("https://api.minimax.io/v1", "minimax"),
            ("https://api.xiaomimimo.com/v1", "mimo"),
            ("https://ark.cn-beijing.volces.com/api/v3", "doubao"),
            ("https://api.hunyuan.cloud.tencent.com", "tencent"),
        ],
    )
    def test_known_urls(self, url: str, key: str) -> None:
        assert detect_provider(url) == key


# ---------------------------------------------------------------------------
# thinking_extra_body_for
# ---------------------------------------------------------------------------


class TestThinkingExtraBody:
    def test_reasoning_effort_medium_vendors(self) -> None:
        expected = {"reasoning_effort": "medium"}
        for url in [
            "https://api.openai.com/v1",  # openai
            "https://generativelanguage.googleapis.com/v1beta/openai",  # gemini
            "https://api.x.ai/v1",  # grok
            "https://api.stepfun.ai/v1",  # stepfun
        ]:
            assert thinking_extra_body_for(url) == expected, url

    def test_reasoning_effort_high_for_mistral(self) -> None:
        assert thinking_extra_body_for("https://api.mistral.ai/v1") == {
            "reasoning_effort": "high"
        }

    def test_thinking_enabled_vendors(self) -> None:
        expected = {"thinking": {"type": "enabled"}}
        for url in [
            "https://api.deepseek.com",  # deepseek
            "https://api.anthropic.com/v1",  # anthropic
            "https://api.moonshot.ai/v1",  # kimi_ai
            "https://api.moonshot.cn/v1",  # kimi_cn
            "https://api.minimax.io/v1",  # minimax
            "https://api.xiaomimimo.com/v1",  # mimo
            "https://ark.cn-beijing.volces.com/api/v3",  # doubao
            "https://open.bigmodel.cn/api/paas/v4",  # glm
        ]:
            assert thinking_extra_body_for(url) == expected, url

    def test_qwen_uses_enable_thinking_flag(self) -> None:
        assert thinking_extra_body_for(
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ) == {"enable_thinking": True}

    def test_tencent_has_no_thinking_param(self) -> None:
        # Hunyuan exposes no documented thinking parameter.
        assert thinking_extra_body_for("https://api.hunyuan.cloud.tencent.com") is None

    def test_custom_returns_none(self) -> None:
        assert thinking_extra_body_for(None) is None
        assert thinking_extra_body_for("https://unknown.example/v1") is None


# ---------------------------------------------------------------------------
# is_reasoning_model
# ---------------------------------------------------------------------------


class TestIsReasoningModel:
    @pytest.mark.parametrize(
        "model",
        [
            "deepseek-reasoner",
            "DeepSeek-R1",
            "qwq-32b",
            "qwq-plus",
            "magistral-medium-latest",
            "grok-4",
            "step-3",
            "mimo-v2-preview",
            "glm-5-air",
            "glm-4.7",
            "doubao-seed-1-6",
        ],
    )
    def test_detected_as_reasoning(self, model: str) -> None:
        assert is_reasoning_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "claude-3-5-sonnet",
            "qwen-plus",
            "moonshot-v1-8k",
            "mistral-large-latest",
            "abab6.5-chat",
        ],
    )
    def test_not_reasoning(self, model: str) -> None:
        assert is_reasoning_model(model) is False

    def test_hints_are_lowercase(self) -> None:
        # Registry invariant: hints are matched against a lowercased model name,
        # so any uppercase hint would silently never match.
        assert all(h == h.lower() for h in REASONING_MODEL_HINTS)


# ---------------------------------------------------------------------------
# Spec shape — guards future ProviderSpec additions against silent mistakes.
# ---------------------------------------------------------------------------


class TestProviderSpecShape:
    def test_all_fields_populated(self) -> None:
        for spec in PROVIDERS:
            assert isinstance(spec, ProviderSpec)
            assert spec.key, f"empty key: {spec}"
            assert spec.name, f"empty name: {spec}"
            # default_model may be "" only for custom.
            if spec.key != CUSTOM_KEY:
                assert spec.default_model, f"{spec.key} has no default_model"

    def test_custom_has_no_patterns(self) -> None:
        assert get_provider(CUSTOM_KEY).url_patterns == ()
        assert get_provider(CUSTOM_KEY).base_url is None
