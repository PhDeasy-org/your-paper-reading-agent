"""Single source of truth for LLM provider knowledge.

Everything vendor-specific — endpoints, default models, how a provider is
detected from a ``base_url``, and the per-provider "thinking/reasoning"
parameters — lives here. To support a new provider, add one ``ProviderSpec``
to :data:`PROVIDERS`; both :mod:`ppagent.llm` and :mod:`ppagent.tui` pick it
up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProviderSpec:
    """Static description of one OpenAI-compatible LLM provider.

    Attributes:
        key: Canonical identifier used throughout the app (config, TUI,
            detection). ``"custom"`` is reserved for user-defined endpoints.
        name: Human-friendly name shown in the TUI.
        base_url: Default API endpoint. ``None`` for the ``"custom"`` entry
            (the user supplies their own URL).
        default_model: Model name pre-filled when a user first visits this
            provider's settings page.
        url_patterns: Substrings matched against a lowercased ``base_url`` to
            identify this provider. Order matters only relative to other specs
            whose patterns overlap — keep specific patterns first.
        thinking_extra_body: The ``extra_body`` payload sent when
            ``enable_thinking`` is on, or ``None`` if this provider has no
            known thinking parameter.
    """

    key: str
    name: str
    base_url: str | None
    default_model: str
    url_patterns: tuple[str, ...] = field(default=())
    thinking_extra_body: dict[str, Any] | None = None
    supports_responses_api: bool = False


# Two shapes of "extended thinking" parameters seen across vendors.
_THINKING_ENABLED: dict[str, Any] = {"thinking": {"type": "enabled"}}
_REASONING_MEDIUM: dict[str, Any] = {"reasoning_effort": "medium"}
_REASONING_HIGH: dict[str, Any] = {"reasoning_effort": "high"}
_QWEN_THINKING: dict[str, Any] = {"enable_thinking": True}


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        key="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        url_patterns=("openai.com",),
        thinking_extra_body=_REASONING_MEDIUM,
        supports_responses_api=True,
    ),
    ProviderSpec(
        key="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        url_patterns=("deepseek.com", "deepseek"),
        thinking_extra_body=_THINKING_ENABLED,
    ),
    ProviderSpec(
        key="mistral",
        name="Mistral",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-large-latest",
        url_patterns=("mistral.ai",),
        thinking_extra_body=_REASONING_HIGH,
    ),
    ProviderSpec(
        key="gemini",
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-2.0-flash",
        url_patterns=("googleapis.com", "google"),
        thinking_extra_body=_REASONING_MEDIUM,
    ),
    ProviderSpec(
        key="anthropic",
        name="Anthropic",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-3-5-sonnet-latest",
        url_patterns=("anthropic.com",),
        thinking_extra_body=_THINKING_ENABLED,
    ),
    ProviderSpec(
        key="qwen",
        name="Qwen (Alibaba)",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        url_patterns=("dashscope", "aliyuncs.com"),
        thinking_extra_body=_QWEN_THINKING,
    ),
    # kimi_cn must precede kimi_ai: "moonshot.cn"/"kimi.ai" are checked before
    # the broader "moonshot.ai" pattern. The patterns do not actually overlap,
    # but the ordering mirrors the original detection precedence.
    ProviderSpec(
        key="kimi_cn",
        name="Kimi (Moonshot) - China",
        base_url="https://api.moonshot.cn/v1",
        default_model="moonshot-v1-8k",
        url_patterns=("moonshot.cn", "kimi.ai"),
        thinking_extra_body=_THINKING_ENABLED,
    ),
    ProviderSpec(
        key="kimi_ai",
        name="Kimi (Moonshot) - International",
        base_url="https://api.moonshot.ai/v1",
        default_model="moonshot-v1-8k",
        url_patterns=("moonshot.ai",),
        thinking_extra_body=_THINKING_ENABLED,
    ),
    ProviderSpec(
        key="glm",
        name="GLM (Zhipu)",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-plus",
        url_patterns=("bigmodel.cn", "z.ai"),
        thinking_extra_body=_THINKING_ENABLED,
    ),
    ProviderSpec(
        key="grok",
        name="Grok (xAI)",
        base_url="https://api.x.ai/v1",
        default_model="grok-2-latest",
        url_patterns=("x.ai",),
        thinking_extra_body=_REASONING_MEDIUM,
        supports_responses_api=True,
    ),
    ProviderSpec(
        key="stepfun",
        name="StepFun",
        base_url="https://api.stepfun.ai/v1",
        default_model="step-1-8k",
        url_patterns=("stepfun",),
        thinking_extra_body=_REASONING_MEDIUM,
    ),
    ProviderSpec(
        key="minimax",
        name="MiniMax",
        base_url="https://api.minimax.io/v1",
        default_model="abab6.5-chat",
        url_patterns=("minimax",),
        thinking_extra_body=_THINKING_ENABLED,
    ),
    ProviderSpec(
        key="mimo",
        name="MiMo (Xiaomi)",
        base_url="https://api.xiaomimimo.com/v1",
        default_model="mimo-v1",
        url_patterns=("xiaomimimo.com", "mimo"),
        thinking_extra_body=_THINKING_ENABLED,
    ),
    ProviderSpec(
        key="doubao",
        name="Doubao (ByteDance)",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        default_model="doubao-pro-32k",
        url_patterns=("volces.com", "volcengine.com"),
        thinking_extra_body=_THINKING_ENABLED,
    ),
    ProviderSpec(
        key="tencent",
        name="Tencent Hunyuan",
        base_url="https://api.hunyuan.cloud.tencent.com",
        default_model="hunyuan-pro",
        url_patterns=("tencent.com", "hunyuan"),
        # Hunyuan exposes no documented thinking parameter; leave as None.
        thinking_extra_body=None,
    ),
    ProviderSpec(
        key="custom",
        name="Custom OpenAI Compatible",
        base_url=None,
        default_model="",
        url_patterns=(),
        thinking_extra_body=None,
    ),
)

# Index for O(1) lookup by key.
_PROVIDERS_BY_KEY: dict[str, ProviderSpec] = {p.key: p for p in PROVIDERS}

# Canonical key for the fallback / user-defined provider.
CUSTOM_KEY = "custom"

# Model-name substrings that indicate a reasoning/thinking model. These are
# cross-vendor heuristics: if any hint appears in the (lowercased) model name,
# structured output is forced onto the more robust MD_JSON mode.
REASONING_MODEL_HINTS: tuple[str, ...] = (
    "deepseek",
    "reasoner",
    "r1",
    "qwq",
    "qwq-plus",
    "magistral",
    "grok-4",
    "step-3",
    "mimo-v2",
    "glm-5",
    "glm-4.7",
    "doubao-seed",
)


def detect_provider(base_url: str | None) -> str:
    """Identify a provider key from a ``base_url``.

    Returns the canonical provider key, or :data:`CUSTOM_KEY` when the URL is
    empty/``None`` or matches no known provider.
    """
    if not base_url:
        return CUSTOM_KEY
    base = base_url.lower()
    for spec in PROVIDERS:
        if any(pattern in base for pattern in spec.url_patterns):
            return spec.key
    return CUSTOM_KEY


def get_provider(key: str) -> ProviderSpec | None:
    """Return the :class:`ProviderSpec` registered under ``key``, or ``None``."""
    return _PROVIDERS_BY_KEY.get(key)


def thinking_extra_body_for(base_url: str | None) -> dict[str, Any] | None:
    """Return the thinking ``extra_body`` for the provider behind ``base_url``.

    Returns ``None`` when the provider is unknown (``"custom"``) or has no
    documented thinking parameter.
    """
    spec = get_provider(detect_provider(base_url))
    return spec.thinking_extra_body if spec else None


def is_reasoning_model(model: str) -> bool:
    """Heuristic: does ``model`` look like a reasoning/thinking model?"""
    model_lower = model.lower()
    return any(hint in model_lower for hint in REASONING_MODEL_HINTS)


def supports_responses_api_for(base_url: str | None) -> bool:
    """Return whether the provider behind ``base_url`` supports the Responses API."""
    if not base_url:
        return False
    spec = get_provider(detect_provider(base_url))
    return spec.supports_responses_api if spec else False
