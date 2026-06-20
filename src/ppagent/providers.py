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
        latest_models: Stable "-latest" aliases the provider exposes which
            always route to the newest version of each model family (e.g.
            ``grok-4-latest``, ``gemini-2.5-pro-latest``). When non-empty, the
            TUI offers a "Latest Models" picker so the user can select one
            instead of typing a model name by hand. Leave empty for providers
            that have no documented "-latest" alias convention.
    """

    key: str
    name: str
    base_url: str | None
    default_model: str
    url_patterns: tuple[str, ...] = field(default=())
    thinking_extra_body: dict[str, Any] | None = None
    supports_responses_api: bool = False
    thinking_incompatible_with_tools: bool = False
    latest_models: tuple[str, ...] = field(default=())


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
        # Source: developers.openai.com/api/docs/models/all (June 2026).
        # GPT-5.5 is the current frontier; 5.4 / 5.4-mini are lower-latency tiers.
        latest_models=("gpt-5.5", "gpt-5.4", "gpt-5.4-mini"),
    ),
    ProviderSpec(
        key="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        url_patterns=("deepseek.com", "deepseek"),
        thinking_extra_body=_THINKING_ENABLED,
        # Source: api-docs.deepseek.com + Volcengine Ark model list
        # (deepseek-v4-pro-260425 / deepseek-v4-flash-260425 confirm the v4
        # pro/flash names). deepseek-chat/-reasoner were deprecated 2026-07-24.
        latest_models=("deepseek-v4-flash", "deepseek-v4-pro"),
    ),
    ProviderSpec(
        key="mistral",
        name="Mistral",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-large-latest",
        url_patterns=("mistral.ai",),
        thinking_extra_body=_REASONING_HIGH,
        # Source: docs.mistral.ai. Mistral's "-latest" aliases are moving
        # pointers to the newest Medium / Small generation.
        latest_models=("mistral-medium-latest", "mistral-small-latest"),
    ),
    ProviderSpec(
        key="gemini",
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        # gemini-2.0-flash is shut down; gemini-2.5-flash-latest is the live
        # "-latest" alias for the 2.5 Flash family on the OpenAI-compatible
        # endpoint and keeps the original "fast/lightweight" intent.
        default_model="gemini-2.5-flash-latest",
        url_patterns=("googleapis.com", "google"),
        thinking_extra_body=_REASONING_MEDIUM,
        # Source: ai.google.dev/gemini-api/docs/changelog. gemini-pro-latest
        # was RENAMED to gemini-3-pro-preview; gemini-flash-latest is still a
        # live alias (now pointing at a Gemini 3 Flash preview).
        latest_models=(
            "gemini-3-pro-preview",
            "gemini-flash-latest",
            "gemini-2.5-flash-latest",
        ),
    ),
    ProviderSpec(
        key="anthropic",
        name="Anthropic",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-3-5-sonnet-latest",
        url_patterns=("anthropic.com",),
        thinking_extra_body=_THINKING_ENABLED,
        # Source: platform.claude.com/docs/en/about-claude/models/overview.
        # Claude Opus/Sonnet 4 retire 2026-06-15; the live point releases are
        # opus-4-1, sonnet-4-6, and haiku-4-5.
        latest_models=(
            "claude-opus-4-1",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ),
    ),
    ProviderSpec(
        key="qwen",
        name="Qwen (Alibaba)",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        url_patterns=("dashscope", "aliyuncs.com"),
        thinking_extra_body=_QWEN_THINKING,
        thinking_incompatible_with_tools=True,
        # Source: help.aliyun.com/zh/model-studio/models (Model Studio model
        # list). qwen3.7-max is the flagship; 3.7-plus / 3.6-plus are the
        # high-performance tiers.
        latest_models=("qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus"),
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
        thinking_incompatible_with_tools=True,
        # Source: platform.kimi.com/docs/models. The kimi-k2 series was
        # discontinued 2026-05-25; kimi-k2.7-code (coding) and kimi-k2.6
        # (flagship) are the current models.
        latest_models=("kimi-k2.7-code", "kimi-k2.6"),
    ),
    ProviderSpec(
        key="kimi_ai",
        name="Kimi (Moonshot) - International",
        base_url="https://api.moonshot.ai/v1",
        default_model="moonshot-v1-8k",
        url_patterns=("moonshot.ai",),
        thinking_extra_body=_THINKING_ENABLED,
        thinking_incompatible_with_tools=True,
        # Same model lineup as the China endpoint (platform.kimi.ai).
        latest_models=("kimi-k2.7-code", "kimi-k2.6"),
    ),
    # zai must precede glm: its base URL also contains "z.ai", but the more
    # specific "z.ai/api/coding" pattern must win for the coding token-plan
    # endpoint. Other z.ai URLs (e.g. the regular paas/v4 endpoint) still fall
    # through to glm below. Unlike glm, this endpoint only exposes the OpenAI
    # chat completions API — not the Responses API.
    ProviderSpec(
        key="zai",
        name="Z.AI (Coding Token Plan)",
        base_url="https://api.z.ai/api/coding/paas/v4",
        default_model="glm-4.6",
        url_patterns=("z.ai/api/coding",),
        thinking_extra_body=_THINKING_ENABLED,
    ),
    ProviderSpec(
        key="glm",
        name="GLM (Zhipu)",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-plus",
        url_patterns=("bigmodel.cn", "z.ai"),
        thinking_extra_body=_THINKING_ENABLED,
        # Source: docs.bigmodel.cn/cn/guide/start/model-overview. glm-5.2 is
        # the new flagship; glm-5 the base; glm-5v-turbo the vision variant.
        # (There is no standalone glm-5-flash — flash lives in the 4.x line.)
        latest_models=("glm-5.2", "glm-5", "glm-5v-turbo"),
    ),
    ProviderSpec(
        key="grok",
        name="Grok (xAI)",
        base_url="https://api.x.ai/v1",
        default_model="grok-2-latest",
        url_patterns=("x.ai",),
        thinking_extra_body=_REASONING_MEDIUM,
        supports_responses_api=True,
        # xAI documents "<model>-latest" aliases that auto-route to the newest
        # version of each Grok family. grok-4-latest is the current flagship.
        latest_models=("grok-4-latest", "grok-3-latest", "grok-2-latest"),
    ),
    ProviderSpec(
        key="stepfun",
        name="StepFun",
        base_url="https://api.stepfun.ai/v1",
        default_model="step-1-8k",
        url_patterns=("stepfun",),
        thinking_extra_body=_REASONING_MEDIUM,
        # Source: platform.stepfun.com/docs/zh/guides/model-migration. step-3
        # is deprecated → migrate to step-3.7-flash; step-3.5-flash is the
        # open-source foundation model.
        latest_models=("step-3.7-flash", "step-3.5-flash"),
    ),
    ProviderSpec(
        key="minimax",
        name="MiniMax",
        base_url="https://api.minimax.io/v1",
        default_model="abab6.5-chat",
        url_patterns=("minimax",),
        thinking_extra_body=_THINKING_ENABLED,
        # Source: minimax.io / platform.minimaxi.com. MiniMax-M3 is the current
        # frontier coding & agentic model (1M context). Casing per the official
        # site.
        latest_models=("MiniMax-M3",),
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
        # Source: volcengine.com/docs/82379/1330310 (火山方舟 model list,
        # dated 2026.05.29). The current generation is the Seed series;
        # the classic doubao-pro/1-5-pro line is discontinued.
        latest_models=(
            "doubao-seed-2-0-pro",
            "doubao-seed-2-0-mini",
            "doubao-seed-1-6",
        ),
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


def thinking_incompatible_with_tools_for(base_url: str | None) -> bool:
    """Return whether the provider's thinking mode is incompatible with tool_choice.

    Some providers (e.g. Qwen, Kimi) reject ``tool_choice`` when thinking mode
    is enabled. For these, structured output must use MD_JSON instead of TOOLS.
    """
    spec = get_provider(detect_provider(base_url))
    return spec.thinking_incompatible_with_tools if spec else False


def supports_responses_api_for(base_url: str | None) -> bool:
    """Return whether the provider behind ``base_url`` supports the Responses API."""
    if not base_url:
        return False
    spec = get_provider(detect_provider(base_url))
    return spec.supports_responses_api if spec else False
