---
name: llm-provider-docs
description: When writing code that uses LLM-specific features (e.g. tool calling, structured output, web search, streaming, function calling, vision), consult the provider's official documentation to use native features correctly. Lists official doc sites for OpenAI, Gemini, Anthropic, ZAI, Moonshot, DeepSeek, Qwen, MiniMax, MiMo, Doubao, StepFun, Mistral, and Grok.
---

# LLM Provider Documentation Reference

## Rule

When writing code that touches a provider-specific LLM feature (tool/function calling, structured output, web search, streaming, vision, caching, batch API, etc.), **always fetch the provider's official documentation first** using `WebFetch`. Do not rely on memory — APIs evolve rapidly.

## Provider Documentation Sites

### Global Providers

| Provider | Docs URL |
|----------|----------|
| OpenAI | https://platform.openai.com/docs |
| Gemini (Google) | https://ai.google.dev/docs |
| Anthropic | https://docs.anthropic.com |
| Mistral | https://docs.mistral.ai |
| Grok (xAI) | https://docs.x.ai |

### Chinese Providers (International / Chinese docs)

| Provider | International Docs | Chinese Docs |
|----------|-------------------|--------------|
| ZAI (ZhipuAI) | https://docs.z.ai | https://open.bigmodel.cn/dev/api |
| Moonshot (Kimi) | https://platform.kimi.ai/docs/overview | https://platform.kimi.com/docs/guide/start-using-kimi-api |
| DeepSeek | https://api-docs.deepseek.com | — |
| Qwen (Alibaba) | https://www.alibabacloud.com/help/en/model-studio/ | https://help.aliyun.com/zh/model-studio/ |
| MiniMax | https://platform.minimax.io/docs | https://platform.minimaxi.com/docs |
| MiMo (Xiaomi) | https://platform.xiaomimimo.com/docs/en-US | — |
| Doubao (ByteDance) | — | https://www.volcengine.com/docs/82379 |
| StepFun | https://platform.stepfun.ai/docs/en | https://platform.stepfun.com/docs/zh |

## Workflow

1. **Identify the provider** from the task (e.g. user says "use OpenAI tool calling" or config specifies `deepseek`).
2. **Locate the relevant docs page** — append the feature path to the base URL (e.g. `https://platform.openai.com/docs/guides/function-calling`).
3. **Fetch the page** with `WebFetch` and extract the current API shape: request/response schema, required fields, streaming behavior, error codes.
4. **Implement using the native API** — do not invent parameters or reuse patterns from other providers.
5. **Prefer OpenAI-compatible SDKs** only when the provider officially supports them; otherwise use the provider's own SDK.

## Common Feature Doc Paths

These are starting points — always verify by fetching the actual page.

| Feature | OpenAI | Anthropic | Gemini |
|---------|--------|-----------|--------|
| Tool/Function calling | `/docs/guides/function-calling` | `/en/docs/build-with-claude/tool-use` | `/docs/function-calling` |
| Structured output | `/docs/guides/structured-outputs` | `/en/docs/build-with-claude/structured-output` | — |
| Vision | `/docs/guides/vision` | `/en/docs/build-with-claude/vision` | `/docs/vision` |
| Streaming | `/docs/guides/streaming` | `/en/docs/build-with-claude/streaming` | `/docs/streaming` |

For other providers, search their docs site for the equivalent feature page.

## Notes

- Most Chinese providers expose an **OpenAI-compatible endpoint**; however, provider-specific features (e.g. native web search, caching) often require their own SDK or custom headers. Always check.
- When the user hasn't specified a provider, default to the one configured in the project's settings (e.g. `config/settings.toml`).
