# Migration of LLM Client to Responses API

This design document outlines the transition of the `ppagent` LLM client and agent tool-calling execution framework from the legacy OpenAI `chat.completions` API to the newer `responses` API.

## Background & Objectives

The Responses API is a streamlined, more flexible interface in the OpenAI Python SDK (and compatible backends) for handling complex agent interactions, tool calls, and structured outputs.
Our goal is to refactor `LLMClient` to invoke `responses.create` instead of `chat.completions.create` and update the agent tool loops to handle the new response output format natively.

## Proposed Changes

### 1. LLM Client Wrapper (`src/ppagent/llm.py`)

- **Instructor Mode Resolution:** Standardize default tool calling on `instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS`.
- **Token Usage Tracking:** Add support in `_record_usage()` for mapping `input_tokens` $\rightarrow$ `prompt_tokens` and `output_tokens` $\rightarrow$ `completion_tokens` from `ResponseUsage`.
- **API Call Interfaces:**
  - Update `chat()` to call `self._client.responses.create(...)`, renaming `messages` to `input` and `max_tokens` to `max_output_tokens`.
  - Update `chat_structured()` to conditionally check if the instructor mode is a responses mode, and call `self._instructor.responses.create_with_completion(...)` accordingly.
- **Multimodal (`chat_vision`):** Access response text via the `.output_text` property instead of `.choices[0]`.

### 2. Base Agent Class (`src/ppagent/agents/base.py`)

- **Call Convenience wrapper (`_call_llm`):** Access text response via `.output_text`.
- **Tool execution loop (`_run_with_tools`):**
  - Check for tool execution requests in `resp.output` matching `item.type == "function_call"`.
  - Extend the conversation history with `resp.output`.
  - Append tool execution results using the new structured format:
    ```python
    {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": str(result),
    }
    ```

## Verification & Testing

- Update mocks in `tests/test_llm_errors.py` to intercept `client.responses.create` instead of `client.chat.completions.create`.
- Ensure all 178 unit tests pass successfully.
