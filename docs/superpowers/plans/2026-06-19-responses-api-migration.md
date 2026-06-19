# Responses API Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the LLM client and agent tool-calling loop from chat completions to the new OpenAI Responses API.

**Architecture:** Update `LLMClient` to delegate to `self._client.responses.create` and use `instructor` with `RESPONSES_TOOLS_WITH_INBUILT_TOOLS`. Update `AgentWithTools._run_with_tools` to parse output items and serialize tool responses using the new `function_call` and `function_call_output` models.

**Tech Stack:** Python 3.12, openai 2.41.0, instructor, pytest

## Global Constraints
- Target python files must follow strictly formatted Python 3.12+ features.
- All modified files must pass `ruff` formatting and linting.
- The unit test suite must remain 100% green.

---

### Task 1: Migrate LLMClient and update its Error/Usage handling [COMPLETE]

**Files:**
- Modify: `src/ppagent/llm.py`
- Modify: `tests/test_llm_errors.py`

**Interfaces:**
- Consumes: None
- Produces: `LLMClient.chat` returning `openai.types.responses.Response` and `LLMClient.chat_structured` returning Pydantic model via `instructor.responses.create_with_completion`.

- [x] **Step 1: Write/Update tests to expect responses.create mocks**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Modify `src/ppagent/llm.py`**
- [x] **Step 4: Run test to verify it passes**
- [x] **Step 5: Commit changes**

---

### Task 2: Migrate Agent Calling Logic and Tool Loop [COMPLETE]

**Files:**
- Modify: `src/ppagent/agents/base.py`

**Interfaces:**
- Consumes: `LLMClient.chat` returning `openai.types.responses.Response`
- Produces: `AgentWithTools._run_with_tools` executing tools and accumulating history with new responses format.

- [x] **Step 1: Write/Update Agent tool tests or review existing tests**
- [x] **Step 2: Modify `src/ppagent/agents/base.py`**
- [x] **Step 3: Run the entire test suite**
- [x] **Step 4: Commit changes**

---

### Task 3: Restore Multi-Provider Compatibility, Unified Response Wrapper, and Type Safety

**Files:**
- Modify: `src/ppagent/providers.py`
- Modify: `src/ppagent/llm.py`
- Modify: `src/ppagent/agents/base.py`
- Modify: `tests/test_llm_errors.py`

**Interfaces:**
- Consumes: `ProviderSpec` definition
- Produces:
  - `LLMClient.use_responses: bool` indicating whether the client supports the Responses API.
  - `LLMResponse` wrapper class in `llm.py` encapsulating both `Response` and `ChatCompletion` objects.
  - Type-safe, dict-serialized message updates in `base.py`'s `_run_with_tools`.
  - Comprehensive unit testing for both Responses-supporting and standard completions-supporting clients.

- [ ] **Step 1: Update `src/ppagent/providers.py`**

  Add `supports_responses_api: bool = False` to `ProviderSpec` class:
  ```python
  @dataclass(frozen=True)
  class ProviderSpec:
      key: str
      name: str
      base_url: str | None
      default_model: str
      url_patterns: tuple[str, ...] = field(default=())
      thinking_extra_body: dict[str, Any] | None = None
      supports_responses_api: bool = False
  ```
  Set `supports_responses_api=True` for `"openai"` and `"grok"` specs in `PROVIDERS`.
  Add helper function `supports_responses_api_for(base_url: str | None) -> bool`:
  ```python
  def supports_responses_api_for(base_url: str | None) -> bool:
      """Return whether the provider behind ``base_url`` supports the Responses API."""
      if not base_url:
          return False
      spec = get_provider(detect_provider(base_url))
      return spec.supports_responses_api if spec else False
  ```

- [ ] **Step 2: Define `LLMResponse` and update `src/ppagent/llm.py`**

  Define the `LLMResponse` compatibility wrapper:
  ```python
  @dataclass
  class _CompatToolCall:
      id: str
      call_id: str
      name: str
      arguments: str
      type: str = "function_call"

  class LLMResponse:
      """Unified wrapper around OpenAI ChatCompletion or Response."""
      def __init__(self, raw: Any) -> None:
          self.raw = raw

      @property
      def output_text(self) -> str:
          if hasattr(self.raw, "output_text"):
              return self.raw.output_text
          try:
              return self.raw.choices[0].message.content or ""
          except (AttributeError, IndexError):
              return ""

      @property
      def output(self) -> list[Any]:
          if hasattr(self.raw, "output"):
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
                              type="function_call"
                          )
                      )
          except (AttributeError, IndexError):
              pass
          return items

      @property
      def usage(self) -> Any:
          return self.raw.usage
  ```

  Update `LLMClient`:
  - `self.use_responses = supports_responses_api_for(config.base_url)` in `__init__`.
  - In `_resolve_instructor_mode()`, only resolve to Responses API modes (`RESPONSES_TOOLS_WITH_INBUILT_TOOLS`) if `self.use_responses` is True.
  - In `chat()`, call `self._client.responses.create` if `self.use_responses` is True, otherwise `self._client.chat.completions.create`. Return `LLMResponse(resp)`.
  - In `_call_with_retry()`, switch dynamically:
    ```python
    if self.use_responses:
        resp = self._client.responses.create(**kwargs)
    else:
        resp = self._client.chat.completions.create(**kwargs)
    ```

- [ ] **Step 3: Update `src/ppagent/agents/base.py`**

  Update `_run_with_tools()` to serialize all model output items to dicts before appending to history, and support both formats:
  ```python
  for iteration in range(max_iterations):
      resp = self.llm.chat(messages, tools=tool_defs if tool_defs else None)

      if self.llm.use_responses:
          tool_calls = [item for item in resp.output if item.type == "function_call"]
          if not tool_calls:
              return resp.output_text

          # Serialize Pydantic output items to dicts
          for item in resp.output:
              if hasattr(item, "model_dump"):
                  messages.append(item.model_dump())
              else:
                  messages.append(item)

          for call in tool_calls:
              fn_name = call.name
              try:
                  fn_args = json.loads(call.arguments)
              except json.JSONDecodeError:
                  fn_args = {}

              handler = tool_map.get(fn_name)
              if handler:
                  result = handler(**fn_args)
              else:
                  result = f"Error: unknown tool '{fn_name}'"

              messages.append(
                  {
                      "type": "function_call_output",
                      "call_id": call.call_id,
                      "output": str(result),
                  }
              )
      else:
          try:
              choice = resp.raw.choices[0]
              message = choice.message
          except (AttributeError, IndexError):
              return resp.output_text

          if not message.tool_calls:
              return resp.output_text

          messages.append(message.model_dump())

          for call in message.tool_calls:
              fn_name = call.function.name
              try:
                  fn_args = json.loads(call.function.arguments)
              except json.JSONDecodeError:
                  fn_args = {}

              handler = tool_map.get(fn_name)
              if handler:
                  result = handler(**fn_args)
              else:
                  result = f"Error: unknown tool '{fn_name}'"

              messages.append(
                  {
                      "role": "tool",
                      "content": str(result),
                      "tool_call_id": call.id,
                  }
              )
  ```

- [ ] **Step 4: Update `tests/test_llm_errors.py`**

  Restore `client._client.chat.completions.create` mocks for Kimi provider (non-responses provider) and add specific tests targeting `responses.create` under an OpenAI provider to verify both code paths.

- [ ] **Step 5: Run tests and commit**

  Run: `pytest`
  Expected: PASS (all 180 tests pass)
  Run: `git commit -am "feat: restore multi-provider compatibility and unified LLMResponse wrapper"`
