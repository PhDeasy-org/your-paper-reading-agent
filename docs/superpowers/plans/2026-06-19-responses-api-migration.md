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

### Task 1: Migrate LLMClient and update its Error/Usage handling

**Files:**
- Modify: `src/ppagent/llm.py`
- Modify: `tests/test_llm_errors.py`

**Interfaces:**
- Consumes: None
- Produces: `LLMClient.chat` returning `openai.types.responses.Response` and `LLMClient.chat_structured` returning Pydantic model via `instructor.responses.create_with_completion`.

- [ ] **Step 1: Write/Update tests to expect responses.create mocks**

  Modify `tests/test_llm_errors.py` by changing all calls to `client._client.chat.completions.create` to `client._client.responses.create`. Also update mock response assertions (e.g., mock choices).
  For example, update `test_auth_error_raises_immediately_without_retry`:
  ```python
  def test_auth_error_raises_immediately_without_retry(self) -> None:
      client = _make_client()
      exc = _make_status_error(401, message="Invalid Authentication")
      auth_exc = openai.AuthenticationError(
          message="Invalid Authentication",
          response=exc.response,
          body=exc.body,
      )
      client._client = MagicMock()
      client._client.responses.create.side_effect = auth_exc

      with pytest.raises(RuntimeError, match="Authentication failed"):
          client._call_with_retry({"model": "m", "input": []})

      assert client._client.responses.create.call_count == 1
  ```
  Ensure all other test methods in `tests/test_llm_errors.py` are updated to mock `client._client.responses.create` instead of `client._client.chat.completions.create`.

- [ ] **Step 2: Run test to verify it fails**

  Run: `pytest tests/test_llm_errors.py -v`
  Expected: FAIL with `AttributeError` or similar because `LLMClient` is still calling `chat.completions.create`.

- [ ] **Step 3: Modify `src/ppagent/llm.py`**

  Apply the following changes to `src/ppagent/llm.py`:
  - Map `instructor.Mode.TOOLS` to `instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS`.
  - Update `_record_usage()`:
    ```python
    def _record_usage(self, usage: Any | None) -> None:
        if not usage:
            return
        local_usage = self._get_local_usage()
        prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
        total = getattr(usage, "total_tokens", 0) or 0
        if total == 0:
            total = prompt + completion
        local_usage["prompt_tokens"] += prompt
        local_usage["completion_tokens"] += completion
        local_usage["total_tokens"] += total
    ```
  - Update `chat()` to call `self._client.responses.create`:
    ```python
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> openai.types.responses.Response:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "input": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_output_tokens": self._clamp_max_tokens(max_tokens),
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
    ```
  - Update `chat_structured()` to use responses if in response mode:
    ```python
    def chat_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> BaseModel:
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
            response, raw_completion = self._instructor.responses.create_with_completion(**kwargs)
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
            response, raw_completion = self._instructor.chat.completions.create_with_completion(**kwargs)

        self._record_usage(raw_completion.usage)
        return response
    ```
  - Update `chat_vision()`:
    ```python
    def chat_vision(
        self,
        system: str,
        user_text: str,
        images: list[Path],
    ) -> str:
        # Construct and call self.chat
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
        return resp.output_text
    ```
  - Update `_call_with_retry()`:
    ```python
    def _call_with_retry(self, kwargs: dict[str, Any]) -> openai.types.responses.Response:
        config_desc = self._describe_config()
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.responses.create(**kwargs)
                return resp
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
                logger.warning("LLM API error (attempt %d/%d): %s — retrying in %ds",
                               attempt + 1, _MAX_RETRIES, exc, wait)
                time.sleep(wait)
        msg = self._friendly_error(last_err, config_desc) if last_err else "LLM API call failed"
        logger.error(msg)
        raise RuntimeError(f"{msg}\n  (failed after {_MAX_RETRIES} retries)") from last_err
    ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `pytest tests/test_llm_errors.py -v`
  Expected: PASS

- [ ] **Step 5: Commit changes**

  Run: `git commit -am "feat: migrate LLMClient to Responses API"`

---

### Task 2: Migrate Agent Calling Logic and Tool Loop

**Files:**
- Modify: `src/ppagent/agents/base.py`

**Interfaces:**
- Consumes: `LLMClient.chat` returning `openai.types.responses.Response`
- Produces: `AgentWithTools._run_with_tools` executing tools and accumulating history with new responses format.

- [ ] **Step 1: Write/Update Agent tool tests or review existing tests**

  Check if we have tool calling tests. `tests/test_xai_tool.py` already uses responses mock.
  Let's review if there are other tests calling tool loop. We will run pytest on tests to verify what fails first when base.py changes.

- [ ] **Step 2: Modify `src/ppagent/agents/base.py`**

  In `src/ppagent/agents/base.py`:
  - Update `_call_llm()`:
    ```python
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel] | None = None,
    ) -> Any:
        messages = LLMClient.build_messages(system_prompt, user_prompt)
        if response_model:
            return self.llm.chat_structured(messages, response_model)
        resp = self.llm.chat(messages)
        return resp.output_text
    ```
  - Update `_run_with_tools()`:
    ```python
    def _run_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        max_iterations: int = 5,
    ) -> str:
        all_tool_defs: list[ToolDef] = [
            *self.tools,
            *(at.definition for at in self.agent_tools),
        ]

        tool_map: dict[str, Any] = {}
        for t in all_tool_defs:
            handler = getattr(self, f"_tool_{t.name}", None)
            if handler:
                tool_map[t.name] = handler

        tool_defs = [t.to_openai_tool() for t in all_tool_defs]

        for iteration in range(max_iterations):
            resp = self.llm.chat(messages, tools=tool_defs if tool_defs else None)

            tool_calls = [item for item in resp.output if item.type == "function_call"]
            if not tool_calls:
                return resp.output_text

            messages.extend(resp.output)

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

            logger.debug(
                "Tool iteration %d/%d: called %d tools",
                iteration + 1,
                max_iterations,
                len(tool_calls),
            )

        return resp.output_text
    ```

- [ ] **Step 3: Run the entire test suite**

  Run: `pytest`
  Expected: PASS (all 178 tests pass)

- [ ] **Step 4: Commit changes**

  Run: `git commit -am "feat: update agent tool-calling loop for Responses API compatibility"`
