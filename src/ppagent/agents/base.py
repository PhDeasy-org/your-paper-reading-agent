"""Base classes for ppagent agents."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ppagent.config import AppConfig
from ppagent.llm import LLMClient
from ppagent.models import AgentResult

if TYPE_CHECKING:
    from ppagent.agents.tools import AgentTool

logger = logging.getLogger(__name__)


class ToolDef(BaseModel):
    """Definition of a tool that an agent can use."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema

    def to_openai_tool(self) -> dict[str, Any]:
        """Chat Completions API tool format (nested under ``function``)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_responses_tool(self) -> dict[str, Any]:
        """Responses API tool format (flat, no ``function`` wrapper)."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class AgentBase(ABC):
    """Base class for all ppagent agents."""

    name: str = "base"
    description: str = ""

    def __init__(self, llm: LLMClient, config: AppConfig) -> None:
        self.llm = llm
        self.config = config

    @abstractmethod
    def run(self, **kwargs: Any) -> AgentResult:
        """Execute the agent's task and return a structured result."""
        ...

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel] | None = None,
    ) -> Any:
        """Convenience wrapper that builds messages and calls the LLM."""
        messages = LLMClient.build_messages(system_prompt, user_prompt)
        if response_model:
            return self.llm.chat_structured(messages, response_model)
        resp = self.llm.chat(messages)
        return resp.output_text


class AgentWithTools(AgentBase):
    """Agent that can use tools via LLM function calling.

    Subclasses may populate:

    * ``self.tools`` — bare :class:`ToolDef` objects whose handlers are
      implemented as ``_tool_<name>`` methods on the subclass.
    * ``self.agent_tools`` — :class:`~ppagent.agents.tools.AgentTool` objects
      from the shared :mod:`ppagent.agents.tools` catalogue.  Their handlers
      are bound automatically; no ``_tool_<name>`` method is needed.
    """

    def __init__(self, llm: LLMClient, config: AppConfig) -> None:
        super().__init__(llm, config)
        self.tools: list[ToolDef] = []
        self.agent_tools: list[AgentTool] = []

    def _run_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        max_iterations: int = 5,
        on_text: Callable[[str], None] | None = None,
    ) -> str:
        """Agentic loop: call LLM → if tool_calls, execute → feed back → repeat.

        Shared :class:`~ppagent.agents.tools.AgentTool` entries listed in
        ``self.agent_tools`` are bound and merged with ``self.tools`` before
        the loop starts.

        ``on_text``, when provided, streams text deltas to the callback for the
        final no-tool turn (the prose the loop ultimately returns). Tool turns
        are never streamed.
        """
        # Bind shared AgentTool handlers onto self before building tool_map
        for at in self.agent_tools:
            at.bind(self)

        all_tool_defs: list[ToolDef] = [
            *self.tools,
            *(at.definition for at in self.agent_tools),
        ]

        tool_map: dict[str, Any] = {}
        for t in all_tool_defs:
            handler = getattr(self, f"_tool_{t.name}", None)
            if handler:
                tool_map[t.name] = handler

        if self.llm.use_responses:
            tool_defs = [t.to_responses_tool() for t in all_tool_defs]
        else:
            tool_defs = [t.to_openai_tool() for t in all_tool_defs]

        for iteration in range(max_iterations):
            # Pass tools through so the LLM can decide to call one. When tools
            # are present, LLMClient.chat falls back to its non-streaming path
            # automatically (so tool-call parsing stays unchanged). Only the
            # final no-tool turn — when tool_defs is empty — actually streams.
            resp = self.llm.chat(
                messages,
                tools=tool_defs if tool_defs else None,
                stream=on_text is not None,
                on_text=on_text,
            )

            if self.llm.use_responses:
                tool_calls = [
                    item for item in resp.output if item.type == "function_call"
                ]
                if not tool_calls:
                    return resp.output_text

                # Serialize Pydantic output items to dicts
                for item in resp.output:
                    if isinstance(item, BaseModel):
                        messages.append(item.model_dump(exclude_none=True))
                    elif hasattr(item, "model_dump") and type(item).__name__ not in (
                        "MagicMock",
                        "Mock",
                    ):
                        messages.append(item.model_dump(exclude_none=True))
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
                num_calls = len(tool_calls)
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
                num_calls = len(message.tool_calls)

            logger.debug(
                "Tool iteration %d/%d: called %d tools",
                iteration + 1,
                max_iterations,
                num_calls,
            )

        # If we exhausted iterations, return last content
        return resp.output_text
