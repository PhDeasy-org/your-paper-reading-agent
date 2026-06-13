"""Base classes for ppagent agents."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from ppagent.config import AppConfig
from ppagent.llm import LLMClient
from ppagent.models import AgentResult

logger = logging.getLogger(__name__)


class ToolDef(BaseModel):
    """Definition of a tool that an agent can use."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
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
        return resp.choices[0].message.content or ""


class AgentWithTools(AgentBase):
    """Agent that can use tools via LLM function calling."""

    def __init__(self, llm: LLMClient, config: AppConfig) -> None:
        super().__init__(llm, config)
        self.tools: list[ToolDef] = []

    def _run_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        max_iterations: int = 5,
    ) -> str:
        """Agentic loop: call LLM → if tool_calls, execute → feed back → repeat."""
        tool_map: dict[str, Any] = {}
        for t in self.tools:
            handler = getattr(self, f"_tool_{t.name}", None)
            if handler:
                tool_map[t.name] = handler

        tool_defs = [t.to_openai_tool() for t in self.tools]

        for iteration in range(max_iterations):
            resp = self.llm.chat(messages, tools=tool_defs if tool_defs else None)
            choice = resp.choices[0]

            if not choice.message.tool_calls:
                return choice.message.content or ""

            # Append the assistant message with tool calls
            messages.append(choice.message.model_dump())

            for call in choice.message.tool_calls:
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

            logger.debug(
                "Tool iteration %d/%d: called %d tools",
                iteration + 1,
                max_iterations,
                len(choice.message.tool_calls),
            )

        # If we exhausted iterations, return last content
        return resp.choices[0].message.content or ""
