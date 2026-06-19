from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ppagent.agents.base import AgentBase, AgentWithTools, ToolDef
from ppagent.config import AppConfig
from ppagent.models import AgentResult


class DummyAgent(AgentWithTools):
    name = "dummy"
    description = "a dummy agent for testing"

    def __init__(self, llm: Any, config: AppConfig) -> None:
        super().__init__(llm, config)
        self.tools = [
            ToolDef(
                name="dummy_tool",
                description="A test tool",
                parameters={
                    "type": "object",
                    "properties": {"arg1": {"type": "string"}},
                    "required": ["arg1"],
                },
            )
        ]

    def run(self, **kwargs: Any) -> AgentResult:
        return AgentResult(agent_name=self.name, success=True)

    def _tool_dummy_tool(self, arg1: str) -> str:
        return f"result: {arg1}"


@pytest.fixture
def mock_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "search": {
                "default_date": "today",
                "default_limit": 50,
                "sort": "trending",
                "profile_path": "config/profile.md",
                "relevance_threshold": 0.6,
                "max_reports_per_run": 5,
            },
            "report": {
                "output_dir": "output",
                "template_dir": "templates",
                "formats": ["md", "html"],
                "download_pdf": True,
                "pdf_cache_dir": ".cache/pdfs",
                "custom_agents": [],
                "language": "English",
            },
        }
    )


def test_call_llm_uses_output_text(mock_config: AppConfig) -> None:
    mock_llm = MagicMock()
    # Mock llm.chat to return a Response-like mock
    mock_response = MagicMock()
    mock_response.output_text = "mocked response text"
    # Ensure there is no choices attribute to simulate the new Response object
    del mock_response.choices

    mock_llm.chat.return_value = mock_response

    class SimpleAgent(AgentBase):
        def run(self, **kwargs: Any) -> AgentResult:
            return AgentResult(agent_name=self.name, success=True)

    agent = SimpleAgent(mock_llm, mock_config)
    res = agent._call_llm("system", "user")

    assert res == "mocked response text"
    mock_llm.chat.assert_called_once()


def test_run_with_tools_executes_tool(mock_config: AppConfig) -> None:
    mock_llm = MagicMock()

    # Define mock response for iteration 0 with a tool call
    mock_call = MagicMock()
    mock_call.type = "function_call"
    mock_call.name = "dummy_tool"
    mock_call.arguments = '{"arg1": "hello"}'
    mock_call.call_id = "call_123"

    mock_resp1 = MagicMock()
    mock_resp1.output = [mock_call]
    mock_resp1.output_text = "iteration 0"
    del mock_resp1.choices

    # Define mock response for iteration 1 with no tool calls
    mock_resp2 = MagicMock()
    mock_resp2.output = []
    mock_resp2.output_text = "final output text"
    del mock_resp2.choices

    mock_llm.chat.side_effect = [mock_resp1, mock_resp2]

    agent = DummyAgent(mock_llm, mock_config)
    messages = [{"role": "user", "content": "run dummy"}]
    res = agent._run_with_tools(messages)

    assert res == "final output text"
    assert len(messages) == 3
    # Check original user message is still first
    assert messages[0] == {"role": "user", "content": "run dummy"}
    # Check resp1.output was extended (which is the mock_call item)
    assert messages[1] == mock_call
    # Check tool output was appended
    assert messages[2] == {
        "type": "function_call_output",
        "call_id": "call_123",
        "output": "result: hello",
    }
