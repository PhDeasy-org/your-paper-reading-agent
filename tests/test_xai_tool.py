from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from ppagent.agents.tools import _web_search


class TestXaiWebSearch:
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_returns_error(self) -> None:
        result = _web_search("test query")
        assert "API key for the active provider is not set" in result

    @patch.dict(os.environ, {"XAI_API_KEY": "test-grok-key"})
    @patch("ppagent.agents.tools.OpenAI")
    def test_web_search_success_default_grok(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.output_text = "This is a search result content."
        mock_response.citations = None

        mock_client.responses.create.return_value = mock_response

        result = _web_search("Who won the last space race?")

        # Verify OpenAI called correctly with default Grok config
        mock_openai_cls.assert_called_once_with(
            base_url="https://api.x.ai/v1", api_key="test-grok-key"
        )
        mock_client.responses.create.assert_called_once_with(
            model="grok-4.3",
            input=[{"role": "user", "content": "Who won the last space race?"}],
            tools=[{"type": "web_search"}],
        )

        assert result == "This is a search result content."

    @patch.dict(os.environ, {}, clear=True)
    @patch("ppagent.agents.tools.OpenAI")
    def test_web_search_success_agent_custom_openai(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.output_text = "OpenAI search results."
        mock_response.citations = None
        mock_client.responses.create.return_value = mock_response

        # Mock agent with OpenAI config
        mock_agent = MagicMock()
        mock_agent.llm.config.base_url = "https://api.openai.com/v1"
        mock_agent.llm.config.api_key = "openai-api-key"
        mock_agent.llm.config.model = "gpt-4o"

        result = _web_search("OpenAI updates", agent=mock_agent)

        mock_openai_cls.assert_called_once_with(
            base_url="https://api.openai.com/v1", api_key="openai-api-key"
        )
        mock_client.responses.create.assert_called_once_with(
            model="gpt-4o",
            input=[{"role": "user", "content": "OpenAI updates"}],
            tools=[{"type": "web_search"}],
        )
        assert result == "OpenAI search results."

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "qwen-api-key"}, clear=True)
    @patch("ppagent.agents.tools.OpenAI")
    def test_web_search_success_agent_qwen_with_env_key(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.output_text = "Qwen search results."
        mock_response.citations = None
        mock_client.responses.create.return_value = mock_response

        # Mock agent with Qwen config (no api_key in config, relies on env var)
        mock_agent = MagicMock()
        mock_agent.llm.config.base_url = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        mock_agent.llm.config.api_key = ""
        mock_agent.llm.config.model = "qwen-plus"

        result = _web_search("Qwen updates", agent=mock_agent)

        mock_openai_cls.assert_called_once_with(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="qwen-api-key",
        )
        assert result == "Qwen search results."

    @patch.dict(os.environ, {"ARK_API_KEY": "doubao-api-key"}, clear=True)
    @patch("ppagent.agents.tools.OpenAI")
    def test_web_search_success_agent_doubao_with_env_key(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.output_text = "Doubao search results."
        mock_response.citations = None
        mock_client.responses.create.return_value = mock_response

        # Mock agent with Doubao config
        mock_agent = MagicMock()
        mock_agent.llm.config.base_url = "https://ark.cn-beijing.volces.com/api/v3"
        mock_agent.llm.config.api_key = ""
        mock_agent.llm.config.model = "doubao-pro-32k"

        result = _web_search("Doubao updates", agent=mock_agent)

        mock_openai_cls.assert_called_once_with(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="doubao-api-key",
        )
        assert result == "Doubao search results."

    @patch.dict(os.environ, {"XAI_API_KEY": "test-grok-key"})
    @patch("ppagent.agents.tools.OpenAI")
    def test_web_search_success_with_citations(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.output_text = "Grok search results."
        mock_response.citations = [
            "https://example.com/site1",
            "https://example.com/site2",
        ]

        mock_client.responses.create.return_value = mock_response

        result = _web_search("Space news")

        assert "Grok search results." in result
        assert "Citations:" in result
        assert "- https://example.com/site1" in result
        assert "- https://example.com/site2" in result

    @patch.dict(os.environ, {"XAI_API_KEY": "test-grok-key"})
    @patch("ppagent.agents.tools.OpenAI")
    def test_web_search_failure_returns_error_message(
        self, mock_openai_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.responses.create.side_effect = Exception("API connection timed out")

        result = _web_search("query")
        assert "Web search failed: API connection timed out" in result
