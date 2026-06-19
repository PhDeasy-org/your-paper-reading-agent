from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from ppagent.agents.tools import _web_search


class TestXaiWebSearch:
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_returns_error(self) -> None:
        result = _web_search("test query")
        assert "XAI_API_KEY environment variable is not set" in result

    @patch.dict(os.environ, {"XAI_API_KEY": "test-grok-key"})
    @patch("ppagent.agents.tools.OpenAI")
    def test_web_search_success_without_citations(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        
        # Mock choice response
        mock_choice = MagicMock()
        mock_choice.message.content = "This is a search result content."
        # No citations attribute
        if hasattr(mock_choice.message, "citations"):
            delattr(mock_choice.message, "citations")
            
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        result = _web_search("Who won the last space race?")
        
        # Verify OpenAI called correctly
        mock_openai_cls.assert_called_once_with(base_url="https://api.x.ai/v1", api_key="test-grok-key")
        mock_client.chat.completions.create.assert_called_once_with(
            model="grok-2-latest",
            messages=[
                {"role": "system", "content": "You are a helpful search assistant. Use the web search tool to find detailed, accurate information for the user's query."},
                {"role": "user", "content": "Who won the last space race?"}
            ],
            tools=[{"type": "web_search"}]
        )
        
        assert result == "This is a search result content."

    @patch.dict(os.environ, {"XAI_API_KEY": "test-grok-key"})
    @patch("ppagent.agents.tools.OpenAI")
    def test_web_search_success_with_citations(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        
        # Mock choice response with citations
        mock_choice = MagicMock()
        mock_choice.message.content = "Grok search results."
        mock_choice.message.citations = ["https://example.com/site1", "https://example.com/site2"]
            
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        result = _web_search("Space news")
        
        assert "Grok search results." in result
        assert "Citations:" in result
        assert "- https://example.com/site1" in result
        assert "- https://example.com/site2" in result

    @patch.dict(os.environ, {"XAI_API_KEY": "test-grok-key"})
    @patch("ppagent.agents.tools.OpenAI")
    def test_web_search_failure_returns_error_message(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API connection timed out")

        result = _web_search("query")
        assert "Web search failed: API connection timed out" in result
