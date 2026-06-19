"""Tests for Paper Type Classifier Agent."""

from __future__ import annotations

import pytest

from ppagent.agents.classifier import ClassifierAgent
from ppagent.agents.prompts import DEFAULT_PAPER_TYPE
from ppagent.config import AppConfig
from ppagent.models import ClassifierOutput, Paper, PaperContent


class MockLLMClient:
    """Mock LLMClient for structured classification testing."""

    def __init__(self, response_value: ClassifierOutput | Exception):
        self.response_value = response_value
        self.messages_called = []
        self.response_model_called = None
        self.usage = {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
        }

    def reset_usage(self):
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def get_usage(self):
        return self.usage

    @staticmethod
    def build_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def chat_structured(
        self, messages: list[dict[str, str]], response_model: type
    ) -> ClassifierOutput:
        self.messages_called = messages
        self.response_model_called = response_model
        if isinstance(self.response_value, Exception):
            raise self.response_value
        return self.response_value


@pytest.fixture
def minimal_config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def sample_paper_content() -> PaperContent:
    paper = Paper(
        id="2506.12345",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        summary="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
    )
    return PaperContent(
        paper=paper,
        markdown="Full paper content goes here...",
    )


def test_classifier_success(minimal_config, sample_paper_content):
    """Verify that a successful classification returns the correct paper type and confidence."""
    mock_output = ClassifierOutput(
        paper_type="method",
        confidence=0.95,
        reasoning="Introduces a new method called Transformer.",
    )
    mock_llm = MockLLMClient(mock_output)

    agent = ClassifierAgent(mock_llm, minimal_config)  # type: ignore[arg-type]
    result = agent.run(content=sample_paper_content)

    assert result.success is True
    assert result.data["paper_type"] == "method"
    assert result.data["confidence"] == 0.95
    assert result.data["reasoning"] == "Introduces a new method called Transformer."

    # Verify the structured response model requested was correct
    assert mock_llm.response_model_called is ClassifierOutput
    assert len(mock_llm.messages_called) == 2
    # Verify prompts contain relevant types and paper info
    system_content = mock_llm.messages_called[0]["content"]
    user_content = mock_llm.messages_called[1]["content"]
    assert "method" in system_content
    assert "benchmark" in system_content
    assert "Attention Is All You Need" in user_content
    assert "Transformer, based solely on attention mechanisms" in user_content


def test_classifier_invalid_type_fallback(minimal_config, sample_paper_content):
    """If the LLM returns an invalid paper type, it should fall back to the default type."""
    mock_output = ClassifierOutput(
        paper_type="invalid_type_xyz",
        confidence=0.8,
        reasoning="Not sure.",
    )
    mock_llm = MockLLMClient(mock_output)

    agent = ClassifierAgent(mock_llm, minimal_config)  # type: ignore[arg-type]
    result = agent.run(content=sample_paper_content)

    # The agent run should still succeed, but use the fallback type
    assert result.success is True
    assert result.data["paper_type"] == DEFAULT_PAPER_TYPE
    assert result.data["confidence"] == 0.8
    assert result.data["reasoning"] == "Not sure."


def test_classifier_llm_failure(minimal_config, sample_paper_content):
    """If the LLM call raises an exception, the agent should return a failure result."""
    mock_llm = MockLLMClient(RuntimeError("API error"))

    agent = ClassifierAgent(mock_llm, minimal_config)  # type: ignore[arg-type]
    result = agent.run(content=sample_paper_content)

    assert result.success is False
    assert "API error" in result.error
    assert result.data == {}


def test_classifier_missing_summary_fallback(minimal_config):
    """If the paper has no summary/abstract, the agent should fall back to the first 2000 chars of markdown."""
    paper = Paper(
        id="2506.12345",
        title="Attention Is All You Need",
        summary="",  # missing summary
    )
    long_markdown = "Introduction: " + ("abc " * 1000)  # > 2000 chars
    content = PaperContent(paper=paper, markdown=long_markdown)

    mock_output = ClassifierOutput(
        paper_type="survey",
        confidence=0.9,
        reasoning="It is a survey.",
    )
    mock_llm = MockLLMClient(mock_output)

    agent = ClassifierAgent(mock_llm, minimal_config)  # type: ignore[arg-type]
    result = agent.run(content=content)

    assert result.success is True
    assert result.data["paper_type"] == "survey"

    # Verify the user prompt got the snippet of the markdown (capped to 2000 chars)
    user_content = mock_llm.messages_called[1]["content"]
    assert "Introduction:" in user_content
    assert len(user_content) > 1000
    assert len(user_content) < 2500
