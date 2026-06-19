from __future__ import annotations

from unittest.mock import MagicMock

from ppagent.agents.finder import FinderAgent
from ppagent.agents.tools import (
    XAI_WEB_SEARCH,
    HF_PAPER_INFO,
    HF_READ_PAPER,
    HF_SEARCH_PAPERS,
)
from ppagent.config import AppConfig


def test_finder_agent_registers_xai_web_search() -> None:
    # Set up mocks
    mock_llm = MagicMock()
    config = AppConfig.model_validate(
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

    # Instantiate FinderAgent
    agent = FinderAgent(mock_llm, config)

    # Assert registration of the new web search tool
    assert XAI_WEB_SEARCH in agent.agent_tools
    assert HF_PAPER_INFO in agent.agent_tools
    assert HF_READ_PAPER in agent.agent_tools

    # Assert HF search is not registered
    assert HF_SEARCH_PAPERS not in agent.agent_tools
