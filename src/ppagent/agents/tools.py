"""Shared tool definitions for ppagent agents.

Each :class:`AgentTool` bundles a JSON-schema :class:`ToolDef` with its
handler callable so that any :class:`AgentWithTools` subclass can pick up
tools without re-implementing the same logic.

Usage example (inside an ``AgentWithTools`` subclass)::

    from ppagent.agents.tools import HF_SEARCH_PAPERS, HF_PAPER_INFO

    class MyAgent(AgentWithTools):
        def __init__(self, llm, config):
            super().__init__(llm, config)
            self.agent_tools = [HF_SEARCH_PAPERS, HF_PAPER_INFO]

``AgentWithTools._run_with_tools`` will discover all registered handlers via
:meth:`AgentTool.bind`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI

from ppagent.agents.base import ToolDef
from ppagent import hf

logger = logging.getLogger(__name__)


@dataclass
class AgentTool:
    """Pairs a :class:`ToolDef` (LLM schema) with its Python handler.

    Call :meth:`bind` to attach the handler to an agent instance under the
    ``_tool_<name>`` attribute that :class:`AgentWithTools` expects.
    """

    definition: ToolDef
    handler: Callable[..., str]

    @property
    def name(self) -> str:
        return self.definition.name

    def bind(self, agent: Any) -> None:
        """Install ``handler`` as ``_tool_<name>`` on *agent*."""
        setattr(agent, f"_tool_{self.name}", self.handler)


# ---------------------------------------------------------------------------
# HuggingFace paper tools
# ---------------------------------------------------------------------------


def _search_papers(query: str, limit: int = 5) -> str:
    """Search HuggingFace papers and return JSON summary."""
    try:
        papers = hf.search_papers(query, limit=limit)
        if not papers:
            return "No papers found."
        results = [
            {
                "id": p.id,
                "title": p.title,
                "upvotes": p.upvotes,
                "summary": p.summary[:300] if p.summary else "",
            }
            for p in papers
        ]
        return json.dumps(results, indent=2)
    except Exception as exc:
        return f"Search failed: {exc}"


def _paper_info(paper_id: str) -> str:
    """Fetch detailed info for a paper by arXiv ID and return JSON."""
    try:
        paper = hf.paper_info(paper_id)
        return json.dumps(
            {
                "id": paper.id,
                "title": paper.title,
                "authors": paper.authors,
                "upvotes": paper.upvotes,
                "summary": paper.summary[:800] if paper.summary else "",
            },
            indent=2,
        )
    except Exception as exc:
        return f"Paper info failed: {exc}"


def _read_paper(paper_id: str) -> str:
    """Read the full markdown text of a paper, truncated to fit context."""
    try:
        markdown = hf.paper_read(paper_id)
        max_chars = 8000
        if len(markdown) > max_chars:
            markdown = markdown[:max_chars] + "\n\n... [truncated]"
        return markdown
    except Exception as exc:
        return f"Read paper failed: {exc}"


HF_SEARCH_PAPERS = AgentTool(
    definition=ToolDef(
        name="search_papers",
        description=(
            "Search for papers on HuggingFace by query string. "
            "Use this to look up unfamiliar concepts, methods, architectures, "
            "benchmarks, or datasets. "
            "Returns a list of papers with IDs, titles, and summaries."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query (e.g., method name, concept, benchmark, "
                        "dataset, or cited work title)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    handler=_search_papers,
)

HF_PAPER_INFO = AgentTool(
    definition=ToolDef(
        name="paper_info",
        description=(
            "Get detailed metadata and abstract for a specific paper by its arXiv ID. "
            "Use this to quickly understand what a cited paper is about without "
            "reading its full text."
        ),
        parameters={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The arXiv paper ID (e.g., '2301.08210').",
                },
            },
            "required": ["paper_id"],
        },
    ),
    handler=_paper_info,
)

HF_READ_PAPER = AgentTool(
    definition=ToolDef(
        name="read_paper",
        description=(
            "Read the full text (as markdown) of a specific paper by its arXiv ID. "
            "Use this when you need deep context about a cited paper's method, "
            "architecture, or findings — not just its abstract. "
            "Prefer paper_info for quick lookups; use read_paper when the abstract "
            "alone is not enough."
        ),
        parameters={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The arXiv paper ID (e.g., '2301.08210').",
                },
            },
            "required": ["paper_id"],
        },
    ),
    handler=_read_paper,
)

#: Convenience bundle — all HuggingFace paper tools.
HF_TOOLS: list[AgentTool] = [HF_SEARCH_PAPERS, HF_PAPER_INFO, HF_READ_PAPER]


def _web_search(query: str) -> str:
    """Search the web in real-time using xAI's Grok model with native web search.

    Use this to search the internet for related papers, blog posts, code,
    benchmarks, or other general web information.
    """
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        return "Error: XAI_API_KEY environment variable is not set."

    try:
        client = OpenAI(base_url="https://api.x.ai/v1", api_key=api_key)
        response = client.responses.create(
            model="grok-4.3",
            input=[
                {"role": "user", "content": query},
            ],
            tools=[{"type": "web_search"}],
        )
        content = getattr(response, "output_text", "") or ""
        citations = getattr(response, "citations", None)
        if citations:
            content += "\n\nCitations:\n" + "\n".join(f"- {c}" for c in citations)
        return content
    except Exception as exc:
        return f"Web search failed: {exc}"


XAI_WEB_SEARCH = AgentTool(
    definition=ToolDef(
        name="web_search",
        description=(
            "Search the web in real-time for any information. "
            "Use this to look up related papers, authors, github repositories, "
            "blog posts, explanations of concepts, benchmarks, or datasets. "
            "Returns a detailed summary of search results with sources."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the internet.",
                },
            },
            "required": ["query"],
        },
    ),
    handler=_web_search,
)
