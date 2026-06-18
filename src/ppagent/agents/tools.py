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
from dataclasses import dataclass
from typing import Any, Callable

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
                "summary": p.summary[:200] if p.summary else "",
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
                "summary": paper.summary[:500] if paper.summary else "",
            },
            indent=2,
        )
    except Exception as exc:
        return f"Paper info failed: {exc}"


HF_SEARCH_PAPERS = AgentTool(
    definition=ToolDef(
        name="search_papers",
        description=(
            "Search for papers on HuggingFace by query string. "
            "Returns a list of papers with IDs, titles, and upvotes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., method name, topic, benchmark name).",
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
        description="Get detailed info about a specific paper by its arXiv ID.",
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

#: Convenience bundle — all HuggingFace paper tools.
HF_TOOLS: list[AgentTool] = [HF_SEARCH_PAPERS, HF_PAPER_INFO]
