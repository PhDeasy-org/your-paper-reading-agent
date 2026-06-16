"""Finder agent — discovers related/previous works using tool-calling."""

from __future__ import annotations

import json
import logging

from ppagent.agents import register_agent
from ppagent.agents.base import AgentWithTools, ToolDef
from ppagent.agents.prompts import (
    FINDER_SYSTEM_PROMPT,
    FINDER_STRUCTURED_SYSTEM_PROMPT,
    FINDER_USER_PROMPT_TEMPLATE,
    FINDER_STRUCTURED_USER_PROMPT_TEMPLATE,
)
from ppagent.llm import LLMClient
from ppagent import hf
from ppagent.models import AgentResult, FinderOutput, PaperContent, Paper

logger = logging.getLogger(__name__)


@register_agent
class FinderAgent(AgentWithTools):
    """Discovers related/previous works via HuggingFace paper search."""

    name = "finder"
    description = "Searches for impactful related works to a given paper."

    def __init__(self, llm: LLMClient, config) -> None:
        super().__init__(llm, config)
        self.tools = [
            ToolDef(
                name="search_papers",
                description="Search for papers on HuggingFace by query string. Returns a list of papers with IDs, titles, and upvotes.",
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
            ToolDef(
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
        ]

    def _tool_search_papers(self, query: str, limit: int = 5) -> str:
        try:
            papers = hf.search_papers(query, limit=limit)
            if not papers:
                return "No papers found."
            results = []
            for p in papers:
                results.append(
                    {
                        "id": p.id,
                        "title": p.title,
                        "upvotes": p.upvotes,
                        "summary": p.summary[:200] if p.summary else "",
                    }
                )
            return json.dumps(results, indent=2)
        except Exception as exc:
            return f"Search failed: {exc}"

    def _tool_paper_info(self, paper_id: str) -> str:
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

    def run(self, *, content: PaperContent) -> AgentResult:
        self.llm.reset_usage()
        user_prompt = FINDER_USER_PROMPT_TEMPLATE.format(
            title=content.paper.title,
            authors=", ".join(content.paper.authors),
            summary=content.paper.summary,
            excerpt=content.markdown[:3000],
        )

        messages = LLMClient.build_messages(FINDER_SYSTEM_PROMPT, user_prompt)

        lang = self.config.report.language
        if lang and lang.lower() != "english":
            messages[0]["content"] += (
                f"\n\nIMPORTANT: Write ALL narrative text in {lang}. Keep paper titles and technical terms in their original language."
            )

        try:
            narrative = self._run_with_tools(messages)
        except Exception as exc:
            logger.error("Finder agent failed: %s", exc)
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(exc),
                usage=self.llm.get_usage(),
            )

        # Now do a structured call to extract the final output
        final_prompt = FINDER_STRUCTURED_USER_PROMPT_TEMPLATE.format(
            narrative=narrative
        )

        try:
            output: FinderOutput = self.llm.chat_structured(
                LLMClient.build_messages(
                    FINDER_STRUCTURED_SYSTEM_PROMPT,
                    final_prompt,
                ),
                response_model=FinderOutput,
            )
        except Exception as exc:
            logger.warning(
                "Finder structured output failed: %s — using narrative only", exc
            )
            output = FinderOutput(narrative=narrative)

        # Build Paper objects from related works
        related_papers: list[Paper] = []
        for rw in output.related_works:
            if rw.paper_id:
                related_papers.append(Paper(id=rw.paper_id, title=rw.title))

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "narrative": output.narrative or narrative,
                "related_works": related_papers,
            },
            usage=self.llm.get_usage(),
        )
