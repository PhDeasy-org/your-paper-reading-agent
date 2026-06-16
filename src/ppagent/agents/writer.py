"""Writer agent — generates structured report sections from paper content.

When ``config.report.writer_research`` is enabled, the writer first runs a
multi-turn research phase using tool-calling (search_papers, paper_info,
read_paper) to look up unfamiliar concepts, cited works, and benchmarks.
The gathered research notes are then fed into the structured output call
so the final analysis is more accurate and thorough.
"""

from __future__ import annotations

import json
import logging

from ppagent.agents import register_agent
from ppagent.agents.base import AgentWithTools, ToolDef
from ppagent.agents.prompts import (
    WRITER_RESEARCH_SYSTEM_PROMPT,
    WRITER_RESEARCH_USER_PROMPT_TEMPLATE,
    WRITER_SYSTEM_PROMPT,
    WRITER_USER_PROMPT_TEMPLATE,
    WRITER_WITH_RESEARCH_USER_PROMPT_TEMPLATE,
)
from ppagent.llm import LLMClient
from ppagent import hf
from ppagent.models import AgentResult, PaperContent, WriterOutput

logger = logging.getLogger(__name__)

_MAX_RESEARCH_ITERATIONS = 8


@register_agent
class WriterAgent(AgentWithTools):
    """Generates structured report sections from paper content.

    When writer_research is enabled, performs multi-turn tool-based research
    before producing the final structured analysis.
    """

    name = "writer"
    description = "Analyzes paper content and produces structured report sections."

    def __init__(self, llm: LLMClient, config) -> None:
        super().__init__(llm, config)
        self.tools = [
            ToolDef(
                name="search_papers",
                description=(
                    "Search for papers on HuggingFace by query string. "
                    "Use this to look up unfamiliar concepts, methods, architectures, "
                    "benchmarks, or datasets mentioned in the paper. "
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
            ToolDef(
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
            ToolDef(
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
        ]

    # --------------------------------------------------------------------- #
    # Tool handlers                                                           #
    # --------------------------------------------------------------------- #

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
                        "summary": p.summary[:300] if p.summary else "",
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
                    "summary": paper.summary[:800] if paper.summary else "",
                },
                indent=2,
            )
        except Exception as exc:
            return f"Paper info failed: {exc}"

    def _tool_read_paper(self, paper_id: str) -> str:
        try:
            markdown = hf.paper_read(paper_id)
            # Truncate to avoid overwhelming the context window.
            max_chars = 8000
            if len(markdown) > max_chars:
                markdown = markdown[:max_chars] + "\n\n... [truncated]"
            return markdown
        except Exception as exc:
            return f"Read paper failed: {exc}"

    # --------------------------------------------------------------------- #
    # Research phase                                                          #
    # --------------------------------------------------------------------- #

    def _do_research(self, content: PaperContent) -> str:
        """Run the multi-turn tool-calling research loop.

        Returns the research notes produced by the LLM.
        """
        user_prompt = WRITER_RESEARCH_USER_PROMPT_TEMPLATE.format(
            title=content.paper.title,
            authors=", ".join(content.paper.authors),
            published=(
                content.paper.published_at.strftime("%Y-%m-%d")
                if content.paper.published_at
                else "Unknown"
            ),
            markdown=content.markdown,
        )

        messages = LLMClient.build_messages(WRITER_RESEARCH_SYSTEM_PROMPT, user_prompt)

        lang = self.config.report.language
        if lang and lang.lower() != "english":
            messages[0]["content"] += (
                f"\n\nIMPORTANT: Write ALL research notes in {lang}. "
                "Keep paper titles, author names, and technical terms in their original language."
            )

        research_notes = self._run_with_tools(
            messages, max_iterations=_MAX_RESEARCH_ITERATIONS
        )
        return research_notes

    # --------------------------------------------------------------------- #
    # Main entry point                                                        #
    # --------------------------------------------------------------------- #

    def run(self, *, content: PaperContent) -> AgentResult:
        self.llm.reset_usage()

        # Phase 1: research (optional)
        research_notes = ""
        if self.config.report.writer_research:
            try:
                logger.info("Writer: starting multi-turn research phase")
                research_notes = self._do_research(content)
                logger.info(
                    "Writer: research phase complete (%d chars of notes)",
                    len(research_notes),
                )
            except Exception as exc:
                logger.warning(
                    "Writer: research phase failed (%s); proceeding without research notes",
                    exc,
                )
                research_notes = ""

        # Phase 2: structured output
        if research_notes:
            user_prompt = WRITER_WITH_RESEARCH_USER_PROMPT_TEMPLATE.format(
                title=content.paper.title,
                authors=", ".join(content.paper.authors),
                published=(
                    content.paper.published_at.strftime("%Y-%m-%d")
                    if content.paper.published_at
                    else "Unknown"
                ),
                markdown=content.markdown,
                research_notes=research_notes,
            )
        else:
            user_prompt = WRITER_USER_PROMPT_TEMPLATE.format(
                title=content.paper.title,
                authors=", ".join(content.paper.authors),
                published=(
                    content.paper.published_at.strftime("%Y-%m-%d")
                    if content.paper.published_at
                    else "Unknown"
                ),
                markdown=content.markdown,
            )

        try:
            system_prompt = WRITER_SYSTEM_PROMPT
            lang = self.config.report.language
            if lang and lang.lower() != "english":
                system_prompt += (
                    f"\n\nIMPORTANT: Write ALL output text in {lang}. "
                    "Keep paper titles, author names, and technical terms in their original language."
                )
            output: WriterOutput = self.llm.chat_structured(
                LLMClient.build_messages(system_prompt, user_prompt),
                response_model=WriterOutput,
            )
        except Exception as exc:
            logger.error("Writer LLM call failed: %s", exc)
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(exc),
                usage=self.llm.get_usage(),
            )

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "keywords": output.metadata_keywords,
                "affiliations": output.affiliations,
                "benchmarks": output.benchmarks,
                "tldr": output.tldr,
                "previous_works": output.previous_works_summary,
                "method": output.method_details,
                "evaluation": output.performance_evaluation,
            },
            usage=self.llm.get_usage(),
        )
