"""Writer agent — generates structured report sections from paper content.

When ``config.report.writer_research`` is enabled, the writer first runs a
multi-turn research phase using tool-calling (search_papers, paper_info,
read_paper) to look up unfamiliar concepts, cited works, and benchmarks.
The gathered research notes are then fed into the structured output call
so the final analysis is more accurate and thorough.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from ppagent.agents import register_agent
from ppagent.agents.base import AgentWithTools
from ppagent.agents.prompts import (
    WRITER_RESEARCH_SYSTEM_PROMPT,
    WRITER_RESEARCH_USER_PROMPT_TEMPLATE,
    WRITER_SYSTEM_PROMPTS,
    WRITER_USER_PROMPT_TEMPLATE,
    WRITER_WITH_RESEARCH_USER_PROMPT_TEMPLATE,
    DEFAULT_PAPER_TYPE,
)
from ppagent.agents.tools import HF_TOOLS
from ppagent.config import AppConfig
from ppagent.llm import LLMClient
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

    def __init__(self, llm: LLMClient, config: AppConfig) -> None:
        super().__init__(llm, config)
        self.agent_tools = list(HF_TOOLS)

    # --------------------------------------------------------------------- #
    # Research phase                                                          #
    # --------------------------------------------------------------------- #

    def _do_research(
        self,
        content: PaperContent,
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> str:
        """Run the multi-turn tool-calling research loop.

        Returns the research notes produced by the LLM. When ``on_text`` is
        provided, the final no-tool turn's text deltas are streamed to it.
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
            messages, max_iterations=_MAX_RESEARCH_ITERATIONS, on_text=on_text
        )
        return research_notes

    # --------------------------------------------------------------------- #
    # Main entry point                                                        #
    # --------------------------------------------------------------------- #

    def run(
        self,
        *,
        content: PaperContent,
        paper_type: str = DEFAULT_PAPER_TYPE,
        on_text: Callable[[str], None] | None = None,
    ) -> AgentResult:
        self.llm.reset_usage()

        # Phase 1: research (optional)
        research_notes = ""
        if self.config.report.writer_research:
            try:
                logger.info("Writer: starting multi-turn research phase")
                research_notes = self._do_research(content, on_text=on_text)
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
            system_prompt = WRITER_SYSTEM_PROMPTS.get(
                paper_type, WRITER_SYSTEM_PROMPTS[DEFAULT_PAPER_TYPE]
            )
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
                "blog_url": output.blog_url,
                "paper_type": paper_type,
            },
            usage=self.llm.get_usage(),
        )
