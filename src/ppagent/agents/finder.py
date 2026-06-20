"""Finder agent — discovers related/previous works using tool-calling."""

from __future__ import annotations

import logging
from collections.abc import Callable

from ppagent.agents import register_agent
from ppagent.agents.base import AgentWithTools
from ppagent.agents.prompts import (
    FINDER_SYSTEM_PROMPT,
    FINDER_STRUCTURED_SYSTEM_PROMPT,
    FINDER_USER_PROMPT_TEMPLATE,
    FINDER_STRUCTURED_USER_PROMPT_TEMPLATE,
)
from ppagent.agents.tools import HF_PAPER_INFO, HF_READ_PAPER, XAI_WEB_SEARCH
from ppagent.llm import LLMClient
from ppagent.models import AgentResult, FinderOutput, PaperContent

logger = logging.getLogger(__name__)


@register_agent
class FinderAgent(AgentWithTools):
    """Discovers related/previous works via HuggingFace paper search."""

    name = "finder"
    description = "Searches for impactful related works to a given paper."

    def __init__(self, llm: LLMClient, config) -> None:
        super().__init__(llm, config)
        self.agent_tools = [XAI_WEB_SEARCH, HF_PAPER_INFO, HF_READ_PAPER]

    def run(
        self,
        *,
        content: PaperContent,
        on_text: Callable[[str], None] | None = None,
    ) -> AgentResult:
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
            narrative = self._run_with_tools(messages, on_text=on_text)
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

        # Keep only related works that carry an identifier we can link to.
        related_works = [rw for rw in output.related_works if rw.paper_id]

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "narrative": output.narrative or narrative,
                "related_works": related_works,
            },
            usage=self.llm.get_usage(),
        )
