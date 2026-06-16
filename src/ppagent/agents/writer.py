"""Writer agent — generates structured report sections from paper content."""

from __future__ import annotations

import logging

from ppagent.agents import register_agent
from ppagent.agents.base import AgentBase
from ppagent.agents.prompts import (
    WRITER_SYSTEM_PROMPT,
    WRITER_USER_PROMPT_TEMPLATE,
)
from ppagent.llm import LLMClient
from ppagent.models import AgentResult, PaperContent, WriterOutput

logger = logging.getLogger(__name__)


@register_agent
class WriterAgent(AgentBase):
    """Generates structured report sections from paper content."""

    name = "writer"
    description = "Analyzes paper content and produces structured report sections."

    def run(self, *, content: PaperContent) -> AgentResult:
        self.llm.reset_usage()
        user_prompt = WRITER_USER_PROMPT_TEMPLATE.format(
            title=content.paper.title,
            authors=", ".join(content.paper.authors),
            published=content.paper.published_at.strftime("%Y-%m-%d") if content.paper.published_at else "Unknown",
            markdown=content.markdown,
        )

        try:
            system_prompt = WRITER_SYSTEM_PROMPT
            lang = self.config.report.language
            if lang and lang.lower() != "english":
                system_prompt += f"\n\nIMPORTANT: Write ALL output text in {lang}. Keep paper titles, author names, and technical terms in their original language."
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
