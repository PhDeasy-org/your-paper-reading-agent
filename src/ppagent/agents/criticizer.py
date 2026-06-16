"""Criticizer agent — strictly audits a paper and identifies limitations."""

from __future__ import annotations

import logging
from typing import Any

from ppagent.agents import register_agent
from ppagent.agents.base import AgentBase
from ppagent.agents.prompts import (
    CRITICIZER_SYSTEM_PROMPTS,
    CRITICIZER_USER_PROMPT_TEMPLATE,
    CRITICIZER_WRITER_CONTEXT_TEMPLATE,
    DEFAULT_PAPER_TYPE,
)
from ppagent.llm import LLMClient
from ppagent.models import AgentResult, CriticizerOutput, PaperContent

logger = logging.getLogger(__name__)


@register_agent
class CriticizerAgent(AgentBase):
    """Strictly audits a paper and identifies limitations."""

    name = "criticizer"
    description = "Performs rigorous critical analysis of a paper."

    def run(
        self,
        *,
        content: PaperContent,
        writer_sections: dict[str, Any] | None = None,
        paper_type: str = DEFAULT_PAPER_TYPE,
    ) -> AgentResult:
        self.llm.reset_usage()
        # Include writer's analysis for additional context
        writer_context = ""
        if writer_sections:
            writer_context = CRITICIZER_WRITER_CONTEXT_TEMPLATE.format(
                method=writer_sections.get("method", "N/A"),
                evaluation=writer_sections.get("evaluation", "N/A"),
                previous_works=writer_sections.get("previous_works", "N/A"),
            )

        user_prompt = CRITICIZER_USER_PROMPT_TEMPLATE.format(
            title=content.paper.title,
            authors=", ".join(content.paper.authors),
            writer_context=writer_context,
            markdown=content.markdown,
        )

        try:
            system_prompt = CRITICIZER_SYSTEM_PROMPTS.get(paper_type, CRITICIZER_SYSTEM_PROMPTS[DEFAULT_PAPER_TYPE])
            lang = self.config.report.language
            if lang and lang.lower() != "english":
                system_prompt += f"\n\nIMPORTANT: Write ALL output text in {lang}. Keep paper titles and technical terms in their original language."
            output: CriticizerOutput = self.llm.chat_structured(
                LLMClient.build_messages(system_prompt, user_prompt),
                response_model=CriticizerOutput,
            )
        except Exception as exc:
            logger.error("Criticizer LLM call failed: %s", exc)
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(exc),
                usage=self.llm.get_usage(),
            )

        # Format findings into a readable critique
        critique_parts = [output.summary]
        for f in output.findings:
            critique_parts.append(f"- **[{f.severity.upper()}]** {f.finding}")

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "critique": "\n\n".join(critique_parts),
                "findings_count": len(output.findings),
                "high_severity": sum(
                    1 for f in output.findings if f.severity == "high"
                ),
            },
            usage=self.llm.get_usage(),
        )
