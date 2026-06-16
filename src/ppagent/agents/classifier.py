"""Classifier agent — identifies the paper type before report generation."""

from __future__ import annotations

import logging

from ppagent.agents import register_agent
from ppagent.agents.base import AgentBase
from ppagent.agents.prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFIER_USER_PROMPT_TEMPLATE,
    DEFAULT_PAPER_TYPE,
    PAPER_TYPES,
)
from ppagent.llm import LLMClient
from ppagent.models import AgentResult, ClassifierOutput, PaperContent

logger = logging.getLogger(__name__)


@register_agent
class ClassifierAgent(AgentBase):
    """Classifies a paper into one of the predefined paper types."""

    name = "classifier"
    description = "Identifies the paper type (method, benchmark, survey, etc.)."

    def run(self, *, content: PaperContent) -> AgentResult:
        self.llm.reset_usage()

        # Build the type descriptions list for the prompt
        type_descriptions = "\n".join(
            f"- **{tid}**: {desc}" for tid, desc in PAPER_TYPES.items()
        )

        system_prompt = CLASSIFIER_SYSTEM_PROMPT.format(
            type_descriptions=type_descriptions,
        )

        # Use title + summary/abstract for classification (lightweight)
        summary = content.paper.summary
        if not summary:
            # Fall back to first ~2000 chars of the full text
            summary = content.markdown[:2000] if content.markdown else "No content available."

        user_prompt = CLASSIFIER_USER_PROMPT_TEMPLATE.format(
            title=content.paper.title,
            summary=summary,
        )

        try:
            output: ClassifierOutput = self.llm.chat_structured(
                LLMClient.build_messages(system_prompt, user_prompt),
                response_model=ClassifierOutput,
            )
        except Exception as exc:
            logger.error("Classifier LLM call failed: %s", exc)
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(exc),
                usage=self.llm.get_usage(),
            )

        # Validate the returned paper type
        paper_type = output.paper_type.lower().strip()
        if paper_type not in PAPER_TYPES:
            logger.warning(
                "Classifier returned unknown type %r; falling back to %r",
                paper_type,
                DEFAULT_PAPER_TYPE,
            )
            paper_type = DEFAULT_PAPER_TYPE

        logger.info(
            "Classified paper as %r (confidence=%.2f): %s",
            paper_type,
            output.confidence,
            output.reasoning,
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "paper_type": paper_type,
                "confidence": output.confidence,
                "reasoning": output.reasoning,
            },
            usage=self.llm.get_usage(),
        )
