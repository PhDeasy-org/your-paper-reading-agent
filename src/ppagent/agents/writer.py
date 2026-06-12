"""Writer agent — generates structured report sections from paper content."""

from __future__ import annotations

import logging

from ppagent.agents import register_agent
from ppagent.agents.base import AgentBase
from ppagent.llm import LLMClient
from ppagent.models import AgentResult, PaperContent, WriterOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert research paper analyst. Given the full text of a paper, produce a \
detailed structured analysis. Be precise, thorough, and technical.

For each section:
- **Keywords**: Extract 5-8 key technical terms/concepts from the paper.
- **Affiliations**: List the institutional affiliations of the authors.
- **Benchmarks**: List all benchmarks, datasets, and evaluation metrics used. If none, write "None reported."
- **TL;DR**: Write a concise 2-3 sentence summary of the paper's core contribution.
- **Previous Works Summary**: Summarize the related work section — what prior methods exist and what are their limitations that motivate this work.
- **Method Details**: Describe the proposed method in detail, including architecture, training procedure, key equations, and novel components. Be technical and thorough.
- **Performance Evaluation**: Summarize the experimental results — main results, comparisons with baselines, ablation studies, and key findings. Include specific numbers where available.\
"""


@register_agent
class WriterAgent(AgentBase):
    """Generates structured report sections from paper content."""

    name = "writer"
    description = "Analyzes paper content and produces structured report sections."

    def run(self, *, content: PaperContent) -> AgentResult:
        user_prompt = f"""\
## Paper: {content.paper.title}

**Authors**: {', '.join(content.paper.authors)}
**Published**: {content.paper.published_at.strftime('%Y-%m-%d') if content.paper.published_at else 'Unknown'}

## Full Paper Content

{content.markdown}
"""

        try:
            system_prompt = _SYSTEM_PROMPT
            lang = self.config.report.language
            if lang and lang.lower() != "english":
                system_prompt += f"\n\nIMPORTANT: Write ALL output text in {lang}. Keep paper titles, author names, and technical terms in their original language."
            output: WriterOutput = self.llm.chat_structured(
                LLMClient.build_messages(system_prompt, user_prompt),
                response_model=WriterOutput,
            )
        except Exception as exc:
            logger.error("Writer LLM call failed: %s", exc)
            return AgentResult(agent_name=self.name, success=False, error=str(exc))

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
        )
