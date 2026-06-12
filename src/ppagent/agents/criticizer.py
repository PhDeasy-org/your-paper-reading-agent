"""Criticizer agent — strictly audits a paper and identifies limitations."""

from __future__ import annotations

import logging
from typing import Any

from ppagent.agents import register_agent
from ppagent.agents.base import AgentBase
from ppagent.llm import LLMClient
from ppagent.models import AgentResult, CriticizerOutput, PaperContent

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = r"""\
You are a rigorous, skeptical senior researcher performing a critical audit of a paper. \
Your role is to find limitations, weaknesses, and potential issues. Be thorough and honest.

IMPORTANT: Use LaTeX formatting with `$` delimiters for all inline mathematical variables, symbols, and expressions (e.g., `$x_i$`, `$\mathcal{M}$`, `$\beta$`), and `$$` delimiters for block equations. Make sure all math content is enclosed in these delimiters for proper rendering.

Evaluate the paper across these dimensions:
1. **Methodology**: Are there methodological weaknesses? Missing ablations? Unjustified \
   design choices? Is the method clearly reproducible?
2. **Experimental Design**: Are the baselines fair and comprehensive? Are there \
   missing comparisons? Are the benchmarks representative?
3. **Results & Claims**: Do the results support the claims? Are there over-claimed \
   results? Are error bars or statistical significance reported?
4. **Scope & Generalization**: How well does the method generalize? Are there \
   unstated assumptions about data, domains, or distributions?
5. **Reproducibility**: Is sufficient detail provided to reproduce the work?
6. **Ethics & Broader Impact**: Are there unaddressed ethical concerns?

Rate each finding's severity:
- **high**: Fundamental flaw that undermines the paper's core contribution
- **medium**: Notable weakness that affects the paper's reliability or scope
- **low**: Minor issue or missed opportunity that doesn't significantly impact conclusions

Be specific and cite evidence from the paper where possible.\
"""


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
    ) -> AgentResult:
        # Include writer's analysis for additional context
        writer_context = ""
        if writer_sections:
            writer_context = f"""
## Writer's Analysis Summary
- **Method**: {writer_sections.get('method', 'N/A')}
- **Evaluation**: {writer_sections.get('evaluation', 'N/A')}
- **Previous Works**: {writer_sections.get('previous_works', 'N/A')}
"""

        user_prompt = f"""\
## Paper: {content.paper.title}

**Authors**: {', '.join(content.paper.authors)}
{writer_context}
## Full Paper Content

{content.markdown}
"""

        try:
            system_prompt = _SYSTEM_PROMPT
            lang = self.config.report.language
            if lang and lang.lower() != "english":
                system_prompt += f"\n\nIMPORTANT: Write ALL output text in {lang}. Keep paper titles and technical terms in their original language."
            output: CriticizerOutput = self.llm.chat_structured(
                LLMClient.build_messages(system_prompt, user_prompt),
                response_model=CriticizerOutput,
            )
        except Exception as exc:
            logger.error("Criticizer LLM call failed: %s", exc)
            return AgentResult(agent_name=self.name, success=False, error=str(exc))

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
                "high_severity": sum(1 for f in output.findings if f.severity == "high"),
            },
        )
