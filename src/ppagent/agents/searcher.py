"""Searcher agent — filters and ranks papers based on user profile."""

from __future__ import annotations

import logging

from ppagent.agents import register_agent
from ppagent.agents.base import AgentBase
from ppagent.llm import LLMClient
from ppagent.models import AgentResult, Paper, SearcherOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research paper recommendation agent. Given a user's research profile and a \
list of recently published papers (with titles and abstracts), score each paper's \
relevance to the user's interests on a scale from 0.0 to 1.0.

Scoring guidelines:
- 0.9-1.0: Directly addresses the user's core research areas
- 0.7-0.89: Highly relevant to the user's specific interests
- 0.5-0.69: Somewhat related, may contain useful insights
- 0.3-0.49: Tangentially related
- 0.0-0.29: Not relevant

Score ALL papers provided. Be precise and differentiate well between papers.\
"""

_BATCH_SIZE = 20  # papers per LLM call to avoid context overflow


@register_agent
class SearcherAgent(AgentBase):
    """Filters and ranks papers based on user research profile."""

    name = "searcher"
    description = "Scores papers by relevance to the user's research profile."

    def run(
        self,
        *,
        papers: list[Paper],
        profile: str,
        threshold: float | None = None,
    ) -> AgentResult:
        threshold = threshold or self.config.search.relevance_threshold

        # Build paper summaries for the LLM
        paper_list_text = self._format_papers(papers)

        user_prompt = f"""\
## User Research Profile

{profile}

## Papers to Score

{paper_list_text}

Please score each paper's relevance to this user profile.\
"""

        try:
            output: SearcherOutput = self.llm.chat_structured(
                LLMClient.build_messages(_SYSTEM_PROMPT, user_prompt),
                response_model=SearcherOutput,
            )
        except Exception as exc:
            logger.error("Searcher LLM call failed: %s", exc)
            return AgentResult(agent_name=self.name, success=False, error=str(exc))

        # Build score lookup
        score_map = {sp.paper_id: sp for sp in output.scored_papers}

        # Filter and sort
        scored: list[tuple[Paper, float]] = []
        for paper in papers:
            sp = score_map.get(paper.id)
            if sp and sp.relevance_score >= threshold:
                scored.append((paper, sp.relevance_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        result_papers = [p for p, _ in scored]
        scores = {p.id: s for p, s in scored}

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "papers": result_papers,
                "scores": scores,
                "total_scored": len(output.scored_papers),
                "total_passed": len(result_papers),
            },
        )

    def _format_papers(self, papers: list[Paper]) -> str:
        """Format papers into a compact text list for the LLM."""
        parts = []
        for i, p in enumerate(papers, 1):
            summary = p.summary[:500] if p.summary else "No abstract available."
            parts.append(
                f"{i}. **{p.title}** (ID: {p.id})\n"
                f"   Authors: {', '.join(p.authors[:5])}{'...' if len(p.authors) > 5 else ''}\n"
                f"   Abstract: {summary}"
            )
        return "\n\n".join(parts)
