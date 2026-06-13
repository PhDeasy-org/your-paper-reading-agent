"""Figure selector agent — picks the best pipeline/method figure via vision LLM."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ppagent.agents import register_agent
from ppagent.agents.base import AgentBase
from ppagent.figures import Figure
from ppagent.models import AgentResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert at reading research papers. You are shown several figures "
    "extracted from a single paper, each labeled by its figure number and caption. "
    "Select the ONE figure that best represents the paper's overall method, "
    "architecture, or pipeline (i.e. an overview/framework diagram) — NOT a raw "
    "results plot or ablation chart, unless no overview figure exists.\n\n"
    "Respond with ONLY a JSON object: {\"figure_number\": <int>, \"reason\": \"<short>\"}. "
    "Do not include any other text."
)

# Matches a JSON object {"figure_number": N, ...} even if wrapped in prose/markdown.
_JSON_RE = re.compile(r"\{[^{}]*\"figure_number\"[^{}]*\}", re.IGNORECASE | re.DOTALL)


@register_agent
class FigureSelectorAgent(AgentBase):
    """Picks the most representative method/pipeline figure from candidates."""

    name = "figure_selector"
    description = "Selects the best pipeline/method figure from extracted PDF figures."

    def run(self, *, figures: list[Figure], base_dir: Path) -> AgentResult:  # type: ignore[override]
        """Select the best figure.

        ``base_dir`` is the paper's report directory (parent of ``figures/``),
        used to resolve each figure's relative ``image_path`` to an absolute file.
        """
        self.llm.reset_usage()

        if not figures:
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={"selected_figure": None, "figures": []},
                usage=self.llm.get_usage(),
            )

        # Single candidate: pick it directly, no LLM call needed.
        if len(figures) == 1:
            logger.info("Only one figure (%s); selecting without LLM call", figures[0])
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={"selected_figure": figures[0], "figures": figures},
                usage=self.llm.get_usage(),
            )

        # Build a textual catalog and resolve image file paths.
        catalog_lines = [
            f"Figure {f.figure_number}: {f.caption}" for f in figures
        ]
        user_text = (
            "Here are the figures from the paper. Choose the single best "
            "method/architecture/pipeline overview figure.\n\n"
            + "\n".join(catalog_lines)
        )
        image_paths = [base_dir / f.image_path for f in figures]

        try:
            raw = self.llm.chat_vision(_SYSTEM_PROMPT, user_text, image_paths)
        except Exception as exc:
            logger.warning("Figure selection LLM call failed: %s — defaulting to Figure 1", exc)
            fallback = self._lowest_numbered(figures)
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={"selected_figure": fallback, "figures": figures},
                usage=self.llm.get_usage(),
            )

        chosen = self._parse_choice(raw, figures)
        if chosen is None:
            logger.warning("Could not parse figure choice from LLM response: %r — defaulting to Figure 1", raw)
            chosen = self._lowest_numbered(figures)

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={"selected_figure": chosen, "figures": figures},
            usage=self.llm.get_usage(),
        )

    @staticmethod
    def _parse_choice(raw: str, figures: list[Figure]) -> Figure | None:
        """Extract the chosen figure_number from the LLM's text response."""
        if not raw:
            return None
        by_number = {f.figure_number: f for f in figures}
        # Try direct JSON parse first.
        m = _JSON_RE.search(raw)
        candidate = m.group(0) if m else raw
        try:
            obj = json.loads(candidate)
            num = int(obj.get("figure_number", -1))
            if num in by_number:
                return by_number[num]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        # Fallback: scan for the first "Figure N" mention present in our set.
        for m in re.finditer(r"Figure\s+(\d+)", raw, re.IGNORECASE):
            num = int(m.group(1))
            if num in by_number:
                return by_number[num]
        return None

    @staticmethod
    def _lowest_numbered(figures: list[Figure]) -> Figure:
        return min(figures, key=lambda f: f.figure_number)
