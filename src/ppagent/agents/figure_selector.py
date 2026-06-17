"""Figure selector agent — lets a vision LLM decide which figures to insert.

The LLM may return any subset of the candidate figures (including none) and
assigns each chosen figure to the report section it best illustrates. This
replaces the previous "always pick exactly one" behaviour: we no longer force
a figure into the report — the LLM decides whether, how many, and where.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ppagent.agents import register_agent
from ppagent.agents.base import AgentBase
from ppagent.agents.prompts import (
    FIGURE_SELECTOR_SYSTEM_PROMPT,
    FIGURE_SELECTOR_USER_PROMPT_TEMPLATE,
)
from ppagent.figures import FIGURE_SECTIONS, Figure, SelectedFigure
from ppagent.models import AgentResult

logger = logging.getLogger(__name__)

# Matches a JSON object containing a "selected" key (possibly wrapped in prose).
_JSON_RE = re.compile(r"\{.*\"selected\".*\}", re.IGNORECASE | re.DOTALL)


@register_agent
class FigureSelectorAgent(AgentBase):
    """Lets the vision LLM decide which figures (if any) to insert, and where."""

    name = "figure_selector"
    description = "Selects which paper figures to insert and their report section via vision LLM."

    def run(self, *, figures: list[Figure], base_dir: Path) -> AgentResult:  # type: ignore[override]
        """Select figures to insert.

        ``base_dir`` is the paper's report directory (parent of ``figures/``),
        used to resolve each figure's relative ``image_path`` to an absolute file.

        Returns an AgentResult whose ``data`` has:
            - ``selected_figures``: list[SelectedFigure] (possibly empty)
            - ``figures``: the full candidate list
            - ``none_reason``: optional string explaining an empty selection
        """
        self.llm.reset_usage()

        if not figures:
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={"selected_figures": [], "figures": [], "none_reason": None},
                usage=self.llm.get_usage(),
            )

        catalog_lines = [
            f"Figure {f.figure_number}: {f.caption}" for f in figures
        ]
        user_text = FIGURE_SELECTOR_USER_PROMPT_TEMPLATE.format(
            catalog="\n".join(catalog_lines)
        )
        image_paths = [base_dir / f.image_path for f in figures]

        try:
            raw = self.llm.chat_vision(FIGURE_SELECTOR_SYSTEM_PROMPT, user_text, image_paths)
        except Exception as exc:
            logger.warning(
                "Figure selection LLM call failed: %s — inserting no figures", exc
            )
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={
                    "selected_figures": [],
                    "figures": figures,
                    "none_reason": f"LLM call failed: {exc}",
                },
                usage=self.llm.get_usage(),
            )

        selected, none_reason = self._parse_choice(raw, figures)
        if not selected:
            logger.info(
                "Figure selector returned no figures to insert; reason: %s",
                none_reason or "not specified",
            )

        return AgentResult(
            agent_name=self.name,
            success=True,
            data={
                "selected_figures": selected,
                "figures": figures,
                "none_reason": none_reason,
            },
            usage=self.llm.get_usage(),
        )

    @staticmethod
    def _parse_choice(
        raw: str, figures: list[Figure]
    ) -> tuple[list[SelectedFigure], str | None]:
        """Extract the chosen figures + their sections from the LLM response.

        Returns (selected_figures, none_reason). Unknown figure numbers are
        dropped, unknown sections are coerced to "method", and duplicate figure
        numbers keep only the first occurrence.
        """
        if not raw:
            return [], None

        by_number = {f.figure_number: f for f in figures}

        obj: dict | None = None
        m = _JSON_RE.search(raw)
        candidate = m.group(0) if m else raw
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        if not isinstance(obj, dict):
            return [], None

        none_reason = obj.get("none_reason")
        raw_selected = obj.get("selected")
        if not isinstance(raw_selected, list):
            return [], _none_reason_str(none_reason)

        seen: set[int] = set()
        out: list[SelectedFigure] = []
        for entry in raw_selected:
            if not isinstance(entry, dict):
                continue
            try:
                num = int(entry.get("figure_number", -1))
            except (TypeError, ValueError):
                continue
            if num not in by_number or num in seen:
                continue
            section = str(entry.get("section", "method")).strip().lower()
            if section not in FIGURE_SECTIONS:
                section = "method"
            out.append(SelectedFigure(figure=by_number[num], section=section))
            seen.add(num)

        return out, _none_reason_str(none_reason)


def _none_reason_str(value: object) -> str | None:
    return str(value).strip() if isinstance(value, str) and value.strip() else None
