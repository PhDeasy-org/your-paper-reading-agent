"""Fetch and parse arXiv HTML papers into markdown text + figures.

arXiv serves an HTML rendering of most recent papers at
``https://arxiv.org/html/{paper_id}``. This module owns fetching that page,
walking it with stdlib :mod:`html.parser` to produce (a) a markdown string for
the analysis agents and (b) a list of author-provided figures (caption +
downloaded local image), each deterministically assigned to the report section
it best illustrates.

The :class:`Figure`, :class:`SelectedFigure`, and :const:`FIGURE_SECTIONS`
symbols migrated here from the deleted ``ppagent.figures`` module so the
assembler and templates keep working unchanged.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Report sections that a selected figure may be assigned to. Kept stable
# across paper types: the heading *text* varies (see WRITER_SECTION_LABELS),
# but the report always renders these section blocks.
FIGURE_SECTIONS: tuple[str, ...] = (
    "method",
    "evaluation",
    "benchmarks",
    "previous_works",
)


@dataclass
class Figure:
    """A single extracted figure from a paper (image + caption)."""

    figure_number: int
    caption: str
    image_path: str  # relative path (e.g. "figures/figure_1.png") for portability

    def __str__(self) -> str:
        return f"Figure {self.figure_number}"


@dataclass
class SelectedFigure:
    """A figure chosen for the report, with its target report section
    (one of :const:`FIGURE_SECTIONS`).
    """

    figure: Figure
    section: str

    def __str__(self) -> str:
        return f"Figure {self.figure.figure_number} → {self.section}"


# Maps a paper-section keyword (lowercased substring) to a FIGURE_SECTIONS key.
# The first matching row wins; checked in order.
_SECTION_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("previous work", "previous_works"),
    ("related", "previous_works"),
    ("prior", "previous_works"),
    ("benchmark", "benchmarks"),
    ("dataset", "benchmarks"),
    ("setup", "benchmarks"),
    ("eval", "evaluation"),
    ("result", "evaluation"),
    ("experiment", "evaluation"),
    ("finding", "evaluation"),
    ("analysis", "evaluation"),
    ("intro", "method"),
    ("method", "method"),
    ("approach", "method"),
    ("framework", "method"),
    ("model", "method"),
    ("architecture", "method"),
    ("preliminar", "method"),
)

# Strip arXiv's leading numeric prefix like "3.4 " or "1 " from a heading.
_NUMERIC_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s+")


def _normalize_section_title(title: str) -> str:
    """Lowercase + strip arXiv's leading numeric section prefix + whitespace."""
    cleaned = _NUMERIC_PREFIX_RE.sub("", title).strip().lower()
    return cleaned


def _map_section(paper_section_title: str, caption: str) -> str:
    """Deterministically map a figure to a report section.

    The paper section the figure appears in is the primary signal; caption
    keywords break ties when the section title is generic. Returns one of
    :const:`FIGURE_SECTIONS`, defaulting to ``"method"``.
    """
    section_norm = _normalize_section_title(paper_section_title)
    caption_norm = (caption or "").lower()
    # Check section title first, then caption as tiebreaker.
    for haystack in (section_norm, caption_norm):
        for keyword, target in _SECTION_KEYWORDS:
            if keyword in haystack:
                return target
    return "method"
