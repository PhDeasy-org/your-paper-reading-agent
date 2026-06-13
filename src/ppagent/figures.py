"""Figure extraction from paper PDFs.

Extracts captioned figures (especially pipeline/method diagrams) from a paper
PDF by locating ``Figure N`` caption text and rendering the page region
immediately above each caption. This captures both raster images and
vector-drawn diagrams (the common case for pipeline figures).

Pure PyMuPDF — no LLM calls here. Selection of the "best" figure is handled
separately by the figure_selector agent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Matches a caption line like "Figure 1:", "Figure 1 |", "Fig. 1."
_CAPTION_RE = re.compile(r"^\s*(?:Figure|Fig\.?)\s*(\d+)", re.IGNORECASE)

# Render at 150 DPI for crisp inline display without huge files.
_RENDER_DPI = 150
# Half-inch margin on each side when clipping to the text column.
_X_MARGIN = 18
# Render the figure plus a thin strip of the caption line itself.
_CAPTION_STRIP = 14

_MAX_FIGURES = 8
# Minimum figure height (points) to keep — filters out stray tiny matches.
_MIN_FIGURE_HEIGHT = 60


@dataclass
class Figure:
    """A single extracted figure from a paper PDF."""

    figure_number: int
    caption: str
    image_path: str  # relative path (e.g. "figures/figure_1.png") for portability

    def __str__(self) -> str:
        return f"Figure {self.figure_number}"


def _find_caption_lines(page: fitz.Page) -> list[tuple[int, fitz.Rect, str]]:
    """Return (figure_number, bbox, full_caption_text) for each caption on the page."""
    results: list[tuple[int, fitz.Rect, str]] = []
    try:
        d = page.get_text("dict")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("get_text('dict') failed on page: %s", exc)
        return results

    for block in d.get("blocks", []):
        if block.get("type") != 0:  # 0 == text
            continue
        # Join all spans in a line; captions often span multiple lines.
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            m = _CAPTION_RE.match(text)
            if not m:
                continue
            # Captions are usually small/italic; grab the full line bbox.
            bbox = fitz.Rect(line["bbox"])
            # Try to extend the caption text by walking forward through
            # subsequent lines in the same block (multi-line captions).
            full = text
            # Use block lines index to extend; collect following lines until a gap or new block.
            results.append((int(m.group(1)), bbox, full))
    return results


def _figure_region(page: fitz.Page, caption_bbox: fitz.Rect) -> fitz.Rect | None:
    """Compute the figure's clip rect: from the largest vertical gap above the
    caption down to the caption line.

    Figures occupy the vertical whitespace between the preceding paragraph and
    their caption, so we look for the biggest gap in text above the caption and
    treat everything from the end of that gap down to the caption as the figure.
    """
    try:
        d = page.get_text("dict")
    except Exception:  # pragma: no cover - defensive
        return None

    page_rect = page.rect
    caption_y = caption_bbox.y0
    # Collect y-ranges of all text lines above the caption (and below top margin).
    line_ys: list[tuple[float, float]] = []
    top_margin = 50
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            r = fitz.Rect(line["bbox"])
            if r.y1 <= caption_y and r.y0 > top_margin:
                line_ys.append((r.y0, r.y1))

    if not line_ys:
        # No text above; the figure likely starts near the top of the page.
        fig_top = top_margin
    else:
        line_ys.sort()
        # Find the largest gap between consecutive lines' y1 and next y0.
        prev_end: float | None = None
        max_gap = 0.0
        gap_end = line_ys[0][0]
        for y0, y1 in line_ys:
            if prev_end is not None:
                g = y0 - prev_end
                if g > max_gap:
                    max_gap = g
                    gap_end = y0
            prev_end = max(prev_end if prev_end is not None else y1, y1)
        # If the largest gap is just above the caption region, use it; otherwise
        # fall back to the first line above the caption (whole region is figure).
        fig_top = gap_end if max_gap >= 20 else line_ys[0][0]

    # Clip to text-column width (most papers are single-column here).
    x0 = max(page_rect.x0, caption_bbox.x0 - _X_MARGIN)
    x1 = min(page_rect.x1, caption_bbox.x1 + _X_MARGIN)
    # Include a small strip of the caption so the label is visible in the crop.
    y1 = caption_bbox.y1
    if y1 - fig_top < _MIN_FIGURE_HEIGHT:
        # Region too small to be a real figure (e.g. inline "Figure" mention).
        return None
    return fitz.Rect(x0, fig_top, x1, y1)


def extract_figures(pdf_path: Path, out_dir: Path) -> list[Figure]:
    """Extract captioned figures from a paper PDF.

    Renders each figure region to ``out_dir/figure_{N}.png`` and returns Figure
    metadata with *relative* ``image_path`` values (``figures/figure_N.png``) so
    the generated report stays portable.

    Returns an empty list if the PDF cannot be opened or no captions are found.
    """
    if not pdf_path.exists():
        logger.warning("PDF not found for figure extraction: %s", pdf_path)
        return []

    figures: list[Figure] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.warning("Could not open PDF for figures (%s): %s", pdf_path, exc)
        return []

    try:
        seen_numbers: set[int] = set()
        zoom = _RENDER_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            for fig_num, cap_bbox, cap_text in _find_caption_lines(page):
                if fig_num in seen_numbers:
                    continue
                region = _figure_region(page, cap_bbox)
                if region is None:
                    continue
                try:
                    pix = page.get_pixmap(clip=region, matrix=mat)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Pixmap render failed for figure %d: %s", fig_num, exc)
                    continue
                # `out_dir` is the paper's report directory; images go into
                # out_dir/figures/figure_N.png and are referenced by the
                # relative path "figures/figure_N.png" so the report stays
                # portable (HTML/MD live in out_dir too).
                rel_path = f"figures/figure_{fig_num}.png"
                figures_subdir = out_dir / "figures"
                figures_subdir.mkdir(parents=True, exist_ok=True)
                abs_path = figures_subdir / f"figure_{fig_num}.png"
                pix.save(str(abs_path))
                seen_numbers.add(fig_num)
                figures.append(
                    Figure(
                        figure_number=fig_num,
                        caption=cap_text,
                        image_path=rel_path,
                    )
                )
                logger.info(
                    "Extracted figure %d → %s (%dx%d)",
                    fig_num,
                    rel_path,
                    pix.width,
                    pix.height,
                )
                if len(figures) >= _MAX_FIGURES:
                    return figures
    finally:
        doc.close()

    return figures
