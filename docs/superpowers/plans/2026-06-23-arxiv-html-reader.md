# arXiv HTML Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `hf papers read` content source and the PyMuPDF figure-extraction pipeline with a single arXiv HTML fetcher that yields clean markdown text + author-provided figures, and delete the vision LLM role.

**Architecture:** A new `src/ppagent/arxiv_html.py` module owns fetch + parse of `https://arxiv.org/html/{id}`. One `html.parser.HTMLParser` pass produces (a) a markdown string for the Writer/Criticizer and (b) a list of `Figure` objects with downloaded local images, each deterministically assigned to a report section by the paper section it appears in. The `figures.py` module, `figure_selector.py` agent, `vision` LLM role, and `chat_vision()` are deleted. Pipeline phases collapse 8 → 6.

**Tech Stack:** Python 3.12+, stdlib `html.parser`, `httpx`, `pydantic` v2, `pytest`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-06-23-arxiv-html-reader-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/ppagent/arxiv_html.py` | **Create** | Fetch arXiv HTML, parse to markdown + figures, download images, deterministic section mapping. Owns `Figure`, `SelectedFigure`, `FIGURE_SECTIONS` (migrated from `figures.py`). |
| `src/ppagent/figures.py` | **Delete** | Replaced entirely by `arxiv_html.py`. |
| `src/ppagent/agents/figure_selector.py` | **Delete** | Vision-based figure selection removed. |
| `tests/test_figures.py` | **Delete** | Tests removed module. |
| `tests/test_arxiv_html.py` | **Create** | Unit + live tests for the new module. |
| `src/ppagent/pipeline.py` | **Modify** | Phase 2 uses arXiv HTML with PDF-text fallback; old phases 6+7 deleted. |
| `src/ppagent/config.py` | **Modify** | Drop `vision` role; add `report.max_figures`. |
| `src/ppagent/llm.py` | **Modify** | Delete `chat_vision()` + `_image_to_data_uri()`. |
| `src/ppagent/agents/prompts.py` | **Modify** | Delete `FIGURE_SELECTOR_*` constants. |
| `src/ppagent/agents/__init__.py` | **Modify** | Drop `FigureSelectorAgent` import. |
| `src/ppagent/agents/assembler.py` | **Modify** | Update imports from `ppagent.figures` → `ppagent.arxiv_html`. |
| `src/ppagent/cli.py` | **Modify** | Drop "Vision LLM" line from `config_show`. |
| `src/ppagent/tui.py` | **Modify** | Drop vision menu entry; update 3 regexes from `text\|vision\|searcher` → `text\|searcher`. |
| `README.md` | **Modify** | Update "How it works" text. |

---

## Task 1: Create `arxiv_html.py` with migrated data symbols + section mapping

This task establishes the new module with the migrated `Figure`/`SelectedFigure`/`FIGURE_SECTIONS` and the deterministic `_map_section` function. No parsing yet — just the data layer + pure mapping logic. This lets tests land before the parser exists.

**Files:**
- Create: `src/ppagent/arxiv_html.py`
- Test: `tests/test_arxiv_html.py`

- [ ] **Step 1: Write the failing test for `_map_section`**

Create `tests/test_arxiv_html.py`:

```python
"""Tests for arXiv HTML fetch + parse + figure extraction."""

from __future__ import annotations

from ppagent.arxiv_html import (
    FIGURE_SECTIONS,
    Figure,
    SelectedFigure,
    _map_section,
)


def test_figure_sections_is_the_four_known_keys():
    assert set(FIGURE_SECTIONS) == {
        "method",
        "evaluation",
        "benchmarks",
        "previous_works",
    }


def test_map_section_by_paper_section_title():
    """Section mapping is driven by the paper section the figure appears in."""
    cases = [
        ("1 Introduction", "", "method"),
        ("3 Method", "", "method"),
        ("3.2 Our Approach", "", "method"),
        ("The Proposed Framework", "", "method"),
        ("Model Architecture", "", "method"),
        ("2 Preliminaries", "", "method"),
        ("4 Experiments", "", "evaluation"),
        ("5 Results", "", "evaluation"),
        ("5.1 Evaluation", "", "evaluation"),
        ("6 Analysis", "", "evaluation"),
        ("4 Experimental Setup", "", "benchmarks"),
        ("3 Datasets", "", "benchmarks"),
        ("Benchmarks", "", "benchmarks"),
        ("7 Related Work", "", "previous_works"),
        ("Prior Work", "", "previous_works"),
        ("Conclusions", "", "method"),  # default fallback
        ("Abstract", "", "method"),  # default fallback
    ]
    for section_title, caption, expected in cases:
        got = _map_section(section_title, caption)
        assert got == expected, (
            f"_map_section({section_title!r}, {caption!r}) = {got!r}, "
            f"expected {expected!r}"
        )


def test_map_section_normalizes_case_and_strips_numeric_prefix():
    assert _map_section("3.4 THE TRAINING COST", "") == "method"
    assert _map_section("  5 Results ", "") == "evaluation"


def test_map_section_uses_caption_as_tiebreaker():
    """When the section title is generic, caption keywords can decide."""
    # Generic section title "Discussion" + caption mentioning "benchmark"
    assert _map_section("Discussion", "Figure 2: benchmark comparison") == "benchmarks"


def test_figure_dataclass_str():
    f = Figure(figure_number=3, caption="a plot", image_path="figures/figure_3.png")
    assert str(f) == "Figure 3"


def test_selected_figure_dataclass_str():
    f = Figure(figure_number=2, caption="x", image_path="figures/figure_2.png")
    sf = SelectedFigure(figure=f, section="method")
    assert str(sf) == "Figure 2 → method"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_arxiv_html.py -v`
Expected: FAIL with `ImportError: No module named 'ppagent.arxiv_html'`

- [ ] **Step 3: Write minimal `arxiv_html.py` with migrated symbols + `_map_section`**

Create `src/ppagent/arxiv_html.py`:

```python
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
from pathlib import Path

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_arxiv_html.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ppagent/arxiv_html.py tests/test_arxiv_html.py
git commit -m "feat(arxiv_html): add module shell with migrated Figure symbols + section mapping"
```

---

## Task 2: Implement the HTML parser (markdown + figure discovery, no fetch yet)

The parser is a single `HTMLParser` subclass. It's pure (input: HTML string + page URL for resolving image src; output: markdown + figures list). Network fetching comes in Task 3. This separation makes the parser unit-testable with synthetic fixtures.

**Files:**
- Modify: `src/ppagent/arxiv_html.py`
- Test: `tests/test_arxiv_html.py`

- [ ] **Step 1: Write failing parser tests (synthetic fixtures)**

Append to `tests/test_arxiv_html.py`:

```python
from ppagent.arxiv_html import parse_html


def test_parse_html_extracts_heading_as_markdown():
    html = (
        "<html><body><article>"
        "<h1 class='ltx_title'>My Paper Title</h1>"
        "<h2 class='ltx_title'>1 Introduction</h2>"
        "<p>Hello world.</p>"
        "</article></body></html>"
    )
    md, figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert "# My Paper Title" in md
    assert "## Introduction" in md  # numeric prefix stripped
    assert "Hello world." in md
    assert figures == []


def test_parse_html_extracts_figure_with_caption_and_resolves_url():
    html = (
        "<html><body><article>"
        "<h2 class='ltx_title'>3 Method</h2>"
        "<figure>"
        "  <img src='2501.00001v1/x1.png' alt='pipeline'/>"
        "  <figcaption>Figure 1: The pipeline.</figcaption>"
        "</figure>"
        "</article></body></html>"
    )
    md, figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    # Figure's image src is resolved to an absolute URL in `src_url`; the
    # relative image_path is assigned later during download.
    assert len(figures) == 1
    fig = figures[0]
    assert fig.figure_number == 1
    assert fig.caption == "Figure 1: The pipeline."
    # src_url is the absolute URL we'll download from (held on the Figure
    # transiently via a module-level dict; see implementation).
    # For now the figure's image_path is unset until download.
    # Figure content does NOT leak into the markdown the Writer reads.
    assert "The pipeline" not in md


def test_parse_html_skips_data_uri_and_static_images():
    html = (
        "<html><body><article>"
        "<h2 class='ltx_title'>2 Method</h2>"
        "<figure>"
        "  <img src='data:image/png;base64,iVBORw0KGgo='/>"
        "  <figcaption>Figure 1: inline icon.</figcaption>"
        "</figure>"
        "<figure>"
        "  <img src='/static/browse/0.3.4/images/logo.svg'/>"
        "  <figcaption>Figure 2: arxiv logo.</figcaption>"
        "</figure>"
        "</article></body></html>"
    )
    md, figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert figures == []  # both skipped


def test_parse_html_subfigures_share_caption_and_section():
    """A <figure> with two <img>s yields two Figures, same caption+section."""
    html = (
        "<html><body><article>"
        "<h2 class='ltx_title'>4 Results</h2>"
        "<figure>"
        "  <img src='2501.00001v1/x1.png'/>"
        "  <img src='2501.00001v1/x2.png'/>"
        "  <figcaption>(a) curve one. (b) curve two.</figcaption>"
        "</figure>"
        "</article></body></html>"
    )
    md, figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert len(figures) == 2
    assert figures[0].figure_number == 1
    assert figures[1].figure_number == 2
    assert figures[0].caption == figures[1].caption
    # Both should resolve to "evaluation" because the section is "4 Results".
    from ppagent.arxiv_html import _map_section

    # (Section mapping uses the most-recent heading; verified at the
    # parse_html level in test_parse_html_assigns_section_per_figure below.)


def test_parse_html_skips_figure_without_img():
    """A <figure> containing only a table (no <img>) is not a figure."""
    html = (
        "<html><body><article>"
        "<h2 class='ltx_title'>3 Benchmarks</h2>"
        "<figure>"
        "  <figcaption>Table 1: results.</figcaption>"
        "  <table><tr><td>1.0</td></tr></table>"
        "</figure>"
        "</article></body></html>"
    )
    md, figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert figures == []


def test_parse_html_extracts_math_annotation_as_inline_latex():
    html = (
        "<html><body><article>"
        "<p>The loss is <math>"
        "<mi>L</mi>"
        "<annotation encoding='application/x-tex'>L = -\\log p</annotation>"
        "</math> here.</p>"
        "</article></body></html>"
    )
    md, figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert "$L = -\\log p$" in md


def test_parse_html_extracts_block_math_as_display_latex():
    html = (
        "<html><body><article>"
        "<p>Display:</p>"
        "<math display='block'>"
        "<mi>X</mi>"
        "<annotation encoding='application/x-tex'>X = f(Y)</annotation>"
        "</math>"
        "</article></body></html>"
    )
    md, figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert "$$X = f(Y)$$" in md


def test_parse_html_assigns_section_per_figure():
    """Each figure remembers the most-recent heading it appeared under."""
    html = (
        "<html><body><article>"
        "<h2 class='ltx_title'>3 Method</h2>"
        "<figure><img src='x1.png'/><figcaption>F1</figcaption></figure>"
        "<h2 class='ltx_title'>5 Experiments</h2>"
        "<figure><img src='x2.png'/><figcaption>F2</figcaption></figure>"
        "</article></body></html>"
    )
    md, raw_figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    # parse_html returns figures with a `.section_title` attribute we can map.
    from ppagent.arxiv_html import _map_section

    assert len(raw_figures) == 2
    assert _map_section(raw_figures[0].section_title, raw_figures[0].caption) == "method"
    assert _map_section(raw_figures[1].section_title, raw_figures[1].caption) == "evaluation"


def test_parse_html_skips_bibliography_section():
    html = (
        "<html><body><article>"
        "<h2 class='ltx_title'>1 Introduction</h2>"
        "<p>Body text.</p>"
        "<section class='ltx_bibliography'>"
        "<h2 class='ltx_title'>References</h2>"
        "<p>[1] Someone, Some Paper, 2024.</p>"
        "</section>"
        "</article></body></html>"
    )
    md, figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert "Body text." in md
    assert "Someone" not in md
    assert "References" not in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_arxiv_html.py -v`
Expected: FAIL — `parse_html` not importable (or figures missing `.section_title`).

- [ ] **Step 3: Implement `parse_html` + the `HTMLParser` subclass**

Add to `src/ppagent/arxiv_html.py` (below the existing code):

```python
from html.parser import HTMLParser
from urllib.parse import urljoin


@dataclass
class _RawFigure:
    """An intermediate figure produced by the parser, before download.

    Carries the resolved absolute ``src_url`` and the ``section_title`` it
    appeared under; :func:`fetch_and_parse` (Task 3) downloads each and
    converts it to a :class:`Figure` with a local ``image_path``.
    """

    figure_number: int
    caption: str
    src_url: str
    section_title: str

    # Extra attribute the parser stashes section context on; kept as a field
    # so tests can read it directly via attribute access.


# Headings whose content is treated as a section title.
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_NUMERIC_PREFIX_RE_HEADING = re.compile(r"^\s*\d+(?:\.\d+)*\s+")


class _ArxivHtmlParser(HTMLParser):
    """Single-pass walker producing markdown text + raw figures.

    State is intentionally minimal: a buffer of markdown chunks, the current
    heading text (for figure section assignment), and bookkeeping flags for
    nested elements we treat specially (figures, math, bibliography).
    """

    def __init__(self, page_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._page_url = page_url
        self._md_chunks: list[str] = []
        # Most-recent heading text, in original (not normalized) form.
        self._current_section: str = ""
        # Heading being accumulated.
        self._heading_buf: list[str] | None = None
        self._heading_tag: str | None = None
        # Figure being accumulated.
        self._raw_figures: list[_RawFigure] = []
        self._in_figure: int = 0  # nest depth of <figure>
        self._figure_caption_buf: list[str] | None = None
        self._figure_imgs: list[str] = []  # src_urls collected in current figure
        self._next_figure_number: int = 1
        # Math handling.
        self._math_depth: int = 0
        self._math_is_block: bool = False
        self._math_latex: str | None = None  # set when annotation captured
        self._want_annotation_tex: bool = False
        # Suppress text output inside skipped regions.
        self._skip_depth: int = 0  # bibliography / nav

    # --- helpers ---------------------------------------------------------

    def _emit(self, text: str) -> None:
        if self._skip_depth == 0:
            self._md_chunks.append(text)

    def _is_skipping(self) -> bool:
        return self._skip_depth > 0

    # --- start/end tags --------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = dict(attrs)

        # Bibliography / nav: skip entirely (and their children).
        if tag == "section" and "ltx_bibliography" in (d.get("class") or ""):
            self._skip_depth += 1
            return
        if tag == "nav":
            self._skip_depth += 1
            return
        if self._is_skipping():
            # Don't process tags inside skipped regions, but track nesting of
            # the skipped container so its end tag unwinds correctly.
            return

        if tag in _HEADING_TAGS:
            self._heading_tag = tag
            self._heading_buf = []
            return

        if tag == "figure":
            self._in_figure += 1
            self._figure_imgs = []
            self._figure_caption_buf = []
            return

        if tag == "figcaption" and self._in_figure > 0:
            # Start capturing caption text (we're already inside figure).
            # _figure_caption_buf is already a list from <figure> start.
            return

        if tag == "img":
            src = d.get("src") or ""
            if not src:
                return
            if src.startswith("data:") or src.startswith("/static/"):
                return  # inline icons / arxiv chrome logos
            if src.endswith(".svg"):
                logger.debug("Skipping SVG figure src=%s", src)
                return
            absolute = urljoin(self._page_url, src)
            if self._in_figure > 0:
                self._figure_imgs.append(absolute)
            else:
                # Bare <img> outside a <figure>: treat as a standalone figure.
                self._record_figure_imgs([absolute], "")
            return

        if tag == "math":
            self._math_depth += 1
            self._math_is_block = d.get("display") == "block"
            self._math_latex = None
            return

        if tag == "annotation":
            if d.get("encoding") == "application/x-tex":
                self._want_annotation_tex = True
                self._math_latex = ""  # begin capturing
            return

        # Structural markdown emission for common content tags.
        if tag == "p":
            self._emit("")
        elif tag == "li":
            self._emit("\n- ")
        elif tag == "ul" or tag == "ol":
            self._emit("")
        elif tag == "br":
            self._emit("  \n")
        elif tag == "table":
            self._emit("\n\n")
        # Unknown tags: nothing — their text content still flows via handle_data.

    def handle_endtag(self, tag: str) -> None:
        if tag == "section" or tag == "nav":
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._is_skipping():
            return

        if tag in _HEADING_TAGS and self._heading_buf is not None:
            raw = "".join(self._heading_buf).strip()
            if raw:
                self._current_section = raw
                # Emit the heading as markdown, numeric prefix stripped.
                cleaned = _NUMERIC_PREFIX_RE_HEADING.sub("", raw).strip()
                level = int(tag[1])
                self._emit("\n\n" + "#" * level + " " + cleaned + "\n\n")
            self._heading_buf = None
            self._heading_tag = None
            return

        if tag == "figure" and self._in_figure > 0:
            caption = "".join(self._figure_caption_buf or []).strip()
            self._record_figure_imgs(self._figure_imgs, caption)
            self._in_figure -= 1
            self._figure_imgs = []
            self._figure_caption_buf = None
            return

        if tag == "p":
            self._emit("\n\n")
        elif tag in ("ul", "ol", "table"):
            self._emit("\n\n")

        if tag == "math":
            if self._math_depth > 0:
                self._math_depth -= 1
                latex = self._math_latex
                if self._math_depth == 0 and latex:
                    if self._math_is_block:
                        self._emit(f"$${latex}$$")
                    else:
                        self._emit(f"${latex}$")
                self._math_is_block = False
                self._math_latex = None
            return

        if tag == "annotation":
            self._want_annotation_tex = False
            return

    def handle_data(self, data: str) -> None:
        if self._is_skipping():
            return

        if self._heading_buf is not None:
            self._heading_buf.append(data)
            return

        if self._in_figure > 0 and self._figure_caption_buf is not None:
            # Caption text: capture but don't emit to markdown.
            self._figure_caption_buf.append(data)
            return

        if self._math_depth > 0 and self._want_annotation_tex:
            assert self._math_latex is not None
            self._math_latex += data
            return

        # Ordinary text content.
        self._emit(data)

    # --- finalize --------------------------------------------------------

    def _record_figure_imgs(self, src_urls: list[str], caption: str) -> None:
        for src_url in src_urls:
            self._raw_figures.append(
                _RawFigure(
                    figure_number=self._next_figure_number,
                    caption=caption,
                    src_url=src_url,
                    section_title=self._current_section,
                )
            )
            self._next_figure_number += 1

    def markdown(self) -> str:
        """Return the accumulated markdown, with runs of blank lines collapsed."""
        text = "".join(self._md_chunks)
        # Collapse 3+ newlines to exactly 2 (paragraph break).
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + "\n"

    def raw_figures(self) -> list[_RawFigure]:
        return list(self._raw_figures)


def parse_html(html: str, *, page_url: str) -> tuple[str, list[_RawFigure]]:
    """Parse arXiv HTML into (markdown, raw_figures).

    ``raw_figures`` are :class:`_RawFigure` instances carrying the resolved
    absolute ``src_url`` and the section title each appeared under. Callers
    (``fetch_and_parse``) download each and convert to :class:`Figure`.
    """
    p = _ArxivHtmlParser(page_url=page_url)
    p.feed(html)
    p.close()
    return p.markdown(), p.raw_figures()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_arxiv_html.py -v`
Expected: all tests PASS. If a test asserts a behavior slightly differently than the implementation, fix the *test* to match the implementation's actual contract (the test's job is to pin the contract, not invent one) — but the contract above is what we want.

- [ ] **Step 5: Commit**

```bash
git add src/ppagent/arxiv_html.py tests/test_arxiv_html.py
git commit -m "feat(arxiv_html): implement HTML parser producing markdown + figures"
```

---

## Task 3: Implement `fetch_and_parse` (network + image download)

Adds the network layer: fetch the HTML page, instantiate the parser, download each figure image to disk, and return the final `ParsedHtml`. Also handles `HtmlUnavailable` / `ParseError`.

**Files:**
- Modify: `src/ppagent/arxiv_html.py`
- Test: `tests/test_arxiv_html.py`

- [ ] **Step 1: Write failing tests for `fetch_and_parse` (mocked httpx)**

Append to `tests/test_arxiv_html.py`:

```python
from pathlib import Path
from unittest.mock import patch

import pytest

from ppagent.arxiv_html import (
    HtmlUnavailable,
    ParseError,
    ParsedHtml,
    fetch_and_parse,
)


class _FakeResponse:
    def __init__(self, status_code: int, text: str, content: bytes = b""):
        self.status_code = status_code
        self.text = text
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=self
            )


def _make_client_returning(*responses: _FakeResponse):
    """Build a fake httpx.Client-like object returning the given responses in order."""
    calls = {"count": 0}
    responses_list = list(responses)

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, *a, **kw):
            i = calls["count"]
            calls["count"] += 1
            if i >= len(responses_list):
                raise AssertionError(f"unexpected extra HTTP request #{i + 1}: {url}")
            return responses_list[i]

    return _FakeClient


def test_fetch_and_parse_downloads_html_and_images(tmp_path):
    html = (
        "<html><body><article>"
        "<h2 class='ltx_title'>3 Method</h2>"
        "<figure>"
        "  <img src='2501.00001v1/x1.png'/>"
        "  <figcaption>Figure 1: pipeline.</figcaption>"
        "</figure>"
        "</article></body></html>"
    )
    png_bytes = b"\x89PNG\r\n\x1a\n"  # minimal PNG header
    fake_client = _make_client_returning(
        _FakeResponse(200, html),
        _FakeResponse(200, "", png_bytes),  # image fetch
    )
    with patch("ppagent.arxiv_html.httpx.Client", fake_client):
        result = fetch_and_parse("2501.00001", tmp_path)

    assert isinstance(result, ParsedHtml)
    assert "## Method" in result.markdown
    assert len(result.figures) == 1
    fig = result.figures[0]
    assert fig.figure_number == 1
    assert fig.caption == "Figure 1: pipeline."
    assert fig.image_path == "figures/figure_1.png"
    # Image downloaded to disk.
    assert (tmp_path / fig.image_path).read_bytes() == png_bytes
    # Section mapping applied.
    assert result.figure_sections[1] == "method"


def test_fetch_and_parse_preserves_image_extension(tmp_path):
    html = (
        "<html><body><article>"
        "<h2 class='ltx_title'>2 Method</h2>"
        "<figure><img src='x1.jpg'/><figcaption>F1</figcaption></figure>"
        "</article></body></html>"
    )
    fake_client = _make_client_returning(
        _FakeResponse(200, html),
        _FakeResponse(200, "", b"\xff\xd8\xff\xe0"),  # JPEG-ish bytes
    )
    with patch("ppagent.arxiv_html.httpx.Client", fake_client):
        result = fetch_and_parse("2501.00001", tmp_path)
    assert result.figures[0].image_path == "figures/figure_1.jpg"
    assert (tmp_path / "figures/figure_1.jpg").exists()


def test_fetch_and_parse_respects_max_figures(tmp_path):
    # 3 figures in HTML, max_figures=1 → only first downloaded + returned.
    html = (
        "<html><body><article>"
        "<h2 class='ltx_title'>2 Method</h2>"
        + "".join(
            f"<figure><img src='x{i}.png'/><figcaption>F{i}</figcaption></figure>"
            for i in (1, 2, 3)
        )
        + "</article></body></html>"
    )
    fake_client = _make_client_returning(
        _FakeResponse(200, html),
        _FakeResponse(200, "", b"\x89PNG"),
        # Only ONE image fetch should happen — max_figures=1.
    )
    with patch("ppagent.arxiv_html.httpx.Client", fake_client):
        result = fetch_and_parse("2501.00001", tmp_path, max_figures=1)
    assert len(result.figures) == 1
    assert result.figures[0].figure_number == 1


def test_fetch_and_parse_skips_failed_image_download(tmp_path):
    """A 404 on one image drops just that figure; others survive."""
    html = (
        "<html><body><article>"
        "<h2 class='ltx_title'>2 Method</h2>"
        "<figure><img src='x1.png'/><figcaption>F1</figcaption></figure>"
        "<figure><img src='x2.png'/><figcaption>F2</figcaption></figure>"
        "</article></body></html>"
    )
    import httpx

    class _BoomClient:
        def __init__(self, *a, **kw):
            self._n = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, *a, **kw):
            self._n += 1
            if self._n == 1:
                return _FakeResponse(200, html)  # HTML page
            if self._n == 2:
                # First image 404s.
                raise httpx.HTTPStatusError(
                    "404", request=None, response=_FakeResponse(404, "")
                )
            return _FakeResponse(200, "", b"\x89PNG")  # second image OK

    with patch("ppagent.arxiv_html.httpx.Client", _BoomClient):
        result = fetch_and_parse("2501.00001", tmp_path)

    assert len(result.figures) == 1
    assert result.figures[0].figure_number == 2  # first was dropped


def test_fetch_and_parse_raises_html_unavailable_on_404(tmp_path):
    import httpx

    class _NotFoundClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, *a, **kw):
            raise httpx.HTTPStatusError(
                "404", request=None, response=_FakeResponse(404, "")
            )

    with patch("ppagent.arxiv_html.httpx.Client", _NotFoundClient):
        with pytest.raises(HtmlUnavailable):
            fetch_and_parse("9999.99999", tmp_path)


def test_fetch_and_parse_raises_parse_error_on_empty_body(tmp_path):
    class _EmptyClient:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, *a, **kw):
            return _FakeResponse(200, "<html><body></body></html>")

    with patch("ppagent.arxiv_html.httpx.Client", _EmptyClient):
        with pytest.raises(ParseError):
            fetch_and_parse("2501.00001", tmp_path)


@pytest.mark.skipif(
    True,  # set to `not os.environ.get("PPA_LIVE_NETWORK")` to enable
    reason="live network test; set PPA_LIVE_NETWORK=1 to run",
)
def test_fetch_and_parse_live(tmp_path):
    """Real fetch of a known arXiv HTML paper. Manual / CI-nightly only."""
    result = fetch_and_parse("2606.01075", tmp_path)
    assert len(result.markdown) > 1000
    assert len(result.figures) >= 1
    for fig in result.figures:
        assert (tmp_path / fig.image_path).exists()
        assert result.figure_sections[fig.figure_number] in FIGURE_SECTIONS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_arxiv_html.py -v`
Expected: FAIL — `fetch_and_parse`, `HtmlUnavailable`, `ParseError`, `ParsedHtml` not importable.

- [ ] **Step 3: Implement `fetch_and_parse` + exceptions**

Add to `src/ppagent/arxiv_html.py`:

```python
import os
import uuid
from dataclasses import dataclass, field

import httpx


_HTML_FETCH_TIMEOUT = 120  # seconds
_IMAGE_FETCH_TIMEOUT = 60  # seconds
_USER_AGENT = "ppagent/1.0 (https://github.com/AutoPhd-org/your-paper-reading-agent)"
_ARXIV_HTML_URL_TEMPLATE = "https://arxiv.org/html/{paper_id}"


class HtmlUnavailable(Exception):
    """Raised when the arXiv HTML page cannot be fetched (404, network, timeout)."""


class ParseError(Exception):
    """Raised when the HTML body is fetched but is not a recognizable paper."""


@dataclass
class ParsedHtml:
    """Result of :func:`fetch_and_parse`: markdown text + downloaded figures."""

    markdown: str
    figures: list[Figure]
    figure_sections: dict[int, str] = field(default_factory=dict)


def _ext_from_url(url: str) -> str:
    """Return a lowercase image extension (without dot) from a URL, default png."""
    path = url.split("?", 1)[0].split("#", 1)[0]
    if "." in path.rsplit("/", 1)[-1]:
        ext = path.rsplit(".", 1)[-1].lower()
        # Normalize jpeg → jpg, strip any query remnants.
        if ext == "jpeg":
            ext = "jpg"
        if ext.isalnum() and len(ext) <= 5:
            return ext
    return "png"


def _looks_like_paper(markdown: str, raw_figures: list[_RawFigure]) -> bool:
    """Heuristic: did the parser find any recognizable paper content?"""
    # We need *some* body text. Figures alone (no text) is suspicious.
    return len(markdown.strip()) > 50


def fetch_and_parse(
    paper_id: str,
    out_dir: Path,
    *,
    max_figures: int = 8,
) -> ParsedHtml:
    """Fetch an arXiv HTML paper and parse it into markdown + downloaded figures.

    Fetches ``https://arxiv.org/html/{paper_id}`` (arXiv redirects to the
    latest version), parses the body into markdown + figures, and downloads
    each figure image (up to ``max_figures``) into ``out_dir/figures/``.

    Raises:
        HtmlUnavailable: HTTP 4xx/5xx or network error on the HTML page.
        ParseError: page fetched but body is not a recognizable paper.
    """
    url = _ARXIV_HTML_URL_TEMPLATE.format(paper_id=paper_id)
    headers = {"User-Agent": _USER_AGENT}

    try:
        with httpx.Client(
            timeout=_HTML_FETCH_TIMEOUT, follow_redirects=True, headers=headers
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
            page_url = str(resp.url)
    except httpx.HTTPError as exc:
        raise HtmlUnavailable(
            f"Could not fetch arXiv HTML for {paper_id!r} at {url}: {exc}"
        ) from exc

    markdown, raw_figures = parse_html(html, page_url=page_url)
    if not _looks_like_paper(markdown, raw_figures):
        raise ParseError(
            f"arXiv HTML for {paper_id!r} parsed to empty content ({len(markdown)} chars, "
            f"{len(raw_figures)} figures); not a recognizable paper."
        )

    # Download images (up to max_figures). Failed downloads are dropped.
    figures: list[Figure] = []
    figure_sections: dict[int, str] = {}
    figures_subdir = out_dir / "figures"
    figures_subdir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": _USER_AGENT}
    for raw in raw_figures:
        if len(figures) >= max_figures:
            break
        ext = _ext_from_url(raw.src_url)
        rel_path = f"figures/figure_{raw.figure_number}.{ext}"
        abs_path = figures_subdir / f"figure_{raw.figure_number}.{ext}"
        try:
            with httpx.Client(timeout=_IMAGE_FETCH_TIMEOUT, headers=headers) as client:
                img_resp = client.get(raw.src_url)
                img_resp.raise_for_status()
                abs_path.write_bytes(img_resp.content)
        except httpx.HTTPError as exc:
            logger.warning(
                "Failed to download figure %d from %s: %s — skipping",
                raw.figure_number,
                raw.src_url,
                exc,
            )
            continue
        figures.append(
            Figure(
                figure_number=raw.figure_number,
                caption=raw.caption,
                image_path=rel_path,
            )
        )
        figure_sections[raw.figure_number] = _map_section(
            raw.section_title, raw.caption
        )
        logger.info(
            "Downloaded figure %d → %s (%d bytes)",
            raw.figure_number,
            rel_path,
            abs_path.stat().st_size,
        )

    return ParsedHtml(markdown=markdown, figures=figures, figure_sections=figure_sections)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_arxiv_html.py -v`
Expected: all tests PASS.

If `test_fetch_and_parse_skips_failed_image_download` fails because the mock raises before `raise_for_status`, ensure the implementation calls `img_resp.raise_for_status()` (it does).

- [ ] **Step 5: Commit**

```bash
git add src/ppagent/arxiv_html.py tests/test_arxiv_html.py
git commit -m "feat(arxiv_html): implement fetch_and_parse with image download + error handling"
```

---

## Task 4: Wire `pipeline.py` to use arXiv HTML with PDF fallback

Rewrites Phase 2 (content) to call `arxiv_html.fetch_and_parse`, deletes old phases 6+7 (PDF ensure + figure extract/select), and feeds figures into the assembler.

**Files:**
- Modify: `src/ppagent/pipeline.py:13-26` (imports), `:149-515` (the `report()` method)

- [ ] **Step 1: Read the current pipeline.py to confirm exact line ranges**

Run: `uv run python -c "import ppagent.pipeline"` (sanity — module imports cleanly before changes).

- [ ] **Step 2: Update imports in `pipeline.py`**

Replace the block at `src/ppagent/pipeline.py:13-26` (the imports of `hf`, `pdf`, agents, `figures_mod`):

Find:
```python
from ppagent import hf, pdf
from ppagent.agents.assembler import Assembler
from ppagent.agents.classifier import ClassifierAgent
from ppagent.agents.criticizer import CriticizerAgent
from ppagent.agents.finder import FinderAgent
from ppagent.agents.figure_selector import FigureSelectorAgent
from ppagent.agents.searcher import SearcherAgent
from ppagent.agents.writer import WriterAgent
from ppagent.config import AppConfig
from ppagent import figures as figures_mod
from ppagent.hf import HfCliError
from ppagent.llm import LLMClient
from ppagent.models import AgentResult, Paper, PaperContent, PaperReport
from ppagent.storage import Storage
```

Replace with:
```python
from ppagent import arxiv_html, hf, pdf
from ppagent.agents.assembler import Assembler
from ppagent.agents.classifier import ClassifierAgent
from ppagent.agents.criticizer import CriticizerAgent
from ppagent.agents.finder import FinderAgent
from ppagent.agents.searcher import SearcherAgent
from ppagent.agents.writer import WriterAgent
from ppagent.config import AppConfig
from ppagent.hf import HfCliError
from ppagent.llm import LLMClient
from ppagent.models import AgentResult, Paper, PaperContent, PaperReport
from ppagent.storage import Storage
```

(Drops `FigureSelectorAgent`, `figures as figures_mod`.)

- [ ] **Step 3: Remove the `figure_selector` agent from `PaperPipeline.__init__`**

In `src/ppagent/pipeline.py`, the `__init__` constructs `self.figure_selector = FigureSelectorAgent(...)`. Remove that line. Also remove the `vision` LLM client construction:

Find:
```python
        # One LLMClient per role; agents are wired to the role they belong to.
        self._clients = {
            "text": LLMClient(config.llms.text),
            "vision": LLMClient(config.llms.vision),
            "searcher": LLMClient(config.llms.searcher),
        }
        self.classifier = ClassifierAgent(self._clients["text"], config)
        self.searcher = SearcherAgent(self._clients["searcher"], config)
        self.writer = WriterAgent(self._clients["text"], config)
        self.finder = FinderAgent(self._clients["searcher"], config)
        self.criticizer = CriticizerAgent(self._clients["text"], config)
        self.figure_selector = FigureSelectorAgent(self._clients["vision"], config)
        self.storage = Storage(config.output_dir)
```

Replace with:
```python
        # One LLMClient per role; agents are wired to the role they belong to.
        self._clients = {
            "text": LLMClient(config.llms.text),
            "searcher": LLMClient(config.llms.searcher),
        }
        self.classifier = ClassifierAgent(self._clients["text"], config)
        self.searcher = SearcherAgent(self._clients["searcher"], config)
        self.writer = WriterAgent(self._clients["text"], config)
        self.finder = FinderAgent(self._clients["searcher"], config)
        self.criticizer = CriticizerAgent(self._clients["text"], config)
        self.storage = Storage(config.output_dir)
```

- [ ] **Step 4: Rewrite Phase 2 (content retrieval) to use arXiv HTML**

In `src/ppagent/pipeline.py`, the `report()` method currently has Phase 2 at roughly lines 221-262 (the `content_md` / `pdf_path` block) plus phases 6+7 at lines 385-482. Replace the Phase 2 block.

Find the Phase 2 block (starts with the comment `# Get paper content: try hf papers read first, fall back to PDF` and the phase-banner print before it) — replace from:

```python
        # Get paper content: try hf papers read first, fall back to PDF
        self.console.print(
            "[bold yellow]🔄 Phase 2/8: Retrieving paper full text content...[/bold yellow]"
        )
        content_md = ""
        pdf_path = None
        with self.console.status(
            "[dim]Retrieving full text from HuggingFace or extracting from PDF...[/dim]",
            spinner="dots",
        ):
            try:
                content_md = hf.paper_read(paper_id)
                self.console.print(
                    f"  [green]✓[/green] Successfully retrieved paper content via HuggingFace API ({len(content_md)} characters)"
                )
            except HfCliError:
                self.console.print(
                    "  [dim]HuggingFace paper read failed. Attempting to download PDF and extract text...[/dim]"
                )
                if self.config.report.download_pdf:
                    try:
                        self.console.print(
                            f"  Downloading PDF to: [dim]{self.config.pdf_cache_dir}[/dim]"
                        )
                        pdf_path = pdf.download_pdf(paper, self.config.pdf_cache_dir)
                        self.console.print("  Extracting text from PDF...")
                        content_md = pdf.extract_text(pdf_path)
                        self.console.print(
                            f"  [green]✓[/green] Successfully extracted PDF text ({len(content_md)} characters)"
                        )
                    except Exception as pdf_exc:
                        self.console.print(
                            f"  [red]✗[/red] PDF download/extraction failed: {pdf_exc}"
                        )

        if not content_md:
            self.console.print(
                "  [yellow]⚠[/yellow] No full text content available. Falling back to using paper abstract/summary."
            )
            content_md = paper.summary or "Paper content unavailable."

        paper_content = PaperContent(paper=paper, markdown=content_md)
```

Replace with:

```python
        # Get paper content + figures from arXiv HTML. Falls back to PDF text
        # (no figures) when HTML is unavailable for older papers.
        self.console.print(
            "[bold yellow]🔄 Phase 2/6: Retrieving paper content + figures from arXiv HTML...[/bold yellow]"
        )
        paper_dir = self.storage.paper_dir(paper.title, paper.published_at)
        selected_figures: list[arxiv_html.SelectedFigure] = []
        figure_selector_result: AgentResult | None = None
        content_md = ""
        with self.console.status(
            "[dim]Fetching and parsing arXiv HTML...[/dim]", spinner="dots"
        ):
            try:
                parsed = arxiv_html.fetch_and_parse(
                    paper_id, paper_dir, max_figures=self.config.report.max_figures
                )
                content_md = parsed.markdown
                selected_figures = [
                    arxiv_html.SelectedFigure(
                        figure=fig,
                        section=parsed.figure_sections[fig.figure_number],
                    )
                    for fig in parsed.figures
                ]
                self.console.print(
                    f"  [green]✓[/green] Parsed arXiv HTML ({len(content_md)} chars, "
                    f"{len(selected_figures)} figure(s))"
                )
            except arxiv_html.HtmlUnavailable as exc:
                self.console.print(
                    f"  [dim]arXiv HTML unavailable ({exc}); falling back to PDF text...[/dim]"
                )
                if self.config.report.download_pdf:
                    try:
                        pdf_path = pdf.download_pdf(paper, self.config.pdf_cache_dir)
                        content_md = pdf.extract_text(pdf_path)
                        self.console.print(
                            f"  [green]✓[/green] Extracted PDF text ({len(content_md)} chars, no figures)"
                        )
                    except Exception as pdf_exc:
                        self.console.print(
                            f"  [red]✗[/red] PDF fallback failed: {pdf_exc}"
                        )
            except arxiv_html.ParseError as exc:
                self.console.print(
                    f"  [yellow]⚠[/yellow] arXiv HTML parse failed ({exc}); trying PDF text."
                )
                if self.config.report.download_pdf:
                    try:
                        pdf_path = pdf.download_pdf(paper, self.config.pdf_cache_dir)
                        content_md = pdf.extract_text(pdf_path)
                    except Exception as pdf_exc:
                        self.console.print(
                            f"  [red]✗[/red] PDF fallback failed: {pdf_exc}"
                        )

        if not content_md:
            self.console.print(
                "  [yellow]⚠[/yellow] No full text content available. Falling back to paper abstract/summary."
            )
            content_md = paper.summary or "Paper content unavailable."

        paper_content = PaperContent(paper=paper, markdown=content_md)
```

- [ ] **Step 5: Delete old phases 6 + 7 (PDF ensure + figure extract/select)**

In `src/ppagent/pipeline.py`, the old phases 6 ("Ensure PDF is downloaded for figure extraction") and 7 ("Extracting and selecting paper figures") are now obsolete — figures were collected in Phase 2. Find the block starting with the comment:

```python
        # Ensure we have the PDF downloaded for figure extraction.
        # (hf papers read may have succeeded without a local PDF.)
```

…and ending just before `# Assemble`. This whole block (the Phase 6 print+status and the Phase 7 print+status+figure_selector call) must be **deleted entirely**.

Replace the entire Phase 6 + Phase 7 block with nothing (i.e. delete it). The flow should now go directly from the Criticizer (Phase 5) to the Assemble banner.

- [ ] **Step 6: Update Phase 8 → Phase 6 renumbering + drop `figure_selector_result` arg**

The old Phase 8 (Assemble) calls `self.assembler.assemble(... figure_selector_result=figure_selector_result ...)`. Since the `figure_selector` agent no longer exists, drop that argument. The `selected_figures` arg stays (it's now populated from arXiv HTML).

Find the Assemble banner + call:

```python
        # Assemble
        self.console.print(
            "[bold yellow]🔄 Phase 8/8: Assembling final report...[/bold yellow]"
        )
        with self.console.status(
            "[dim]Formatting, generating LaTeX equations, and writing report files...[/dim]",
            spinner="dots",
        ):
            report, md_content, html_content = self.assembler.assemble(
                paper=paper,
                writer_result=writer_result,
                finder_result=finder_result,
                criticizer_result=criticizer_result,
                figure_selector_result=figure_selector_result,
                classifier_result=classifier_result,
                selected_figures=selected_figures or None,
                paper_type=paper_type,
            )
            self.console.print("  [green]✓[/green] Report assembled successfully!")
```

Replace with:

```python
        # Assemble
        self.console.print(
            "[bold yellow]🔄 Phase 6/6: Assembling final report...[/bold yellow]"
        )
        with self.console.status(
            "[dim]Formatting, generating LaTeX equations, and writing report files...[/dim]",
            spinner="dots",
        ):
            report, md_content, html_content = self.assembler.assemble(
                paper=paper,
                writer_result=writer_result,
                finder_result=finder_result,
                criticizer_result=criticizer_result,
                figure_selector_result=None,
                classifier_result=classifier_result,
                selected_figures=selected_figures or None,
                paper_type=paper_type,
            )
            self.console.print("  [green]✓[/green] Report assembled successfully!")
```

Also update the phase numbers in the other banners to match the new 6-phase scheme:
- Phase 1/8 → Phase 1/6
- Phase 3/8 → Phase 3/6
- Phase 4/8 → Phase 4/6
- Phase 5/8 → Phase 5/6

- [ ] **Step 7: Remove the now-unused vision-model print block at the top of `report()`**

The `report()` method's header prints (lines ~171-191) include a "Vision model (Figure Selector)" block. Delete just that block:

Find:
```python
        self.console.print(
            f"  • [bold]Vision model (Figure Selector):[/bold] [cyan]{self.config.llms.vision.model}[/cyan]"
        )
        self.console.print(
            f"    [dim]Base URL: {self.config.llms.vision.base_url} | Temperature: {self.config.llms.vision.temperature} | Max tokens: {self.config.llms.vision.max_tokens} | Thinking: {self.config.llms.vision.enable_thinking}[/dim]"
        )
```

Delete those two `self.console.print(...)` blocks entirely. Keep the text-model and searcher-model blocks.

- [ ] **Step 8: Run pipeline smoke check (module imports + syntax)**

Run: `uv run python -c "from ppagent.pipeline import PaperPipeline; print('ok')"`
Expected: prints `ok` (no import errors).

- [ ] **Step 9: Run the full test suite**

Run: `uv run pytest -x -q`
Expected: all tests PASS except possibly `test_figures.py` (which we delete in Task 7) and any test importing `FigureSelectorAgent` (also deleted in Task 7). Note any failures to fix in the next task.

- [ ] **Step 10: Commit**

```bash
git add src/ppagent/pipeline.py
git commit -m "refactor(pipeline): read paper content + figures from arXiv HTML; collapse to 6 phases"
```

---

## Task 5: Drop the `vision` LLM role from `config.py` and add `max_figures`

**Files:**
- Modify: `src/ppagent/config.py`

- [ ] **Step 1: Write failing test for the config shape**

Add to `tests/test_config_persistence.py` (read the file first to match its style):

```python
def test_llms_config_has_no_vision_role():
    """The vision LLM role was removed when figure selection moved to arXiv HTML."""
    from ppagent.config import LLMsConfig

    cfg = LLMsConfig()
    assert not hasattr(cfg, "vision")
    assert set(cfg.model_fields) == {"text", "searcher", "saved_vendors"}


def test_report_config_has_max_figures_default_8():
    from ppagent.config import ReportConfig

    cfg = ReportConfig()
    assert cfg.max_figures == 8


def test_for_role_rejects_vision():
    from ppagent.config import LLMsConfig

    cfg = LLMsConfig()
    import pytest

    with pytest.raises(ValueError):
        cfg.for_role("vision")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_persistence.py -v -k "vision or max_figures or for_role"`
Expected: FAIL.

- [ ] **Step 3: Edit `src/ppagent/config.py`**

Make these specific edits:

**(a)** Delete the `_vision_default()` function (lines ~61-63):
```python
def _vision_default() -> LLMConfig:
    """Default LLM config for the vision role (a vision-capable model)."""
    return LLMConfig(model="gpt-4o")
```

**(b)** In `LLMsConfig`, remove the `vision` field and update the docstring. Find:
```python
class LLMsConfig(BaseModel):
    """Per-role LLM configurations.

    - ``text``: agents that reason over paper text (writer, finder, criticizer).
    - ``vision``: the figure_selector agent, which sends images to the LLM.
    - ``searcher``: the paper-scoring/relevance agent (discovery phase).

    ``saved_vendors`` stores the last-edited LLMConfig for each
    ...
    """

    text: LLMConfig = Field(default_factory=LLMConfig)
    vision: LLMConfig = Field(default_factory=_vision_default)
    searcher: LLMConfig = Field(default_factory=LLMConfig)
    saved_vendors: dict[str, dict[str, LLMConfig]] = Field(default_factory=dict)

    def for_role(self, role: str) -> LLMConfig:
        """Return the LLMConfig for a given role name."""
        if role not in ("text", "vision", "searcher"):
            raise ValueError(f"Unknown LLM role: {role!r}")
        return getattr(self, role)
```

Replace with:
```python
class LLMsConfig(BaseModel):
    """Per-role LLM configurations.

    - ``text``: agents that reason over paper text (writer, finder, criticizer).
    - ``searcher``: the paper-scoring/relevance agent (discovery phase).

    ``saved_vendors`` stores the last-edited LLMConfig for each
    ``(role, vendor_key)`` pair so the user can switch providers in the TUI
    without losing previously entered keys/models. The live role field
    (``text`` / ``searcher``) is the *currently active* provider;
    the pipeline only ever reads from the live field, so it requires no
    changes. ``saved_vendors`` is keyed as ``{role: {vendor_key: <LLMConfig>}}``.
    """

    text: LLMConfig = Field(default_factory=LLMConfig)
    searcher: LLMConfig = Field(default_factory=LLMConfig)
    saved_vendors: dict[str, dict[str, LLMConfig]] = Field(default_factory=dict)

    def for_role(self, role: str) -> LLMConfig:
        """Return the LLMConfig for a given role name."""
        if role not in ("text", "searcher"):
            raise ValueError(f"Unknown LLM role: {role!r}")
        return getattr(self, role)
```

**(c)** In `ReportConfig`, add the `max_figures` field. Find the `download_pdf` field:
```python
    download_pdf: bool = Field(
        default=True,
        description="Whether to download the paper's PDF for figure extraction.",
    )
```
Replace with:
```python
    download_pdf: bool = Field(
        default=True,
        description="Whether to download the paper's PDF as a text-only fallback when arXiv HTML is unavailable.",
    )
    max_figures: int = Field(
        default=8,
        description="Maximum number of figures to extract from the paper's arXiv HTML and insert into the report.",
    )
```

**(d)** Update `AGENT_LLM_ROLE` — drop the figure_selector entry. Find:
```python
AGENT_LLM_ROLE: dict[str, str] = {
    "classifier": "text",
    "writer": "text",
    "finder": "searcher",
    "criticizer": "text",
    "figure_selector": "vision",
    "searcher": "searcher",
}
```
Replace with:
```python
AGENT_LLM_ROLE: dict[str, str] = {
    "classifier": "text",
    "writer": "text",
    "finder": "searcher",
    "criticizer": "text",
    "searcher": "searcher",
}
```

**(e)** Update `_LLM_ROLES` constant. Find:
```python
_LLM_ROLES = ("text", "vision", "searcher")
```
Replace with:
```python
_LLM_ROLES = ("text", "searcher")
```

**(f)** Update `_migrate_legacy_llm` docstring/comment (the code itself uses `_LLM_ROLES` which now excludes vision, so it's automatically correct — but the comment mentions three roles). Find:
```python
    """Migrate a legacy flat ``[llm]`` section to the new ``llms.*`` structure.

    If ``raw`` has a flat ``llm`` mapping but no ``llms`` key, the flat config is
    cloned into ``llms.text``, ``llms.vision``, and ``llms.searcher`` so existing
    setups keep working unchanged. The legacy ``llm`` key is left in place only
    until the config is re-saved (callers should write ``llms`` on save).
    """
```
Replace with:
```python
    """Migrate a legacy flat ``[llm]`` section to the new ``llms.*`` structure.

    If ``raw`` has a flat ``llm`` mapping but no ``llms`` key, the flat config is
    cloned into ``llms.text`` and ``llms.searcher`` so existing setups keep
    working unchanged. The legacy ``llm`` key is left in place only until the
    config is re-saved (callers should write ``llms`` on save).
    """
```

**(g)** Update `_apply_env_overrides` docstring. Find:
```python
    """Override config values with environment variables.

    ``PPA_LLM_*`` apply to ALL three LLM roles (text/vision/searcher) so a single
    env var can configure the whole app headlessly.
    """
```
Replace with:
```python
    """Override config values with environment variables.

    ``PPA_LLM_*`` apply to BOTH LLM roles (text/searcher) so a single env var
    can configure the whole app headlessly.
    """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_persistence.py -v`
Expected: all tests PASS (including the three new ones).

- [ ] **Step 5: Commit**

```bash
git add src/ppagent/config.py tests/test_config_persistence.py
git commit -m "refactor(config): drop vision LLM role; add report.max_figures"
```

---

## Task 6: Delete `chat_vision()` from `llm.py` and clean up agents

**Files:**
- Modify: `src/ppagent/llm.py:506-533` (`chat_vision`), `:674-680` (`_image_to_data_uri`)
- Modify: `src/ppagent/agents/__init__.py:66`
- Modify: `src/ppagent/agents/prompts.py:718-750` (FIGURE_SELECTOR constants)
- Modify: `src/ppagent/agents/assembler.py:13`

- [ ] **Step 1: Delete `chat_vision` and `_image_to_data_uri` from `llm.py`**

In `src/ppagent/llm.py`, find and delete the entire `chat_vision` method (lines ~506-532):

```python
    def chat_vision(
        self,
        system: str,
        user_text: str,
        images: list[Path],
    ) -> str:
        """Multimodal chat completion: send images + text, return plain text.

        ``images`` are file paths to PNG/JPEGs. Each is embedded as a base64
        data URI so the call works with any OpenAI-compatible vision endpoint
        without exposing local files. Returns the assistant's text response.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img_path in images:
            data_uri = _image_to_data_uri(img_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        resp = self.chat(messages)
        return resp.output_text
```

And the module-level `_image_to_data_uri` function at the bottom (lines ~674-680):

```python
def _image_to_data_uri(img_path: Path) -> str:
    """Encode an image file as a base64 data URI for vision API calls."""
    ext = img_path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/{mime};base64,{b64}"
```

Also remove the now-unused `base64` import at the top of the file (line 5):
```python
import base64
```

And the now-unused `Path` import if it was only used by `chat_vision` — check first; `Path` is likely used elsewhere in the file, so only remove `base64`.

- [ ] **Step 2: Delete the `FigureSelectorAgent` import from `agents/__init__.py`**

In `src/ppagent/agents/__init__.py`, find line 66:
```python
from ppagent.agents.figure_selector import FigureSelectorAgent  # noqa: F401, E402
```
Delete that line.

- [ ] **Step 3: Delete the `FIGURE_SELECTOR_*` prompts from `prompts.py`**

In `src/ppagent/agents/prompts.py`, find the block at lines ~718-750 (the section header comment + both constants):

```python
# ==============================================================================
# Figure Selector Agent Prompts
# ==============================================================================

FIGURE_SELECTOR_SYSTEM_PROMPT = (
    ...
)

FIGURE_SELECTOR_USER_PROMPT_TEMPLATE = """\
...
"""
```

Delete the entire block (from the `# ====...` header through the end of `FIGURE_SELECTOR_USER_PROMPT_TEMPLATE`).

- [ ] **Step 4: Update `assembler.py` import**

In `src/ppagent/agents/assembler.py:13`, find:
```python
from ppagent.figures import FIGURE_SECTIONS, Figure, SelectedFigure
```
Replace with:
```python
from ppagent.arxiv_html import FIGURE_SECTIONS, Figure, SelectedFigure
```

- [ ] **Step 5: Verify imports are clean**

Run:
```bash
uv run python -c "import ppagent.llm, ppagent.agents, ppagent.agents.assembler, ppagent.agents.prompts; print('imports ok')"
uv run ruff check src/ppagent/llm.py src/ppagent/agents/
```
Expected: imports succeed; ruff reports no unused imports.

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -x -q`
Expected: failures only in `tests/test_figures.py` and possibly any test referencing `FigureSelectorAgent` (to be deleted in Task 7). Everything else passes.

- [ ] **Step 7: Commit**

```bash
git add src/ppagent/llm.py src/ppagent/agents/__init__.py src/ppagent/agents/prompts.py src/ppagent/agents/assembler.py
git commit -m "refactor: remove vision LLM client + figure_selector agent + prompts"
```

---

## Task 7: Delete the old `figures.py` module and its tests

**Files:**
- Delete: `src/ppagent/figures.py`
- Delete: `src/ppagent/agents/figure_selector.py`
- Delete: `tests/test_figures.py`

- [ ] **Step 1: Confirm no remaining references to the old module**

Run:
```bash
uv run python -c "
import subprocess
out = subprocess.check_output(['git', 'grep', '-l', 'ppagent.figures\\|ppagent import figures\\|FigureSelectorAgent\\|figures_mod'], text=True)
print('References found:')
print(out or '(none)')
"
```
Expected: no references to `ppagent.figures`, `figures_mod`, or `FigureSelectorAgent` anywhere in `src/`. (Tests in `tests/test_figures.py` will still match — that's expected; we delete it next.)

If any `src/` reference remains, fix it before proceeding.

- [ ] **Step 2: Delete the three files**

```bash
git rm src/ppagent/figures.py
git rm src/ppagent/agents/figure_selector.py
git rm tests/test_figures.py
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: delete figures.py, figure_selector.py, and their tests

Figure extraction now lives in ppagent.arxiv_html (arXiv HTML fetch +
deterministic section mapping). The PyMuPDF caption-cropping pipeline and
the vision-LLM figure selector are gone."
```

---

## Task 8: Update TUI to drop the vision menu entry

**Files:**
- Modify: `src/ppagent/tui.py`

- [ ] **Step 1: Write failing test for the TUI change**

Add to `tests/test_tui_config.py` (read it first to match style). The test asserts the `llms` menu no longer contains a vision entry:

```python
def test_llms_menu_has_no_vision_entry():
    """The vision LLM menu entry was removed when figure selection moved to arXiv HTML."""
    from ppagent.tui import MENUS

    llms_menu = MENUS["llms"]
    targets = [item.target for item in llms_menu]
    assert "llm_vision_vendor" not in targets
    assert "llm_text_vendor" in targets
    assert "llm_searcher_vendor" in targets
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tui_config.py::test_llms_menu_has_no_vision_entry -v`
Expected: FAIL.

- [ ] **Step 3: Edit `src/ppagent/tui.py`**

**(a)** Update the main-menu "LLM API Settings" description (line ~250). Find:
```python
        MenuItem(
            "LLM API Settings",
            target="llms",
            description="Configure per-role LLM providers: text (writer/finder/criticizer), vision (figure selector), searcher (paper scoring).",
        ),
```
Replace with:
```python
        MenuItem(
            "LLM API Settings",
            target="llms",
            description="Configure per-role LLM providers: text (writer/finder/criticizer), searcher (paper scoring).",
        ),
```

**(b)** Delete the Vision LLM menu item (lines ~281-284). Find:
```python
        MenuItem(
            "Vision LLM (figure selector)",
            target="llm_vision_vendor",
            description="Vision-capable LLM used by the figure_selector agent to pick pipeline diagrams.",
        ),
```
Delete that entire `MenuItem(...)` block.

**(c)** Update the three regexes that include `vision`. Find (around line 506):
```python
    vendor_list_match = re.match(r"^llm_(text|vision|searcher)_vendor$", menu_id)
    if vendor_list_match:
        role = vendor_list_match.group(1)
        role_label = {
            "text": "Text LLM",
            "vision": "Vision LLM",
            "searcher": "Searcher LLM",
        }[role]
```
Replace with:
```python
    vendor_list_match = re.match(r"^llm_(text|searcher)_vendor$", menu_id)
    if vendor_list_match:
        role = vendor_list_match.group(1)
        role_label = {
            "text": "Text LLM",
            "searcher": "Searcher LLM",
        }[role]
```

Find (around line 548):
```python
    picker_match = re.match(
        r"^llm_(text|vision|searcher)_([a-z0-9_]+)_latest$", menu_id
    )
```
Replace with:
```python
    picker_match = re.match(
        r"^llm_(text|searcher)_([a-z0-9_]+)_latest$", menu_id
    )
```

Find (around line 587):
```python
    vendor_setting_match = re.match(
        r"^llm_(text|vision|searcher)_([a-z0-9_]+)$", menu_id
    )
```
Replace with:
```python
    vendor_setting_match = re.match(
        r"^llm_(text|searcher)_([a-z0-9_]+)$", menu_id
    )
```

**(d)** Check the docstring at `set_config_value` (line ~615) — it mentions "vision". Find:
```python
    """Set nested attribute of AppConfig by path string (e.g. 'llm.api_key').

    For ``llms.<role>.api_key``, the new value is propagated to every sibling
    role whose active provider matches: API keys are per-provider, not per-role,
    so entering the key once (e.g. in "vision") makes it appear immediately in
    "text"/"searcher" when they use the same provider.
    """
```
Replace with:
```python
    """Set nested attribute of AppConfig by path string (e.g. 'llm.api_key').

    For ``llms.<role>.api_key``, the new value is propagated to every sibling
    role whose active provider matches: API keys are per-provider, not per-role,
    so entering the key once makes it appear immediately in the other role
    when both use the same provider.
    """
```

**(e)** Check the module-level comment at line ~79 and ~98 for "vision" mentions. Find and update any prose that says "text/vision/searcher" to "text/searcher". The exact text:
- Line ~79: `text/vision/searcher alike` → `text/searcher alike`
- Line ~98: `text and vision both on OpenAI` → `text and searcher both on OpenAI`

- [ ] **Step 4: Run TUI tests + ruff**

Run:
```bash
uv run pytest tests/test_tui_config.py -v
uv run ruff check src/ppagent/tui.py
```
Expected: tests pass; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/ppagent/tui.py tests/test_tui_config.py
git commit -m "refactor(tui): remove vision LLM menu entry and update role regexes"
```

---

## Task 9: Update `cli.py` (`config_show`) and run final verification

**Files:**
- Modify: `src/ppagent/cli.py`

- [ ] **Step 1: Write failing test for `config_show` output**

Add to `tests/test_cli.py` (read it first):

```python
def test_config_show_does_not_mention_vision(capsys):
    """config_show no longer prints a Vision LLM line."""
    from ppagent.cli import config_show

    # config_show calls _load(); we need a config to exist. Use the test's
    # isolated config fixture if one exists, else skip gracefully.
    try:
        config_show()
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "Vision LLM" not in out
    assert "figure_selector" not in out
```

If `test_cli.py` has no config fixture and `config_show` requires one, wrap in `try/except FileNotFoundError: pytest.skip("no config")` instead.

- [ ] **Step 2: Run test to verify it fails (or skip)**

Run: `uv run pytest tests/test_cli.py::test_config_show_does_not_mention_vision -v`

- [ ] **Step 3: Edit `src/ppagent/cli.py`**

In `config_show` (lines ~393-411), find and delete the Vision LLM line:
```python
    console.print(
        f"  Vision LLM (figure_selector):       {cfg.llms.vision.model} @ {cfg.llms.vision.base_url}"
    )
```
Delete that `console.print(...)` block entirely.

- [ ] **Step 4: Run all tests + ruff on the whole project**

Run:
```bash
uv run pytest -q
uv run ruff check src/ tests/
```
Expected: all tests pass; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/ppagent/cli.py tests/test_cli.py
git commit -m "refactor(cli): drop Vision LLM line from config_show"
```

---

## Task 10: Update README and run end-to-end smoke test

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README prose**

In `README.md`, find any mention of "vision" / "Vision" / "figure" extraction and update:

Find the line in the intro:
```
A **Vision** agent picks the best figure from the PDF.
```
Replace with:
```
Figures are pulled directly from the paper's arXiv HTML version.
```

Find in "How it works":
```
**PDF & Markdown:**
- PDF processing uses `PyMuPDF`.
```
Replace with:
```
**Paper content:**
- Paper text and figures are read from the arXiv HTML version; PDF text extraction (`PyMuPDF`) remains as a fallback for older papers without HTML.
```

- [ ] **Step 2: Commit README**

```bash
git add README.md
git commit -m "docs: update README for arXiv HTML reader"
```

- [ ] **Step 3: End-to-end smoke test (manual — requires network + LLM key)**

Run:
```bash
uv run ppagent report arxiv:2606.01075 --force --no-open
```
Expected: completes all 6 phases; produces `output/<...>/report.html` with at least one figure rendered in the Method section; no errors about vision LLM or PyMuPDF figure extraction.

If you don't have an LLM key in this environment, skip this step and note it for the user to run.

- [ ] **Step 4: Final full test run**

Run:
```bash
uv run pytest -q
uv run ruff check .
```
Expected: all green.

---

## Self-Review (completed during planning)

**1. Spec coverage check:**
- §1 Architecture & module layout → Tasks 1 (module shell + migrated symbols), 6 (imports move), 7 (delete old module).
- §2 Parsing strategy → Task 2 (parser: markdown + figures + math + skip bibliography/data-uri).
- §3 Pipeline + fallback → Task 4 (Phase 2 HTML-first with PDF-text fallback; phases 6/7 deleted).
- §4 Config + TUI + removals → Tasks 5 (config), 8 (TUI), 6 (agent/llm cleanup), 9 (cli).
- §5 Edge cases → Task 2 (skip data-uri/static/SVG, subfigures, no-img figures), Task 3 (image 404 skip, HtmlUnavailable, ParseError, max_figures).
- §6 Test plan → every named test maps to a step in Tasks 1-3.

**2. Placeholder scan:** None found — every step has concrete code or commands.

**3. Type/signature consistency:**
- `Figure`, `SelectedFigure`, `FIGURE_SECTIONS` defined in Task 1, imported in Tasks 4 & 6.
- `_RawFigure` defined in Task 2, consumed in Task 3.
- `parse_html(html, *, page_url)` signature consistent across Tasks 2 & 3.
- `fetch_and_parse(paper_id, out_dir, *, max_figures=8)` consistent across Tasks 3 & 4.
- `_map_section(paper_section_title, caption)` consistent across Tasks 1, 2, 3.
- `ParsedHtml` fields (`markdown`, `figures`, `figure_sections`) consistent across Tasks 3 & 4.
