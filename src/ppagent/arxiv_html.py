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
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import httpx

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


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------


@dataclass
class _RawFigure:
    """An intermediate figure produced by the parser, before download.

    Carries the resolved absolute ``src_url`` and the ``section_title`` it
    appeared under; :func:`fetch_and_parse` downloads each and converts it to
    a :class:`Figure` with a local ``image_path``.
    """

    figure_number: int
    caption: str
    src_url: str
    section_title: str


# Headings whose content is treated as a section title.
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")

# Classes on a <section> that mark it as the bibliography (skip entirely).
_SKIP_SECTION_CLASSES = ("ltx_bibliography",)

# Tags whose entire content (text + children) is dropped from the markdown.
# ``<script>``/``<style>`` carry JS/CSS that would otherwise leak as prose;
# ``<noscript>`` mirrors fallback content we don't want either.
_SKIP_CONTENT_TAGS = ("script", "style", "noscript")

# arXiv HTML wraps the actual paper in <article class="ltx_document ...">.
# Everything outside it (page chrome: feedback modals, nav, header/footer) is
# noise we must not feed to the analysis agents. When present, we parse only
# the article subtree; if absent we fall back to the whole document.
_ARTICLE_OPEN_RE = re.compile(r"<article\b[^>]*>", re.IGNORECASE)


def _extract_article(html: str) -> str:
    """Return the inner HTML of the first <article> element, or ``html`` itself.

    arXiv's paper content lives inside ``<article class="ltx_document">``; the
    surrounding page (feedback modal, theme controls, nav) is chrome that would
    otherwise leak into the markdown. We slice from the first ``<article>`` open
    tag to its matching close tag by depth-counting (regex can't balance, but
    articles don't nest in arXiv HTML).
    """
    open_match = _ARTICLE_OPEN_RE.search(html)
    if not open_match:
        return html
    start = open_match.end()
    depth = 1
    pos = start
    tag_re = re.compile(r"<(/?)article\b[^>]*>", re.IGNORECASE)
    while depth > 0:
        m = tag_re.search(html, pos)
        if not m:
            # Unbalanced — return everything from the first <article> onward.
            return html[start:]
        if m.group(1) == "/":
            depth -= 1
            if depth == 0:
                return html[start:m.start()]
        else:
            depth += 1
        pos = m.end()
    return html[start:m.start()]


def _img_src_is_skippable(src: str) -> bool:
    """Return True for image sources we never treat as paper figures."""
    if not src:
        return True
    if src.startswith("data:"):
        return True  # inline base64 icons
    if src.startswith("/static/"):
        return True  # arxiv chrome logos
    if src.endswith(".svg"):
        return True  # SVGs deferred (rare; revisit later)
    return False


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
        # Heading being accumulated: non-None only between start and end tag.
        self._heading_buf: list[str] | None = None
        # Figure being accumulated.
        self._raw_figures: list[_RawFigure] = []
        self._in_figure: int = 0  # nest depth of <figure>
        self._figure_caption_buf: list[str] | None = None
        self._figure_imgs: list[str] = []  # src_urls collected in current figure
        self._next_figure_number: int = 1
        # Math handling. arXiv HTML embeds the original LaTeX inside
        # <annotation encoding="application/x-tex">...</annotation>.
        self._math_depth: int = 0
        self._math_is_block: bool = False
        self._math_latex: str | None = None  # buffer while capturing annotation
        self._want_annotation_tex: bool = False
        # Stack of tags that opened a skip region, so each closing tag only
        # unwinds the skip it actually started. A `<section>` only skips when
        # it carries a bibliography class, so we record "section" only then;
        # a plain `</section>` inside a `<nav>` must not pop the nav's skip.
        self._skip_stack: list[str] = []

    # --- helpers ---------------------------------------------------------

    def _emit(self, text: str) -> None:
        if not self._skip_stack:
            self._md_chunks.append(text)

    def _is_skipping(self) -> bool:
        return bool(self._skip_stack)

    def _open_skip(self, tag: str) -> None:
        self._skip_stack.append(tag)

    # --- start/end tags --------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = dict(attrs)

        # Bibliography / nav / script / style: skip entirely (and children).
        if tag == "section" and any(
            cls in (d.get("class") or "") for cls in _SKIP_SECTION_CLASSES
        ):
            self._open_skip("section")
            return
        if tag in ("nav", *_SKIP_CONTENT_TAGS):
            self._open_skip(tag)
            return
        if self._is_skipping():
            return

        if tag in _HEADING_TAGS:
            self._heading_buf = []
            return

        if tag == "figure":
            self._in_figure += 1
            self._figure_imgs = []
            self._figure_caption_buf = []
            return

        if tag == "img":
            src = d.get("src") or ""
            if _img_src_is_skippable(src):
                logger.debug("Skipping image src=%s", src)
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
                self._math_latex = ""
            return

        # Structural markdown emission for common content tags.
        if tag == "p":
            self._emit("")
        elif tag == "li":
            self._emit("\n- ")
        elif tag in ("ul", "ol", "table"):
            self._emit("")
        elif tag == "br":
            self._emit("  \n")
        # Unknown tags: nothing — their text content still flows via handle_data.

    def handle_endtag(self, tag: str) -> None:
        # Pop the skip stack only when the closing tag matches the tag that
        # opened the current skip region. This keeps a plain `</section>`
        # (which never opened a skip) from prematurely unwinding a <nav> skip.
        if self._skip_stack and self._skip_stack[-1] == tag:
            self._skip_stack.pop()
            return
        if self._is_skipping():
            return

        if tag in _HEADING_TAGS and self._heading_buf is not None:
            raw = "".join(self._heading_buf).strip()
            if raw:
                self._current_section = raw
                cleaned = _NUMERIC_PREFIX_RE.sub("", raw).strip()
                level = int(tag[1])
                self._emit("\n\n" + "#" * level + " " + cleaned + "\n\n")
            self._heading_buf = None
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

        # Inside a figure, text only contributes to the caption (not the
        # markdown the analysis agents read).
        if self._in_figure > 0 and self._figure_caption_buf is not None:
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
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + "\n"

    def raw_figures(self) -> list[_RawFigure]:
        return list(self._raw_figures)


def parse_html(html: str, *, page_url: str) -> tuple[str, list[_RawFigure]]:
    """Parse arXiv HTML into (markdown, raw_figures).

    ``raw_figures`` are :class:`_RawFigure` instances carrying the resolved
    absolute ``src_url`` and the section title each appeared under. Callers
    (``fetch_and_parse``) download each and convert to :class:`Figure`.

    Only the ``<article>`` subtree (the actual paper) is parsed when present;
    surrounding page chrome (feedback modal, nav, theme controls) is dropped.
    """
    article_html = _extract_article(html)
    p = _ArxivHtmlParser(page_url=page_url)
    p.feed(article_html)
    p.close()
    return p.markdown(), p.raw_figures()


# ---------------------------------------------------------------------------
# Network fetch + image download
# ---------------------------------------------------------------------------

_HTML_FETCH_TIMEOUT = 120  # seconds
_IMAGE_FETCH_TIMEOUT = 60  # seconds
_USER_AGENT = "ppagent/1.0 (https://github.com/AutoPhd-org/your-paper-reading-agent)"
_ARXIV_HTML_URL_TEMPLATE = "https://arxiv.org/html/{paper_id}"
# A fetched page must yield at least this much markdown OR at least one figure
# to count as a real paper. Truly empty responses (login walls, 200-OK errors,
# malformed stubs) produce neither and trigger ParseError.
_MIN_PAPER_MARKDOWN_LEN = 20


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
    last_segment = path.rsplit("/", 1)[-1]
    if "." in last_segment:
        ext = last_segment.rsplit(".", 1)[-1].lower()
        if ext == "jpeg":
            ext = "jpg"
        if ext.isalnum() and len(ext) <= 5:
            return ext
    return "png"


def fetch_and_parse(
    paper_id: str,
    out_dir: Path,
    *,
    max_figures: int = 8,
) -> ParsedHtml:
    """Fetch an arXiv HTML paper and parse it into markdown + downloaded figures.

    Fetches ``https://arxiv.org/html/{paper_id}`` (arXiv redirects to the
    latest version, which ``httpx`` follows), parses the body into markdown +
    figures, and downloads each figure image (up to ``max_figures``) into
    ``out_dir/figures/``.

    Raises:
        HtmlUnavailable: HTTP 4xx/5xx or network error on the HTML page.
        ParseError: page fetched but body is not a recognizable paper.
    """
    url = _ARXIV_HTML_URL_TEMPLATE.format(paper_id=paper_id)
    headers = {"User-Agent": _USER_AGENT}

    # One shared client for the HTML page + all image downloads. Per-request
    # timeouts (via client.get(..., timeout=)) keep the page and image limits
    # independent without churning connections.
    with httpx.Client(
        timeout=_HTML_FETCH_TIMEOUT, follow_redirects=True, headers=headers
    ) as client:
        try:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
            page_url = str(resp.url)
        except httpx.HTTPError as exc:
            raise HtmlUnavailable(
                f"Could not fetch arXiv HTML for {paper_id!r} at {url}: {exc}"
            ) from exc

        markdown, raw_figures = parse_html(html, page_url=page_url)
        # A real paper yields either meaningful prose or at least one figure.
        # Empty/short markdown with no figures signals a non-paper response
        # (login wall, malformed stub, withdrawn-paper notice).
        if len(markdown.strip()) < _MIN_PAPER_MARKDOWN_LEN and not raw_figures:
            raise ParseError(
                f"arXiv HTML for {paper_id!r} parsed to empty content ({len(markdown)} chars, "
                f"{len(raw_figures)} figures); not a recognizable paper."
            )

        # Download images (up to max_figures). A failed download drops just
        # that figure rather than failing the whole parse.
        figures: list[Figure] = []
        figure_sections: dict[int, str] = {}
        figures_subdir = out_dir / "figures"
        figures_subdir.mkdir(parents=True, exist_ok=True)

        for raw in raw_figures:
            if len(figures) >= max_figures:
                break
            ext = _ext_from_url(raw.src_url)
            rel_path = f"figures/figure_{raw.figure_number}.{ext}"
            abs_path = figures_subdir / f"figure_{raw.figure_number}.{ext}"
            try:
                img_resp = client.get(raw.src_url, timeout=_IMAGE_FETCH_TIMEOUT)
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

    return ParsedHtml(
        markdown=markdown, figures=figures, figure_sections=figure_sections
    )
