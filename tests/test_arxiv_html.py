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


# ---------------------------------------------------------------------------
# parse_html — HTML → markdown + raw figures
# ---------------------------------------------------------------------------

from ppagent.arxiv_html import parse_html  # noqa: E402


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
    assert len(figures) == 1
    fig = figures[0]
    assert fig.figure_number == 1
    assert fig.caption == "Figure 1: The pipeline."
    assert fig.src_url == "https://arxiv.org/html/2501.00001v1/x1.png"
    assert fig.section_title == "3 Method"
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
    """A <figure> with two <img>s yields two figures, same caption+section."""
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
    _md, figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert len(figures) == 2
    assert figures[0].figure_number == 1
    assert figures[1].figure_number == 2
    assert figures[0].caption == figures[1].caption == "(a) curve one. (b) curve two."
    assert figures[0].section_title == figures[1].section_title == "4 Results"


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
    _md, figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
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
    md, _figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
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
    md, _figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert "$$X = f(Y)$$" in md


def test_parse_html_assigns_section_per_figure():
    """Each figure remembers the most-recent heading it appeared under."""
    from ppagent.arxiv_html import _map_section

    html = (
        "<html><body><article>"
        "<h2 class='ltx_title'>3 Method</h2>"
        "<figure><img src='x1.png'/><figcaption>F1</figcaption></figure>"
        "<h2 class='ltx_title'>5 Experiments</h2>"
        "<figure><img src='x2.png'/><figcaption>F2</figcaption></figure>"
        "</article></body></html>"
    )
    _md, raw_figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert len(raw_figures) == 2
    assert (
        _map_section(raw_figures[0].section_title, raw_figures[0].caption) == "method"
    )
    assert (
        _map_section(raw_figures[1].section_title, raw_figures[1].caption)
        == "evaluation"
    )


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
    md, _figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert "Body text." in md
    assert "Someone" not in md
    assert "References" not in md


def test_parse_html_bare_img_outside_figure_becomes_standalone_figure():
    """An <img> not wrapped in <figure> is still captured as a figure."""
    html = (
        "<html><body><article>"
        "<h2 class='ltx_title'>1 Introduction</h2>"
        "<p><img src='2501.00001v1/inline.png'/></p>"
        "</article></body></html>"
    )
    _md, figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert len(figures) == 1
    assert figures[0].figure_number == 1
    assert figures[0].src_url == "https://arxiv.org/html/2501.00001v1/inline.png"


def test_parse_html_skips_script_and_style_content():
    """JS/CSS inside <script>/<style>/<noscript> must not leak into markdown."""
    html = (
        "<html><head>"
        "<style>body { color: red; } .hide { display: none; }</style>"
        "<script>function initialize() { localStorage.getItem('theme'); }</script>"
        "</head><body><article>"
        "<noscript>Enable JS for the best experience.</noscript>"
        "<h2 class='ltx_title'>1 Introduction</h2>"
        "<p>Real paper content.</p>"
        "</article></body></html>"
    )
    md, _figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert "Real paper content." in md
    assert "initialize" not in md
    assert "localStorage" not in md
    assert "color: red" not in md
    assert "display: none" not in md
    assert "Enable JS" not in md


def test_parse_html_drops_page_chrome_outside_article():
    """arXiv page chrome (feedback modal, nav) lives outside <article> and
    must not leak into the markdown the analysis agents read."""
    html = (
        "<html><body>"
        "<nav><a>arXiv</a></nav>"
        "<form class='modal'><h5>Report GitHub Issue</h5>"
        "<button>Submit without GitHub</button></form>"
        "<article class='ltx_document'>"
        "<h2 class='ltx_title'>1 Introduction</h2>"
        "<p>Real paper content.</p>"
        "</article>"
        "<footer>Built with ar5iv</footer>"
        "</body></html>"
    )
    md, _figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert "Real paper content." in md
    assert "Report GitHub Issue" not in md
    assert "Submit without GitHub" not in md
    assert "Built with ar5iv" not in md
    assert "arXiv" not in md


def test_parse_html_falls_back_to_full_doc_when_no_article():
    """When there's no <article> wrapper, parse the whole document."""
    html = (
        "<html><body>"
        "<h2 class='ltx_title'>1 Introduction</h2>"
        "<p>Content without article wrapper.</p>"
        "</body></html>"
    )
    md, _figures = parse_html(html, page_url="https://arxiv.org/html/2501.00001v1")
    assert "Content without article wrapper." in md


# ---------------------------------------------------------------------------
# fetch_and_parse — network fetch + image download
# ---------------------------------------------------------------------------

import os  # noqa: E402
from unittest.mock import patch  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from ppagent.arxiv_html import (  # noqa: E402
    HtmlUnavailable,
    ParseError,
    ParsedHtml,
    fetch_and_parse,
)


class _FakeResponse:
    """Minimal httpx.Response stand-in for fetch_and_parse tests."""

    def __init__(self, status_code: int, text: str = "", content: bytes = b""):
        self.status_code = status_code
        self.text = text
        self.content = content
        # `fetch_and_parse` reads `resp.url` to resolve relative image URLs.
        self.url = httpx.URL("https://arxiv.org/html/2501.00001v1")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=self
            )


def _client_returning(*responses: _FakeResponse):
    """Build a fake httpx.Client substitute returning the given responses in order.

    Raises if the code under test makes more requests than responses provided,
    so tests fail loudly on unexpected extra fetches.
    """

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            self._queue = list(responses)
            self._calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, *args, **kwargs):
            if self._calls >= len(self._queue):
                raise AssertionError(
                    f"unexpected extra HTTP request #{self._calls + 1}: {url}"
                )
            r = self._queue[self._calls]
            self._calls += 1
            if isinstance(r, Exception):
                raise r
            return r

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
    fake_client = _client_returning(
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
    fake_client = _client_returning(
        _FakeResponse(200, html),
        _FakeResponse(200, "", b"\xff\xd8\xff\xe0"),  # JPEG-ish bytes
    )
    with patch("ppagent.arxiv_html.httpx.Client", fake_client):
        result = fetch_and_parse("2501.00001", tmp_path)
    assert result.figures[0].image_path == "figures/figure_1.jpg"
    assert (tmp_path / "figures/figure_1.jpg").exists()


def test_fetch_and_parse_respects_max_figures(tmp_path):
    """3 figures in HTML, max_figures=1 → only first downloaded + returned,
    and only one image fetch is made."""
    html = (
        "<html><body><article>"
        "<h2 class='ltx_title'>2 Method</h2>"
        + "".join(
            f"<figure><img src='x{i}.png'/><figcaption>F{i}</figcaption></figure>"
            for i in (1, 2, 3)
        )
        + "</article></body></html>"
    )
    fake_client = _client_returning(
        _FakeResponse(200, html),
        _FakeResponse(200, "", b"\x89PNG"),  # only ONE image fetch expected
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

    class _BoomClient:
        def __init__(self, *args, **kwargs):
            self._n = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, *args, **kwargs):
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
    fake_client = _client_returning(
        httpx.HTTPStatusError("404", request=None, response=_FakeResponse(404, ""))
    )
    with patch("ppagent.arxiv_html.httpx.Client", fake_client):
        with pytest.raises(HtmlUnavailable):
            fetch_and_parse("9999.99999", tmp_path)


def test_fetch_and_parse_raises_html_unavailable_on_connection_error(tmp_path):
    fake_client = _client_returning(httpx.ConnectError("network down"))
    with patch("ppagent.arxiv_html.httpx.Client", fake_client):
        with pytest.raises(HtmlUnavailable):
            fetch_and_parse("2501.00001", tmp_path)


def test_fetch_and_parse_raises_parse_error_on_empty_body(tmp_path):
    fake_client = _client_returning(
        _FakeResponse(200, "<html><body><article></article></body></html>")
    )
    with patch("ppagent.arxiv_html.httpx.Client", fake_client):
        with pytest.raises(ParseError):
            fetch_and_parse("2501.00001", tmp_path)


@pytest.mark.skipif(
    not os.environ.get("PPA_LIVE_NETWORK"),
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
