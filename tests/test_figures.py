"""Tests for figure extraction and selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from ppagent.agents.figure_selector import FigureSelectorAgent
from ppagent.figures import Figure, extract_figures

# The cached paper PDF shipped with the repo for this paper.
CACHED_PDF = Path(__file__).resolve().parent.parent / ".cache" / "pdfs" / "2606.01075.pdf"


@pytest.fixture
def tmp_report_dir(tmp_path):
    return tmp_path


@pytest.mark.skipif(not CACHED_PDF.exists(), reason="cached test PDF not available")
def test_extract_figures_finds_captioned_figures(tmp_report_dir):
    """Extraction should find Figure 1..4 and write PNGs into figures/ subdir."""
    figures = extract_figures(CACHED_PDF, tmp_report_dir)

    assert len(figures) >= 1
    numbers = {f.figure_number for f in figures}
    assert 1 in numbers  # Figure 1 (taxonomy/pipeline) should always be present

    for fig in figures:
        assert fig.caption  # non-empty caption
        assert fig.image_path.startswith("figures/figure_")
        assert fig.image_path.endswith(".png")
        # File should actually exist on disk at <report_dir>/<image_path>
        abs_path = tmp_report_dir / fig.image_path
        assert abs_path.exists()
        assert abs_path.stat().st_size > 0


@pytest.mark.skipif(not CACHED_PDF.exists(), reason="cached test PDF not available")
def test_extract_figures_returns_empty_for_missing_pdf(tmp_report_dir):
    assert extract_figures(Path("/nonexistent/x.pdf"), tmp_report_dir) == []


def test_figure_selector_no_figures(mock_llm_client_unused):
    """With zero candidates the selector returns None without an LLM call."""
    cfg = _minimal_config()
    agent = FigureSelectorAgent(mock_llm_client_unused, cfg)
    result = agent.run(figures=[], base_dir=Path("/tmp"))

    assert result.success
    assert result.data["selected_figure"] is None
    assert result.data["figures"] == []


def test_figure_selector_single_candidate_skips_llm(tmp_path):
    """A single figure is selected directly — no LLM call should happen."""
    fig = Figure(figure_number=3, caption="only one", image_path="figures/figure_3.png")

    class BoomLLM:
        def reset_usage(self):
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def get_usage(self):
            return self.usage

        def chat_vision(self, *a, **k):
            raise AssertionError("chat_vision must not be called for a single figure")

    agent = FigureSelectorAgent(BoomLLM(), _minimal_config())  # type: ignore[arg-type]
    result = agent.run(figures=[fig], base_dir=tmp_path)

    assert result.success
    assert result.data["selected_figure"] is fig


def test_figure_selector_parses_llm_choice(tmp_path):
    """The selector should map the LLM's JSON figure_number to the right Figure."""
    figs = [
        Figure(figure_number=1, caption="taxonomy", image_path="figures/figure_1.png"),
        Figure(figure_number=2, caption="plots", image_path="figures/figure_2.png"),
    ]
    # Copy real PNG bytes so chat_vision can read them.
    if CACHED_PDF.exists():
        extract_figures(CACHED_PDF, tmp_path)
        # ensure both figure files exist (may differ in numbering); create stubs
    for f in figs:
        p = tmp_path / f.image_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal stub

    class FakeLLM:
        def reset_usage(self):
            self.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        def get_usage(self):
            return self.usage

        def chat_vision(self, system, user_text, images):
            return 'Here is my choice: {"figure_number": 1, "reason": "overview"}'

    agent = FigureSelectorAgent(FakeLLM(), _minimal_config())  # type: ignore[arg-type]
    result = agent.run(figures=figs, base_dir=tmp_path)

    assert result.success
    chosen = result.data["selected_figure"]
    assert chosen is not None
    assert chosen.figure_number == 1


def test_figure_selector_llm_failure_falls_back(tmp_path):
    """If the LLM call raises, we fall back to the lowest-numbered figure."""
    figs = [
        Figure(figure_number=2, caption="b", image_path="figures/figure_2.png"),
        Figure(figure_number=5, caption="e", image_path="figures/figure_5.png"),
    ]
    for f in figs:
        p = tmp_path / f.image_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x89PNG\r\n\x1a\n")

    class BrokenLLM:
        def reset_usage(self):
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def get_usage(self):
            return self.usage

        def chat_vision(self, *a, **k):
            raise RuntimeError("vision endpoint down")

    agent = FigureSelectorAgent(BrokenLLM(), _minimal_config())  # type: ignore[arg-type]
    result = agent.run(figures=figs, base_dir=tmp_path)

    assert result.success
    assert result.data["selected_figure"].figure_number == 2  # lowest-numbered


# --- helpers ---


def _minimal_config():
    from ppagent.config import AppConfig

    return AppConfig()


@pytest.fixture
def mock_llm_client_unused():
    """An LLM client stub that is never expected to be called in no-figure path."""

    class UnusedLLM:
        def reset_usage(self):
            self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def get_usage(self):
            return self.usage

    return UnusedLLM()
