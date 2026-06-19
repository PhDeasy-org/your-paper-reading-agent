"""Tests for figure extraction and selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from ppagent.agents.figure_selector import FigureSelectorAgent
from ppagent.figures import Figure, extract_figures

# The cached paper PDF shipped with the repo for this paper.
CACHED_PDF = (
    Path(__file__).resolve().parent.parent / ".cache" / "pdfs" / "2606.01075.pdf"
)


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
    """With zero candidates the selector returns an empty list without an LLM call."""
    cfg = _minimal_config()
    agent = FigureSelectorAgent(mock_llm_client_unused, cfg)
    result = agent.run(figures=[], base_dir=Path("/tmp"))

    assert result.success
    assert result.data["selected_figures"] == []
    assert result.data["figures"] == []


def test_figure_selector_single_candidate_calls_llm(tmp_path):
    """A single figure still goes through the LLM (the LLM may decline it)."""
    fig = Figure(figure_number=3, caption="only one", image_path="figures/figure_3.png")
    p = tmp_path / fig.image_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\n")

    class AcceptLLM:
        def reset_usage(self):
            self.usage = {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
            }

        def get_usage(self):
            return self.usage

        def chat_vision(self, system, user_text, images):
            return '{"selected": [{"figure_number": 3, "section": "method", "reason": "overview"}], "none_reason": ""}'

    agent = FigureSelectorAgent(AcceptLLM(), _minimal_config())  # type: ignore[arg-type]
    result = agent.run(figures=[fig], base_dir=tmp_path)

    assert result.success
    assert len(result.data["selected_figures"]) == 1
    assert result.data["selected_figures"][0].figure is fig
    assert result.data["selected_figures"][0].section == "method"


def test_figure_selector_parses_llm_choice(tmp_path):
    """The selector should map the LLM's JSON selected list to SelectedFigure objects."""
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
            self.usage = {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }

        def get_usage(self):
            return self.usage

        def chat_vision(self, system, user_text, images):
            return '{"selected": [{"figure_number": 1, "section": "method", "reason": "overview"}], "none_reason": ""}'

    agent = FigureSelectorAgent(FakeLLM(), _minimal_config())  # type: ignore[arg-type]
    result = agent.run(figures=figs, base_dir=tmp_path)

    assert result.success
    selected = result.data["selected_figures"]
    assert len(selected) == 1
    assert selected[0].figure.figure_number == 1
    assert selected[0].section == "method"


def test_figure_selector_llm_failure_returns_empty(tmp_path):
    """If the LLM call raises, the selector returns an empty selection (no forced fallback)."""
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
    assert result.data["selected_figures"] == []
    assert "LLM call failed" in (result.data.get("none_reason") or "")


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
