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
