"""Tests for paper-type prompts, section labels, and assembler rendering of all paper types."""

from __future__ import annotations

from pathlib import Path
import pytest

from ppagent.agents.assembler import Assembler
from ppagent.agents.prompts import (
    PAPER_TYPES,
    WRITER_SYSTEM_PROMPTS,
    WRITER_SECTION_LABELS,
)
from ppagent.models import AgentResult, Paper
from ppagent.storage import Storage


def test_prompts_completeness():
    """Verify that every defined paper type has corresponding prompts and section labels."""
    from ppagent.agents.prompts import CRITICIZER_SYSTEM_PROMPTS

    for paper_type in PAPER_TYPES:
        assert paper_type in WRITER_SYSTEM_PROMPTS, f"Missing writer system prompt for {paper_type}"
        assert paper_type in CRITICIZER_SYSTEM_PROMPTS, f"Missing criticizer system prompt for {paper_type}"
        assert paper_type in WRITER_SECTION_LABELS, f"Missing section labels for {paper_type}"

        # Verify structure of section labels
        labels = WRITER_SECTION_LABELS[paper_type]
        assert "benchmarks_heading" in labels
        assert "method_heading" in labels
        assert "evaluation_heading" in labels


def test_assembler_renders_all_paper_types(sample_paper, writer_data, finder_data, template_dir, tmp_path):
    """Test that assembling a report with any of the paper types renders the correct dynamic headings in MD and HTML."""
    storage = Storage(tmp_path)
    
    writer_result = AgentResult(agent_name="writer", success=True, data=writer_data)
    finder_result = AgentResult(agent_name="finder", success=True, data=finder_data)
    criticizer_result = AgentResult(agent_name="criticizer", success=True, data={"critique": "Critique detail"})

    for paper_type in PAPER_TYPES:
        assembler = Assembler(
            template_dir=template_dir,
            storage=storage,
            model_map={
                "classifier": "mock-text",
                "writer": "mock-text",
                "finder": "mock-text",
                "criticizer": "mock-text",
            },
        )

        report, md_content, html_content = assembler.assemble(
            paper=sample_paper,
            writer_result=writer_result,
            finder_result=finder_result,
            criticizer_result=criticizer_result,
            paper_type=paper_type,
        )

        assert report.paper_type == paper_type

        # Verify markdown contains the correct headings for this paper type
        labels = WRITER_SECTION_LABELS[paper_type]
        assert f"## {labels['method_heading']}" in md_content
        assert f"## {labels['evaluation_heading']}" in md_content

        # Verify HTML contains the correct headings
        assert labels['benchmarks_heading'] in html_content
        assert labels['method_heading'] in html_content
        assert labels['evaluation_heading'] in html_content

        # For non-default paper types, verify the paper type metadata is shown in the output
        if paper_type != "method":
            # Just verify paper_type is referenced in some form
            assert paper_type in md_content.lower() or PAPER_TYPES[paper_type].split(" —")[0].lower() in md_content.lower()
            assert paper_type in html_content.lower() or PAPER_TYPES[paper_type].split(" —")[0].lower() in html_content.lower()
