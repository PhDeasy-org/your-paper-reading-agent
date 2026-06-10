"""Assembler — combines all agent outputs into final Markdown + HTML reports."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ppagent.models import AgentResult, Paper, PaperReport, ReportSection
from ppagent.storage import Storage

logger = logging.getLogger(__name__)


class Assembler:
    """Deterministic combiner that assembles agent outputs into final reports.

    This is NOT an LLM agent — it simply validates, orders, and renders sections.
    """

    def __init__(self, template_dir: Path, storage: Storage, model_used: str = "") -> None:
        self.storage = storage
        self.model_used = model_used
        if template_dir.is_dir():
            self.env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=False,
            )
        else:
            self.env = None
            logger.warning("Template directory not found: %s", template_dir)

    def assemble(
        self,
        *,
        paper: Paper,
        writer_result: AgentResult,
        finder_result: AgentResult,
        criticizer_result: AgentResult,
    ) -> tuple[PaperReport, str, str]:
        """Assemble all agent results into a PaperReport + rendered Markdown + HTML.

        Returns (report, md_content, html_content).
        """
        w = writer_result.data if writer_result.success else {}
        f = finder_result.data if finder_result.success else {}
        c = criticizer_result.data if criticizer_result.success else {}

        # Build sections
        metadata = ReportSection(
            name="metadata",
            content=self._build_metadata_text(paper, w),
        )
        benchmarks = ReportSection(
            name="benchmarks",
            content=w.get("benchmarks", "None reported."),
        )
        tldr = ReportSection(
            name="tldr",
            content=w.get("tldr", "TL;DR generation failed."),
        )
        previous_works = ReportSection(
            name="previous_works",
            content=w.get("previous_works", "Previous works summary unavailable."),
        )
        method = ReportSection(
            name="method",
            content=w.get("method", "Method details unavailable."),
        )
        evaluation = ReportSection(
            name="evaluation",
            content=w.get("evaluation", "Evaluation details unavailable."),
        )
        critique = ReportSection(
            name="critique",
            content=c.get("critique", "Critical analysis unavailable."),
        )

        report = PaperReport(
            paper=paper,
            metadata=metadata,
            benchmarks=benchmarks,
            tldr=tldr,
            previous_works=previous_works,
            method=method,
            evaluation=evaluation,
            critique=critique,
            related_works=f.get("related_works", []),
            generated_at=datetime.now(),
            model_used=self.model_used,
        )

        # Render templates
        md_content = self._render_md(report, w, f)
        html_content = self._render_html(report, w, f, md_content)

        # Save to disk
        self.storage.save_report(report, md_content=md_content, html_content=html_content)

        return report, md_content, html_content

    def _build_metadata_text(self, paper: Paper, writer_data: dict) -> str:
        """Build the metadata section text."""
        lines = [
            f"| **Paper** | [arXiv:{paper.id}]({paper.arxiv_url}) |",
            f"| **HuggingFace** | [Link]({paper.hf_url}) |",
            f"| **Published** | {paper.published_at.strftime('%Y-%m-%d') if paper.published_at else 'Unknown'} |",
            f"| **Authors** | {', '.join(paper.authors)} |",
            f"| **Affiliations** | {', '.join(writer_data.get('affiliations', ['N/A']))} |",
            f"| **Keywords** | {', '.join(writer_data.get('keywords', []))} |",
        ]
        return "\n".join(lines)

    def _template_context(self, report: PaperReport, writer_data: dict, finder_data: dict) -> dict:
        return {
            "paper": report.paper,
            "metadata": report.metadata,
            "benchmarks": report.benchmarks,
            "tldr": report.tldr,
            "previous_works": report.previous_works,
            "method": report.method,
            "evaluation": report.evaluation,
            "critique": report.critique,
            "related_works": report.related_works,
            "generated_at": report.generated_at,
            "model_used": report.model_used,
            "keywords": writer_data.get("keywords", []),
            "affiliations": writer_data.get("affiliations", []),
            "finder_narrative": finder_data.get("narrative", ""),
        }

    def _render_md(self, report: PaperReport, writer_data: dict, finder_data: dict) -> str:
        """Render the Markdown report."""
        if self.env:
            try:
                template = self.env.get_template("report.md.jinja2")
                return template.render(**self._template_context(report, writer_data, finder_data))
            except Exception as exc:
                logger.warning("MD template rendering failed: %s — using fallback", exc)

        return self._fallback_md(report)

    def _render_html(self, report: PaperReport, writer_data: dict, finder_data: dict, md_content: str) -> str:
        """Render the HTML report."""
        if self.env:
            try:
                template = self.env.get_template("report.html.jinja2")
                return template.render(
                    **self._template_context(report, writer_data, finder_data),
                    markdown_content=md_content,
                )
            except Exception as exc:
                logger.warning("HTML template rendering failed: %s — using fallback", exc)

        return self._fallback_html(report, md_content)

    def _fallback_md(self, report: PaperReport) -> str:
        """Fallback Markdown when template is unavailable."""
        p = report.paper
        return f"""\
# {p.title}

> **TL;DR**: {report.tldr.content}

| Field | Value |
|-------|-------|
| **Paper** | [arXiv:{p.id}]({p.arxiv_url}) |
| **Published** | {p.published_at.strftime('%Y-%m-%d') if p.published_at else 'Unknown'} |
| **Authors** | {', '.join(p.authors)} |

## Benchmarks
{report.benchmarks.content}

## Previous Work & Limitations
{report.previous_works.content}

## Method
{report.method.content}

## Performance Evaluation
{report.evaluation.content}

## Critical Analysis
{report.critique.content}

## Related Papers
{chr(10).join(f'- [{rp.title}]({rp.arxiv_url})' for rp in report.related_works) or 'None found.'}

---
*Generated by ppagent on {report.generated_at.strftime('%Y-%m-%d %H:%M')} using {report.model_used}*
"""

    def _fallback_html(self, report: PaperReport, md_content: str) -> str:
        """Fallback HTML when template is unavailable."""
        import markdown as md_lib
        body = md_lib.markdown(md_content, extensions=["tables", "fenced_code"])
        return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{report.paper.title} — ppagent Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; color: #333; }}
  h1 {{ border-bottom: 2px solid #2563eb; padding-bottom: 0.5rem; }}
  h2 {{ color: #1e40af; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
  blockquote {{ border-left: 4px solid #2563eb; margin-left: 0; padding-left: 1rem; color: #555; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 2rem 0; }}
  a {{ color: #2563eb; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
