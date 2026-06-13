"""Assembler — combines all agent outputs into final Markdown + HTML reports."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ppagent.models import AgentResult, Paper, PaperReport, ReportSection
from ppagent.storage import Storage

logger = logging.getLogger(__name__)

LLM_PRICING = {
    "deepseek": {
        "provider": "DeepSeek",
        "model": "DeepSeek-V4-Pro",
        "input_price": 0.435,  # per 1M tokens
        "output_price": 0.87,  # per 1M tokens
    },
    "qwen": {
        "provider": "Qwen",
        "model": "Qwen-Plus",
        "input_price": 0.40,
        "output_price": 1.20,
    },
    "openai": {
        "provider": "OpenAI",
        "model": "GPT-5.5 / GPT-4o-mini",
        "input_price": 5.00,
        "output_price": 30.00,
    },
    "anthropic": {
        "provider": "Anthropic",
        "model": "Claude Fable 5 / Sonnet 4.6",
        "input_price": 10.00,
        "output_price": 50.00,
    },
    "google": {
        "provider": "Google",
        "model": "Gemini 3.1 Pro / 3.5 Flash",
        "input_price": 2.00,
        "output_price": 12.00,
    },
    "grok": {
        "provider": "Grok",
        "model": "Grok 4.3 (Flagship)",
        "input_price": 1.25,
        "output_price": 2.50,
    },
    "kimi": {
        "provider": "Kimi",
        "model": "Kimi K2.7 Code",
        "input_price": 0.95,
        "output_price": 4.00,
    },
    "glm": {
        "provider": "GLM",
        "model": "Zhipu GLM-5",
        "input_price": 1.00,
        "output_price": 3.20,
    },
    "minimax": {
        "provider": "MiniMax",
        "model": "MiniMax M3",
        "input_price": 0.30,
        "output_price": 1.20,
    },
    "bytedance-seed": {
        "provider": "ByteDance-Seed",
        "model": "Seed-2.0-Mini",
        "input_price": 0.10,
        "output_price": 0.40,
    },
    "mistral": {
        "provider": "Mistral",
        "model": "Mistral Large 3",
        "input_price": 2.00,
        "output_price": 6.00,
    },
    "mimo": {
        "provider": "Mimo",
        "model": "Mimo-v2.5",
        "input_price": 0.14,
        "output_price": 0.28,
    },
    "tencent-hy": {
        "provider": "Tencent-Hy",
        "model": "Tencent Hy3",
        "input_price": 0.06,
        "output_price": 0.21,
    },
    "stepfun": {
        "provider": "Stepfun",
        "model": "Step-3",
        "input_price": 0.57,
        "output_price": 1.42,
    },
}

def calculate_cost(model_name: str, usage: dict[str, int]) -> dict[str, Any] | None:
    m = model_name.lower()
    matched_key = None
    
    if "deepseek" in m:
        matched_key = "deepseek"
    elif "qwen" in m:
        matched_key = "qwen"
    elif "gpt" in m or "openai" in m:
        matched_key = "openai"
        if "mini" in m or "nano" in m:
            input_price = 0.15
            output_price = 0.60
            input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * input_price
            output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * output_price
            return {
                "provider": "OpenAI",
                "model": model_name,
                "input_price": input_price,
                "output_price": output_price,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost
            }
    elif "claude" in m or "anthropic" in m:
        matched_key = "anthropic"
        if "sonnet" in m:
            input_price = 3.00
            output_price = 15.00
            input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * input_price
            output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * output_price
            return {
                "provider": "Anthropic",
                "model": model_name,
                "input_price": input_price,
                "output_price": output_price,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost
            }
    elif "gemini" in m or "google" in m:
        matched_key = "google"
        if "flash" in m:
            input_price = 1.50
            output_price = 9.00
            input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * input_price
            output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * output_price
            return {
                "provider": "Google",
                "model": model_name,
                "input_price": input_price,
                "output_price": output_price,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost
            }
    elif "grok" in m:
        matched_key = "grok"
    elif "kimi" in m or "moonshot" in m:
        matched_key = "kimi"
    elif "glm" in m or "zhipu" in m:
        matched_key = "glm"
    elif "minimax" in m or "abab" in m:
        matched_key = "minimax"
    elif "doubao" in m or "seed" in m:
        matched_key = "bytedance-seed"
    elif "mistral" in m:
        matched_key = "mistral"
    elif "mimo" in m:
        matched_key = "mimo"
    elif "hunyuan" in m or "tencent" in m or "hy-" in m:
        matched_key = "tencent-hy"
    elif "stepfun" in m or "step-" in m:
        matched_key = "stepfun"

    if matched_key:
        info = LLM_PRICING[matched_key]
        ip = info["input_price"]
        op = info["output_price"]
        input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * ip
        output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * op
        return {
            "provider": info["provider"],
            "model": model_name,
            "input_price": ip,
            "output_price": op,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost
        }
    return None

def render_markdown_with_math(text: str) -> str:
    """Renders markdown while protecting LaTeX math blocks from being escaped or mangled."""
    import markdown as md_lib

    placeholders = {}

    # Match block math: $$...$$ and \[...\]
    block_pattern = re.compile(r'(\$\$.*?\$\$|\\\[.*?\\\])', re.DOTALL)
    # Match inline math: $...$ and \(...\)
    inline_pattern = re.compile(r'(\$(?!\s)[^\$\n]+?(?<!\s)\$|\\\(.*?\\\))')

    temp_text = text or ""

    def replace_match(match: re.Match) -> str:
        placeholder = f"<!--MATH_PLACEHOLDER_{len(placeholders)}-->"
        placeholders[placeholder] = match.group(0)
        return placeholder

    # Replace block math first, then inline math
    temp_text = block_pattern.sub(replace_match, temp_text)
    temp_text = inline_pattern.sub(replace_match, temp_text)

    # Render markdown
    html = md_lib.markdown(temp_text, extensions=["tables", "fenced_code"])

    # Restore math blocks
    for placeholder, original in placeholders.items():
        html = html.replace(placeholder, original)

    return html


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
            try:
                self.env.filters["markdown"] = render_markdown_with_math
            except Exception:
                pass
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

        # Aggregate token usage
        results = [writer_result, finder_result, criticizer_result]
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for res in results:
            if res and res.usage:
                for k in usage:
                    usage[k] += res.usage.get(k, 0)

        # Calculate cost
        cost_report = calculate_cost(self.model_used, usage)

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
            usage=usage,
            cost_report=cost_report,
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
            "usage": report.usage,
            "cost_report": report.cost_report,
            "price_table": LLM_PRICING,
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
        body = render_markdown_with_math(md_content)
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
<script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
      displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
      processEscapes: true,
      processEnvironments: true
    }},
    options: {{
      skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
    }}
  }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" async></script>
</head>
<body>
{body}
</body>
</html>
"""
