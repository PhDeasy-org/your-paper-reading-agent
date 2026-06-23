"""Tests for LaTeX math rendering in report generation.

Covers:
- render_markdown_with_math(): placeholder-based math preservation
- HTML template MathJax configuration
- Assembler integration with LaTeX-rich report sections
- Edge cases: empty input, dollar signs in text, special characters
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ppagent.agents.assembler import Assembler, render_markdown_with_math
from ppagent.models import AgentResult, PaperReport
from ppagent.storage import Storage


# ---------------------------------------------------------------------------
# render_markdown_with_math — core function
# ---------------------------------------------------------------------------


class TestInlineMathPreservation:
    """Inline math ($...$) must survive markdown rendering intact."""

    def test_simple_inline_math(self):
        text = "The variable $x$ is important."
        html = render_markdown_with_math(text)
        assert "$x$" in html

    def test_inline_math_with_subscript(self):
        text = "We use $x_i$ and $y_j$ as indices."
        html = render_markdown_with_math(text)
        assert "$x_i$" in html
        assert "$y_j$" in html

    def test_inline_math_with_superscript(self):
        text = "The term $\\alpha^2$ appears in the equation."
        html = render_markdown_with_math(text)
        assert "$\\alpha^2$" in html

    def test_inline_math_with_greek_letters(self):
        text = "Parameters $\\beta$, $\\gamma$, and $\\theta$ are learned."
        html = render_markdown_with_math(text)
        assert "$\\beta$" in html
        assert "$\\gamma$" in html
        assert "$\\theta$" in html

    def test_inline_math_with_cal(self):
        text = "The model $\\mathcal{M}$ is trained end-to-end."
        html = render_markdown_with_math(text)
        assert "$\\mathcal{M}$" in html

    def test_inline_math_complex_expression(self):
        text = "We compute $\\frac{\\partial L}{\\partial \\theta}$ via backprop."
        html = render_markdown_with_math(text)
        assert "$\\frac{\\partial L}{\\partial \\theta}$" in html

    def test_inline_math_with_text_command(self):
        text = "Score is $\\text{BLEU} = 28.4$."
        html = render_markdown_with_math(text)
        assert "$\\text{BLEU} = 28.4$" in html

    def test_multiple_inline_math_same_line(self):
        text = "Given $Q$, $K$, and $V$, compute attention."
        html = render_markdown_with_math(text)
        assert "$Q$" in html
        assert "$K$" in html
        assert "$V$" in html

    def test_inline_math_attention_formula(self):
        text = "The attention is $\\text{Attention}(Q,K,V) = \\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V$."
        html = render_markdown_with_math(text)
        assert (
            "$\\text{Attention}(Q,K,V) = \\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V$"
            in html
        )

    def test_inline_math_paren_delimiters(self):
        text = "The variable \\(x\\) is important."
        html = render_markdown_with_math(text)
        assert "\\(x\\)" in html

    def test_inline_math_paren_complex(self):
        text = "We compute \\(\\alpha + \\beta\\) here."
        html = render_markdown_with_math(text)
        assert "\\(\\alpha + \\beta\\)" in html


class TestBlockMathPreservation:
    """Block/display math ($$...$$) must survive markdown rendering intact."""

    def test_simple_block_math(self):
        text = "The equation:\n\n$$E = mc^2$$\n\nis famous."
        html = render_markdown_with_math(text)
        assert "$$E = mc^2$$" in html

    def test_block_math_multiline(self):
        text = "We have:\n\n$$\\sum_{i=1}^{n} x_i = \\frac{n(n+1)}{2}$$\n\nDone."
        html = render_markdown_with_math(text)
        assert "$$\\sum_{i=1}^{n} x_i = \\frac{n(n+1)}{2}$$" in html

    def test_block_math_multihead_attention(self):
        expr = "$$\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1,\\dots,\\text{head}_h)W^O$$"
        text = f"Multi-head:\n\n{expr}\n\nDone."
        html = render_markdown_with_math(text)
        assert expr in html

    def test_block_math_with_frac(self):
        expr = "$$\\frac{\\partial \\mathcal{L}}{\\partial \\theta} = \\sum_{t} \\nabla_\\theta \\log \\pi_\\theta(a_t|s_t) R_t$$"
        text = f"Policy gradient:\n\n{expr}\n"
        html = render_markdown_with_math(text)
        assert expr in html

    def test_block_math_bracket_delimiters(self):
        text = "Equation:\n\n\\[E = mc^2\\]\n\nDone."
        html = render_markdown_with_math(text)
        assert "\\[E = mc^2\\]" in html

    def test_block_math_bracket_complex(self):
        expr = "\\[\\mathcal{L} = -\\sum_{i} y_i \\log \\hat{y}_i\\]"
        text = f"Loss:\n\n{expr}\n"
        html = render_markdown_with_math(text)
        assert expr in html

    def test_multiple_block_equations(self):
        eq1 = "$$a = b + c$$"
        eq2 = "$$d = e + f$$"
        text = f"First:\n\n{eq1}\n\nSecond:\n\n{eq2}\n"
        html = render_markdown_with_math(text)
        assert eq1 in html
        assert eq2 in html


class TestMixedMathAndMarkdown:
    """Markdown formatting must work alongside preserved math."""

    def test_bold_with_inline_math(self):
        text = "The **model $\\mathcal{M}$** is trained."
        html = render_markdown_with_math(text)
        assert "<strong>" in html
        assert "$\\mathcal{M}$" in html

    def test_italic_with_inline_math(self):
        text = "The *variable $x$* is key."
        html = render_markdown_with_math(text)
        assert "<em>" in html
        assert "$x$" in html

    def test_markdown_link_preserved_with_math(self):
        text = "See [the paper](https://arxiv.org) for $\\alpha$ details."
        html = render_markdown_with_math(text)
        assert '<a href="https://arxiv.org">' in html
        assert "$\\alpha$" in html

    def test_list_with_inline_math(self):
        text = "Methods:\n\n- First uses $\\alpha$\n- Second uses $\\beta$"
        html = render_markdown_with_math(text)
        assert "$\\alpha$" in html
        assert "$\\beta$" in html
        assert "<li>" in html

    def test_heading_with_inline_math(self):
        text = "## The $\\mathcal{L}$ Function\n\nDetails here."
        html = render_markdown_with_math(text)
        assert "<h2>" in html
        assert "$\\mathcal{L}$" in html

    def test_paragraph_separation_with_math(self):
        text = "First paragraph with $x$.\n\nSecond paragraph with $y$."
        html = render_markdown_with_math(text)
        assert "$x$" in html
        assert "$y$" in html
        assert html.count("<p>") == 2

    def test_table_with_math_in_cells(self):
        text = "| Model | Score |\n|-------|-------|\n| Baseline | $28.4$ |\n| Ours | $30.1$ |"
        html = render_markdown_with_math(text)
        assert "$28.4$" in html
        assert "$30.1$" in html
        assert "<table>" in html or "<table" in html

    def test_code_block_not_affected_by_math(self):
        text = "```python\nx = 5\n```\n\nAnd $y$ is a variable."
        html = render_markdown_with_math(text)
        assert "$y$" in html
        assert "language-python" in html or "<code" in html


class TestEdgeCases:
    """Edge cases and potential failure modes."""

    def test_empty_string(self):
        html = render_markdown_with_math("")
        assert html == ""

    def test_none_input(self):
        html = render_markdown_with_math(None)
        assert html == ""

    def test_no_math_content(self):
        text = "Just a regular paragraph with no math."
        html = render_markdown_with_math(text)
        assert "regular paragraph" in html

    def test_dollar_sign_not_math(self):
        text = "The price is $5 and the cost is $10."
        html = render_markdown_with_math(text)
        # Should not be treated as math: "$5 and the cost is $" would be wrong
        assert "$5" in html or "5" in html

    def test_math_at_start_of_line(self):
        text = "$x$ is the first token."
        html = render_markdown_with_math(text)
        assert "$x$" in html

    def test_math_at_end_of_line(self):
        text = "The result is $x$"
        html = render_markdown_with_math(text)
        assert "$x$" in html

    def test_math_with_adjacent_punctuation(self):
        text = "Compute ($x$), then use [$y$]; finally $z$."
        html = render_markdown_with_math(text)
        assert "$x$" in html
        assert "$y$" in html
        assert "$z$" in html

    def test_consecutive_inline_math(self):
        text = "$a$$b$ should render both."
        html = render_markdown_with_math(text)
        # Both should be present — the regex must handle adjacent math
        assert "$a$" in html
        assert "$b$" in html

    def test_block_math_with_internal_single_dollar(self):
        text = "$$\\text{cost} = \\$5 + x$$\n\nDone."
        html = render_markdown_with_math(text)
        # The block math must be preserved as a whole
        assert "$$" in html

    def test_only_whitespace_between_dollars(self):
        text = "Not math: $ $ but this is: $x$."
        html = render_markdown_with_math(text)
        assert "$x$" in html

    def test_escaped_dollar(self):
        text = "Price is \\$5 and math is $x$."
        html = render_markdown_with_math(text)
        assert "$x$" in html

    def test_math_with_ampersand(self):
        text = "Align: $a & b$ in a table."
        html = render_markdown_with_math(text)
        assert "$a & b$" in html or "$a &amp; b$" in html or "$a & b$" in html

    def test_math_with_underscore_not_italicized(self):
        text = "Variable $x_i$ is the input feature."
        html = render_markdown_with_math(text)
        assert "$x_i$" in html
        # The underscore inside math must not produce markdown emphasis
        assert html.count("<em>") == 0

    def test_math_with_asterisk_not_bolded(self):
        text = "Product $a * b$ is the element-wise multiplication."
        html = render_markdown_with_math(text)
        assert "$a * b$" in html
        # The asterisk inside math must not produce markdown bold
        assert html.count("<strong>") == 0

    def test_long_complex_formula(self):
        formula = "$\\mathcal{L}_{\\text{DPO}}(\\pi_\\theta;\\pi_{\\text{ref}}) = -\\mathbb{E}\\big[\\log\\sigma(\\beta\\log\\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta\\log\\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)})\\big]$"
        text = f"Minimize {formula} during training."
        html = render_markdown_with_math(text)
        assert formula in html

    def test_realistic_method_section(self):
        """Test with content similar to actual generated reports."""
        text = (
            "The model $\\mathcal{M}$ uses attention where the generator $\\mathcal{G}$ "
            "produces $k$ candidates $\\{\\hat{y}_1,\\dots,\\hat{y}_k\\}$ per query $q$.\n\n"
            "The verifier outputs $\\mathcal{V}(q,\\hat{y}_i) \\in \\{\\text{Correct},\\text{Incorrect}\\}$. "
            "Threshold $\\tau \\in [0.5,0.8]$ controls the decision boundary."
        )
        html = render_markdown_with_math(text)
        assert "$\\mathcal{M}$" in html
        assert "$\\mathcal{G}$" in html
        assert "$k$" in html
        assert "$q$" in html
        assert "$\\tau \\in [0.5,0.8]$" in html


class TestMathNotMangledByMarkdown:
    """Verify that markdown processing doesn't escape or corrupt math content."""

    def test_underscore_in_math_not_escaped(self):
        text = "Index $x_{ij}$ is valid."
        html = render_markdown_with_math(text)
        assert "$x_{ij}$" in html
        assert "x_{ij}" not in html.replace("$x_{ij}$", "")

    def test_backslash_in_math_preserved(self):
        text = "Use $\\frac{a}{b}$ for fractions."
        html = render_markdown_with_math(text)
        assert "\\frac" in html

    def test_curly_braces_in_math_preserved(self):
        text = "Sum $\\sum_{i=1}^{n}$ is computed."
        html = render_markdown_with_math(text)
        assert "\\sum_{i=1}^{n}" in html

    def test_angle_brackets_in_math_preserved(self):
        text = "Expectation $\\langle x \\rangle$ is computed."
        html = render_markdown_with_math(text)
        # Angle brackets should not be HTML-escaped inside math
        assert "$\\langle x \\rangle$" in html

    def test_pipe_in_math_preserved(self):
        text = "Conditional $p(x|y)$ is defined."
        html = render_markdown_with_math(text)
        assert "$p(x|y)$" in html

    def test_hash_in_math_preserved(self):
        text = "The count is $#S|$ elements."
        html = render_markdown_with_math(text)
        assert "$#S|$" in html


# ---------------------------------------------------------------------------
# HTML template — MathJax configuration
# ---------------------------------------------------------------------------


class TestMathJaxConfiguration:
    """The HTML template must include correct MathJax config for client-side rendering."""

    @pytest.fixture
    def html_template_content(self) -> str:
        template_path = (
            Path(__file__).resolve().parent.parent / "templates" / "report.html.jinja2"
        )
        return template_path.read_text()

    def test_mathjax_script_included(self, html_template_content: str):
        assert "mathjax" in html_template_content.lower()
        assert (
            "tex-chtml" in html_template_content or "tex-mml" in html_template_content
        )

    def test_inline_math_delimiters_dollar(self, html_template_content: str):
        assert "['$', '$']" in html_template_content

    def test_inline_math_delimiters_paren(self, html_template_content: str):
        assert r"'\('" in html_template_content or "'\\\\('" in html_template_content

    def test_display_math_delimiters_dollar(self, html_template_content: str):
        assert "['$$', '$$']" in html_template_content

    def test_display_math_delimiters_bracket(self, html_template_content: str):
        assert r"'\['" in html_template_content or "'\\\\['" in html_template_content

    def test_process_escapes_enabled(self, html_template_content: str):
        assert "processEscapes" in html_template_content

    def test_skip_html_tags_configured(self, html_template_content: str):
        assert "skipHtmlTags" in html_template_content
        assert "pre" in html_template_content
        assert "code" not in html_template_content or "script" in html_template_content


class TestFallbackMathJaxConfiguration:
    """The fallback HTML generator must also have correct MathJax config."""

    def test_fallback_html_has_mathjax(
        self, sample_report: PaperReport, tmp_path: Path
    ):
        storage = Storage(tmp_path)
        assembler = Assembler(
            template_dir=Path("/nonexistent"), storage=storage, model_used="test"
        )
        md = assembler._fallback_md(sample_report)
        html = assembler._fallback_html(sample_report, md)

        assert "MathJax" in html
        assert "mathjax" in html.lower()

    def test_fallback_html_has_inline_delimiters(
        self, sample_report: PaperReport, tmp_path: Path
    ):
        storage = Storage(tmp_path)
        assembler = Assembler(
            template_dir=Path("/nonexistent"), storage=storage, model_used="test"
        )
        md = assembler._fallback_md(sample_report)
        html = assembler._fallback_html(sample_report, md)

        assert "['$', '$']" in html

    def test_fallback_html_has_display_delimiters(
        self, sample_report: PaperReport, tmp_path: Path
    ):
        storage = Storage(tmp_path)
        assembler = Assembler(
            template_dir=Path("/nonexistent"), storage=storage, model_used="test"
        )
        md = assembler._fallback_md(sample_report)
        html = assembler._fallback_html(sample_report, md)

        assert "['$$', '$$']" in html


# ---------------------------------------------------------------------------
# Assembler integration — full report rendering with LaTeX
# ---------------------------------------------------------------------------


class TestAssemblerMarkdownRendering:
    """The assembler must produce valid Markdown with preserved LaTeX."""

    def test_md_report_preserves_inline_math(
        self, sample_report, writer_data, finder_data, template_dir, tmp_path
    ):
        storage = Storage(tmp_path)
        assembler = Assembler(
            template_dir=template_dir, storage=storage, model_used="test"
        )

        writer_result = AgentResult(agent_name="writer", success=True, data=writer_data)
        finder_result = AgentResult(agent_name="finder", success=True, data=finder_data)
        criticizer_result = AgentResult(
            agent_name="criticizer", success=True, data={"critique": "OK"}
        )

        _, md_content, _ = assembler.assemble(
            paper=sample_report.paper,
            writer_result=writer_result,
            finder_result=finder_result,
            criticizer_result=criticizer_result,
        )

        assert "$Q$" in md_content
        assert "$K$" in md_content
        assert "$V$" in md_content

    def test_md_report_preserves_block_math(
        self, sample_report, writer_data, finder_data, template_dir, tmp_path
    ):
        storage = Storage(tmp_path)
        assembler = Assembler(
            template_dir=template_dir, storage=storage, model_used="test"
        )

        writer_result = AgentResult(agent_name="writer", success=True, data=writer_data)
        finder_result = AgentResult(agent_name="finder", success=True, data=finder_data)
        criticizer_result = AgentResult(
            agent_name="criticizer", success=True, data={"critique": "OK"}
        )

        _, md_content, _ = assembler.assemble(
            paper=sample_report.paper,
            writer_result=writer_result,
            finder_result=finder_result,
            criticizer_result=criticizer_result,
        )

        assert "$$\\text{MultiHead}" in md_content

    def test_md_report_has_sections(
        self, sample_report, writer_data, finder_data, template_dir, tmp_path
    ):
        storage = Storage(tmp_path)
        assembler = Assembler(
            template_dir=template_dir, storage=storage, model_used="test"
        )

        writer_result = AgentResult(agent_name="writer", success=True, data=writer_data)
        finder_result = AgentResult(agent_name="finder", success=True, data=finder_data)
        criticizer_result = AgentResult(
            agent_name="criticizer", success=True, data={"critique": "OK"}
        )

        _, md_content, _ = assembler.assemble(
            paper=sample_report.paper,
            writer_result=writer_result,
            finder_result=finder_result,
            criticizer_result=criticizer_result,
        )

        assert "## Method" in md_content
        assert "## Benchmarks" in md_content
        assert "## Performance Evaluation" in md_content


class TestAssemblerHTMLRendering:
    """The assembler must produce HTML with rendered markdown and preserved LaTeX."""

    def test_html_report_has_mathjax(
        self, sample_report, writer_data, finder_data, template_dir, tmp_path
    ):
        storage = Storage(tmp_path)
        assembler = Assembler(
            template_dir=template_dir, storage=storage, model_used="test"
        )

        writer_result = AgentResult(agent_name="writer", success=True, data=writer_data)
        finder_result = AgentResult(agent_name="finder", success=True, data=finder_data)
        criticizer_result = AgentResult(
            agent_name="criticizer", success=True, data={"critique": "OK"}
        )

        _, _, html_content = assembler.assemble(
            paper=sample_report.paper,
            writer_result=writer_result,
            finder_result=finder_result,
            criticizer_result=criticizer_result,
        )

        assert "MathJax" in html_content

    def test_html_report_preserves_inline_math_in_sections(
        self, sample_report, writer_data, finder_data, template_dir, tmp_path
    ):
        storage = Storage(tmp_path)
        assembler = Assembler(
            template_dir=template_dir, storage=storage, model_used="test"
        )

        writer_result = AgentResult(agent_name="writer", success=True, data=writer_data)
        finder_result = AgentResult(agent_name="finder", success=True, data=finder_data)
        criticizer_result = AgentResult(
            agent_name="criticizer", success=True, data={"critique": "OK"}
        )

        _, _, html_content = assembler.assemble(
            paper=sample_report.paper,
            writer_result=writer_result,
            finder_result=finder_result,
            criticizer_result=criticizer_result,
        )

        assert "$\\text{Attention}" in html_content
        assert "\\sqrt{d_k}" in html_content

    def test_html_report_preserves_block_math_in_sections(
        self, sample_report, writer_data, finder_data, template_dir, tmp_path
    ):
        storage = Storage(tmp_path)
        assembler = Assembler(
            template_dir=template_dir, storage=storage, model_used="test"
        )

        writer_result = AgentResult(agent_name="writer", success=True, data=writer_data)
        finder_result = AgentResult(agent_name="finder", success=True, data=finder_data)
        criticizer_result = AgentResult(
            agent_name="criticizer", success=True, data={"critique": "OK"}
        )

        _, _, html_content = assembler.assemble(
            paper=sample_report.paper,
            writer_result=writer_result,
            finder_result=finder_result,
            criticizer_result=criticizer_result,
        )

        assert "$$\\text{MultiHead}" in html_content

    def test_html_report_renders_markdown_in_sections(
        self, sample_report, writer_data, finder_data, template_dir, tmp_path
    ):
        storage = Storage(tmp_path)
        assembler = Assembler(
            template_dir=template_dir, storage=storage, model_used="test"
        )

        writer_result = AgentResult(agent_name="writer", success=True, data=writer_data)
        finder_result = AgentResult(agent_name="finder", success=True, data=finder_data)
        criticizer_result = AgentResult(
            agent_name="criticizer", success=True, data={"critique": "OK"}
        )

        _, _, html_content = assembler.assemble(
            paper=sample_report.paper,
            writer_result=writer_result,
            finder_result=finder_result,
            criticizer_result=criticizer_result,
        )

        # Markdown links in previous_works should be rendered to <a> tags
        assert '<a href="https://arxiv.org/abs/1409.0473">' in html_content

    def test_html_report_has_proper_structure(
        self, sample_report, writer_data, finder_data, template_dir, tmp_path
    ):
        storage = Storage(tmp_path)
        assembler = Assembler(
            template_dir=template_dir, storage=storage, model_used="test"
        )

        writer_result = AgentResult(agent_name="writer", success=True, data=writer_data)
        finder_result = AgentResult(agent_name="finder", success=True, data=finder_data)
        criticizer_result = AgentResult(
            agent_name="criticizer", success=True, data={"critique": "OK"}
        )

        _, _, html_content = assembler.assemble(
            paper=sample_report.paper,
            writer_result=writer_result,
            finder_result=finder_result,
            criticizer_result=criticizer_result,
        )

        assert "<!DOCTYPE html>" in html_content
        assert "<html" in html_content
        assert "</html>" in html_content
        assert "<h1>" in html_content
        assert "Attention Is All You Need" in html_content


class TestRenderMarkdownWithMathFilter:
    """Test the Jinja2 filter registration and behavior."""

    def test_filter_registered(self, template_dir, tmp_path):
        storage = Storage(tmp_path)
        assembler = Assembler(template_dir=template_dir, storage=storage)
        assert assembler.env is not None
        assert "markdown" in assembler.env.filters

    def test_filter_is_render_markdown_with_math(self, template_dir, tmp_path):
        storage = Storage(tmp_path)
        assembler = Assembler(template_dir=template_dir, storage=storage)
        assert assembler.env.filters["markdown"] is render_markdown_with_math


# ---------------------------------------------------------------------------
# Regression tests — specific patterns from actual generated reports
# ---------------------------------------------------------------------------


class TestRegressionFromActualReports:
    """Regression tests based on actual generated report content."""

    def test_dpo_loss_formula(self):
        """The DPO loss formula from the self-evolution paper must render correctly."""
        text = (
            "Minimizing $\\mathcal{L}_{\\text{DPO}}(\\pi_\\theta;\\pi_{\\text{ref}}) = "
            "-\\mathbb{E}\\big[\\log\\sigma(\\beta\\log\\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} "
            "- \\beta\\log\\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)})\\big]$."
        )
        html = render_markdown_with_math(text)
        assert "\\mathcal{L}_{\\text{DPO}}" in html
        assert "\\pi_\\theta" in html
        assert "\\pi_{\\text{ref}}" in html

    def test_generator_verifier_game_notation(self):
        """GV game notation from actual reports."""
        text = (
            "A base model $\\mathcal{M}$ acts as generator $\\mathcal{G}$ and verifier $\\mathcal{V}$. "
            "Given $\\mathcal{D}$, produce $\\{\\hat{y}_1,\\dots,\\hat{y}_k\\}$ per query $q$."
        )
        html = render_markdown_with_math(text)
        assert "$\\mathcal{M}$" in html
        assert "$\\mathcal{G}$" in html
        assert "$\\mathcal{V}$" in html
        assert "$\\mathcal{D}$" in html
        assert "$q$" in html

    def test_threshold_notation(self):
        """Threshold notation with \\in and brackets."""
        text = "With $\\tau \\in [0.5,0.8]$ and $\\hat{p} \\ge \\tau$."
        html = render_markdown_with_math(text)
        assert "$\\tau \\in [0.5,0.8]$" in html
        assert "$\\hat{p} \\ge \\tau$" in html

    def test_iterative_se_notation(self):
        """Iterative self-evolution mathematical notation."""
        text = "$\\mathcal{P}_t = \\text{GV}(\\mathcal{M}_{t-1},\\mathcal{D},T)$, $\\mathcal{M}_t = \\text{Finetune}(\\mathcal{M}_{t-1},\\mathcal{P}_t)$."
        html = render_markdown_with_math(text)
        assert (
            "$\\mathcal{P}_t = \\text{GV}(\\mathcal{M}_{t-1},\\mathcal{D},T)$" in html
        )
        assert (
            "$\\mathcal{M}_t = \\text{Finetune}(\\mathcal{M}_{t-1},\\mathcal{P}_t)$"
            in html
        )

    def test_performance_numbers_with_math(self):
        """Performance text mixing numbers and math notation."""
        text = (
            "Gemma 3 4B: base accuracy 31.0%. SimpleSE $\\tau=0.6$: 40.7%; "
            "RevisionSE: 42.2%; gap of ~8–13% to oracle."
        )
        html = render_markdown_with_math(text)
        assert "$\\tau=0.6$" in html
        assert "31.0%" in html
        assert "40.7%" in html

    def test_multiline_inline_math(self):
        """Inline math containing a single newline must be preserved correctly."""
        text = (
            "The objective is $\\mathcal{L}_{\\text{pre}} = -\\sum_{i,j} \\log p_{\\theta \\oplus \\alpha\n"
            "\\Delta_i^{\\text{pre}}}(z_{i,j} \\mid q_i, z_{i,<j>})$."
        )
        html = render_markdown_with_math(text)
        assert "&lt;j&gt;" in html
        assert "<!--MATH_PLACEHOLDER" not in html
        assert "\\mathcal{L}_{\\text{pre}}" in html

    def test_math_with_html_special_chars(self):
        """LaTeX expressions containing <, >, and & must have those characters escaped to prevent browser tag parsing."""
        text = "Comparison $a < b$ and $c > d$ and alignment $x & y$."
        html = render_markdown_with_math(text)
        assert "$a &lt; b$" in html
        assert "$c &gt; d$" in html
        assert "$x &amp; y$" in html


class TestMathInCodeBlocks:
    """Math blocks/inline math inside code/pre blocks must be replaced properly and not show as placeholders."""

    def test_math_in_inline_code(self):
        text = "Check `math $x$` in code."
        html = render_markdown_with_math(text)
        assert "<!--MATH_PLACEHOLDER" not in html
        assert "$x$" in html

    def test_math_in_fenced_code_block(self):
        text = "```python\n# formula: $E = mc^2$\nx = m * c**2\n```"
        html = render_markdown_with_math(text)
        assert "<!--MATH_PLACEHOLDER" not in html
        assert "$E = mc^2$" in html
