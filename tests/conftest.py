"""Shared fixtures for ppagent tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from ppagent.models import (
    Paper,
    PaperReport,
    ReportSection,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        id="2506.12345",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        published_at=datetime(2025, 6, 10),
        arxiv_url="https://arxiv.org/abs/2506.12345",
        hf_url="https://huggingface.co/papers/2506.12345",
    )


@pytest.fixture
def sample_report(sample_paper: Paper) -> PaperReport:
    return PaperReport(
        paper=sample_paper,
        metadata=ReportSection(
            name="metadata", content="| **Paper** | [arXiv:2506.12345] |"
        ),
        benchmarks=ReportSection(name="benchmarks", content="WMT 2014, PTB"),
        tldr=ReportSection(
            name="tldr",
            content="The Transformer model uses $Q$, $K$, $V$ matrices for self-attention.",
        ),
        previous_works=ReportSection(
            name="previous_works",
            content="Prior work on [RNNs](https://arxiv.org/abs/1409.0473) and CNNs.",
        ),
        method=ReportSection(
            name="method",
            content=(
                "The attention function is $\\text{Attention}(Q,K,V) = \\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V$.\n\n"
                "Multi-head attention:\n\n"
                "$$\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1,\\dots,\\text{head}_h)W^O$$\n\n"
                "where $\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$."
            ),
        ),
        evaluation=ReportSection(
            name="evaluation",
            content="BLEU score of $28.4$ on WMT 2014 EN-DE, surpassing all previous models by $+2.0$ BLEU.",
        ),
        critique=ReportSection(
            name="critique",
            content="Solid work but limited to sequence-to-sequence tasks.",
        ),
        related_works=[
            Paper(
                id="2506.99999",
                title="BERT: Pre-training of Deep Bidirectional Transformers",
            ),
        ],
        generated_at=datetime(2025, 6, 13, 12, 0),
        model_used="gpt-4o",
    )


@pytest.fixture
def writer_data() -> dict:
    return {
        "keywords": ["transformer", "attention", "NLP"],
        "affiliations": ["Google Brain", "Google Research"],
        "benchmarks": "WMT 2014, PTB",
        "tldr": "The Transformer model uses $Q$, $K$, $V$ matrices for self-attention.",
        "previous_works": "Prior work on [RNNs](https://arxiv.org/abs/1409.0473) and CNNs.",
        "method": (
            "The attention function is $\\text{Attention}(Q,K,V) = \\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V$.\n\n"
            "Multi-head attention:\n\n"
            "$$\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1,\\dots,\\text{head}_h)W^O$$\n\n"
            "where $\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$."
        ),
        "evaluation": "BLEU score of $28.4$ on WMT 2014 EN-DE, surpassing all previous models by $+2.0$ BLEU.",
    }


@pytest.fixture
def finder_data() -> dict:
    return {
        "narrative": "Related landscape includes $\\text{ELMo}$ and $\\text{GPT}$.",
        "related_works": [],
    }


@pytest.fixture
def criticizer_data() -> dict:
    return {
        "critique": "Solid work but limited scope.",
    }


@pytest.fixture
def template_dir() -> Path:
    return PROJECT_ROOT / "templates"
