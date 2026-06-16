"""Pydantic data models for ppagent."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


import re


class Paper(BaseModel):
    """Represents a paper from HuggingFace daily papers."""

    id: str  # arXiv ID e.g. "2506.12345"
    title: str
    authors: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    upvotes: int = 0
    summary: str = ""
    arxiv_url: str = ""
    pdf_url: str = ""
    hf_url: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.arxiv_url and self.id:
            self.arxiv_url = f"https://arxiv.org/abs/{self.id}"
        if not self.pdf_url and self.id:
            self.pdf_url = f"https://arxiv.org/pdf/{self.id}"
        if not self.hf_url and self.id:
            self.hf_url = f"https://huggingface.co/papers/{self.id}"
        if not self.published_at and self.id:
            cleaned = self.id.split('/')[-1]
            match = re.match(r'^(\d{2})(\d{2})\.\d+', cleaned)
            if match:
                yy, mm = match.groups()
                try:
                    y_val = int(yy)
                    year = 2000 + y_val if y_val < 90 else 1900 + y_val
                    month = int(mm)
                    if 1 <= month <= 12:
                        self.published_at = datetime(year, month, 1)
                except ValueError:
                    pass
            else:
                match_old = re.match(r'^(\d{2})(\d{2})\d+', cleaned)
                if match_old:
                    yy, mm = match_old.groups()
                    try:
                        y_val = int(yy)
                        year = 2000 + y_val if y_val < 90 else 1900 + y_val
                        month = int(mm)
                        if 1 <= month <= 12:
                            self.published_at = datetime(year, month, 1)
                    except ValueError:
                        pass



class PaperContent(BaseModel):
    """Full text content of a paper."""

    paper: Paper
    markdown: str = ""
    sections: dict[str, str] = Field(default_factory=dict)


class ReportSection(BaseModel):
    """A single section of the generated report."""

    name: str  # "metadata", "tldr", "method", etc.
    content: str
    confidence: float = 1.0


class PaperReport(BaseModel):
    """Complete report for a single paper."""

    paper: Paper
    paper_type: str = "method"
    metadata: ReportSection
    benchmarks: ReportSection
    tldr: ReportSection
    previous_works: ReportSection
    method: ReportSection
    evaluation: ReportSection
    critique: ReportSection
    related_works: list[Paper] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)
    model_used: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    cost_report: dict[str, Any] | None = None


class AgentResult(BaseModel):
    """Standardized output from any agent."""

    agent_name: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    usage: dict[str, int] = Field(default_factory=dict)


# --- Structured LLM output models ---


class ClassifierOutput(BaseModel):
    """Structured output from the Classifier agent."""

    paper_type: str = Field(
        default="method",
        description=(
            "The paper type. Must be one of: method, benchmark, survey, "
            "analysis, empirical, framework, position, application."
        ),
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score for the classification (0.0 to 1.0).",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this paper type was chosen.",
    )


class ScoredPaper(BaseModel):
    """A paper with a relevance score assigned by the Searcher agent."""

    paper_id: str
    title: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    justification: str = ""


class SearcherOutput(BaseModel):
    """Structured output from the Searcher agent."""

    scored_papers: list[ScoredPaper] = Field(default_factory=list)


class WriterOutput(BaseModel):
    """Structured output from the Writer agent."""

    metadata_keywords: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    benchmarks: str = "None reported."
    tldr: str = ""
    previous_works_summary: str = Field(
        default="",
        description=(
            "Summary of the related work section, highlighting prior methods and their limitations. "
            "For EACH mentioned prior work, method, framework, baseline, or dataset, you MUST include a "
            "Markdown hyperlink (e.g. `[Work Name](URL)`) to its official paper, arXiv page, or a search link "
            "on Google Scholar or arXiv search."
        ),
    )
    method_details: str = ""
    performance_evaluation: str = ""


class RelatedWork(BaseModel):
    """A related work found by the Finder agent."""

    paper_id: str
    title: str
    relevance: str = ""


class FinderOutput(BaseModel):
    """Structured output from the Finder agent."""

    narrative: str = ""
    related_works: list[RelatedWork] = Field(default_factory=list)


class CritiqueFinding(BaseModel):
    """A single finding from the Criticizer agent."""

    finding: str
    severity: str = "medium"  # low / medium / high


class CriticizerOutput(BaseModel):
    """Structured output from the Criticizer agent."""

    summary: str = ""
    findings: list[CritiqueFinding] = Field(default_factory=list)
