"""Pydantic data models for ppagent."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


import re


class Paper(BaseModel):
    """Represents a paper from HuggingFace daily papers."""

    id: str = Field(
        description="Unique identifier for the paper, typically an arXiv ID (e.g., '2506.12345')."
    )
    title: str = Field(description="The full title of the paper.")
    authors: list[str] = Field(
        default_factory=list, description="A list of the paper's authors."
    )
    published_at: datetime | None = Field(
        default=None,
        description="The date and time the paper was officially published.",
    )
    upvotes: int = Field(
        default=0,
        description="The number of upvotes the paper has received (e.g., on HuggingFace).",
    )
    summary: str = Field(
        default="",
        description="The abstract or a brief summary of the paper's contents.",
    )
    arxiv_url: str = Field(
        default="", description="The URL to the paper's landing page on arXiv."
    )
    pdf_url: str = Field(
        default="", description="The direct URL to download the paper's PDF."
    )
    hf_url: str = Field(
        default="", description="The URL to the paper's page on HuggingFace."
    )
    github_repo: str = Field(
        default="",
        description="The official GitHub repository URL for the paper's code (if any).",
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.arxiv_url and self.id:
            self.arxiv_url = f"https://arxiv.org/abs/{self.id}"
        if not self.pdf_url and self.id:
            self.pdf_url = f"https://arxiv.org/pdf/{self.id}"
        if not self.hf_url and self.id:
            self.hf_url = f"https://huggingface.co/papers/{self.id}"
        if not self.published_at and self.id:
            cleaned = self.id.split("/")[-1]
            match = re.match(r"^(\d{2})(\d{2})\.\d+", cleaned)
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
                match_old = re.match(r"^(\d{2})(\d{2})\d+", cleaned)
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

    paper: Paper = Field(description="The paper object containing metadata.")
    markdown: str = Field(
        default="",
        description="The full extracted text content of the paper, formatted in Markdown.",
    )
    sections: dict[str, str] = Field(
        default_factory=dict,
        description="A mapping of section headings to their respective text content.",
    )


class ReportSection(BaseModel):
    """A single section of the generated report."""

    name: str = Field(
        description="The identifier for the section (e.g., 'metadata', 'tldr', 'method')."
    )
    content: str = Field(
        description="The actual text content generated for this section."
    )
    confidence: float = Field(
        default=1.0,
        description="The confidence score of the generated content, from 0.0 to 1.0.",
    )


class PaperReport(BaseModel):
    """Complete report for a single paper."""

    paper: Paper = Field(description="The underlying paper being reported on.")
    paper_type: str = Field(
        default="method",
        description="The classified type of the paper (e.g., 'method', 'survey').",
    )
    metadata: ReportSection = Field(
        description="Section containing extracted keywords, affiliations, and metadata."
    )
    benchmarks: ReportSection = Field(
        description="Section detailing benchmarks and datasets used in the paper."
    )
    tldr: ReportSection = Field(
        description="Section containing a short 'Too Long; Didn't Read' summary."
    )
    previous_works: ReportSection = Field(
        description="Section discussing prior works and context."
    )
    method: ReportSection = Field(
        description="Section explaining the proposed method or framework."
    )
    evaluation: ReportSection = Field(
        description="Section summarizing the experimental results and evaluation."
    )
    critique: ReportSection = Field(
        description="Section providing a critical analysis of the paper's strengths and weaknesses."
    )
    related_works: list[RelatedWork] = Field(
        default_factory=list,
        description="A list of other papers related to this one, each with an explanation of its relationship to the main paper.",
    )
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="The timestamp when this report was generated.",
    )
    model_used: str = Field(
        default="",
        description="The primary LLM model used to generate the text of this report.",
    )
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics aggregated across all generation steps.",
    )
    cost_report: dict[str, Any] | None = Field(
        default=None, description="Detailed cost breakdown based on model token usage."
    )


class AgentResult(BaseModel):
    """Standardized output from any agent."""

    agent_name: str = Field(
        description="The name of the agent that produced this result (e.g., 'writer', 'searcher')."
    )
    success: bool = Field(
        description="Whether the agent's operation completed successfully."
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="The structured output data returned by the agent.",
    )
    error: str = Field(
        default="", description="An error message if the operation failed."
    )
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage statistics reported by the LLM for this operation.",
    )


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

    paper_id: str = Field(description="The unique identifier for the paper.")
    title: str = Field(description="The title of the paper.")
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="A score from 0.0 to 1.0 indicating how relevant the paper is to the user's profile.",
    )
    justification: str = Field(
        default="",
        description="A short explanation of why this relevance score was assigned.",
    )


class SearcherOutput(BaseModel):
    """Structured output from the Searcher agent."""

    scored_papers: list[ScoredPaper] = Field(default_factory=list)


class WriterOutput(BaseModel):
    """Structured output from the Writer agent."""

    metadata_keywords: list[str] = Field(
        default_factory=list,
        description="A list of key terms describing the paper's core topics.",
    )
    affiliations: list[str] = Field(
        default_factory=list,
        description="A list of institutions the authors are affiliated with.",
    )
    benchmarks: str = Field(
        default="None reported.",
        description="A summary of the benchmarks or datasets used.",
    )
    tldr: str = Field(
        default="", description="A concise, high-level summary of the paper."
    )
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
    blog_url: str = Field(
        default="",
        description=(
            "The URL of an official blog post / webpage explaining this paper (e.g. on "
            "the authors' company or project blog), if one is linked in the paper text. "
            "Leave empty if none is mentioned. Must be a full http(s) URL."
        ),
    )


class RelatedWork(BaseModel):
    """A related work found by the Finder agent."""

    paper_id: str = Field(description="The identifier of the related work.")
    title: str = Field(description="The title of the related work.")
    relevance: str = Field(
        default="",
        description="An explanation of how this work is related to the main paper.",
    )

    @property
    def arxiv_url(self) -> str:
        """The arXiv landing page URL for this related work."""
        return f"https://arxiv.org/abs/{self.paper_id}" if self.paper_id else ""


class FinderOutput(BaseModel):
    """Structured output from the Finder agent."""

    narrative: str = ""
    related_works: list[RelatedWork] = Field(default_factory=list)


class CritiqueFinding(BaseModel):
    """A single finding from the Criticizer agent."""

    finding: str = Field(
        description="A specific critique, limitation, or observation about the paper."
    )
    severity: str = Field(
        default="medium",
        description="The severity or importance of the finding: 'low', 'medium', or 'high'.",
    )


class CriticizerOutput(BaseModel):
    """Structured output from the Criticizer agent."""

    summary: str = ""
    findings: list[CritiqueFinding] = Field(default_factory=list)
