"""Orchestrates the full paper discovery → report generation pipeline."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ppagent import hf, pdf
from ppagent.agents.assembler import Assembler
from ppagent.agents.criticizer import CriticizerAgent
from ppagent.agents.finder import FinderAgent
from ppagent.agents.searcher import SearcherAgent
from ppagent.agents.writer import WriterAgent
from ppagent.config import AppConfig
from ppagent.hf import HfCliError
from ppagent.llm import LLMClient
from ppagent.models import AgentResult, Paper, PaperContent, PaperReport
from ppagent.storage import Storage

logger = logging.getLogger(__name__)


class PaperPipeline:
    """Full pipeline: search → (writer ‖ finder) → criticizer → assembler."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm = LLMClient(config.llm)
        self.searcher = SearcherAgent(self.llm, config)
        self.writer = WriterAgent(self.llm, config)
        self.finder = FinderAgent(self.llm, config)
        self.criticizer = CriticizerAgent(self.llm, config)
        self.storage = Storage(config.output_dir)
        self.assembler = Assembler(
            template_dir=config.template_dir,
            storage=self.storage,
            model_used=config.llm.model,
        )

    def search(
        self,
        *,
        date: str | None = None,
        limit: int | None = None,
    ) -> list[Paper]:
        """Fetch daily papers and filter by user profile."""
        papers = hf.list_papers(
            date=date or self.config.search.default_date,
            limit=limit or self.config.search.default_limit,
            sort=self.config.search.sort,
        )
        logger.info("Fetched %d papers from HuggingFace", len(papers))

        profile_path = self.config.profile_path
        if not profile_path.exists():
            logger.warning("Profile not found at %s — returning all papers", profile_path)
            return papers[: self.config.search.max_reports_per_run]

        profile = profile_path.read_text()
        result = self.searcher.run(papers=papers, profile=profile)

        if not result.success:
            logger.error("Searcher failed: %s", result.error)
            return []

        matched = result.data["papers"]
        logger.info("Searcher matched %d / %d papers", len(matched), len(papers))
        return matched[: self.config.search.max_reports_per_run]

    def report(self, paper_id: str) -> PaperReport:
        """Generate a full report for a single paper."""
        # Fetch paper info
        try:
            paper = hf.paper_info(paper_id)
        except HfCliError:
            # Fallback: construct minimal Paper from ID
            paper = Paper(id=paper_id, title=paper_id)

        # Get paper content: try hf papers read first, fall back to PDF
        content_md = ""
        try:
            content_md = hf.paper_read(paper_id)
            logger.info("Got paper content via hf papers read (%d chars)", len(content_md))
        except HfCliError as exc:
            logger.warning("hf papers read failed: %s — trying PDF fallback", exc)
            if self.config.report.download_pdf:
                try:
                    pdf_path = pdf.download_pdf(paper, self.config.pdf_cache_dir)
                    content_md = pdf.extract_text(pdf_path)
                    logger.info("Got paper content via PDF (%d chars)", len(content_md))
                except Exception as pdf_exc:
                    logger.error("PDF fallback also failed: %s", pdf_exc)

        if not content_md:
            content_md = paper.summary or "Paper content unavailable."
            logger.warning("Using abstract as content for %s", paper_id)

        paper_content = PaperContent(paper=paper, markdown=content_md)

        # Run writer and finder in parallel
        writer_result: AgentResult | None = None
        finder_result: AgentResult | None = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            writer_future = executor.submit(self.writer.run, content=paper_content)
            finder_future = executor.submit(self.finder.run, content=paper_content)

            for future in as_completed([writer_future, finder_future]):
                result = future.result()
                if result.agent_name == "writer":
                    writer_result = result
                elif result.agent_name == "finder":
                    finder_result = result

        # Ensure results exist
        if writer_result is None:
            writer_result = AgentResult(agent_name="writer", success=False, error="Writer did not complete")
        if finder_result is None:
            finder_result = AgentResult(agent_name="finder", success=False, error="Finder did not complete")

        # Criticizer depends on writer output
        criticizer_result = self.criticizer.run(
            content=paper_content,
            writer_sections=writer_result.data if writer_result.success else None,
        )

        # Assemble
        report, _, _ = self.assembler.assemble(
            paper=paper,
            writer_result=writer_result,
            finder_result=finder_result,
            criticizer_result=criticizer_result,
        )

        logger.info("Report generated for %s: %s", paper_id, paper.title)
        return report

    def run(
        self,
        *,
        date: str | None = None,
        limit: int | None = None,
        prompt_replace: bool = True,
    ) -> list[PaperReport]:
        """Full pipeline: search + report generation."""
        papers = self.search(date=date, limit=limit)
        if not papers:
            logger.info("No papers to process.")
            return []

        reports: list[PaperReport] = []
        for paper in papers:
            if self.storage.report_exists(paper.title, paper.published_at):
                if prompt_replace:
                    try:
                        answer = input(f"Report for \"{paper.title}\" already exists. Regenerate? [y/N] ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        answer = ""
                    if answer != "y":
                        logger.info("Skipping %s", paper.title)
                        continue
                else:
                    logger.info("Skipping %s — report already exists", paper.title)
                    continue

            try:
                report = self.report(paper.id)
                reports.append(report)
            except Exception as exc:
                logger.error("Failed to generate report for %s: %s", paper.id, exc)

        return reports
