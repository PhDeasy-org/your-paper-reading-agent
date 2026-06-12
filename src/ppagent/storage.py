"""Local filesystem storage for generated reports."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from ppagent.models import PaperReport

logger = logging.getLogger(__name__)


class Storage:
    """Manages report files on the local filesystem."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _paper_dir(self, paper_id: str, date: str | None = None) -> Path:
        """Get the directory for a paper's report files."""
        date_str = date or datetime.now().strftime("%Y-%m-%d")
        safe_id = paper_id.replace("/", "_")
        return self.output_dir / date_str / safe_id

    def save_report(self, report: PaperReport, *, md_content: str = "", html_content: str = "") -> Path:
        """Save a report's Markdown, HTML, and metadata to disk.

        Returns the path to the paper's output directory.
        """
        paper_dir = self._paper_dir(report.paper.id)
        paper_dir.mkdir(parents=True, exist_ok=True)

        if md_content:
            md_path = paper_dir / "report.md"
            md_path.write_text(md_content, encoding="utf-8")
            logger.info("Wrote %s", md_path)

        if html_content:
            html_path = paper_dir / "report.html"
            html_path.write_text(html_content, encoding="utf-8")
            logger.info("Wrote %s", html_path)

        meta_path = paper_dir / "metadata.json"
        meta_path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote %s", meta_path)

        return paper_dir

    def report_exists(self, paper_id: str, date: str | None = None) -> bool:
        """Check if a report has already been generated."""
        paper_dir = self._paper_dir(paper_id, date)
        return (paper_dir / "metadata.json").exists()

    def list_reports(self, date: str | None = None) -> list[Path]:
        """List previously generated report directories."""
        if date:
            base = self.output_dir / date
            return sorted(base.iterdir()) if base.is_dir() else []

        results: list[Path] = []
        for date_dir in sorted(self.output_dir.iterdir()):
            if date_dir.is_dir():
                results.extend(sorted(date_dir.iterdir()))
        return results
