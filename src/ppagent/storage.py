"""Local filesystem storage for generated reports."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from ppagent.models import PaperReport

logger = logging.getLogger(__name__)


class Storage:
    """Manages report files on the local filesystem."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_filename(title: str, published_at: datetime | None = None) -> str:
        """Convert a paper title into a filesystem-safe directory name with YY-MM prefix."""
        name = re.sub(r'[^\w\s-]', '', title)
        name = re.sub(r'[\s_]+', '-', name.strip())
        name = re.sub(r'-{2,}', '-', name)
        name = name[:200] or 'untitled'
        if published_at:
            prefix = published_at.strftime("%y-%m")
            return f"{prefix}-{name}"
        return name

    def _paper_dir(self, title: str, published_at: datetime | None = None) -> Path:
        """Get the directory for a paper's report files."""
        return self.output_dir / self._safe_filename(title, published_at)

    def paper_dir(self, title: str, published_at: datetime | None = None) -> Path:
        """Public accessor for a paper's report directory (creates it if needed)."""
        d = self._paper_dir(title, published_at)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_report(self, report: PaperReport, *, md_content: str = "", html_content: str = "") -> Path:
        """Save a report's Markdown, HTML, and metadata to disk.

        Returns the path to the paper's output directory.
        """
        paper_dir = self._paper_dir(report.paper.title, report.paper.published_at)
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

    def report_exists(self, title: str, published_at: datetime | None = None) -> bool:
        """Check if a report has already been generated."""
        paper_dir = self._paper_dir(title, published_at)
        return (paper_dir / "metadata.json").exists()

    def list_reports(self) -> list[Path]:
        """List previously generated report directories."""
        return sorted(
            p for p in self.output_dir.iterdir() if p.is_dir()
        )
