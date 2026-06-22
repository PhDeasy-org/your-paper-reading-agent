"""Base class for publishers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ppagent.models import PaperReport


class PublisherBase(ABC):
    """Abstract base class for all publishers."""

    name: str = "base"

    @abstractmethod
    def publish(
        self,
        report: PaperReport,
        *,
        md_content: str,
        html_content: str,
        report_dir: Path | None = None,
    ) -> bool:
        """Publish the report.

        ``report_dir`` is the on-disk directory holding ``report.html``,
        ``report.md``, ``metadata.json``, and (when figures were selected)
        a ``figures/`` subdirectory. Webhook-style publishers may ignore it;
        file-based publishers (e.g. GitHub Pages) copy from it.

        Returns True on success, False on failure.
        """
        ...

    def validate_config(self) -> bool:
        """Validate that required configuration is present. Override in subclasses."""
        return True
