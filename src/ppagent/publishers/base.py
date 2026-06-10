"""Base class for publishers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ppagent.models import PaperReport


class PublisherBase(ABC):
    """Abstract base class for all publishers."""

    name: str = "base"

    @abstractmethod
    def publish(self, report: PaperReport, *, md_content: str, html_content: str) -> bool:
        """Publish the report. Returns True on success, False on failure."""
        ...

    def validate_config(self) -> bool:
        """Validate that required configuration is present. Override in subclasses."""
        return True
