"""Generic blog/webhook publisher."""

from __future__ import annotations

import logging

import httpx

from ppagent.models import PaperReport
from ppagent.publishers import register_publisher
from ppagent.publishers.base import PublisherBase

logger = logging.getLogger(__name__)


@register_publisher
class BlogPublisher(PublisherBase):
    """Publishes reports to a personal blog via webhook or REST API."""

    name = "blog"

    def __init__(self, webhook_url: str = "", api_key: str = "") -> None:
        self.webhook_url = webhook_url
        self.api_key = api_key

    def validate_config(self) -> bool:
        if not self.webhook_url:
            logger.error("Blog publisher requires webhook_url")
            return False
        return True

    def publish(self, report: PaperReport, *, md_content: str, html_content: str) -> bool:
        if not self.validate_config():
            return False

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "title": report.paper.title,
            "paper_id": report.paper.id,
            "arxiv_url": report.paper.arxiv_url,
            "published_at": report.paper.published_at.isoformat() if report.paper.published_at else None,
            "content_markdown": md_content,
            "content_html": html_content,
            "tldr": report.tldr.content,
            "keywords": [s for s in report.metadata.content.split(", ") if s],
        }

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(self.webhook_url, headers=headers, json=payload)
                resp.raise_for_status()
            logger.info("Blog publish successful for %s", report.paper.title)
            return True
        except Exception as exc:
            logger.error("Blog publish failed: %s", exc)
            return False
