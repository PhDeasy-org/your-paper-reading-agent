"""Notion page publisher."""

from __future__ import annotations

import logging

import httpx

from ppagent.models import PaperReport
from ppagent.publishers import register_publisher
from ppagent.publishers.base import PublisherBase

logger = logging.getLogger(__name__)

_NOTION_API = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


@register_publisher
class NotionPublisher(PublisherBase):
    """Publishes reports as pages in a Notion database."""

    name = "notion"

    def __init__(self, api_key: str = "", database_id: str = "") -> None:
        self.api_key = api_key
        self.database_id = database_id

    def validate_config(self) -> bool:
        if not self.api_key or not self.database_id:
            logger.error("Notion publisher requires api_key and database_id")
            return False
        return True

    def publish(self, report: PaperReport, *, md_content: str, html_content: str) -> bool:
        if not self.validate_config():
            return False

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Notion-Version": _NOTION_VERSION,
        }

        # Convert markdown content to Notion blocks (simplified: paragraph blocks)
        blocks = []
        for line in md_content.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("# "):
                blocks.append({"object": "block", "type": "heading_1",
                               "heading_1": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]}})
            elif line.startswith("## "):
                blocks.append({"object": "block", "type": "heading_2",
                               "heading_2": {"rich_text": [{"type": "text", "text": {"content": line[3:]}}]}})
            elif line.startswith("### "):
                blocks.append({"object": "block", "type": "heading_3",
                               "heading_3": {"rich_text": [{"type": "text", "text": {"content": line[4:]}}]}})
            else:
                # Truncate long lines to Notion's 2000 char limit per text object
                text = line[:2000]
                blocks.append({"object": "block", "type": "paragraph",
                               "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]}})

        payload = {
            "parent": {"database_id": self.database_id},
            "properties": {
                "title": {"title": [{"text": {"content": report.paper.title}}]},
            },
            "children": blocks[:100],  # Notion limits to 100 blocks per request
        }

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(f"{_NOTION_API}/pages", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            page_id = data.get("id", "unknown")
            logger.info("Notion page created: %s", page_id)
            return True
        except Exception as exc:
            logger.error("Notion publish failed: %s", exc)
            return False
