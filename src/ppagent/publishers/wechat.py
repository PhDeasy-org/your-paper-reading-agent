"""WeChat Official Account publisher."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from ppagent.models import PaperReport
from ppagent.publishers import register_publisher
from ppagent.publishers.base import PublisherBase

logger = logging.getLogger(__name__)

_WECHAT_TOKEN_URL = "https://api.weixin.qq.com/cgi-bin/token"
_WECHAT_DRAFT_URL = "https://api.weixin.qq.com/cgi-bin/draft/add"


@register_publisher
class WeChatPublisher(PublisherBase):
    """Publishes reports as drafts to a WeChat Official Account."""

    name = "wechat"

    def __init__(self, appid: str = "", secret: str = "") -> None:
        self.appid = appid
        self.secret = secret

    def validate_config(self) -> bool:
        if not self.appid or not self.secret:
            logger.error("WeChat publisher requires appid and secret")
            return False
        return True

    def _get_access_token(self) -> str:
        """Obtain a WeChat API access token."""
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                _WECHAT_TOKEN_URL,
                params={
                    "grant_type": "client_credential",
                    "appid": self.appid,
                    "secret": self.secret,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        if "access_token" not in data:
            raise RuntimeError(f"WeChat token error: {data}")
        return data["access_token"]

    def publish(
        self,
        report: PaperReport,
        *,
        md_content: str,
        html_content: str,
        report_dir: Path | None = None,
    ) -> bool:
        if not self.validate_config():
            return False
        try:
            token = self._get_access_token()
            article = {
                "title": report.paper.title,
                "content": html_content,
                "author": "ppagent",
                "digest": report.tldr.content[:120],
            }
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    _WECHAT_DRAFT_URL,
                    params={"access_token": token},
                    json={"articles": [article]},
                )
                resp.raise_for_status()
                data = resp.json()
            if "media_id" in data:
                logger.info("WeChat draft created: %s", data["media_id"])
                return True
            else:
                logger.error("WeChat draft failed: %s", data)
                return False
        except Exception as exc:
            logger.error("WeChat publish failed: %s", exc)
            return False
