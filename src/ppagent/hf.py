"""HuggingFace CLI wrapper for paper discovery."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from typing import Any

from ppagent.models import Paper

logger = logging.getLogger(__name__)

_HF_CMD = "hf"


class HfCliError(Exception):
    """Raised when the hf CLI command fails."""


def _run_hf(args: list[str], timeout: int = 60) -> str:
    """Execute an hf CLI command and return stdout."""
    cmd = [_HF_CMD, *args]
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise HfCliError(
            "hf CLI not found. Install it with: pip install huggingface_hub[cli]"
        )
    except subprocess.TimeoutExpired:
        raise HfCliError(f"hf command timed out after {timeout}s: {' '.join(cmd)}")

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise HfCliError(f"hf command failed (rc={result.returncode}): {stderr}")
    return result.stdout.strip()


def _parse_paper(raw: dict[str, Any]) -> Paper:
    """Parse a raw JSON dict from hf papers ls into a Paper model."""
    paper_id = raw.get("id") or raw.get("arxiv_id") or raw.get("paperId", "")
    published_at = None
    for key in ("publishedAt", "published_at", "submittedAt"):
        val = raw.get(key)
        if val:
            try:
                published_at = datetime.fromisoformat(val.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
            break

    authors = raw.get("authors", [])
    if authors and isinstance(authors[0], dict):
        authors = [a.get("name", a.get("user", str(a))) for a in authors]

    return Paper(
        id=paper_id,
        title=raw.get("title", ""),
        authors=authors,
        published_at=published_at,
        upvotes=raw.get("upvotes", 0),
        summary=raw.get("summary", raw.get("abstract", "")),
    )


def list_papers(
    *,
    date: str | None = None,
    limit: int = 50,
    sort: str = "trending",
    submitter: str | None = None,
    filter_: str | None = None,
) -> list[Paper]:
    """List papers via `hf papers ls --format json`."""
    args = ["papers", "ls", "--format", "json"]
    if date and date != "today":
        args.extend(["--date", date])
    if limit:
        args.extend(["--limit", str(limit)])
    if sort:
        args.extend(["--sort", sort])
    if submitter:
        args.extend(["--submitter", submitter])
    if filter_:
        args.extend(["--filter", filter_])

    stdout = _run_hf(args, timeout=90)
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise HfCliError(f"Failed to parse hf papers ls JSON output: {exc}") from exc

    if isinstance(data, dict):
        data = data.get("papers", data.get("results", [data]))
    return [_parse_paper(item) for item in data if isinstance(item, dict)]


def paper_info(paper_id: str) -> Paper:
    """Get detailed info for a specific paper via `hf papers info`."""
    args = ["papers", "info", paper_id, "--format", "json"]
    stdout = _run_hf(args)
    try:
        raw = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise HfCliError(f"Failed to parse hf papers info JSON: {exc}") from exc
    if isinstance(raw, list):
        raw = raw[0] if raw else {}
    return _parse_paper(raw)


def paper_read(paper_id: str) -> str:
    """Read paper content as markdown via `hf papers read`."""
    args = ["papers", "read", paper_id]
    return _run_hf(args, timeout=120)


def search_papers(query: str, *, limit: int = 10) -> list[Paper]:
    """Search papers via `hf papers search --format json`."""
    args = ["papers", "search", query, "--format", "json", "--limit", str(limit)]
    stdout = _run_hf(args, timeout=90)
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise HfCliError(f"Failed to parse hf papers search JSON: {exc}") from exc

    if isinstance(data, dict):
        data = data.get("papers", data.get("results", [data]))
    return [_parse_paper(item) for item in data if isinstance(item, dict)]
