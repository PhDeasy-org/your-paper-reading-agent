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


def fetch_arxiv_info(paper_id: str) -> Paper | None:
    """Fetch paper details from arXiv API as a fallback."""
    import urllib.request
    import xml.etree.ElementTree as ET
    from datetime import datetime

    cleaned_id = paper_id.split('/')[-1]
    url = f"http://export.arxiv.org/api/query?id_list={cleaned_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ppagent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        title_el = entry.find("atom:title", ns)
        if title_el is None or title_el.text is None:
            return None

        title = title_el.text.strip().replace("\n", " ")
        title = " ".join(title.split())  # normalize whitespaces
        if title.lower() == "error" or not title:
            return None

        pub_el = entry.find("atom:published", ns)
        published_at = None
        if pub_el is not None and pub_el.text:
            try:
                val = pub_el.text.strip()
                published_at = datetime.fromisoformat(val.replace("Z", "+00:00"))
            except Exception:
                pass

        authors = []
        for author in entry.findall("atom:author", ns):
            name_el = author.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        summary = ""
        summary_el = entry.find("atom:summary", ns)
        if summary_el is not None and summary_el.text:
            summary = summary_el.text.strip().replace("\n", " ")
            summary = " ".join(summary.split())

        return Paper(
            id=cleaned_id,
            title=title,
            authors=authors,
            published_at=published_at,
            summary=summary,
        )
    except Exception as e:
        logger.warning("Failed to fetch info from arXiv API for %s: %s", paper_id, e)
        return None

