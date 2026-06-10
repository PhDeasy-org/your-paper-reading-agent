"""PDF download and text extraction for papers."""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF
import httpx

from ppagent.models import Paper

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT = 120  # seconds


def download_pdf(paper: Paper, cache_dir: Path) -> Path:
    """Download a paper's PDF from arXiv and cache it locally.

    Returns the path to the cached PDF file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = cache_dir / f"{paper.id.replace('/', '_')}.pdf"

    if pdf_path.exists():
        logger.debug("Using cached PDF: %s", pdf_path)
        return pdf_path

    url = paper.pdf_url or f"https://arxiv.org/pdf/{paper.id}"
    logger.info("Downloading PDF: %s → %s", url, pdf_path)

    with httpx.Client(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        pdf_path.write_bytes(resp.content)

    return pdf_path


def extract_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF file page by page."""
    doc = fitz.open(pdf_path)
    pages: list[str] = []
    try:
        for page in doc:
            pages.append(page.get_text())
    finally:
        doc.close()
    return "\n\n".join(pages)
