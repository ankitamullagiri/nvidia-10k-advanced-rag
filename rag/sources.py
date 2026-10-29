"""Load the NVIDIA FY2025 filing as a list of page texts.

Prefers the annual-report PDF (the document the project was evaluated on) if it is
present; otherwise downloads the FY2025 Form 10-K from SEC EDGAR.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

PDF_PATH = Path(os.environ.get("NVIDIA_PDF", "NVIDIA-2025-Annual-Report.pdf"))
SEC_URL = "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm"
# SEC asks automated clients to identify themselves in the User-Agent
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "nvidia-10k-advanced-rag demo (github.com/ankitamullagiri)")
WORDS_PER_SECTION = 600  # the HTML filing has no pages, so it is split into sections


def load_pdf(path: Path) -> list[str]:
    from pypdf import PdfReader

    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def load_sec_filing(url: str = SEC_URL) -> list[str]:
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    for hidden in soup.select("ix\\:header, [style*='display:none']"):
        hidden.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    words = text.split()
    return [" ".join(words[i : i + WORDS_PER_SECTION]) for i in range(0, len(words), WORDS_PER_SECTION)]


def load_document() -> tuple[list[str], str]:
    """Return (pages, source_label)."""
    if PDF_PATH.exists():
        return load_pdf(PDF_PATH), f"{PDF_PATH.name} (page numbers)"
    return load_sec_filing(), "SEC EDGAR Form 10-K, FY2025 (section numbers)"
