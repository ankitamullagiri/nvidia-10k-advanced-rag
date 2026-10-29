"""Document loading and chunking. Pure Python, no model dependencies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: int
    text: str
    page: int | None = None
    meta: dict = field(default_factory=dict)


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation followed by a capital or digit."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def chunk_text(text: str, chunk_size: int = 380, overlap: int = 40) -> list[str]:
    """Sentence-aware chunking by word count.

    Mirrors LlamaIndex's SentenceSplitter(512 tokens, 50 overlap) used in the notebook:
    ~380 words is roughly 512 tokens. Sentences are kept whole; when a chunk is full,
    the next one starts with the last `overlap` words of the previous chunk.
    A single sentence longer than `chunk_size` is split on word boundaries.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words_per_sentence: list[list[str]] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        for i in range(0, len(words), chunk_size):  # break up giant "sentences" (tables)
            words_per_sentence.append(words[i : i + chunk_size])

    chunks: list[str] = []
    current: list[str] = []
    for words in words_per_sentence:
        if current and len(current) + len(words) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(words)
    if current and (not chunks or " ".join(current) != chunks[-1]):
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[str], chunk_size: int = 380, overlap: int = 40) -> list[Chunk]:
    """Chunk each page separately so every chunk keeps a page number for citations."""
    out: list[Chunk] = []
    for page_no, page_text in enumerate(pages, start=1):
        for piece in chunk_text(page_text, chunk_size, overlap):
            out.append(Chunk(id=len(out), text=piece, page=page_no))
    return out


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25 (keeps numbers like 130.5 and fy2025)."""
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*[a-z]*", text.lower())
