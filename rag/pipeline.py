"""Baseline and enhanced RAG pipelines over NVIDIA's FY2025 annual report.

Models are injected, so the pipeline can be unit-tested with fakes:
    embed(texts) -> np.ndarray of shape (n, d)
    rerank(query, texts) -> list[float]          (higher = more relevant)
    llm(prompt) -> str
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .bm25 import BM25
from .fusion import reciprocal_rank_fusion
from .text import Chunk

Embedder = Callable[[list[str]], np.ndarray]
Reranker = Callable[[str, list[str]], list[float]]
LLM = Callable[[str], str]

# Prompts are kept verbatim from the evaluated notebook.
VANILLA_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Answer the question: {question}\n"
)

REWRITE_PROMPT = (
    "You are a search query optimizer for a financial document database.\n"
    "Rewrite the following question to improve document retrieval.\n"
    "Add relevant financial terminology and be more specific.\n"
    "Return ONLY the rewritten query, nothing else.\n\n"
    "Original: {query}\n"
    "Rewritten:"
)

GROUNDED_PROMPT = """
You are a financial analyst assistant. Answer the question using ONLY
the context provided below. Do not use any outside knowledge.
If the context does not contain enough information, say so explicitly.
Always cite specific figures and facts from the context.

Retrieved Context:
{context}

Question: {question}

Answer (grounded in context only):
"""


@dataclass
class RAGResult:
    mode: str
    question: str
    answer: str
    sources: list[Chunk]
    rewritten_query: str | None = None
    trace: list[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(
        self,
        chunks: list[Chunk],
        embed: Embedder,
        rerank: Reranker,
        llm: LLM,
        embeddings: np.ndarray | None = None,
        rrf_k: int = 60,
    ):
        if not chunks:
            raise ValueError("No chunks to index")
        self.chunks = chunks
        self.embed = embed
        self.rerank_fn = rerank
        self.llm = llm
        self.rrf_k = rrf_k
        vectors = embeddings if embeddings is not None else embed([c.text for c in chunks])
        self.vectors = _normalise(np.asarray(vectors, dtype=np.float32))
        if self.vectors.shape[0] != len(chunks):
            raise ValueError("embeddings and chunks have different lengths")
        self.bm25 = BM25([c.text for c in chunks])

    # ---------- retrieval ----------
    def vector_search(self, query: str, k: int) -> list[int]:
        q = _normalise(np.asarray(self.embed([query]), dtype=np.float32))[0]
        sims = self.vectors @ q
        return [int(i) for i in np.argsort(-sims, kind="stable")[:k]]

    def hybrid_search(self, query: str, per_retriever: int = 10, top_k: int = 8) -> list[int]:
        fused = reciprocal_rank_fusion(
            [self.bm25.top_k(query, per_retriever), self.vector_search(query, per_retriever)],
            k=self.rrf_k,
            top_k=top_k,
        )
        return [int(doc_id) for doc_id, _ in fused]

    def rerank(self, query: str, ids: list[int], top_k: int = 3) -> list[int]:
        if not ids:
            return []
        scores = self.rerank_fn(query, [self.chunks[i].text for i in ids])
        ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
        return [i for i, _ in ranked[:top_k]]

    def rewrite_query(self, query: str) -> str:
        rewritten = self.llm(REWRITE_PROMPT.format(query=query)).strip()
        return rewritten or query

    # ---------- pipelines ----------
    def baseline(self, question: str, top_k: int = 3) -> RAGResult:
        ids = self.vector_search(question, top_k)
        sources = [self.chunks[i] for i in ids]
        answer = self.llm(VANILLA_PROMPT.format(context=_join(sources), question=question)).strip()
        return RAGResult("baseline", question, answer, sources, trace=[f"Vector search → top {top_k}"])

    def enhanced(self, question: str, candidates: int = 8, top_k: int = 3) -> RAGResult:
        rewritten = self.rewrite_query(question)
        cand_ids = self.hybrid_search(rewritten, top_k=candidates)
        ids = self.rerank(rewritten, cand_ids, top_k=top_k)
        sources = [self.chunks[i] for i in ids]
        answer = self.llm(GROUNDED_PROMPT.format(context=_join(sources), question=question)).strip()
        trace = [
            "LLM query rewrite",
            f"BM25 + vector (top 10 each) → RRF → {len(cand_ids)} candidates",
            f"Cross-encoder rerank → top {len(ids)}",
            "Grounded, citation-required answer",
        ]
        return RAGResult("enhanced", question, answer, sources, rewritten, trace)


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


def _join(chunks: list[Chunk]) -> str:
    return "\n\n---\n\n".join(c.text for c in chunks)
