"""Baseline and enhanced RAG pipelines (same logic as the evaluated notebook).

The pipeline works on LlamaIndex objects but only relies on their interfaces:
    retriever.retrieve(query) -> list[NodeWithScore]
    baseline_engine.query(query) -> Response (with .response and .source_nodes)
    rerank(query, texts) -> list[float]
    llm(prompt) -> str
so the orchestration can be unit-tested with lightweight fakes (see tests/).
The real LlamaIndex + ChromaDB objects are built in rag/build.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .fusion import reciprocal_rank_fusion

# Prompts are kept verbatim from the evaluated notebook.
VANILLA_PROMPT = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Answer the question: {query_str}\n"
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
class Source:
    text: str
    page: str | None


@dataclass
class RAGResult:
    mode: str
    question: str
    answer: str
    sources: list[Source]
    rewritten_query: str | None = None
    trace: list[str] = field(default_factory=list)


def _source(node_with_score: Any) -> Source:
    node = node_with_score.node
    meta = getattr(node, "metadata", {}) or {}
    return Source(node.get_content(), meta.get("page_label") or meta.get("section"))


class RAGPipeline:
    def __init__(
        self,
        bm25_retriever: Any,
        vector_retriever: Any,
        baseline_engine: Any,
        rerank: Callable[[str, list[str]], list[float]],
        llm: Callable[[str], str],
        rrf_k: int = 60,
    ):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.baseline_engine = baseline_engine
        self.rerank_fn = rerank
        self.llm = llm
        self.rrf_k = rrf_k

    # ---------- enhanced pipeline steps ----------
    def rewrite_query(self, query: str) -> str:
        """Short or vague queries embed poorly; rewriting adds financial terminology."""
        rewritten = self.llm(REWRITE_PROMPT.format(query=query)).strip()
        return rewritten or query

    def hybrid_retrieve(self, query: str, top_k: int = 8) -> list[Any]:
        """BM25 + vector search (top 10 each), fused with Reciprocal Rank Fusion."""
        bm25_results = self.bm25_retriever.retrieve(query)
        vector_results = self.vector_retriever.retrieve(query)
        node_map = {n.node.node_id: n for n in [*bm25_results, *vector_results]}
        fused = reciprocal_rank_fusion(
            [[n.node.node_id for n in bm25_results], [n.node.node_id for n in vector_results]],
            k=self.rrf_k,
            top_k=top_k,
        )
        return [node_map[node_id] for node_id, _ in fused]

    def rerank(self, query: str, nodes: list[Any], top_k: int = 3) -> list[Any]:
        """Cross-encoder reads query and chunk together: slower but more accurate."""
        if not nodes:
            return []
        scores = self.rerank_fn(query, [n.node.get_content() for n in nodes])
        ranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
        return [node for node, _ in ranked[:top_k]]

    # ---------- pipelines ----------
    def baseline(self, question: str) -> RAGResult:
        response = self.baseline_engine.query(question)
        sources = [_source(n) for n in response.source_nodes]
        return RAGResult(
            "baseline", question, str(response.response).strip(), sources,
            trace=["ChromaDB vector search → top 3", "Vanilla prompt"],
        )

    def enhanced(self, question: str) -> RAGResult:
        rewritten = self.rewrite_query(question)
        candidates = self.hybrid_retrieve(rewritten, top_k=8)
        top_nodes = self.rerank(rewritten, candidates, top_k=3)
        context = "\n\n---\n\n".join(n.node.get_content() for n in top_nodes)
        answer = self.llm(GROUNDED_PROMPT.format(context=context, question=question)).strip()
        return RAGResult(
            "enhanced", question, answer, [_source(n) for n in top_nodes], rewritten,
            trace=[
                "LLM query rewrite",
                f"BM25 + ChromaDB vector search (top 10 each) → RRF → {len(candidates)} candidates",
                f"Cross-encoder rerank → top {len(top_nodes)}",
                "Grounded, citation-required answer",
            ],
        )
