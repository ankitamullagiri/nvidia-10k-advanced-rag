"""Build the LlamaIndex + ChromaDB pipeline, mirroring the evaluated notebook.

- SimpleDirectoryReader + SentenceSplitter(512, 50) for loading and chunking
- all-MiniLM-L6-v2 embeddings stored in ChromaDB through LlamaIndex's ChromaVectorStore
- LlamaIndex BM25Retriever and VectorIndexRetriever (top 10 each)
- cross-encoder/ms-marco-MiniLM-L-6-v2 for reranking (on GPU when running on ZeroGPU)
- Gemini 2.5 Flash through LlamaIndex, either via OpenRouter (OPENROUTER_API_KEY)
  or directly via Google's API (GOOGLE_API_KEY)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .pipeline import VANILLA_PROMPT, RAGPipeline
from .sources import PDF_PATH, load_sec_filing

log = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")

# Hugging Face ZeroGPU: functions decorated with @spaces.GPU get a GPU attached on demand.
# Outside a ZeroGPU Space (locally, in CI) the `spaces` package is absent, so fall back to CPU.
try:
    import spaces

    gpu = spaces.GPU(duration=20)
    DEVICE = "cuda"
except ImportError:
    def gpu(fn):
        return fn

    DEVICE = "cpu"

_reranker = None


def load_reranker() -> None:
    """Load the cross-encoder at import time, as ZeroGPU expects for models used on GPU."""
    global _reranker
    from sentence_transformers import CrossEncoder

    _reranker = CrossEncoder(RERANK_MODEL, device=DEVICE)
    log.info("Reranker loaded on %s", DEVICE)


# Defined at module level so ZeroGPU always finds a @spaces.GPU function at startup,
# even if building the index fails later (otherwise the Space is shut down).
@gpu
def rerank(query: str, texts: list[str]) -> list[float]:
    """The cross-encoder is the heaviest per-question model call, so it runs on the GPU."""
    return _reranker.predict([(query, t) for t in texts]).tolist()


def load_documents():
    """The annual-report PDF if it's in the Space, otherwise the 10-K from SEC EDGAR."""
    from llama_index.core import Document, SimpleDirectoryReader

    if Path(PDF_PATH).exists():
        docs = SimpleDirectoryReader(input_files=[str(PDF_PATH)]).load_data()
        return docs, f"{Path(PDF_PATH).name} (page numbers)"
    try:
        sections = load_sec_filing()
    except Exception as exc:
        raise RuntimeError(
            f"Couldn't find {Path(PDF_PATH).name} and the SEC download failed ({exc}). "
            f"Upload the annual report PDF as '{Path(PDF_PATH).name}' to the root of the Space."
        ) from exc
    docs = [Document(text=t, metadata={"section": str(i)}) for i, t in enumerate(sections, start=1)]
    return docs, "SEC EDGAR Form 10-K, FY2025 (section numbers)"


def _secret(name: str) -> str:
    """Read a key from the environment, ignoring stray whitespace or quotes from pasting."""
    return os.environ.get(name, "").strip().strip("'\"").strip()


def make_llm():
    """Gemini 2.5 Flash via OpenRouter if that key is set, otherwise via Google directly."""
    if key := _secret("OPENROUTER_API_KEY"):
        from llama_index.llms.openrouter import OpenRouter

        return OpenRouter(api_key=key, model=OPENROUTER_MODEL, max_tokens=1024), f"{OPENROUTER_MODEL} via OpenRouter"
    if key := _secret("GOOGLE_API_KEY"):
        from llama_index.llms.google_genai import GoogleGenAI

        return GoogleGenAI(model=LLM_MODEL, api_key=key), f"{LLM_MODEL} via Google AI"
    raise RuntimeError("Set OPENROUTER_API_KEY or GOOGLE_API_KEY as a Space secret, then restart")


def build_pipeline() -> tuple[RAGPipeline, str]:
    import chromadb
    from llama_index.core import PromptTemplate, Settings, StorageContext, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.vector_stores.chroma import ChromaVectorStore

    # Models (same as the notebook)
    Settings.llm, llm_label = make_llm()
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # Load and chunk
    documents, source_label = load_documents()
    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(documents)
    log.info("Loaded %d documents → %d chunks from %s", len(documents), len(nodes), source_label)

    # Embed into ChromaDB. An in-memory client keeps node ids in sync with the BM25 index
    # built from the same nodes; the Space rebuilds it on each restart (~1 min on CPU).
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection("nvidia_rag")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    # Retrievers and baseline engine
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    baseline_engine = index.as_query_engine(similarity_top_k=3, text_qa_template=PromptTemplate(VANILLA_PROMPT))

    if _reranker is None:
        load_reranker()

    def llm(prompt: str) -> str:
        return Settings.llm.complete(prompt).text

    pipeline = RAGPipeline(bm25_retriever, vector_retriever, baseline_engine, rerank, llm)
    return pipeline, f"{source_label} · {len(nodes)} chunks in ChromaDB · {llm_label}"
