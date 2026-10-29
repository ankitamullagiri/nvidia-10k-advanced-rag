# Advanced RAG over NVIDIA's 2025 Annual Report

A retrieval-augmented generation (RAG) system that answers questions about NVIDIA's 181-page FY2025 Form 10-K. It compares a baseline pipeline against an enhanced one with query rewriting, hybrid retrieval and cross-encoder reranking.

**Context:** MSc Business Analytics, Warwick Business School (Generative AI & AI Applications) · solo project · Python

## Why RAG here

Financial filings need exact figures. An LLM on its own will hallucinate segment revenues or state them with false confidence, and its knowledge stops at a training cutoff. Grounding answers in the filing fixes both.

## Pipeline

| Stage | Baseline | Enhanced |
|---|---|---|
| Query | Raw user input | **LLM query rewriting** into financial terminology (Gemini 2.5 Flash) |
| Retrieval | Vector similarity, top-3 | **Hybrid BM25 + dense vectors**, merged with Reciprocal Rank Fusion (k = 60), top-8 |
| Reranking | None | **Cross-encoder** (`ms-marco-MiniLM-L-6-v2`), keeps the top 3 |
| Prompt | Vanilla | Grounded and citation-required, and says so when the answer isn't in the document |

The 10-K is split into 450 chunks (512 tokens with 50 overlap), embedded with `all-MiniLM-L6-v2` and stored in ChromaDB. The last cell is an interactive search box that switches between the baseline and enhanced modes.

## Evaluation

8 test queries across 4 types: simple factual, deep multi-section context, ambiguous, and edge cases (out-of-corpus or nonsense). Retrieval relevance and answer quality were scored with Gemini as judge.

| Metric | Baseline | Enhanced | Δ |
|---|---|---|---|
| Retrieval precision | 0.42 | 0.50 | +0.08 |
| Correctness (1–5) | 4.12 | 4.50 | +0.38 |
| Grounding (1–5) | 3.75 | **4.75** | **+1.00** |
| Completeness (1–5) | 3.88 | 4.62 | +0.74 |

The biggest win was on the risk-factors query (retrieval precision 0.67 → 1.00), where rewriting named the risk categories explicitly and pulled chunks from three separate sections.

## What didn't work

- **Over-grounding on edge cases:** asked about NVIDIA's "ice cream division", the strict prompt only said the context had no such information, while the baseline gave a more useful answer about NVIDIA's real segments.
- **Modest retrieval precision overall:** filings contain many related passages that don't directly answer the question, and an LLM judge may score relevance more strictly than a human would.
- **Next steps:** human-labelled relevance judgements, metadata filtering by 10-K section, and a fallback that answers out-of-scope questions with what *is* in the filing.

## Run it

```bash
pip install llama-index llama-index-llms-google-genai llama-index-embeddings-huggingface \
            llama-index-vector-stores-chroma llama-index-retrievers-bm25 chromadb \
            sentence-transformers rank_bm25 pypdf nest_asyncio
export GOOGLE_API_KEY=...   # or enter it when the notebook prompts
```

Download the FY2025 annual report (10-K) from [NVIDIA Investor Relations](https://investor.nvidia.com/financial-info/annual-reports-and-proxies/default.aspx), save it as `NVIDIA-2025-Annual-Report.pdf` next to the notebook, then run `advanced_rag.ipynb`.

## Tech stack

Python · LlamaIndex · ChromaDB · BM25 · Sentence-Transformers · cross-encoder reranking · Gemini 2.5 Flash · LLM-as-judge evaluation
