# Advanced RAG over NVIDIA's 2025 Annual Report

[![Live demo](https://img.shields.io/badge/🤗%20Live%20demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/nautankii/nvidia-10k-advanced-rag)
[![tests](https://github.com/ankitamullagiri/nvidia-10k-advanced-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/ankitamullagiri/nvidia-10k-advanced-rag/actions)

A retrieval-augmented generation (RAG) system that answers questions about NVIDIA's 181-page FY2025 Form 10-K. It compares a baseline pipeline against an enhanced one with query rewriting, hybrid retrieval and cross-encoder reranking.

**Context:** MSc Business Analytics, Warwick Business School (Generative AI & AI Applications) · solo project · Python

👉 **[Try it live on Hugging Face](https://huggingface.co/spaces/nautankii/nvidia-10k-advanced-rag)**: ask your own question and compare the baseline and enhanced answers side by side.

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

## Project structure

```
├── advanced_rag.ipynb   # original notebook: build, evaluate, demo log
├── app.py               # Gradio web app (baseline vs enhanced side by side)
├── rag/
│   ├── build.py         # LlamaIndex + ChromaDB: load, chunk, embed, retrievers, Gemini
│   ├── pipeline.py      # query rewrite → hybrid retrieval → RRF → rerank → grounded answer
│   ├── fusion.py        # Reciprocal Rank Fusion
│   └── sources.py       # annual-report PDF, or the 10-K from SEC EDGAR as a fallback
└── tests/               # pytest suite using fakes, so no models or API key needed
```

## Run it

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...    # or GOOGLE_API_KEY=... (free key from aistudio.google.com/apikey)
python app.py                    # opens the Gradio app at http://127.0.0.1:7860
```

Optionally save the FY2025 annual report from [NVIDIA Investor Relations](https://investor.nvidia.com/financial-info/annual-reports-and-proxies/default.aspx) as `NVIDIA-2025-Annual-Report.pdf` in the repo root. Without it, the app loads the 10-K from SEC EDGAR.

**Tests:** `pip install -r requirements-dev.txt && pytest`

## Tech stack

Python · LlamaIndex · ChromaDB · BM25 · Sentence-Transformers · cross-encoder reranking · Gemini 2.5 Flash · LLM-as-judge evaluation · Gradio · pytest · GitHub Actions
