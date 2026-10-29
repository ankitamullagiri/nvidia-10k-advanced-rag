"""Real model backends: MiniLM embeddings, MS MARCO cross-encoder, Gemini 2.5 Flash."""

from __future__ import annotations

import os

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


class MiniLMEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        return self.model.predict([(query, t) for t in texts]).tolist()


class GeminiLLM:
    def __init__(self, api_key: str | None = None, model: str = LLM_MODEL):
        from google import genai

        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        self.client = genai.Client(api_key=key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return (response.text or "").strip()
