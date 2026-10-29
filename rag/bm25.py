"""Okapi BM25 keyword retrieval. Pure Python + NumPy."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .text import tokenize


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75), matching rank_bm25's BM25Okapi defaults."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in documents]
        self.doc_len = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(self.docs) else 0.0
        self.tf = [Counter(d) for d in self.docs]

        df: Counter = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        idf = {t: math.log(n - f + 0.5) - math.log(f + 0.5) for t, f in df.items()}
        # Terms in more than half the docs get negative idf; floor them like rank_bm25 does
        positive = [v for v in idf.values() if v > 0]
        floor = epsilon * (sum(positive) / len(positive)) if positive else 0.0
        self.idf = {t: (v if v > 0 else floor) for t, v in idf.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        if not self.docs or self.avgdl == 0:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            freq = np.array([tf.get(term, 0) for tf in self.tf], dtype=float)
            out += idf * freq * (self.k1 + 1) / (freq + norm)
        return out

    def top_k(self, query: str, k: int) -> list[int]:
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")
        return [int(i) for i in order[:k] if s[i] > 0]
