"""Reciprocal Rank Fusion. Pure Python."""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]],
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[Hashable, float]]:
    """Fuse several ranked lists of document ids.

    Each document scores 1 / (k + rank) in every list it appears in (rank starts at 1),
    and the scores are summed. No score normalisation is needed, which is why RRF works
    well for combining BM25 scores with cosine similarities.

    Ties are broken by first appearance, so results are deterministic.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    order = 0
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = order
                order += 1
    fused = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    return fused[:top_k] if top_k is not None else fused
