import numpy as np
import pytest

from rag.bm25 import BM25
from rag.fusion import reciprocal_rank_fusion
from rag.pipeline import GROUNDED_PROMPT, RAGPipeline
from rag.text import Chunk

DOCS = [
    "Data Center revenue grew strongly driven by demand for Hopper GPUs.",
    "Gaming revenue was flat compared with the prior year.",
    "Risk factors include export controls and supply chain constraints.",
    "Automotive revenue increased from self-driving platforms.",
]


# ---------- RRF ----------
def test_rrf_scores_match_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement_between_retrievers():
    # "b" is only 2nd in each list but found by both; "a" and "c" each top one list only
    ranking = [doc for doc, _ in reciprocal_rank_fusion([["a", "b"], ["c", "b"]])]
    assert ranking[0] == "b"


def test_rrf_top_k_and_deterministic_ties():
    assert reciprocal_rank_fusion([["x", "y"], ["y", "x"]], top_k=1)[0][0] == "x"
    assert len(reciprocal_rank_fusion([["a", "b", "c"]], top_k=2)) == 2


def test_rrf_empty():
    assert reciprocal_rank_fusion([[], []]) == []


# ---------- BM25 ----------
def test_bm25_finds_keyword_match():
    assert BM25(DOCS).top_k("export controls risk", 1) == [2]


def test_bm25_ignores_unknown_terms():
    assert BM25(DOCS).top_k("ice cream division", 3) == []


# ---------- Pipeline with fake models ----------
VOCAB = ["data", "center", "gaming", "risk", "export", "automotive", "revenue"]


def fake_embed(texts):
    """Bag-of-words vectors over a tiny vocabulary: enough to test the plumbing."""
    return np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype=float) + 1e-3


def fake_rerank(query, texts):
    q = set(query.lower().split())
    return [len(q & set(t.lower().split())) for t in texts]


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if prompt.startswith("You are a search query optimizer"):
            return "data center segment revenue"
        return "ANSWER"


@pytest.fixture
def pipe():
    chunks = [Chunk(id=i, text=t, page=i + 1) for i, t in enumerate(DOCS)]
    return RAGPipeline(chunks, fake_embed, fake_rerank, FakeLLM())


def test_enhanced_rewrites_retrieves_and_grounds(pipe):
    result = pipe.enhanced("how did data centers do?", top_k=2)
    assert result.rewritten_query == "data center segment revenue"
    assert result.sources[0].page == 1
    assert len(result.sources) == 2
    final_prompt = pipe.llm.prompts[-1]
    assert final_prompt.startswith(GROUNDED_PROMPT.split("{context}")[0])
    assert "how did data centers do?" in final_prompt  # answers the user's question, not the rewrite


def test_baseline_uses_vector_search_only(pipe):
    result = pipe.baseline("gaming", top_k=1)
    assert result.rewritten_query is None
    assert result.sources[0].page == 2
    assert len(pipe.llm.prompts) == 1  # no rewrite call


def test_empty_rewrite_falls_back_to_original(pipe):
    pipe.llm = lambda prompt: ""
    assert pipe.rewrite_query("risk") == "risk"


def test_mismatched_embeddings_rejected():
    with pytest.raises(ValueError):
        RAGPipeline([Chunk(0, "a")], fake_embed, fake_rerank, FakeLLM(), embeddings=np.zeros((2, 7)))
