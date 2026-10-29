"""Tests for RRF and the pipeline orchestration, using fakes shaped like LlamaIndex objects."""

from dataclasses import dataclass, field

import pytest

from rag.fusion import reciprocal_rank_fusion
from rag.pipeline import GROUNDED_PROMPT, REWRITE_PROMPT, RAGPipeline


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


# ---------- fakes with the same interface as LlamaIndex nodes / retrievers ----------
@dataclass
class FakeNode:
    node_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def get_content(self):
        return self.text


@dataclass
class FakeNodeWithScore:
    node: FakeNode
    score: float = 0.0


class FakeRetriever:
    def __init__(self, nodes):
        self.nodes = nodes
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return [FakeNodeWithScore(n) for n in self.nodes]


@dataclass
class FakeResponse:
    response: str
    source_nodes: list


class FakeEngine:
    def __init__(self, nodes):
        self.nodes = nodes

    def query(self, q):
        return FakeResponse("BASELINE ANSWER", [FakeNodeWithScore(n) for n in self.nodes[:3]])


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return "data center segment revenue" if prompt.startswith(REWRITE_PROMPT[:30]) else "ANSWER"


N = {i: FakeNode(f"n{i}", f"chunk {i} data center" if i in (2, 5) else f"chunk {i}", {"page_label": str(i)}) for i in range(1, 8)}


@pytest.fixture
def pipe():
    bm25 = FakeRetriever([N[1], N[2], N[3], N[4]])
    vector = FakeRetriever([N[5], N[2], N[6], N[7]])

    def rerank(query, texts):  # prefer chunks mentioning "data center"
        return [1.0 if "data center" in t else 0.0 for t in texts]

    return RAGPipeline(bm25, vector, FakeEngine(list(N.values())), rerank, FakeLLM())


def test_hybrid_retrieve_fuses_and_dedupes(pipe):
    ids = [n.node.node_id for n in pipe.hybrid_retrieve("q", top_k=8)]
    assert ids[0] == "n2"  # found by both retrievers
    assert len(ids) == len(set(ids)) == 7


def test_enhanced_rewrites_retrieves_reranks_and_grounds(pipe):
    result = pipe.enhanced("how did data centers do?")
    assert result.rewritten_query == "data center segment revenue"
    assert pipe.bm25_retriever.queries == ["data center segment revenue"]  # retrieval uses the rewrite
    assert [s.page for s in result.sources[:2]] == ["2", "5"]  # reranker pulled both matches up
    assert len(result.sources) == 3
    final_prompt = pipe.llm.prompts[-1]
    assert final_prompt.startswith(GROUNDED_PROMPT.split("{context}")[0])
    assert "how did data centers do?" in final_prompt  # answers the user's question, not the rewrite


def test_baseline_uses_query_engine(pipe):
    result = pipe.baseline("gaming")
    assert result.answer == "BASELINE ANSWER"
    assert result.rewritten_query is None
    assert [s.page for s in result.sources] == ["1", "2", "3"]
    assert pipe.llm.prompts == []  # no rewrite call


def test_empty_rewrite_falls_back_to_original(pipe):
    pipe.llm = lambda prompt: ""
    assert pipe.rewrite_query("risk") == "risk"


def test_rerank_handles_no_candidates(pipe):
    assert pipe.rerank("q", []) == []
