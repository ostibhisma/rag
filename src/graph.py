# src/graph.py
"""LangGraph pipelines for Adaptive Retrieval RAG.
Three pipelines are defined:
- naive_rag: vector retrieval → generate answer
- hybrid_rag: hybrid retrieval → generate answer
- adaptive_rag: query analysis → decision → selected retriever → evidence check → generate (or reformulate) → verify
"""
from typing import List, Tuple

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain.schema import Document

# Import modules
from .controller.query_analyzer import analyze_query
from .controller.decision import decide_strategy, VECTOR, BM25, HYBRID
from .controller.evidence_checker import check_evidence
from .retrieval.vector import get_vector_retriever
from .retrieval.bm25 import get_bm25_retriever
from .retrieval.hybrid import get_hybrid_retriever
from .llm.generate import generate_answer
from .llm.verify import verify_answer


class RAGState(BaseModel):
    """State passed between graph nodes."""
    question: str = Field(default="")
    docs: List[Document] = Field(default_factory=list)
    answer: str = Field(default="")
    is_sufficient: bool = Field(default=False)
    verification: bool = Field(default=False)
    evidence_metrics: dict = Field(default_factory=dict)
    strategy: str = Field(default="")
    # For adaptive pipeline we may need a flag to indicate whether we need reformulation – omitted for simplicity.


# ---- Node implementations -------------------------------------------------

def retrieve_vector(state: RAGState) -> RAGState:
    retriever = get_vector_retriever()
    state.docs = retriever.retrieve(state.question)
    state.strategy = VECTOR
    return state


def retrieve_bm25(state: RAGState) -> RAGState:
    retriever = get_bm25_retriever()
    state.docs = retriever.retrieve(state.question)
    state.strategy = BM25
    return state


def retrieve_hybrid(state: RAGState) -> RAGState:
    retriever = get_hybrid_retriever()
    state.docs = retriever.retrieve(state.question)
    state.strategy = HYBRID
    return state


def analyze(state: RAGState) -> RAGState:
    state.evidence_metrics = {}
    # analysis does not alter docs yet
    return state


def decide(state: RAGState) -> RAGState:
    features = analyze_query(state.question)
    strategy = decide_strategy(features)
    state.strategy = strategy
    return state


def evidence_check(state: RAGState) -> RAGState:
    is_ok, metrics = check_evidence(state.question, state.docs)
    state.is_sufficient = is_ok
    state.evidence_metrics = metrics
    return state


def generate(state: RAGState) -> RAGState:
    state.answer = generate_answer(state.question, state.docs)
    return state


def verify(state: RAGState) -> RAGState:
    supported, feedback = verify_answer(state.question, state.answer, state.docs)
    state.verification = supported
    return state


def end(state: RAGState) -> RAGState:
    return state

# ---- Graph builders ------------------------------------------------------

def build_naive_rag() -> StateGraph:
    graph = StateGraph(RAGState)
    graph.add_node("vector", retrieve_vector)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("end", end)
    graph.set_entry_point("vector")
    graph.add_edge("vector", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "end")
    graph.add_conditional_edges(
        "verify",
        lambda s: END,
        {END: "end"},
    )
    return graph.compile()


def build_hybrid_rag() -> StateGraph:
    graph = StateGraph(RAGState)
    graph.add_node("hybrid", retrieve_hybrid)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("end", end)
    graph.set_entry_point("hybrid")
    graph.add_edge("hybrid", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "end")
    return graph.compile()


def build_adaptive_rag() -> StateGraph:
    graph = StateGraph(RAGState)
    # Nodes
    graph.add_node("decide", decide)
    graph.add_node("vector", retrieve_vector)
    graph.add_node("bm25", retrieve_bm25)
    graph.add_node("hybrid", retrieve_hybrid)
    graph.add_node("evidence", evidence_check)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("end", end)

    graph.set_entry_point("decide")

    # Conditional routing based on chosen strategy
    def route_strategy(state: RAGState):
        if state.strategy == VECTOR:
            return "vector"
        if state.strategy == BM25:
            return "bm25"
        return "hybrid"
    graph.add_conditional_edges("decide", route_strategy, {VECTOR: "vector", BM25: "bm25", HYBRID: "hybrid"})

    # After retrieval, go to evidence checking
    graph.add_edge("vector", "evidence")
    graph.add_edge("bm25", "evidence")
    graph.add_edge("hybrid", "evidence")

    # If evidence sufficient generate, else we could loop – for simplicity we always generate.
    graph.add_edge("evidence", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "end")

    return graph.compile()

# Helper to pick pipeline
def get_pipeline(name: str):
    if name == "naive":
        return build_naive_rag()
    if name == "hybrid":
        return build_hybrid_rag()
    if name == "adaptive":
        return build_adaptive_rag()
    raise ValueError(f"Unknown pipeline: {name}")
