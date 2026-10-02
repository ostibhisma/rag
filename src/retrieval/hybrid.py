# src/retrieval/hybrid.py
"""Hybrid retriever combining vector (FAISS) and BM25 scores.
A simple linear interpolation is used with weight ``config.HYBRID_WEIGHT``.
"""
from typing import List, Tuple
import numpy as np

from langchain.schema import Document

from .vector import get_vector_retriever
from .bm25 import get_bm25_retriever
from ..config import HYBRID_WEIGHT, TOP_K


class HybridRetriever:
    """Retriever that fuses vector and BM25 scores.

    Parameters
    ----------
    vector_retriever : VectorRetriever
        Retrieves documents with FAISS.
    bm25_retriever : BM25Retriever
        Retrieves documents with BM25.
    weight : float
        Weight for vector scores (BM25 gets ``1 - weight``).
    """

    def __init__(self, vector_retriever, bm25_retriever, weight: float = HYBRID_WEIGHT):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weight = weight

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Document]:
        # Get raw results from each retriever
        vec_docs = self.vector_retriever.retrieve(query, top_k=top_k * 2)  # fetch extra for fusion
        bm25_docs = self.bm25_retriever.retrieve(query, top_k=top_k * 2)

        # Build dicts for quick lookup of scores
        # For vector, we need scores; we can approximate by using distances from FAISS index
        # Re‑embed query and compute distances manually
        from ..embeddings import embed_query, load_faiss_index
        q_vec = embed_query(query).astype("float32")
        index = self.vector_retriever.index
        distances, indices = index.search(np.expand_dims(q_vec, 0), top_k * 2)
        vec_scores = {int(idx): -float(dist) for idx, dist in zip(indices[0], distances[0])}
        # BM25 scores are accessible via the BM25 object
        bm25 = self.bm25_retriever.bm25
        tokenized_query = query.split()
        bm25_raw_scores = bm25.get_scores(tokenized_query)
        bm25_scores = {i: float(score) for i, score in enumerate(bm25_raw_scores)}

        # Combine scores for all candidate docs
        candidate_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
        combined: List[Tuple[int, float]] = []
        for doc_id in candidate_ids:
            v_score = vec_scores.get(doc_id, 0.0)
            b_score = bm25_scores.get(doc_id, 0.0)
            combined_score = self.weight * v_score + (1 - self.weight) * b_score
            combined.append((doc_id, combined_score))

        # Sort by combined score descending and pick top_k
        combined.sort(key=lambda x: x[1], reverse=True)
        top_ids = [doc_id for doc_id, _ in combined[:top_k]]
        # Retrieve Document objects from the vector retriever's doc list (same ordering)
        docs = self.vector_retriever.docs
        return [docs[i] for i in top_ids if i < len(docs)]


def get_hybrid_retriever() -> HybridRetriever:
    vector_ret = get_vector_retriever()
    bm25_ret = get_bm25_retriever()
    return HybridRetriever(vector_ret, bm25_ret)
