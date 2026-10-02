# src/retrieval/bm25.py
"""BM25 retriever implementation using rank_bm25.
It builds an inverted index over the raw document texts and scores
queries against it, returning the top‑k most relevant documents.
"""
import os
import pickle
from typing import List

from rank_bm25 import BM25Okapi
from langchain.schema import Document

from ..config import BM25_INDEX_PATH, TOP_K

# Simple persistence utilities

def _save_bm25(bm25: BM25Okapi, docs: List[Document]):
    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump((bm25, docs), f)


def _load_bm25():
    if os.path.exists(BM25_INDEX_PATH):
        with open(BM25_INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return None, []


def build_bm25(docs: List[Document]):
    """Create a BM25 index from *docs* and persist it.
    Each document is tokenized by whitespace.
    """
    tokenized_corpus = [doc.page_content.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    _save_bm25(bm25, docs)
    return bm25, docs


class BM25Retriever:
    """Retriever based on BM25 scores.
    """

    def __init__(self, bm25: BM25Okapi, docs: List[Document]):
        self.bm25 = bm25
        self.docs = docs

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Document]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        # Get indices of top scores (higher is better)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.docs[i] for i in top_indices]


def get_bm25_retriever() -> BM25Retriever:
    bm25, docs = _load_bm25()
    if bm25 is None:
        raise RuntimeError("BM25 index not built yet. Run build_bm25() first.")
    return BM25Retriever(bm25, docs)
