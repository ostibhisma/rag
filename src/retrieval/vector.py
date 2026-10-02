# src/retrieval/vector.py
"""FAISS vector retriever implementation.
Uses the FAISS index built from document embeddings and returns the most similar
documents for a query.
"""
from typing import List

import numpy as np
from langchain.schema import Document

from ..embeddings import embed_query, load_faiss_index
from ..config import TOP_K, INDEX_PATH

# For simplicity we keep the document list in a pickle alongside the index.
import os
import pickle

DOCS_PATH = os.path.join(os.path.dirname(INDEX_PATH), "docs.pkl")


def _load_documents() -> List[Document]:
    if os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, "rb") as f:
            return pickle.load(f)
    return []


class VectorRetriever:
    """Retriever that searches a FAISS index.

    Parameters
    ----------
    index: faiss.IndexFlatL2
        Pre‑loaded FAISS index.
    docs: List[Document]
        List of documents in the same order as the vectors added to the index.
    """

    def __init__(self, index, docs: List[Document]):
        self.index = index
        self.docs = docs

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Document]:
        # Embed the query
        q_vec = embed_query(query).astype("float32")
        distances, indices = self.index.search(np.expand_dims(q_vec, 0), top_k)
        # FAISS returns distances; smaller is more similar
        results = []
        for idx in indices[0]:
            if idx < len(self.docs):
                results.append(self.docs[idx])
        return results

# Helper to create a retriever from the stored index/files
def get_vector_retriever() -> "VectorRetriever":
    index = load_faiss_index()
    docs = _load_documents()
    return VectorRetriever(index, docs)
