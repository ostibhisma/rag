# src/embeddings.py
"""Embedding utilities for Adaptive Retrieval RAG.

Provides functions to generate embeddings for documents and queries using the
configured SentenceTransformer model, and to manage a FAISS index.
"""
import os
import numpy as np
import faiss
from typing import List

from sentence_transformers import SentenceTransformer
from langchain.schema import Document

from .config import EMBEDDING_MODEL, INDEX_PATH, USE_GPU, RANDOM_SEED

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Load embedding model (GPU if available and requested)
def _load_model() -> SentenceTransformer:
    if USE_GPU:
        # sentence-transformers will automatically use CUDA if available
        model = SentenceTransformer(EMBEDDING_MODEL, device="cuda")
    else:
        model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return model

_model = _load_model()


def embed_documents(documents: List[Document]) -> np.ndarray:
    """Return a (N, D) ndarray of embeddings for a list of LangChain Documents.

    Parameters
    ----------
    documents: List[Document]
        Documents whose ``page_content`` will be embedded.
    """
    texts = [doc.page_content for doc in documents]
    embeddings = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Encode a single query string into a 1‑D embedding vector.
    """
    return _model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]


def init_faiss_index(d: int) -> faiss.IndexFlatL2:
    """Initialize a FAISS index for *d*-dimensional vectors.
    """
    index = faiss.IndexFlatL2(d)
    return index


def load_faiss_index() -> faiss.IndexFlatL2:
    """Load an existing FAISS index from ``INDEX_PATH`` or create a new empty one.
    """
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = None
    return index


def save_faiss_index(index: faiss.IndexFlatL2) -> None:
    """Persist the FAISS index to disk.
    """
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)


def add_documents_to_index(index: faiss.IndexFlatL2, docs: List[Document]):
    """Compute embeddings for *docs* and add them to *index*.
    Returns the updated index and the list of docs (for lookup).
    """
    embeddings = embed_documents(docs)
    if index is None:
        index = init_faiss_index(embeddings.shape[1])
    index.add(embeddings)
    return index, docs
