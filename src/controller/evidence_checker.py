# src/controller/evidence_checker.py
"""Evidence checking utilities.
Given retrieved documents and the query, compute simple metrics to decide
whether the evidence is sufficient for answer generation.
"""
from typing import List, Tuple

from langchain.schema import Document

from ..config import EVIDENCE_SIMILARITY_THRESHOLD, EVIDENCE_COVERAGE_THRESHOLD, TOP_K
from ..embeddings import embed_documents, embed_query
import numpy as np


def _average_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> float:
    """Return the average cosine similarity between query and each document vector."""
    # Normalize vectors
    q_norm = query_vec / np.linalg.norm(query_vec)
    d_norm = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    sims = np.dot(d_norm, q_norm)
    return float(sims.mean()) if sims.size else 0.0


def check_evidence(query: str, docs: List[Document]) -> Tuple[bool, dict]:
    """Determine if the retrieved *docs* provide sufficient evidence.

    Returns a tuple ``(is_sufficient, metrics)`` where ``metrics`` contains the
    computed similarity and coverage scores.
    """
    if not docs:
        return False, {"similarity": 0.0, "coverage": 0.0}

    query_vec = embed_query(query)
    doc_vecs = embed_documents(docs)
    similarity = _average_similarity(query_vec, doc_vecs)

    # Coverage: proportion of retrieved docs that contain at least one named entity
    # (simple proxy for relevance). We'll use spaCy for entity detection.
    import spacy
    nlp = spacy.load("en_core_web_sm")
    ent_counts = 0
    for doc in docs:
        spacy_doc = nlp(doc.page_content)
        if spacy_doc.ents:
            ent_counts += 1
    coverage = ent_counts / len(docs)

    is_sufficient = (similarity >= EVIDENCE_SIMILARITY_THRESHOLD) and (coverage >= EVIDENCE_COVERAGE_THRESHOLD)
    metrics = {"similarity": similarity, "coverage": coverage}
    return is_sufficient, metrics
