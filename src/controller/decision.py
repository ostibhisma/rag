# src/controller/decision.py
"""Decision module for Adaptive Retrieval.
Selects a retrieval strategy (vector, bm25, hybrid) based on lightweight query features.
Currently uses a simple rule‑based approach; can be swapped with a trained classifier later.
"""
from typing import Dict

# Strategy constants
VECTOR = "vector"
BM25 = "bm25"
HYBRID = "hybrid"


def decide_strategy(features: Dict[str, float]) -> str:
    """Return the chosen retrieval strategy.

    Rules (tuned to typical query patterns):
    - Very short queries (<= 15 characters) → Vector only.
    - Queries with many named entities (> 2) → Hybrid (covers both lexical & semantic).
    - Low vector similarity (< 0.2) → Hybrid.
    - High BM25 score (> 5.0) → BM25.
    - Otherwise default to Vector.
    """
    length = features.get("length_chars", 0)
    entities = features.get("num_entities", 0)
    vec_score = features.get("top_vector_score", 0)
    bm25_score = features.get("top_bm25_score", 0)

    if length <= 15:
        return VECTOR
    if entities > 2:
        return HYBRID
    if vec_score < 0.2:
        return HYBRID
    if bm25_score > 5.0:
        return BM25
    return VECTOR
