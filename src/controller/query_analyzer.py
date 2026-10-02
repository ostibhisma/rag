# src/controller/query_analyzer.py
"""Query analysis utilities.
Extract lightweight features from a user query to inform the adaptive controller.
"""
import spacy
from typing import Dict

from ..embeddings import embed_query
from ..retrieval.vector import get_vector_retriever
from ..retrieval.bm25 import get_bm25_retriever
from ..config import TOP_K

# Load a small spaCy model for NER and tokenization (en_core_web_sm)
_nlp = spacy.load("en_core_web_sm")


def analyze_query(query: str) -> Dict[str, float]:
    """Return a feature dictionary for *query*.

    Features:
        - length_chars: number of characters
        - length_tokens: number of tokens
        - num_entities: number of named entities
        - avg_idf: average IDF of terms (using BM25 document frequencies)
        - top_vector_score: similarity of top‑k vector result (average distance)
        - top_bm25_score: BM25 score of top result
    """
    doc = nlp(query)
    length_chars = len(query)
    length_tokens = len([t for t in doc])
    num_entities = len(doc.ents)

    # BM25 average IDF approximation: use document frequencies from built BM25 index
    bm25_ret = get_bm25_retriever()
    # Approximate IDF by using BM25's internal corpus size and term frequencies
    # Since rank_bm25 does not expose IDF directly, we approximate using token frequencies
    tokenized = query.split()
    # Use bm25's idf_ attribute if available (private), fallback to 0
    try:
        idf_vals = bm25_ret.bm25.idf
        # Map token to idf if present
        idf_dict = {bm25_ret.bm25.term_freqs[i]: idf_vals[i] for i in range(len(idf_vals))}
        idf_scores = [idf_dict.get(tok.lower(), 0.0) for tok in tokenized]
        avg_idf = sum(idf_scores) / len(idf_scores) if idf_scores else 0.0
    except Exception:
        avg_idf = 0.0

    # Vector top‑k similarity (using distances; smaller distance => higher similarity)
    vector_ret = get_vector_retriever()
    # Get distances from FAISS index directly
    q_vec = embed_query(query).astype("float32")
    distances, _ = vector_ret.index.search(q_vec.reshape(1, -1), TOP_K)
    # Convert distances to similarity (negative distance)
    top_vector_score = -float(distances[0].mean()) if distances.size else 0.0

    # BM25 top score
    bm25_scores = bm25_ret.bm25.get_scores(tokenized)
    top_bm25_score = float(bm25_scores.max()) if bm25_scores.size else 0.0

    return {
        "length_chars": float(length_chars),
        "length_tokens": float(length_tokens),
        "num_entities": float(num_entities),
        "avg_idf": float(avg_idf),
        "top_vector_score": float(top_vector_score),
        "top_bm25_score": float(top_bm25_score),
    }
