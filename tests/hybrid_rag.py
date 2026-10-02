"""
Hybrid RAG example combining Semantic (ChromaDB) and Keyword (BM25) search using Reciprocal Rank Fusion (RRF).
"""

import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# Reuse utilities from ingest.py
from ingest import load_txt, split_documents, SemanticRetriever

def compute_rrf(semantic_results: List[Dict[str, Any]], keyword_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
    """
    Compute Reciprocal Rank Fusion (RRF) for semantic and keyword results.
    """
    rrf_scores: Dict[str, float] = {}

    # rank semantic results
    for rank, res in enumerate(semantic_results):
        content = res["content"]
        if content not in rrf_scores:
            rrf_scores[content] = 0.0
        rrf_scores[content] += 1 / (k + rank + 1)

    # rank keyword results
    for rank, res in enumerate(keyword_results):
        content = res["content"]
        if content not in rrf_scores:
            rrf_scores[content] = 0.0
        rrf_scores[content] += 1 / (k + rank + 1)
        
    # Sort by RRF score descending
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [{"content": content, "rrf_score": score} for content, score in sorted_results]


def bm25_retrieval(chunks: List[Document], query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    Keyword retrieval using BM25.
    """
    tokenized_corpus = [doc.page_content.lower().split(" ") for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split(" ")
    
    scores = bm25.get_scores(tokenized_query)
    
    # Pair scores with docs
    doc_scores = [{"content": chunks[i].page_content, "score": scores[i]} for i in range(len(chunks))]
    
    # Sort by score
    doc_scores = sorted(doc_scores, key=lambda x: x["score"], reverse=True)
    
    return doc_scores[:n_results]


def compare_hybrid(chunks: List[Document], query: str, n: int = 3) -> Dict[str, Any]:
    # 1. Semantic
    sem = SemanticRetriever(collection_name="company_docs_hybrid")
    sem.ingest(chunks)
    sem_results = sem.retrieve(query, n_results=n)
    
    # 2. Keyword (BM25)
    bm25_results = bm25_retrieval(chunks, query, n_results=n)
    
    # 3. Hybrid
    hybrid_results = compute_rrf(sem_results, bm25_results, k=60)
    
    return {
        "query": query,
        "semantic_results": sem_results,
        "keyword_results": bm25_results,
        "hybrid_results": hybrid_results[:n]
    }


if __name__ == "__main__":
    doc_path = Path(__file__).parents[1] / "tests" / "docs" / "company.txt"
    if not doc_path.is_file():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    raw_doc = load_txt(doc_path)
    chunks = split_documents([raw_doc])
    
    query = "What is NexusFlow-X1 use for"
    
    result = compare_hybrid(chunks, query)
    
    print("\nQUESTION:")
    print(result["query"])

    print("\n--- Semantic RESULTS ---")
    for i, r in enumerate(result["semantic_results"], start=1):
        excerpt = r["content"].replace("\n", " ")[:120]
        print(f"  {i}. [dist={r.get('distance', 0):.4f}] {excerpt}...")

    print("\n--- Keyword (BM25) RESULTS ---")
    for i, r in enumerate(result["keyword_results"], start=1):
        excerpt = r["content"].replace("\n", " ")[:120]
        print(f"  {i}. [score={r.get('score', 0):.4f}] {excerpt}...")
        
    print("\n--- Hybrid (RRF) RESULTS ---")
    for i, r in enumerate(result["hybrid_results"], start=1):
        excerpt = r["content"].replace("\n", " ")[:120]
        print(f"  {i}. [rrf_score={r.get('rrf_score', 0):.4f}] {excerpt}...")
