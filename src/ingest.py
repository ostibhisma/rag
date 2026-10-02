# src/ingest.py
"""Data ingestion utilities.
Loads documents from a directory, supports .txt and .pdf files, splits them into
chunks, generates embeddings, and stores them in ChromaDB for semantic retrieval.
"""
import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def _load_txt(file_path: Path) -> Document:
    """Load a plain text file as a LangChain Document."""
    loader = TextLoader(str(file_path))
    docs = loader.load()
    return docs[0]


def _load_pdf(file_path: Path) -> List[Document]:
    """Load a PDF file, returning a list of Documents (one per page)."""
    loader = PyPDFLoader(str(file_path))
    return loader.load_and_split()


def load_documents(source_dir: str) -> List[Document]:
    """Recursively load supported documents from *source_dir*.

    Supported extensions: .txt, .pdf
    """
    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    documents: List[Document] = []
    for file_path in source_path.rglob("*"):
        if file_path.suffix.lower() == ".txt":
            documents.append(_load_txt(file_path))
        elif file_path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(file_path))
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split documents into smaller text chunks.

    Parameters
    ----------
    documents: List[Document]
        The raw documents to split.
    chunk_size: int, default 500
        Maximum number of characters per chunk.
    chunk_overlap: int, default 50
        Number of characters to overlap between consecutive chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


class SemanticRetriever:
    """Handles embedding generation, ChromaDB storage, and semantic search."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        db_path: str = "./chroma_db",
        collection_name: str = "documents",
    ):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def ingest_chunks(self, chunks: List[Document]) -> None:
        """Generate embeddings and store chunks in ChromaDB."""
        documents = [c.page_content for c in chunks]
        embeddings = self.model.encode(documents).tolist()
        self.collection.upsert(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=documents,
            embeddings=embeddings,
        )

    def retrieve(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Semantic search: return top-n relevant chunks for *query*."""
        query_embedding = self.model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return [
            {"content": doc, "distance": dist}
            for doc, dist in zip(results["documents"][0], results["distances"][0])
        ]


def compare_retrieval(chunks: List[Document], query: str, n_results: int = 3) -> Dict[str, Any]:
    """Run keyword-based and semantic retrieval, then return both results for comparison."""
    # --- Keyword / naive retrieval (simple substring match) ---
    query_lower = query.lower()
    keyword_hits = [
        {"content": c.page_content, "score": None}
        for c in chunks
        if query_lower in c.page_content.lower()
    ][:n_results]

    # --- Semantic retrieval ---
    retriever = SemanticRetriever()
    retriever.ingest_chunks(chunks)
    semantic_hits = retriever.retrieve(query, n_results=n_results)

    return {
        "query": query,
        "keyword_results": keyword_hits,
        "semantic_results": semantic_hits,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Ingest, split, and semantically search documents.")
    parser.add_argument("source", help="Directory containing source .txt/.pdf files")
    parser.add_argument("--out", default="./data/chunks.json", help="Output path for JSON dump")
    parser.add_argument("--query", default=None, help="Optional query to run semantic retrieval & comparison")
    args = parser.parse_args()

    docs = load_documents(args.source)
    chunks = split_documents(docs)

    # Save chunks
    serialized = [
        {"page_content": d.page_content, "metadata": d.metadata} for d in chunks
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {args.out}")

    # Optional: run retrieval comparison
    if args.query:
        comparison = compare_retrieval(chunks, args.query)
        print(f"\nQuery: {comparison['query']}")
        print("\n--- Keyword Results ---")
        for i, r in enumerate(comparison["keyword_results"], 1):
            print(f"  {i}. {r['content'][:120]}...")
        print("\n--- Semantic Results ---")
        for i, r in enumerate(comparison["semantic_results"], 1):
            print(f"  {i}. [dist={r['distance']:.4f}] {r['content'][:120]}...")
