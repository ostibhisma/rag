# ingest.py – basic RAG example using LangChain, ChromaDB and an open‑source embedding model
"""A minimal script that:
1. Loads the `company.txt` document.
2. Splits it into small chunks.
3. Generates embeddings with `sentence‑transformers/all‑MiniLM‑L6‑v2`.
4. Stores the vectors in a local ChromaDB collection.
5. Executes a sample query and shows **semantic** vs **keyword** (simple substring) results.

The implementation mirrors the richer utilities found in `src/ingest.py` but stays compact
so it can be run directly from the `tests/` folder.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Helper: load a single text file as a LangChain Document
# ---------------------------------------------------------------------------
def load_txt(file_path: Path) -> Document:
    """Load a plain‑text file and return a LangChain Document."""
    loader = TextLoader(str(file_path))
    docs = loader.load()
    return docs[0]

# ---------------------------------------------------------------------------
# Chunk the documents – a tiny chunk size keeps the demo readable
# ---------------------------------------------------------------------------
def split_documents(docs: List[Document], chunk_size: int = 120, chunk_overlap: int = 5) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

# ---------------------------------------------------------------------------
# Semantic retrieval powered by SentenceTransformer + ChromaDB
# ---------------------------------------------------------------------------
class SemanticRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", db_path: str = "./chroma_db", collection_name: str = "company_docs"):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def ingest(self, chunks: List[Document]) -> None:
        texts = [c.page_content for c in chunks]
        embeddings = self.model.encode(texts).tolist()
        self.collection.upsert(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=texts,
            embeddings=embeddings,
        )

    def retrieve(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        query_emb = self.model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_emb], n_results=n_results)
        return [
            {"content": doc, "distance": dist}
            for doc, dist in zip(results["documents"][0], results["distances"][0])
        ]

# ---------------------------------------------------------------------------
# Simple keyword (naïve) retrieval – case‑insensitive substring match
# ---------------------------------------------------------------------------
def keyword_retrieval(chunks: List[Document], query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    q = query.lower()
    hits = []
    for chunk in chunks:
        if q in chunk.page_content.lower():
            hits.append({"content": chunk.page_content, "score": None})
        if len(hits) >= n_results:
            break
    return hits

# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------
def compare(chunks: List[Document], query: str, n: int = 3) -> Dict[str, Any]:
    kw = keyword_retrieval(chunks, query, n_results=n)
    sem = SemanticRetriever()
    sem.ingest(chunks)
    sr = sem.retrieve(query, n_results=n)
    return {
        "query": query,
        "keyword_results": kw,
        "semantic_results": sr,
    }

# ---------------------------------------------------------------------------
# Main execution – ingest `company.txt` and run a demonstration query
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load the example document (relative to the repository root)
    doc_path = Path(__file__).parents[1]/ "tests" / "docs" / "company.txt"
    if not doc_path.is_file():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    # 1️⃣ Load & split
    raw_doc = load_txt(doc_path)
    chunks = split_documents([raw_doc])
    print("Number of chunks:", len(chunks))

    # 2️⃣ Sample query
    query = "where is company location ?"

    # 3️⃣ Run comparison
    result = compare(chunks, query)

    # -------------------------------------------------------------------
    # Pretty‑print the outcomes – mimicking the earlier demo output
    # -------------------------------------------------------------------
    print("\nQUESTION:")
    print(result["query"])

    print("\n--- Keyword (BM25‑like) RESULTS ---")
    for i, r in enumerate(result["keyword_results"], start=1):
        excerpt = r["content"].replace("\n", " ")[:120]
        print(f"  {i}. {excerpt}...")

    print("\n--- Semantic RESULTS ---")
    for i, r in enumerate(result["semantic_results"], start=1):
        excerpt = r["content"].replace("\n", " ")
        print(f"  {i}. [dist={r['distance']:.4f}] {excerpt}...")

    # Clean up – close the Chroma client (optional but tidy)
    # The client will be closed when the script exits.
