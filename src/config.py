# src/config.py
"""Configuration constants for the Adaptive Retrieval RAG project.
All settings are chosen based on the user's preferences.
"""

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# LLM model for generation and verification (using huggingface Transformers)
LLM_MODEL = "meta-llama/Llama-2-7b-chat"

# Vector store selection
VECTOR_STORE = "faiss"  # options: faiss, chromadb, milvus

# Retrieval thresholds (these can be tuned later)
TOP_K = 5
HYBRID_WEIGHT = 0.5  # weight for linear interpolation between vector and BM25 scores

# Evidence sufficiency thresholds
EVIDENCE_SIMILARITY_THRESHOLD = 0.6
EVIDENCE_COVERAGE_THRESHOLD = 0.5

# GPU configuration (user has ~8GB GPU)
USE_GPU = True

# Paths
PROJECT_ROOT = "./adaptive_rag"
DATA_DIR = f"{PROJECT_ROOT}/data"
INDEX_PATH = f"{PROJECT_ROOT}/data/faiss_index.faiss"
BM25_INDEX_PATH = f"{PROJECT_ROOT}/data/bm25.pkl"

# Misc
RANDOM_SEED = 42
