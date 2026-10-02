# Adaptive Retrieval RAG

A research‑grade prototype for **adaptive retrieval** in Retrieval‑Augmented Generation (RAG).

## Overview
- Implements three pipelines: Naïve RAG, Hybrid RAG, Adaptive RAG.
- Built with **Python**, **LangGraph** (workflow engine), **LangChain** utilities, and an **open‑source LLM**.
- Modular codebase (data ingestion, embedding, retrieval, controller, LLM wrappers, graph definition).
- Includes scripts for experiments and a Jupyter notebook demo.

## Quick Start
```bash
# Clone the repo (if not already)
git clone <repo‑url> adaptive_rag
cd adaptive_rag

# Create environment (conda example)
conda create -n adaptive_rag python=3.10 -y
conda activate adaptive_rag

# Install dependencies
pip install -r requirements.txt

# Run the demo notebook
jupyter notebook notebooks/demo.ipynb
```

## Directory Layout
```
adaptive_rag/
├─ data/                # raw docs, chunked texts, embeddings
├─ src/
│  ├─ __init__.py
│  ├─ config.py        # global settings (model ids, thresholds)
│  ├─ ingest.py        # document loading & chunking
│  ├─ embeddings.py    # embedding generation & vector store init
│  ├─ retrieval/
│  │   ├─ __init__.py
│  │   ├─ base.py      # abstract Retriever interface
│  │   ├─ vector.py    # FAISS wrapper
│  │   ├─ bm25.py      # BM25 wrapper
│  │   └─ hybrid.py    # hybrid combiner
│  ├─ controller/
│  │   ├─ __init__.py
│  │   ├─ query_analyzer.py   # lightweight feature extractor
│  │   ├─ decision.py         # rule‑based / tiny classifier
│  │   └─ evidence_checker.py # scoring & sufficiency heuristics
│  ├─ llm/
│  │   ├─ __init__.py
│  │   ├─ generate.py   # generation wrapper (open‑source LLM)
│  │   └─ verify.py     # post‑generation claim verification
│  ├─ graph.py          # LangGraph definition of pipelines
│  └─ run.py            # CLI entry point
├─ notebooks/           # demo notebooks
├─ experiments/         # scripts for baseline / adaptive runs
├─ tests/               # unit tests
└─ README.md
```
