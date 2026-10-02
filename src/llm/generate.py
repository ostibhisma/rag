# src/llm/generate.py
"""LLM generation wrapper.
Uses a HuggingFace pipeline to generate answers given a query and retrieved evidence.
"""
from typing import List

from transformers import pipeline, set_seed
from langchain.schema import Document

from ..config import LLM_MODEL, USE_GPU

# Initialize generation pipeline (text-generation). For simplicity we use default params.
_device = 0 if USE_GPU else -1
_generator = pipeline("text-generation", model=LLM_MODEL, device=_device)

# Optional: set a deterministic seed for reproducibility
set_seed(42)

def _format_context(docs: List[Document]) -> str:
    """Concatenate document contents into a single context string.
    Each document is separated by a newline.
    """
    return "\n".join(doc.page_content.strip() for doc in docs)

def generate_answer(question: str, docs: List[Document]) -> str:
    """Generate an answer to *question* using *docs* as context.

    The prompt follows a simple RAG template:
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    """
    context = _format_context(docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    # Generate with a max length of 200 tokens; adjust as needed.
    generated = _generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
    # The pipeline returns a list of dicts with 'generated_text'
    answer_text = generated[0]["generated_text"]
    # Strip the prompt part to keep only the answer.
    answer = answer_text.split("Answer:", 1)[-1].strip()
    return answer
