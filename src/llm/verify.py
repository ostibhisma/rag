# src/llm/verify.py
"""Verification wrapper for generated answers.
Uses the same LLM to check whether each statement in the answer is supported
by the retrieved evidence. Returns a boolean flag and optional feedback.
"""
from typing import List, Tuple

from transformers import pipeline
from langchain.schema import Document

from ..config import LLM_MODEL, USE_GPU

_device = 0 if USE_GPU else -1
_verifier = pipeline("text-generation", model=LLM_MODEL, device=_device)

def _format_context(docs: List[Document]) -> str:
    return "\n".join(doc.page_content.strip() for doc in docs)

def verify_answer(question: str, answer: str, docs: List[Document]) -> Tuple[bool, str]:
    """Return ``(is_supported, feedback)``.
    The prompt asks the LLM to state whether all claims in *answer* are
    substantiated by *docs*. If any claim is unsupported, it should respond with
    "No" and optionally list missing evidence.
    """
    context = _format_context(docs)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        "Is the answer fully supported by the evidence above? Respond with 'Yes' or 'No' and provide a brief justification."
    )
    generated = _verifier(prompt, max_new_tokens=50, do_sample=False)
    response = generated[0]["generated_text"].split("Answer:", 1)[-1].strip()
    # Simple parsing: look for Yes/No at start
    lowered = response.lower()
    is_supported = lowered.startswith("yes")
    return is_supported, response
