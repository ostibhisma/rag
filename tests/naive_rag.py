import json
import re
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi


# ============================================================
# 1. LOAD QWEN
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 3. SIMPLE DOCUMENT COLLECTION
# ============================================================

documents = [
    "The Eiffel Tower is located in Paris, France.",
    
    "Paris is the capital city of France.",
    
    "The Eiffel Tower was completed in 1889.",
    
    "Albert Einstein developed the theory of relativity.",
    
    "Einstein was born in Germany in 1879.",
    
    "The theory of relativity includes special relativity and general relativity.",
    
    "Python is a programming language widely used in data science.",
    
    "Machine learning is a subset of artificial intelligence.",
    
    "Retrieval Augmented Generation combines information retrieval with language generation.",
    
    "RAG can reduce hallucination by providing external evidence to a language model."
]


# ============================================================
# 4. CREATE VECTOR INDEX
# ============================================================

embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# ============================================================
# 5. BM25 INDEX
# ============================================================

tokenized_docs = [
    doc.lower().split()
    for doc in documents
]

bm25 = BM25Okapi(tokenized_docs)


# ============================================================
# 6. LLM-BASED QUESTION & COMPLEXITY CLASSIFIER
# ============================================================

def classify_question_llm(question):
    """
    LLM-based classifier using Qwen to analyze user questions.
    Determines whether RAG (retrieval) is required and assesses query complexity.
    """
    prompt = f"""You are an expert query classifier for an Adaptive RAG system.
Analyze the user question and classify it into:
1. "needs_rag": boolean (true or false).
   - Set false ONLY for general greetings (e.g. "Hello"), casual small talk, or broad high-level concept definitions (e.g. "What is Python?").
   - Set true for specific factual questions asking for exact locations, dates, historical facts, entity relationships, comparison between theories, or domain document details.
2. "complexity": string ("simple", "medium", or "complex").
   - "simple": Direct single-fact query or specific lookup.
   - "medium": Multi-entity question or relational query.
   - "complex": Comparative analysis, multi-hop reasoning, or deep synthesis.

Return STRICTLY a valid JSON object with no additional text:
{{
    "needs_rag": true,
    "complexity": "simple",
    "reasoning": "<brief explanation>"
}}

Question: "{question}"
JSON Response:"""

    messages = [
        {"role": "system", "content": "You are a precise JSON query classifier for RAG systems."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.1,
        do_sample=False
    )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            needs_rag = bool(data.get("needs_rag", True))
            complexity = str(data.get("complexity", "simple")).lower()
            if complexity not in ["simple", "medium", "complex"]:
                complexity = "simple"
            reasoning = str(data.get("reasoning", ""))
            return {
                "needs_rag": needs_rag,
                "complexity": complexity,
                "reasoning": reasoning
            }
    except Exception:
        pass

    return {
        "needs_rag": True,
        "complexity": "simple",
        "reasoning": "Defaulting due to parsing."
    }


def question_needs_rag(question):
    return classify_question_llm(question)["needs_rag"]


def classify_complexity(question):
    return classify_question_llm(question)["complexity"]


# ============================================================
# 8. NAIVE RAG
# ============================================================

def naive_rag(question, k=3):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = [
        documents[i]
        for i in indices[0]
    ]

    return results


# ============================================================
# 9. HYBRID RAG
# ============================================================

def hybrid_rag(question, k=3):

    # Vector retrieval
    vector_results = naive_rag(question, k=5)

    # BM25 retrieval
    tokenized_query = question.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:5]

    bm25_results = [
        documents[i]
        for i in bm25_indices
    ]

    # Combine results
    combined = []

    for doc in vector_results + bm25_results:

        if doc not in combined:
            combined.append(doc)

    return combined[:k]


# ============================================================
# 10. ADAPTIVE RAG
# ============================================================

def adaptive_rag(question):

    complexity = classify_complexity(question)

    if complexity == "simple":

        print("Adaptive decision: Simple question")

        return naive_rag(question)

    elif complexity == "medium":

        print("Adaptive decision: Medium question")

        return hybrid_rag(question)

    else:

        print("Adaptive decision: Complex question")

        # Start with hybrid retrieval
        results = hybrid_rag(question, k=5)

        # In a more advanced version you can:
        #
        # 1. Check evidence
        # 2. Reformulate question
        # 3. Retrieve again
        # 4. Verify answer

        return results


# ============================================================
# 11. QWEN GENERATION
# ============================================================

def generate_answer(question, context=None):

    if context:

        context_text = "\n".join(
            f"- {doc}"
            for doc in context
        )

        prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context_text}

Question:
{question}

Give a concise and factual answer.
"""

    else:

        prompt = f"""
Answer the following question directly.

Question:
{question}
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200
    )

    generated = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    )


# ============================================================
# 12. COMPLETE ADAPTIVE RAG PIPELINE
# ============================================================

def adaptive_pipeline(question):

    print("\nQUESTION:")
    print(question)

    # ---------------------------------------
    # STEP 1: LLM-BASED QUESTION CLASSIFICATION
    # ---------------------------------------

    cls_result = classify_question_llm(question)
    needs_rag = cls_result["needs_rag"]
    complexity = cls_result["complexity"]
    reasoning = cls_result.get("reasoning", "")

    if not needs_rag:

        print("\nRAG decision: NO RAG")
        if reasoning:
            print("Reasoning:", reasoning)

        answer = generate_answer(question)

        return {
            "question": question,
            "rag": False,
            "algorithm": "None",
            "answer": answer
        }

    # ---------------------------------------
    # STEP 2: QUESTION COMPLEXITY
    # ---------------------------------------

    print("\nRAG decision: YES")
    print("Complexity:", complexity)
    if reasoning:
        print("Reasoning:", reasoning)

    # ---------------------------------------
    # STEP 3: SELECT RAG ALGORITHM
    # ---------------------------------------

    if complexity == "simple":

        algorithm = "Naive RAG"

        context = naive_rag(question)

    elif complexity == "medium":

        algorithm = "Hybrid RAG"

        context = hybrid_rag(question)

    else:

        algorithm = "Adaptive RAG"

        context = adaptive_rag(question)

    print("Selected algorithm:", algorithm)

    print("\nRetrieved context:")

    for doc in context:
        print("-", doc)

    # ---------------------------------------
    # STEP 4: GENERATE ANSWER
    # ---------------------------------------

    answer = generate_answer(
        question,
        context
    )

    return {
        "question": question,
        "rag": True,
        "complexity": complexity,
        "algorithm": algorithm,
        "context": context,
        "answer": answer
    }


# ============================================================
# 13. TEST
# ============================================================

questions = [
    "What is Python?",
    "Where is the Eiffel Tower located?",
    "When was the Eiffel Tower completed?",
    "What is the difference between special relativity and general relativity?",
    "How is Einstein related to the theory of relativity?"
]


for question in questions:

    result = adaptive_pipeline(question)

    print("\n==============================")
    print("FINAL ANSWER:")
    print(result["answer"])
    print("==============================")