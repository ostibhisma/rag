# tests/classify_question.py
"""
Question Classifier using Open-Source Qwen LLM for Adaptive RAG.

Classifies user questions into 3 complexity levels:
  Level 1: Simple (No retrieval needed, LLM answers directly)
  Level 2: Moderate (Requires single-source / specific context -> Naive RAG)
  Level 3: Complex (Requires multi-document / analytical reasoning -> Hybrid RAG)
"""

import json
import re
from typing import Dict, Any, Optional

# Global cache for Qwen LLM classifier instance to prevent re-loading on each call
_QWEN_CLASSIFIER_INSTANCE = None


def classify_question_rule_based(question: str) -> dict:
    """
    Fallback keyword-based query classification.
    """
    question_lower = question.lower()
    
    complex_keywords = ["compare", "analyze", "difference", "impact", "evaluate", "why", "trends"]
    moderate_keywords = ["what", "who", "where", "when", "how much", "details about", "company"]
    
    level = 1
    if any(keyword in question_lower for keyword in complex_keywords):
        level = 3
    elif any(keyword in question_lower for keyword in moderate_keywords):
        level = 2
        
    return {
        "level": level,
        "reasoning": "Keyword-based rule matching."
    }


class QwenLLMClassifier:
    """
    Open-Source Qwen LLM Classifier supporting HuggingFace Transformers and Ollama backends.
    Default Model: 'Qwen/Qwen2.5-1.5B-Instruct' (or 'qwen2.5' for Ollama)
    """

    CLASSIFICATION_PROMPT = """You are an expert query classifier for an Adaptive RAG system.
Analyze the given question and classify it into exactly one of three complexity levels:

Level 1 (Simple): General greetings, basic conversation, definitions, or broad general knowledge that an LLM can answer directly without external company or private document retrieval.
Level 2 (Moderate): Specific factual queries, definitions, or lookups requiring context from private/company documents (retrieval needed, Naive RAG).
Level 3 (Complex): Analytical, comparative, multi-part, or trend questions requiring context synthesis across multiple documents or complex reasoning (retrieval needed, Hybrid RAG).

Return your decision strictly as a valid JSON object with the following format:
{{
    "level": <1, 2, or 3>,
    "reasoning": "<brief one sentence explanation>"
}}

Question: "{question}"
JSON Response:"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        backend: str = "transformers",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.backend = backend
        self.pipeline = None
        self.tokenizer = None
        self.model = None

        if self.backend == "transformers":
            self._init_transformers(device)

    def _init_transformers(self, device: str):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch
        except ImportError:
            print("Warning: HuggingFace 'transformers' or 'torch' not installed. Falling back to rule-based.")
            return

        print(f"Loading Open-Source Qwen model '{self.model_name}' via HuggingFace Transformers...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
                trust_remote_code=True
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            print("Qwen LLM classifier loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load Qwen model '{self.model_name}': {e}")
            self.pipeline = None

    def classify_with_transformers(self, question: str) -> Dict[str, Any]:
        prompt = self.CLASSIFICATION_PROMPT.format(question=question)
        
        messages = [
            {"role": "system", "content": "You are a precise JSON query classifier for RAG systems."},
            {"role": "user", "content": prompt}
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        outputs = self.pipeline(
            formatted_prompt,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False
        )

        generated_text = outputs[0]["generated_text"]
        if isinstance(generated_text, list):
            raw_response = generated_text[-1]["content"]
        else:
            raw_response = generated_text[len(formatted_prompt):] if generated_text.startswith(formatted_prompt) else generated_text

        return self._parse_json_response(raw_response)

    def classify_with_ollama(self, question: str, ollama_url: str = "http://localhost:11434/api/generate") -> Dict[str, Any]:
        try:
            import requests
        except ImportError:
            print("Warning: 'requests' library not available for Ollama backend.")
            return classify_question_rule_based(question)

        prompt = self.CLASSIFICATION_PROMPT.format(question=question)
        payload = {
            "model": self.model_name if "qwen" in self.model_name.lower() else "qwen2.5",
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        try:
            res = requests.post(ollama_url, json=payload, timeout=30)
            res.raise_for_status()
            raw_response = res.json().get("response", "")
            return self._parse_json_response(raw_response)
        except Exception as e:
            print(f"Ollama request failed: {e}. Falling back to rule-based.")
            return classify_question_rule_based(question)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                level = int(data.get("level", 1))
                reasoning = str(data.get("reasoning", "Qwen LLM classification."))
                if level not in [1, 2, 3]:
                    level = 1
                return {"level": level, "reasoning": reasoning}
        except Exception:
            pass
        return {"level": 1, "reasoning": "Defaulting to Level 1 due to output parsing."}

    def classify(self, question: str) -> Dict[str, Any]:
        if self.backend == "ollama":
            return self.classify_with_ollama(question)
        elif self.pipeline is not None:
            return self.classify_with_transformers(question)
        else:
            return classify_question_rule_based(question)


def classify_question(
    question: str,
    use_llm: bool = True,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    backend: str = "transformers"
) -> dict:
    """
    Classifies a question into 3 complexity levels and determines the retrieval strategy.
    
    Levels:
    1: Simple (No retrieval needed, LLM can answer directly)
    2: Moderate (Needs specific context, Naive RAG)
    3: Complex (Needs analytical or multi-document context, Hybrid RAG)
    
    Args:
        question: User query string.
        use_llm: Whether to use open-source Qwen LLM as classifier (default: False for fast test runs).
        model_name: Qwen model identifier (e.g., 'Qwen/Qwen2.5-1.5B-Instruct' or 'qwen2.5').
        backend: Backend framework ('transformers' or 'ollama').
    """
    global _QWEN_CLASSIFIER_INSTANCE

    if use_llm:
        if _QWEN_CLASSIFIER_INSTANCE is None or _QWEN_CLASSIFIER_INSTANCE.model_name != model_name or _QWEN_CLASSIFIER_INSTANCE.backend != backend:
            _QWEN_CLASSIFIER_INSTANCE = QwenLLMClassifier(model_name=model_name, backend=backend)
        cls_result = _QWEN_CLASSIFIER_INSTANCE.classify(question)
    else:
        cls_result = classify_question_rule_based(question)

    level = cls_result.get("level", 1)
    reasoning = cls_result.get("reasoning", "")

    if level == 3:
        needs_retrieval = True
        retrieval_algo = "Hybrid Retrieval"
    elif level == 2:
        needs_retrieval = True
        retrieval_algo = "Naive Retrieval"
    else:
        level = 1
        needs_retrieval = False
        retrieval_algo = "None (LLM directly)"
        
    return {
        "question": question,
        "level": level,
        "needs_retrieval": needs_retrieval,
        "retrieval_algo": retrieval_algo,
        "reasoning": reasoning,
        "classifier": f"Qwen ({backend})" if use_llm else "Rule-based"
    }


if __name__ == "__main__":
    test_questions = [
        # Level 1: Simple
        "Hello, how are you?",
        "What is the capital of France?", 
        
        # Level 2: Moderate (requires context)
        "What are the company's core values?",
        "Who is the CEO of the company?",
        
        # Level 3: Complex (requires multi-hop or analytical context)
        "Compare our Q3 revenue with Q2 and analyze the impact of the new marketing strategy.",
        "Why did the stock price drop last week and what are the trends?"
    ]

    print("\n=== Testing Rule-based Classifier ===")
    for q in test_questions:
        result = classify_question(q, use_llm=False)
        print(f"Question: '{result['question']}'")
        print(f"  Level: {result['level']}")
        print(f"  Needs Retrieval: {result['needs_retrieval']}")
        print(f"  Retrieval Algorithm: {result['retrieval_algo']}")
        print(f"  Classifier: {result['classifier']}")
        print("-" * 50)

    # To ÷run with open-source Qwen LLM Classifier (Transformers or Ollama backend):
    result = classify_question(
        question="Compare Q3 and Q2 revenue trends.",
        use_llm=True,
        model_name="Qwen/Qwen2.5-1.5B-Instruct", # or "qwen2.5" for Ollama
        backend="transformers"                   # or "ollama"
    )
    print(result)
    result = classify_question(
        question="who is president of usa?",
        use_llm=True,
        model_name="Qwen/Qwen2.5-1.5B-Instruct", # or "qwen2.5" for Ollama
        backend="transformers"                   # or "ollama"
    )
    print(result)


