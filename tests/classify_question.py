def classify_question(question: str) -> dict:
    """
    Classifies a question into 3 complexity levels and determines the retrieval strategy.
    
    Levels:
    1: Simple (No retrieval needed, LLM can answer directly)
    2: Moderate (Needs specific context, Naive RAG)
    3: Complex (Needs analytical or multi-document context, Hybrid RAG)
    """
    
    question_lower = question.lower()
    
    # Simple keyword-based mock classification for demonstration purposes
    complex_keywords = ["compare", "analyze", "difference", "impact", "evaluate", "why", "trends"]
    moderate_keywords = ["what", "who", "where", "when", "how much", "details about", "company"]
    
    # Default to simple
    level = 1
    needs_retrieval = False
    retrieval_algo = "None (LLM directly)"
    
    if any(keyword in question_lower for keyword in complex_keywords):
        level = 3
        needs_retrieval = True
        retrieval_algo = "Hybrid Retrieval"
    elif any(keyword in question_lower for keyword in moderate_keywords):
        level = 2
        needs_retrieval = True
        retrieval_algo = "Naive Retrieval"
        
    return {
        "question": question,
        "level": level,
        "needs_retrieval": needs_retrieval,
        "retrieval_algo": retrieval_algo
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
    
    for q in test_questions:
        result = classify_question(q)
        print(f"Question: '{result['question']}'")
        print(f"  Level: {result['level']}")
        print(f"  Needs Retrieval: {result['needs_retrieval']}")
        print(f"  Retrieval Algorithm: {result['retrieval_algo']}")
        print("-" * 50)
