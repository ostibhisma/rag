# src/retrieval/base.py
"""Abstract base class for retrievers used in Adaptive Retrieval RAG.
All retrievers must implement a ``retrieve`` method returning a list of
`langchain.schema.Document` objects ordered by relevance.
"""
from abc import ABC, abstractmethod
from typing import List

from langchain.schema import Document


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Return the top *top_k* documents for *query*.
        """
        raise NotImplementedError
