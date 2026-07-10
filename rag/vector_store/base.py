"""
rag/vector_store/base.py — VectorStoreBackend abstract base class.

To add a new vector store (e.g. FAISS, Pinecone):
  1. Create a new file (e.g. rag/vector_store/faiss.py)
  2. Subclass VectorStoreBackend and implement all three methods
  3. Register in rag/retriever.py VECTOR_STORE_BACKENDS dict
"""
from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class VectorStoreBackend(ABC):
    """
    Abstract vector store backend. Handles chunking, embedding,
    persistence, and retrieval behind a uniform interface.
    """

    @abstractmethod
    def build(self, documents: List[Document], collection_name: str) -> None:
        """
        Chunk, embed, and persist a list of documents.
        Called once during ingestion (scripts/ingest.py).
        """
        ...

    @abstractmethod
    def get_retriever(self, collection_name: str, k: int = 5) -> BaseRetriever:
        """
        Return a LangChain retriever for the given collection.
        Called at agent startup to wire the RAG tool.
        """
        ...

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check whether a collection has already been indexed.
        Used to skip re-ingestion on subsequent runs.
        """
        ...
