"""
rag/embeddings/base.py — EmbeddingProvider abstract base class.

To add a new embedding backend:
  1. Create a new file (e.g. rag/embeddings/huggingface.py)
  2. Subclass EmbeddingProvider and implement get_embeddings()
  3. Register in rag/retriever.py EMBEDDING_PROVIDERS dict
"""
from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings


class EmbeddingProvider(ABC):
    """
    Abstract embedding provider. Returns a LangChain-compatible Embeddings
    object that can be consumed by any VectorStoreBackend.
    """

    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Return an initialized embeddings instance."""
        ...
