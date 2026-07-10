"""
data_handlers/base_loader.py — BaseDataLoader abstract base class.

Every data source plugin must implement this interface.
The RAG core (rag/) and agent core (core/) never import from data_handlers/
directly — only the factory and ingestion script do.

To add a new data source:
  1. Create a new file (e.g. data_handlers/pdf_loader.py)
  2. Subclass BaseDataLoader and implement load() + collection_name()
  3. Register it in data_handlers/registry.py
"""
from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseDataLoader(ABC):
    """
    Abstract data loader. Converts any data source into a list of
    LangChain Documents ready for PII sanitization and embedding.
    """

    @abstractmethod
    def load(self) -> List[Document]:
        """
        Load data and return as LangChain Documents.

        Each Document should have:
          - page_content: the primary text to embed (searchable)
          - metadata: structured fields (not embedded, but retrievable)
        """
        ...

    @abstractmethod
    def collection_name(self) -> str:
        """
        Return a unique, stable identifier for this data source's
        vector store collection.

        Used as the ChromaDB collection name. Keep it snake_case and
        lowercase (e.g. "flipkart_products", "company_hr_docs").
        """
        ...
