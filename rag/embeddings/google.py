"""
rag/embeddings/google.py — Google GenerativeAI embedding provider.

Uses the text-embedding-004 model by default (same GEMINI_API_KEY).
"""
import os
from rag.embeddings.base import EmbeddingProvider


class GoogleEmbeddingProvider(EmbeddingProvider):
    """
    Google Generative AI embeddings via langchain-google-genai.
    Requires GEMINI_API_KEY to be set in environment.
    """

    def __init__(self, model_name: str = "text-embedding-004"):
        self.model_name = model_name

    def get_embeddings(self):
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file: GEMINI_API_KEY=your_key_here"
            )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=self.model_name)


# Future alternative — local, no API key required:
#
# class HuggingFaceEmbeddingProvider(EmbeddingProvider):
#     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
#         self.model_name = model_name
#     def get_embeddings(self):
#         from langchain_huggingface import HuggingFaceEmbeddings
#         return HuggingFaceEmbeddings(model_name=self.model_name)
