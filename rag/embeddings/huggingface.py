"""
rag/embeddings/huggingface.py — HuggingFace local embedding provider.
"""
from rag.embeddings.base import EmbeddingProvider

class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        
    def get_embeddings(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=self.model_name)
