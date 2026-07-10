"""
rag/vector_store/chroma.py — ChromaDB vector store backend.

File-persisted, zero-infrastructure local vector store.
Swap to Pinecone or Weaviate by adding a new backend class — no other code changes.
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.vector_store.base import VectorStoreBackend


class ChromaBackend(VectorStoreBackend):
    """
    ChromaDB-backed vector store with recursive character chunking.

    Persists to disk at `persist_dir`. Subsequent runs detect the existing
    collection and skip re-indexing (unless --force is passed to ingest.py).
    """

    def __init__(
        self,
        embeddings,
        persist_dir: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.embeddings = embeddings
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def build(self, documents: List[Document], collection_name: str) -> None:
        """Chunk, embed, and persist documents to ChromaDB in rate-limited batches."""
        from langchain_chroma import Chroma
        import time

        print(f"  -> Splitting {len(documents)} documents into chunks "
              f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        chunks = self._splitter.split_documents(documents)
        print(f"  -> {len(chunks)} chunks produced. Embedding and persisting in rate-limited batches...")

        # Delete existing collection if it exists to avoid dimension mismatch errors
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            existing = [c.name for c in client.list_collections()]
            if collection_name in existing:
                client.delete_collection(collection_name)
                print(f"  -> Deleted existing collection '{collection_name}' for clean rebuild.")
        except Exception as e:
            print(f"  -> Note: could not delete collection directly: {e}")

        # Batch write to stay strictly below the 100 requests per minute free quota
        # (each chunk in a batch counts as 1 request toward the 100/min ceiling)
        batch_size = 80
        store = None
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            print(f"     Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} (size {len(batch)})...")
            
            if i == 0:
                store = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    collection_name=collection_name,
                    persist_directory=self.persist_dir,
                )
            else:
                store.add_documents(batch)
                
            if i + batch_size < len(chunks):
                print("     Sleeping 65 seconds to reset Google free-tier rate limits...")
                time.sleep(65)

        print(f"  -> Persisted to '{self.persist_dir}' (collection: '{collection_name}').")


    def get_retriever(self, collection_name: str, k: int = 5):
        """Return a retriever wired to the specified ChromaDB collection."""
        from langchain_chroma import Chroma

        store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        return store.as_retriever(search_kwargs={"k": k})

    def collection_exists(self, collection_name: str) -> bool:
        """Check if the collection has already been indexed."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            existing = [c.name for c in client.list_collections()]
            return collection_name in existing
        except Exception:
            return False
