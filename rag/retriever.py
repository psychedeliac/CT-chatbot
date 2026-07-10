"""
rag/retriever.py — Builds the RAG tool from AgentConfig.

build_rag_tool(config) resolves the configured embedding provider and vector
store backend, then wraps the retriever as a named LangChain tool that the
agent can call just like any other tool.
"""
from langchain.tools import tool
from langchain_core.documents import Document

from config import AgentConfig
from rag.embeddings.google import GoogleEmbeddingProvider
from rag.embeddings.huggingface import HuggingFaceEmbeddingProvider

# ── Provider Registries ────────────────────────────────────────────────────────
# Add new providers here; nothing else in the codebase needs to change.

EMBEDDING_PROVIDERS = {
    "google": GoogleEmbeddingProvider,
    "huggingface": HuggingFaceEmbeddingProvider,
}

from rag.vector_store.chroma import ChromaBackend

VECTOR_STORE_BACKENDS = {
    "chroma": ChromaBackend,
    # "faiss": FAISSBackend,
    # "pinecone": PineconeBackend,
}


def build_rag_tool(config: AgentConfig):
    """
    Resolve embedding + vector store from config and return a LangChain @tool
    that the agent can call to retrieve relevant documents.

    The returned tool is registered under the name "rag" in the tool registry
    by the caller (main.py or scripts/ingest.py).
    """
    # 1. Resolve embedding provider
    emb_cls = EMBEDDING_PROVIDERS.get(config.embedding_provider)
    if not emb_cls:
        raise ValueError(
            f"Unknown embedding provider: '{config.embedding_provider}'. "
            f"Available: {list(EMBEDDING_PROVIDERS.keys())}"
        )
    embeddings = emb_cls(model_name=config.embedding_model).get_embeddings()

    # 2. Resolve vector store backend
    vs_cls = VECTOR_STORE_BACKENDS.get(config.vector_store_backend)
    if not vs_cls:
        raise ValueError(
            f"Unknown vector store backend: '{config.vector_store_backend}'. "
            f"Available: {list(VECTOR_STORE_BACKENDS.keys())}"
        )
    vector_store = vs_cls(
        embeddings=embeddings,
        persist_dir=config.vector_store_persist_dir,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    # 3. Get semantic store directly to access distance scores
    from langchain_chroma import Chroma
    chroma_store = Chroma(
        collection_name=config.rag_collection,
        embedding_function=embeddings,
        persist_directory=config.vector_store_persist_dir,
    )

    # 4. Initialize BM25 keyword retriever
    from data_handlers.enriched_loader import EnrichedLoader
    from langchain_community.retrievers import BM25Retriever
    import re
    
    # Common stop words and ubiquitous domain terms to ignore in keyword searches
    STOP_WORDS = {
        # Question words
        "what", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "has", "have", "had",
        "who", "whom", "whose", "which", "where", "when", "why", "how",
        # Pronouns
        "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        # Prepositions, articles & conjunctions
        "a", "an", "the", "in", "on", "for", "with", "about", "of", "to", "and", "or", "but", "because", "as", "until", "while",
        "at", "by", "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", "up", "down",
        "out", "off", "over", "under", "again", "further", "then", "once",
        # Domain terms
        "corporate", "turnaround", "company", "business", "businesses", "program", "service", "services", "client", "clients"
    }
    
    def preprocess_func(text: str) -> list[str]:
        # Lowercase and split into alphanumeric tokens
        tokens = re.findall(r'\b\w+\b', text.lower())
        filtered = [t for t in tokens if t not in STOP_WORDS]
        # If everything was filtered out, fallback to original tokens to avoid empty queries
        return filtered if filtered else tokens
    
    bm25_retriever = None
    try:
        loader = EnrichedLoader()
        documents = loader.load()
        if documents:
            bm25_retriever = BM25Retriever.from_documents(
                documents,
                preprocess_func=preprocess_func
            )
            bm25_retriever.k = config.rag_k
    except Exception as e:
        print(f"[Warning] Failed to initialize BM25 retriever: {e}. Falling back to semantic search only.")

    # 5. Wrap search tool with custom rank matching logic
    @tool
    def rag_search(query: str) -> str:
        """
        Search the indexed knowledge base to answer questions.
        Use this for domain-specific questions about products, data, or
        any content that has been ingested into the knowledge base.
        """
        k = config.rag_k or 5
        chroma_docs_with_score = []
        # 1. Query semantic/synonym search (Chroma) with distance score filtering
        try:
            chroma_docs_with_score = chroma_store.similarity_search_with_score(query, k=k)
            # Distance <= 0.80 filters out noisy, out-of-domain chunks
            chroma_docs = [doc for doc, score in chroma_docs_with_score if score <= 0.80]
        except Exception as e:
            print(f"[Warning] Chroma search failed: {e}")
            chroma_docs = []
        
        # 2. Query keyword search (BM25)
        if bm25_retriever:
            try:
                bm25_docs = bm25_retriever.invoke(query)[:k]
            except Exception as e:
                print(f"[Warning] BM25 retrieval failed: {e}")
                bm25_docs = []
        else:
            bm25_docs = []

        # Filter BM25 docs to ensure they actually contain at least one of the query keywords (avoiding LangChain's 0-score fallback)
        filtered_query_tokens = preprocess_func(query)
        valid_bm25_docs = []
        if filtered_query_tokens:
            for doc in bm25_docs:
                doc_text_lower = doc.page_content.lower()
                if any(token in doc_text_lower for token in filtered_query_tokens):
                    valid_bm25_docs.append(doc)

        # 3. Match and select (overlapping/matching chunks get prioritized as best)
        matching_docs = []
        unique_chroma_docs = []
        unique_bm25_docs = []
        
        # Build set of normalized page content for intersection lookup
        bm25_contents = {doc.page_content.strip() for doc in valid_bm25_docs}
        
        # Find matching docs (in both) and unique Chroma docs
        seen_matching = set()
        for doc in chroma_docs:
            content = doc.page_content.strip()
            if content in bm25_contents:
                matching_docs.append(doc)
                seen_matching.add(content)
            else:
                unique_chroma_docs.append(doc)
                
        # Find unique BM25 docs
        for doc in valid_bm25_docs:
            content = doc.page_content.strip()
            if content not in seen_matching:
                unique_bm25_docs.append(doc)
                
        # Combine: matching first, then unique semantic/synonym, then unique keyword
        final_docs = matching_docs + unique_chroma_docs + unique_bm25_docs
        
        # Cap to top-5 chunks
        docs = final_docs[:k]
        
        # Build score lookup map from Chroma results
        content_scores = {}
        for doc, score in chroma_docs_with_score:
            content_scores[doc.page_content.strip()] = score

        formatted_chunks = []
        for doc in docs:
            content = doc.page_content.strip()
            score = content_scores.get(content, None)
            
            # Parse existing Title/Section from content if present
            lines = content.split("\n")
            if len(lines) >= 2 and lines[0].startswith("Title: ") and lines[1].startswith("Section: "):
                title = lines[0].replace("Title: ", "").strip()
                section = lines[1].replace("Section: ", "").strip()
                text = "\n".join(lines[2:]).strip()
            else:
                title = doc.metadata.get("title", "Unknown Source")
                section = doc.metadata.get("section_heading", "Unknown Section")
                text = content
                
            score_str = f"{score:.4f}" if score is not None else "Keyword Match"
            formatted_chunks.append(f"Title: {title}\nSection: {section}\nScore: {score_str}\n\n{text}")
            
        if not formatted_chunks:
            return "No relevant information found in the knowledge base for this query."
            
        return "\n\n---\n\n".join(formatted_chunks)

    return rag_search
