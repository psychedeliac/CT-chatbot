"""
scripts/diagnose_retrieval.py — Read-only retrieval diagnostics.

Reproduces the exact retrieval logic in rag/retriever.py::build_rag_tool,
but with full visibility into every intermediate stage: raw Chroma scores
(unfiltered), which docs pass the 0.80 distance cutoff, raw BM25 results,
which pass the keyword-overlap filter, and how the final merge decides
what the LLM sees.

Does NOT modify any pipeline code or data. Safe to run repeatedly.

Usage:
    python scripts/diagnose_retrieval.py
    python scripts/diagnose_retrieval.py --query "what is islam"
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config import load_config

DEFAULT_QUERIES = [
    "what is islam",
    "what is mca",
    "what is unsecured debt",
    "how do I settle my business debt",
    "what is the capital of france",
    "tell me a joke",
]

STOP_WORDS = {
    "what", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "has", "have", "had",
    "who", "whom", "whose", "which", "where", "when", "why", "how",
    "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "a", "an", "the", "in", "on", "for", "with", "about", "of", "to", "and", "or", "but", "because", "as", "until", "while",
    "at", "by", "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", "up", "down",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "corporate", "turnaround", "company", "business", "businesses", "program", "service", "services", "client", "clients"
}


def preprocess_func(text: str) -> list[str]:
    tokens = re.findall(r'\b\w+\b', text.lower())
    filtered = [t for t in tokens if t not in STOP_WORDS]
    return filtered if filtered else tokens


def short(text: str, n: int = 90) -> str:
    text = text.replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")


def main():
    parser = argparse.ArgumentParser(description="Diagnose the RAG retrieval pipeline.")
    parser.add_argument("--query", action="append", help="Query to test (repeatable). Defaults to a built-in set.")
    args = parser.parse_args()

    queries = args.query if args.query else DEFAULT_QUERIES

    config = load_config()

    print(f"\n{'='*70}")
    print("  Retrieval Diagnostics")
    print(f"{'='*70}")
    print(f"  Collection:       {config.rag_collection}")
    print(f"  Embedding:        {config.embedding_provider} / {config.embedding_model}")
    print(f"  Vector store dir: {config.vector_store_persist_dir}")
    print(f"  rag_k:            {config.rag_k}")
    print(f"{'='*70}\n")

    # ── Collection metadata (distance metric check) ────────────────────────────
    import chromadb
    client = chromadb.PersistentClient(path=config.vector_store_persist_dir)
    try:
        collection = client.get_collection(config.rag_collection)
        print(f"[Collection Metadata] {collection.metadata}")
        space = (collection.metadata or {}).get("hnsw:space", "l2 (chroma default — not explicitly set)")
        print(f"[Distance Metric]     {space}")
        print(f"[Document Count]      {collection.count()}")
    except Exception as e:
        print(f"[Error] Could not inspect collection: {e}")
        return
    print()

    # ── Build embeddings + Chroma store (mirrors rag/retriever.py) ─────────────
    from rag.embeddings.google import GoogleEmbeddingProvider
    from rag.embeddings.huggingface import HuggingFaceEmbeddingProvider

    EMBEDDING_PROVIDERS = {
        "google": GoogleEmbeddingProvider,
        "huggingface": HuggingFaceEmbeddingProvider,
    }
    emb_cls = EMBEDDING_PROVIDERS.get(config.embedding_provider)
    embeddings = emb_cls(model_name=config.embedding_model).get_embeddings()

    from langchain_chroma import Chroma
    chroma_store = Chroma(
        collection_name=config.rag_collection,
        embedding_function=embeddings,
        persist_directory=config.vector_store_persist_dir,
    )

    # ── Build BM25 retriever (mirrors rag/retriever.py) ─────────────────────────
    from data_handlers.enriched_loader import EnrichedLoader
    from langchain_community.retrievers import BM25Retriever

    bm25_retriever = None
    try:
        loader = EnrichedLoader()
        documents = loader.load()
        print(f"[BM25 Source] {len(documents)} documents loaded from EnrichedLoader\n")

        # ── Data quality scan: flag empty/placeholder docs feeding the index ────
        placeholder_docs = [
            d for d in documents
            if "Unknown" in d.page_content and d.page_content.count("Unknown") >= 2
        ]
        empty_docs = [d for d in documents if len(d.page_content.strip()) < 40]
        print(f"[Data Quality] {len(placeholder_docs)} docs contain >=2 'Unknown' fields (likely placeholder/garbage)")
        print(f"[Data Quality] {len(empty_docs)} docs have <40 chars of content (likely empty)")
        for d in placeholder_docs[:5]:
            print(f"    -> {short(d.page_content, 120)}")
        print()

        if documents:
            bm25_retriever = BM25Retriever.from_documents(documents, preprocess_func=preprocess_func)
            bm25_retriever.k = config.rag_k
    except Exception as e:
        print(f"[Warning] Failed to initialize BM25 retriever: {e}\n")

    # ── Run each query through the full pipeline with full visibility ──────────
    k = config.rag_k or 5

    for query in queries:
        print(f"\n{'#'*70}")
        print(f"  QUERY: {query!r}")
        print(f"{'#'*70}")

        # 1. Raw Chroma results (UNFILTERED)
        chroma_docs_with_score = chroma_store.similarity_search_with_score(query, k=k)
        print(f"\n[Chroma] Raw results (unfiltered), k={k}:")
        for i, (doc, score) in enumerate(chroma_docs_with_score):
            passed = "PASS" if score <= 0.80 else "FAIL"
            print(f"  #{i+1} [{passed}] score={score:.4f}  {short(doc.page_content)}")

        chroma_docs = [doc for doc, score in chroma_docs_with_score if score <= 0.80]
        print(f"  -> {len(chroma_docs)}/{len(chroma_docs_with_score)} passed the <=0.80 filter")

        # 2. Raw BM25 results
        filtered_query_tokens = preprocess_func(query)
        print(f"\n[BM25] Filtered query tokens: {filtered_query_tokens}")
        if bm25_retriever:
            bm25_docs = bm25_retriever.invoke(query)[:k]
            print(f"[BM25] Raw results, k={k}:")
            valid_bm25_docs = []
            for i, doc in enumerate(bm25_docs):
                doc_text_lower = doc.page_content.lower()
                has_overlap = any(t in doc_text_lower for t in filtered_query_tokens) if filtered_query_tokens else False
                if has_overlap:
                    valid_bm25_docs.append(doc)
                tag = "PASS" if has_overlap else "FAIL(no token overlap)"
                print(f"  #{i+1} [{tag}] {short(doc.page_content)}")
            print(f"  -> {len(valid_bm25_docs)}/{len(bm25_docs)} passed the token-overlap filter")
        else:
            valid_bm25_docs = []
            print("[BM25] retriever not initialized")

        # 3. Final merge (mirrors rag/retriever.py logic exactly)
        bm25_contents = {doc.page_content.strip() for doc in valid_bm25_docs}
        matching_docs, unique_chroma_docs, unique_bm25_docs = [], [], []
        seen_matching = set()
        for doc in chroma_docs:
            content = doc.page_content.strip()
            if content in bm25_contents:
                matching_docs.append(doc)
                seen_matching.add(content)
            else:
                unique_chroma_docs.append(doc)
        for doc in valid_bm25_docs:
            content = doc.page_content.strip()
            if content not in seen_matching:
                unique_bm25_docs.append(doc)

        final_docs = matching_docs + unique_chroma_docs + unique_bm25_docs
        docs = final_docs[:k]

        content_scores = {doc.page_content.strip(): score for doc, score in chroma_docs_with_score}

        print(f"\n[Final Merge] {len(matching_docs)} matched-both, {len(unique_chroma_docs)} chroma-only, "
              f"{len(unique_bm25_docs)} bm25-only -> returning top {len(docs)}:")
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            score = content_scores.get(content)
            if doc in matching_docs:
                source = "MATCHED(both)"
            elif doc in unique_chroma_docs:
                source = "CHROMA-only"
            else:
                source = "BM25-only"
            score_str = f"{score:.4f}" if score is not None else "Keyword Match"
            # Flag the inconsistency: a displayed score above the cutoff means
            # it did NOT come through the chroma_docs filter path.
            flag = ""
            if score is not None and score > 0.80 and source != "BM25-only":
                flag = "  <-- ANOMALY: score exceeds 0.80 filter but not sourced from BM25"
            print(f"  #{i+1} [{source}] score={score_str}  {short(content)}{flag}")

    print(f"\n{'='*70}")
    print("  Done. Re-run after any pipeline change to compare before/after.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
