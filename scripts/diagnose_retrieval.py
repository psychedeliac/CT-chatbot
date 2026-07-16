"""
scripts/diagnose_retrieval.py — Read-only retrieval diagnostics.

Calls into RetrievalPipeline (rag/pipeline.py) — the same code path the live
agent uses — so this script can never drift from production behavior. Shows
every intermediate stage: raw Chroma scores, raw/filtered BM25 results, the
RRF-fused pool, cross-encoder rerank scores, and the final gated output.

Does NOT modify any pipeline code or data. Safe to run repeatedly.

Usage:
    python scripts/diagnose_retrieval.py
    python scripts/diagnose_retrieval.py --query "what is islam"
"""
import argparse
import os
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
    "who is the current us president",
]


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
    print(f"  rag_k:            {config.rag_k}  (candidate pool: {config.rag_candidate_pool_k})")
    print(f"  Reranker:         {'enabled (' + config.rerank_model + ')' if config.rerank_enabled else 'disabled'}")
    print(f"  Gate threshold:   {config.rerank_score_threshold}")
    print(f"{'='*70}\n")

    # ── Collection metadata (distance metric check) ────────────────────────────
    import chromadb
    client = chromadb.PersistentClient(path=config.vector_store_persist_dir)
    try:
        collection = client.get_collection(config.rag_collection)
        space = (collection.metadata or {}).get("hnsw:space", "l2 (chroma default — not explicitly set)")
        print(f"[Distance Metric]     {space}")
        print(f"[Document Count]      {collection.count()}")
    except Exception as e:
        print(f"[Error] Could not inspect collection: {e}")
        return
    print()

    # ── Build the pipeline (single source of truth for retrieval logic) ────────
    from rag.pipeline import RetrievalPipeline
    pipeline = RetrievalPipeline(config)

    # ── Data quality scan (not retrieval logic — a separate diagnostic) ────────
    from data_handlers.enriched_loader import EnrichedLoader
    documents = EnrichedLoader().load()
    placeholder_docs = [
        d for d in documents
        if "Unknown" in d.page_content and d.page_content.count("Unknown") >= 2
    ]
    empty_docs = [d for d in documents if len(d.page_content.strip()) < 40]
    print(f"[Data Quality] {len(placeholder_docs)} docs contain >=2 'Unknown' fields (likely placeholder/garbage)")
    print(f"[Data Quality] {len(empty_docs)} docs have <40 chars of content (likely empty)")

    # ── Run each query through the full pipeline with full visibility ──────────
    for query in queries:
        print(f"\n{'#'*70}")
        print(f"  QUERY: {query!r}")
        print(f"{'#'*70}")

        trace = pipeline.retrieve_with_trace(query)

        print(f"\n[Chroma] Raw results (unfiltered), pool_k={config.rag_candidate_pool_k}:")
        for i, (doc, score) in enumerate(trace.chroma_raw):
            print(f"  #{i+1} score={score:.4f}  {short(doc.page_content)}")

        print(f"\n[BM25] Raw: {len(trace.bm25_raw)}  ->  token-overlap filtered: {len(trace.bm25_filtered)}")
        for i, doc in enumerate(trace.bm25_filtered):
            print(f"  #{i+1} {short(doc.page_content)}")

        print(f"\n[RRF Fusion] {len(trace.rrf_fused)} candidates after fusing Chroma + BM25:")
        for i, doc in enumerate(trace.rrf_fused[:10]):
            print(f"  #{i+1} {short(doc.page_content)}")
        if len(trace.rrf_fused) > 10:
            print(f"  ... and {len(trace.rrf_fused) - 10} more")

        print(f"\n[Rerank + Gate] threshold={config.rerank_score_threshold}:")
        for c in trace.reranked[:10]:
            score_str = f"{c.rerank_score:.4f}" if c.rerank_score is not None else "N/A"
            tag = "PASS" if c.passed_gate else "FAIL"
            print(f"  rrf_rank={c.rrf_rank:<3} [{tag}] score={score_str:<10} {short(c.document.page_content)}")

        print(f"\n[Final] {len(trace.final)} chunk(s) returned to the LLM"
              + (" -> triggers 'No relevant information found'" if not trace.final else ""))

    print(f"\n{'='*70}")
    print("  Done. Re-run after any pipeline change to compare before/after.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
