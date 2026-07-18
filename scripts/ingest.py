"""
scripts/ingest.py — Standalone data ingestion pipeline.

Completely independent of the agent runtime. Run this once (or when data changes)
to embed a data source into the vector store. The agent (main.py) then reads
from the persisted store at startup.

Usage:
    python scripts/ingest.py --loader corporate_turnaround
    python scripts/ingest.py --loader corporate_turnaround --max-rows 1000   # dev/test sample
    python scripts/ingest.py --loader corporate_turnaround --force           # re-index even if exists
"""
import argparse
import sys
import os

# Allow running from scripts/ directory or project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config import load_config
from data_handlers.registry import get_loader
from rag.embeddings.google import GoogleEmbeddingProvider
from rag.embeddings.huggingface import HuggingFaceEmbeddingProvider
from rag.vector_store.chroma import ChromaBackend
from rag.guardrails.pii_detector import PIIGuardrail

# ── Provider registries (mirrors rag/retriever.py) ────────────────────────────
EMBEDDING_PROVIDERS = {
    "google": GoogleEmbeddingProvider,
    "huggingface": HuggingFaceEmbeddingProvider,
}

VECTOR_STORE_BACKENDS = {
    "chroma": ChromaBackend,
}


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a data source into the RAG vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py --loader corporate_turnaround
  python scripts/ingest.py --loader corporate_turnaround --max-rows 500
  python scripts/ingest.py --loader corporate_turnaround --force
        """,
    )
    parser.add_argument(
        "--loader",
        required=True,
        help="Data loader name (e.g. 'corporate_turnaround'). See data_handlers/registry.py.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index even if the collection already exists.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit the number of rows to ingest (useful for testing).",
    )
    args = parser.parse_args()

    config = load_config()

    print(f"\n{'='*62}")
    print(f"  RAG Ingestion Pipeline")
    print(f"{'='*62}")
    print(f"  Loader:           {args.loader}")
    print(f"  LLM Provider:     {config.llm_provider} / {config.llm_model}")
    print(f"  Embeddings:       {config.embedding_provider} / {config.embedding_model}")
    print(f"  Vector Store:     {config.vector_store_backend} -> {config.vector_store_persist_dir}")
    print(f"  Chunk Size:       {config.chunk_size} (overlap: {config.chunk_overlap})")
    print(f"  PII Enabled:      {config.pii.enabled} (strategy: {config.pii.strategy})")
    if args.max_rows:
        print(f"  Max Rows:         {args.max_rows} (sampling mode)")
    print(f"{'='*62}\n")

    # ── Step 1: Resolve loader ─────────────────────────────────────────────────
    loader_cls = get_loader(args.loader)
    loader_kwargs = {}
    if args.max_rows:
        loader_kwargs["max_rows"] = args.max_rows
    loader = loader_cls(**loader_kwargs)
    collection = loader.collection_name()

    # Update config so the rest of the pipeline uses the correct collection
    config.rag_collection = collection

    # ── Step 2: Resolve vector store backend ──────────────────────────────────
    emb_cls = EMBEDDING_PROVIDERS.get(config.embedding_provider)
    if not emb_cls:
        print(f"[Error] Unknown embedding provider: '{config.embedding_provider}'")
        sys.exit(1)
    embeddings = emb_cls(model_name=config.embedding_model).get_embeddings()

    vs_cls = VECTOR_STORE_BACKENDS.get(config.vector_store_backend)
    if not vs_cls:
        print(f"[Error] Unknown vector store backend: '{config.vector_store_backend}'")
        sys.exit(1)
    vector_store = vs_cls(
        embeddings=embeddings,
        persist_dir=config.vector_store_persist_dir,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    # ── Step 3: Skip if already indexed ───────────────────────────────────────
    if vector_store.collection_exists(collection) and not args.force:
        print(f"[OK] Collection '{collection}' already exists. Skipping ingestion.")
        print(f"  Use --force to re-index.\n")
        return

    # ── Step 4: Load documents ─────────────────────────────────────────────────
    print(f"[Step 1/3] Loading data via {loader_cls.__name__}...")
    documents = loader.load()
    print(f"  -> {len(documents)} documents loaded.\n")

    # ── Step 5: PII sanitization ───────────────────────────────────────────────
    n_redacted = 0
    if config.pii.enabled and config.pii.scrub_on_ingest:
        print(f"[Step 2/3] Running PII guardrail on {len(documents)} documents...")
        guardrail = PIIGuardrail(config.pii)
        documents, n_redacted = guardrail.sanitize_documents(documents)
        print(f"  -> {n_redacted} PII entities redacted.")
        print(f"  -> {len(documents)} documents passed to embedding.\n")
    else:
        print("[Step 2/3] PII guardrail disabled — skipping.\n")

    # ── Step 6: Build vector store ─────────────────────────────────────────────
    print(f"[Step 3/3] Building vector store collection '{collection}'...")
    vector_store.build(documents, collection)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  [OK] Ingestion complete!")
    print(f"  Collection:       {collection}")
    print(f"  Documents indexed:{len(documents)}")
    print(f"  PII entities:     {n_redacted if config.pii.enabled else 'N/A (disabled)'}")
    print(f"  Persist dir:      {config.vector_store_persist_dir}")
    print(f"\n  Next step: python main.py --mode rag")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
