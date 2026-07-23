"""
main.py — Entry point for the plug-and-play RAG agent.

Usage:
    python main.py                  

All agent parameters are driven by config.py (or env var overrides).
No implementation code needs to change between deployments.
"""
import argparse
import asyncio
import os

from dotenv import load_dotenv
load_dotenv()

from config import load_config, AgentConfig


# ── Retrieval Setup ────────────────────────────────────────────────────────────

def _setup_rag_tool(config: AgentConfig) -> None:
    """
    Build the shared RetrievalPipeline for this config, paying its one-time
    cost (BM25 index over the whole corpus) before anyone is waiting on it.
    Warns (but does not crash) if the collection hasn't been indexed yet.

    Named for history: every entrypoint calls it via warmup.build_shared_config.
    """
    from rag.vector_store.chroma import ChromaBackend
    from rag.embeddings.google import GoogleEmbeddingProvider
    from rag.embeddings.huggingface import HuggingFaceEmbeddingProvider

    EMBEDDING_PROVIDERS = {
        "google": GoogleEmbeddingProvider,
        "huggingface": HuggingFaceEmbeddingProvider,
    }
    VECTOR_STORE_BACKENDS = {"chroma": ChromaBackend}

    # Check collection exists before wiring the retriever
    if config.embedding_provider not in EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unknown embedding_provider '{config.embedding_provider}'. "
            f"Available: {list(EMBEDDING_PROVIDERS)}"
        )
    if config.vector_store_backend not in VECTOR_STORE_BACKENDS:
        raise ValueError(
            f"Unknown vector_store_backend '{config.vector_store_backend}'. "
            f"Available: {list(VECTOR_STORE_BACKENDS)}"
        )

    emb_cls = EMBEDDING_PROVIDERS[config.embedding_provider]
    embeddings = emb_cls(model_name=config.embedding_model).get_embeddings()
    vs_cls = VECTOR_STORE_BACKENDS[config.vector_store_backend]
    vector_store = vs_cls(
        embeddings=embeddings,
        persist_dir=config.vector_store_persist_dir,
    )

    if not vector_store.collection_exists(config.rag_collection):
        print(
            f"\n[Warning] Vector store collection '{config.rag_collection}' not found.\n"
            f"  Run ingestion first:\n"
            f"    python scripts/ingest.py --loader enriched\n"
        )

    from rag.pipeline import get_pipeline
    get_pipeline(config)
    print(f"[RAG] Retrieval ready for collection: '{config.rag_collection}'")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Corporate Turnaround Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
  python scripts/ingest.py --loader enriched
  python main.py
        """,
    )
    parser.parse_args()

    # ── Config ─────────────────────────────────────────────────────────────────
    config = load_config()

    # Verify appropriate API key is set
    if config.llm_provider == "gemini":
        if not os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") == "your_api_key_here":
            print("[Error] GEMINI_API_KEY not set. Add it to your .env file.")
            return
    elif config.llm_provider == "groq":
        if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "your_api_key_here":
            print("[Error] GROQ_API_KEY not set. Add it to your .env file.")
            return

    print(f"\n{'='*62}")
    print(f"  Plug-and-Play RAG Agent")
    print(f"{'='*62}")
    print(f"  LLM:        {config.llm_provider} / {config.llm_model}")
    print(f"  PII Guard:  {'enabled (' + config.pii.strategy + ')' if config.pii.enabled else 'disabled'}")
    print(f"{'='*62}\n")

    # ── Retrieval setup ────────────────────────────────────────────────────────
    _setup_rag_tool(config)

    # ── Build the answer engine ────────────────────────────────────────────────
    print("Initializing agent...")
    from core.rag_chat import RagChat, Turn
    chat = RagChat(config)
    history: list[Turn] = []
    print("Agent ready! Type 'exit' or 'quit' to stop.\n" + "-" * 50)

    # ── Interactive loop ───────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("\nYou: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue

            # Checkpoint 2: Query-time PII scrub
            from core.utils import apply_pii_query_guard, apply_pii_response_guard
            user_input = apply_pii_query_guard(user_input, config)

            # Single-pass RAG (core/rag_chat.py). Prefix cleanup and the
            # grounding backstop run inside it, so the CLI, Streamlit and the
            # public API share one implementation of them.
            async def answer_turn() -> str:
                async for kind, payload in chat.stream(history, user_input):
                    if kind == "done":
                        return payload.text
                return ""

            final_message = asyncio.run(answer_turn())

            # Checkpoint 3: Output-time PII scrub
            final_message = apply_pii_response_guard(final_message, config)
            history.append(Turn(user=user_input, assistant=final_message))

            print(f"\nAgent: {final_message}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            # Never surface the raw provider error to the user: it can carry
            # internal details (a 400 from a malformed tool call echoes the
            # tool schema). Log the detail, show a calm, actionable message.
            import sys
            print(f"[Error] Agent invocation failed: {type(e).__name__}: {e}", file=sys.stderr)
            print(
                "\nAgent: Sorry -- something went wrong on our end while looking that up. "
                "Please try again in a moment, or call us at 1-800-889-0232 and we can help you directly."
            )


if __name__ == "__main__":
    main()
