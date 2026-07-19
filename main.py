"""
main.py — Entry point for the plug-and-play RAG agent.

Usage:
    python main.py                  

All agent parameters are driven by config.py (or env var overrides).
No implementation code needs to change between deployments.
"""
import os
import argparse

from dotenv import load_dotenv
load_dotenv()

from config import load_config, AgentConfig
from core.factory import AgentFactory
from core.tools.registry import register_tool


# ── RAG Tool Setup ─────────────────────────────────────────────────────────────

def _setup_rag_tool(config: AgentConfig) -> None:
    """
    Build the RAG retriever tool and register it in the tool registry.
    Warns (but does not crash) if the collection hasn't been indexed yet.
    """
    from rag.retriever import build_rag_tool
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

    rag_tool = build_rag_tool(config)
    register_tool("rag", rag_tool)
    print(f"[RAG] Tool registered for collection: '{config.rag_collection}'")


MODE_PRESETS = {
    "rag": {
        "tools": ["rag"],
    }
}


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
    parser.add_argument(
        "--mode",
        choices=["rag"],
        default="rag",
        help="Agent mode (default: rag)",
    )
    args = parser.parse_args()

    # ── Config ─────────────────────────────────────────────────────────────────
    config = load_config()
    preset = MODE_PRESETS[args.mode]
    config.tools = preset["tools"]

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
    print(f"  Mode:       {args.mode}")
    print(f"  LLM:        {config.llm_provider} / {config.llm_model}")
    print(f"  Tools:      {config.tools}")
    print(f"  PII Guard:  {'enabled (' + config.pii.strategy + ')' if config.pii.enabled else 'disabled'}")
    print(f"{'='*62}\n")

    # ── RAG tool setup (if needed) ─────────────────────────────────────────────
    if "rag" in config.tools:
        _setup_rag_tool(config)

    # ── Build agent ────────────────────────────────────────────────────────────
    print("Initializing agent...")
    agent_executor = AgentFactory.create(config)
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

            from core.utils import build_user_query, wrap_user_query
            session_config = {"configurable": {"thread_id": "session_1"}}
            response = agent_executor.invoke(
                {"messages": [("user", wrap_user_query(build_user_query(user_input)))]},
                config=session_config,
            )

            # Extract final message text
            final_message = response["messages"][-1].content
            if isinstance(final_message, list):
                final_message = "".join(
                    part["text"] if isinstance(part, dict) and "text" in part else str(part)
                    for part in final_message
                )

            # Clean RAG prefix noise
            from core.utils import clean_response_prefix, enforce_grounding_refusal
            final_message = clean_response_prefix(final_message)

            # Deterministic backstop: if rag_search found nothing this turn,
            # don't trust the LLM to have actually refused as instructed.
            final_message = enforce_grounding_refusal(response, final_message)

            # Checkpoint 3: Output-time PII scrub
            final_message = apply_pii_response_guard(final_message, config)

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
