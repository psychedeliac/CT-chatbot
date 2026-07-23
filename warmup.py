"""
warmup.py — build the retrieval machinery once, before anyone is waiting on it.

`streamlit run app.py` only executes app.py when a browser connects, so the ~40s
BM25 build + cross-encoder load was paid by the first visitor. Running
`python warmup.py` instead does that work in the server process at boot, then
hands off to Streamlit's bootstrap. The lru_cache below is process-global, so
the app.py session that follows gets the already-built objects for free and
keeps them for the life of the process.
"""
import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def build_shared_config():
    """
    Build config and the shared RetrievalPipeline ONCE per server process.

    Retrieval is the expensive part (BM25 index + cross-encoder) and holds no
    per-conversation state, so every entrypoint shares one copy. Conversation
    history lives with whoever owns the session (api/chat.py, app.py).
    """
    from config import load_config
    from main import _setup_rag_tool

    config = load_config()
    _setup_rag_tool(config)

    # Warm the cross-encoder. RetrievalPipeline loads it lazily on first rerank
    # (rag/pipeline.py:_get_cross_encoder), ~14s. One throwaway query pays it here.
    from rag.pipeline import get_pipeline
    try:
        get_pipeline(config).retrieve("warmup")
    except Exception as exc:
        # A failed warmup must not take the app down -- the model still loads
        # lazily on the first real query, just slowly.
        print(f"[Warning] Retrieval warmup failed (first query will be slow): {exc}")

    # Warm the PII engines too: the first sanitize_query() otherwise loads
    # spaCy en_core_web_lg (~2.5s) inside a user's turn.
    if config.pii.enabled:
        from core.utils import apply_pii_query_guard
        try:
            apply_pii_query_guard("warmup", config)
        except Exception as exc:
            print(f"[Warning] PII warmup failed (first query will be slow): {exc}")

    return config


if __name__ == "__main__":
    from streamlit.web import bootstrap

    flag_options = {}
    bootstrap.load_config_options(flag_options)
    # flush=True: stdout is block-buffered when the server runs under a pipe,
    # which otherwise hides these until the process exits.
    print("Warming retrieval (BM25 index + cross-encoder)...", flush=True)
    build_shared_config()
    print("Warm. Starting Streamlit...", flush=True)
    bootstrap.run(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"),
                  False, [], flag_options)
