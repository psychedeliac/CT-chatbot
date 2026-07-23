import warnings
import sys
import os

# Allow running from scripts/ directory or project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Running import smoke test...\n")

try:
    from config import load_config, PIIConfig
    from core.llms import GeminiProvider, LLMProvider
    from core.factory import build_llm
    from core.rag_chat import RagChat, build_retrieval_query
    from rag.embeddings.base import EmbeddingProvider
    from rag.embeddings.google import GoogleEmbeddingProvider
    from rag.vector_store.base import VectorStoreBackend
    from rag.vector_store.chroma import ChromaBackend
    from rag.pipeline import RetrievalPipeline, get_pipeline
    from data_handlers.base_loader import BaseDataLoader
    from data_handlers.ct_loader import CorporateTurnaroundLoader
    from data_handlers.registry import get_loader
    print("OK  All module imports successful")
except ImportError as e:
    print(f"FAIL  Import error: {e}")
    sys.exit(1)

# Test 1: Config instantiation
try:
    cfg = load_config()
    assert cfg.llm_provider in ["gemini", "groq"]
    assert isinstance(cfg.pii.enabled, bool)
    print(f"OK  Config: llm={cfg.llm_provider}/{cfg.llm_model}, pii_enabled={cfg.pii.enabled}")
except Exception as e:
    print(f"FAIL  Config: {e}")
    sys.exit(1)

# Test 2: PII guardrail (regex fallback, no spacy model needed)
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from rag.guardrails.pii_detector import PIIGuardrail
        pii_cfg = PIIConfig(detect_person_names=False)  # disable NLP-heavy entity
        g = PIIGuardrail(pii_cfg)
        clean, n = g._detect_and_anonymize("Contact us at test@example.com or call +1-800-555-1234")
        assert n >= 1, f"Expected at least 1 PII entity, got {n}"
        assert "test@example.com" not in clean
        print(f"OK  PIIGuardrail: {n} entity/entities redacted -> '{clean[:80]}'")
except Exception as e:
    print(f"FAIL  PIIGuardrail: {e}")
    sys.exit(1)

# Test 3: Loader registry
try:
    loader_cls = get_loader("corporate_turnaround")
    assert loader_cls.__name__ == "CorporateTurnaroundLoader"
    loader = loader_cls()
    assert loader.collection_name() == "corporate_turnaround"
    print(f"OK  Loader registry: 'corporate_turnaround' -> {loader_cls.__name__} (collection: {loader.collection_name()})")
except Exception as e:
    print(f"FAIL  Loader registry: {e}")
    sys.exit(1)

# Test 4: Follow-up queries are standalone (no router LLM call does it now)
try:
    from core.rag_chat import Turn
    query = build_retrieval_query(
        [Turn(user="Do you help with merchant cash advances?", assistant="Yes...")],
        "how do I get out of it?",
    )
    assert "merchant cash advances" in query, query
    print(f"OK  Retrieval query rewriting: follow-up resolved to '{query[:60]}...'")
except Exception as e:
    print(f"FAIL  Retrieval query rewriting: {e}")
    sys.exit(1)

# Test 5: LLM provider hierarchy
try:
    assert issubclass(GeminiProvider, LLMProvider)
    print(f"OK  LLM hierarchy: GeminiProvider implements LLMProvider ABC")
except Exception as e:
    print(f"FAIL  LLM hierarchy: {e}")
    sys.exit(1)

print("\nAll checks passed. RAG pipeline is ready.")
