"""
core/factory.py — composition root for LLM provider selection.

build_llm(config) is the ONLY place a provider is resolved. Everything else
talks to the BaseChatModel abstraction.

The LangGraph ReAct agent that used to live here is gone: every entrypoint
now answers through core/rag_chat.py (retrieve, then one LLM call). The agent
spent a whole LLM round-trip per turn deciding to run the search the system
prompt mandates anyway.
"""
from config import AgentConfig
from core.llms import GeminiProvider, GroqProvider, LLMProvider


# ── LLM Provider Registry ──────────────────────────────────────────────────────
# Add new providers here (no changes elsewhere needed).
LLM_PROVIDERS: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    # "openai": OpenAIProvider,   # uncomment when implemented
    # "ollama": OllamaProvider,   # uncomment when implemented
}


def build_llm(config: AgentConfig):
    """Resolve config.llm_provider to a ready chat model."""
    provider_cls = LLM_PROVIDERS.get(config.llm_provider)
    if not provider_cls:
        raise ValueError(
            f"Unknown LLM provider: '{config.llm_provider}'. "
            f"Available: {list(LLM_PROVIDERS.keys())}"
        )
    kwargs = {"model_name": config.llm_model, "temperature": config.llm_temperature}
    # Only Gemini takes a cap today; passing it blindly would break Groq.
    if provider_cls is GeminiProvider:
        kwargs["max_output_tokens"] = config.llm_max_output_tokens
    return provider_cls(**kwargs).get_llm()
