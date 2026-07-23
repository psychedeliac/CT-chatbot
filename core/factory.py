"""
core/factory.py — Composition root for the plug-and-play agent.

AgentFactory.create(config) is the ONLY place where provider selection happens.
Everything else talks to abstractions.
"""
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from config import AgentConfig
from core.llms import GeminiProvider, GroqProvider, LLMProvider
from core.tools.registry import get_tools


# ── LLM Provider Registry ──────────────────────────────────────────────────────
# Add new providers here (no changes elsewhere needed).
LLM_PROVIDERS: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    # "openai": OpenAIProvider,   # uncomment when implemented
    # "ollama": OllamaProvider,   # uncomment when implemented
}


def build_llm(config: AgentConfig):
    """Resolve config.llm_provider to a ready chat model. The single place
    provider selection happens -- the ReAct agent and the single-pass path in
    core/rag_chat.py both come through here."""
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


class AgentFactory:

    @staticmethod
    def create(config: AgentConfig):
        """
        Build a fully configured LangGraph ReAct agent from AgentConfig.

        Steps:
          1. Resolve the LLM provider from config.llm_provider
          2. Assemble tools from the named tool registry
          3. Wire into a LangGraph agent with in-memory conversation state
        """
        # 1. Resolve LLM
        llm = build_llm(config)

        # 2. Assemble tools
        tools = get_tools(config.tools)
        if not tools:
            raise ValueError(
                "No tools could be resolved. "
                "Check config.tools and ensure all required tools are initialized."
            )

        # 3. Build LangGraph agent
        memory = MemorySaver()
        agent_executor = create_react_agent(
            llm,
            tools=tools,
            prompt=config.tool_retrieval_contract + config.system_prompt,
            checkpointer=memory,
        )
        return agent_executor
