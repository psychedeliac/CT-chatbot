"""
core/tools/registry.py — Named tool registry.

Add new tools here without touching factory.py or main.py.
The agent enables/disables tools purely via config.tools list.
"""
from core.tools.web_search import web_search
from core.tools.url_reader import url_reader

# ── Tool Registry ─────────────────────────────────────────────────────────────
# Maps tool name → LangChain tool callable.
# "rag" starts as None; populated at runtime by build_rag_tool() + register_tool().
TOOL_REGISTRY: dict = {
    "web_search": web_search,
    "url_reader": url_reader,
    "rag": None,              # registered dynamically after vector store is ready
}


def register_tool(name: str, tool) -> None:
    """
    Register (or update) a tool by name.
    Used to inject the RAG tool after the vector store is initialized.
    """
    TOOL_REGISTRY[name] = tool


def get_tools(names: list) -> list:
    """
    Resolve a list of tool names to their callable implementations.

    - Skips tools registered as None (not yet initialized) with a warning.
    - Raises ValueError for completely unknown tool names.
    """
    import warnings

    tools = []
    for name in names:
        if name not in TOOL_REGISTRY:
            raise ValueError(
                f"Unknown tool '{name}'. "
                f"Available tools: {list(TOOL_REGISTRY.keys())}"
            )
        tool = TOOL_REGISTRY[name]
        if tool is None:
            warnings.warn(
                f"Tool '{name}' is registered but not yet initialized. "
                f"Skipping. (Did you forget to run scripts/ingest.py?)",
                stacklevel=2,
            )
            continue
        tools.append(tool)
    return tools
