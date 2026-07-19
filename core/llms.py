"""
core/llms.py — LLM provider abstraction.

Defines the LLMProvider ABC and the concrete GeminiProvider.
To add a new LLM (e.g. OpenAI), create a class that inherits LLMProvider
and implement get_llm(). Register it in the LLM_PROVIDERS dict in factory.py.
"""
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


def _gemini_api_keys() -> list[str]:
    """All configured Gemini keys, in order: GEMINI_API_KEY, GEMINI_API_KEY_1, _2, ...

    Multiple keys let the app rotate past a key that has hit its daily
    free-tier quota (20 requests/day) instead of dying on a 429.
    """
    keys = []
    primary = os.environ.get("GEMINI_API_KEY")
    if primary and primary != "your_api_key_here":
        keys.append(primary)
    i = 1
    while True:
        k = os.environ.get(f"GEMINI_API_KEY_{i}")
        if not k:
            break
        if k != "your_api_key_here":
            keys.append(k)
        i += 1
    return keys


def _is_failover_error(exc: Exception) -> bool:
    """True for a per-key error worth retrying on the NEXT key rather than
    surfacing: a quota/rate limit (429 / RESOURCE_EXHAUSTED) or a key whose
    project is disabled/denied (403 / PERMISSION_DENIED). If every key hits
    one of these, the last error still propagates."""
    s = str(exc)
    return (
        "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower()
        or "403" in s or "PERMISSION_DENIED" in s
    )


class RotatingGeminiLLM(BaseChatModel):
    """ChatGoogleGenerativeAI wrapper that fails over across multiple API keys
    on a quota (429) error. On each call it tries keys round-robin starting
    from the last known-good one; a key that 429s is skipped to the next.

    Works transparently with LangGraph: it's a BaseChatModel, so create_react_agent
    binds tools to it via bind_tools() and reuses the bound instance.
    """

    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    api_keys: list[str] = []

    _bound_tools: Optional[Sequence] = PrivateAttr(default=None)
    _bound_kwargs: dict = PrivateAttr(default_factory=dict)
    _next: int = PrivateAttr(default=0)
    _clients: dict = PrivateAttr(default_factory=dict)

    def _client(self, key: str):
        if key not in self._clients:
            from langchain_google_genai import ChatGoogleGenerativeAI
            client = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=key,
            )
            if self._bound_tools is not None:
                client = client.bind_tools(self._bound_tools, **self._bound_kwargs)
            self._clients[key] = client
        return self._clients[key]

    def bind_tools(self, tools: Sequence, **kwargs: Any) -> "RotatingGeminiLLM":
        clone = RotatingGeminiLLM(
            model_name=self.model_name,
            temperature=self.temperature,
            api_keys=self.api_keys,
        )
        clone._bound_tools = tools
        clone._bound_kwargs = kwargs
        return clone

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        if not self.api_keys:
            raise ValueError("No Gemini API keys configured (set GEMINI_API_KEY).")
        n = len(self.api_keys)
        last_exc: Exception = RuntimeError("no Gemini API keys available")
        for offset in range(n):
            idx = (self._next + offset) % n
            try:
                msg = self._client(self.api_keys[idx]).invoke(messages, stop=stop, **kwargs)
                self._next = (idx + 1) % n
                return ChatResult(generations=[ChatGeneration(message=msg)])
            except Exception as exc:
                if not _is_failover_error(exc):
                    raise
                last_exc = exc  # quota/denied key: try the next key
        raise last_exc  # every key exhausted

    @property
    def _llm_type(self) -> str:
        return "rotating-gemini"


# ── Abstract Base ──────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Abstract LLM provider. All concrete providers must implement get_llm().
    The agent layer consumes only this interface — it never knows which
    specific model is underneath.
    """

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """Return an initialized, ready-to-use LangChain chat model."""
        ...


# ── Concrete Providers ─────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    """Google Gemini via langchain-google-genai. Requires GEMINI_API_KEY."""

    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    def get_llm(self) -> BaseChatModel:
        keys = _gemini_api_keys()
        if not keys:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file: GEMINI_API_KEY=your_key_here"
            )
        return RotatingGeminiLLM(
            model_name=self.model_name,
            temperature=self.temperature,
            api_keys=keys,
        )


class GroqProvider(LLMProvider):
    """Groq API wrapper via langchain_groq. Requires GROQ_API_KEY."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    def get_llm(self) -> BaseChatModel:
        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file: GROQ_API_KEY=your_key_here"
            )
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
        )



# Future providers — implement LLMProvider and add to factory.LLM_PROVIDERS:
#
# class OpenAIProvider(LLMProvider):
#     def get_llm(self):
#         from langchain_openai import ChatOpenAI
#         return ChatOpenAI(model=self.model_name, temperature=self.temperature)
#
# class OllamaProvider(LLMProvider):
#     def get_llm(self):
#         from langchain_community.chat_models import ChatOllama
#         return ChatOllama(model=self.model_name)


# ── Backward-compatible shim ───────────────────────────────────────────────────

def get_llm(model_name: str = "gemini-2.5-flash", temperature: float = 0) -> BaseChatModel:
    """
    Backward-compatible helper. Prefer AgentFactory.create(config) for new code.
    """
    return GeminiProvider(model_name=model_name, temperature=temperature).get_llm()
