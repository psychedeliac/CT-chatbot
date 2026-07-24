"""
core/llms.py — LLM provider abstraction.

Defines the LLMProvider ABC and the concrete GeminiProvider.
To add a new LLM (e.g. OpenAI), create a class that inherits LLMProvider
and implement get_llm(). Register it in the LLM_PROVIDERS dict in factory.py.
"""
import itertools
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
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


def _first_chunk_and_rest(stream):
    """
    Pull the first chunk eagerly, then hand back an iterator replaying it
    followed by the remainder.

    A generator is lazy: calling client.stream() does no work and raises
    nothing, so a quota error would surface only later, outside the failover
    loop. Forcing the first chunk makes the 429 happen while another key can
    still be tried. next(..., None) rather than bare next() because a
    StopIteration crossing a generator frame becomes a RuntimeError (PEP 479).
    """
    iterator = iter(stream)
    first = next(iterator, None)
    if first is None:
        return iter(())
    return itertools.chain([first], iterator)


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
    max_output_tokens: Optional[int] = None

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
                max_output_tokens=self.max_output_tokens,
                # The SDK defaults to 6 retries with backoff. On a 429 that
                # means sitting on an exhausted key for tens of seconds before
                # this class even gets the chance to rotate -- a measured 47s
                # for one answer. Rotating to the next key is both the faster
                # and the more likely fix, so retrying here is pure latency.
                # Transient non-quota errors now surface immediately; the API
                # layer turns those into a retryable message for the client.
                max_retries=0,
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
            max_output_tokens=self.max_output_tokens,
        )
        clone._bound_tools = tools
        clone._bound_kwargs = kwargs
        return clone

    def _try_each_key(self, call):
        """Run `call(client)` against each key in turn, rotating past quota/denied
        keys. Shared by _generate and _stream so both fail over identically.

        A non-failover error (a transient network blip, a Google-side 500 --
        not a quota/permission issue) gets ONE immediate retry on the same key
        before surfacing. max_retries=0 on the client deliberately skips the
        SDK's 6-retry backoff (tens of seconds on a 429), but that same setting
        was also eating one-off transient errors with zero retries at all. A
        single retry with no backoff catches the blip without reintroducing
        the latency the SDK's own retries were removed for.
        """
        if not self.api_keys:
            raise ValueError("No Gemini API keys configured (set GEMINI_API_KEY).")
        n = len(self.api_keys)
        last_exc: Exception = RuntimeError("no Gemini API keys available")
        for offset in range(n):
            idx = (self._next + offset) % n
            client = self._client(self.api_keys[idx])
            for attempt in range(2):
                try:
                    result = call(client)
                    self._next = (idx + 1) % n
                    return result
                except Exception as exc:
                    if _is_failover_error(exc):
                        last_exc = exc  # quota/denied key: try the next key
                        break
                    if attempt == 0:
                        continue  # one fast, no-backoff retry on the same key
                    raise  # retried once and still failing: not a key problem
        raise last_exc  # every key exhausted

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self._try_each_key(lambda client: client.invoke(messages, stop=stop, **kwargs))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        """
        Real token-by-token streaming.

        Without this, BaseChatModel's default streaming emits the entire answer
        as a single chunk once generation has finished -- so a UI that "streams"
        still shows nothing until the last token, and time-to-first-token equals
        full response time (seconds). Async callers get this for free:
        langchain_core's default _astream runs _stream in an executor.

        Failover happens on the FIRST chunk only. Once tokens have been handed
        to the caller, switching keys mid-answer would replay the reply from the
        start, so a mid-stream failure propagates instead.
        """
        stream = self._try_each_key(
            lambda client: _first_chunk_and_rest(client.stream(messages, stop=stop, **kwargs))
        )
        for message_chunk in stream:
            # The _stream contract is ChatGenerationChunk, not the raw
            # AIMessageChunk the underlying client yields. Passing the message
            # straight through blows up downstream in _generate_with_cache.
            chunk = ChatGenerationChunk(message=message_chunk)
            if run_manager and chunk.text:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

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

    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0,
                 max_output_tokens: Optional[int] = None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

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
            max_output_tokens=self.max_output_tokens,
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
