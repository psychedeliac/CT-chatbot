"""
core/llms.py — LLM provider abstraction.

Defines the LLMProvider ABC and the concrete GeminiProvider.
To add a new LLM (e.g. OpenAI), create a class that inherits LLMProvider
and implement get_llm(). Register it in the LLM_PROVIDERS dict in factory.py.
"""
import os
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


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
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file: GEMINI_API_KEY=your_key_here"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
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
