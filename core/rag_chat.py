"""
core/rag_chat.py — single-pass RAG: retrieve, then ONE LLM call.

Why this exists alongside the ReAct agent (core/factory.py):

The agent spends a full LLM round-trip deciding to call rag_search, every
single turn, because the system prompt orders it to search every single turn.
Measured here: 0.7-3.5s of the ~2.8s median answer, for a decision whose
outcome is fixed. Collapsing sequential LLM steps is the standard fix
("make fewer requests" -- OpenAI's latency guide); published comparisons put
agentic RAG at 3-5x the latency of single-pass for exactly this reason.

What is deliberately NOT lost by dropping the agent:
  - Retrieval is the same RetrievalPipeline, same RRF fusion, same
    cross-encoder gate, called with the same kind of standalone query.
  - The grounding backstop is the same core/utils.enforce_grounding.
  - The PII checkpoints are unchanged (they live in the caller).
What IS lost: the model can no longer choose to search a second time mid-turn,
or to skip searching. Neither happened in practice -- the prompt mandates one
search per turn -- and the deterministic path is the more auditable one for a
regulated context.

Query rewriting: the agent used its router call to resolve "how do I get out
of it?" against the previous turn. Without that call, a follow-up carries the
previous user message as prefix context for retrieval (hybrid BM25+semantic
absorbs the extra terms). First turns -- the majority of widget traffic --
retrieve on the raw message, which is exactly what scripts/eval_retrieval.py
calibrated the thresholds against.
"""
import asyncio
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import AgentConfig
from core.utils import build_user_query, clean_response_prefix, enforce_grounding, wrap_user_query
from rag.pipeline import get_pipeline
from rag.retriever import NO_RESULTS_MESSAGE

# How many past turns to replay into the prompt. Each turn is ~2 messages of
# a few hundred tokens; the model only needs recent context to resolve
# references, and unbounded history is unbounded prefill latency.
MAX_HISTORY_TURNS = 6


@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str


@dataclass(frozen=True)
class Answer:
    text: str
    grounded: bool


def build_retrieval_query(history: list[Turn], message: str) -> str:
    """
    A standalone query for retrieval.

    First turn: the message as-is. Follow-up: prefixed with the previous user
    message so pronouns and elisions ("what about the fees?", "how do I get
    out of it?") still carry their subject into BM25 and the embedding.
    """
    if not history:
        return message
    return f"{history[-1].user} {message}"


class RagChat:
    """
    Stateless answer engine: history is passed in, never stored here.

    The caller owns sessions (api/chat.py bounds and expires them), which
    keeps this class safe to share across all concurrent conversations.
    """

    def __init__(self, config: AgentConfig):
        self._config = config
        self._pipeline = get_pipeline(config)
        from core.factory import build_llm
        self._llm = build_llm(config)

    def _messages(self, history: list[Turn], message: str, context: str) -> list:
        """
        System prompt first and byte-identical every turn: Gemini's implicit
        caching keys on the longest shared prefix, so anything variable placed
        ahead of it would forfeit the discount and the prefill saving.
        """
        prompt = self._config.inline_retrieval_contract + self._config.system_prompt
        messages: list = [SystemMessage(content=prompt)]
        for turn in history[-MAX_HISTORY_TURNS:]:
            messages.append(HumanMessage(content=turn.user))
            messages.append(AIMessage(content=turn.assistant))
        user_block = wrap_user_query(build_user_query(message))
        messages.append(HumanMessage(content=f"{user_block}\n\n{context}"))
        return messages

    async def retrieve(self, history: list[Turn], message: str) -> tuple[str, bool]:
        """Returns (context_block, grounded). Runs off the event loop -- the
        cross-encoder is CPU-bound and would otherwise stall every other
        concurrent request for the duration."""
        query = build_retrieval_query(history, message)
        chunks = await asyncio.to_thread(self._pipeline.retrieve, query)
        context = self._pipeline.format_for_llm(chunks)
        return (context or NO_RESULTS_MESSAGE), bool(chunks)

    async def stream(self, history: list[Turn], message: str):
        """
        Yields ("delta", text) as tokens arrive, then ("done", Answer).

        Deltas are provisional: enforce_grounding can replace the whole answer,
        so the caller must render the final Answer over whatever it streamed.
        """
        context, grounded = await self.retrieve(history, message)
        messages = self._messages(history, message, context)

        parts: list[str] = []
        async for chunk in self._llm.astream(messages):
            text = _flatten(chunk.content)
            if not text:
                continue
            parts.append(text)
            yield "delta", text

        answer = enforce_grounding(grounded, clean_response_prefix("".join(parts)))
        yield "done", Answer(text=answer, grounded=grounded)


def _flatten(content) -> str:
    """LLM content is either a string or a list of content parts."""
    if isinstance(content, list):
        return "".join(
            part["text"] if isinstance(part, dict) and "text" in part else str(part)
            for part in content
        )
    return content or ""
