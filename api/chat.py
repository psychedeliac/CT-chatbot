"""
api/chat.py — one shared answer engine serving many concurrent conversations.

The Streamlit app builds a ReAct agent per browser session because each
carries its own MemorySaver. That does not scale: N users means N agents.
Here a SINGLE stateless RagChat is shared and conversations are separated by
session id, with history held in this module. Everything that must be bounded
for a public deployment is bounded here: live sessions, history length,
cached answers, concurrent LLM turns, and per-turn wall time.

The post-processing chain (prefix clean -> grounding backstop -> PII scrub) is
the same one core/utils.py exposes to the CLI and Streamlit. It is not
reimplemented -- a divergence there would mean the public endpoint enforcing
weaker compliance guarantees than the internal UIs.
"""
import asyncio
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass

from cachetools import TTLCache

from config import APIConfig
from core.rag_chat import MAX_HISTORY_TURNS, RagChat, Turn
from core.utils import apply_pii_query_guard, apply_pii_response_guard

logger = logging.getLogger(__name__)

SESSION_ID_BYTES = 24
# Session ids are opaque tokens we issue. Anything else is refused rather than
# adopted: a caller who can name an arbitrary thread_id can inject turns into a
# conversation they do not own.
SESSION_ID_LENGTH = 32


class CapacityError(Exception):
    """Raised when the server is at its concurrency ceiling. Maps to HTTP 503."""


class TurnTimeout(Exception):
    """Raised when a single turn exceeds turn_timeout_seconds. Maps to HTTP 504."""


@dataclass(frozen=True)
class TurnResult:
    session_id: str
    answer: str
    cached: bool


def new_session_id() -> str:
    return secrets.token_urlsafe(SESSION_ID_BYTES)


def is_valid_session_id(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SESSION_ID_LENGTH
        and all(char.isalnum() or char in "-_" for char in value)
    )


def _cache_key(message: str) -> str:
    """Normalized so 'What are your fees?' and 'what are your fees' share an entry.

    Exact-match by design. Embedding-similarity ("semantic") caching is the
    usual next step and was measured here first: with all-MiniLM-L6-v2,
    "Do you settle business debt?" vs "Do you settle personal credit card
    debt?" scores 0.757 while the genuine paraphrase "How much does your
    program cost?" vs "How much do your services cost?" scores only 0.590.
    No threshold separates them, so a semantic cache in this corpus would
    serve the personal-debt answer to a business-debt question -- the one
    distinction this assistant may never blur. Revisit only with an embedding
    model that ranks those pairs correctly.
    """
    normalized = " ".join(message.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class ChatService:
    def __init__(self, agent_config, api_config: APIConfig):
        self._config = agent_config
        self._api = api_config
        self._chat = RagChat(agent_config)
        # session_id -> last-seen monotonic timestamp. Plain dict rather than a
        # TTLCache because expiry has to run a side effect (dropping the
        # conversation's history), and cachetools evicts silently.
        self._sessions: dict[str, float] = {}
        self._history: dict[str, list[Turn]] = {}
        self._answers = TTLCache(
            maxsize=api_config.answer_cache_size,
            ttl=api_config.answer_cache_ttl_seconds,
        )
        self._semaphore = asyncio.Semaphore(api_config.max_concurrent_turns)

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def _prune_sessions(self) -> None:
        """
        Drop expired sessions and their conversation state.

        Without this, _history accumulates every conversation the process has
        ever seen and the container is eventually OOM-killed.

        ponytail: O(n) scan per turn over at most max_sessions entries (5k by
        default) -- microseconds, and it keeps eviction and thread deletion in
        one obvious place. Switch to a heap if max_sessions grows past ~100k.
        """
        cutoff = time.monotonic() - self._api.session_ttl_seconds
        expired = [sid for sid, seen in self._sessions.items() if seen < cutoff]

        # Hard ceiling as well as TTL: a traffic spike can create sessions
        # faster than they expire.
        overflow = len(self._sessions) - len(expired) - self._api.max_sessions
        if overflow > 0:
            oldest = sorted(self._sessions.items(), key=lambda item: item[1])
            expired.extend(
                sid for sid, _ in oldest[:overflow] if sid not in set(expired)
            )

        for session_id in expired:
            self._sessions.pop(session_id, None)
            self._history.pop(session_id, None)

    def _resolve_session(self, session_id: str | None) -> tuple[str, bool]:
        """Returns (session_id, is_new). Unknown or expired ids get a fresh one."""
        is_known = bool(
            session_id and is_valid_session_id(session_id) and session_id in self._sessions
        )
        resolved = session_id if (is_known and session_id) else new_session_id()
        self._sessions[resolved] = time.monotonic()
        # Prune AFTER inserting, so the ceiling actually holds at max_sessions
        # rather than max_sessions + 1. The session just touched is the newest,
        # so oldest-first eviction can never drop the one being served.
        self._prune_sessions()
        return resolved, not is_known

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    # ── Turn execution ────────────────────────────────────────────────────────

    async def stream_turn(self, message: str, session_id: str | None):
        """
        Yield SSE-shaped events for one turn:
          {"type": "session", "session_id": ...}   always first
          {"type": "delta",   "text": ...}         provisional tokens
          {"type": "done",    "answer": ...}       authoritative final answer

        Clients MUST render the `done` answer over whatever they accumulated
        from deltas. The guards below can replace an answer wholesale (an
        ungrounded reply becomes a refusal), and streaming raw tokens straight
        to the page would let a non-compliant answer finish typing before the
        backstop swapped it.
        """
        resolved_id, is_new = self._resolve_session(session_id)
        yield {"type": "session", "session_id": resolved_id}

        cached = self._cached_answer(message) if is_new else None
        if cached is not None:
            self._record_cached_turn(resolved_id, message, cached)
            yield {"type": "delta", "text": cached}
            yield {"type": "done", "answer": cached, "cached": True}
            return

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self._api.queue_timeout_seconds
            )
        except (asyncio.TimeoutError, TimeoutError):
            raise CapacityError("No capacity for a new turn")

        try:
            async with asyncio.timeout(self._api.turn_timeout_seconds):
                async for event in self._generate(resolved_id, message, is_new):
                    yield event
        except (asyncio.TimeoutError, TimeoutError):
            raise TurnTimeout(f"Turn exceeded {self._api.turn_timeout_seconds}s")
        finally:
            self._semaphore.release()

    async def _generate(self, session_id: str, message: str, is_new: bool):
        scrubbed = apply_pii_query_guard(message, self._config)
        history = self._history.get(session_id, [])

        answer = ""
        async for kind, payload in self._chat.stream(history, scrubbed):
            # The retrieved context is for QA UIs only -- it must never be
            # streamed to a public caller.
            if kind == "delta":
                yield {"type": "delta", "text": payload}
            elif kind == "done":
                answer = apply_pii_response_guard(payload.text, self._config)

        self._append_turn(session_id, scrubbed, answer)
        if is_new:
            self._store_answer(message, answer)
        yield {"type": "done", "answer": answer, "cached": False}

    def _append_turn(self, session_id: str, message: str, answer: str) -> None:
        """Record the exchange, bounded to the window RagChat replays -- older
        turns are dead weight in memory that would never reach a prompt."""
        turns = self._history.setdefault(session_id, [])
        turns.append(Turn(user=message, assistant=answer))
        del turns[:-MAX_HISTORY_TURNS]

    async def answer(self, message: str, session_id: str | None) -> TurnResult:
        """Non-streaming turn. Drains stream_turn so both paths share one implementation."""
        resolved_id = session_id or ""
        final = ""
        cached = False
        async for event in self.stream_turn(message, session_id):
            if event["type"] == "session":
                resolved_id = event["session_id"]
            elif event["type"] == "done":
                final = str(event["answer"])
                cached = bool(event["cached"])
        return TurnResult(session_id=resolved_id, answer=final, cached=cached)

    # ── Answer cache ──────────────────────────────────────────────────────────
    # Only first turns are cacheable: a follow-up's answer depends on history
    # that is unique to its session, so keying it on the message alone would
    # serve one user's context to another.

    def _cached_answer(self, message: str) -> str | None:
        if not self._api.answer_cache_enabled:
            return None
        return self._answers.get(_cache_key(message))

    def _store_answer(self, message: str, answer: str) -> None:
        if self._api.answer_cache_enabled:
            self._answers[_cache_key(message)] = answer

    def _record_cached_turn(self, session_id: str, message: str, answer: str) -> None:
        """
        Write a cache-served exchange into the conversation history anyway.

        Skipping this is a real bug, not an optimization: the next turn would
        run against an empty history and the model would have no idea what the
        user just asked about.
        """
        self._append_turn(session_id, message, answer)
