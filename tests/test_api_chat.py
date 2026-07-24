"""
Checks for the bounded-resource logic in api/chat.py -- session expiry, the
size ceiling, conversation history, and the first-turn-only answer cache.
These are the parts that decide whether the process survives a day of public
traffic, and none of them are exercised by the retrieval or guardrail tests.

No LLM and no vector store: the answer engine is faked, so this runs in
milliseconds and costs no API quota.
"""
import asyncio
import time

import pytest

from api import chat as chat_module
from api.chat import ChatService, is_valid_session_id, new_session_id
from config import APIConfig
from core.rag_chat import Answer


class FakeChat:
    """Minimal stand-in for RagChat: streams a canned answer and records the
    history it was handed."""

    def __init__(self, answer="grounded answer"):
        self.answer = answer
        self.stream_calls = 0
        self.seen_histories = []

    async def stream(self, history, message):
        self.stream_calls += 1
        self.seen_histories.append(list(history))
        yield "delta", self.answer
        yield "done", Answer(text=self.answer, grounded=True)


@pytest.fixture
def service(monkeypatch):
    chat = FakeChat()
    monkeypatch.setattr(chat_module, "RagChat", lambda cfg: chat)
    # The PII guards have their own tests; here they would only obscure what
    # these assertions are about.
    monkeypatch.setattr(chat_module, "apply_pii_query_guard", lambda text, cfg: text)
    monkeypatch.setattr(chat_module, "apply_pii_response_guard", lambda text, cfg: text)

    api_config = APIConfig(session_ttl_seconds=60, max_sessions=3, answer_cache_size=8)
    svc = ChatService(agent_config=object(), api_config=api_config)
    svc.chat = chat
    return svc


def _answer(service, message, session_id=None):
    return asyncio.run(service.answer(message, session_id))


def test_issued_session_ids_pass_validation_and_junk_does_not():
    assert is_valid_session_id(new_session_id())
    assert not is_valid_session_id("")
    assert not is_valid_session_id("../../etc/passwd")
    assert not is_valid_session_id("short")


def test_unknown_session_id_gets_a_fresh_one_instead_of_being_adopted(service):
    forged = new_session_id()
    result = _answer(service, "hello", forged)
    assert result.session_id != forged


def test_same_session_id_is_reused_across_turns(service):
    first = _answer(service, "hello")
    second = _answer(service, "and then?", first.session_id)
    assert second.session_id == first.session_id


def test_conversation_history_is_replayed_into_the_next_turn(service):
    first = _answer(service, "Do you handle SBA loans?")
    _answer(service, "what about the fees?", first.session_id)

    history = service.chat.seen_histories[-1]
    assert [turn.user for turn in history] == ["Do you handle SBA loans?"]
    assert history[0].assistant == first.answer


def test_history_is_bounded(service):
    first = _answer(service, "opening question")
    for i in range(chat_module.MAX_HISTORY_TURNS + 4):
        _answer(service, f"follow up {i}", first.session_id)
    assert len(service._history[first.session_id]) <= chat_module.MAX_HISTORY_TURNS


def test_expired_sessions_are_dropped_with_their_conversation_state(service):
    result = _answer(service, "hello")
    # Age the session past its TTL.
    service._sessions[result.session_id] = time.monotonic() - 999

    _answer(service, "someone else's first message")

    assert result.session_id not in service._sessions
    assert result.session_id not in service._history


def test_session_count_stays_under_the_ceiling(service):
    for i in range(10):
        _answer(service, f"message {i}")
    assert service.active_sessions <= service._api.max_sessions


def test_repeated_first_question_is_served_from_cache(service):
    first = _answer(service, "What are your fees?")
    calls_after_first = service.chat.stream_calls

    second = _answer(service, "what are   your FEES?")

    assert second.cached is True
    assert not first.cached
    assert second.answer == first.answer
    # The whole point: no second trip to the LLM.
    assert service.chat.stream_calls == calls_after_first
    # ...but the exchange still lands in history, or the follow-up turn would
    # run against an empty conversation.
    assert service._history[second.session_id][-1].assistant == second.answer


def test_follow_up_turns_are_never_cache_served(service):
    first = _answer(service, "What are your fees?")
    calls = service.chat.stream_calls

    follow_up = _answer(service, "What are your fees?", first.session_id)

    assert follow_up.cached is False
    assert service.chat.stream_calls == calls + 1


# ── LLM health tracking (drives /health) ────────────────────────────────────

class FailingChat:
    """Stand-in RagChat whose stream() always raises -- simulates every
    Gemini key being exhausted or the API being down."""

    async def stream(self, history, message):
        raise RuntimeError("all keys exhausted")
        yield  # pragma: no cover -- makes this an async generator


def test_healthy_with_no_traffic_yet(service):
    assert service.is_llm_degraded is False


def test_stays_healthy_after_a_single_failure(service):
    service._chat = FailingChat()
    with pytest.raises(RuntimeError):
        _answer(service, "hello")
    assert service.is_llm_degraded is False


def test_flips_degraded_after_consecutive_failures(service):
    service._chat = FailingChat()
    for i in range(chat_module.LLM_HEALTH_FAILURE_THRESHOLD):
        with pytest.raises(RuntimeError):
            _answer(service, f"message {i}")
    assert service.is_llm_degraded is True


def test_recovers_once_a_turn_succeeds(service):
    service._chat = FailingChat()
    for i in range(chat_module.LLM_HEALTH_FAILURE_THRESHOLD):
        with pytest.raises(RuntimeError):
            _answer(service, f"message {i}")
    assert service.is_llm_degraded is True

    service._chat = FakeChat()  # simulates the outage clearing
    _answer(service, "one more try")

    assert service.is_llm_degraded is False


def test_cache_hits_do_not_count_as_llm_outcomes(service):
    """A cached first-turn answer never reaches _generate, so it must not mask
    (or be mistaken for) a real LLM failure."""
    _answer(service, "What are your fees?")
    second = _answer(service, "What are your fees?")
    assert second.cached is True
    assert len(service._recent_llm_outcomes) == 1  # only the first, uncached turn
