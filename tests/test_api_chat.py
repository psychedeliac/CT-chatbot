"""
Checks for the bounded-resource logic in api/chat.py -- session expiry, the
size ceiling, and the first-turn-only answer cache. These are the parts that
decide whether the process survives a day of public traffic, and none of them
are exercised by the retrieval or guardrail tests.

No LLM and no vector store: the agent is faked, so this runs in milliseconds
and costs no API quota.
"""
import asyncio
import time

import pytest

from api import chat as chat_module
from api.chat import ChatService, is_valid_session_id, new_session_id
from config import APIConfig


class FakeAgent:
    """Minimal stand-in for a LangGraph agent: streams a canned answer."""

    def __init__(self, answer="grounded answer"):
        self.answer = answer
        self.deleted_threads = []
        self.recorded_states = []
        self.stream_calls = 0
        self.checkpointer = self

    def delete_thread(self, thread_id):
        self.deleted_threads.append(thread_id)

    def update_state(self, config, values):
        self.recorded_states.append((config["configurable"]["thread_id"], values))

    async def astream(self, payload, config, stream_mode):
        self.stream_calls += 1

        class _Token:
            content = self.answer

        yield "messages", (_Token(), {})
        yield "values", {"messages": [_Token()]}


@pytest.fixture
def service(monkeypatch):
    agent = FakeAgent()
    monkeypatch.setattr(chat_module.AgentFactory, "create", staticmethod(lambda cfg: agent))
    # The grounding backstop and PII guards have their own tests; here they
    # would only obscure what these assertions are about.
    monkeypatch.setattr(chat_module, "enforce_grounding_refusal", lambda state, text: text)
    monkeypatch.setattr(chat_module, "apply_pii_query_guard", lambda text, cfg: text)
    monkeypatch.setattr(chat_module, "apply_pii_response_guard", lambda text, cfg: text)

    api_config = APIConfig(session_ttl_seconds=60, max_sessions=3, answer_cache_size=8)
    svc = ChatService(agent_config=object(), api_config=api_config)
    svc.agent = agent
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


def test_expired_sessions_are_dropped_with_their_conversation_state(service):
    result = _answer(service, "hello")
    # Age the session past its TTL.
    service._sessions[result.session_id] = time.monotonic() - 999

    _answer(service, "someone else's first message")

    assert result.session_id not in service._sessions
    assert result.session_id in service._agent.deleted_threads


def test_session_count_stays_under_the_ceiling(service):
    for i in range(10):
        _answer(service, f"message {i}")
    assert service.active_sessions <= service._api.max_sessions


def test_repeated_first_question_is_served_from_cache(service):
    first = _answer(service, "What are your fees?")
    calls_after_first = service._agent.stream_calls

    second = _answer(service, "what are   your FEES?")

    assert second.cached is True
    assert not first.cached
    assert second.answer == first.answer
    # The whole point: no second trip to the LLM.
    assert service._agent.stream_calls == calls_after_first
    # ...but the exchange still lands in history, or the follow-up turn would
    # run against an empty conversation.
    assert service._agent.recorded_states[-1][0] == second.session_id


def test_follow_up_turns_are_never_cache_served(service):
    first = _answer(service, "What are your fees?")
    calls = service._agent.stream_calls

    follow_up = _answer(service, "What are your fees?", first.session_id)

    assert follow_up.cached is False
    assert service._agent.stream_calls == calls + 1
