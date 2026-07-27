"""
tests/test_api_chat.py — what the public endpoint guarantees, asserted from
outside it.

The previous version reached into `service._history`, `service._sessions` and
`service._recent_llm_outcomes` and asserted those dicts looked right. Internal
state being correct is not the guarantee anyone cares about, and it means a
refactor breaks green tests while a real leak between two users' conversations
would not have been noticed at all. It also monkeypatched BOTH PII guards away
in the shared fixture, so the compliance chain this module's docstring promises
("the public endpoint can never enforce a weaker rule than the internal UIs")
had no test on it whatsoever.

Everything here goes through the public surface -- stream_turn(), answer(),
active_sessions, is_llm_degraded -- plus one legitimate observation point: what
the fake RagChat was handed, which is exactly the data that would reach a real
LLM. The clock is faked where a test needs time to pass; nothing else is.

RagChat is faked at the boundary, so no corpus and no API quota: milliseconds.
"""
import asyncio

import pytest

from api import chat as chat_module
from api.chat import (
    CapacityError,
    ChatService,
    TurnTimeout,
    is_valid_session_id,
    new_session_id,
)
from config import APIConfig, PIIConfig
from core.rag_chat import Answer, Turn


class FakeChat:
    """Stands in for RagChat at the network boundary. Echoes back what it was
    given, so a test can see exactly which message and which history would
    have reached the model."""

    def __init__(self, answer="grounded answer"):
        self.answer = answer
        self.turns = []  # (history, message) per call

    async def stream(self, history, message):
        self.turns.append((list(history), message))
        yield "context", "RETRIEVED-CONTEXT-INTERNAL-ONLY"
        yield "delta", self.answer
        yield "done", Answer(text=self.answer, grounded=True)

    @property
    def call_count(self):
        return len(self.turns)

    def history_of_last_turn(self):
        return self.turns[-1][0]

    def message_of_last_turn(self):
        return self.turns[-1][1]


class EchoHistoryChat(FakeChat):
    """Answers with the conversation it was given -- makes history leakage
    between sessions directly visible in the returned answer."""

    async def stream(self, history, message):
        self.turns.append((list(history), message))
        reply = f"you previously said: {[t.user for t in history]}"
        yield "delta", reply
        yield "done", Answer(text=reply, grounded=True)


class OutageChat:
    """Delegates to whatever fake is currently installed, so a test can take
    the LLM down and bring it back without reaching into ChatService."""

    def __init__(self, inner):
        self.inner = inner

    def stream(self, history, message):
        return self.inner.stream(history, message)


def build_service(chat=None, pii_enabled=False, **api_kwargs):
    """A ChatService whose answer engine is a fake. Everything else -- session
    handling, the cache, the PII chain -- is the real implementation."""
    settings = dict(session_ttl_seconds=60, max_sessions=3, answer_cache_size=8)
    settings.update(api_kwargs)

    agent_config = type("Cfg", (), {"pii": PIIConfig(enabled=pii_enabled)})()
    original = chat_module.RagChat
    chat_module.RagChat = lambda cfg: (chat if chat is not None else FakeChat())
    try:
        return ChatService(agent_config=agent_config, api_config=APIConfig(**settings))
    finally:
        chat_module.RagChat = original


def answer(service, message, session_id=None):
    return asyncio.run(service.answer(message, session_id))


def events(service, message, session_id=None):
    async def drain():
        return [event async for event in service.stream_turn(message, session_id)]
    return asyncio.run(drain())


# ── Session identity ────────────────────────────────────────────────────────

def test_only_ids_we_issued_are_accepted():
    """A caller who can name an arbitrary session id can inject turns into a
    conversation they do not own."""
    assert is_valid_session_id(new_session_id())
    assert not is_valid_session_id("")
    assert not is_valid_session_id("short")
    assert not is_valid_session_id("../../etc/passwd")
    assert not is_valid_session_id("a" * 31)


def test_an_unissued_session_id_starts_an_empty_conversation_rather_than_adopting_one():
    chat = EchoHistoryChat()
    service = build_service(chat)

    forged = new_session_id()  # well-formed, but never issued by this process
    result = answer(service, "hello", forged)

    assert result.session_id != forged
    assert chat.history_of_last_turn() == []


def test_a_conversation_continues_under_the_id_it_was_issued():
    service = build_service()
    first = answer(service, "Do you handle SBA loans?")
    second = answer(service, "and the fees?", first.session_id)
    assert second.session_id == first.session_id


# ── Isolation between conversations ─────────────────────────────────────────

def test_one_users_conversation_never_reaches_anothers_turn():
    """The failure this exists for: a shared RagChat plus a shared history
    store answering user B with user A's disclosed details."""
    chat = EchoHistoryChat()
    service = build_service(chat)

    alice = answer(service, "my restaurant owes 3 lenders")
    answer(service, "what should I do", alice.session_id)

    bob = answer(service, "do you handle payroll taxes")

    assert "restaurant" not in bob.answer
    assert chat.history_of_last_turn() == []
    assert bob.session_id != alice.session_id


def test_history_is_replayed_into_the_next_turn_of_the_same_conversation():
    chat = FakeChat()
    service = build_service(chat)

    first = answer(service, "Do you handle SBA loans?")
    answer(service, "what about the fees?", first.session_id)

    replayed = chat.history_of_last_turn()
    assert [turn.user for turn in replayed] == ["Do you handle SBA loans?"]
    assert replayed[0].assistant == first.answer


def test_replayed_history_is_capped_at_what_the_model_will_read():
    """Anything older than RagChat replays is memory held for nothing."""
    chat = FakeChat()
    service = build_service(chat)

    first = answer(service, "opening question")
    for i in range(chat_module.MAX_HISTORY_TURNS + 5):
        answer(service, f"follow up {i}", first.session_id)

    assert len(chat.history_of_last_turn()) <= chat_module.MAX_HISTORY_TURNS


# ── Bounded resources ───────────────────────────────────────────────────────

def test_a_conversation_idle_past_its_ttl_is_dropped_with_its_history(monkeypatch):
    """Expiry must take the id AND the conversation. If the id went and the
    history stayed, a later session could inherit it; if the history went and
    the id stayed, a returning user would silently talk to an empty thread.

    Note the trigger: pruning runs on each incoming turn, so it is OTHER
    traffic that collects an idle session. _resolve_session tests membership
    before it touches the timestamp, so a user returning to a quiet instance
    can still resume past the TTL -- harmless (it is their own conversation)
    but it means the TTL is a lazy sweep, not a hard deadline.
    """
    chat = EchoHistoryChat()
    service = build_service(chat, session_ttl_seconds=60)

    now = [1000.0]
    monkeypatch.setattr(chat_module.time, "monotonic", lambda: now[0])

    first = answer(service, "my shop owes money to 4 lenders")
    now[0] += 61                       # the conversation goes idle past its TTL
    answer(service, "unrelated visitor")  # ...and someone else's turn sweeps it

    resumed = answer(service, "so what now", first.session_id)

    assert resumed.session_id != first.session_id
    assert "shop" not in resumed.answer
    assert chat.history_of_last_turn() == []


def test_session_count_cannot_be_grown_past_the_ceiling():
    """Without this the process accumulates every conversation it has ever
    seen and is eventually OOM-killed."""
    service = build_service(max_sessions=3)
    for i in range(25):
        answer(service, f"message {i}")
    assert service.active_sessions <= 3


def test_the_conversation_being_served_is_never_the_one_evicted():
    """Eviction is oldest-first and runs on every turn; evicting the session
    mid-turn would silently reset the user who is talking right now."""
    service = build_service(max_sessions=2)

    live = answer(service, "first message")
    for i in range(5):
        answer(service, f"other user {i}")          # churn past the ceiling
        follow_up = answer(service, "still here?", live.session_id)
        live = follow_up if follow_up.session_id == live.session_id else live

    assert service.active_sessions <= 2


def test_a_hung_turn_is_cut_off_instead_of_holding_the_connection_open():
    class HangingChat:
        async def stream(self, history, message):
            await asyncio.sleep(5)
            yield "done", Answer(text="too late", grounded=True)

    service = build_service(HangingChat(), turn_timeout_seconds=0.05)
    with pytest.raises(TurnTimeout):
        answer(service, "hello")


def test_traffic_past_the_concurrency_ceiling_is_refused_rather_than_queued_forever():
    class SlowChat:
        async def stream(self, history, message):
            await asyncio.sleep(0.2)
            yield "done", Answer(text="ok", grounded=True)

    service = build_service(
        SlowChat(), max_concurrent_turns=1, queue_timeout_seconds=0.01
    )

    async def two_at_once():
        return await asyncio.gather(
            service.answer("first", None),
            service.answer("second", None),
            return_exceptions=True,
        )

    outcomes = asyncio.run(two_at_once())
    assert any(isinstance(o, CapacityError) for o in outcomes)


# ── The answer cache ────────────────────────────────────────────────────────

def test_an_identical_opening_question_is_served_without_a_second_llm_call():
    chat = FakeChat()
    service = build_service(chat)

    first = answer(service, "What are your fees?")
    calls = chat.call_count

    second = answer(service, "what are   your FEES?")   # normalized to the same key

    assert second.cached is True and not first.cached
    assert second.answer == first.answer
    assert chat.call_count == calls


def test_a_cached_answer_is_never_reused_for_a_turn_that_depends_on_history():
    """Only first turns are cacheable. Keying a follow-up on its message alone
    would serve one user's conversational context to another."""
    chat = EchoHistoryChat()
    service = build_service(chat)

    first = answer(service, "What are your fees?")
    calls = chat.call_count

    follow_up = answer(service, "What are your fees?", first.session_id)

    assert follow_up.cached is False
    assert chat.call_count == calls + 1
    assert "What are your fees?" in follow_up.answer  # answered against real history


def test_a_cache_served_turn_still_becomes_part_of_the_conversation():
    """Skipping this is a real bug, not an optimization: the next turn would
    run against an empty history and the model would not know what was asked."""
    chat = EchoHistoryChat()
    service = build_service(chat)

    answer(service, "What are your fees?")                      # populates the cache
    cached_turn = answer(service, "What are your fees?")         # served from it
    assert cached_turn.cached is True

    answer(service, "and how long does it take?", cached_turn.session_id)

    assert [t.user for t in chat.history_of_last_turn()] == ["What are your fees?"]


# ── What may leave the process ──────────────────────────────────────────────

def test_retrieved_context_is_never_emitted_to_a_public_caller():
    """RagChat yields the retrieved block for the internal QA panel. Forwarding
    it to the widget would publish the corpus verbatim, disclaimers and
    scope-guard instructions included."""
    service = build_service(FakeChat())

    emitted = events(service, "Do you handle SBA loans?")

    assert not any(
        "RETRIEVED-CONTEXT-INTERNAL-ONLY" in str(event) for event in emitted
    )
    assert [event["type"] for event in emitted] == ["session", "delta", "done"]


def test_the_done_event_is_authoritative_when_it_differs_from_the_deltas():
    """The guards can replace an answer wholesale. A client that renders the
    deltas and ignores `done` would leave a non-compliant answer on screen, so
    the two must be distinguishable in the stream."""
    class RevisingChat:
        async def stream(self, history, message):
            yield "delta", "Settlement saves 60% guaranteed."
            yield "done", Answer(text="Results vary by situation.", grounded=False)

    service = build_service(RevisingChat())
    emitted = events(service, "how much will I save")

    delta = next(e for e in emitted if e["type"] == "delta")
    done = next(e for e in emitted if e["type"] == "done")
    assert delta["text"] == "Settlement saves 60% guaranteed."
    assert done["answer"] == "Results vary by situation."


def test_pii_is_stripped_before_the_model_sees_it_and_before_the_answer_goes_out():
    """The real Presidio guards, on the real public path. Previously both were
    monkeypatched out of every test in this file."""
    chat = FakeChat(answer="Reach our team at 1-800-889-0232, or Bob on 415-555-0134.")
    service = build_service(chat, pii_enabled=True)

    result = answer(service, "I'm Bob Smith, my cell is 415-555-0134, I owe $40k")

    assert "415-555-0134" not in chat.message_of_last_turn(), "PII reached the model"
    assert "415-555-0134" not in result.answer, "PII went back to the caller"
    assert "1-800-889-0232" in result.answer, "our own published line was redacted"


# ── Health signal (drives /health, and therefore load-balancer routing) ──────

class FailingChat:
    """Every key exhausted, or Gemini down."""

    async def stream(self, history, message):
        raise RuntimeError("all keys exhausted")
        yield  # pragma: no cover -- makes this an async generator


def _fail_turns(service, count):
    for i in range(count):
        with pytest.raises(RuntimeError):
            answer(service, f"message {i}")


def test_a_single_failure_does_not_take_the_instance_out_of_rotation():
    service = build_service(FailingChat())
    assert service.is_llm_degraded is False

    _fail_turns(service, 1)

    assert service.is_llm_degraded is False


def test_an_unbroken_run_of_failures_reports_degraded():
    service = build_service(FailingChat())
    _fail_turns(service, chat_module.LLM_HEALTH_FAILURE_THRESHOLD)
    assert service.is_llm_degraded is True


def test_health_recovers_on_the_first_turn_that_succeeds():
    outage = OutageChat(FailingChat())
    service = build_service(outage)
    _fail_turns(service, chat_module.LLM_HEALTH_FAILURE_THRESHOLD)

    outage.inner = FakeChat()   # the outage clears
    answer(service, "one more try")

    assert service.is_llm_degraded is False


def test_cache_hits_cannot_fake_a_recovery():
    """A cached answer never reaches the LLM, so it says nothing about whether
    the LLM is reachable. Counting it as a success would flap an unhealthy
    instance back into rotation on repeat traffic alone."""
    outage = OutageChat(FakeChat())
    service = build_service(outage)
    answer(service, "What are your fees?")          # populates the cache

    outage.inner = FailingChat()
    _fail_turns(service, chat_module.LLM_HEALTH_FAILURE_THRESHOLD)
    assert service.is_llm_degraded is True

    cached = answer(service, "What are your fees?")  # served from cache

    assert cached.cached is True
    assert service.is_llm_degraded is True
