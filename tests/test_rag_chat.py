"""
Single-pass RAG path (core/rag_chat.py).

Covers the two behaviours that changed when the ReAct router call was
removed: the retrieval query must still be standalone on follow-up turns
(the router used to rewrite it), and the grounding backstop must still fire
when retrieval comes back empty.

No LLM and no corpus: RagChat is exercised through its pure pieces plus a
stub, so this runs in CI without burning API quota.
"""
import asyncio

from core.rag_chat import Turn, build_retrieval_query
from core.utils import REFUSAL_MESSAGE, enforce_grounding


def test_first_turn_retrieves_on_the_raw_message():
    assert build_retrieval_query([], "Do you handle SBA loans?") == "Do you handle SBA loans?"


def test_followup_carries_the_previous_question():
    history = [Turn(user="Do you help with merchant cash advances?", assistant="Yes...")]
    query = build_retrieval_query(history, "how do I get out of it?")
    assert "merchant cash advances" in query
    assert "how do I get out of it?" in query


def test_followup_uses_only_the_most_recent_turn():
    history = [
        Turn(user="Do you handle SBA loans?", assistant="..."),
        Turn(user="What about payroll taxes?", assistant="..."),
    ]
    query = build_retrieval_query(history, "and the penalties?")
    assert "payroll taxes" in query
    assert "SBA" not in query


def test_ungrounded_substantive_answer_is_replaced():
    answer = (
        "Business debt settlement typically reduces balances by 40-60% over 24 months, "
        "and creditors generally accept these terms once an account is 90 days delinquent."
    )
    assert enforce_grounding(False, answer) == REFUSAL_MESSAGE


def test_ungrounded_deflection_survives():
    answer = "I'm an AI assistant for Corporate Turnaround -- what's going on with your business?"
    assert enforce_grounding(False, answer) == answer


def test_grounded_answer_is_untouched():
    answer = "We negotiate SBA workouts directly with lenders. Call 1-800-889-0232."
    assert enforce_grounding(True, answer) == answer


def test_stream_yields_deltas_then_final_answer():
    """The contract api/chat.py depends on: deltas first, one ('done', Answer) last."""
    from core.rag_chat import Answer, RagChat

    chat = RagChat.__new__(RagChat)  # bypass __init__: no corpus, no LLM needed

    async def fake_retrieve(history, message):
        return "<retrieved_context>SBA workouts</retrieved_context>", True

    class _Chunk:
        def __init__(self, content):
            self.content = content

    class _LLM:
        async def astream(self, messages):
            for piece in ("We negotiate ", "SBA workouts."):
                yield _Chunk(piece)

    chat.retrieve = fake_retrieve
    chat._llm = _LLM()
    chat._config = type("C", (), {"system_prompt": "sys", "inline_retrieval_contract": "ctx"})()

    async def drain():
        return [event async for event in chat.stream([], "Do you handle SBA loans?")]

    events = asyncio.run(drain())
    assert [kind for kind, _ in events] == ["delta", "delta", "done"]
    final = events[-1][1]
    assert isinstance(final, Answer)
    assert final.text == "We negotiate SBA workouts."
    assert final.grounded is True
