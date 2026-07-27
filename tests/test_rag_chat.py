"""
tests/test_rag_chat.py — the answer path, wired the way production wires it.

The previous version of this file built a RagChat with `__new__` and replaced
BOTH retrieval and the LLM with stubs, then asserted the stub's output came
back. That cannot fail for any reason a user would notice: retrieval, the
prompt assembly, and the grounding backstop were all absent from the thing
under test.

Here only the NETWORK is faked. Retrieval is the real RetrievalPipeline over
the real corpus; _messages, clean_response_prefix and enforce_grounding are
the real ones. The fake LLM records exactly what the model would have been
sent and returns what a misbehaving model would return -- which is the point,
because the backstop exists for models that ignore the prompt.

No API quota. Loads the corpus + reranker once (~20s); skip with -m "not corpus".
"""
import asyncio

import pytest

from config import AgentConfig
from core.rag_chat import MAX_HISTORY_TURNS, Answer, RagChat, Turn, build_retrieval_query
from core.utils import REFUSAL_MESSAGE
from rag.pipeline import get_pipeline

pytestmark = pytest.mark.corpus


class FakeLLM:
    """Stands in for the Gemini client at the network boundary. Streams the
    text it was constructed with and keeps every message list it was sent."""

    def __init__(self, reply="We negotiate SBA workouts directly with your lender."):
        self.reply = reply
        self.calls = []

    async def astream(self, messages):
        self.calls.append(messages)
        # Fixed-width slices rather than word splits: the concatenation has to
        # reproduce the reply byte for byte, or the test asserts against text
        # the fake invented.
        for i in range(0, len(self.reply), 8):
            yield type("Chunk", (), {"content": self.reply[i:i + 8]})()

    @property
    def last_messages(self):
        return self.calls[-1]


def build_chat(reply):
    """Real config, real pipeline, real prompt assembly -- fake network."""
    chat = RagChat.__new__(RagChat)
    chat._config = AgentConfig()
    chat._pipeline = get_pipeline(chat._config)
    chat._llm = FakeLLM(reply)
    return chat


def run_turn(chat, message, history=()):
    async def drain():
        return [event async for event in chat.stream(list(history), message)]
    return asyncio.run(drain())


def final_answer(events) -> Answer:
    return events[-1][1]


# ── The grounding contract, end to end ──────────────────────────────────────

def test_an_off_topic_question_is_refused_even_when_the_model_answers_it():
    """The whole chain: a question the corpus cannot ground, a model that
    answers it anyway (which is what an LLM does), and the deterministic
    backstop that has to catch it. Nothing here is stubbed except the network.
    """
    chat = build_chat(
        "The capital of France is Paris, home to about 2,100,000 people."
    )
    events = run_turn(chat, "what is the capital of France")

    answer = final_answer(events)
    assert answer.grounded is False
    assert answer.text == REFUSAL_MESSAGE


def test_a_grounded_answer_is_handed_back_untouched():
    chat = build_chat("We negotiate SBA workouts directly with your lender.")
    events = run_turn(chat, "Do you handle SBA loans?")

    answer = final_answer(events)
    assert answer.grounded is True
    assert answer.text == "We negotiate SBA workouts directly with your lender."


def test_an_ungrounded_greeting_is_not_replaced_by_the_refusal():
    """Every widget conversation opens with one of these. Replacing a greeting
    with 'I don't have the specifics on that one' is the failure mode that made
    the backstop worth testing at this level."""
    chat = build_chat(
        "Hello! I'm an AI assistant for Corporate Turnaround -- what's going on "
        "with your business?"
    )
    events = run_turn(chat, "hi")

    assert final_answer(events).text.startswith("Hello!")


def test_the_streamed_deltas_are_provisional_and_the_done_event_overrides_them():
    """The contract api/chat.py and the widget both depend on: when the
    backstop fires, what was typed on screen is NOT the answer."""
    chat = build_chat("Debt settlement saves 60% in 18 months, guaranteed.")
    events = run_turn(chat, "how do I fix a leaking kitchen tap")

    streamed = "".join(payload for kind, payload in events if kind == "delta")
    final = final_answer(events).text
    assert "60%" in streamed          # the model really did say it
    assert final == REFUSAL_MESSAGE   # ...and the user never gets it
    assert final != streamed


# ── What the model is actually shown ────────────────────────────────────────

def test_the_user_message_and_the_retrieved_context_arrive_in_separate_tagged_blocks():
    """The system prompt instructs the model to treat <user_query> as ground
    truth and <retrieved_context> as reference material. Both tags have to
    exist in what is actually sent, or the instruction points at nothing."""
    chat = build_chat("...")
    run_turn(chat, "Do you handle SBA loans?")

    sent = chat._llm.last_messages[-1].content
    assert "<user_query>" in sent and "</user_query>" in sent
    assert "<retrieved_context>" in sent and "</retrieved_context>" in sent
    # The user's own words go inside the user block, ahead of any context.
    # (Compared against the CLOSING tag: the user-query preamble itself
    # mentions "<retrieved_context>" when explaining the distinction.)
    assert sent.index("Do you handle SBA loans?") < sent.index("</user_query>")
    assert sent.index("</user_query>") < sent.index("</retrieved_context>")


def test_qa_pair_context_is_labelled_so_it_is_not_read_as_the_users_own_story():
    """Q&A records are written in the first person ("i got three mca advances").
    Unlabelled, the model adopts the scenario as this user's facts and starts
    referring to advances they never mentioned."""
    chat = build_chat("...")
    run_turn(chat, "i got three mca advances and my sales dropped hard")

    sent = chat._llm.last_messages[-1].content
    assert "Similar past case (NOT the current user)" in sent


def test_the_system_prompt_is_first_and_byte_identical_across_turns():
    """Gemini's implicit caching keys on the longest shared prefix; anything
    variable ahead of the system prompt forfeits the prefill discount."""
    chat = build_chat("...")
    run_turn(chat, "Do you handle SBA loans?")
    run_turn(chat, "what about payroll taxes", history=[Turn("a", "b")])

    firsts = [messages[0] for messages in chat._llm.calls]
    assert len({m.content for m in firsts}) == 1
    assert firsts[0].content == AgentConfig().system_prompt


def test_replayed_history_is_bounded_so_prefill_cannot_grow_without_limit():
    long_history = [Turn(f"question {i}", f"answer {i}") for i in range(MAX_HISTORY_TURNS + 8)]
    chat = build_chat("...")
    run_turn(chat, "and after that?", history=long_history)

    sent = chat._llm.last_messages
    # system + 2 per replayed turn + the current user message
    assert len(sent) == 1 + 2 * MAX_HISTORY_TURNS + 1
    assert "question 0" not in str(sent)


# ── Event contract ──────────────────────────────────────────────────────────

def test_stream_yields_context_once_first_then_deltas_then_exactly_one_done():
    chat = build_chat("We can help with that.")
    kinds = [kind for kind, _ in run_turn(chat, "Do you handle SBA loans?")]

    assert kinds[0] == "context"
    assert kinds.count("context") == 1
    assert kinds[-1] == "done"
    assert kinds.count("done") == 1
    assert set(kinds[1:-1]) == {"delta"}


def test_the_context_event_carries_what_retrieval_actually_returned():
    """The QA panel renders this instead of running retrieval a second time,
    so it has to be the same block the model was given."""
    chat = build_chat("...")
    events = run_turn(chat, "Do you handle SBA loans?")

    context = next(payload for kind, payload in events if kind == "context")
    assert context in chat._llm.last_messages[-1].content


# ── Follow-up query construction ────────────────────────────────────────────

def test_a_follow_up_carries_its_subject_into_retrieval():
    """"how do I get out of it?" retrieves nothing on its own. The ReAct router
    used to rewrite it; without that call the previous turn has to supply the
    subject, or every follow-up in a conversation abstains."""
    history = [Turn(user="Do you help with merchant cash advances?", assistant="Yes...")]
    query = build_retrieval_query(history, "how do I get out of it?")
    assert "merchant cash advances" in query and "how do I get out of it?" in query


def test_only_the_most_recent_turn_is_carried_so_stale_subjects_do_not_bleed_in():
    history = [
        Turn(user="Do you handle SBA loans?", assistant="..."),
        Turn(user="What about payroll taxes?", assistant="..."),
    ]
    query = build_retrieval_query(history, "and the penalties?")
    assert "payroll taxes" in query and "SBA" not in query


def test_a_follow_up_actually_retrieves_its_subjects_content():
    """The point of the prefix, checked against the corpus rather than against
    the string it produces. On its own "how do I get out of it?" pulls generic
    filler (a consultation blurb); with the subject carried it has to reach
    actual MCA material, or every follow-up in a conversation is answered from
    whatever happened to score highest on a pronoun."""
    pipeline = get_pipeline(AgentConfig())
    history = [Turn(user="Do you help with merchant cash advances?", assistant="Yes...")]

    def mca_records(query):
        return [
            c.document.metadata.get("record_id", "")
            for c in pipeline.retrieve(query)
            if "mca" in c.document.metadata.get("record_id", "").lower()
        ]

    assert not mca_records("how do I get out of it?")
    assert mca_records(build_retrieval_query(history, "how do I get out of it?"))
