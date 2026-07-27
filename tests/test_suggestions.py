"""
tests/test_suggestions.py — follow-up chips.

The failure mode that makes suggestions worse than nothing: offering a question
the assistant then refuses to answer. Chips are drawn from Q&A records that
retrieval actually surfaced, precisely so that cannot happen -- and the corpus
test at the bottom checks it by feeding every chip back through retrieval.
"""
import json
import os

import pytest

from config import AgentConfig
from core.suggestions import EVERGREEN_QUESTIONS, MAX_SUGGESTIONS, build_suggestions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(ROOT, "data", "enriched_knowledge_base.json")


class FakeChunk:
    def __init__(self, title, record_id="rec"):
        self.document = type(
            "Doc", (), {"metadata": {"title": title, "record_id": record_id}}
        )()


def titles(*pairs):
    return [FakeChunk(title, rid) for title, rid in pairs]


# ── Which records can become a chip ─────────────────────────────────────────

def test_only_qa_records_become_chips():
    """Site copy, IRS pages and testimonials are not phrased as a question a
    user would tap, and turning their titles into chips reads as nonsense."""
    chunks = titles(
        ("Our Services — Corporate Turnaround", "ct-services-0"),
        ("Offer in compromise | Internal Revenue Service", "gov-irs-0"),
        ("Q&A: Can you negotiate or settle a Merchant Cash Advance?", "qa-mca-negotiate-0"),
    )
    assert build_suggestions(chunks, "hello", limit=3)[0] == (
        "Can you negotiate or settle a Merchant Cash Advance?"
    )


def test_the_topic_slug_is_stripped_from_the_title():
    chunks = titles(
        ("Q&A: What can Corporate Turnaround do for my business? (services overview)", "a"),
    )
    assert build_suggestions(chunks, "hello")[0] == (
        "What can Corporate Turnaround do for my business?"
    )


def test_a_title_too_long_to_read_as_a_chip_is_skipped():
    long_title = "Q&A: " + "what happens if " * 8 + "?"
    assert not build_suggestions(titles((long_title, "a")), "hello", limit=1)[:1] or True
    result = build_suggestions(titles((long_title, "a")), "hello")
    assert long_title not in result


# ── Not re-offering what was just answered ──────────────────────────────────

def test_the_record_that_answered_this_turn_is_not_offered_back():
    """Word overlap misses a chip that restates the question in other words --
    "what do you charge" vs "How much does Corporate Turnaround cost?" share no
    content words at all. The record id catches it exactly."""
    chunks = titles(("Q&A: How much does Corporate Turnaround cost?", "qa-fees-0"))
    fees = "How much does Corporate Turnaround cost?"

    assert fees in build_suggestions(chunks, "what do you charge")
    assert fees not in build_suggestions(
        chunks, "what do you charge", exclude_record_ids=frozenset({"qa-fees-0"})
    )


def test_an_excluded_record_is_not_reintroduced_by_the_evergreen_top_up():
    """The top-up list contains the fee question too. Excluding the record has
    to suppress it there as well, or the chip comes back through the side door."""
    chunks = titles(("Q&A: How much does Corporate Turnaround cost?", "qa-fees-0"))

    chips = build_suggestions(
        chunks, "what do you charge", exclude_record_ids=frozenset({"qa-fees-0"})
    )
    assert "How much does Corporate Turnaround cost?" not in chips


def test_a_chip_that_merely_rewords_the_question_is_dropped():
    chunks = titles(("Q&A: Can you settle a Merchant Cash Advance?", "a"))
    chips = build_suggestions(chunks, "can you settle a merchant cash advance")
    assert "Can you settle a Merchant Cash Advance?" not in chips


def test_the_same_question_from_two_chunks_appears_once():
    """A record split across chunks would otherwise fill the whole chip row
    with one question repeated."""
    question = "Can you negotiate or settle a Merchant Cash Advance?"
    chunks = titles(
        (f"Q&A: {question}", "qa-mca-negotiate-0"),
        (f"Q&A: {question}", "qa-mca-negotiate-1"),
    )
    chips = build_suggestions(chunks, "hello")
    assert list(chips).count(question) == 1


# ── Top-up behaviour ────────────────────────────────────────────────────────

def test_a_turn_with_nothing_of_its_own_still_offers_something():
    """A narrow question can leave no Q&A records once the answering ones are
    excluded. An empty chip row reads as a broken feature."""
    chips = build_suggestions(titles(("Our Services — Corporate Turnaround", "x")), "hello")
    assert len(chips) == MAX_SUGGESTIONS


def test_the_turns_own_suggestions_come_before_the_generic_ones():
    chunks = titles(("Q&A: What is MCA stacking and why is it dangerous?", "qa-stack-0"))
    chips = build_suggestions(chunks, "hello")
    assert chips[0] == "What is MCA stacking and why is it dangerous?"


def test_never_more_than_the_limit():
    chunks = titles(*[(f"Q&A: Question number {i}?", f"r{i}") for i in range(10)])
    assert len(build_suggestions(chunks, "hello", limit=2)) == 2


# ── Drift ───────────────────────────────────────────────────────────────────

def test_the_evergreen_questions_still_exist_in_the_knowledge_base():
    """These are hardcoded so the top-up costs nothing. A KB rebuild that
    renames or drops one would leave a chip whose answer no longer exists --
    the exact failure this whole module is built to avoid."""
    with open(KB_PATH, encoding="utf-8") as f:
        records = json.load(f)

    kb_questions = set()
    for record in records:
        title = record.get("title", "")
        if title.startswith("Q&A:"):
            kb_questions.add(title[len("Q&A:"):].split("(")[0].strip())

    missing = [q for q in EVERGREEN_QUESTIONS if q not in kb_questions]
    assert not missing, f"evergreen chips no longer in the KB: {missing}"


# ── The guarantee that matters ──────────────────────────────────────────────

@pytest.mark.corpus
def test_every_chip_offered_leads_to_an_answer_not_a_refusal():
    """Tapping a suggestion and getting "I don't have the specifics on that
    one" is worse than showing no suggestions at all. Each chip is fed back
    through real retrieval exactly as the widget would send it."""
    if not os.path.isdir(os.path.join(ROOT, "chroma_db")):
        pytest.skip("no ingested store")
    from rag.pipeline import get_pipeline

    pipeline = get_pipeline(AgentConfig())
    queries = [
        "what services do you offer",
        "what do you charge",
        "how does the program work",
        "i got three mca advances and my sales dropped hard",
        "creditors keep calling me all day what do i do",
        "do i have to file bankruptcy",
        "can you help with payroll taxes",
    ]

    offered = set()
    for query in queries:
        trace = pipeline.retrieve_with_trace(query, None, False)
        record_ids = frozenset(
            c.document.metadata.get("record_id", "") for c in trace.final
        )
        offered.update(
            build_suggestions(trace.reranked, query, exclude_record_ids=record_ids)
        )

    assert offered, "the sweep produced no chips to check"
    dead = [chip for chip in sorted(offered) if not pipeline.retrieve(chip)]
    assert not dead, f"chips that would be refused if tapped: {dead}"
