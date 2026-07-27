"""
tests/test_guardrails.py — the compliance layer, checked on real inputs.

What changed from the previous version: every check on the context-formatting
layer used to hand format_for_llm a Document the test itself had just built
with the exact metadata the assertion was about. That proves the formatter can
format; it proves nothing about whether a chunk that actually reaches the model
in production carries its marker, which is the only thing that matters -- a
NerdWallet article voiced as our own advice, or a fee record answered past its
deflection, is a regulatory problem, not a formatting one.

So the marker checks now sweep real queries through real retrieval and assert
the invariant over whatever comes back: any chunk flagged in the corpus must
arrive at the model with its warning attached. The sweep also asserts it saw
each flag at least once, so the invariant can never pass vacuously.

The pure-function checks (grounding backstop, prefix cleaner, PII allowlist)
stay pure -- they have no corpus dependency and no useful integration form --
but the adversarial cases are broadened past the ones that happened to ship.
"""
import collections
import json
import os

import pytest

from config import AgentConfig, PIIConfig, VALID_PII_STRATEGIES
from core.utils import (
    REFUSAL_MESSAGE,
    clean_response_prefix,
    enforce_grounding,
    apply_pii_query_guard,
    apply_pii_response_guard,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(ROOT, "data", "enriched_knowledge_base.json")


# ── The grounding backstop ──────────────────────────────────────────────────
# enforce_grounding is the deterministic half of the grounding rule: the prompt
# asks the model to deflect when retrieval is empty, and this catches the turns
# where it doesn't. Both directions are failures -- refusing a greeting reads as
# broken, and letting a figure through is an unsubstantiated claim.

@pytest.mark.parametrize("reply", [
    "Debt settlement typically saves 40% and takes 24 months.",
    "Most clients settle for around 50 cents on the dollar.",
    "We can usually cut your MCA payment by $3,000 a month.",
    "Expect the process to take 18-36 months.",
    "Our success rate is 94%.",
    "Islam is a monotheistic religion. " * 30,          # off-topic essay
    "Here is a Python script:\n" + "print(1)\n" * 40,   # jailbreak-shaped
])
def test_an_ungrounded_substantive_answer_is_replaced(reply):
    assert enforce_grounding(False, reply) == REFUSAL_MESSAGE


@pytest.mark.parametrize("reply", [
    "Hi there! I'm Corporate Turnaround's AI assistant. What's going on with your business?",
    "I don't have the specifics on that, but our team does -- call 1-800-889-0232.",
    "That's outside what we handle, but I'm glad to help with business debt.",
    # The two figures config.system_prompt itself supplies to the model.
    "Hello! Since 1998 we have worked with over 10,000 small business owners.",
    # Crisis routing. A self-harm message grounds nothing in a debt KB, so this
    # arrives here as an "ungrounded" reply -- and replacing it with the debt
    # refusal is the one substitution that can never be allowed to happen.
    "Please reach out to the 988 Suicide & Crisis Lifeline -- call or text 988, any time, 24/7.",
    "Our client service line is 1-800-411-1113 if you're already enrolled.",
])
def test_a_short_figure_free_deflection_survives(reply):
    assert enforce_grounding(False, reply) == reply


def test_a_grounded_answer_is_never_touched():
    """Retrieval succeeded, so figures in the reply came from the corpus."""
    answer = "We negotiate within a budget you can afford -- over 10,000 owners since 1998."
    assert enforce_grounding(True, answer) == answer


def test_the_refusal_does_not_sound_like_a_search_engine():
    """Users are talking to Corporate Turnaround, not to a retrieval system."""
    assert "knowledge base" not in REFUSAL_MESSAGE.lower()
    assert "context" not in REFUSAL_MESSAGE.lower()
    assert "1-800-889-0232" in REFUSAL_MESSAGE


# ── Response prefix cleanup ─────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("Based on the context, you have options.", "You have options."),
    ("According to the documents, call us.", "Call us."),
    ("BASED ON the provided information, we can help.", "We can help."),
])
def test_rag_preamble_is_stripped(raw, expected):
    assert clean_response_prefix(raw) == expected


@pytest.mark.parametrize("kept", [
    "Basing your plan on revenue is wise.",
    "According to your lender, the balance is due -- that's worth checking.",
    "",
])
def test_sentences_that_merely_look_like_a_preamble_are_left_alone(kept):
    assert clean_response_prefix(kept) == kept


# ── PII (Presidio, real) ────────────────────────────────────────────────────

def test_our_published_numbers_survive_the_scrub_and_a_private_one_does_not():
    """The recognizer cannot tell our published line from a caller's cell.
    Without the allowlist, answers rendered as "call us at [REDACTED_PHONE_NUMBER]"
    -- redacting the assistant's only call to action."""
    from rag.guardrails.pii_detector import PIIGuardrail

    guardrail = PIIGuardrail(PIIConfig(enabled=True, strategy="anonymize"))
    out = guardrail.scrub_response(
        "Call us at 1-800-889-0232 or 1-800-411-1113. "
        "Ask for Sarah Jenkins on 415-555-0134."
    )

    assert "1-800-889-0232" in out and "1-800-411-1113" in out
    assert "415-555-0134" not in out


def test_both_guards_no_op_when_pii_is_switched_off():
    config = AgentConfig()
    config.pii = PIIConfig(enabled=False)
    text = "Call 1-800-889-0232 or 415-555-0134"
    assert apply_pii_query_guard(text, config) == text
    assert apply_pii_response_guard(text, config) == text


def test_ingest_time_scrubbing_stays_off():
    """Scrubbing this corpus at ingest redacted the company's own number, turned
    "About Us" into "About [REDACTED_LOCATION]", and damaged 120 of 798 docs.
    Checkpoints 2 and 3 stay on; checkpoint 1 must stay off by default."""
    assert PIIConfig().scrub_on_ingest is False


def test_a_typo_in_the_strategy_name_cannot_silently_disable_pii_guarding():
    """"redact" is not a strategy, and setting it disabled PII guarding
    entirely rather than failing."""
    assert "redact" not in VALID_PII_STRATEGIES
    assert VALID_PII_STRATEGIES == {"anonymize", "block"}


# ── KB tagging (no retrieval) ───────────────────────────────────────────────

def _kb_records():
    with open(KB_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_every_record_carries_the_tags_the_format_layer_enforces_voice_with():
    """A missing tag silently downgrades a scope guard to an ordinary chunk --
    the LLM then answers a fee question with whatever the text implies."""
    untagged = [
        r["id"] for r in _kb_records()
        if not r.get("authority") or not r.get("answer_policy")
    ]
    assert not untagged, f"untagged records: {untagged[:5]}"


def test_the_scope_guards_are_still_in_the_corpus():
    """Correct refusals are records here, not prompt rules. If a KB rebuild
    drops them, fee/savings/legal questions stop being deflected and start
    being answered."""
    guards = [r for r in _kb_records() if r.get("answer_policy") == "deflect"]
    assert len(guards) >= 8, f"only {len(guards)} scope-guard records left"


# ── Context markers, over real retrieval ────────────────────────────────────

# Queries chosen to pull each flagged category out of the live corpus: our own
# canonical answers, third-party educational material, deflection topics, and
# records the scrapers flagged as outcome/regulated claims.
MARKER_SWEEP = [
    "what services do you offer",
    "what do you charge",
    "how much will I save",
    "what is a merchant cash advance",
    "how does the program work",
    "do i have to file bankruptcy",
    "can you help with payroll taxes",
    "creditors keep calling me all day what do i do",
    "what is a confession of judgment",
    "are you available in my state",
    "can i get sued by my mca lender",
    "i got three mca advances and my sales dropped hard",
]


@pytest.fixture(scope="module")
def formatted_contexts():
    """(chunk, formatted block) for every chunk the sweep actually sends to the
    model. One retrieval pass, shared by the marker checks below."""
    if not os.path.isdir(os.path.join(ROOT, "chroma_db")):
        pytest.skip("no ingested store -- run scripts/ingest.py --loader enriched --force")
    from rag.pipeline import get_pipeline

    pipeline = get_pipeline(AgentConfig())
    pairs = []
    for query in MARKER_SWEEP:
        for chunk in pipeline.retrieve(query):
            pairs.append((query, chunk, pipeline.format_for_llm([chunk])))
    return pairs


@pytest.mark.corpus
def test_the_sweep_is_not_vacuous(formatted_contexts):
    """Guards the three checks below: if retrieval stopped returning flagged
    content they would pass by having nothing to check."""
    seen = collections.Counter()
    for _, chunk, _ in formatted_contexts:
        meta = chunk.document.metadata
        seen["background"] += meta.get("authority") == "background"
        seen["deflect"] += meta.get("answer_policy") == "deflect"
        seen["disclaimer"] += bool(meta.get("requires_disclaimer"))
    assert seen["background"] and seen["deflect"] and seen["disclaimer"], seen


@pytest.mark.corpus
def test_third_party_material_is_never_handed_over_as_our_own_voice(formatted_contexts):
    """270 of the corpus's records are consumer-finance articles, IRS pages and
    SBA guidance. Unmarked, the model states their figures and thresholds as
    Corporate Turnaround's advice."""
    for query, chunk, block in formatted_contexts:
        if chunk.document.metadata.get("authority") == "background":
            assert "third-party educational material" in block, (
                f"{query!r} -> {chunk.document.metadata.get('record_id')} "
                f"reached the model unmarked"
            )


@pytest.mark.corpus
def test_restricted_topics_arrive_with_their_restriction_attached(formatted_contexts):
    """Fees, savings estimates and legal conclusions are answerable only to the
    extent the canonical deflection says. The record alone reads like an
    ordinary answer; the POLICY tag is what stops the model elaborating."""
    for query, chunk, block in formatted_contexts:
        if chunk.document.metadata.get("answer_policy") == "deflect":
            assert "[POLICY: Restricted topic." in block, (
                f"{query!r} -> deflection record sent without its policy tag"
            )


@pytest.mark.corpus
def test_outcome_claims_arrive_with_their_compliance_warning(formatted_contexts):
    """The scrapers flagged these and nothing read the flag for a while, so
    specific outcomes reached the model indistinguishable from explanation --
    which is how a past result becomes an implied promise."""
    for query, chunk, block in formatted_contexts:
        if chunk.document.metadata.get("requires_disclaimer"):
            assert "COMPLIANCE:" in block, (
                f"{query!r} -> outcome claim sent without its disclaimer"
            )


@pytest.mark.corpus
def test_the_context_wrapper_the_system_prompt_names_actually_exists(formatted_contexts):
    """The prompt tells the model its background material arrives under
    <retrieved_context>. For a while nothing emitted that tag, so the
    instruction referred to something that never existed."""
    for _, _, block in formatted_contexts:
        assert "<retrieved_context>" in block and "</retrieved_context>" in block
