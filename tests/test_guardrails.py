"""
tests/test_guardrails.py — Regression checks for the guardrail layer.

Run directly (no pytest needed):
    python tests/test_guardrails.py

These cover defects that actually shipped, not hypotheticals.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AgentConfig, PIIConfig, VALID_PII_STRATEGIES
from core.utils import clean_response_prefix, apply_pii_query_guard, apply_pii_response_guard


def test_company_phone_survives_pii_scrub() -> None:
    """
    The phone recognizer cannot tell the company's published line from a
    private number. Before the allowlist, answers rendered as "call us at
    [REDACTED_PHONE_NUMBER]" -- redacting the assistant's only call to action.
    """
    from rag.guardrails.pii_detector import PIIGuardrail

    config = PIIConfig(enabled=True, strategy="anonymize")
    guardrail = PIIGuardrail(config)

    out = guardrail.scrub_response(
        "Call us at 1-800-889-0232. Ask for Sarah Jenkins on 415-555-0134."
    )
    assert "1-800-889-0232" in out, f"company line was redacted: {out}"
    assert "415-555-0134" not in out, f"private number leaked: {out}"


def test_pii_guards_noop_when_disabled() -> None:
    config = AgentConfig()
    config.pii = PIIConfig(enabled=False)
    text = "Call 1-800-889-0232 or 415-555-0134"
    assert apply_pii_query_guard(text, config) == text
    assert apply_pii_response_guard(text, config) == text


def test_ingest_scrub_is_off_by_default() -> None:
    """
    Scrubbing this corpus at ingest redacted the company's own number, turned
    "About Us" into "About [REDACTED_LOCATION]", and damaged 120 of 798 docs.
    Checkpoints 2 and 3 stay on; checkpoint 1 must stay off by default.
    """
    assert PIIConfig().scrub_on_ingest is False


def test_invalid_pii_strategy_is_rejected() -> None:
    """"redact" is not a strategy; it silently disabled PII guarding entirely."""
    assert "redact" not in VALID_PII_STRATEGIES
    assert VALID_PII_STRATEGIES == {"anonymize", "block"}


def test_clean_response_prefix_strips_rag_preamble() -> None:
    assert clean_response_prefix("Based on the context, you have options.") == "You have options."
    assert clean_response_prefix("According to the documents, call us.") == "Call us."
    # Must not mangle a sentence that merely starts with a similar word.
    assert clean_response_prefix("Basing your plan on revenue is wise.").startswith("Basing")


def test_retrieved_context_wrapper_is_emitted() -> None:
    """
    The system prompt tells the model that background material arrives under
    <retrieved_context>. Nothing emitted that tag, so the instruction referred
    to something that never existed.
    """
    from langchain_core.documents import Document
    from rag.pipeline import RetrievalPipeline, RetrievedChunk

    chunk = RetrievedChunk(
        document=Document(page_content="Title: T\nSection: S\n\nBody text here.", metadata={}),
        rrf_rank=1,
        rerank_score=1.0,
    )
    out = RetrievalPipeline.format_for_llm(None, [chunk])  # type: ignore[arg-type]
    assert "<retrieved_context>" in out and "</retrieved_context>" in out


def test_compliance_marker_added_for_flagged_sources() -> None:
    from langchain_core.documents import Document
    from rag.pipeline import RetrievalPipeline, RetrievedChunk

    chunk = RetrievedChunk(
        document=Document(
            page_content="Title: T\nSection: S\n\nClient saved $40,000.",
            metadata={"requires_disclaimer": True},
        ),
        rrf_rank=1,
        rerank_score=1.0,
    )
    out = RetrievalPipeline.format_for_llm(None, [chunk])  # type: ignore[arg-type]
    assert "COMPLIANCE:" in out


def test_background_sources_are_marked_third_party() -> None:
    """
    KB v2: 270 of 402 records are third-party educational/regulatory content.
    Unmarked, the LLM voices consumer-finance articles and IRS form details as
    Corporate Turnaround's own advice. authority=background must add framing.
    """
    from langchain_core.documents import Document
    from rag.pipeline import RetrievalPipeline, RetrievedChunk

    chunk = RetrievedChunk(
        document=Document(
            page_content="Title: T\nSection: S\n\nYou can send a written dispute within 30 days.",
            metadata={"authority": "background"},
        ),
        rrf_rank=1,
        rerank_score=1.0,
    )
    out = RetrievalPipeline.format_for_llm(None, [chunk])  # type: ignore[arg-type]
    assert "third-party educational material" in out


def test_deflect_policy_tag_is_emitted() -> None:
    """Scope-guard records (fees, savings, legal/bankruptcy advice) carry
    answer_policy=deflect; the LLM must see the restriction inline."""
    from langchain_core.documents import Document
    from rag.pipeline import RetrievalPipeline, RetrievedChunk

    chunk = RetrievedChunk(
        document=Document(
            page_content="Title: Q&A: fees\nSection: S\n\nA: Fees depend on your situation.",
            metadata={"answer_policy": "deflect"},
        ),
        rrf_rank=1,
        rerank_score=1.0,
    )
    out = RetrievalPipeline.format_for_llm(None, [chunk])  # type: ignore[arg-type]
    assert "[POLICY: Restricted topic." in out


def test_kb_v2_has_no_untagged_records() -> None:
    """Every KB record must carry authority + answer_policy so the format
    layer can enforce voice and scope. A missing tag silently downgrades a
    scope guard to an ordinary chunk."""
    import json

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "enriched_knowledge_base.json")
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    untagged = [r["id"] for r in records
                if not r.get("authority") or not r.get("answer_policy")]
    assert not untagged, f"untagged records: {untagged[:5]}"
    guards = [r for r in records if r.get("answer_policy") == "deflect"]
    assert len(guards) >= 8, "scope-guard records missing from KB"


def _agent_response(rag_content: str | None) -> dict:
    """Minimal fake LangGraph response: one human turn, optionally one
    rag_search ToolMessage with the given content."""
    from langchain_core.messages import HumanMessage, ToolMessage

    messages: list = [HumanMessage(content="hi")]
    if rag_content is not None:
        messages.append(ToolMessage(content=rag_content, name="rag_search", tool_call_id="t1"))
    return {"messages": messages}


def test_grounding_backstop_lets_safe_deflections_through() -> None:
    """
    The old backstop replaced EVERY empty-retrieval reply with one canned
    refusal -- so 'hello' and 'I'm going to lose everything' both got a
    robotic 'not in my knowledge base' answer. Short, figure-free deflections
    (greeting, dodge, phone handoff) must survive.
    """
    from core.utils import enforce_grounding_refusal
    from rag.retriever import NO_RESULTS_MESSAGE

    response = _agent_response(NO_RESULTS_MESSAGE)
    greeting = "Hi there! I'm Corporate Turnaround's AI assistant. What's going on with your business?"
    handoff = "I don't have the specifics on that, but our team does -- call us at 1-800-889-0232, the consultation is free."
    assert enforce_grounding_refusal(response, greeting) == greeting
    assert enforce_grounding_refusal(response, handoff) == handoff


def test_grounding_backstop_blocks_ungrounded_answers() -> None:
    """A substantive parametric answer (figures, or essay-length) after empty
    retrieval must still be replaced with the canned refusal."""
    from core.utils import REFUSAL_MESSAGE, enforce_grounding_refusal
    from rag.retriever import NO_RESULTS_MESSAGE

    response = _agent_response(NO_RESULTS_MESSAGE)
    with_figures = "Debt settlement typically saves 40% and takes 24 months."
    essay = "Islam is a monotheistic religion. " * 30
    assert enforce_grounding_refusal(response, with_figures) == REFUSAL_MESSAGE
    assert enforce_grounding_refusal(response, essay) == REFUSAL_MESSAGE


def test_grounding_backstop_noop_when_retrieval_succeeded() -> None:
    from core.utils import enforce_grounding_refusal

    response = _agent_response("<retrieved_context>real content</retrieved_context>")
    answer = "We negotiate with your creditors within a budget you can afford, saving you $10,000."
    assert enforce_grounding_refusal(response, answer) == answer


def test_refusal_message_never_mentions_knowledge_base() -> None:
    """Users are talking to Corporate Turnaround, not to a search engine."""
    from core.utils import REFUSAL_MESSAGE

    assert "knowledge base" not in REFUSAL_MESSAGE.lower()
    assert "1-800-889-0232" in REFUSAL_MESSAGE


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    print(f"\n{'All guardrail checks passed.' if not failures else f'{failures} failure(s).'}")
    sys.exit(1 if failures else 0)
