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
