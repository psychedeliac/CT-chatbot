"""
tests/test_retrieval_contract.py — what retrieval owes the LLM, checked against
the REAL corpus.

Why this file exists: the unit tests around retrieval all built their own
Document objects, so they asserted that the code does what the code does. They
passed at 1.00 eval accuracy for weeks while the answer-less `Also asked:`
sibling of every canonical record was outscoring its own answer chunk and
evicting it -- the LLM was handed a list of questions with no answer, and
nothing in the suite could see it, because the eval harness only checks WHICH
record was retrieved, never whether the text handed over contains an answer.

So these assert properties of the bytes that actually reach the LLM, over the
live Chroma store and the live reranker. Every check here is phrased against
the contract ("no chunk may reach the LLM without answer text"), never against
an implementation detail -- a rewrite of dedupe, chunking, or gating that keeps
the contract must keep these green.

No LLM: retrieval only, so this costs no API quota. It does load the embedding
model, BM25 index, and cross-encoder once (~20s). Skip with `-m "not corpus"`.

Requires an ingested store: python scripts/ingest.py --loader enriched --force
"""
import json
import os
import re

import pytest

from config import AgentConfig
from rag.pipeline import get_pipeline

pytestmark = pytest.mark.corpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(ROOT, "data", "enriched_knowledge_base.json")

# Real questions in the shapes people actually type: plain, formal, informal,
# and mid-conversation follow-ups (which arrive concatenated with the previous
# turn -- see core.rag_chat.build_retrieval_query).
IN_DOMAIN_QUERIES = [
    "what services do you offer",
    "tell me all the services you provide, not just MCAs",
    "Do you handle SBA loans?",
    "can you help with payroll taxes",
    "what do you charge",
    "how does the program work",
    "are you available in my state",
    "i got three mca advances and my sales dropped hard",
    "creditors keep calling me all day what do i do",
    "do i have to file bankruptcy",
    "Do you help with merchant cash advances? how do I get out of it?",
    "what happens during the free consultation",
]

# Nothing in a business-debt KB answers these. Retrieval must abstain rather
# than hand the LLM the nearest debt chunk it can find.
OUT_OF_DOMAIN_QUERIES = [
    "tell me a joke",
    "what is the capital of France",
    "how do I fix a leaking kitchen tap",
    "write me a python script to sort a list",
]

# (query, terms that must appear in the text handed to the LLM). These are the
# substance of the answer, not the record's title or its question phrasings --
# a variants-only chunk, a header-only chunk, or the wrong record all fail.
ANSWER_SUBSTANCE = [
    ("what services do you offer", ["restructuring", "tax"]),
    ("tell me all the services you provide, not just MCAs", ["restructuring", "tax"]),
    ("what do you charge", ["free"]),
    ("can you help with payroll taxes", ["tax"]),
    ("are you available in my state", ["not necessarily available", "1-800-889-0232"]),
]


@pytest.fixture(scope="module")
def pipeline():
    if not os.path.isdir(os.path.join(ROOT, "chroma_db")):
        pytest.skip("no ingested store -- run scripts/ingest.py --loader enriched --force")
    return get_pipeline(AgentConfig())


def answer_text(chunk) -> str:
    """The chunk's body with its retrieval scaffolding removed: the Title/
    Section header, the Contextual Retrieval prefix's `Also asked:` lines, and
    blank lines. What is left is what the LLM can actually answer from."""
    lines = [ln.strip() for ln in chunk.document.page_content.splitlines() if ln.strip()]
    return "\n".join(
        ln for ln in lines
        if not ln.startswith(("Also asked:", "Title:", "Section:"))
    )


# ── The contract ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query", IN_DOMAIN_QUERIES)
def test_every_chunk_reaching_the_llm_carries_answer_text(pipeline, query):
    """No context slot may be spent on a chunk the LLM cannot answer from.

    This is the check that was missing. A chunk of nothing but `Also asked:`
    phrasings is retrieval telling the model "here is the question you were
    asked", which is worse than returning nothing -- the model then improvises
    or gives the one-line non-answer that was reported from production.
    """
    for chunk in pipeline.retrieve(query):
        body = answer_text(chunk)
        assert len(body) >= 40, (
            f"{query!r} -> chunk {chunk.document.metadata.get('record_id')} "
            f"reached the LLM with no answer text: {chunk.document.page_content[:120]!r}"
        )


@pytest.mark.parametrize("query", IN_DOMAIN_QUERIES)
def test_one_record_never_occupies_two_context_slots(pipeline, query):
    """rag_k is small; a record split across chunks must not spend two of them
    on itself and crowd out a different source."""
    ids = [c.document.metadata.get("record_id") for c in pipeline.retrieve(query)]
    present = [i for i in ids if i]
    assert len(present) == len(set(present)), f"{query!r} -> duplicate records: {ids}"


@pytest.mark.parametrize("query,terms", ANSWER_SUBSTANCE)
def test_the_answer_itself_is_what_gets_handed_over(pipeline, query, terms):
    """Retrieving the right RECORD is not the same as handing over its ANSWER.
    The eval harness checks the former; this checks the latter."""
    context = pipeline.format_for_llm(pipeline.retrieve(query)).lower()
    missing = [t for t in terms if t.lower() not in context]
    assert not missing, f"{query!r} -> answer substance missing from context: {missing}"


def test_a_services_question_gets_more_than_one_service_line(pipeline):
    """The reported production defect: asked for every service, the widget
    named one. The context has to contain the whole service list for the model
    to have any chance of listing it."""
    context = pipeline.format_for_llm(pipeline.retrieve("what services do you offer")).lower()
    service_lines = [
        "negotiat",       # business debt negotiation and settlement
        "merchant cash advance",
        "restructuring",
        "creditor harassment",
        "sba",
        "tax",
    ]
    found = [s for s in service_lines if s in context]
    assert len(found) >= 5, f"only {found} of {service_lines} in the context"


@pytest.mark.parametrize("query", OUT_OF_DOMAIN_QUERIES)
def test_out_of_domain_queries_get_no_context_at_all(pipeline, query):
    """An empty return is what triggers the grounded=False refusal path. A
    near-miss debt chunk here is how off-topic questions get answered with
    confident debt advice."""
    assert pipeline.retrieve(query) == [], f"{query!r} pulled context it should not have"


def test_formatted_context_has_a_body_under_every_header(pipeline):
    """format_for_llm emits Title/Section headers per chunk. A block whose
    header is followed by nothing is a slot the model reads as an empty source."""
    context = pipeline.format_for_llm(pipeline.retrieve("how does the program work"))
    blocks = [b.strip() for b in context.split("Title: ") if b.strip()]
    assert blocks, "no context produced for a core in-domain question"
    for block in blocks:
        body = "\n".join(
            ln for ln in block.splitlines()[1:]
            if ln.strip() and not ln.startswith(("Section:", "Similar past case", "Relevance"))
        )
        assert len(body.strip()) >= 40, f"header with no body:\n{block[:200]}"


# ── Corpus invariants (no retrieval, no model load) ─────────────────────────

def test_every_canonical_record_has_an_answer_a_chunk_can_carry():
    """A canonical record whose variant list dwarfs its answer chunks badly:
    the variants block splits off, and if the answer is thin there is nothing
    substantial left to retrieve. Catches the problem at KB-authoring time
    rather than in production."""
    with open(KB_PATH, encoding="utf-8") as f:
        records = json.load(f)

    canonical = [r for r in records if r.get("authority") == "canonical"]
    assert canonical, "no canonical records in the KB -- did build_kb_v2.py run?"

    thin = []
    for record in canonical:
        body = re.sub(r"^Also asked:.*$", "", record.get("chunk_text", ""), flags=re.M)
        if len(body.strip()) < 120:
            thin.append(record["id"])
    assert not thin, f"canonical records with no substantive answer text: {thin}"


# ── Plain phrasing must reach our own voice ─────────────────────────────────

# Questions about US. The canonical Q&A was authored almost entirely in one
# voice -- urgent, first-person, misspelled -- so the calmest phrasing of the
# same question matched third-party articles instead, which is how "what
# services do you offer" missed its own answer record. Asked plainly, a
# question about Corporate Turnaround must still reach Corporate Turnaround's
# own words.
PLAIN_QUESTIONS_ABOUT_US = [
    "what services do you offer",
    "what does corporate turnaround do",
    "how does the program work",
    "what happens during the free consultation",
    "who is this program for",
    "are you a legitimate company",
    "what do you charge",
]


@pytest.mark.parametrize("query", PLAIN_QUESTIONS_ABOUT_US)
def test_plain_questions_about_us_reach_our_own_voice(pipeline, query):
    """Answered from `authority=background` alone, the model voices NerdWallet
    and the IRS as Corporate Turnaround's own policy -- or, as reported from
    production, gives a one-line non-answer because nothing authoritative was
    in front of it."""
    authorities = [
        c.document.metadata.get("authority") for c in pipeline.retrieve(query)
    ]
    assert any(a in ("canonical", "company") for a in authorities), (
        f"{query!r} -> only third-party sources retrieved: {authorities}"
    )
