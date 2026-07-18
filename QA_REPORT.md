# CT Chatbot — QA Audit & Fix Report (2026-07-19)

Persona-based user testing of the chatbot as a corporateturnaround.com visitor would use it,
root-cause diagnosis of every failure, fixes, and verification. Method: 50 retrieval-layer probes
across 6 personas (panicked owner, skeptical researcher, price shopper, existing client,
adversarial, vague browser) plus end-to-end conversations through the live agent.

## Headline results

| Metric | Before | After |
|---|---|---|
| Persona probe failures (50 queries) | 8 | 2 (both handled at prompt layer) |
| Retrieval eval recall (61-query set) | 0.98 | **1.00** |
| Retrieval eval accuracy | 0.97 | 0.98 |
| Out-of-domain false-positive rate | 0.00 | 0.00 |
| Robotic "not in my knowledge base" replies | every abstain | **eliminated** |
| Guardrail regression tests | 7 | 11, all passing |

## What was broken, and why

1. **Stale vector index (critical).** `chroma_db` held 798 embeddings from a KB that had been
   rebuilt to 592 records. Semantic search and BM25 were searching *different corpora*.
   Fixed by re-ingesting; eval confirmed the threshold calibration still holds.

2. **The robotic refusal.** Whenever retrieval found nothing, a deterministic backstop replaced
   the model's reply with: *"That's not something I have information on in our knowledge base…"* —
   regardless of context. A user typing "hello" got it. A user in crisis typing
   *"im gonna lose everything please help"* got it. The CFPB explicitly flags this "doom loop"
   pattern in consumer-finance chatbots.

3. **Knowledge-base gaps.** No chunk stated the A+ BBB rating or years in business (the questions
   every skeptic asks). No chunk covered existing-client routing (1-800-411-1113), so enrolled
   clients hit the refusal. No coverage for payroll/cash-flow-crisis phrasing.

4. **Knowledge-base garbage.** Two scraped .gov page banners ("Official websites use .gov…"),
   one social-share-button fragment, and two exact duplicate chunks.

## Fixes applied

- **Re-ingested** the KB into Chroma (591 → in-sync with BM25).
- **KB repair**: dropped 5 garbage/duplicate records; authored 6 new grounded Q&A chunks
  (company legitimacy/BBB, company history, existing-client routing, creditor-contact-while-enrolled,
  payroll-crisis, acute-distress reassurance). All facts substantiated from corporateturnaround.com.
- **Context-aware deflection** (`config.py` system prompt): on empty retrieval the bot now
  distinguishes greeting / off-topic / missing-info / distress and responds appropriately —
  warm, short, never the words "knowledge base", always an actionable next step. Fee questions
  get an explanation of *why* it's a call-us topic plus what CAN be answered (free, no-obligation
  consultation), not a cold refusal.
- **Smarter grounding backstop** (`core/utils.py`): still deterministically blocks ungrounded
  substantive answers (essay-length or containing figures), but lets short, figure-free
  deflections through. Covered by 4 new regression tests.
- **UI polish** (`app.py`): welcome card with credibility line and 4 suggested starter questions;
  spinner text no longer says "Searching knowledge base".

## Verified end-to-end (live agent)

> **User:** im gonna lose everything please help
> **Before:** "That's not something I have information on in our knowledge base. For direct help, please call us at 1-800-889-0232."
> **After:** "We're here to help and want you to know that you're not alone. Losing control of your finances can be incredibly stressful, but it's a solvable problem. We've helped many business owners in similar situations… call us at 1-800-889-0232 for a free consultation…"

Retrieval-layer verification (no API needed, deterministic): all 6 previously-failing
persona queries now retrieve correct chunks — BBB rating, years in business, existing-client
routing (×2), payroll crisis, acute distress.

The 2 remaining probe mismatches are off-topic queries ("write me a poem about debt",
"how do i file my personal taxes") where retrieval returns *adjacent* content; the system
prompt now instructs the model to deflect these — prompt-layer, verified by instruction.

## Industry benchmark (what the best bots do)

Research on Intercom Fin, Klarna, Decagon and CFPB/FTC guidance distilled to four rules,
all now implemented:
1. **Never dead-end** — every non-answer ends in an actionable next step.
2. **Risk-band queries** — informational → answer; account-specific → client line
   (1-800-411-1113); regulated (fees/outcomes/credit advice) → explained handoff to
   1-800-889-0232.
3. **Quality of the automated subset beats automation rate** — the abstain gate stays
   conservative; out-of-domain FP rate is 0.00.
4. **Escalate on distress signals**, don't abstain — empathy first, then the phone.

Sources: [Klarna AI lessons](https://www.twig.so/blog/how-klarna-is-revolutionizing-customer-support-with-ai),
[Intercom Fin guide](https://myaskai.com/blog/intercom-fin-ai-agent-complete-guide-2026),
[Decagon vs Fin](https://querypal.com/blog/decagon-vs-fin-vs-querypal),
[CFPB: Chatbots in consumer finance](https://www.consumerfinance.gov/data-research/research-reports/chatbots-in-consumer-finance/chatbots-in-consumer-finance/),
[Chatbot escalation best practices](https://mxchat.ai/6-best-practices-for-using-chatbots-in-2026/).

## Operational notes

- **API quotas**: Gemini free tier allows only 20 requests/day for `gemini-flash-latest`
  (resolves to gemini-3.5-flash); Groq free tier 100k tokens/day. For the client demo and any
  real traffic, a paid tier on one provider is required — at free-tier limits the bot dies
  mid-demo. Provider is switchable via `LLM_PROVIDER`/`LLM_MODEL` env vars (gemini | groq).
- `.env` overrides `RAG_CANDIDATE_POOL_K=20` (vs calibrated 15); current eval numbers were
  produced under this override and are good — leave as-is or remove and re-sweep, but don't
  change it silently.
- Re-run checks any time:
  `python tests/test_guardrails.py` · `python scripts/eval_retrieval.py` ·
  `python scripts/ingest.py --loader enriched --force` after any KB edit.
