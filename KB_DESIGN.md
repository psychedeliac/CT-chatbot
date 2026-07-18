# Knowledge Base v2 — Design & Curation (2026-07-19)

How the KB is structured so the chatbot cannot produce mishits, out-of-scope
answers, or wrong information — modeled on how the best AI support stacks
(Intercom Fin, Decagon, Sierra) treat knowledge as tiered, policied content
rather than a flat document pile.

## The problem with KB v1 (593 records)

- **63% of the corpus was third-party content** (329 educational + 43
  regulatory chunks) written *for consumers*, voiced by the bot as if it were
  Corporate Turnaround's own advice — HELOCs on your home, personal credit
  repair, debt after death, IRS Form 656 details.
- **133 near-identical testimonials** (all titled "Client Success Story /
  Debt Resolution Case Example") flooding the candidate pool.
- **10 whole topic areas had zero answerable content** — every brainstormed
  visitor question about UCC liens, vendor debt, personal guarantees,
  business credit, out-of-court restructuring, creditor-call handling, and
  CT's own services overview failed groundedness (see
  `data/raw/qa_unanswerable.json`: 153 real-style questions with no answer).
- No metadata told retrieval or the LLM whose voice a chunk speaks in or how
  far an answer may go.

## KB v2 structure (402 records)

Every record now carries two fields, flowing loader → Chroma/BM25 metadata →
`format_for_llm` → the LLM's context:

**`authority` — whose voice is this?**

| Tier | Count | Meaning | Treatment |
|---|---|---|---|
| `canonical` | 80 | Hand-authored Q&A in CT's voice; the text IS the approved answer | Widened retrieval rescue window (`rerank_canonical_rescue_rrf_rank_cutoff`) |
| `company` | 34 | Verbatim first-party site/services content | Used as-is |
| `evidence` | 18 | Curated client stories | Compliance-flagged (results vary) |
| `background` | 270 | Third-party educational/regulatory | Tagged inline: "third-party material — do not present as CT's own advice or quote its figures as recommendations" |

**`answer_policy` — how far may an answer go?**

| Policy | Count | Meaning |
|---|---|---|
| `answer` | 281 | Answer freely from the chunk |
| `careful` | 112 | Compliance marker added (outcome claims, figures) |
| `deflect` | 8 | Scope guard: reply with the reference deflection only — no fees, percentages, timeframes, legal conclusions |
| `route_client_line` | 1 | Existing-client routing (1-800-411-1113) |

## Scope guards: negative knowledge as records

The industry-standard fix for "RAG retrieves whatever is semantically
closest" is to make the *correct refusal itself retrievable*. Eight guard
records (`category: ct-scope`) cover the known trap topics — fees, savings/
timeframes, personal debt, personal taxes, legal advice, "should I stop
paying?", "should I file bankruptcy?", state availability. A trap query now
*hits* a chunk whose text is the approved deflection, tagged `[POLICY:
Restricted topic…]`, instead of missing and letting the model improvise or
pull adjacent third-party content.

## Canonical records with question variants

Each new canonical record bakes 3–10 real user phrasings (harvested from the
unanswerable/brainstormed question sets) into the embedded text as
"Also asked:" lines — one record covers a whole cluster of informal,
misspelled, panicked phrasings for both BM25 and the embedder. 26 new
records close the 10 zero-coverage topics plus company facts (track record,
guarantee, program mechanics, qualification, DIY-vs-CT, not-a-loan, contact,
confidentiality).

Grounding discipline: every fact traces to first-party scraped content
(`data/raw/ct_site.json`, `ct_services.json`). The four LLM-generated Q&As
with arithmetic errors (factor-rate math) remain excluded
(`data/raw/qa_arithmetic_flagged.json`).

## What was removed

- **102 chunks / 15 documents of wrong-audience consumer content** (HELOC,
  personal credit cards, charge-offs on personal credit reports, debt after
  death, "how much Americans owe", psychology of debt…). These were the raw
  material of adjacent-but-wrong answers.
- **115 redundant testimonials** — 133 curated to 18, two per theme across 9
  themes (MCA, tax, vendor, lawsuit, harassment, hardship, bankruptcy
  avoided, payment plan, settled), retitled so they're distinguishable in
  retrieval and reporting.

## Defense in depth (why wrong answers are structurally hard)

1. **Corpus**: wrong-audience content deleted; math-error QAs excluded.
2. **Retrieval**: hybrid RRF + rerank + calibrated abstain gate; canonical
   records get a wider rescue window so guards win trap queries.
3. **Chunk tags**: `[COMPLIANCE]`, `[SOURCE: third-party…]`, `[POLICY:
   Restricted topic…]` emitted inline by `rag/pipeline.format_for_llm`.
4. **Prompt**: context-aware deflection rules (config.py).
5. **Deterministic backstop**: `enforce_grounding_refusal` still replaces
   ungrounded substantive answers after empty retrieval.

## Results (77-query eval, `scripts/eval_retrieval.py`)

| Metric | KB v1 | KB v2 |
|---|---|---|
| Recall (in-domain) | 0.98 | **1.00** |
| Precision | 0.98 | **1.00** |
| Accuracy | 0.97 | **1.00** |
| Out-of-domain FP rate | 0.00 | 0.00 |
| Eval set size | 61 | 77 (+16 gap/guard queries) |
| Guardrail tests | 11 | 17, all passing |

## Maintenance workflow

1. Edit `data/canonical_qa_v2.json` (new canonical answers / variants) —
   never hand-edit `enriched_knowledge_base.json` for canonical content.
2. `python scripts/build_kb_v2.py` (idempotent; re-applies curation + merge).
3. `python scripts/ingest.py --loader enriched --force` (Chroma does not
   auto-sync).
4. `python scripts/eval_retrieval.py` + `python tests/test_guardrails.py`.
5. Gap-driven iteration: mine unanswered/abstained production queries into
   new canonical records or scope guards — that's the loop Intercom Fin and
   Decagon teams run weekly.

Sources consulted: [Fin AI knowledge-base guide](https://fin.ai/learn/ai-knowledge-base),
[Decagon platform guides](https://www.getmacha.com/blog/decagon-ai-complete-guide),
[Contentful on RAG hallucinations & structured metadata](https://www.contentful.com/blog/rag-hallucinations-structured-data-fix/),
[Towards Data Science: preventing RAG hallucinations](https://towardsdatascience.com/5-techniques-to-prevent-hallucinations-in-your-rag-question-answering/),
[CFPB: Chatbots in consumer finance](https://www.consumerfinance.gov/data-research/research-reports/chatbots-in-consumer-finance/chatbots-in-consumer-finance/).
