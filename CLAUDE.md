# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RAG chatbot for Corporate Turnaround (business debt-settlement company). LangGraph ReAct agent + hybrid retrieval (Chroma semantic + BM25 keyword, RRF fusion, cross-encoder rerank) over a curated JSON knowledge base, served via Streamlit or a CLI loop.

## Commands

```bash
source venv/bin/activate                      # local venv at ./venv

# Ingest KB into Chroma (required before first run, and after ANY KB edit)
python scripts/ingest.py --loader enriched --force

# Run
streamlit run app.py                          # web UI (primary)
python main.py                                # CLI chat loop

# Tests / eval
pytest tests/                                 # unit tests (guardrails)
python scripts/smoke_test.py                  # end-to-end smoke (uses LLM API quota)
python scripts/eval_retrieval.py              # retrieval eval vs data/eval_retrieval_set.json
python scripts/diagnose_retrieval.py "query"  # inspect what retrieval returns for one query

# KB rebuild (raw sources → enriched_knowledge_base.json)
python scripts/build_kb_v2.py
```

Run pytest with a single test: `pytest tests/test_guardrails.py -k <name>`.

## Critical rules

- **Editing anything under `data/` changes nothing at runtime until you re-ingest** (`python scripts/ingest.py --loader enriched --force`). The agent reads only from `./chroma_db`.
- Gemini free-tier API quota is limited — don't loop LLM-calling scripts (smoke_test, synthesize_qa_answers) unnecessarily. Retrieval-only scripts (eval, diagnose) use no LLM quota.
- Secrets live in `.env` (`GEMINI_API_KEY`, optionally `GROQ_API_KEY`); `.env.example` documents them.
- The PII phone allowlist in `config.py` exists because Presidio otherwise redacts CT's own published phone numbers — don't remove it. Ingest-time PII scrubbing is deliberately off (see comments in `PIIConfig`).

## Architecture

Two independent phases: **ingestion** (offline, `scripts/ingest.py`) and **inference** (runtime, `app.py`/`main.py`). Detailed docs: `architecture.md` (retrieval mechanics), `KB_DESIGN.md` (KB v2 tiering), `QA_REPORT.md` (known failure modes).

- `config.py` — the single deployer touchpoint. `AgentConfig` dataclass + env-var overrides drive everything: LLM provider/model, embedding provider, retrieval knobs (`rag_k`, rerank threshold, canonical rescue cutoff), PII tiers, system prompt. Change behavior here, not in implementation files.
- `rag/pipeline.py` — `RetrievalPipeline`, the **single implementation of retrieval**. The LangChain tool (`rag/retriever.py`), diagnostics, and eval harness all call it — never reimplement Chroma/BM25/merge logic elsewhere; that drift is exactly what this module was created to fix.
- `core/factory.py` + `core/llms.py` — build the LangGraph ReAct agent with per-session `MemorySaver`; provider registries (gemini live, groq lazy-imported).
- `core/tools/registry.py` — tools registered by name (`register_tool("rag", ...)`); `main.py:_setup_rag_tool` wires retrieval and is reused by `app.py`.
- `core/utils.py` — response post-processing chain used by both UIs: `build_user_query`/`wrap_user_query` → PII query guard → `clean_response_prefix` → `enforce_grounding_refusal` (deterministic refusal backstop when rag_search returns nothing) → PII response guard.
- `data_handlers/` — loader registry pattern; `enriched_loader.py` reads `data/enriched_knowledge_base.json` and carries `authority`/`answer_policy` metadata into Chroma.
- `rag/guardrails/pii_detector.py` — Presidio-based, three checkpoints (ingest/query/output).
- `app.py` — Streamlit. Retrieval machinery is shared process-wide via `st.cache_resource` (BM25 build + cross-encoder load ≈ 40s); the agent stays per-session because its MemorySaver holds conversation state.

## Knowledge base (v2)

Every record in `data/enriched_knowledge_base.json` carries `authority` (canonical / company / evidence / background — whose voice) and `answer_policy` (answer / careful / deflect / route_client_line — how far an answer may go). These flow loader → vector-store metadata → LLM context. Scope guards are *records*: correct refusals made retrievable. Canonical Q&A gets a widened retrieval rescue window. Preserve both fields when adding or editing KB content; see `KB_DESIGN.md` before restructuring.

Raw scraped sources live in `data/raw/`; scrapers in `scripts/scrapers/`; `scripts/build_kb_v2.py` assembles the enriched KB from them.
