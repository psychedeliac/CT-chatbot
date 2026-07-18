"""
scripts/contextualize_kb.py — Anthropic-style Contextual Retrieval pass.

Generates a short (~50-100 token) description situating each chunk inside its
parent document, and stores it on the record as `context_prefix`. EnrichedLoader
prepends that prefix to page_content at ingest, so both the embedding and the
BM25 index see the disambiguated text.

Why this matters here: chunks in this corpus routinely lose their referent.
A chunk reading "you repay it daily or weekly using a percentage of card sales"
never says "merchant cash advance", so a query for MCA repayment has nothing
lexical to match and only weak semantic signal. Reported effect of this
technique is ~35% fewer top-20 retrieval failures, ~49% combined with BM25,
and ~67% with reranking on top.

The raw chunk_text is never modified, so this is reversible: delete the
context_prefix fields and re-ingest to get the previous behaviour back.

Usage:
    python scripts/contextualize_kb.py --limit 10   # sample first
    python scripts/contextualize_kb.py              # full run (idempotent)
    python scripts/contextualize_kb.py --force      # regenerate everything
    python scripts/contextualize_kb.py --clear      # strip all prefixes
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config import load_config

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "enriched_knowledge_base.json",
)

# Keep the per-call document window bounded so cost stays predictable, and so a
# whole-corpus pass fits inside free-tier tokens-per-minute limits. 6000 chars
# (~1500 tokens/call) exhausted Groq's 12k TPM after ~80 records.
MAX_DOC_CHARS = 2000
MAX_WORKERS = 3
MAX_RETRIES = 6
PROMPT = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Give a short, succinct context (one or two sentences, under 60 words) that situates
this chunk within the overall document, naming the specific product, program, or legal
process it refers to (for example "merchant cash advance", "Chapter 11 bankruptcy",
"IRS offer in compromise") so the chunk can be found by search on its own.

Rules:
- State the subject matter directly. Do NOT hedge with "likely", "probably",
  "appears to be" or "this chunk is from".
- Do NOT guess or name a publisher, agency or company unless that name actually
  appears in the document text above. Wrongly attributing content to an agency is
  worse than omitting the source entirely.
- Describe only what the text actually says.

Answer ONLY with the context, no preamble."""


# Boilerplate openers the model emits despite being told not to. Stripping these
# is a pure text transform -- no extra API calls -- and worth doing: the phrase
# "This chunk describes" appeared verbatim in 96% of prefixes, contributing zero
# discriminative signal to the embedding while adding "chunk"/"describes" as
# high-frequency noise terms to the BM25 index.
_META_OPENER = re.compile(
    r"^\s*(this\s+(chunk|section|document|text|passage)\s+"
    r"(is\s+)?(likely\s+|probably\s+)?"
    r"(describes|refers\s+to|discusses|is\s+about|outlines|explains|details|covers|"
    r"provides|contains|appears\s+to\s+be|is\s+part\s+of|is\s+from|is\s+found\s+in)\s*"
    r"(that\s+)?)",
    re.IGNORECASE,
)
_HEDGES = re.compile(r"\b(likely|probably|appears to be|seems to be)\s+", re.IGNORECASE)


def clean_prefix(text: str) -> str:
    """Strip meta-boilerplate and hedging so the prefix reads as plain subject matter."""
    if not text:
        return text
    cleaned = _META_OPENER.sub("", text)
    cleaned = _HEDGES.sub("", cleaned)
    cleaned = " ".join(cleaned.split())
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _save(path: str, records: list[dict]) -> None:
    """Write via a temp file + rename so an interrupted write can't truncate the KB."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def body_of(record: dict) -> str:
    text = record.get("chunk_text", "")
    if not text and "customer_problem" in record:
        text = (
            f"Customer Problem: {record.get('customer_problem', '')}\n"
            f"Solution: {record.get('ct_solution', '')}\n"
            f"Outcome: {record.get('outcome', '')}"
        )
    return text or record.get("content", "")


def build_documents(records: list[dict]) -> dict[str, str]:
    """Reconstruct a rough parent document per topic to give the model context."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        key = record.get("topic") or record.get("source_type") or "misc"
        grouped[key].append(body_of(record))
    return {k: " ".join(v)[:MAX_DOC_CHARS] for k, v in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Add contextual retrieval prefixes to the KB.")
    parser.add_argument("--limit", type=int, help="Only process the first N records (sampling).")
    parser.add_argument("--force", action="store_true", help="Regenerate prefixes that already exist.")
    parser.add_argument("--clear", action="store_true", help="Remove all context_prefix fields and exit.")
    parser.add_argument("--path", default=DATA_PATH)
    parser.add_argument(
        "--provider",
        choices=["groq", "gemini"],
        default="groq",
        help=(
            "Which LLM runs the batch. Defaults to groq: this is a one-off offline "
            "job over the whole corpus, and the Gemini free tier allows only 20 "
            "requests PER DAY, which cannot cover a 592-chunk pass (and would eat "
            "the quota the live chatbot needs)."
        ),
    )
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.clear:
        cleared = [{k: v for k, v in r.items() if k != "context_prefix"} for r in records]
        _save(args.path, cleared)
        print(f"Cleared context_prefix from {len(cleared)} records.")
        return

    config = load_config()
    if args.provider == "groq":
        from core.llms import GroqProvider
        llm = GroqProvider(model_name="llama-3.1-8b-instant", temperature=0.0).get_llm()
    else:
        from core.llms import GeminiProvider
        llm = GeminiProvider(model_name=config.llm_model, temperature=0.0).get_llm()
    print(f"Using provider: {args.provider}")

    documents = build_documents(records)
    targets = [
        (i, r) for i, r in enumerate(records)
        if (args.force or not r.get("context_prefix")) and body_of(r).strip()
    ]
    if args.limit:
        targets = targets[: args.limit]

    print(f"Contextualizing {len(targets)} of {len(records)} records "
          f"({len(records) - len(targets)} already done or empty)...")

    lock = threading.Lock()
    done = {"n": 0, "failed": 0}

    def work(item):
        index, record = item
        key = record.get("topic") or record.get("source_type") or "misc"
        prompt = PROMPT.format(document=documents.get(key, ""), chunk=body_of(record))

        # Free-tier TPM limits are hit constantly on a whole-corpus pass, and a
        # rate-limit rejection is retryable, not a failure. Back off and retry
        # rather than silently dropping the record's context.
        context = None
        for attempt in range(MAX_RETRIES):
            try:
                raw = llm.invoke(prompt).content
                if isinstance(raw, list):
                    raw = "".join(
                        p["text"] if isinstance(p, dict) and "text" in p else str(p) for p in raw
                    )
                context = clean_prefix(" ".join(str(raw).split()))
                break
            except Exception as exc:
                retryable = "429" in str(exc) or "rate_limit" in str(exc).lower()
                if not retryable or attempt == MAX_RETRIES - 1:
                    with lock:
                        done["failed"] += 1
                        if done["failed"] <= 3:
                            print(f"  [warn] record {record.get('id')}: {type(exc).__name__}: {exc}")
                    return
                time.sleep(min(2 ** attempt + 1, 30))

        if not context:
            with lock:
                done["failed"] += 1
            return
        with lock:
            records[index] = {**record, "context_prefix": context}
            done["n"] += 1
            if done["n"] % 25 == 0:
                # Checkpoint to disk. Free-tier rate limits stretch a full pass
                # over tens of minutes; writing only at the end meant a Ctrl-C
                # or a crash threw away every call made so far. Since the run is
                # idempotent (records with a context_prefix are skipped), a
                # partial write makes the job resumable by just re-running it.
                _save(args.path, records)
                print(f"  {done['n']}/{len(targets)} (checkpointed)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(work, targets))

    _save(args.path, records)

    print(f"\nDone. {done['n']} contextualized, {done['failed']} failed.")
    if done["n"]:
        sample = next((r for r in records if r.get("context_prefix")), None)
        if sample:
            print(f"\nSample prefix:\n  {sample['context_prefix'][:220]}")
    print("\nNext: python scripts/ingest.py --loader enriched --force")


if __name__ == "__main__":
    main()
