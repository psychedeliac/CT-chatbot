"""
scripts/audit_kb_gaps.py — Read-only knowledge-base coverage audit.

Runs a formal + informal probe query per known content-gap topic (UCC liens,
personal guarantees, MCA negotiation, SBA default/OIC, business bankruptcy
types, out-of-court restructuring, CT fee/timeline specifics — see
implementation_plan.md) through the same RetrievalPipeline the live agent
uses, and classifies each topic as MISSING / WEAK / COVERED.

Does NOT modify any pipeline code or data. Safe to run repeatedly — run once
before adding new scraped content / QA pairs, then again after, to confirm
gap topics moved to COVERED.

Usage:
    python scripts/audit_kb_gaps.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config import load_config

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SKIPPED_URLS_PATH = os.path.join(DATA_DIR, "raw", "skipped_urls.txt")


def build_gap_probe_queries() -> list[dict]:
    """One formal + one informal probe query per known gap topic."""
    topics = [
        ("ucc_liens", "what is a UCC-1 lien and how do MCA lenders use it",
         "they say they have a lien on everything i own is that even real"),
        ("personal_guarantee", "what does signing a personal guarantee mean for a business owner",
         "did i just become personally responsible for my businesses debt"),
        ("mca_negotiation", "can a merchant cash advance be negotiated or settled",
         "can i actually get my mca lender to lower what i owe"),
        ("sba_default", "what happens if a business defaults on an SBA loan",
         "i missed payments on my sba loan what happens now"),
        ("business_bankruptcy", "what is the difference between chapter 7 and chapter 11 bankruptcy",
         "should my business just file for bankruptcy at this point"),
        ("out_of_court_restructuring", "what is out-of-court business debt restructuring",
         "is there a way to fix this without going to court"),
        ("ct_fees_timeline", "how does Corporate Turnaround charge for its services and how long does the program take",
         "how much does corporate turnaround cost and how long till im done"),
        ("confession_of_judgment", "what is a confession of judgment in an MCA contract",
         "they froze my bank account with no warning is that legal"),
    ]
    queries = []
    for topic_id, formal, informal in topics:
        queries.append({"topic_id": topic_id, "query": formal, "style": "formal"})
        queries.append({"topic_id": topic_id, "query": informal, "style": "informal"})
    return queries


def probe(pipeline, queries: list[dict]) -> list[dict]:
    results = []
    for item in queries:
        trace = pipeline.retrieve_with_trace(item["query"])
        if not trace.final:
            status = "MISSING"
        else:
            # WEAK if every surviving chunk only passed via the rescue gate
            # (i.e. none cleared the primary high-confidence threshold).
            only_rescued = all(
                c.rerank_score is not None
                and c.rerank_score < pipeline.config.rerank_score_threshold
                for c in trace.final
            )
            status = "WEAK" if only_rescued else "COVERED"

        top = trace.final[0] if trace.final else None
        top_title = None
        if top:
            # Title lives in page_content ("Title: X\nSection: Y\n\n...body"),
            # not metadata — same parsing RetrievalPipeline.format_for_llm() does.
            lines = top.document.page_content.split("\n")
            top_title = lines[0].replace("Title: ", "").strip() if lines and lines[0].startswith("Title: ") else "?"

        results.append({
            "topic_id": item["topic_id"],
            "query": item["query"],
            "style": item["style"],
            "status": status,
            "n_hits": len(trace.final),
            "top_score": f"{top.rerank_score:.4f}" if top and top.rerank_score is not None else "N/A",
            "top_source_type": top.document.metadata.get("source", "?") if top else None,
            "top_title": top_title,
        })
    return results


def _status_rank(status: str) -> int:
    return {"MISSING": 0, "WEAK": 1, "COVERED": 2}[status]


def report(results: list[dict]) -> None:
    skipped_urls = set()
    if os.path.exists(SKIPPED_URLS_PATH):
        with open(SKIPPED_URLS_PATH, "r", encoding="utf-8") as f:
            skipped_urls = {line.strip() for line in f if line.strip()}

    print(f"\n{'='*90}")
    print("  Knowledge Base Gap Audit")
    print(f"{'='*90}\n")

    by_topic: dict[str, list[dict]] = {}
    for r in results:
        by_topic.setdefault(r["topic_id"], []).append(r)

    # Rank topics worst-first (MISSING before WEAK before COVERED).
    ordered_topics = sorted(
        by_topic.items(),
        key=lambda kv: min(_status_rank(r["status"]) for r in kv[1]),
    )

    for topic_id, items in ordered_topics:
        worst = min(items, key=lambda r: _status_rank(r["status"]))
        print(f"[{worst['status']:<8}] {topic_id}")
        for r in items:
            print(f"    ({r['style']:<8}) {r['query']!r}")
            print(f"        -> {r['n_hits']} hit(s), top_score={r['top_score']}, "
                  f"top_source_type={r['top_source_type']}, top_title={r['top_title']!r}")
        print()

    n_missing = sum(1 for r in results if r["status"] == "MISSING")
    n_weak = sum(1 for r in results if r["status"] == "WEAK")
    n_covered = sum(1 for r in results if r["status"] == "COVERED")
    print(f"{'-'*90}")
    print(f"  Totals: {n_missing} MISSING, {n_weak} WEAK, {n_covered} COVERED (of {len(results)} probes)")
    if skipped_urls:
        print(f"\n  {len(skipped_urls)} URL(s) previously failed to scrape (data/raw/skipped_urls.txt):")
        for url in sorted(skipped_urls):
            print(f"    - {url}")
    print(f"{'='*90}\n")


def main():
    config = load_config()
    from rag.pipeline import RetrievalPipeline
    pipeline = RetrievalPipeline(config)

    queries = build_gap_probe_queries()
    results = probe(pipeline, queries)
    report(results)


if __name__ == "__main__":
    main()
