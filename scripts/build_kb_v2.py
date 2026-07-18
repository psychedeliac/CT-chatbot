"""
scripts/build_kb_v2.py — Deterministic KB v2 curation.

Rebuilds data/enriched_knowledge_base.json from the current KB plus the
hand-authored canonical records in data/canonical_qa_v2.json:

  1. Purges consumer/personal-finance documents (wrong audience for a
     business-debt bot — they are what caused "adjacent-but-wrong" answers).
  2. Drops exact-duplicate chunks (home vs index page scrapes).
  3. Curates 133 near-identical testimonials down to a diverse, retitled set.
  4. Tags every record with `authority` (canonical | company | evidence |
     background) and `answer_policy` (answer | careful | deflect |
     route_client_line) — surfaced to the LLM by rag/pipeline.format_for_llm.
  5. Merges the canonical v2 Q&A records, baking real user question variants
     into the embedded text ("Also asked:") for BM25/semantic recall.

Run, then re-ingest:
    python scripts/build_kb_v2.py
    python scripts/ingest.py --loader enriched --force
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KB_PATH = os.path.join("data", "enriched_knowledge_base.json")
CANONICAL_PATH = os.path.join("data", "canonical_qa_v2.json")

# Consumer/personal-finance documents (by title). A business-debt assistant
# retrieving these produces wrong-audience advice (HELOCs on your home,
# personal credit repair, debt after death). Eval-set dependencies were
# checked: none of these back an eval query.
PURGE_TITLES = {
    "Breaking Up With Toxic Money Habits and Debt for Good",
    "Can Debt Follow You to Another Country? What to Know",
    "How Credit Cards Keep You in Debt—and What You Can Do",
    "How Long Will It Take to Pay Off Your U.S. Credit Card Debt?",
    "How Much Americans Are in Debt and 5 Ways to Break Free",
    "How To Get Rid Of Charge Offs On Credit Report",
    "How We Define Unsecured Credit Cards—And How to Get One",
    "How to Find Out If You Have Debt and Who You Owe",
    "How to Freeze a Bank Account—and When It Makes Sense",
    "Should You Use a HELOC to Pay Off Debt?",
    "Signs You Might Be Relying Too Much on Credit Cards",
    "Statute of Limitations on Debt After Death: What You Need to Know",
    "The Psychology of Debt: Debt Addiction and Financial Health",
    "What Percentage of Americans Are in Debt?",
    "Debt Parking: Understanding Hidden Debt and How to Fight Back",
}

AUTHORITY_BY_SOURCE_TYPE = {
    "qa_pair": "canonical",
    "corporate_turnaround_site": "company",
    "ct_services": "company",
    "testimonial": "evidence",
    "educational": "background",
    "regulatory": "background",
}

# Testimonial diversity buckets: (slug, human label, keywords matched against
# problem+solution+outcome). First bucket that matches wins; max 2 per bucket.
TESTIMONIAL_BUCKETS = [
    ("mca", "Merchant cash advance relief", ("merchant cash", "mca", "daily payment", "cash advance")),
    ("tax", "IRS and tax debt resolved", ("irs", "tax")),
    ("vendor", "Supplier and vendor debt settled", ("vendor", "supplier", "invoice")),
    ("lawsuit", "Lawsuit or judgment resolved", ("lawsuit", "sued", "judgment", "legal action", "attorney")),
    ("harassment", "Creditor harassment stopped", ("harass", "collection call", "creditor calls", "stopped calling")),
    ("hardship", "Personal hardship behind the debt", ("illness", "medical", "family", "storm", "disaster", "hurricane", "divorce")),
    ("bankruptcy", "Bankruptcy avoided", ("bankruptcy",)),
    ("payment-plan", "Affordable payment plan negotiated", ("payment plan", "afford", "budget", "monthly")),
    ("settled", "Debt settled and business saved", ("settl", "saved", "reduc")),
]
MAX_PER_BUCKET = 2


def testimonial_text(rec: dict) -> str:
    return " ".join(
        str(rec.get(k, "")) for k in ("customer_problem", "ct_solution", "outcome")
    ).lower()


def curate_testimonials(testimonials: list[dict]) -> list[dict]:
    kept: list[dict] = []
    counts = {slug: 0 for slug, _, _ in TESTIMONIAL_BUCKETS}
    for rec in testimonials:
        text = testimonial_text(rec)
        for slug, label, keywords in TESTIMONIAL_BUCKETS:
            if counts[slug] >= MAX_PER_BUCKET or not any(k in text for k in keywords):
                continue
            counts[slug] += 1
            kept.append({
                **rec,
                "id": f"testimonial-{slug}-{counts[slug]}",
                "title": f"Client Success Story — {label}",
                "section_heading": f"Case example: {label.lower()}",
            })
            break
    return kept


def bake_variants(rec: dict) -> dict:
    """Fold variant question phrasings into the embedded text so BM25 and the
    embedder see the vocabulary real users type."""
    variants = rec.get("variant_questions") or []
    if not variants:
        return rec
    variant_block = "\n".join(f"Also asked: {q}" for q in variants)
    return {**rec, "chunk_text": f"{rec['chunk_text']}\n\n{variant_block}"}


def with_tokens(rec: dict) -> dict:
    text = rec.get("chunk_text", "")
    return {**rec, "chunk_tokens": max(1, len(text) // 4)} if text else rec


def main() -> None:
    with open(KB_PATH, encoding="utf-8") as f:
        kb = json.load(f)
    with open(CANONICAL_PATH, encoding="utf-8") as f:
        canonical_v2 = json.load(f)

    purged = [r for r in kb if r.get("title") in PURGE_TITLES]
    kept = [r for r in kb if r.get("title") not in PURGE_TITLES]

    # Exact-duplicate chunk bodies (e.g. home vs index scrape) — first wins.
    seen_bodies: set[str] = set()
    deduped: list[dict] = []
    dups = 0
    for r in kept:
        body = (r.get("chunk_text") or testimonial_text(r)).strip()
        if body in seen_bodies:
            dups += 1
            continue
        seen_bodies.add(body)
        deduped.append(r)

    testimonials = [r for r in deduped if r.get("source_type") == "testimonial"]
    non_testimonials = [r for r in deduped if r.get("source_type") != "testimonial"]
    kept_testimonials = curate_testimonials(testimonials)

    def tag(rec: dict) -> dict:
        authority = rec.get("authority") or AUTHORITY_BY_SOURCE_TYPE.get(
            rec.get("source_type", ""), "background"
        )
        policy = rec.get("answer_policy") or (
            "careful" if rec.get("requires_disclaimer") else "answer"
        )
        return {**rec, "authority": authority, "answer_policy": policy}

    v2_ids = {r["id"] for r in canonical_v2}
    merged = (
        [tag(r) for r in non_testimonials if r["id"] not in v2_ids]
        + [tag(r) for r in kept_testimonials]
        + [with_tokens(bake_variants(tag(r))) for r in canonical_v2]
    )

    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"KB v2 built: {len(kb)} -> {len(merged)} records")
    print(f"  purged wrong-audience docs: {len(purged)} chunks "
          f"({len(PURGE_TITLES)} documents)")
    print(f"  exact-duplicate chunks dropped: {dups}")
    print(f"  testimonials curated: {len(testimonials)} -> {len(kept_testimonials)}")
    print(f"  canonical v2 records merged: {len(canonical_v2)}")
    print("\nNext: python scripts/ingest.py --loader enriched --force")


if __name__ == "__main__":
    main()
