"""
scripts/repair_kb.py — One-time repair pass over the enriched knowledge base.

Fixes data defects that are safely correctable in place. Run after
format_raw_to_chunks.py, before ingest.py:

    python scripts/repair_kb.py            # writes data/enriched_knowledge_base.json
    python scripts/repair_kb.py --dry-run  # report only

What it fixes
  1. Testimonial records carrying NO metadata at all (no title, section_heading,
     source_type, category). 133 of 799 records shipped this way, rendering as
     "Title: / Section:" blanks to the LLM and "Unknown Source" in the UI.
  2. Category casing collisions ("Debt Collection Rights" vs
     "debt-collection-rights" vs "Debt-Collection-Rights" were three separate
     buckets for one category), which make metadata useless for filtering.
  3. Missing spaces left by HTML extraction ("real time.When" -> "real time. When").
     Only punctuation-boundary cases are repaired; letter-boundary damage
     ("ourbusiness") is unrecoverable here and needs a re-scrape.
  4. Inconsistent rendering of the main contact number.

What it deliberately does NOT do
  Re-join and re-split the mid-sentence chunks. ~48% of records end mid-sentence
  because the old scraper cut every N words, but chunk ordering within a section
  cannot be reliably reconstructed from the shipped ids, so an automatic merge
  risks stitching text together in the wrong order. The chunker itself is fixed
  (scripts/scrapers/utils.py); a re-scrape is required to clear the existing
  truncation.
"""
import argparse
import json
import os
import re
from typing import Any

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "enriched_knowledge_base.json",
)

CT_PHONE = "1-800-889-0232"          # new enquiries / free consultation
CT_CLIENT_PHONE = "1-800-411-1113"   # existing clients & creditor inquiries

# Testimonials ship with only id/customer_problem/ct_solution/outcome.
TESTIMONIAL_DEFAULTS = {
    "source_type": "testimonial",
    "title": "Client Success Story",
    "section_heading": "Debt Resolution Case Example",
    "category": "client-outcomes",
    "topic": "client-outcomes",
    # Outcome claims need a "results vary" framing under FTC guidance for
    # testimonial advertising -- these are specific dollar results.
    "requires_disclaimer": True,
}


def normalize_category(value: str) -> str:
    """Collapse casing/separator variants onto one canonical slug."""
    return re.sub(r"[\s_]+", "-", str(value or "").strip().lower())


def fix_spacing(text: str) -> str:
    """
    Restore spaces lost by BeautifulSoup get_text(strip=True) joining inline
    elements. Only punctuation boundaries are touched -- inserting a space at
    every lowercase/uppercase boundary would corrupt legitimate casing.
    """
    if not text:
        return text
    text = re.sub(r"(?<=[a-z0-9])([.;:!?])(?=[A-Z])", r"\1 ", text)
    text = re.sub(r"(?<=[a-z])(,)(?=[A-Z])", r"\1 ", text)
    return text


def normalize_phone(text: str) -> str:
    """Render the main contact line one consistent way."""
    if not text:
        return text
    # No \b anchors: the get_text spacing damage welds the number to adjacent
    # words ("call800.889.0232for a free consultation"), and \b never matches
    # between "l" and "8". Pad with spaces so the number separates cleanly.
    text = re.sub(r"1?[-.\s]?800[-.\s]?889[-.\s]?0232", f" {CT_PHONE} ", text)
    text = re.sub(r"1?[-.\s]?800[-.\s]?411[-.\s]?1113", f" {CT_CLIENT_PHONE} ", text)
    return text.replace("  ", " ").strip()


def repair(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "testimonials_backfilled": 0,
        "categories_normalized": 0,
        "spacing_fixed": 0,
        "phones_normalized": 0,
    }
    repaired = []

    for record in records:
        new = dict(record)  # never mutate the input

        # 1. Backfill metadata-less testimonial records.
        if "customer_problem" in new and not str(new.get("title", "")).strip():
            new = {**new, **TESTIMONIAL_DEFAULTS}
            stats["testimonials_backfilled"] += 1

        # 2. Canonicalize category.
        if new.get("category"):
            canonical = normalize_category(new["category"])
            if canonical != new["category"]:
                stats["categories_normalized"] += 1
            new["category"] = canonical

        # 3 & 4. Text repairs across every free-text field.
        for field in ("chunk_text", "customer_problem", "ct_solution", "outcome",
                      "title", "section_heading"):
            original = new.get(field)
            if not isinstance(original, str) or not original:
                continue
            fixed = normalize_phone(fix_spacing(original))
            if fixed != original:
                if fix_spacing(original) != original:
                    stats["spacing_fixed"] += 1
                if normalize_phone(original) != original:
                    stats["phones_normalized"] += 1
                new[field] = fixed

        repaired.append(new)

    return repaired, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair the enriched knowledge base in place.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    parser.add_argument("--path", default=DATA_PATH, help="Path to enriched_knowledge_base.json")
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        records = json.load(f)

    repaired, stats = repair(records)

    print(f"\n=== KB Repair ({len(records)} records) ===")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' '):<28}: {value}")

    if args.dry_run:
        print("\n  (dry run — nothing written)")
        return

    with open(args.path, "w", encoding="utf-8") as f:
        json.dump(repaired, f, ensure_ascii=False, indent=2)
    print(f"\n  Written: {args.path}")
    print("  Next: python scripts/ingest.py --loader enriched --force")


if __name__ == "__main__":
    main()
