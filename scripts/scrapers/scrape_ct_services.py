"""
scripts/scrapers/scrape_ct_services.py — Scrapes the Corporate Turnaround
services page (corpo-nine.vercel.app/services), which lists the firm's actual
named service lines (Debt Negotiation & Settlement, MCA Debt Relief, Business
Restructuring, Creditor Harassment Protection, SBA & Business Loan Workouts,
Tax Debt Resolution).

This is the single highest-priority source for advice-seeking "what can I do"
/ "i need help" queries: it's the one place in the corpus that names concrete,
actionable services rather than generic debt-relief advice, so it should be
what surfaces when a distressed user asks what their options are.

Usage:
    python -m scripts.scrapers.scrape_ct_services
"""
import os
import sys

# --fresh: replace reproduced topics wholesale (use after changing the chunker)
FRESH = "--fresh" in sys.argv
import re
from bs4 import BeautifulSoup
from scripts.scrapers.utils import iter_content_tags, fetch, parse_body, split_into_chunks, make_chunk, save_raw, merge_with_existing, logger, is_boilerplate

SERVICES_URL = "https://corpo-nine.vercel.app/services"

# The page embeds a floating support-chat widget (placeholder phone number,
# hours, greeting text) whose markup sits alongside real body content and
# gets swept up by the tag-walk. Strip it rather than reject the whole
# section, since real content shares the same paragraph/section.
WIDGET_JUNK_RE = re.compile(r"Call Us Directly.*?How can I help you today\?", re.IGNORECASE | re.DOTALL)


def extract_chunks_from_html(html: str, url: str) -> list[dict]:
    soup = parse_body(html)

    title_tag = soup.find("title")
    page_title = title_tag.get_text(" ", strip=True) if title_tag else "Corporate Turnaround Services"
    # Site's <title> has a mangled separator glyph (encoding artifact) -- normalize it.
    page_title = page_title.split("�")[0].strip() or "Corporate Turnaround Services"

    chunks = []
    sections = []
    current_heading = page_title
    current_text = []

    for tag in iter_content_tags(soup):
        if tag.name in ["h1", "h2", "h3"]:
            if current_text:
                sections.append((current_heading, " ".join(current_text)))
            current_heading = tag.get_text(" ", strip=True)
            current_text = []
        else:
            text = tag.get_text(" ", strip=True)
            if len(text) > 30:
                current_text.append(text)

    if current_text:
        sections.append((current_heading, " ".join(current_text)))

    chunk_counter = 0
    for heading, raw_text in sections:
        text = WIDGET_JUNK_RE.sub("", raw_text).strip()
        if not text:
            continue
        text_chunks = split_into_chunks(text)
        for text_chunk in text_chunks:
            if is_boilerplate(text_chunk):
                continue
            chunk_id = f"ct-services-{chunk_counter}"
            chunk = make_chunk(
                id=chunk_id,
                source_type="ct_services",
                topic="services",
                category="ct-services",
                tags=["services", "advice-priority"],
                title=page_title,
                heading=heading,
                text=text_chunk,
                requires_disclaimer=False,
            )
            chunks.append(chunk)
            chunk_counter += 1

    return chunks


def run():
    html = fetch(SERVICES_URL, delay=1.0)
    if not html:
        logger.error(f"Failed to fetch {SERVICES_URL}")
        return

    new_chunks = extract_chunks_from_html(html, SERVICES_URL)
    logger.info(f"Extracted {len(new_chunks)} chunks from {SERVICES_URL}")

    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "ct_services.json"
    )
    merged = merge_with_existing(new_chunks, save_path, fresh_topics=FRESH)
    save_raw(merged, save_path)


if __name__ == "__main__":
    run()
