import json
import os
import sys

# --fresh: replace reproduced topics wholesale (use after changing the chunker)
FRESH = "--fresh" in sys.argv
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from scripts.scrapers.utils import iter_content_tags, fetch, parse_body, split_into_chunks, make_chunk, save_raw, logger, is_boilerplate, merge_with_existing

BLOG_SOURCES = [
    # NerdWallet — confirmed accessible. Several 2024-era URLs in this list
    # were retired (site restructure, 404 as of 2026-07); replaced with the
    # current live equivalents where NerdWallet still covers the topic.
    {
        "domain": "nerdwallet",
        "selector": "article",  # confirmed <article> tag wraps body
        "urls": [
            "https://www.nerdwallet.com/article/small-business/merchant-cash-advance",
            "https://www.nerdwallet.com/article/small-business/ucc-filing",
            "https://www.nerdwallet.com/article/small-business/personal-guarantee-business-loan",
            "https://www.nerdwallet.com/article/small-business/business-loan-default",
        ],
    },
    # Investopedia dropped: its `.article-body` content div is served
    # inconsistently (present in some fetches, reduced to a CSS-only stub in
    # others — likely bot-tiered rendering), so scrapes of it are not
    # reliably reproducible. NerdWallet + uscourts.gov below cover the same
    # ground (personal guarantee, business loan default, bankruptcy chapters)
    # with reliable static HTML.
    # U.S. Courts — public domain, closes the "business bankruptcy types" gap
    # (Chapter 7 liquidation vs. 11 reorganization vs. 13) with zero licensing
    # risk, unlike paraphrasing a commercial blog.
    {
        "domain": "uscourts",
        "selector": "article",
        "urls": [
            "https://www.uscourts.gov/court-programs/bankruptcy/bankruptcy-basics/chapter-7-bankruptcy-basics",
            "https://www.uscourts.gov/court-programs/bankruptcy/bankruptcy-basics/chapter-11-bankruptcy-basics",
            "https://www.uscourts.gov/court-programs/bankruptcy/bankruptcy-basics/chapter-13-bankruptcy-basics",
        ],
    },
]

# Domain-level category overrides — the default heuristic in
# extract_chunks_from_html (category = "mca-education" if "mca" appears in
# the URL else "business-debt") misclassifies bankruptcy-basics pages, which
# never contain "mca" but are also not general business-debt-mechanics content.
CATEGORY_OVERRIDES = {
    "uscourts": "business-bankruptcy",
}

def extract_chunks_from_html(html: str, url: str, source: dict) -> list[dict]:
    soup = parse_body(html)
    
    title_tag = soup.find('title')
    page_title = title_tag.get_text(" ", strip=True) if title_tag else f"{source['domain']} Article"
    
    chunks = []
    sections = []
    current_heading = page_title
    current_text = []
    
    # Try to find the specific content wrapper
    content_area = None
    for selector in source["selector"].split(","):
        content_area = soup.select_one(selector.strip())
        if content_area:
            break
            
    if not content_area:
        content_area = soup  # Fallback to whole body if selector fails
    
    for tag in iter_content_tags(content_area):
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
    path = urlparse(url).path.strip('/').replace('/', '-') or 'home'
    topic = f"{source['domain']}-{path}"
    
    # Simple category heuristic based on URL, with a per-domain override for
    # sources that don't fit the mca/business-debt binary (see CATEGORY_OVERRIDES).
    if source["domain"] in CATEGORY_OVERRIDES:
        category = CATEGORY_OVERRIDES[source["domain"]]
    else:
        category = "mca-education" if "mca" in url or "merchant-cash-advance" in url else "business-debt"
    
    for heading, text in sections:
        text_chunks = split_into_chunks(text)
        for i, text_chunk in enumerate(text_chunks):
            if is_boilerplate(text_chunk):
                continue
            chunk_id = f"edu-{topic}-{chunk_counter}"
            chunk = make_chunk(
                id=chunk_id,
                source_type="educational",
                topic=topic,
                category=category,
                tags=[source['domain']],
                title=page_title,
                heading=heading,
                text=text_chunk,
                requires_disclaimer=False
            )
            chunks.append(chunk)
            chunk_counter += 1
            
    return chunks

def run():
    all_chunks = []
    skipped_urls = []
    manifest = []

    for source in BLOG_SOURCES:
        for url in source["urls"]:
            html = fetch(url, delay=2.0)
            if not html:
                logger.warning(f"[SKIPPED] {url} — blocked or unavailable. Add to manual review list.")
                skipped_urls.append(url)
                manifest.append({"url": url, "http_ok": False, "n_chunks": 0})
                continue

            page_chunks = extract_chunks_from_html(html, url, source)
            logger.info(f"Extracted {len(page_chunks)} chunks from {url}")
            manifest.append({"url": url, "http_ok": True, "n_chunks": len(page_chunks)})
            if not page_chunks:
                # Fetch succeeded (2xx) but no chunks were extracted — usually
                # a stale CSS selector or an anti-bot/consent stub page. This
                # previously went unlogged once the run finished, making the
                # failure invisible to anyone auditing skipped_urls.txt.
                logger.warning(f"[EMPTY] {url} — page fetched OK but yielded 0 chunks (selector mismatch?).")
                skipped_urls.append(url)
                continue
            all_chunks.extend(page_chunks)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    save_path = os.path.join(base_dir, "data", "raw", "industry_blogs.json")
    save_raw(merge_with_existing(all_chunks, save_path, fresh_topics=FRESH), save_path)

    manifest_path = os.path.join(base_dir, "data", "raw", "industry_blogs_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved per-URL manifest to {manifest_path}")

    if skipped_urls:
        skipped_path = os.path.join(base_dir, "data", "raw", "skipped_urls.txt")
        os.makedirs(os.path.dirname(skipped_path), exist_ok=True)
        with open(skipped_path, "w", encoding="utf-8") as f:
            for url in skipped_urls:
                f.write(url + "\n")
        logger.warning(f"Saved {len(skipped_urls)} skipped URLs to {skipped_path}")

if __name__ == "__main__":
    run()
