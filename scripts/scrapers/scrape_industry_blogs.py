import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from scripts.scrapers.utils import fetch, parse_body, split_into_chunks, make_chunk, save_raw, logger

BLOG_SOURCES = [
    # NerdWallet — confirmed accessible
    {
        "domain": "nerdwallet",
        "selector": "article",  # confirmed <article> tag wraps body
        "urls": [
            "https://www.nerdwallet.com/article/small-business/merchant-cash-advance",
            "https://www.nerdwallet.com/article/small-business/business-debt-consolidation",
            "https://www.nerdwallet.com/article/small-business/how-to-get-out-of-business-debt",
            "https://www.nerdwallet.com/article/small-business/merchant-cash-advance-alternatives",
            "https://www.nerdwallet.com/article/small-business/ucc-lien",
            "https://www.nerdwallet.com/article/small-business/sba-loan-default",
            "https://www.nerdwallet.com/article/small-business/personal-guarantee",
        ],
    },
    # Investopedia — may be blocked (460); best-effort
    {
        "domain": "investopedia",
        "selector": "div[data-testid='article-body']",
        "urls": [
            "https://www.investopedia.com/terms/m/merchant-cash-advance.asp",
            "https://www.investopedia.com/terms/c/chapter11.asp",
            "https://www.investopedia.com/terms/c/chapter7.asp",
            "https://www.investopedia.com/terms/u/ucc-1-statement.asp",
            "https://www.investopedia.com/terms/p/personal-guarantee.asp",
        ],
    },
    # Nav.com
    {
        "domain": "nav",
        "selector": "article, .article-content, .post-content",
        "urls": [
            "https://www.nav.com/resource/merchant-cash-advance/",
            "https://www.nav.com/resource/business-credit-score/",
            "https://www.nav.com/resource/ucc-filing/",
            "https://www.nav.com/resource/business-debt-relief/",
        ],
    },
    # Fundera (NerdWallet subdomain)
    {
        "domain": "fundera",
        "selector": "article, .post-content",
        "urls": [
            "https://www.fundera.com/business-loans/guides/merchant-cash-advance",
            "https://www.fundera.com/business-loans/guides/mca-stacking",
            "https://www.fundera.com/blog/out-of-court-debt-settlement",
        ],
    },
]

def extract_chunks_from_html(html: str, url: str, source: dict) -> list[dict]:
    soup = parse_body(html)
    
    title_tag = soup.find('title')
    page_title = title_tag.get_text(strip=True) if title_tag else f"{source['domain']} Article"
    
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
    
    for tag in content_area.find_all(["h1", "h2", "h3", "p", "li"]):
        if tag.name in ["h1", "h2", "h3"]:
            if current_text:
                sections.append((current_heading, " ".join(current_text)))
            current_heading = tag.get_text(strip=True)
            current_text = []
        else:
            text = tag.get_text(strip=True)
            if len(text) > 30:
                current_text.append(text)
                
    if current_text:
        sections.append((current_heading, " ".join(current_text)))
        
    chunk_counter = 0
    path = urlparse(url).path.strip('/').replace('/', '-') or 'home'
    topic = f"{source['domain']}-{path}"
    
    # Simple category heuristic based on URL
    category = "mca-education" if "mca" in url or "merchant-cash-advance" in url else "business-debt"
    
    for heading, text in sections:
        text_chunks = split_into_chunks(text)
        for i, text_chunk in enumerate(text_chunks):
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
    
    for source in BLOG_SOURCES:
        for url in source["urls"]:
            html = fetch(url, delay=2.0)
            if not html:
                logger.warning(f"[SKIPPED] {url} — blocked or unavailable. Add to manual review list.")
                skipped_urls.append(url)
                continue
                
            page_chunks = extract_chunks_from_html(html, url, source)
            logger.info(f"Extracted {len(page_chunks)} chunks from {url}")
            all_chunks.extend(page_chunks)
            
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    save_path = os.path.join(base_dir, "data", "raw", "industry_blogs.json")
    save_raw(all_chunks, save_path)
    
    if skipped_urls:
        skipped_path = os.path.join(base_dir, "data", "raw", "skipped_urls.txt")
        os.makedirs(os.path.dirname(skipped_path), exist_ok=True)
        with open(skipped_path, "w", encoding="utf-8") as f:
            for url in skipped_urls:
                f.write(url + "\n")
        logger.warning(f"Saved {len(skipped_urls)} skipped URLs to {skipped_path}")

if __name__ == "__main__":
    run()
