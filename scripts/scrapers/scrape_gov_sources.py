import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from scripts.scrapers.utils import fetch, parse_body, split_into_chunks, make_chunk, save_raw, logger

GOV_URLS = [
    # FTC — FDCPA & debt rights
    ("ftc", "https://consumer.ftc.gov/articles/debt-collection-faqs"),
    ("ftc", "https://consumer.ftc.gov/articles/what-do-if-debt-collector-sues-you"),
    ("ftc", "https://consumer.ftc.gov/articles/merchant-cash-advances"),

    # CFPB — debt collection & small biz
    ("cfpb", "https://www.consumerfinance.gov/consumer-tools/debt-collection/"),
    ("cfpb", "https://www.consumerfinance.gov/consumer-tools/credit-reports-and-scores/"),

    # SBA — loans, defaults, financial health
    ("sba", "https://www.sba.gov/funding-programs/loans/7a-loans"),
    ("sba", "https://www.sba.gov/business-guide/manage-your-business/manage-your-finances"),

    # IRS — tax debt for context
    ("irs", "https://www.irs.gov/businesses/small-businesses-self-employed/collection-procedures-for-businesses"),
    ("irs", "https://www.irs.gov/payments/offer-in-compromise"),
]

def extract_chunks_from_html(html: str, url: str, agency: str) -> list[dict]:
    soup = parse_body(html)
    
    title_tag = soup.find('title')
    page_title = title_tag.get_text(strip=True) if title_tag else f"{agency.upper()} Resource"
    
    chunks = []
    sections = []
    current_heading = page_title
    current_text = []
    
    # Simple extraction heuristic
    for tag in soup.find_all(["h1", "h2", "h3", "p", "li"]):
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
    topic = f"{agency}-{path}"
    category = "business-debt"
    
    for heading, text in sections:
        text_chunks = split_into_chunks(text)
        for i, text_chunk in enumerate(text_chunks):
            chunk_id = f"gov-{topic}-{chunk_counter}"
            chunk = make_chunk(
                id=chunk_id,
                source_type="regulatory",
                topic=topic,
                category=category,
                tags=[agency],
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
    
    for agency, url in GOV_URLS:
        # FTC robots.txt requires 5 seconds delay
        delay = 5.0 if agency == 'ftc' else 2.0
        html = fetch(url, delay=delay)
        if not html:
            logger.warning(f"Failed to fetch {url}")
            continue
            
        page_chunks = extract_chunks_from_html(html, url, agency)
        logger.info(f"Extracted {len(page_chunks)} chunks from {url}")
        all_chunks.extend(page_chunks)
        
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "gov_sources.json")
    save_raw(all_chunks, save_path)

if __name__ == "__main__":
    run()
