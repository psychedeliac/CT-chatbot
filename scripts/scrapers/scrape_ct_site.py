import os
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from scripts.scrapers.utils import fetch, parse_body, split_into_chunks, make_chunk, save_raw, logger, is_boilerplate

CT_SEED_URLS = [
    "https://www.corporateturnaround.com/",
    "https://www.corporateturnaround.com/aboutus.html",
    "https://www.corporateturnaround.com/whatwedo.html",
    "https://www.corporateturnaround.com/news.html",
    "https://www.corporateturnaround.com/freeconsultation.php",
    "https://www.corporateturnaround.com/whatwedo/businesstools.html",
    "https://www.corporateturnaround.com/whatwedo/score.html",
    "https://www.corporateturnaround.com/directory.html",
]

DOMAIN = "www.corporateturnaround.com"

def get_internal_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    links = []
    for a in soup.find_all("a", href=True):
        href = a['href']
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if parsed.netloc == DOMAIN and "testimonials.html" not in parsed.path:
            # Normalize URL (remove fragments)
            clean_url = full_url.split('#')[0]
            if clean_url not in links:
                links.append(clean_url)
    return links

def extract_chunks_from_html(html: str, url: str) -> list[dict]:
    soup = parse_body(html)
    
    # Try to get page title
    title_tag = soup.find('title')
    page_title = title_tag.get_text(strip=True) if title_tag else "Corporate Turnaround"
    
    chunks = []
    sections = []
    current_heading = page_title
    current_text = []
    
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
    topic = urlparse(url).path.strip('/').replace('.html', '').replace('.php', '') or 'home'
    
    for heading, text in sections:
        text_chunks = split_into_chunks(text)
        for i, text_chunk in enumerate(text_chunks):
            if is_boilerplate(text_chunk):
                continue
            chunk_id = f"ct-site-{topic}-{chunk_counter}"
            chunk = make_chunk(
                id=chunk_id,
                source_type="corporate_turnaround_site",
                topic=topic,
                category="ct-process",
                tags=[topic],
                title=page_title,
                heading=heading,
                text=text_chunk,
                requires_disclaimer=False
            )
            chunks.append(chunk)
            chunk_counter += 1
            
    return chunks

def run():
    visited = set()
    to_visit = list(CT_SEED_URLS)
    all_chunks = []
    max_pages = 30
    
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
            
        visited.add(url)
        html = fetch(url, delay=3.0)
        if not html:
            continue
            
        soup = BeautifulSoup(html, "lxml")
        links = get_internal_links(soup, url)
        
        for link in links:
            if link not in visited and link not in to_visit:
                # strictly allow only html/php or paths without extension
                parsed_link = urlparse(link)
                path = parsed_link.path.lower()
                ext = os.path.splitext(path)[1]
                if ext in ['.html', '.htm', '.php', ''] or path.endswith('/'):
                    to_visit.append(link)
                
        page_chunks = extract_chunks_from_html(html, url)
        logger.info(f"Extracted {len(page_chunks)} chunks from {url}")
        all_chunks.extend(page_chunks)
        
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "ct_site.json")
    save_raw(all_chunks, save_path)

if __name__ == "__main__":
    run()
