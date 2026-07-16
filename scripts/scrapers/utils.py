import time
import json
import logging
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch(url: str, delay: float = 2.0, max_retries: int = 3) -> str:
    """Fetches a URL with a delay, retries, and a standard user agent."""
    time.sleep(delay)
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching {url} (Attempt {attempt+1}/{max_retries})")
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            # Do not retry on 404 Not Found
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                return ""
                
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                return ""
    return ""

def parse_body(html: str) -> BeautifulSoup:
    """Parses HTML and strips out non-content tags."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["nav", "footer", "header", "script", "style", "aside", "form", "iframe", "noscript"]):
        tag.decompose()
    return soup

def split_into_chunks(text: str, max_tokens: int = 120) -> List[str]:
    """Splits text into chunks, respecting a rough max token limit."""
    words = text.split()
    max_words = int(max_tokens / 1.3)
    
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks

BOILERPLATE_REGEXES = [
    re.compile(
        r"the \.gov means it.?s official\.?\s*federal government websites often end in "
        r"\.gov or \.mil\.?\s*before sharing sensitive information,?\s*make sure you.?re on "
        r"a federal government site\.?",
        re.IGNORECASE,
    ),
    re.compile(r"click here\s*or call\s*[\d.\-]+\s*for a free consultation\.?", re.IGNORECASE),
    re.compile(
        r"the site is secure\.?\s*the\s*https://\s*ensures that you are connecting to the "
        r"official website and that any information you provide is encrypted and transmitted "
        r"securely\.?",
        re.IGNORECASE,
    ),
]


def is_boilerplate(text: str, dominance_threshold: float = 0.6) -> bool:
    """
    Flags nav/banner/CTA snippets that survive tag-based section extraction
    but carry no real content (e.g. gov-site ".gov means it's official"
    banners, site CTAs like "Click Here or call 800...for a free
    consultation."). Strips known boilerplate SENTENCES (not short keyword
    fragments — a fragment match would flag any chunk that merely mentions
    the phrase once, even one dominated by real FAQ/program content) and
    flags the chunk only if most of it disappears in the process.
    """
    stripped = text.strip()
    if not stripped:
        return True
    residual = stripped
    for pattern in BOILERPLATE_REGEXES:
        residual = pattern.sub("", residual)
    removed_fraction = 1 - (len(residual.strip()) / len(stripped))
    return removed_fraction >= dominance_threshold


def make_chunk(id: str, source_type: str, topic: str, category: str, tags: List[str], title: str, heading: str, text: str, requires_disclaimer: bool = False) -> Dict[str, Any]:
    """Creates a standard JSON chunk dictionary."""
    tokens = int(len(text.split()) * 1.3)
    return {
        "id": id,
        "source_type": source_type,
        "topic": topic,
        "category": category,
        "topic_tags": tags,
        "title": title,
        "section_heading": heading,
        "chunk_text": text,
        "chunk_tokens": tokens,
        "requires_disclaimer": requires_disclaimer
    }

def merge_with_existing(new_chunks: List[Dict[str, Any]], path: str) -> List[Dict[str, Any]]:
    """
    Unions freshly-scraped chunks with whatever is already on disk, deduped
    by exact chunk_text, instead of letting a full-overwrite save silently
    destroy good content from an earlier run. This matters for two distinct
    failure modes seen in practice: a transient fetch failure that drops a
    whole topic, and a live page redesign that keeps returning chunks but
    loses specific content (e.g. a definition moved into a JS-rendered
    calculator widget our tag-walking extraction can't see) — in both cases
    "new chunks exist for this topic" does NOT mean "new chunks are a
    strict improvement," so old content is kept alongside new rather than
    replaced by topic. format_raw_to_chunks.py's own exact-text dedup
    collapses any genuine repeats at merge time.
    """
    if not os.path.exists(path):
        return new_chunks

    with open(path, "r", encoding="utf-8") as f:
        existing = json.load(f)

    seen_texts = {c["chunk_text"].strip() for c in new_chunks}
    carried_over = [c for c in existing if c["chunk_text"].strip() not in seen_texts]
    if carried_over:
        logger.warning(
            f"Carrying over {len(carried_over)} chunk(s) from the previous run "
            f"not reproduced by this run's scrape."
        )
    return new_chunks + carried_over


def save_raw(chunks: List[Dict[str, Any]], path: str):
    """Saves a list of chunks to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(chunks)} chunks to {path}")
