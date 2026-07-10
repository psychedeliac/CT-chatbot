import time
import json
import logging
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

def save_raw(chunks: List[Dict[str, Any]], path: str):
    """Saves a list of chunks to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(chunks)} chunks to {path}")
