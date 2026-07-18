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

CONTENT_TAGS = ["h1", "h2", "h3", "p", "li"]


def iter_content_tags(soup):
    """
    Yield content tags, skipping any nested inside another content tag.

    A bare find_all(CONTENT_TAGS) returns a <p> wrapped in an <li> twice -- once
    when the <li> is visited and its full text taken, and again for the <p>
    itself. That duplicated ~7% of all sentences in the scraped corpus, which
    both wastes context and biases BM25 term frequencies toward whatever the
    site happens to mark up with nested tags.
    """
    for tag in soup.find_all(CONTENT_TAGS):
        if tag.find_parent(CONTENT_TAGS) is None:
            yield tag


def split_into_chunks(text: str, max_tokens: int = 400, overlap_ratio: float = 0.15) -> List[str]:
    """
    Split text into chunks on sentence boundaries, with overlap between them.

    The previous implementation hard-cut every N words with no regard for
    punctuation and no overlap, which left ~48% of the shipped corpus ending
    mid-sentence and ~27% starting mid-sentence (chunks trailing off like
    "...creditors seize your assets and"). Both halves of a split sentence
    then embed poorly and read badly when handed to the LLM.

    Defaults follow current RAG practice: ~300-500 tokens per chunk with
    10-15% overlap so a fact spanning a boundary survives in one piece.
    """
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    max_words = max(1, int(max_tokens / 1.3))
    overlap_words = int(max_words * overlap_ratio)

    chunks: List[str] = []
    current: List[str] = []

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        # ponytail: an individual sentence longer than max_words becomes its own
        # oversized chunk rather than being cut mid-thought. Split long sentences
        # on clause boundaries only if such sentences actually show up.
        if current and len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_words:] if overlap_words else []
        current.extend(words)

    if current:
        chunks.append(" ".join(current))
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

def merge_with_existing(
    new_chunks: List[Dict[str, Any]],
    path: str,
    fresh_topics: bool = False,
) -> List[Dict[str, Any]]:
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

    if fresh_topics:
        # Deliberate re-chunk: any topic reproduced this run is replaced wholesale.
        # Exact-text dedup cannot collapse a re-chunk, because new chunk
        # boundaries never match the old ones -- so the default union would keep
        # BOTH chunkings of the same page (a re-scrape of chapter-11 went from 43
        # fresh chunks to 140 total). Topics that failed to scrape at all are
        # still carried over, preserving the transient-failure protection.
        fresh = {c.get("topic") for c in new_chunks}
        carried_over = [c for c in existing if c.get("topic") not in fresh]
        if carried_over:
            logger.warning(
                f"fresh_topics: replaced {len(fresh)} reproduced topic(s); "
                f"carried over {len(carried_over)} chunk(s) from topics that did not scrape."
            )
        return new_chunks + carried_over

    seen_texts = {c["chunk_text"].strip() for c in new_chunks}
    carried_over = [c for c in existing if c["chunk_text"].strip() not in seen_texts]
    if carried_over:
        logger.warning(
            f"Carrying over {len(carried_over)} chunk(s) from the previous run "
            f"not reproduced by this run's scrape. If this was an intentional "
            f"re-chunk, pass fresh_topics=True -- otherwise both chunkings persist."
        )
    return new_chunks + carried_over


def save_raw(chunks: List[Dict[str, Any]], path: str):
    """Saves a list of chunks to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(chunks)} chunks to {path}")
