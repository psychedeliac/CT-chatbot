"""
data_handlers/enriched_loader.py — Enriched RAG data loader.
"""
import json
import os
from typing import List, Optional

from langchain_core.documents import Document
from data_handlers.base_loader import BaseDataLoader


class EnrichedLoader(BaseDataLoader):
    """
    Loads the unified enriched knowledge base JSON.
    """
    DEFAULT_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "enriched_knowledge_base.json"
    )

    def __init__(
        self,
        path: Optional[str] = None,
        max_rows: Optional[int] = None,
    ):
        self.path = path or self.DEFAULT_PATH
        self.max_rows = max_rows

    def load(self) -> List[Document]:
        documents: List[Document] = []
        skipped = 0

        if os.path.exists(self.path):
            print(f"[EnrichedLoader] Reading data: {self.path}")
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if self.max_rows:
                    data = data[:self.max_rows]
                for item in data:
                    title = item.get("title", "")
                    section = item.get("section_heading", "")
                    text = item.get("chunk_text", "")

                    # Handle different chunk schemas (e.g., testimonials)
                    if not text:
                        if "customer_problem" in item:
                            text = (f"Customer Problem: {item.get('customer_problem', '')}\n"
                                    f"Solution: {item.get('ct_solution', '')}\n"
                                    f"Outcome: {item.get('outcome', '')}")
                        else:
                            text = item.get("content", "")

                    page_content = f"Title: {title}\nSection: {section}\n\n{text}"

                    if self._is_low_quality(page_content):
                        skipped += 1
                        continue

                    metadata = {
                        "source": item.get("source_type", "unknown"),
                        "topic": item.get("topic", ""),
                        "category": item.get("category", "")
                    }
                    documents.append(Document(page_content=page_content, metadata=metadata))
        else:
            print(f"[EnrichedLoader] Warning: File not found at {self.path}")

        print(f"[EnrichedLoader] {len(documents)} documents created ({skipped} skipped as low-quality/placeholder).")
        return documents

    @staticmethod
    def _is_low_quality(page_content: str) -> bool:
        """
        Flags records that shouldn't be embedded:
          - Templated placeholders where the source fields were missing
            (e.g. testimonials with "Unknown" for problem/solution/outcome)
          - Near-empty chunks with no real body text
        """
        body = page_content.split("\n\n", 1)[-1].strip()
        if len(body) < 40:
            return True
        if page_content.count("Unknown") >= 2:
            return True
        return False

    def collection_name(self) -> str:
        return "enriched_knowledge_base"
