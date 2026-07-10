"""
data_handlers/ct_loader.py — Corporate Turnaround data loader.

Reads data/cleaned_blogs_rag.json and data/cleaned_rag_testimonials.json and converts
each entry into a LangChain Document.
"""
import json
import os
from typing import List, Optional

from langchain_core.documents import Document
from data_handlers.base_loader import BaseDataLoader


class CorporateTurnaroundLoader(BaseDataLoader):
    """
    Loads Corporate Turnaround blogs and testimonials as LangChain Documents.
    """
    DEFAULT_BLOGS_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "cleaned_blogs_rag.json"
    )
    DEFAULT_TESTIMONIALS_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "cleaned_rag_testimonials.json"
    )

    def __init__(
        self,
        blogs_path: Optional[str] = None,
        testimonials_path: Optional[str] = None,
        max_rows: Optional[int] = None,
    ):
        self.blogs_path = blogs_path or self.DEFAULT_BLOGS_PATH
        self.testimonials_path = testimonials_path or self.DEFAULT_TESTIMONIALS_PATH
        self.max_rows = max_rows

    def load(self) -> List[Document]:
        documents: List[Document] = []
        
        # Load Blogs
        if os.path.exists(self.blogs_path):
            print(f"[CorporateTurnaroundLoader] Reading blogs: {self.blogs_path}")
            with open(self.blogs_path, "r", encoding="utf-8") as f:
                blogs = json.load(f)
                if self.max_rows:
                    blogs = blogs[:self.max_rows]
                for item in blogs:
                    title = item.get("title", "")
                    section = item.get("section_heading", "")
                    text = item.get("chunk_text", "")
                    
                    page_content = f"Title: {title}\nSection: {section}\n\n{text}"
                    metadata = {
                        "source": "blog",
                        "topic": item.get("topic", ""),
                        "category": item.get("category", "")
                    }
                    documents.append(Document(page_content=page_content, metadata=metadata))
        else:
            print(f"[CorporateTurnaroundLoader] Warning: Blogs file not found at {self.blogs_path}")

        # Load Testimonials
        if os.path.exists(self.testimonials_path):
            print(f"[CorporateTurnaroundLoader] Reading testimonials: {self.testimonials_path}")
            with open(self.testimonials_path, "r", encoding="utf-8") as f:
                testimonials = json.load(f)
                if self.max_rows:
                    testimonials = testimonials[:self.max_rows]
                for item in testimonials:
                    problem = item.get("customer_problem", "")
                    solution = item.get("ct_solution", "")
                    outcome = item.get("outcome", "")
                    
                    page_content = (f"Customer Problem: {problem}\n"
                                    f"Corporate Turnaround Solution: {solution}\n"
                                    f"Outcome: {outcome}")
                    metadata = {
                        "source": "testimonial",
                        "id": str(item.get("id", ""))
                    }
                    documents.append(Document(page_content=page_content, metadata=metadata))
        else:
            print(f"[CorporateTurnaroundLoader] Warning: Testimonials file not found at {self.testimonials_path}")

        print(f"[CorporateTurnaroundLoader] {len(documents)} documents created.")
        return documents

    def collection_name(self) -> str:
        return "corporate_turnaround"
