"""
rag/pipeline.py — RetrievalPipeline: the single implementation of retrieval.

Framework-agnostic core (no @tool, no LangGraph). Everything that needs to
search the knowledge base — the LangChain tool in rag/retriever.py, the
diagnostics in scripts/diagnose_retrieval.py, and the eval harness in
scripts/eval_retrieval.py — calls into this module instead of reimplementing
retrieval logic. That duplication (diagnostics hand-copying the tool's Chroma
+ BM25 + merge logic) previously caused real drift between what the
diagnostics showed and what the live app actually did.

Pipeline stages:
  1. Chroma (semantic) + BM25 (keyword) each fetch a deep candidate pool.
  2. EnsembleRetriever fuses both pools via real Reciprocal Rank Fusion (RRF).
  3. A cross-encoder reranks the fused pool, producing one interpretable
     relevance score per candidate.
  4. Candidates below rerank_score_threshold are dropped (the abstain gate);
     survivors are capped to rag_k.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from config import AgentConfig
from rag.embeddings.google import GoogleEmbeddingProvider
from rag.embeddings.huggingface import HuggingFaceEmbeddingProvider

EMBEDDING_PROVIDERS = {
    "google": GoogleEmbeddingProvider,
    "huggingface": HuggingFaceEmbeddingProvider,
}

# Common stop words and ubiquitous domain terms to ignore in keyword searches.
STOP_WORDS = {
    # Question words
    "what", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "has", "have", "had",
    "who", "whom", "whose", "which", "where", "when", "why", "how",
    # Pronouns
    "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    # Prepositions, articles & conjunctions
    "a", "an", "the", "in", "on", "for", "with", "about", "of", "to", "and", "or", "but", "because", "as", "until", "while",
    "at", "by", "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", "up", "down",
    "out", "off", "over", "under", "again", "further", "then", "once",
    # Domain terms
    "corporate", "turnaround", "company", "business", "businesses", "program", "service", "services", "client", "clients"
}


def preprocess_func(text: str) -> list[str]:
    """Lowercase, tokenize, and strip stop words. Falls back to raw tokens if empty."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    filtered = [t for t in tokens if t not in STOP_WORDS]
    return filtered if filtered else tokens


class FilteredBM25Retriever(BaseRetriever):
    """
    Wraps a BM25Retriever and drops results that share no keyword with the
    query. Plain BM25 always returns its top-k regardless of relevance
    (falling back to an arbitrary corpus-order tie-break when nothing
    matches) — this guard keeps that noise out of the RRF fusion pool.
    """
    inner: object

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        docs = self.inner.invoke(query)
        query_tokens = preprocess_func(query)
        if not query_tokens:
            return docs
        return [
            doc for doc in docs
            if any(tok in doc.page_content.lower() for tok in query_tokens)
        ]


@dataclass
class RetrievedChunk:
    document: Document
    rrf_rank: int                          # 1-based position after RRF fusion, before rerank
    rerank_score: Optional[float] = None   # cross-encoder score; None if reranking disabled
    passed_gate: bool = True


@dataclass
class RetrievalTrace:
    chroma_raw: list[tuple[Document, float]] = field(default_factory=list)
    bm25_raw: list[Document] = field(default_factory=list)
    bm25_filtered: list[Document] = field(default_factory=list)
    rrf_fused: list[Document] = field(default_factory=list)
    reranked: list[RetrievedChunk] = field(default_factory=list)
    final: list[RetrievedChunk] = field(default_factory=list)


class RetrievalPipeline:
    """
    Builds and owns the Chroma store, BM25 retriever, RRF ensemble, and
    (lazily) the cross-encoder reranker for one AgentConfig. Construction
    does the expensive setup once; retrieve()/retrieve_with_trace() are
    cheap to call repeatedly.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._cross_encoder = None  # lazy-loaded on first rerank call

        emb_cls = EMBEDDING_PROVIDERS.get(config.embedding_provider)
        if not emb_cls:
            raise ValueError(
                f"Unknown embedding provider: '{config.embedding_provider}'. "
                f"Available: {list(EMBEDDING_PROVIDERS.keys())}"
            )
        embeddings = emb_cls(model_name=config.embedding_model).get_embeddings()

        from langchain_chroma import Chroma
        self.chroma_store = Chroma(
            collection_name=config.rag_collection,
            embedding_function=embeddings,
            persist_directory=config.vector_store_persist_dir,
        )

        pool_k = config.rag_candidate_pool_k
        chroma_retriever = self.chroma_store.as_retriever(search_kwargs={"k": pool_k})

        self.bm25_retriever = None
        filtered_bm25 = None
        try:
            from data_handlers.enriched_loader import EnrichedLoader
            from langchain_community.retrievers import BM25Retriever

            documents = EnrichedLoader().load()
            if documents:
                self.bm25_retriever = BM25Retriever.from_documents(
                    documents, preprocess_func=preprocess_func
                )
                self.bm25_retriever.k = pool_k
                filtered_bm25 = FilteredBM25Retriever(inner=self.bm25_retriever)
        except Exception as e:
            print(f"[Warning] Failed to initialize BM25 retriever: {e}. Falling back to semantic search only.")

        from langchain_classic.retrievers import EnsembleRetriever
        retrievers = [chroma_retriever] + ([filtered_bm25] if filtered_bm25 else [])
        weights = [0.5, 0.5] if filtered_bm25 else [1.0]
        self.ensemble = EnsembleRetriever(retrievers=retrievers, weights=weights, c=config.rrf_c)

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self.config.rerank_model)
        return self._cross_encoder

    def retrieve_with_trace(self, query: str, k: Optional[int] = None) -> RetrievalTrace:
        k = k or self.config.rag_k
        trace = RetrievalTrace()

        # Raw per-retriever visibility (for diagnostics; not used for fusion itself).
        try:
            trace.chroma_raw = self.chroma_store.similarity_search_with_score(
                query, k=self.config.rag_candidate_pool_k
            )
        except Exception as e:
            print(f"[Warning] Chroma search failed: {e}")

        if self.bm25_retriever:
            trace.bm25_raw = self.bm25_retriever.invoke(query)[: self.config.rag_candidate_pool_k]
            query_tokens = preprocess_func(query)
            trace.bm25_filtered = [
                doc for doc in trace.bm25_raw
                if not query_tokens or any(tok in doc.page_content.lower() for tok in query_tokens)
            ]

        # RRF fusion.
        trace.rrf_fused = self.ensemble.invoke(query)

        # Rerank + gate.
        if not self.config.rerank_enabled or not trace.rrf_fused:
            trace.reranked = [
                RetrievedChunk(document=doc, rrf_rank=i + 1, rerank_score=None, passed_gate=True)
                for i, doc in enumerate(trace.rrf_fused)
            ]
        else:
            pairs = [(query, doc.page_content) for doc in trace.rrf_fused]
            scores = self._get_cross_encoder().predict(pairs)
            scored = [
                RetrievedChunk(
                    document=doc,
                    rrf_rank=i + 1,
                    rerank_score=float(score),
                    # Two-tier gate: Tier A is the calibrated high-confidence
                    # threshold. Tier B rescues a candidate that RRF fusion
                    # (BM25 + semantic) independently ranked very early, even
                    # if the cross-encoder alone scored it low -- the
                    # cross-encoder has a formality bias against informal/
                    # urgent phrasing that RRF fusion doesn't share.
                    passed_gate=(
                        float(score) >= self.config.rerank_score_threshold
                        or (
                            float(score) >= self.config.rerank_rescue_score_threshold
                            and (i + 1) <= self.config.rerank_rescue_rrf_rank_cutoff
                        )
                    ),
                )
                for i, (doc, score) in enumerate(zip(trace.rrf_fused, scores))
            ]
            trace.reranked = sorted(scored, key=lambda c: c.rerank_score, reverse=True)

        trace.final = [c for c in trace.reranked if c.passed_gate][:k]
        return trace

    def retrieve(self, query: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        """Thin wrapper around retrieve_with_trace().final — guarantees the
        tool path and the diagnostic/eval path can never drift apart."""
        return self.retrieve_with_trace(query, k=k).final

    def format_for_llm(self, chunks: list[RetrievedChunk]) -> str:
        """Format chunks as Title/Section/Score blocks for the LLM. Returns
        "" if chunks is empty — the caller decides the abstain message."""
        if not chunks:
            return ""

        formatted = []
        for c in chunks:
            content = c.document.page_content.strip()
            lines = content.split("\n")
            if len(lines) >= 2 and lines[0].startswith("Title: ") and lines[1].startswith("Section: "):
                title = lines[0].replace("Title: ", "").strip()
                section = lines[1].replace("Section: ", "").strip()
                text = "\n".join(lines[2:]).strip()
            else:
                title = c.document.metadata.get("title", "Unknown Source")
                section = c.document.metadata.get("section_heading", "Unknown Section")
                text = content

            score_str = f"{c.rerank_score:.4f}" if c.rerank_score is not None else "N/A (reranking disabled)"
            formatted.append(f"Title: {title}\nSection: {section}\nScore: {score_str}\n\n{text}")

        return "\n\n---\n\n".join(formatted)
