"""
rag/retriever.py — Builds the RAG tool from AgentConfig.

build_rag_tool(config) wraps a RetrievalPipeline (rag/pipeline.py) as a named
LangChain tool the agent can call. All retrieval logic (RRF fusion,
reranking, gating) lives in RetrievalPipeline — this module is purely the
LangChain adapter over it.
"""
from langchain.tools import tool

from config import AgentConfig
from rag.pipeline import get_pipeline

NO_RESULTS_MESSAGE = "No relevant information found in the knowledge base for this query."


def build_rag_tool(config: AgentConfig):
    """
    Build a RetrievalPipeline from config and return a LangChain @tool that
    the agent can call to retrieve relevant documents.

    The returned tool is registered under the name "rag" in the tool
    registry by the caller (main.py or scripts/ingest.py).
    """
    pipeline = get_pipeline(config)

    @tool
    def rag_search(query: str) -> str:
        """
        Search the indexed knowledge base to answer questions.
        Use this for domain-specific questions about products, data, or
        any content that has been ingested into the knowledge base.
        """
        chunks = pipeline.retrieve(query)
        return pipeline.format_for_llm(chunks) or NO_RESULTS_MESSAGE

    return rag_search
