"""
rag/retriever.py — what an empty retrieval looks like to the rest of the app.

The LangChain @tool wrapper that used to live here went away with the ReAct
agent: core/rag_chat.py calls RetrievalPipeline directly. This message stays
because the grounding backstop and the QA UI both key off it.
"""
NO_RESULTS_MESSAGE = "No relevant information found in the knowledge base for this query."
