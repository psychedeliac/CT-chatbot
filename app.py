import streamlit as st
import os
import sys
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from config import load_config
from core.factory import AgentFactory
from main import _setup_rag_tool, MODE_PRESETS
from langchain_core.messages import ToolMessage

# Streamlit App Setup
st.set_page_config(
    page_title="Corporate Turnaround AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_executor" not in st.session_state:
    # Build the agent
    config = load_config()
    preset = MODE_PRESETS["rag"]
    config.tools = preset["tools"]
    
    # Setup RAG tool
    _setup_rag_tool(config)
    
    # Build executor
    st.session_state.agent_executor = AgentFactory.create(config)
    st.session_state.config = config

# Helper to parse chunks
def parse_retrieved_chunks(content: str) -> list:
    if not content or content == "No relevant information found in the knowledge base for this query.":
        return []
    chunks = content.split("\n\n---\n\n")
    parsed_chunks = []
    for chunk in chunks:
        lines = chunk.strip().split("\n")
        title = "Unknown Source"
        section = "Unknown Section"
        score = "Keyword Match"
        text = chunk
        
        if len(lines) >= 3 and lines[0].startswith("Title: ") and lines[1].startswith("Section: ") and lines[2].startswith("Score: "):
            title = lines[0].replace("Title: ", "").strip()
            section = lines[1].replace("Section: ", "").strip()
            score = lines[2].replace("Score: ", "").strip()
            text = "\n".join(lines[3:]).strip()
        elif len(lines) >= 2 and lines[0].startswith("Title: ") and lines[1].startswith("Section: "):
            title = lines[0].replace("Title: ", "").strip()
            section = lines[1].replace("Section: ", "").strip()
            text = "\n".join(lines[2:]).strip()
            
        parsed_chunks.append({
            "title": title,
            "section": section,
            "score": score,
            "text": text
        })
    return parsed_chunks

# Helper to build a QA export of the conversation + retrieved chunks
def build_qa_export(messages: list, config) -> str:
    turns = []
    pending_user = None
    for msg in messages:
        if msg["role"] == "user":
            pending_user = msg["content"]
        else:
            turns.append({
                "user_message": pending_user,
                "assistant_response": msg["content"],
                "retrieved_chunks": [
                    {
                        "title": c["title"],
                        "section": c["section"],
                        "score": c["score"],
                        "text": c["text"],
                    }
                    for c in msg.get("chunks", [])
                ],
            })
            pending_user = None

    export = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "rag_collection": config.rag_collection,
        "embedding_model": config.embedding_model,
        "turns": turns,
    }
    return json.dumps(export, indent=2, ensure_ascii=False)


# Sidebar info
with st.sidebar:
    st.title("💼 Corporate Turnaround")
    st.subheader("AI Assistant Dashboard")
    st.markdown("---")
    
    config = st.session_state.config
    st.markdown("**System Settings**")
    st.text(f"LLM Provider: {config.llm_provider}")
    st.text(f"LLM Model: {config.llm_model}")
    st.text(f"Collection: {config.rag_collection}")
    st.text(f"Embeddings: {config.embedding_model}")
    
    st.markdown("---")
    st.download_button(
        "📥 Export Chat for QA",
        data=build_qa_export(st.session_state.messages, config),
        file_name=f"qa_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        disabled=not st.session_state.messages,
        help="Downloads the full conversation, including every retrieved chunk per turn, as JSON for QA review.",
    )

    st.markdown("---")
    if st.button("Clear Conversation & Reset Agent"):
        st.session_state.messages = []
        if "agent_executor" in st.session_state:
            del st.session_state.agent_executor
        st.rerun()

st.title("💬 Chat Assistant")

# Display conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Display chunks if retrieved for this message
        if msg.get("chunks"):
            with st.expander("🔍 View Retrieved Context Chunks"):
                for idx, chunk in enumerate(msg["chunks"]):
                    score_info = f" | Score: `{chunk['score']}`" if chunk.get("score") else ""
                    st.markdown(f"**Chunk {idx+1}: {chunk['title']}** (Section: *{chunk['section']}*{score_info})")
                    st.markdown(chunk["text"])
                    st.markdown("---")

# User Input
if user_input := st.chat_input("Ask a question about business debt or Corporate Turnaround..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    # Query LangGraph Agent
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and thinking..."):
            from core.utils import build_user_query, wrap_user_query
            session_config = {"configurable": {"thread_id": "session_streamlit"}}
            response = st.session_state.agent_executor.invoke(
                {"messages": [("user", wrap_user_query(build_user_query(user_input)))]},
                config=session_config,
            )
            
            # Find RAG Tool Messages from the CURRENT turn only (and deduplicate them)
            retrieved_chunks = []
            seen_texts = set()
            
            # Find the last human message index
            last_human_idx = 0
            for idx, msg in enumerate(response["messages"]):
                if msg.type == "human" or (hasattr(msg, "role") and msg.role == "user"):
                    last_human_idx = idx
                    
            # Scan messages after the last user input
            for msg in response["messages"][last_human_idx:]:
                if isinstance(msg, ToolMessage) and msg.name == "rag_search":
                    for chunk in parse_retrieved_chunks(msg.content):
                        clean_text = chunk["text"].strip()
                        if clean_text not in seen_texts:
                            seen_texts.add(clean_text)
                            retrieved_chunks.append(chunk)
            
            # Get final AI message
            final_message = response["messages"][-1].content
            if isinstance(final_message, list):
                final_message = "".join(
                    part["text"] if isinstance(part, dict) and "text" in part else str(part)
                    for part in final_message
                )
            
            # Clean RAG prefix noise
            from core.utils import clean_response_prefix, enforce_grounding_refusal
            final_message = clean_response_prefix(final_message)

            # Deterministic backstop: if rag_search found nothing this turn,
            # don't trust the LLM to have actually refused as instructed.
            final_message = enforce_grounding_refusal(response, final_message)

            st.write(final_message)
            
            # Show chunks
            if retrieved_chunks:
                with st.expander("🔍 View Retrieved Context Chunks", expanded=True):
                    for idx, chunk in enumerate(retrieved_chunks):
                        score_info = f" | Score: `{chunk['score']}`" if chunk.get("score") else ""
                        st.markdown(f"**Chunk {idx+1}: {chunk['title']}** (Section: *{chunk['section']}*{score_info})")
                        st.markdown(chunk["text"])
                        st.markdown("---")
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_message,
                "chunks": retrieved_chunks
            })
