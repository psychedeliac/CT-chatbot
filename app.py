import streamlit as st
import os
import sys
import json
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from config import load_config
from core.factory import AgentFactory
from main import _setup_rag_tool, MODE_PRESETS
from rag.retriever import NO_RESULTS_MESSAGE
from langchain_core.messages import ToolMessage

# Streamlit App Setup
st.set_page_config(
    page_title="Corporate Turnaround AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner="Loading knowledge base and retrieval models...")
def build_shared_config():
    """
    Build config and register the RAG tool ONCE per server process.

    RetrievalPipeline construction is the expensive part of startup: it builds a
    BM25 index over the whole corpus and loads a cross-encoder, ~40s. That was
    happening per browser session, which is fine for a demo and untenable under
    real traffic. st.cache_resource shares one instance across sessions.

    Only the retrieval machinery is shared. The agent itself stays per-session
    below, because its MemorySaver holds conversation state -- sharing that
    would leak one user's conversation into another's.
    """
    config = load_config()
    config.tools = MODE_PRESETS["rag"]["tools"]
    _setup_rag_tool(config)

    # Warm the cross-encoder. RetrievalPipeline loads it lazily on first rerank
    # (rag/pipeline.py:_get_cross_encoder), which cost ~14s -- paid by whichever
    # real user happened to ask the first question after a deploy or restart.
    # Running one throwaway query here moves that cost into server boot, where
    # nobody is waiting on it.
    from core.tools.registry import get_tools
    try:
        get_tools(["rag"])[0].invoke({"query": "warmup"})
    except Exception as exc:
        # A failed warmup must not take the app down -- the model still loads
        # lazily on the first real query, just slowly.
        print(f"[Warning] Retrieval warmup failed (first query will be slow): {exc}")

    return config


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    # Distinct per browser session. The previous hard-coded "session_streamlit"
    # was only safe by accident -- it relied on each session happening to build
    # its own MemorySaver.
    st.session_state.thread_id = f"session_{uuid.uuid4().hex[:12]}"
if "agent_executor" not in st.session_state:
    config = build_shared_config()
    st.session_state.agent_executor = AgentFactory.create(config)
    st.session_state.config = config

# Helper to parse chunks
# Q&A chunks use a "Similar past case" label instead of "Section" (see
# RetrievalPipeline.format_for_llm), which the old Section-only parser failed
# on -- those chunks rendered as "Unknown Source / Unknown Section".
SECTION_PREFIXES = ("Section: ", "Similar past case (NOT the current user): ")


def escape_dollars(text: str) -> str:
    """
    Streamlit's markdown reads a pair of '$' as LaTeX math, so "saved $207,000
    of $262,000" renders as garbled math instead of two dollar amounts. In a
    business-debt assistant nearly every substantive answer contains at least
    two dollar figures, so this has to be escaped everywhere text is displayed.
    """
    return text.replace("$", r"\$") if text else text


def parse_retrieved_chunks(content: str) -> list:
    if not content or content == NO_RESULTS_MESSAGE:
        return []

    # Strip the <retrieved_context> wrapper the pipeline adds for the LLM.
    body = content.split("</retrieved_context>")[0]
    if "<retrieved_context>" in body:
        body = body.split("\n\n", 1)[-1]

    parsed_chunks = []
    for chunk in body.split("\n\n---\n\n"):
        lines = chunk.strip().split("\n")
        title = "Unknown Source"
        section = "Unknown Section"
        score = "Keyword Match"
        text = chunk

        section_prefix = next(
            (p for p in SECTION_PREFIXES if len(lines) >= 2 and lines[1].startswith(p)),
            None,
        )
        if lines and lines[0].startswith("Title: ") and section_prefix:
            title = lines[0].replace("Title: ", "").strip()
            section = lines[1][len(section_prefix):].strip().strip('"')
            if len(lines) >= 3 and lines[2].startswith("Score: "):
                score = lines[2].replace("Score: ", "").strip()
                text = "\n".join(lines[3:]).strip()
            else:
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

# Suggested starter questions, shown only on the empty state. A button click
# triggers a Streamlit rerun on which it returns True; the prompt then flows
# through the same path as typed chat input below.
STARTER_PROMPTS = (
    "What does Corporate Turnaround do?",
    "I'm behind on my merchant cash advance payments — what are my options?",
    "How is debt restructuring different from bankruptcy?",
    "Is the consultation really free?",
)

queued_prompt = None
if not st.session_state.messages:
    st.markdown(
        "#### 👋 Welcome!\n"
        "I'm Corporate Turnaround's AI assistant. Since 1998 we've helped over "
        "10,000 small business owners work through business debt. Tell me what's "
        "going on with your business, or start with one of these:"
    )
    starter_cols = st.columns(2)
    for i, prompt_text in enumerate(STARTER_PROMPTS):
        if starter_cols[i % 2].button(prompt_text, use_container_width=True):
            queued_prompt = prompt_text

# Display conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(escape_dollars(msg["content"]))
        
        # Display chunks if retrieved for this message
        if msg.get("chunks"):
            with st.expander("🔍 View Retrieved Context Chunks"):
                for idx, chunk in enumerate(msg["chunks"]):
                    score_info = f" | Score: `{chunk['score']}`" if chunk.get("score") else ""
                    st.markdown(f"**Chunk {idx+1}: {chunk['title']}** (Section: *{chunk['section']}*{score_info})")
                    st.markdown(escape_dollars(chunk["text"]))
                    st.markdown("---")

# User Input (typed, or a starter prompt clicked above)
if user_input := (st.chat_input("Ask a question about business debt or Corporate Turnaround...") or queued_prompt):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    # Query LangGraph Agent
    with st.chat_message("assistant"):
        with st.spinner("Looking into that for you..."):
            from core.utils import (
                build_user_query,
                wrap_user_query,
                apply_pii_query_guard,
                apply_pii_response_guard,
            )

            config = st.session_state.config
            scrubbed_input = apply_pii_query_guard(user_input, config)

            session_config = {"configurable": {"thread_id": st.session_state.thread_id}}

            # Stream rather than invoke. The work takes the same ~6-10s either
            # way, but invoke() showed a blank spinner for all of it; streaming
            # puts words on screen at ~2s.
            #
            # Tokens go into a placeholder, NOT straight to the page, because
            # the post-processing chain below (clean_response_prefix ->
            # enforce_grounding_refusal -> PII guard) can rewrite or wholly
            # replace the text. Streaming it directly would let an ungrounded
            # answer finish typing out before the backstop swapped it.
            #
            # stream_mode=["messages", "values"] gives both: "messages" for
            # per-token chunks, "values" for the full final state that the
            # chunk-parsing and grounding backstop below still need.
            stream_placeholder = st.empty()
            streamed_text = ""
            response = None
            try:
                for mode, payload in st.session_state.agent_executor.stream(
                    {"messages": [("user", wrap_user_query(build_user_query(scrubbed_input)))]},
                    config=session_config,
                    stream_mode=["messages", "values"],
                ):
                    if mode == "values":
                        response = payload
                    elif mode == "messages":
                        chunk, _metadata = payload
                        # Only the assistant's prose. Tool-call argument chunks
                        # and ToolMessage results must not reach the user.
                        if isinstance(chunk, ToolMessage) or not chunk.content:
                            continue
                        text = chunk.content
                        if isinstance(text, list):
                            text = "".join(
                                part["text"] if isinstance(part, dict) and "text" in part else str(part)
                                for part in text
                            )
                        streamed_text += text
                        stream_placeholder.markdown(escape_dollars(streamed_text))

                if response is None:
                    raise RuntimeError("Agent stream ended without emitting final state")
            except Exception as exc:
                stream_placeholder.empty()
                # A raw Streamlit traceback is a poor thing to show someone who
                # came here about their debt -- and it leaks internals. Log the
                # detail server-side, show a calm, actionable message.
                print(f"[Error] Agent invocation failed: {type(exc).__name__}: {exc}")
                error_text = (
                    "Sorry — something went wrong on our end while looking that up. "
                    "Please try again in a moment, or call us at 1-800-889-0232 and "
                    "we can help you directly."
                )
                st.error(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text, "chunks": []}
                )
                st.stop()

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

            # Checkpoint 3: Output-time PII scrub (parity with main.py).
            final_message = apply_pii_response_guard(final_message, config)

            # Overwrite the streamed text with the post-processed version. Same
            # content in the common case; the swap only shows when a guard
            # above actually rewrote the answer.
            stream_placeholder.markdown(escape_dollars(final_message))

            # Show chunks
            if retrieved_chunks:
                with st.expander("🔍 View Retrieved Context Chunks", expanded=True):
                    for idx, chunk in enumerate(retrieved_chunks):
                        score_info = f" | Score: `{chunk['score']}`" if chunk.get("score") else ""
                        st.markdown(f"**Chunk {idx+1}: {chunk['title']}** (Section: *{chunk['section']}*{score_info})")
                        st.markdown(escape_dollars(chunk["text"]))
                        st.markdown("---")
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_message,
                "chunks": retrieved_chunks
            })
