import asyncio
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

from core.rag_chat import RagChat, Turn
from rag.retriever import NO_RESULTS_MESSAGE
from warmup import build_shared_config

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
if "thread_id" not in st.session_state:
    # Distinct per browser session, for logging and QA exports.
    st.session_state.thread_id = f"session_{uuid.uuid4().hex[:12]}"
if "history" not in st.session_state:
    # Conversation state for RagChat. It is stateless by design, so history
    # lives with the session that owns it -- here, the browser tab.
    st.session_state.history = []
if "chat" not in st.session_state:
    config = build_shared_config()
    # RagChat holds no per-conversation state, so this could be process-global;
    # it stays in session_state so "Clear Conversation" can rebuild it.
    st.session_state.chat = RagChat(config)
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
        st.session_state.history = []
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
        
    # Answer via the single-pass RAG path (core/rag_chat.py) -- the same engine
    # the public API runs, so what QA sees here is what visitors get.
    with st.chat_message("assistant"):
        with st.spinner("Looking into that for you..."):
            from core.utils import apply_pii_query_guard, apply_pii_response_guard

            config = st.session_state.config
            scrubbed_input = apply_pii_query_guard(user_input, config)

            # Tokens go into a placeholder, NOT straight to the page: the
            # grounding backstop inside RagChat can replace the answer
            # wholesale, and streaming raw tokens to the page would let an
            # ungrounded answer finish typing before the swap.
            stream_placeholder = st.empty()
            turn = {"streamed": "", "context": "", "answer": ""}

            async def run_turn():
                async for kind, payload in st.session_state.chat.stream(
                    st.session_state.history, scrubbed_input
                ):
                    if kind == "context":
                        turn["context"] = payload
                    elif kind == "delta":
                        turn["streamed"] += payload
                        stream_placeholder.markdown(escape_dollars(turn["streamed"]))
                    else:
                        turn["answer"] = payload.text

            try:
                asyncio.run(run_turn())
            except Exception as exc:
                stream_placeholder.empty()
                # A raw Streamlit traceback is a poor thing to show someone who
                # came here about their debt -- and it leaks internals. Log the
                # detail server-side, show a calm, actionable message.
                print(f"[Error] Turn failed: {type(exc).__name__}: {exc}")
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

            # What retrieval actually returned this turn, deduplicated.
            retrieved_chunks = []
            seen_texts = set()
            for chunk in parse_retrieved_chunks(turn["context"]):
                clean_text = chunk["text"].strip()
                if clean_text not in seen_texts:
                    seen_texts.add(clean_text)
                    retrieved_chunks.append(chunk)

            # Checkpoint 3: Output-time PII scrub (parity with main.py).
            # clean_response_prefix and the grounding backstop already ran
            # inside RagChat.
            final_message = apply_pii_response_guard(turn["answer"], config)
            st.session_state.history.append(
                Turn(user=scrubbed_input, assistant=final_message)
            )

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
