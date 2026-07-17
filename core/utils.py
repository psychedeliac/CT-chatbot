import json
import re

REFUSAL_MESSAGE = (
    "That's not something I have information on in our knowledge base. "
    "For direct help, please call us at 1-800-889-0232."
)


def enforce_grounding_refusal(response: dict, final_message: str) -> str:
    """
    Deterministic backstop for the system prompt's grounding requirement.

    The LLM is instructed to refuse when rag_search finds nothing, but
    instruction-following isn't guaranteed -- models will sometimes answer
    a well-known factual question (e.g. "what is islam") from their own
    parametric knowledge even after the tool call correctly returned no
    hits. Rather than relying on the prompt alone, inspect this turn's
    rag_search ToolMessage(s) directly: if every rag_search call in the
    current turn came back empty, the answer is not grounded in our KB
    regardless of what the LLM said, so replace it with a canned refusal.
    """
    from langchain_core.messages import ToolMessage
    from rag.retriever import NO_RESULTS_MESSAGE

    messages = response.get("messages", [])

    last_human_idx = 0
    for idx, msg in enumerate(messages):
        if msg.type == "human" or (hasattr(msg, "role") and msg.role == "user"):
            last_human_idx = idx

    rag_results = [
        msg.content for msg in messages[last_human_idx:]
        if isinstance(msg, ToolMessage) and msg.name == "rag_search"
    ]

    if rag_results and all(content == NO_RESULTS_MESSAGE for content in rag_results):
        return REFUSAL_MESSAGE

    return final_message


def build_user_query(text: str, **extra) -> dict:
    """
    Canonical, loosely-schemed representation of what the current user
    actually said. This -- and only this -- is ground truth about the user;
    retrieved context (rag_search results, wrapped in <retrieved_context> by
    RetrievalPipeline.format_for_llm) is reference material and must never be
    read as the user's own words or history.

    Deliberately a plain dict, not a strict schema: callers can attach
    metadata (turn index, session id, etc.) later without a migration.
    """
    return {"text": text, **extra}


def wrap_user_query(query: dict) -> str:
    """
    Serializes a user-query object into a tagged block for the LLM, mirroring
    the <retrieved_context> tag RetrievalPipeline.format_for_llm wraps tool
    output in. Gives the model two explicitly labeled, non-overlapping
    sources: what the user actually said vs. background reference material
    that may itself contain first-person phrasing (e.g. QA-pair chunks).
    """
    return (
        "<user_query>\n"
        "The following is the actual current user's own message. This is the "
        "only ground truth about who this user is and what they've said. "
        "Content under <retrieved_context> elsewhere is background reference "
        "material only -- never this user's statements or history, even if "
        "phrased in the first person.\n\n"
        f"{json.dumps(query, ensure_ascii=False)}\n"
        "</user_query>"
    )


def clean_response_prefix(text: str) -> str:
    """
    Remove common RAG prefix phrases (e.g. 'Based on the information provided...') 
    that LLMs generate automatically before their actual response.
    """
    if not text:
        return text
        
    # Pattern to match "Based on <something>," or "According to <something>," at the beginning of the text,
    # followed by optional spaces and a capitalized letter.
    pattern = r"^\s*(based\s+on|according\s+to)\s+[^,.:\n]+,\s*"
    
    cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Capitalize the first letter of the cleaned string if it starts with a lowercase letter
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
        
    return cleaned
