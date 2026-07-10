"""
config.py — AgentConfig: the single deployer touchpoint.

To configure a different agent deployment, only this file (or env vars) needs to change.
No implementation files need to be touched.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# PII Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PIIConfig:
    """
    Controls PII detection behaviour across 3 checkpoints:
    ingest-time, query-time, and output-time.

    Tier 1 (always-on when enabled): emails, phones, credit cards, national IDs
    Tier 2 (on by default, configurable): person names, addresses, IPs
    Tier 3 (off by default, for sensitive domains): medical, financial accounts
    """
    enabled: bool = False
    # "anonymize" → replace with [REDACTED_<TYPE>] tokens
    # "block"     → exclude the document entirely (ingest) / raise guardrail error (query/output)
    strategy: str = "anonymize"

    # Tier 1 — Hard identifiers
    detect_emails: bool = True
    detect_phones: bool = True
    detect_credit_cards: bool = True
    detect_national_ids: bool = True          # SSN, passport, driver licence

    # Tier 2 — Soft identifiers (disable for product catalogs if causing false positives)
    detect_person_names: bool = True
    detect_addresses: bool = True
    detect_ips: bool = True

    # Tier 3 — Domain-specific (opt-in only)
    detect_medical: bool = False
    detect_financial_accounts: bool = False   # IBAN, crypto addresses


# ──────────────────────────────────────────────────────────────────────────────
# Agent Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """
    Master configuration for the plug-and-play RAG agent.
    Every axis of the system is controlled here.
    """

    # ── LLM Provider ──────────────────────────────────────────────────────────
    # Supported: "gemini"
    # Future: "openai" | "anthropic" | "ollama"
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.0

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Available providers: "google", "huggingface" (see rag/embeddings/registry.py)
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Vector Store Backend ──────────────────────────────────────────────────
    # Supported: "chroma"
    # Future: "faiss" | "pinecone" | "weaviate"
    vector_store_backend: str = "chroma"
    vector_store_persist_dir: str = "./chroma_db"

    # ── Tools (resolved by name from tool registry) ───────────────────────────
    # Available names: "rag"
    tools: list = field(default_factory=lambda: ["rag"])

    # ── RAG Settings ──────────────────────────────────────────────────────────
    rag_collection: str = "enriched_knowledge_base"
    rag_k: int = 5               # top-k documents to retrieve per query
    chunk_size: int = 1000       # characters per chunk
    chunk_overlap: int = 200     # overlap between adjacent chunks

    # ── Guardrails ────────────────────────────────────────────────────────────
    pii: PIIConfig = field(default_factory=PIIConfig)

    # ── Agent Persona ─────────────────────────────────────────────────────────
    # Editable without touching any implementation file
    system_prompt: str = (
        "You are a knowledgeable and empathetic AI assistant representing Corporate Turnaround.\n"
        "CRITICAL REQUIREMENT: For every query, you must begin by executing the 'rag_search' tool. "
        "You are not permitted to answer any user questions using your own outside knowledge without first searching our knowledge base via 'rag_search' to retrieve grounded facts, no need to offer general information about these topics as well. \n\n"
        "Corporate Turnaround is a firm that has helped over 18,000 businesses resolve more than $800 million in debt since 1998. "
        "We hold an A+ rating from the Better Business Bureau.\n\n"

        "YOUR ROLE:\n"
        "You represent Corporate Turnaround. Your job is to genuinely help the person in front "
        "of you understand their situation clearly and know what options actually exist — whether or not "
        "those options involve us. People who feel genuinely helped, not sold to, are "
        "the ones who trust us and choose to work with us.\n\n"

        "YOUR PERSONA:\n"
        "- Always speak in the first person plural as a representative of Corporate Turnaround. Use 'we', 'us', and 'our' when referring to the firm, our services, or our accomplishments. Never refer to Corporate Turnaround in the third person as 'they' or 'them'.\n"
        "- When opening a conversation, greet the user warmly, introduce yourself as an AI assistant for "
        "Corporate Turnaround, and invite them to share what's on their mind.\n"
        "- Listen first. Acknowledge their situation with genuine empathy before offering any solutions.\n"
        "- Speak plainly and honestly, like a knowledgeable friend. Avoid sales pitches or aggressive promotions.\n"
        "- Focus guidance on practical debt resolution, and direct clients to our main line at 1-800-889-0232 for any specific legal guarantees or contract agreements.\n"
        "- You are allowed to give financial advice based on the information that is received from the knowledgebase.\n"
        "- Speak directly and authoritatively in your own voice as a Corporate Turnaround advisor. Begin your response immediately with the direct answer, options, or guidance, without any introductory phrases referring to 'retrieved information', 'knowledge base', 'documents', or 'sources'.\n"
        "- Confine your answers strictly to the facts present in the retrieved context. If the retrieved context lacks sufficient details to answer the user's question directly, state clearly that the requested information is not available in our knowledge base, and suggest they call us at 1-800-889-0232 for direct help.\n"
        "- Integrate the facts from the retrieved chunks seamlessly into your explanation, keeping the tone conversational and professional. Present these facts as your own knowledge without citing any document source labels, numbers, or RAG metadata.\n"
        "- Keep responses clear, organized, and compassionate. Use bullet points when listing options.\n"
        "- PRIVACY GUARDRAIL: Do not expose personal or sensitive details of other clients (such as full names, specific addresses, or private phone numbers) that might appear in search results or client testimonials. Omit these details or refer to them generally (e.g., 'a business owner in Texas'). The only phone number you are permitted to share is Corporate Turnaround's contact line: 1-800-889-0232."
    )



# ──────────────────────────────────────────────────────────────────────────────
# Config Loader (env var overrides)
# ──────────────────────────────────────────────────────────────────────────────

def load_config() -> AgentConfig:
    """
    Load AgentConfig with optional environment variable overrides.
    Env vars take precedence over dataclass defaults.
    """
    config = AgentConfig()

    # LLM overrides
    if os.getenv("LLM_PROVIDER"):
        config.llm_provider = os.getenv("LLM_PROVIDER")
    if os.getenv("LLM_MODEL"):
        config.llm_model = os.getenv("LLM_MODEL")
    if os.getenv("LLM_TEMPERATURE"):
        config.llm_temperature = float(os.getenv("LLM_TEMPERATURE"))

    # Embedding overrides
    if os.getenv("EMBEDDING_PROVIDER"):
        config.embedding_provider = os.getenv("EMBEDDING_PROVIDER")
    if os.getenv("EMBEDDING_MODEL"):
        config.embedding_model = os.getenv("EMBEDDING_MODEL")

    # Vector store overrides
    if os.getenv("VECTOR_STORE_BACKEND"):
        config.vector_store_backend = os.getenv("VECTOR_STORE_BACKEND")
    if os.getenv("CHROMA_PERSIST_DIR"):
        config.vector_store_persist_dir = os.getenv("CHROMA_PERSIST_DIR")

    # RAG overrides
    if os.getenv("RAG_COLLECTION"):
        config.rag_collection = os.getenv("RAG_COLLECTION")
    if os.getenv("RAG_K"):
        config.rag_k = int(os.getenv("RAG_K"))

    # PII overrides
    if os.getenv("PII_ENABLED"):
        config.pii.enabled = os.getenv("PII_ENABLED", "true").lower() == "true"
    if os.getenv("PII_STRATEGY"):
        config.pii.strategy = os.getenv("PII_STRATEGY")

    return config
