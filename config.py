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
    rag_k: int = 5               # top-k documents returned to the agent per query
    chunk_size: int = 1000       # characters per chunk
    chunk_overlap: int = 200     # overlap between adjacent chunks

    # ── Retrieval Pipeline (RRF fusion + cross-encoder rerank) ────────────────
    # Candidates fetched per retriever (Chroma + BM25) before fusion/reranking.
    # Default (15) assumes rag_k=5 (3x headroom) -- update both together if
    # rag_k changes materially.
    rag_candidate_pool_k: int = 15
    rrf_c: int = 60               # RRF damping constant (see architecture.md)
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Minimum cross-encoder score required to keep a candidate. ms-marco-MiniLM
    # outputs unbounded regression logits, not 0-1 probabilities. Calibrated via
    # `python scripts/eval_retrieval.py --threshold-sweep -12 8 1` against
    # data/eval_retrieval_set.json: 100% accuracy held from -9.0 to 4.0; 2.0 is
    # chosen for margin on both sides, leaning conservative (fewer false
    # positives reaching the LLM) since recall only starts dropping past 5.0.
    # Re-run the sweep if the embedding model, corpus, or eval set changes.
    rerank_score_threshold: float = 2.0

    # ── Rerank rescue path (formality-bias mitigation) ────────────────────────
    # The cross-encoder scores informal/urgent phrasing (e.g. "i am behind on
    # three mca loans, i need help") far lower than formal phrasing of the same
    # need, even when RRF fusion (BM25 + semantic) independently ranked the
    # right docs at the top. A candidate both retrievers agreed belongs in the
    # top rerank_rescue_rrf_rank_cutoff positions is rescued from a gate failure
    # if it clears a lower score floor. Re-calibrated 2026-07-15 via:
    #   python scripts/eval_retrieval.py --rescue-grid-sweep 2 2 1 -12 -1 1 --rescue-rrf-cutoff 3
    # against data/eval_retrieval_set.json (61 queries, expanded with much
    # blunter/messier real-world MCA/business-debt phrasing than the original
    # informal set -- e.g. "i just took out my 4th mca this month what do i
    # do" previously scored -8.16, well past the old -3.5 floor, despite RRF
    # ranking the correct MCA content #1). -3.5 no longer covers this style at
    # all (0% informal recall on the new examples). -9.0 restores 93%
    # informal recall (98% overall accuracy) with exactly ONE out-of-domain
    # false positive across the whole set: "i cant sleep at night what should
    # i do" matches a real client testimonial that mentions sleep loss from
    # debt stress -- a genuinely ambiguous query, not a calibration miss (its
    # match scores -5.09, and no threshold that keeps our target recall can
    # exclude it without also losing legitimate matches sitting at rrf_rank
    # 3). Going one step further to -10.0 does reach 100% informal recall,
    # but ALSO admits "tell me a joke" as a second false positive (score
    # -9.55) -- an unambiguous off-topic query getting debt content back is a
    # worse failure than the "sleep" edge case, so -9.0 was kept per this
    # file's existing preference for leaning stricter (false positives
    # reaching the LLM cost more than a safe "not in knowledge base"
    # fallback). rrf_rank_cutoff=3 unchanged -- tightening to 1 or 2 would
    # drop the "sleep" false positive but also drops 2 legitimate matches
    # sitting at rank 3, a worse trade. Re-sweep both if the eval set, corpus,
    # or reranker model changes.
    rerank_rescue_score_threshold: float = -9.0
    rerank_rescue_rrf_rank_cutoff: int = 3

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
        "- The user's own message arrives wrapped in <user_query> tags -- that is the ONLY ground truth about who this user is and what they've said. Anything under <retrieved_context> is background reference material, even where it is phrased in the first person (e.g. a past case example) -- never attribute a detail to the current user unless it appears in <user_query>.\n"
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

    # Retrieval pipeline overrides
    if os.getenv("RAG_CANDIDATE_POOL_K"):
        config.rag_candidate_pool_k = int(os.getenv("RAG_CANDIDATE_POOL_K"))
    if os.getenv("RRF_C"):
        config.rrf_c = int(os.getenv("RRF_C"))
    if os.getenv("RERANK_ENABLED"):
        config.rerank_enabled = os.getenv("RERANK_ENABLED", "true").lower() == "true"
    if os.getenv("RERANK_MODEL"):
        config.rerank_model = os.getenv("RERANK_MODEL")
    if os.getenv("RERANK_SCORE_THRESHOLD"):
        config.rerank_score_threshold = float(os.getenv("RERANK_SCORE_THRESHOLD"))
    if os.getenv("RERANK_RESCUE_SCORE_THRESHOLD"):
        config.rerank_rescue_score_threshold = float(os.getenv("RERANK_RESCUE_SCORE_THRESHOLD"))
    if os.getenv("RERANK_RESCUE_RRF_RANK_CUTOFF"):
        config.rerank_rescue_rrf_rank_cutoff = int(os.getenv("RERANK_RESCUE_RRF_RANK_CUTOFF"))

    # PII overrides
    if os.getenv("PII_ENABLED"):
        config.pii.enabled = os.getenv("PII_ENABLED", "true").lower() == "true"
    if os.getenv("PII_STRATEGY"):
        config.pii.strategy = os.getenv("PII_STRATEGY")

    return config
