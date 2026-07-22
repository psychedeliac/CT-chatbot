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

VALID_PII_STRATEGIES = {"anonymize", "block"}


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

    # Checkpoint 1 (ingest-time) is OFF even when PII guarding is enabled.
    # This knowledge base is public marketing, regulatory and educational
    # content -- it contains no third-party PII to protect, and scrubbing it
    # measurably destroys the corpus: enabling this redacted Corporate
    # Turnaround's OWN contact number (the one number the assistant is required
    # to give out), turned the title "About Us" into "About [REDACTED_LOCATION]"
    # (spaCy reads "Us" as the United States), and stripped named credentials,
    # damaging 120 of 798 documents. Real user PII arrives at checkpoints 2 and
    # 3 (query-time and output-time), which stay on. Only enable this if a
    # corpus genuinely carries third-party personal data.
    scrub_on_ingest: bool = False

    # Strings the PII detector must never redact. The phone recognizer cannot
    # distinguish the company's own published contact line from a private
    # number, so without this the assistant literally answered "call us at
    # [REDACTED_PHONE_NUMBER]" -- destroying the call to action every answer is
    # required to end with. Applies to all three checkpoints.
    allowlist: tuple = (
        # New enquiries / free consultation
        "1-800-889-0232",
        "800-889-0232",
        "800.889.0232",
        # Existing-client service and creditor inquiries (verified on the live
        # site 2026-07-18). Allowlisted so it survives scrubbing when the
        # assistant legitimately gives it to an existing client.
        "1-800-411-1113",
        "800-411-1113",
        "800.411.1113",
    )

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
# HTTP API Configuration (api/)
# ──────────────────────────────────────────────────────────────────────────────

def _csv_env(name: str) -> tuple:
    raw = os.getenv(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class APIConfig:
    """
    Settings for the public HTTP API that backs the website chat widget.

    Every field is env-overridable via load_api_config() so the same image can
    run in dev and prod. Defaults are the safe/dev values: CORS closed to
    everything except localhost, conservative rate limits.
    """

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Exact origins allowed to call the API from a browser. A public widget is
    # embedded on known domains, so this is an allowlist, never "*" -- "*" plus
    # credentials is rejected by browsers anyway, and without it any site could
    # burn the LLM quota from a visitor's IP.
    allowed_origins: tuple = ("http://localhost:3000", "http://localhost:8501")

    # ── Rate limiting (per client IP) ─────────────────────────────────────────
    # Two windows: a burst allowance for a human typing quickly, and an hourly
    # cap so a single IP cannot drain the daily LLM quota.
    rate_limit_burst: str = "10/minute"
    rate_limit_sustained: str = "100/hour"
    # "memory://" keeps counters in-process -- correct for ONE instance only.
    # Multi-instance deployments must point this at Redis
    # (RATE_LIMIT_STORAGE_URI=redis://host:6379) or each replica enforces its
    # own separate limit.
    rate_limit_storage_uri: str = "memory://"
    # Header naming the real client IP when running behind a proxy/CDN. Left
    # unset by default: trusting X-Forwarded-For when nothing strips it lets a
    # caller spoof their IP and bypass rate limiting entirely.
    client_ip_header: str = ""

    # ── Input limits ──────────────────────────────────────────────────────────
    max_message_chars: int = 2000

    # ── Concurrency & timeouts ────────────────────────────────────────────────
    # Bounds how many LLM/retrieval turns run at once. Past this, requests wait;
    # past the queue timeout they get 503 rather than piling up until the box
    # dies. Sized for the LLM provider's rate limit, not the CPU.
    max_concurrent_turns: int = 8
    queue_timeout_seconds: float = 10.0
    turn_timeout_seconds: float = 90.0

    # ── Sessions ──────────────────────────────────────────────────────────────
    # Conversation state lives in LangGraph's in-process MemorySaver, so it is
    # bounded here or it grows until the process is OOM-killed.
    session_ttl_seconds: int = 3600
    max_sessions: int = 5000

    # ── Answer cache ──────────────────────────────────────────────────────────
    # First-turn answers only (see api/chat.py). On a public site the same
    # handful of questions dominate traffic; caching them cuts both latency and
    # LLM spend. Follow-up turns are never cached -- they depend on history.
    answer_cache_enabled: bool = True
    answer_cache_size: int = 512
    answer_cache_ttl_seconds: int = 1800


def load_api_config() -> APIConfig:
    """Build APIConfig from env vars, falling back to the dataclass defaults."""
    defaults = APIConfig()
    origins = _csv_env("CORS_ALLOWED_ORIGINS") or defaults.allowed_origins

    def _int(name: str, fallback: int) -> int:
        raw = os.getenv(name)
        return int(raw) if raw else fallback

    def _float(name: str, fallback: float) -> float:
        raw = os.getenv(name)
        return float(raw) if raw else fallback

    config = APIConfig(
        allowed_origins=origins,
        rate_limit_burst=os.getenv("RATE_LIMIT_BURST", defaults.rate_limit_burst),
        rate_limit_sustained=os.getenv("RATE_LIMIT_SUSTAINED", defaults.rate_limit_sustained),
        rate_limit_storage_uri=os.getenv("RATE_LIMIT_STORAGE_URI", defaults.rate_limit_storage_uri),
        client_ip_header=os.getenv("CLIENT_IP_HEADER", defaults.client_ip_header),
        max_message_chars=_int("MAX_MESSAGE_CHARS", defaults.max_message_chars),
        max_concurrent_turns=_int("MAX_CONCURRENT_TURNS", defaults.max_concurrent_turns),
        queue_timeout_seconds=_float("QUEUE_TIMEOUT_SECONDS", defaults.queue_timeout_seconds),
        turn_timeout_seconds=_float("TURN_TIMEOUT_SECONDS", defaults.turn_timeout_seconds),
        session_ttl_seconds=_int("SESSION_TTL_SECONDS", defaults.session_ttl_seconds),
        max_sessions=_int("MAX_SESSIONS", defaults.max_sessions),
        answer_cache_enabled=os.getenv("ANSWER_CACHE_ENABLED", "true").lower() == "true",
        answer_cache_size=_int("ANSWER_CACHE_SIZE", defaults.answer_cache_size),
        answer_cache_ttl_seconds=_int("ANSWER_CACHE_TTL_SECONDS", defaults.answer_cache_ttl_seconds),
    )

    if "*" in config.allowed_origins:
        raise ValueError(
            "CORS_ALLOWED_ORIGINS must list explicit origins, not '*'. A wildcard "
            "lets any site drive this API from a visitor's browser and IP."
        )
    return config


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
    # Alias rather than a pinned version: Google retires specific versions for
    # new API keys (gemini-2.5-flash now 404s), which breaks the app silently.
    # flash-lite over flash: measured 1.80s vs 5.70s for the same grounded
    # answer, and generation is the largest single component of response time.
    # The tradeoff is weaker instruction-following, which matters here because
    # the compliance rules in system_prompt are instructions -- so this leans on
    # the deterministic backstops (enforce_grounding_refusal, the PII guards)
    # rather than on the model behaving. Set LLM_MODEL=gemini-flash-latest to
    # revert if answer quality regresses.
    llm_model: str = "gemini-flash-lite-latest"
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
    # Tokens of each candidate the cross-encoder actually reads. This is the
    # single biggest lever on answer latency: reranking ~28 candidates costs
    # ~3.0s at 512 tokens and ~0.8s at 128 on a 4-thread CPU, and it sits
    # directly in the user's wait. Lowering it truncates each chunk, which can
    # change scores -- validate any change with scripts/eval_retrieval.py
    # rather than by eye. Measured 2026-07-22 with scripts/eval_retrieval.py
    # over the 77-query eval set: accuracy/precision/recall all 1.00 and
    # out-of-domain FP rate 0.00 at 512, 256, 192 AND 128 -- identical results,
    # so the shorter window costs nothing measurable here. Chunks average ~860
    # chars (~215 tokens) and carry their Contextual Retrieval prefix up front,
    # which is why truncating the tail does not change what they score as.
    # Re-run the eval if the chunker, corpus, or reranker model changes.
    rerank_max_length: int = 128

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
    # RECALIBRATED 2026-07-18 after the corpus was rebuilt (re-scraped with a
    # sentence-aware chunker plus Contextual Retrieval prefixes; 799 -> 592
    # records). Re-swept with:
    #   python scripts/eval_retrieval.py --rescue-grid-sweep 0 4 1 -14 -1 1 --rescue-rrf-cutoff 3
    # On the rebuilt corpus the old -9.0 leaks an out-of-domain query into debt
    # content; -7.0 does not. Verified end state at -7.0, after the Contextual
    # Retrieval prefixes were stripped of "This chunk describes..." boilerplate:
    #   accuracy 0.97, recall 0.98, precision 0.98, out-of-domain FP rate 0.00,
    #   informal in-domain recall 0.97 (was 0.66 under the stale .env override).
    # The primary threshold made no difference anywhere in the sweep (identical
    # results 0.0 through 4.0), so it keeps its existing value.
    #
    # NOTE: these are overridable by RERANK_* env vars, and an override in .env
    # silently wins over everything above. A stale .env is why this pipeline ran
    # at 0.82 accuracy / 0.66 informal recall for a while. Check the env before
    # concluding the corpus regressed.
    rerank_rescue_score_threshold: float = -7.0
    rerank_rescue_rrf_rank_cutoff: int = 3

    # ── Authority-aware rescue (KB v2) ────────────────────────────────────────
    # Records with authority == "canonical" are hand-authored answers that are
    # safe by construction — their text IS the approved response (including
    # scope guards whose "answer" is the correct deflection for fee/savings/
    # legal questions). Missing one costs a correct answer; admitting one can't
    # produce a wrong answer. They therefore get a wider RRF rescue window than
    # third-party background content ("what do you charge" put the fee guard at
    # RRF rank 8 behind six loan-charge-off chunks — inside no sane cutoff for
    # background content, but exactly what this wider cutoff is for).
    rerank_canonical_rescue_rrf_rank_cutoff: int = 10

    # Allow the pipeline to start with BM25 missing (semantic-only retrieval).
    # Off by default: silent degradation is how a missing rank_bm25 install went
    # unnoticed for a whole commit while the system advertised hybrid search.
    allow_degraded_retrieval: bool = False

    # ── Guardrails ────────────────────────────────────────────────────────────
    pii: PIIConfig = field(default_factory=PIIConfig)

    # ── Agent Persona ─────────────────────────────────────────────────────────
    # Editable without touching any implementation file
    system_prompt: str = (
        "You are a knowledgeable and empathetic AI assistant representing Corporate Turnaround.\n"
        "CRITICAL REQUIREMENT: For every query, you must begin by executing the 'rag_search' tool. "
        "You are not permitted to answer any user questions using your own outside knowledge without first searching our knowledge base via 'rag_search' to retrieve grounded facts, no need to offer general information about these topics as well. \n"
        "SEARCH QUERY REWRITING: 'rag_search' receives ONLY the query string you pass it -- it cannot see the conversation. So rewrite the user's message into a STANDALONE query that resolves pronouns and references to earlier turns. If they asked about merchant cash advances and then say 'how do I get out of it?', search for something like 'how to get out of a merchant cash advance', never the bare 'how do I get out of it?'.\n\n"
        # Every figure here must be substantiable from corporateturnaround.com.
        # The previous wording claimed "over 18,000 businesses" and "more than
        # $800 million in debt" -- neither phrase appears anywhere on the live
        # site (verified by re-scrape 2026-07-18). Unsubstantiated performance
        # claims are exactly what the FTC pursues in debt relief, so this now
        # states only what the site itself says.
        "Corporate Turnaround has worked with over 10,000 small business owners since 1998, "
        "and is rated A+ with the Better Business Bureau.\n"
        "Do not inflate, round up, or add to these figures, and do not volunteer any other "
        "statistic about our track record, amounts resolved, or success rates.\n\n"

        # Length is a latency setting, not just a style one: this assistant
        # answers in a chat widget on a website, where every extra sentence is
        # another second the person watches a cursor blink. Generation is the
        # single largest component of response time (~60 tokens/sec), so an
        # unbounded "comprehensive" answer costs 5s+ where a focused one costs
        # under 2s. It also reads better in a narrow chat column.
        "LENGTH — THIS IS A HARD RULE:\n"
        "- Keep every reply under 110 words. Shorter is better.\n"
        "- Lead with the direct answer in one or two sentences. No preamble.\n"
        "- Use at most 3 short bullets, and only when genuinely listing options.\n"
        "- Answer what was actually asked. Do not pre-empt follow-up questions or "
        "explain adjacent topics -- invite them to ask instead.\n\n"

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
        "- Confine your answers strictly to the facts present in the retrieved context. Never use the words "
        "'knowledge base', 'documents', 'sources', 'search results', or 'retrieved' in a reply -- the user is "
        "talking to Corporate Turnaround, not to a search engine.\n\n"

        "WHEN rag_search RETURNS NO RESULTS (or nothing relevant), do NOT answer from your own knowledge. "
        "Instead pick the response that fits the situation, keep it under 60 words, and never state any fact, "
        "figure, or claim about debt, finance, or our company in these replies:\n"
        "- Greeting or small talk ('hello', 'how are you'): greet them warmly, introduce yourself as Corporate "
        "Turnaround's AI assistant, and invite them to share what's going on with their business. No phone number needed.\n"
        "- Off-topic requests (weather, jokes, poems, sports, general knowledge): decline with warmth and a light "
        "touch, then steer back: you're here to help with business debt questions.\n"
        "- On-topic question you simply lack details for: say honestly that you don't have the specifics on that, "
        "and offer the free consultation at 1-800-889-0232 where a specialist can answer directly. If they are an "
        "existing client, give 1-800-411-1113 instead.\n"
        "- Distress with no specific question ('I'm going to lose everything'): lead with genuine empathy, remind "
        "them this is solvable and they're not alone, and encourage the free consultation at 1-800-889-0232. Then "
        "invite them to tell you more so you can actually help in chat.\n"
        "Apply the same judgment when rag_search DOES return results but they don't genuinely fit the request: "
        "an off-topic ask (a poem, a joke) that happens to match debt articles still gets the warm off-topic "
        "deflection, and a personal-finance matter (personal taxes, personal credit cards) gets a note that we "
        "specialize in BUSINESS debt, plus an answer only if the retrieved content truly applies to them.\n"
        "- The user's own message arrives wrapped in <user_query> tags -- that is the ONLY ground truth about who this user is and what they've said. Anything under <retrieved_context> is background reference material, even where it is phrased in the first person (e.g. a past case example) -- never attribute a detail to the current user unless it appears in <user_query>.\n"
        "- Integrate the facts from the retrieved chunks seamlessly into your explanation, keeping the tone conversational and professional. Present these facts as your own knowledge without citing any document source labels, numbers, or RAG metadata.\n"
        "- Keep responses clear, organized, and compassionate. Use bullet points when listing options.\n"
        "\nCOMPLIANCE (this is a regulated debt-relief context -- incorrect or overstated "
        "information from an AI assistant is itself a violation risk, not merely a quality issue):\n"
        "- If a retrieved chunk carries a [COMPLIANCE: ...] marker, it describes a specific outcome or "
        "regulated process. Never restate it as a guaranteed or typical result. Frame it as one client's "
        "situation and make clear that results vary with the individual circumstances.\n"
        "- Never quote, estimate, or commit to fees, settlement percentages, savings amounts, or timeframes. "
        "These carry mandatory disclosure requirements that cannot be met in a chat message. When asked, don't "
        "refuse coldly: explain that fees and outcomes genuinely depend on their specific debts and budget, which "
        "is exactly what the free consultation works out -- then give 1-800-889-0232. Answer what you CAN "
        "(e.g. the consultation is free and no-obligation) before deflecting what you can't.\n"
        "- Never advise anyone to stop paying their creditors, and never state or predict what will happen "
        "to their credit score or their legal position.\n"
        "- If you are asked whether you are a human, say plainly and immediately that you are an AI assistant.\n"
        "- If someone describes an emergency, threatened legal action, or a deadline within days, tell them "
        "directly to call 1-800-889-0232 rather than working through it in chat.\n\n"

        "- PRIVACY GUARDRAIL: Do not expose personal or sensitive details of other clients (such as full names, specific addresses, or private phone numbers) that might appear in search results or client testimonials. Omit these details or refer to them generally (e.g., 'a business owner in Texas').\n"
        "- PHONE NUMBERS: Exactly two numbers may ever be shared, and only in their correct context. "
        "New enquiries and free consultations: 1-800-889-0232 -- this is the default for anyone asking for help. "
        "Existing clients asking about their own account, or creditors making an inquiry: 1-800-411-1113. "
        "Never share any other number that appears in retrieved content, and never give the client-service line to a prospective customer."
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
    if os.getenv("RERANK_MAX_LENGTH"):
        config.rerank_max_length = int(os.getenv("RERANK_MAX_LENGTH"))
    if os.getenv("RERANK_RESCUE_SCORE_THRESHOLD"):
        config.rerank_rescue_score_threshold = float(os.getenv("RERANK_RESCUE_SCORE_THRESHOLD"))
    if os.getenv("RERANK_RESCUE_RRF_RANK_CUTOFF"):
        config.rerank_rescue_rrf_rank_cutoff = int(os.getenv("RERANK_RESCUE_RRF_RANK_CUTOFF"))
    if os.getenv("ALLOW_DEGRADED_RETRIEVAL"):
        config.allow_degraded_retrieval = os.getenv("ALLOW_DEGRADED_RETRIEVAL", "false").lower() == "true"

    # PII overrides
    if os.getenv("PII_ENABLED"):
        config.pii.enabled = os.getenv("PII_ENABLED", "true").lower() == "true"
    if os.getenv("PII_STRATEGY"):
        config.pii.strategy = os.getenv("PII_STRATEGY")
    if os.getenv("PII_SCRUB_ON_INGEST"):
        config.pii.scrub_on_ingest = os.getenv("PII_SCRUB_ON_INGEST", "false").lower() == "true"

    # The rerank thresholds are calibrated against data/eval_retrieval_set.json.
    # An override silently replaces that calibration, and a stale .env carrying
    # copied example values is exactly how this pipeline ended up running at 0.82
    # accuracy / 0.66 informal recall while config.py still documented the tuned
    # numbers. Say so at startup rather than letting it pass unnoticed.
    calibration_overrides = {
        name: os.getenv(name)
        for name in ("RERANK_SCORE_THRESHOLD", "RERANK_RESCUE_SCORE_THRESHOLD",
                     "RERANK_RESCUE_RRF_RANK_CUTOFF", "RAG_CANDIDATE_POOL_K")
        if os.getenv(name)
    }
    if calibration_overrides:
        print(
            "[Config] Calibrated retrieval settings are overridden by environment "
            f"variables: {calibration_overrides}. These replace the values tuned via "
            "scripts/eval_retrieval.py. Remove them from .env unless the override is "
            "deliberate and re-swept."
        )

    # Fail loudly on an unsupported strategy. An unrecognized value (e.g.
    # "redact") previously fell through every branch in PIIGuardrail, so PII
    # guarding silently did nothing while the startup banner still reported
    # it as enabled.
    if config.pii.strategy not in VALID_PII_STRATEGIES:
        raise ValueError(
            f"Invalid PII_STRATEGY '{config.pii.strategy}'. "
            f"Supported: {sorted(VALID_PII_STRATEGIES)}"
        )

    return config
