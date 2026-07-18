"""
rag/guardrails/pii_detector.py — PII detection and anonymization.

Three checkpoints:
  1. Ingest-time  : sanitize_documents() — scrub before embedding
  2. Query-time   : sanitize_query()     — scrub user input before retrieval
  3. Output-time  : scrub_response()     — second-pass on LLM response

PII Tiers (controlled by PIIConfig):
  Tier 1 (always-on): emails, phones, credit cards, national IDs
  Tier 2 (default-on): person names, physical addresses, IP addresses
  Tier 3 (opt-in):    medical terms, financial account numbers

Backend: presidio-analyzer + presidio-anonymizer (industry standard, runs locally).
Fallback: pure-regex patterns used if presidio is not installed.
"""
import re
import warnings
from typing import List, Tuple

from langchain_core.documents import Document
from config import PIIConfig


# ── Regex Fallback Patterns ────────────────────────────────────────────────────
# Used when presidio is not installed. Covers Tier 1 only.
_REGEX_PATTERNS: dict[str, str] = {
    "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "PHONE_NUMBER":  r"\b(\+\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b",
    "CREDIT_CARD":   r"\b(?:\d[ \-]?){13,19}\b",
    "IP_ADDRESS":    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

# ── Presidio availability check ────────────────────────────────────────────────
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False
    warnings.warn(
        "\n[PII Guardrail] presidio-analyzer / presidio-anonymizer not installed.\n"
        "  Falling back to basic regex detection (Tier 1 only, lower accuracy).\n"
        "  Install with: pip install presidio-analyzer presidio-anonymizer\n"
        "  Then download the spacy model: python -m spacy download en_core_web_lg",
        stacklevel=2,
    )


class PIIGuardrail:
    """
    PII detection and anonymization engine.
    Instantiate once and reuse across all three checkpoints.
    """

    def __init__(self, pii_config: PIIConfig):
        self.config = pii_config
        self._entities = self._resolve_entities()

        if _PRESIDIO_AVAILABLE:
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()

    # ── Entity Resolution ──────────────────────────────────────────────────────

    def _resolve_entities(self) -> list[str]:
        """Map PIIConfig boolean flags to presidio entity type names."""
        entities = []

        # Tier 1 — Hard identifiers
        if self.config.detect_emails:
            entities.append("EMAIL_ADDRESS")
        if self.config.detect_phones:
            entities.append("PHONE_NUMBER")
        if self.config.detect_credit_cards:
            entities.append("CREDIT_CARD")
        if self.config.detect_national_ids:
            entities.extend(["US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE"])

        # Tier 2 — Soft identifiers
        if self.config.detect_person_names:
            entities.append("PERSON")
        if self.config.detect_addresses:
            entities.append("LOCATION")
        if self.config.detect_ips:
            entities.append("IP_ADDRESS")

        # Tier 3 — Domain-specific (opt-in)
        if self.config.detect_medical:
            entities.extend(["MEDICAL_LICENSE", "NRP"])
        if self.config.detect_financial_accounts:
            entities.extend(["IBAN_CODE", "CRYPTO"])

        return entities

    # ── Core Anonymization ─────────────────────────────────────────────────────

    def _detect_and_anonymize(self, text: str) -> Tuple[str, int]:
        """
        Detect PII entities and replace them with [REDACTED_<TYPE>] tokens,
        while never touching allowlisted strings.

        The allowlist exists because the phone detector cannot tell Corporate
        Turnaround's own published contact line from a private number. Without
        it, the assistant's answers came out reading "call us at
        [REDACTED_PHONE_NUMBER]" -- redacting the single call to action the
        system prompt requires every answer to end with.

        Allowlisted terms are swapped for inert placeholders before detection
        and restored afterwards, so the guard applies at all three checkpoints
        (ingest, query, output) without special-casing any of them.
        """
        if not text or not text.strip():
            return text, 0

        protected: dict[str, str] = {}
        for i, term in enumerate(self.config.allowlist):
            if term and term in text:
                placeholder = f"ZZALLOWLISTZZ{i}ZZ"
                protected[placeholder] = term
                text = text.replace(term, placeholder)

        cleaned, count = self._detect_and_anonymize_inner(text)

        for placeholder, term in protected.items():
            cleaned = cleaned.replace(placeholder, term)
        return cleaned, count

    def _detect_and_anonymize_inner(self, text: str) -> Tuple[str, int]:
        """Detection proper. See _detect_and_anonymize for the allowlist wrapper."""
        if not text or not text.strip():
            return text, 0

        if _PRESIDIO_AVAILABLE:
            results = self._analyzer.analyze(
                text=text,
                entities=self._entities,
                language="en",
            )
            if not results:
                return text, 0

            operators = {
                entity: OperatorConfig(
                    "replace", {"new_value": f"[REDACTED_{entity}]"}
                )
                for entity in self._entities
            }
            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=operators,
            )
            return anonymized.text, len(results)

        else:
            # Regex fallback (Tier 1 only)
            count = 0
            for label, pattern in _REGEX_PATTERNS.items():
                # Respect individual flags
                if label == "EMAIL_ADDRESS" and not self.config.detect_emails:
                    continue
                if label == "PHONE_NUMBER" and not self.config.detect_phones:
                    continue
                if label == "CREDIT_CARD" and not self.config.detect_credit_cards:
                    continue
                if label == "IP_ADDRESS" and not self.config.detect_ips:
                    continue

                matches = re.findall(pattern, text)
                if matches:
                    count += len(matches)
                    text = re.sub(pattern, f"[REDACTED_{label}]", text)

            return text, count

    # ── Checkpoint 1: Ingest-time ──────────────────────────────────────────────

    def sanitize_documents(
        self, documents: List[Document]
    ) -> Tuple[List[Document], int]:
        """
        Scrub PII from document content before embedding.

        strategy="anonymize" → replaces PII tokens in-place
        strategy="block"     → drops documents containing PII entirely

        Returns (sanitized_documents, total_entities_redacted).
        """
        if not self.config.enabled:
            return documents, 0

        sanitized: List[Document] = []
        total_entities = 0

        for doc in documents:
            clean_text, n = self._detect_and_anonymize(doc.page_content)
            total_entities += n

            if n > 0 and self.config.strategy == "block":
                # Drop this document — it contained PII
                continue

            sanitized.append(Document(page_content=clean_text, metadata=doc.metadata))

        return sanitized, total_entities

    # ── Checkpoint 2: Query-time ───────────────────────────────────────────────

    def sanitize_query(self, query: str) -> str:
        """
        Scrub PII from a user query before sending to the retriever.
        Always anonymizes (cannot block a query); warns the user.

        Returns the cleaned query string.
        """
        if not self.config.enabled:
            return query

        clean_query, n = self._detect_and_anonymize(query)
        if n > 0:
            print(
                f"[PII Guardrail] ⚠  {n} PII entity/entities detected in your query "
                f"and anonymized before retrieval."
            )
        return clean_query

    # ── Checkpoint 3: Output-time ──────────────────────────────────────────────

    def scrub_response(self, response: str) -> str:
        """
        Second-pass redaction on LLM responses before displaying to the user.
        Catches any PII that may have leaked through the retrieved context.

        Returns the scrubbed response string.
        """
        if not self.config.enabled:
            return response

        clean_response, n = self._detect_and_anonymize(response)
        if n > 0:
            print(
                f"[PII Guardrail] ⚠  {n} PII entity/entities scrubbed from agent response."
            )
        return clean_response
