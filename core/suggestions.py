"""
core/suggestions.py — follow-up chips, taken from the corpus rather than invented.

Every strong assistant widget offers a couple of tappable next questions. The
tempting implementation is a second LLM call ("suggest three follow-ups"), which
costs a round trip per turn and, worse, happily proposes questions this KB
cannot answer -- the user taps one and gets the refusal, which is a worse
experience than no chips at all.

So suggestions are drawn from the Q&A records retrieval already surfaced for
this turn. Every chip is therefore a question with a hand-authored answer
sitting behind it, and producing them costs nothing beyond string work on
retrieval output we already have.
"""
import re

MAX_SUGGESTIONS = 3
# Longer than this wraps to three lines in a narrow chat column and stops
# reading as a tappable chip.
MAX_SUGGESTION_CHARS = 72

# Titles of Q&A records look like:
#   "Q&A: Can you negotiate or settle a Merchant Cash Advance? (mca-negotiate)"
_QA_TITLE = re.compile(r"^Q&A:\s*(?P<question>.+?)\s*(?:\([^)]*\))?\s*$")

# Words that carry no signal when deciding whether a chip repeats the question
# the user just asked.
_STOPWORDS = frozenset(
    "a an and are as at be by can co do does for from get got have how i "
    "if in is it me my of on or our so that the their there they this to "
    "we what when where which who why will with you your".split()
)

# Used only to top up a turn that surfaced fewer than `limit` chips of its own.
# The reranked pool is ~20-30 candidates and most are not Q&A records, so a
# narrow question (bankruptcy, "are you legit") can leave nothing to offer once
# the records that answered it are excluded -- and an empty chip row looks like
# a broken feature rather than a deliberate one.
#
# These are verbatim titles of canonical Q&A records, so each one still leads
# to a hand-authored answer. tests/test_suggestions.py fails if a KB rebuild
# renames or drops any of them.
EVERGREEN_QUESTIONS = (
    "What happens during a free consultation with Corporate Turnaround?",
    "How much does Corporate Turnaround cost?",
    "Can you negotiate or settle a Merchant Cash Advance?",
    "Can you actually stop collectors from calling and threatening me?",
    "Are your services available in my state?",
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOPWORDS}


def _question_from_title(title: str) -> str | None:
    """The user-facing question inside a Q&A record title, or None if this
    record is not a Q&A (site copy, IRS pages, testimonials -- none of which
    phrase themselves as a question a user would tap)."""
    match = _QA_TITLE.match(title.strip())
    if not match:
        return None
    question = match.group("question").strip()
    if len(question) > MAX_SUGGESTION_CHARS or not question.endswith("?"):
        return None
    return question


def _is_rephrasing(question: str, asked: str) -> bool:
    """True if the chip is essentially the question just answered. Offering it
    back is the most obvious way for suggestions to look broken."""
    chip, message = _content_words(question), _content_words(asked)
    if not chip:
        return True
    return len(chip & message) / len(chip) >= 0.6


def build_suggestions(
    chunks,
    message: str,
    limit: int = MAX_SUGGESTIONS,
    exclude_record_ids: frozenset = frozenset(),
) -> tuple[str, ...]:
    """
    Follow-up questions for this turn, best-scoring first.

    `chunks` is any ranked sequence of RetrievedChunk -- pass the full reranked
    candidate list rather than just the k that reached the LLM, or a turn whose
    context is all site copy offers nothing.

    `exclude_record_ids` should be the records that answered this turn. Word
    overlap alone does not catch a chip that restates the question in different
    words ("what do you charge" -> "How much does Corporate Turnaround cost?");
    the record it came from does, exactly.
    """
    suggestions: list[str] = []
    seen: set[str] = set()

    for chunk in chunks:
        question = _question_from_title(chunk.document.metadata.get("title", ""))
        if not question:
            continue
        key = question.lower()
        if chunk.document.metadata.get("record_id", "") in exclude_record_ids:
            # Marked seen, not just skipped: otherwise the evergreen top-up
            # below re-offers the very question this turn just answered.
            seen.add(key)
            continue
        if key in seen or _is_rephrasing(question, message):
            continue
        seen.add(key)
        suggestions.append(question)
        if len(suggestions) == limit:
            return tuple(suggestions)

    for question in EVERGREEN_QUESTIONS:
        if len(suggestions) == limit:
            break
        key = question.lower()
        if key in seen or _is_rephrasing(question, message):
            continue
        seen.add(key)
        suggestions.append(question)

    return tuple(suggestions)
