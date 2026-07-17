"""
scripts/scrapers/brainstorm_qa_questions.py — Real-world question brainstorming.

Generates messy, distressed, real-world-phrased questions per knowledge-base
topic (dollar amounts, multiple debts, typos, urgency, emotional framing) —
NOT the clean textbook phrasing a topic list naturally suggests. Style is
anchored to the informal entries already validated in
data/eval_retrieval_set.json, so "real-world phrasing" has one source of
truth instead of drifting between this prompt and the eval set.

This produces QUESTIONS ONLY. Answers are synthesized separately in
synthesize_qa_answers.py, grounded against the actual retrieval pipeline —
never invented here.

Usage:
    python scripts/scrapers/brainstorm_qa_questions.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from scripts.scrapers.utils import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_SET_PATH = os.path.join(BASE_DIR, "data", "eval_retrieval_set.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "qa_questions_brainstormed.json")

QA_TOPICS = [
    # MCA pillar
    ("mca-basics",        "mca-education",    "What is a Merchant Cash Advance (MCA)?"),
    ("mca-factor-rate",   "mca-education",    "How do MCA factor rates work and what do they cost?"),
    ("mca-daily-ach",     "mca-education",    "How do daily ACH debits from MCAs drain business cash flow?"),
    ("mca-stacking",      "mca-education",    "What is MCA stacking and why is it dangerous?"),
    ("mca-ucc-lien",      "mca-education",    "What is a UCC-1 lien and how do MCA lenders use it?"),
    ("mca-coj",           "mca-education",    "What is a Confession of Judgment (COJ) in an MCA contract?"),
    ("mca-default",       "mca-education",    "What happens when you default on a Merchant Cash Advance?"),
    ("mca-vs-loan",       "mca-education",    "How is an MCA different from a business loan?"),
    ("mca-negotiate",     "mca-education",    "Can you negotiate or settle a Merchant Cash Advance?"),
    ("mca-personal-guar", "mca-education",    "Do MCAs require a personal guarantee?"),
    # Multi-debt overwhelm is its own topic, distinct from single-MCA-default
    # phrasing above: the retrieval gap is specifically owners juggling
    # several simultaneous advances, not one.
    ("mca-multi-debt",    "mca-education",    "What should a business owner do when juggling multiple simultaneous MCAs they can't keep up with?"),

    # Business debt pillar
    ("biz-debt-types",    "business-debt",    "What types of debt do small businesses typically have?"),
    ("biz-credit-score",  "business-debt",    "How does business credit work vs. personal credit?"),
    ("biz-personal-guar", "business-debt",    "What does signing a personal guarantee mean for a business owner?"),
    ("biz-sba-default",   "business-debt",    "What happens if you default on an SBA loan?"),
    ("biz-ch7-vs-ch11",   "business-debt",    "What is the difference between Chapter 7 and Chapter 11 bankruptcy?"),
    ("biz-out-of-court",  "business-debt",    "What is out-of-court debt restructuring?"),
    ("biz-vendor-debt",   "business-debt",    "What happens when you can't pay your vendors?"),
    ("biz-ucc-search",    "business-debt",    "How do you find UCC liens filed against your business?"),
    # Dollar-amount-driven overwhelm: owners describing their situation by a
    # specific total-debt figure rather than naming a debt type at all.
    ("biz-dollar-overwhelm", "business-debt", "What should a business owner do when they're overwhelmed by a large, specific total amount of business debt (e.g. $50,000+)?"),

    # Corporate Turnaround process pillar
    ("ct-what-is",        "ct-process",       "What does Corporate Turnaround do?"),
    ("ct-consultation",   "ct-process",       "What happens during a free consultation with Corporate Turnaround?"),
    ("ct-eligibility",    "ct-process",       "Who qualifies for Corporate Turnaround's program?"),
    ("ct-program-steps",  "ct-process",       "What are the steps in the Corporate Turnaround program?"),
    ("ct-debts-handled",  "ct-process",       "What types of debt does Corporate Turnaround handle?"),
    ("ct-debts-not",      "ct-process",       "What debts can Corporate Turnaround NOT help with?"),
    ("ct-fees",           "ct-process",       "How does Corporate Turnaround charge for its services?"),
    ("ct-timeline",       "ct-process",       "How long does the Corporate Turnaround program take?"),
    ("ct-credit-impact",  "ct-process",       "How does enrolling in Corporate Turnaround affect my credit score?"),
    ("ct-vs-bankruptcy",  "ct-process",       "Why choose Corporate Turnaround over filing for bankruptcy?"),
    ("ct-creditor-calls", "ct-process",       "What should I say to creditors while in the Corporate Turnaround program?"),
    ("ct-score-partner",  "ct-process",       "What is SCORE and how does it relate to Corporate Turnaround?"),
    ("ct-mca-help",       "ct-process",       "Can Corporate Turnaround help settle a Merchant Cash Advance?"),
    ("ct-industries",     "ct-process",       "What industries does Corporate Turnaround serve?"),
    ("ct-success-stats",  "ct-process",       "What are Corporate Turnaround's track record and success statistics?"),
    # Bridges generic/topic-specific "what can I do" advice-seeking phrasing
    # (no mention of "Corporate Turnaround" by name) to the concrete named
    # service lines on corpo-nine.vercel.app/services. Without these, that
    # content only retrieves when a query already names the company --
    # generic distress phrasing ("i have 3 mca loans what can i do") never
    # surfaces it, since it's the QA pairs' informal phrasing that wins the
    # keyword/semantic match, not the services page's marketing copy.
    ("ct-services-overview",       "ct-process", "A business owner asks generically what they can do or what their realistic options are for a business debt problem, without naming a specific solution -- what concrete services can Corporate Turnaround offer?"),
    ("ct-mca-services-bridge",     "ct-process", "A business owner asks generically what they can do about multiple MCA debts or MCA cash-flow problems, without mentioning Corporate Turnaround by name -- what specific service does Corporate Turnaround offer for MCA relief?"),
    ("ct-creditor-harassment-bridge", "ct-process", "A business owner asks if anything can be done to stop aggressive creditor calls, legal threats, or bank account levies -- what specific service and protection does Corporate Turnaround offer?"),
]

# Topic ids added after the initial brainstorm run -- used by
# scripts/scrapers/append_new_qa_topics.py to generate only the new topics'
# questions and append them, rather than re-running the full list (which
# would regenerate every existing topic's questions at temperature 0.9 and
# break synthesize_qa_answers.py's already_answered resume matching).
NEWLY_ADDED_TOPIC_IDS = {
    "ct-services-overview",
    "ct-mca-services-bridge",
    "ct-creditor-harassment-bridge",
}

PROMPT_TEMPLATE = """
You are helping build a test set of REAL, messy questions that struggling small business
owners actually type into a chat box — not polished textbook questions.

Topic the questions should relate to: {topic_desc}

Here are examples of the exact style to match (real distressed/informal phrasing
already validated for this project):
{fewshots}

Generate {n} NEW questions in that same style, about the topic above. Vary the style:
- some with specific dollar amounts or numbers of debts (e.g. "3 mca loans", "$50k", "120 grand")
- some with typos, missing punctuation, run-on sentences
- some with urgency/panic ("what do i do right now", "is it too late")
- some short and blunt, some longer and rambling
Do not repeat the example questions verbatim. Do not answer the questions.

Respond ONLY with a valid JSON array of strings, each a single question. No Markdown,
no extra text.
"""


def load_informal_fewshots(eval_set_path: str = EVAL_SET_PATH) -> list[str]:
    """Pull style=='informal' queries straight from the eval set as few-shot
    anchors, so 'real-world phrasing' has one source of truth instead of
    drifting between this prompt and the eval set."""
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    return [item["query"] for item in eval_set if item.get("style") == "informal"]


def brainstorm_questions(llm, topic_desc: str, fewshots: list[str], n: int = 8) -> list[str]:
    fewshot_block = "\n".join(f'  - "{q}"' for q in fewshots)
    prompt = PROMPT_TEMPLATE.format(topic_desc=topic_desc, fewshots=fewshot_block, n=n)

    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        # Newer Gemini models return content as a list of typed blocks
        # (text blocks plus signed reasoning metadata), not a plain string.
        content = "".join(
            part["text"] if isinstance(part, dict) and "text" in part else str(part)
            for part in content
        )
    response_text = content.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    questions = json.loads(response_text)
    return [q for q in questions if isinstance(q, str) and q.strip()]


def run():
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable not set. Please set it in .env file.")
        return

    fewshots = load_informal_fewshots()
    logger.info(f"Loaded {len(fewshots)} informal few-shot anchors from {EVAL_SET_PATH}")

    # gemini-2.5-flash returns 404 "no longer available to new users" for the
    # current API key/project despite being listed by models.list -- an
    # account-level restriction, not a real deprecation. gemini-flash-latest
    # is the current equivalent.
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.9)

    all_questions = []
    for topic_id, category, topic_desc in QA_TOPICS:
        logger.info(f"Brainstorming questions for {topic_id}...")
        try:
            questions = brainstorm_questions(llm, topic_desc, fewshots, n=8)
            for q in questions:
                all_questions.append({
                    "topic_id": topic_id,
                    "category": category,
                    "topic_desc": topic_desc,
                    "question": q,
                    "style": "informal",
                })
            logger.info(f"  -> {len(questions)} questions")
        except Exception as e:
            logger.error(f"Error brainstorming questions for {topic_id}: {e}")

        time.sleep(1.5)  # rate limit

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(all_questions)} brainstormed questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
