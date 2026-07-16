"""
scripts/scrapers/synthesize_qa_answers.py — Grounded QA answer synthesis.

For each brainstormed question (scripts/scrapers/brainstorm_qa_questions.py),
retrieves real chunks via the SAME RetrievalPipeline the live agent uses, and
has an LLM answer strictly from those chunks. This is the anti-hallucination
step: the LLM never answers from parametric knowledge alone. Questions with
no retrievable grounding are hard-refused (logged separately for follow-up),
never silently answered from thin air.

Usage:
    python scripts/scrapers/synthesize_qa_answers.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from config import load_config
from core.llms import GroqProvider
from scripts.scrapers.utils import make_chunk, save_raw, logger
from scripts.verify_qa_arithmetic import check_answer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUESTIONS_PATH = os.path.join(BASE_DIR, "data", "raw", "qa_questions_brainstormed.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "raw", "qa_pairs.json")
UNANSWERABLE_PATH = os.path.join(BASE_DIR, "data", "raw", "qa_unanswerable.json")

ANSWER_PROMPT = """
You are a senior debt relief advisor at Corporate Turnaround, a firm that helps small
business owners resolve business debt without bankruptcy.

A business owner asked:
"{question}"

Answer using ONLY facts in the CONTEXT below. Do not invent numbers, dates, percentages,
interest/factor rates, legal claims, or program specifics that are not present in CONTEXT.
If CONTEXT only partially answers the question, answer the part it supports and say the
rest should be discussed by calling 1-800-889-0232. Keep the tone empathetic and direct,
2-4 sentences.

CONTEXT:
{context}

Respond with ONLY the answer text. No preamble, no "Based on the context" framing.
"""

GROUNDEDNESS_CHECK_PROMPT = """
CONTEXT:
{context}

ANSWER:
{answer}

Does every DOMAIN factual claim in ANSWER (numbers, rates, dates, legal claims, program
specifics) appear in CONTEXT? Ignore mentions of the phone number 1-800-889-0232 or generic
advice to "call us" / "discuss with an advisor" — that redirect is standard house style, not
a claim that needs sourcing. Respond with ONLY "YES" or "NO: <brief reason>".
"""


def synthesize_answer(llm, pipeline, question: str) -> tuple[str | None, list, str | None]:
    """Returns (answer, retrieved_chunks, refusal_reason). answer is None if
    retrieval was empty (hard refuse) or the groundedness check failed."""
    chunks = pipeline.retrieve(question)
    if not chunks:
        return None, [], "no_retrieval"

    context = pipeline.format_for_llm(chunks)
    answer_resp = llm.invoke(ANSWER_PROMPT.format(question=question, context=context))
    answer = answer_resp.content.strip()

    check_resp = llm.invoke(GROUNDEDNESS_CHECK_PROMPT.format(context=context, answer=answer))
    verdict = check_resp.content.strip()
    if not verdict.upper().startswith("YES"):
        return None, chunks, f"groundedness_check_failed: {verdict}"

    # Groundedness only checks that claims are thematically supported by
    # CONTEXT -- a wrong factor-rate product can still "look like" it's
    # using the right inputs. Verify the arithmetic itself deterministically.
    mismatches = check_answer(llm, answer)
    if mismatches:
        return None, chunks, f"arithmetic_check_failed: {mismatches}"

    return answer, chunks, None


def run():
    if not os.path.exists(QUESTIONS_PATH):
        logger.error(f"No brainstormed questions found at {QUESTIONS_PATH}. "
                     f"Run scripts/scrapers/brainstorm_qa_questions.py first.")
        return

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Resume support: skip questions already successfully answered in a prior
    # run (matched by the literal question text embedded in section_heading
    # as "Q: {question}") so re-runs after a quota reset don't re-spend
    # tokens re-answering questions that already succeeded.
    already_answered = set()
    all_chunks = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        already_answered = {
            c["section_heading"][len("Q: "):] for c in all_chunks
            if c.get("section_heading", "").startswith("Q: ")
        }
        logger.info(f"Resuming: {len(already_answered)} question(s) already answered in a prior run.")

    config = load_config()
    from rag.pipeline import RetrievalPipeline
    pipeline = RetrievalPipeline(config)
    llm = GroqProvider(model_name="llama-3.1-8b-instant", temperature=0.2).get_llm()

    unanswerable = []
    counters: dict[str, int] = {}
    for c in all_chunks:
        # Keep id counters consistent with already-saved chunks so a resumed
        # run doesn't reuse an id already assigned to a prior success.
        topic = c.get("topic", "")
        counters[topic] = max(counters.get(topic, 0), int(c["id"].rsplit("-", 1)[-1]) + 1)

    for item in questions:
        topic_id = item["topic_id"]
        question = item["question"]
        if question in already_answered:
            continue
        logger.info(f"Synthesizing answer for [{topic_id}] {question!r}")

        try:
            answer, retrieved, refusal_reason = synthesize_answer(llm, pipeline, question)
        except Exception as e:
            logger.error(f"  Error: {e}")
            unanswerable.append({**item, "reason": f"error: {e}"})
            time.sleep(1.0)
            continue

        if answer is None:
            logger.warning(f"  Refused ({refusal_reason})")
            unanswerable.append({**item, "reason": refusal_reason})
            time.sleep(1.0)
            continue

        idx = counters.get(topic_id, 0)
        counters[topic_id] = idx + 1
        chunk = make_chunk(
            id=f"qa-{topic_id}-{idx}",
            source_type="qa_pair",
            topic=topic_id,
            category=item["category"],
            tags=[item["category"], "qa", "distressed-phrasing"],
            title=f"Q&A: {item['topic_desc']}",
            heading=f"Q: {question}",
            text=f"A: {answer}",
            requires_disclaimer=item["category"] in ("ct-process", "mca-education"),
        )
        chunk["grounded_source_ids"] = [
            c.document.metadata.get("topic", "?") for c in retrieved
        ]
        all_chunks.append(chunk)
        logger.info(f"  -> grounded in {len(retrieved)} chunk(s)")

        time.sleep(1.0)  # rate limit

    save_raw(all_chunks, OUTPUT_PATH)

    if unanswerable:
        with open(UNANSWERABLE_PATH, "w", encoding="utf-8") as f:
            json.dump(unanswerable, f, ensure_ascii=False, indent=2)
        logger.warning(f"Saved {len(unanswerable)} unanswerable/refused questions to {UNANSWERABLE_PATH}")

    logger.info(f"Done: {len(all_chunks)} grounded QA pairs, {len(unanswerable)} refused.")


if __name__ == "__main__":
    run()
