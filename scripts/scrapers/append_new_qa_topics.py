"""
scripts/scrapers/append_new_qa_topics.py — Brainstorm questions for only the
newly-added QA_TOPICS entries (NEWLY_ADDED_TOPIC_IDS) and append them to the
existing data/raw/qa_questions_brainstormed.json, rather than re-running
brainstorm_qa_questions.py's full run() -- which would regenerate every
existing topic's questions at temperature 0.9 and break
synthesize_qa_answers.py's already_answered resume matching (it matches on
literal question text).

Usage:
    python -m scripts.scrapers.append_new_qa_topics
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
from scripts.scrapers.brainstorm_qa_questions import (
    QA_TOPICS,
    NEWLY_ADDED_TOPIC_IDS,
    OUTPUT_PATH,
    brainstorm_questions,
    load_informal_fewshots,
)


def run():
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable not set. Please set it in .env file.")
        return

    existing = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing_topic_ids = {item["topic_id"] for item in existing}

    new_topics = [t for t in QA_TOPICS if t[0] in NEWLY_ADDED_TOPIC_IDS and t[0] not in existing_topic_ids]
    if not new_topics:
        logger.info("No new topics to brainstorm -- all NEWLY_ADDED_TOPIC_IDS already present.")
        return

    fewshots = load_informal_fewshots()
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.9)

    added = []
    for topic_id, category, topic_desc in new_topics:
        logger.info(f"Brainstorming questions for {topic_id}...")
        try:
            questions = brainstorm_questions(llm, topic_desc, fewshots, n=8)
            for q in questions:
                added.append({
                    "topic_id": topic_id,
                    "category": category,
                    "topic_desc": topic_desc,
                    "question": q,
                    "style": "informal",
                })
            logger.info(f"  -> {len(questions)} questions")
        except Exception as e:
            logger.error(f"Error brainstorming questions for {topic_id}: {e}")
        time.sleep(1.5)

    merged = existing + added
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    logger.info(f"Appended {len(added)} new question(s). Total now {len(merged)} in {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
