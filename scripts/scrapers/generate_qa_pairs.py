import os
import json
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from scripts.scrapers.utils import make_chunk, save_raw, logger

load_dotenv()

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

    # Business debt pillar
    ("biz-debt-types",    "business-debt",    "What types of debt do small businesses typically have?"),
    ("biz-credit-score",  "business-debt",    "How does business credit work vs. personal credit?"),
    ("biz-personal-guar", "business-debt",    "What does signing a personal guarantee mean for a business owner?"),
    ("biz-sba-default",   "business-debt",    "What happens if you default on an SBA loan?"),
    ("biz-ch7-vs-ch11",   "business-debt",    "What is the difference between Chapter 7 and Chapter 11 bankruptcy?"),
    ("biz-out-of-court",  "business-debt",    "What is out-of-court debt restructuring?"),
    ("biz-vendor-debt",   "business-debt",    "What happens when you can't pay your vendors?"),
    ("biz-ucc-search",    "business-debt",    "How do you find UCC liens filed against your business?"),

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
]

PROMPT_TEMPLATE = """
You are a senior debt relief advisor at Corporate Turnaround, a firm that helps small
business owners resolve business debt without bankruptcy.

Generate 5 realistic questions a struggling small business owner might ask about the
following topic, along with clear, accurate, empathetic answers (2-4 sentences each).

Topic: {topic}

Respond ONLY with a valid JSON array. Each element must have:
  - "question": the owner's question (natural language)
  - "answer": the advisor's response

Do not include any text outside the JSON array. Do not use Markdown block syntax around the JSON. Return raw JSON text.
"""

def run():
    all_chunks = []
    
    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable not set. Please set it in .env file.")
        return
        
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API: {e}")
        return
        
    for topic_id, category, topic_desc in QA_TOPICS:
        logger.info(f"Generating Q&A for {topic_id}...")
        prompt = PROMPT_TEMPLATE.format(topic=topic_desc)
        
        try:
            response = llm.invoke(prompt)
            # Try to parse JSON from the response, removing any markdown code blocks if present
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
                
            qa_pairs = json.loads(response_text)
            
            for idx, pair in enumerate(qa_pairs):
                question = pair.get("question", "")
                answer = pair.get("answer", "")
                
                if question and answer:
                    chunk_id = f"qa-{topic_id}-{idx}"
                    requires_disclaimer = category in ["ct-process", "mca-education"]
                    
                    chunk = make_chunk(
                        id=chunk_id,
                        source_type="qa_pair",
                        topic=topic_id,
                        category=category,
                        tags=[category, "qa"],
                        title=f"Q&A: {topic_desc}",
                        heading=f"Q: {question}",
                        text=f"A: {answer}",
                        requires_disclaimer=requires_disclaimer
                    )
                    all_chunks.append(chunk)
                    
        except Exception as e:
            logger.error(f"Error generating Q&A for {topic_id}: {e}")
            
        # Rate limit
        time.sleep(1.5)
        
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    save_path = os.path.join(base_dir, "data", "raw", "qa_pairs.json")
    save_raw(all_chunks, save_path)

if __name__ == "__main__":
    run()
