"""
scripts/verify_qa_arithmetic.py — Deterministic arithmetic verification for
QA-pair content.

Scoped to the one defect class actually observed in production: MCA
factor-rate calculations (principal x factor_rate = total_repayment;
total_repayment - principal = fee). The existing groundedness check in
synthesize_qa_answers.py verifies that claims are *thematically* supported
by the retrieved context, but a wrong product can still "look like" it's
using the right inputs -- this catches the arithmetic itself.

Can be run standalone as an audit over an existing qa_pairs.json (this
module), or imported by synthesize_qa_answers.py as a gate on newly
generated answers before they're accepted.

Usage:
    python scripts/verify_qa_arithmetic.py
    python scripts/verify_qa_arithmetic.py --qa-pairs data/raw/qa_pairs.json
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_QA_PAIRS_PATH = os.path.join(BASE_DIR, "data", "raw", "qa_pairs.json")
FLAGGED_PATH = os.path.join(BASE_DIR, "data", "raw", "qa_arithmetic_flagged.json")

TOLERANCE_ABS = 1.0     # dollars
TOLERANCE_REL = 0.005   # 0.5%, for rounding in stated figures

CALCULATION_GATE = re.compile(r"factor rate of\s*[\d.]+", re.IGNORECASE)

EXTRACTION_PROMPT = """
Extract every factor-rate calculation mentioned in ANSWER. A calculation has a
principal advance amount, a factor rate, and (optionally) a stated total
repayment amount and/or a stated fee amount.

ANSWER:
{answer}

Respond with ONLY a JSON array (no markdown fences). Each element:
  {{"principal": <number>, "factor_rate": <number>, "stated_total": <number or null>, "stated_fee": <number or null>}}
If no calculation is present, respond with an empty array: []
"""


def might_contain_calculation(text: str) -> bool:
    """Cheap regex gate so the extraction call only runs on answers that
    plausibly contain a calculation -- most QA answers don't."""
    return bool(CALCULATION_GATE.search(text)) and "$" in text


def extract_calculations(llm, answer_text: str) -> list[dict]:
    """LLM extraction only -- no generation, no self-critique. Extraction is
    a much more reliable task than open-ended math, and handles phrasing
    variance ('fee of $X', 'total repayment of $X', both) a single regex
    can't."""
    resp = llm.invoke(EXTRACTION_PROMPT.format(answer=answer_text))
    text = resp.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [c for c in parsed if isinstance(c, dict) and c.get("principal") and c.get("factor_rate")]


def verify(calculations: list[dict]) -> list[dict]:
    """Pure Python, no LLM. Returns the subset of calculations that don't add up."""
    mismatches = []
    for calc in calculations:
        principal = float(calc["principal"])
        factor_rate = float(calc["factor_rate"])
        expected_total = principal * factor_rate
        expected_fee = expected_total - principal

        def close(stated, expected):
            return abs(stated - expected) <= max(TOLERANCE_ABS, TOLERANCE_REL * expected)

        problems = []
        stated_total = calc.get("stated_total")
        if stated_total is not None and not close(float(stated_total), expected_total):
            problems.append(f"stated_total={stated_total} but principal*factor_rate={expected_total:.2f}")

        stated_fee = calc.get("stated_fee")
        if stated_fee is not None and not close(float(stated_fee), expected_fee):
            problems.append(f"stated_fee={stated_fee} but total-principal={expected_fee:.2f}")

        if problems:
            mismatches.append({
                **calc,
                "expected_total": round(expected_total, 2),
                "expected_fee": round(expected_fee, 2),
                "problems": problems,
            })
    return mismatches


def check_answer(llm, answer_text: str) -> list[dict]:
    """Full pipeline for one answer: gate -> extract -> verify. Returns a
    list of mismatch dicts (empty if the answer has no arithmetic problems,
    including the common case of no calculation at all)."""
    if not might_contain_calculation(answer_text):
        return []
    calculations = extract_calculations(llm, answer_text)
    if not calculations:
        return []
    return verify(calculations)


def run(qa_pairs_path: str = DEFAULT_QA_PAIRS_PATH):
    with open(qa_pairs_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    from core.llms import GroqProvider
    llm = GroqProvider(model_name="llama-3.1-8b-instant", temperature=0.0).get_llm()

    flagged = []
    checked = 0
    for chunk in chunks:
        text = chunk.get("chunk_text", "")
        if not might_contain_calculation(text):
            continue
        checked += 1
        mismatches = check_answer(llm, text)
        if mismatches:
            flagged.append({**chunk, "mismatches": mismatches})
            print(f"[FLAGGED] {chunk['id']}")
            for m in mismatches:
                print(f"    {m['problems']}")

    print(f"\nChecked {checked} chunk(s) with a candidate calculation, out of {len(chunks)} total.")
    print(f"Flagged: {len(flagged)}")

    if flagged:
        with open(FLAGGED_PATH, "w", encoding="utf-8") as f:
            json.dump(flagged, f, ensure_ascii=False, indent=2)
        print(f"Saved flagged entries to {FLAGGED_PATH}")

    return flagged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify arithmetic in QA-pair content.")
    parser.add_argument("--qa-pairs", default=DEFAULT_QA_PAIRS_PATH)
    args = parser.parse_args()
    run(args.qa_pairs)
