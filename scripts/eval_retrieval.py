"""
scripts/eval_retrieval.py — Retrieval evaluation harness.

Runs a labeled query set through RetrievalPipeline and reports precision /
hit-rate / false-positive-rate, broken out by label bucket (in_domain,
borderline, out_of_domain). Use this to measure any pipeline change instead
of eyeballing single examples.

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --eval-set data/eval_retrieval_set.json
    python scripts/eval_retrieval.py --threshold-sweep -5 5 0.5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config import load_config

DEFAULT_EVAL_SET = os.path.join(os.path.dirname(__file__), "..", "data", "eval_retrieval_set.json")


def load_eval_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify(item: dict, chunks) -> tuple[str, bool]:
    """Returns (verdict, correct) where verdict is TP/FP/TN/FN."""
    got_hit = len(chunks) > 0
    expect_hit = item["expect_hit"]

    if expect_hit and got_hit:
        # Precision proxy: if expected substrings given, require at least
        # one returned chunk to actually contain one of them.
        keywords = item.get("expected_content_contains", [])
        if keywords:
            matched = any(
                any(kw.lower() in c.document.page_content.lower() for kw in keywords)
                for c in chunks
            )
            return ("TP", True) if matched else ("FP", False)
        return "TP", True
    if expect_hit and not got_hit:
        return "FN", False
    if not expect_hit and got_hit:
        return "FP", False
    return "TN", True  # not expect_hit and not got_hit


def _print_buckets(title: str, results: list[dict], key: str):
    buckets = {}
    for r in results:
        buckets.setdefault(r[key], []).append(r)
    buckets["OVERALL"] = results

    print(f"\n  {title}")
    for bucket, items in buckets.items():
        tp = sum(1 for i in items if i["verdict"] == "TP")
        fp = sum(1 for i in items if i["verdict"] == "FP")
        tn = sum(1 for i in items if i["verdict"] == "TN")
        fn = sum(1 for i in items if i["verdict"] == "FN")
        n = len(items)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        accuracy = sum(1 for i in items if i["correct"]) / n if n else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        print(f"  [{bucket:<13}] n={n:<3} TP={tp} FP={fp} TN={tn} FN={fn}  "
              f"precision={precision:.2f}  recall={recall:.2f}  fp_rate={fpr:.2f}  accuracy={accuracy:.2f}")


def print_summary(results: list[dict]):
    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    _print_buckets("By label", results, "label")
    _print_buckets("By style", results, "style")
    print(f"\n{'='*70}\n")


def save_results_json(results: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Per-query results written to {path}")


def run_eval(pipeline, eval_set: list[dict], verbose: bool = True) -> list[dict]:
    results = []
    for item in eval_set:
        chunks = pipeline.retrieve(item["query"])
        verdict, correct = classify(item, chunks)
        top_score = f"{chunks[0].rerank_score:.4f}" if chunks and chunks[0].rerank_score is not None else "N/A"
        results.append({
            "query": item["query"],
            "label": item["label"],
            "style": item.get("style", "formal"),
            "verdict": verdict,
            "correct": correct,
            "n_chunks": len(chunks),
            "top_score": top_score,
        })
        if verbose:
            print(f"  [{verdict}] {item['label']:<13} {item['query']!r:<55} -> {len(chunks)} chunks, top_score={top_score}")
    return results


def threshold_sweep(pipeline, eval_set: list[dict], lo: float, hi: float, step: float):
    """
    Re-runs the gate at each threshold WITHOUT rebuilding the pipeline or
    re-embedding/re-reranking per threshold: fetch each query's full traced
    (fused + reranked) candidate list once, then re-apply the gate in
    Python at each threshold value.
    """
    print(f"\n[Threshold Sweep] Tracing {len(eval_set)} queries once (RRF + rerank, no gate applied yet)...")
    traces = []
    for item in eval_set:
        trace = pipeline.retrieve_with_trace(item["query"], k=pipeline.config.rag_k)
        traces.append((item, trace))

    thresholds = []
    t = lo
    while t <= hi + 1e-9:
        thresholds.append(round(t, 4))
        t += step

    print(f"\n{'Threshold':<12}{'in_domain P':<14}{'in_domain R':<14}{'OOD FP-rate':<14}{'Overall Acc':<14}")
    for threshold in thresholds:
        results = []
        for item, trace in traces:
            k = pipeline.config.rag_k
            chunks = [c for c in trace.reranked if c.rerank_score is not None and c.rerank_score >= threshold][:k]
            verdict, correct = classify(item, chunks)
            results.append({"label": item["label"], "verdict": verdict, "correct": correct})

        def bucket_stats(label):
            items = [r for r in results if r["label"] == label]
            tp = sum(1 for i in items if i["verdict"] == "TP")
            fp = sum(1 for i in items if i["verdict"] == "FP")
            fn = sum(1 for i in items if i["verdict"] == "FN")
            tn = sum(1 for i in items if i["verdict"] == "TN")
            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
            fpr = fp / (fp + tn) if (fp + tn) else float("nan")
            return precision, recall, fpr

        in_domain = [r for r in results if r["label"] in ("in_domain", "borderline")]
        tp = sum(1 for i in in_domain if i["verdict"] == "TP")
        fp = sum(1 for i in in_domain if i["verdict"] == "FP")
        fn = sum(1 for i in in_domain if i["verdict"] == "FN")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")

        ood = [r for r in results if r["label"] == "out_of_domain"]
        ood_fp = sum(1 for i in ood if i["verdict"] == "FP")
        ood_fpr = ood_fp / len(ood) if ood else float("nan")

        overall_acc = sum(1 for r in results if r["correct"]) / len(results) if results else float("nan")

        print(f"{threshold:<12}{precision:<14.2f}{recall:<14.2f}{ood_fpr:<14.2f}{overall_acc:<14.2f}")

    print(f"\nPick the threshold that maximizes overall accuracy while keeping OOD FP-rate near 0.")
    print(f"Set config.rerank_score_threshold (or RERANK_SCORE_THRESHOLD env var) to that value.\n")


def _frange(lo: float, hi: float, step: float) -> list[float]:
    values = []
    t = lo
    while t <= hi + 1e-9:
        values.append(round(t, 4))
        t += step
    return values


def rescue_grid_sweep(
    pipeline, eval_set: list[dict],
    primary_lo: float, primary_hi: float, primary_step: float,
    rescue_lo: float, rescue_hi: float, rescue_step: float,
    rrf_cutoff: int,
):
    """
    2D grid sweep over (rerank_score_threshold, rerank_rescue_score_threshold)
    with rerank_rescue_rrf_rank_cutoff held fixed. Traces every query once
    (same caching pattern as threshold_sweep); RetrievedChunk.rrf_rank is
    already available on cached objects, so no pipeline changes are needed
    to re-apply the two-tier gate at each grid cell in pure Python.
    """
    print(f"\n[Rescue Grid Sweep] Tracing {len(eval_set)} queries once "
          f"(RRF + rerank, no gate applied yet). rrf_cutoff fixed at {rrf_cutoff}.")
    traces = []
    for item in eval_set:
        trace = pipeline.retrieve_with_trace(item["query"], k=pipeline.config.rag_k)
        traces.append((item, trace))

    primary_values = _frange(primary_lo, primary_hi, primary_step)
    rescue_values = _frange(rescue_lo, rescue_hi, rescue_step)
    k = pipeline.config.rag_k

    header = "primary\\rescue".ljust(16) + "".join(f"{r:<24}" for r in rescue_values)
    print(f"\n{header}")
    print("(cell = acc / informalR / oodFP)")

    for primary_t in primary_values:
        row = f"{primary_t:<16}"
        for rescue_t in rescue_values:
            results = []
            for item, trace in traces:
                chunks = [
                    c for c in trace.reranked
                    if c.rerank_score is not None and (
                        c.rerank_score >= primary_t
                        or (c.rerank_score >= rescue_t and c.rrf_rank <= rrf_cutoff)
                    )
                ][:k]
                verdict, correct = classify(item, chunks)
                results.append({
                    "label": item["label"],
                    "style": item.get("style", "formal"),
                    "verdict": verdict,
                    "correct": correct,
                })

            acc = sum(1 for r in results if r["correct"]) / len(results) if results else float("nan")

            informal_in_domain = [
                r for r in results
                if r["style"] == "informal" and r["label"] in ("in_domain", "borderline")
            ]
            informal_tp = sum(1 for r in informal_in_domain if r["verdict"] == "TP")
            informal_fn = sum(1 for r in informal_in_domain if r["verdict"] == "FN")
            informal_recall = (
                informal_tp / (informal_tp + informal_fn) if (informal_tp + informal_fn) else float("nan")
            )

            informal_ood = [r for r in results if r["style"] == "informal" and r["label"] == "out_of_domain"]
            informal_ood_fp = sum(1 for r in informal_ood if r["verdict"] == "FP")
            informal_ood_fpr = informal_ood_fp / len(informal_ood) if informal_ood else float("nan")

            row += f"{acc:.2f}/{informal_recall:.2f}/{informal_ood_fpr:.2f}".ljust(24)
        print(row)

    print(f"\nCell format: acc=overall accuracy, informalR=informal in-domain+borderline recall, "
          f"oodFP=informal out-of-domain false-positive rate.")
    print(f"Pick the cell with acc near 1.00 and oodFP near 0.00, preferring primary threshold close to "
          f"the existing calibrated value and the smallest-magnitude rescue threshold that still works.")
    print(f"Set config.rerank_score_threshold / rerank_rescue_score_threshold "
          f"(or RERANK_SCORE_THRESHOLD / RERANK_RESCUE_SCORE_THRESHOLD env vars), "
          f"and rerank_rescue_rrf_rank_cutoff={rrf_cutoff} (or RERANK_RESCUE_RRF_RANK_CUTOFF).\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG retrieval pipeline against a labeled query set.")
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET, help="Path to labeled eval set JSON.")
    parser.add_argument("--threshold-sweep", nargs=3, type=float, metavar=("MIN", "MAX", "STEP"),
                         help="Sweep rerank_score_threshold over MIN..MAX in STEP increments instead of scoring once.")
    parser.add_argument("--rescue-grid-sweep", nargs=6, type=float,
                         metavar=("PRIMARY_MIN", "PRIMARY_MAX", "PRIMARY_STEP", "RESCUE_MIN", "RESCUE_MAX", "RESCUE_STEP"),
                         help="2D grid sweep of (rerank_score_threshold, rerank_rescue_score_threshold) "
                              "with rerank_rescue_rrf_rank_cutoff held fixed at --rescue-rrf-cutoff.")
    parser.add_argument("--rescue-rrf-cutoff", type=int, default=3,
                         help="rrf_rank cutoff held fixed during --rescue-grid-sweep (default: 3).")
    parser.add_argument("--output-json", help="Write per-query results (as JSON) to this path, for before/after diffing.")
    args = parser.parse_args()

    config = load_config()
    eval_set = load_eval_set(args.eval_set)

    print(f"\n{'='*70}")
    print("  Retrieval Eval Harness")
    print(f"{'='*70}")
    print(f"  Eval set:      {args.eval_set} ({len(eval_set)} queries)")
    print(f"  Reranker:      {'enabled (' + config.rerank_model + ')' if config.rerank_enabled else 'disabled'}")
    print(f"  Gate threshold:{config.rerank_score_threshold}")
    print(f"{'='*70}\n")

    from rag.pipeline import RetrievalPipeline
    pipeline = RetrievalPipeline(config)

    if args.threshold_sweep:
        lo, hi, step = args.threshold_sweep
        threshold_sweep(pipeline, eval_set, lo, hi, step)
        return

    if args.rescue_grid_sweep:
        primary_lo, primary_hi, primary_step, rescue_lo, rescue_hi, rescue_step = args.rescue_grid_sweep
        rescue_grid_sweep(
            pipeline, eval_set,
            primary_lo, primary_hi, primary_step,
            rescue_lo, rescue_hi, rescue_step,
            rrf_cutoff=args.rescue_rrf_cutoff,
        )
        return

    results = run_eval(pipeline, eval_set)
    print_summary(results)

    if args.output_json:
        save_results_json(results, args.output_json)


if __name__ == "__main__":
    main()
