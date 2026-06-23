"""
eval/run_eval.py — Evaluation harness for the Insurance RAG pipeline.

Usage:
    python eval/run_eval.py --policy docs/BAJHLIP23020V012223.pdf

Metrics reported:
  - Decision accuracy    : exact match on covered/not_covered/partial/unclear
  - Clause recall        : did expected keywords appear in the answer + justification?
  - Mean confidence      : correct vs incorrect answers
  - Query type breakdown : simple vs complex accuracy
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"


def load_ground_truth():
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        return json.load(f)


def keyword_recall(answer_text: str, justification: str, keywords: list) -> bool:
    """Return True if ANY expected keyword appears in the answer or justification."""
    haystack = (answer_text + " " + justification).lower()
    return any(kw.lower() in haystack for kw in keywords)


def run_pipeline(pdf_path: str, questions: list) -> list:
    """
    Index the PDF exactly once, then run each question sequentially
    through the query processor directly — no re-indexing per question.
    """
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()

    async def _run():
        import os
        from src.parse_documents import load_and_parse_documents
        from src.chunk_documents_optimized import chunk_documents_optimized
        from src.embed_and_index import index_chunks_in_pinecone
        from src.query_processor import QueryProcessor

        pinecone_key = os.getenv("PINECONE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")

        # Step 1: parse + chunk
        print("  📄 Parsing PDF...")
        parsed = load_and_parse_documents([pdf_path])
        transformed = [{
            "document_name": doc.get("document_name", ""),
            "content": doc.get("parsed_output", {}).get("content", ""),
            "ordered_content": doc.get("parsed_output", {}).get("ordered_content", []),
        } for doc in parsed]
        chunks = chunk_documents_optimized(transformed)
        print(f"  ✅ {len(chunks)} chunks")

        # Step 2: index once
        print("  📥 Indexing to Pinecone...")
        result = index_chunks_in_pinecone(
            chunks=chunks,
            pinecone_api_key=pinecone_key,
            pinecone_env="us-east-1",
            index_name="policy-index",
        )
        if result is False:
            raise RuntimeError("Pinecone indexing failed")
        print("  ✅ Indexed")

        # Step 3: build processor once, populate chunk cache
        processor = QueryProcessor(
            pinecone_api_key=pinecone_key,
            groq_api_key=groq_key or "",
            gemini_api_key=gemini_key or "",
            index_name="policy-index",
        )
        processor.populate_chunk_cache(chunks)

        # Step 4: query sequentially with a gap to avoid rate limits
        results = []
        for i, question in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {question[:70]}")
            result = processor.process_query_routed_sync(question)
            evaluation = result.get("evaluation", {})
            results.append({
                "decision":        evaluation.get("decision", "unclear"),
                "confidence":      evaluation.get("confidence", 0.0),
                "answer":          evaluation.get("answer", ""),
                "justification":   evaluation.get("justification", ""),
                "relevant_clauses": evaluation.get("relevant_clauses", []),
            })
            if i < len(questions) - 1:
                time.sleep(3)  # 3s gap — keeps well under Groq's RPM limit

        return results

    return asyncio.run(_run())


def evaluate(results: list, ground_truth: list) -> dict:
    """Compare pipeline results against ground truth and compute metrics."""
    assert len(results) == len(ground_truth), "Result count mismatch"

    correct_decision = 0
    clause_recall_hits = 0
    confidences_correct = []
    confidences_incorrect = []

    simple_correct = simple_total = 0
    complex_correct = complex_total = 0

    rows = []
    for gt, result in zip(ground_truth, results):
        predicted_decision = result.get("decision", "unclear")
        confidence = float(result.get("confidence", 0.0))
        answer_text = result.get("answer", "")
        justification = result.get("justification", "")

        decision_match = predicted_decision == gt["expected_decision"]
        recall_hit = keyword_recall(answer_text, justification, gt["expected_answer_keywords"])

        if decision_match:
            correct_decision += 1
            confidences_correct.append(confidence)
        else:
            confidences_incorrect.append(confidence)

        if recall_hit:
            clause_recall_hits += 1

        qtype = gt["query_type"]
        if qtype == "simple":
            simple_total += 1
            if decision_match:
                simple_correct += 1
        else:
            complex_total += 1
            if decision_match:
                complex_correct += 1

        rows.append({
            "id": gt["id"],
            "question": gt["question"][:60],
            "expected": gt["expected_decision"],
            "predicted": predicted_decision,
            "decision_ok": "✅" if decision_match else "❌",
            "recall_ok": "✅" if recall_hit else "❌",
            "confidence": round(confidence, 2),
            "query_type": qtype,
        })

    n = len(ground_truth)
    return {
        "rows": rows,
        "decision_accuracy": correct_decision / n,
        "clause_recall": clause_recall_hits / n,
        "mean_confidence_correct": sum(confidences_correct) / len(confidences_correct) if confidences_correct else 0.0,
        "mean_confidence_incorrect": sum(confidences_incorrect) / len(confidences_incorrect) if confidences_incorrect else 0.0,
        "simple_accuracy": simple_correct / simple_total if simple_total else 0.0,
        "complex_accuracy": complex_correct / complex_total if complex_total else 0.0,
        "total": n,
        "correct": correct_decision,
        "recall_hits": clause_recall_hits,
    }


def print_report(metrics: dict, elapsed: float):
    rows = metrics["rows"]
    n = metrics["total"]

    print("\n" + "=" * 72)
    print("  EVALUATION RESULTS — Bajaj Allianz Global Health Care")
    print("=" * 72)
    print(f"  {'ID':<5} {'Dec':^3} {'Rcl':^3}  {'Conf':^5}  {'Type':<8}  {'Expected':<12}  {'Predicted':<12}")
    print(f"  {'-'*5} {'-'*3} {'-'*3}  {'-'*5}  {'-'*8}  {'-'*12}  {'-'*12}")
    for r in rows:
        print(
            f"  {r['id']:<5} {r['decision_ok']:^3} {r['recall_ok']:^3}  "
            f"{r['confidence']:>5.2f}  {r['query_type']:<8}  "
            f"{r['expected']:<12}  {r['predicted']:<12}"
        )

    print("=" * 72)
    print(f"  Decision accuracy   : {metrics['correct']}/{n}  ({metrics['decision_accuracy']:.0%})")
    print(f"  Clause recall       : {metrics['recall_hits']}/{n}  ({metrics['clause_recall']:.0%})")
    print(f"  Mean conf (correct) : {metrics['mean_confidence_correct']:.2f}")
    print(f"  Mean conf (wrong)   : {metrics['mean_confidence_incorrect']:.2f}")
    print(f"  Simple accuracy     : {metrics['simple_accuracy']:.0%}")
    print(f"  Complex accuracy    : {metrics['complex_accuracy']:.0%}")
    print(f"  Total time          : {elapsed:.1f}s")
    print("=" * 72 + "\n")


def save_results(metrics: dict, pdf_path: str, elapsed: float):
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{ts}.json"
    payload = {
        "policy": os.path.basename(pdf_path),
        "timestamp": ts,
        "elapsed_seconds": round(elapsed, 2),
        "summary": {k: v for k, v in metrics.items() if k != "rows"},
        "rows": metrics["rows"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Results saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline evaluation")
    parser.add_argument("--policy", required=True, help="Path to the policy PDF")
    parser.add_argument("--save", action="store_true", default=True, help="Save results to eval/results/")
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.policy)
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        sys.exit(1)

    ground_truth = load_ground_truth()
    questions = [gt["question"] for gt in ground_truth]

    print(f"\n📄 Policy : {os.path.basename(pdf_path)}")
    print(f"❓ Questions : {len(questions)}")
    print("⏳ Running pipeline...\n")

    t0 = time.time()
    results = run_pipeline(pdf_path, questions)
    elapsed = time.time() - t0

    if len(results) != len(ground_truth):
        print(f"❌ Got {len(results)} results for {len(ground_truth)} questions — pipeline error")
        sys.exit(1)

    metrics = evaluate(results, ground_truth)
    print_report(metrics, elapsed)

    if args.save:
        save_results(metrics, pdf_path, elapsed)


if __name__ == "__main__":
    main()
