"""
Runs the eval question set against the real pipeline and reports:
  - retrieval hit@k (expected chunk id shows up in top-k)
  - escalation match (did it escalate when it should / shouldn't)

Usage:
    export GEMINI_API_KEY=your-key
    python run_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm import GeminiClient
from pipeline import RAGPipeline
from retriever import Retriever, build_and_save_index

EVAL_FILE = Path(__file__).parent / "eval_questions.json"


def run():
    index_dir = Path(__file__).parent.parent / "index"
    retriever = Retriever.load(index_dir) if (index_dir / "chunks.json").exists() else build_and_save_index()
    pipeline = RAGPipeline(retriever, GeminiClient())

    cases = json.loads(EVAL_FILE.read_text())
    hits, hit_total = 0, 0
    escal_correct, escal_total = 0, 0

    for case in cases:
        ans = pipeline.answer(case["question"], case.get("history", []))

        expected_ids = set(case.get("expected_chunk_ids", []))
        hit_str = "n/a"
        if expected_ids:
            hit_total += 1
            hit = bool(expected_ids & set(ans.retrieved_ids))
            hits += int(hit)
            hit_str = "HIT" if hit else "MISS"

        escal_str = "n/a"
        expect_escalate = case.get("expect_escalate")
        if expect_escalate is not None:
            escal_total += 1
            correct = ans.escalate == expect_escalate
            escal_correct += int(correct)
            escal_str = "OK" if correct else f"MISS (expected {expect_escalate}, got {ans.escalate})"

        print(f"{case['id']:<10} retrieval={hit_str:<6} escalation={escal_str}")

    print("\n--- Summary ---")
    if hit_total:
        print(f"Retrieval hit rate: {hits}/{hit_total}")
    if escal_total:
        print(f"Escalation accuracy: {escal_correct}/{escal_total}")


if __name__ == "__main__":
    run()
