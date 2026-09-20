import argparse
import sys
from pathlib import Path

from llm import GeminiClient
from pipeline import RAGPipeline
from retriever import Retriever, build_and_save_index

INDEX_DIR = Path(__file__).parent.parent / "index"


def load_or_build_retriever():
    if not (INDEX_DIR / "chunks.json").exists():
        return build_and_save_index()
    return Retriever.load()


def main():
    parser = argparse.ArgumentParser(description="LearnForge support assistant (CLI)")
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    retriever = load_or_build_retriever()

    try:
        client = GeminiClient(model_name=args.model)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    pipeline = RAGPipeline(retriever, client, top_k=args.top_k)

    print("LearnForge Support Assistant. Type 'exit' to quit.\n")

    history = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        ans = pipeline.answer(question, history)
        print(f"\nAssistant: {ans.answer_text}")
        print(f"[confidence: {ans.confidence} | sources: {', '.join(ans.sources) or 'none'}]")
        if ans.escalate:
            print(f"⚠️  Escalate to human agent — {ans.escalation_reason}")
        print()
        history.append({"user": question, "assistant": ans.answer_text})


if __name__ == "__main__":
    main()
