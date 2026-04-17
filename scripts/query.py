"""Interactive query script — ask questions against the knowledge base."""

import argparse
import logging
import sys

sys.path.insert(0, ".")

from src.rag_chain import invoke

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _print_result(result: dict) -> None:
    """Pretty-print an answer and its sources."""
    print(f"\n{result['answer']}\n")
    print("Sources:")
    for source in result["sources"]:
        print(f"  - {source}")
    print()


def main() -> None:
    """Run a single query or start an interactive REPL."""
    parser = argparse.ArgumentParser(description="Query the knowledge base")
    parser.add_argument("--question", help="Question to ask (omit for REPL mode)")
    args = parser.parse_args()

    if args.question:
        result = invoke(args.question)
        _print_result(result)
        return

    print("RAG Knowledge Assistant (type 'quit' to exit)")
    print("-" * 48)
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("quit", "exit"):
            break
        try:
            result = invoke(question)
            _print_result(result)
        except Exception:
            logger.exception("Error processing question")


if __name__ == "__main__":
    main()
