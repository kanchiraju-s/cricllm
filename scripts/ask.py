#!/usr/bin/env python3
"""CLI for asking the rulebook a question.

Usage:
    python scripts/ask.py "How many no-balls make an over invalid?"
    python scripts/ask.py "What happens if a fielder deliberately deflects the ball with their helmet?" --top-k 8

This is the "query" half — run_ingestion.py builds the index, this asks
questions against it. All the actual work (embedding the question,
retrieving chunks, calling Gemini) happens in cricllm.qa.QAEngine; this
file and app.py are both just thin wrappers around it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cricllm.config import load_settings  # noqa: E402
from cricllm.qa import QAEngine  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a question against the ICC Laws of Cricket")
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = load_settings()
    engine = QAEngine(settings)

    if not engine.is_ready():
        print("The vector store is empty — run ingestion first:")
        print("  python scripts/run_ingestion.py --input data/icc_rulebook.md")
        return 1

    result = engine.answer(args.question, top_k=args.top_k)

    print(result.answer)
    if not result.sources:
        return 1

    print("\n--- Sources ---")
    for source in result.sources:
        print(f"  ({source.distance:.3f}) {source.header_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
