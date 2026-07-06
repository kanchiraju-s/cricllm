"""Command-line entry point: ``cricllm-ingest --input data/icc_rulebook.md``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cricllm.config import load_settings
from cricllm.logging_config import setup_logging
from cricllm.pipeline import IngestionPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    """Set up the argparse parser for `cricllm-ingest`."""
    parser = argparse.ArgumentParser(
        description=(
            "Ingest the ICC Laws of Cricket Markdown rulebook into a Gemini-embedded, "
            "locally persisted Chroma vector index."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a Markdown file (e.g. data/icc_rulebook.md) or a directory of .md files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest files even if their content hash is unchanged since the last run",
    )
    return parser


def _iter_markdown_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.rglob("*.md"))
    return [input_path]


def main(argv: list[str] | None = None) -> int:
    """Run ingestion from the command line. 0 if everything went fine, 1 if anything failed."""
    args = build_arg_parser().parse_args(argv)
    settings = load_settings()
    logger = setup_logging(settings.log_dir)

    files = _iter_markdown_files(args.input)
    if not files:
        logger.error("No Markdown files found at %s", args.input)
        return 1

    pipeline = IngestionPipeline(settings)
    exit_code = 0
    for path in files:
        stats = pipeline.ingest_file(path, force=args.force)
        if not stats.success:
            exit_code = 1

    logger.info("Vector store now holds %d chunks total", pipeline.vector_store.count())
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
