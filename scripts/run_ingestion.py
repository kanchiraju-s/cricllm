#!/usr/bin/env python3
"""Lets you run ingestion without `pip install -e .`-ing the package first.

Usage:
    python scripts/run_ingestion.py --input data/icc_rulebook.md
    python scripts/run_ingestion.py --input data/ --force
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cricllm.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
