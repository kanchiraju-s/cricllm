"""Small SHA256 helpers — everything else in this project uses these for caching,
spotting duplicates, and figuring out if a file has actually changed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_of_text(text: str) -> str:
    """Hash a string (UTF-8) and return the hex digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_file(path: Path) -> str:
    """Hash a file's raw bytes without loading the whole thing into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()
