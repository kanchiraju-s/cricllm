"""Loads a Markdown file without letting one bad file kill the whole ingestion run.

Handles the stuff that trips up a naive `open().read()`: invalid UTF-8 (we
fall back to charset-normalizer's best guess), empty files, and a quick
sanity check for unbalanced code fences.
"""

from __future__ import annotations

from pathlib import Path

from charset_normalizer import from_bytes

from cricllm.exceptions import CorruptedMarkdownError, EmptyDocumentError, InvalidEncodingError
from cricllm.logging_config import get_logger

logger = get_logger("loader")


def load_markdown_file(path: Path) -> str:
    """Load a Markdown file as text, without choking on encoding weirdness.

    Raises:
        EmptyDocumentError: file's missing, zero-byte, or just whitespace.
        InvalidEncodingError: we genuinely couldn't figure out the encoding.
    """
    if not path.exists():
        raise EmptyDocumentError(f"File does not exist: {path}")

    raw_bytes = path.read_bytes()
    if not raw_bytes:
        raise EmptyDocumentError(f"File is empty: {path}")

    text = _decode_bytes(raw_bytes, path)

    if not text.strip():
        raise EmptyDocumentError(f"File has no usable content after decoding: {path}")

    if "\x00" in text:
        raise CorruptedMarkdownError(f"Null bytes found in decoded text — binary/corrupt file: {path}")

    _check_fence_balance(text, path)
    return text


def _decode_bytes(raw_bytes: bytes, path: Path) -> str:
    """Try strict UTF-8 first; if that fails, let charset-normalizer take a guess."""
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("Invalid UTF-8 in %s, attempting encoding detection", path)

    result = from_bytes(raw_bytes).best()
    if result is None:
        raise InvalidEncodingError(f"Could not detect a usable encoding for {path}")

    logger.warning(
        "Recovered %s using detected encoding=%s (confidence best-effort)", path, result.encoding
    )
    return str(result)


def _check_fence_balance(text: str, path: Path) -> None:
    """Warn if code fences look unbalanced — doesn't fail, just flags it.

    Hand-edited docs end up with a stray ``` fence pretty often. It's not
    fatal (the chunker just treats an unterminated fence as running to the
    end of the file), but it's worth a heads-up that the file might need a
    once-over.
    """
    fence_count = sum(1 for line in text.splitlines() if line.strip().startswith("```"))
    if fence_count % 2 != 0:
        logger.warning(
            "Odd number of ``` fences (%d) in %s — file may have corrupted code blocks; "
            "the trailing fence will be treated as running to end of file",
            fence_count,
            path,
        )
