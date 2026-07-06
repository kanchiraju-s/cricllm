"""Rough token counting, since Gemini won't tell us its real tokenizer.

We fall back to tiktoken's cl100k_base encoding here. It's not exactly what
Gemini uses under the hood, but it's fast, has no weird dependencies, and is
consistent — which is really all we need for deciding when a chunk is
getting too big.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Ballpark token count for ``text`` — good enough for sizing chunks, not exact."""
    if not text:
        return 0
    return len(_encoding().encode(text))
