"""All our own exception types, so callers can tell our failures apart from library ones."""

from __future__ import annotations


class CricLLMError(Exception):
    """Catch-all base — catch this if you just want "something in our code broke"."""


class EmptyDocumentError(CricLLMError):
    """The source file had nothing usable in it (missing, blank, whitespace-only)."""


class InvalidEncodingError(CricLLMError):
    """Couldn't decode the file as text no matter what encoding we tried."""


class CorruptedMarkdownError(CricLLMError):
    """The Markdown is broken badly enough that we can't safely parse it."""


class EmbeddingAPIError(CricLLMError):
    """Gemini's embedding API kept failing even after we retried."""


class VectorStoreError(CricLLMError):
    """Chroma (or whatever's backing the vector store) failed in a way we can't recover from."""
