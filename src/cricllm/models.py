"""Just the result objects the pipeline hands back after ingesting a file."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IngestionStats:
    """A quick summary of what happened when we ingested one file."""

    source: str
    total_chunks: int = 0
    ingested_chunks: int = 0
    duplicate_chunks: int = 0
    skipped_already_ingested: int = 0
    skipped_unchanged: bool = False
    duration_seconds: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        """No error means it's fine — whether it actually ingested or was skipped."""
        return self.error is None
