"""A small SQLite database that remembers what we've already done.

Two things live in here, both keyed off SHA256 hashes:

* ``EmbeddingCache`` — so we never pay to re-embed the same text twice.
* ``FileStateStore`` — so re-running ingestion on an unchanged file is a
  no-op instead of starting over.

(There's also ``IngestedChunkStore`` further down, same idea but for
tracking which chunks actually made it into the vector store.)
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from cricllm.logging_config import get_logger

logger = get_logger("cache")

_SCHEMA = """
-- v2 because we added task_type to the key. Turns out the same text embeds
-- differently depending on RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY, so we
-- can't just key on (content_hash, model) anymore. Went with a new table
-- name instead of an ALTER TABLE migration -- anyone with an old cache file
-- just re-embeds once and moves on, no risk of a broken schema.
CREATE TABLE IF NOT EXISTS embeddings_v2 (
    content_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    task_type TEXT NOT NULL,
    vector TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (content_hash, model, task_type)
);

CREATE TABLE IF NOT EXISTS file_state (
    file_path TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS ingested_chunks (
    chunk_hash TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.executescript(_SCHEMA)
    return conn


class EmbeddingCache:
    """Sticks embedding vectors in SQLite, keyed by hash + model + task_type."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def get_many(self, hashes: list[str], model: str, task_type: str) -> dict[str, list[float]]:
        """Look up whichever of these hashes we already have cached."""
        if not hashes:
            return {}
        with closing(_connect(self._db_path)) as conn:
            placeholders = ",".join("?" for _ in hashes)
            rows = conn.execute(
                f"SELECT content_hash, vector FROM embeddings_v2 "
                f"WHERE model = ? AND task_type = ? AND content_hash IN ({placeholders})",
                (model, task_type, *hashes),
            ).fetchall()
        return {content_hash: json.loads(vector) for content_hash, vector in rows}

    def set_many(self, items: dict[str, list[float]], model: str, task_type: str) -> None:
        """Save freshly computed embeddings so we don't have to ask Gemini again."""
        if not items:
            return
        now = time.time()
        with closing(_connect(self._db_path)) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings_v2 "
                "(content_hash, model, task_type, vector, created_at) VALUES (?, ?, ?, ?, ?)",
                [(h, model, task_type, json.dumps(vec), now) for h, vec in items.items()],
            )
            conn.commit()


class FileStateStore:
    """Remembers each file's SHA256 so we can skip it if nothing changed."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def get_file_hash(self, file_path: str) -> str | None:
        with closing(_connect(self._db_path)) as conn:
            row = conn.execute(
                "SELECT file_hash FROM file_state WHERE file_path = ? AND status = 'completed'",
                (file_path,),
            ).fetchone()
        return row[0] if row else None

    def mark_in_progress(self, file_path: str, file_hash: str) -> None:
        self._upsert(file_path, file_hash, "in_progress")

    def mark_completed(self, file_path: str, file_hash: str) -> None:
        self._upsert(file_path, file_hash, "completed")

    def _upsert(self, file_path: str, file_hash: str, status: str) -> None:
        with closing(_connect(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO file_state (file_path, file_hash, status, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (file_path, file_hash, status, time.time()),
            )
            conn.commit()


class IngestedChunkStore:
    """Keeps a record of which chunks actually made it into the vector store.

    This is the piece that makes resuming after a crash actually safe — if
    a chunk's hash shows up in here, we know it's already in Chroma and skip
    it entirely instead of re-embedding or re-inserting it.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def filter_new(self, chunk_hashes: list[str]) -> set[str]:
        """Filter out whatever's already ingested, leaving only the genuinely new ones."""
        if not chunk_hashes:
            return set()
        with closing(_connect(self._db_path)) as conn:
            placeholders = ",".join("?" for _ in chunk_hashes)
            rows = conn.execute(
                f"SELECT chunk_hash FROM ingested_chunks WHERE chunk_hash IN ({placeholders})",
                chunk_hashes,
            ).fetchall()
        already_done = {row[0] for row in rows}
        return set(chunk_hashes) - already_done

    def mark_ingested(self, chunk_hashes: list[str], source: str) -> None:
        if not chunk_hashes:
            return
        now = time.time()
        with closing(_connect(self._db_path)) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO ingested_chunks (chunk_hash, source, updated_at) "
                "VALUES (?, ?, ?)",
                [(h, source, now) for h in chunk_hashes],
            )
            conn.commit()
