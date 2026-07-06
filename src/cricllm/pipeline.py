"""This ties everything together: load a file, chunk it, embed it, store it.

A few things make it safe to kill and re-run at any point:

* If a file's SHA256 hasn't changed since last time, we skip it entirely.
* Before spending any API quota, we check whether a chunk's already in the
  "ingested" ledger or already sitting in Chroma — that's what makes resume
  and duplicate detection work.
* If a batch fails, its chunks get written to a dead-letter JSON file
  instead of the whole run just falling over.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_core.documents import Document

from cricllm.cache import EmbeddingCache, FileStateStore, IngestedChunkStore
from cricllm.config import Settings
from cricllm.embeddings import CachedGeminiEmbeddings
from cricllm.exceptions import (
    CricLLMError,
    CorruptedMarkdownError,
    EmptyDocumentError,
    InvalidEncodingError,
)
from cricllm.hashing import sha256_of_file
from cricllm.loader import load_markdown_file
from cricllm.logging_config import get_logger
from cricllm.models import IngestionStats
from cricllm.semantic_chunker import chunk_markdown
from cricllm.vectorstore import VectorStore

logger = get_logger("pipeline")


class IngestionPipeline:
    """Owns everything a single ingestion run needs — cache, embedder, vector store."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_directories()
        self.embedding_cache = EmbeddingCache(settings.cache_db)
        self.file_state = FileStateStore(settings.cache_db)
        self.ingested_chunks = IngestedChunkStore(settings.cache_db)
        self.embeddings = CachedGeminiEmbeddings(
            model=settings.embedding_model,
            api_key=settings.google_api_key,
            task_type=settings.embedding_task_type,
            cache=self.embedding_cache,
            batch_size=settings.embed_batch_size,
            max_retries=settings.max_retries,
            retry_min_seconds=settings.retry_min_seconds,
            retry_max_seconds=settings.retry_max_seconds,
        )
        self.vector_store = VectorStore(
            settings.pinecone_api_key, settings.pinecone_index_name, settings.embedding_dimension
        )

    def ingest_file(self, path: Path, force: bool = False) -> IngestionStats:
        """Run one file through the whole pipeline. Always returns stats, even on failure."""
        stats = IngestionStats(source=str(path))
        started = time.monotonic()

        try:
            file_hash = sha256_of_file(path)
        except OSError as exc:
            logger.error("Could not read %s: %s", path, exc)
            stats.error = str(exc)
            return stats

        previous_hash = self.file_state.get_file_hash(str(path))
        if not force and previous_hash == file_hash:
            logger.info("Skipping %s — unchanged since last successful ingestion", path)
            stats.skipped_unchanged = True
            return stats

        self.file_state.mark_in_progress(str(path), file_hash)

        try:
            text = load_markdown_file(path)
        except (EmptyDocumentError, InvalidEncodingError, CorruptedMarkdownError) as exc:
            logger.error("Failed to load %s: %s", path, exc)
            stats.error = str(exc)
            return stats

        documents = chunk_markdown(
            text,
            source=str(path),
            min_tokens=self.settings.min_chunk_tokens,
            max_tokens=self.settings.max_chunk_tokens,
            hard_max_tokens=self.settings.hard_max_chunk_tokens,
            max_embedding_input_tokens=self.settings.max_embedding_input_tokens,
            use_semantic_chunking=self.settings.use_semantic_chunking,
            embeddings=self.embeddings,
            semantic_breakpoint_threshold_type=self.settings.semantic_breakpoint_threshold_type,
        )
        stats.total_chunks = len(documents)
        logger.info("Chunked %s into %d chunks", path, len(documents))

        unique_documents = []
        seen_hashes: set[str] = set()
        for doc in documents:
            chunk_hash = doc.metadata["chunk_hash"]
            if chunk_hash in seen_hashes:
                stats.duplicate_chunks += 1
                continue
            seen_hashes.add(chunk_hash)
            unique_documents.append(doc)

        candidate_ids = [doc.metadata["chunk_hash"] for doc in unique_documents]
        not_yet_marked = self.ingested_chunks.filter_new(candidate_ids)
        not_yet_in_store = not_yet_marked - self.vector_store.existing_ids(list(not_yet_marked))
        stats.skipped_already_ingested = len(unique_documents) - len(not_yet_in_store)

        pending_docs = [
            doc for doc in unique_documents if doc.metadata["chunk_hash"] in not_yet_in_store
        ]

        if pending_docs:
            try:
                texts = [doc.page_content for doc in pending_docs]
                vectors = self.embeddings.embed_documents(texts)
                pending_ids = [doc.metadata["chunk_hash"] for doc in pending_docs]
                self.vector_store.upsert(pending_docs, pending_ids, vectors)
                self.ingested_chunks.mark_ingested(pending_ids, str(path))
                stats.ingested_chunks = len(pending_docs)
            except CricLLMError as exc:
                logger.error("Embedding/upsert failed for %s: %s", path, exc)
                stats.error = str(exc)
                self._write_dead_letter(path, pending_docs, exc)
                return stats

        self.file_state.mark_completed(str(path), file_hash)
        stats.duration_seconds = time.monotonic() - started
        logger.info(
            "Completed %s: %d ingested, %d duplicates skipped, %d already-ingested skipped, "
            "%d total chunks, %.2fs",
            path,
            stats.ingested_chunks,
            stats.duplicate_chunks,
            stats.skipped_already_ingested,
            stats.total_chunks,
            stats.duration_seconds,
        )
        return stats

    def _write_dead_letter(self, path: Path, documents: list[Document], exc: Exception) -> None:
        """Save the chunks that didn't make it, so we can figure out what to do about them later."""
        self.settings.dead_letter_dir.mkdir(parents=True, exist_ok=True)
        dead_letter_path = self.settings.dead_letter_dir / f"{path.stem}_{int(time.time())}.json"
        payload = {
            "source": str(path),
            "error": str(exc),
            "chunk_hashes": [doc.metadata["chunk_hash"] for doc in documents],
        }
        dead_letter_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.error("Wrote dead-letter record for %s to %s", path, dead_letter_path)
