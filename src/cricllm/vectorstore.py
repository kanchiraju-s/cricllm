"""Our Chroma vector store, wrapped up so re-running ingestion is always safe.

We talk to `chromadb` directly instead of going through `langchain_chroma`.
The convenience wrapper would recompute embeddings for us on every insert,
which defeats the whole point of caching them ourselves.

We also use each chunk's content hash as its Chroma document ID. That's the
whole trick behind duplicate detection: re-inserting the same chunk just
overwrites the same row instead of creating a copy.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from cricllm.logging_config import get_logger

logger = get_logger("vectorstore")

_UPSERT_BATCH_SIZE = 100


class VectorStore:
    """Thin wrapper around one persistent Chroma collection."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        # Chroma defaults to L2 distance, but Gemini's embeddings aren't
        # guaranteed to be unit-length, so L2 can rank things differently
        # than actual semantic similarity would. Cosine is the right call
        # here. Only bites for a brand new collection though — Chroma won't
        # let you change the distance metric on one that already exists.
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        current_space = (self._collection.metadata or {}).get("hnsw:space", "l2")
        if current_space != "cosine":
            logger.warning(
                "Collection '%s' was created with '%s' distance, not 'cosine' — Chroma can't "
                "change this on an existing collection. To pick up cosine distance, set "
                "CRICLLM_COLLECTION_NAME to a new name and re-run ingestion; cached embeddings "
                "mean this re-upserts without any new API calls.",
                collection_name,
                current_space,
            )

    def existing_ids(self, ids: list[str]) -> set[str]:
        """Which of these ids are already sitting in the collection?"""
        if not ids:
            return set()
        found: set[str] = set()
        for i in range(0, len(ids), _UPSERT_BATCH_SIZE):
            batch = ids[i : i + _UPSERT_BATCH_SIZE]
            result = self._collection.get(ids=batch, include=[])
            found.update(result["ids"])
        return found

    def upsert(self, documents: list[Document], ids: list[str], embeddings: list[list[float]]) -> None:
        """Write documents + their already-computed embeddings into Chroma, in batches."""
        if not documents:
            return
        for i in range(0, len(documents), _UPSERT_BATCH_SIZE):
            batch_docs = documents[i : i + _UPSERT_BATCH_SIZE]
            batch_ids = ids[i : i + _UPSERT_BATCH_SIZE]
            batch_embeddings = embeddings[i : i + _UPSERT_BATCH_SIZE]
            self._collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,  # type: ignore[arg-type]  # chromadb accepts plain list[list[float]] at runtime
                documents=[doc.page_content for doc in batch_docs],
                metadatas=[_sanitize_metadata(doc.metadata) for doc in batch_docs],
            )
        logger.info("Upserted %d chunks into collection", len(documents))

    def query(self, query_embedding: list[float], n_results: int = 5) -> list[dict]:
        """Find the ``n_results`` chunks closest to this query embedding.

        Each result has ``id``, ``content``, ``metadata``, and ``distance``
        — the smaller the distance, the closer the match.
        """
        result = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]  # chromadb accepts plain list[list[float]] at runtime
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        # mypy thinks these could be None because the type stub covers the
        # case where you didn't ask for them via `include` — we did, so
        # they're always there.
        ids = result["ids"][0]
        documents = result["documents"][0]  # type: ignore[index]
        metadatas = result["metadatas"][0]  # type: ignore[index]
        distances = result["distances"][0]  # type: ignore[index]
        return [
            {"id": doc_id, "content": content, "metadata": metadata, "distance": distance}
            for doc_id, content, metadata, distance in zip(ids, documents, metadatas, distances)
        ]

    def count(self) -> int:
        return self._collection.count()


def _sanitize_metadata(metadata: dict) -> dict:
    """Chroma only accepts str/int/float/bool in metadata, so flatten anything fancier."""
    sanitized: dict = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = ",".join(str(v) for v in value)
        else:
            sanitized[key] = str(value)
    return sanitized
