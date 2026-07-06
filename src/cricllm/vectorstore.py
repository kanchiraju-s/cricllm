"""Our Pinecone-backed vector store.

We used to run this locally on `chromadb`, but its Python client
reproducibly hung the moment it was called from inside a live gunicorn
worker on Render — confirmed through extensive isolation testing to be
neither our code, the data, the API key, nor memory pressure, but something
specific to chromadb's own (Rust-backed) client running in that particular
process context. Pinecone's client is a plain REST/HTTP SDK with no local
persistent Rust core, which sidesteps that entirely — and as a bonus, it
also means the rulebook's actual text no longer needs to live on our own
disk (or in git) at all, since Pinecone hosts it.

Chunk content hashes are used as vector IDs, same trick as before: re-upserting
an unchanged chunk overwrites the same row instead of creating a duplicate.
"""

from __future__ import annotations

from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from cricllm.logging_config import get_logger

logger = get_logger("vectorstore")

_UPSERT_BATCH_SIZE = 100
_FETCH_BATCH_SIZE = 200


class VectorStore:
    """Thin wrapper around one Pinecone index."""

    def __init__(self, api_key: str, index_name: str, dimension: int) -> None:
        self._client = Pinecone(api_key=api_key)
        if not self._client.has_index(index_name):
            logger.info("Creating Pinecone index %r (dimension=%d, metric=cosine)", index_name, dimension)
            self._client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self._index = self._client.Index(index_name)

    def existing_ids(self, ids: list[str]) -> set[str]:
        """Which of these ids are already sitting in the index?"""
        if not ids:
            return set()
        found: set[str] = set()
        for i in range(0, len(ids), _FETCH_BATCH_SIZE):
            batch = ids[i : i + _FETCH_BATCH_SIZE]
            result = self._index.fetch(ids=batch)
            found.update(result.vectors.keys())
        return found

    def upsert(self, documents: list[Document], ids: list[str], embeddings: list[list[float]]) -> None:
        """Write documents + their already-computed embeddings into Pinecone, in batches."""
        if not documents:
            return
        vectors = [
            {
                "id": doc_id,
                "values": vector,
                # Pinecone has nowhere else to put the chunk text, so it
                # rides along in metadata under "content" and gets pulled
                # back out in query().
                "metadata": _sanitize_metadata({**doc.metadata, "content": doc.page_content}),
            }
            for doc_id, doc, vector in zip(ids, documents, embeddings)
        ]
        self._index.upsert(vectors=vectors, batch_size=_UPSERT_BATCH_SIZE, show_progress=False)
        logger.info("Upserted %d chunks into Pinecone index", len(documents))

    def query(self, query_embedding: list[float], n_results: int = 5) -> list[dict]:
        """Find the ``n_results`` chunks closest to this query embedding.

        Each result has ``id``, ``content``, ``metadata``, and ``distance``
        — the smaller the distance, the closer the match.
        """
        result = self._index.query(vector=query_embedding, top_k=n_results, include_metadata=True)
        matches = []
        for match in result.matches:
            metadata = dict(match.metadata or {})
            content = metadata.pop("content", "")
            # Pinecone hands back a cosine SIMILARITY (higher = closer);
            # everything downstream expects a "distance" (lower = closer),
            # same convention Chroma used, so the rest of the app doesn't
            # need to know or care which vector store is behind it.
            distance = 1 - match.score
            matches.append(
                {"id": match.id, "content": content, "metadata": metadata, "distance": distance}
            )
        return matches

    def count(self) -> int:
        return self._index.describe_index_stats().total_vector_count


def _sanitize_metadata(metadata: dict) -> dict:
    """Pinecone metadata is str/int/float/bool/list[str] — no None, nothing fancier."""
    sanitized: dict = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized
