"""Wraps Gemini's embedding API with batching, retries, and caching.

We only ever hit the real hosted API here — no local fallback model. All
the retry/batch/cache machinery below exists for one reason: a ~13k-line
rulebook is a lot of API calls, and we really don't want to pay for the
same text twice or lose a whole run to one flaky request.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from cricllm.cache import EmbeddingCache
from cricllm.exceptions import EmbeddingAPIError
from cricllm.hashing import sha256_of_text
from cricllm.logging_config import get_logger

logger = get_logger("embeddings")


class CachedGeminiEmbeddings(Embeddings):
    """Batches texts, retries on failure, and caches results to disk.

    Use ``task_type="RETRIEVAL_DOCUMENT"`` when ingesting, and spin up a
    separate instance with ``"RETRIEVAL_QUERY"`` when answering questions —
    Gemini embeds the same text differently depending on which one you pick.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        task_type: str,
        cache: EmbeddingCache,
        batch_size: int = 32,
        max_retries: int = 6,
        retry_min_seconds: float = 1.0,
        retry_max_seconds: float = 60.0,
        show_progress: bool = True,
    ) -> None:
        self._model = model
        self._task_type = task_type
        self._cache = cache
        self._batch_size = batch_size
        self._show_progress = show_progress
        self._client = GoogleGenerativeAIEmbeddings(  # type: ignore[call-arg]  # pydantic dynamic init confuses mypy; verified working at runtime
            model=model, google_api_key=api_key, task_type=task_type
        )
        self._embed_with_retry = retry(
            reraise=True,
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=retry_min_seconds, max=retry_max_seconds),
            before_sleep=self._log_retry,
        )(self._client.embed_documents)

    @staticmethod
    def _log_retry(retry_state) -> None:  # noqa: ANN001 - tenacity callback signature
        logger.warning(
            "Embedding API call failed (attempt %d), retrying after backoff: %s",
            retry_state.attempt_number,
            retry_state.outcome.exception(),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts — pull cached ones for free, only pay for the rest."""
        if not texts:
            return []

        hashes = [sha256_of_text(t) for t in texts]
        cached = self._cache.get_many(hashes, self._model, self._task_type)

        results: list[Optional[list[float]]] = [cached.get(h) for h in hashes]
        pending_indices = [i for i, vec in enumerate(results) if vec is None]

        if pending_indices:
            logger.info(
                "%d/%d texts already cached, embedding %d new texts via Gemini API",
                len(texts) - len(pending_indices),
                len(texts),
                len(pending_indices),
            )
            pending_hashes = [hashes[i] for i in pending_indices]
            pending_texts = [texts[i] for i in pending_indices]
            new_vectors = self._embed_in_batches(pending_texts, pending_hashes)
            for idx, vector in zip(pending_indices, new_vectors):
                results[idx] = vector

        return results  # type: ignore[return-value]

    def _embed_in_batches(self, texts: list[str], hashes: list[str]) -> list[list[float]]:
        """Send texts to Gemini in batches, caching each batch as soon as it lands.

        We used to cache only after *all* batches succeeded, which meant a
        rate limit on batch 5 of 8 threw away batches 1-4's work too. Now
        each batch gets saved the moment it comes back, so a failure partway
        through only costs you the batch that actually failed — re-running
        `embed_documents` afterwards picks up right where it left off.
        """
        vectors: list[list[float]] = []
        batch_bounds = [
            (i, min(i + self._batch_size, len(texts))) for i in range(0, len(texts), self._batch_size)
        ]
        iterator = tqdm(batch_bounds, desc="Embedding batches", disable=not self._show_progress)
        for start, end in iterator:
            batch_texts = texts[start:end]
            batch_hashes = hashes[start:end]
            try:
                batch_vectors = self._embed_with_retry(batch_texts)
            except Exception as exc:
                raise EmbeddingAPIError(
                    f"Embedding API failed after retries for a batch of {len(batch_texts)} "
                    f"texts ({len(vectors)} texts from earlier batches in this call are "
                    "already cached and won't be re-embedded on retry)"
                ) from exc
            vectors.extend(batch_vectors)
            self._cache.set_many(dict(zip(batch_hashes, batch_vectors)), self._model, self._task_type)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string — same cache and everything as embed_documents."""
        return self.embed_documents([text])[0]
