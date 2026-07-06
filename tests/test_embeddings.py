import pytest

from cricllm.cache import EmbeddingCache
from cricllm.embeddings import CachedGeminiEmbeddings
from cricllm.exceptions import EmbeddingAPIError
from cricllm.hashing import sha256_of_text

MODEL = "models/gemini-embedding-001"


def test_partial_batch_progress_is_cached_before_a_later_batch_fails(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    embedder = CachedGeminiEmbeddings(
        model=MODEL,
        api_key="dummy-key",
        task_type="RETRIEVAL_DOCUMENT",
        cache=cache,
        batch_size=2,
        max_retries=1,
        show_progress=False,
    )

    call_count = {"n": 0}

    def fake_embed_with_retry(batch):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [[0.1, 0.2] for _ in batch]
        raise RuntimeError("simulated API failure (e.g. rate limit)")

    embedder._embed_with_retry = fake_embed_with_retry

    texts = ["alpha", "beta", "gamma", "delta"]  # batch_size=2 -> two batches

    with pytest.raises(EmbeddingAPIError):
        embedder.embed_documents(texts)

    # The first batch succeeded before the second failed — it must already be
    # durably cached, so a retried run doesn't have to re-embed it.
    cached = cache.get_many(
        [sha256_of_text("alpha"), sha256_of_text("beta")], MODEL, "RETRIEVAL_DOCUMENT"
    )
    assert len(cached) == 2

    not_yet_cached = cache.get_many(
        [sha256_of_text("gamma"), sha256_of_text("delta")], MODEL, "RETRIEVAL_DOCUMENT"
    )
    assert len(not_yet_cached) == 0


def test_a_retried_call_only_re_embeds_what_is_still_missing(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    embedder = CachedGeminiEmbeddings(
        model=MODEL,
        api_key="dummy-key",
        task_type="RETRIEVAL_DOCUMENT",
        cache=cache,
        batch_size=2,
        max_retries=1,
        show_progress=False,
    )

    seen_batches: list[list[str]] = []

    def flaky_then_ok(batch):
        seen_batches.append(list(batch))
        if len(seen_batches) == 1:
            return [[0.1, 0.2] for _ in batch]
        if len(seen_batches) == 2:
            raise RuntimeError("simulated API failure")
        return [[0.3, 0.4] for _ in batch]

    embedder._embed_with_retry = flaky_then_ok

    texts = ["alpha", "beta", "gamma", "delta"]
    with pytest.raises(EmbeddingAPIError):
        embedder.embed_documents(texts)

    # Re-run with the same texts: only "gamma"/"delta" should hit the API again.
    seen_batches.clear()
    vectors = embedder.embed_documents(texts)
    assert len(vectors) == 4
    assert seen_batches == [["gamma", "delta"]]


def test_cache_does_not_leak_between_task_types(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")

    doc_embedder = CachedGeminiEmbeddings(
        model=MODEL, api_key="dummy-key", task_type="RETRIEVAL_DOCUMENT", cache=cache,
        show_progress=False,
    )
    doc_embedder._embed_with_retry = lambda batch: [[1.0, 1.0] for _ in batch]

    query_embedder = CachedGeminiEmbeddings(
        model=MODEL, api_key="dummy-key", task_type="RETRIEVAL_QUERY", cache=cache,
        show_progress=False,
    )
    query_embedder._embed_with_retry = lambda batch: [[9.0, 9.0] for _ in batch]

    same_text = "What is a payment link?"
    doc_vector = doc_embedder.embed_documents([same_text])[0]
    query_vector = query_embedder.embed_query(same_text)

    # Identical text embedded under two task types must not share a cache
    # entry — otherwise the query embedding would silently come back as the
    # document embedding (or vice versa), corrupting retrieval quality.
    assert doc_vector == [1.0, 1.0]
    assert query_vector == [9.0, 9.0]
