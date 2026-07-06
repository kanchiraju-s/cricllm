import logging

from cricllm.vectorstore import VectorStore


def test_new_collection_uses_cosine_distance(tmp_path):
    store = VectorStore(tmp_path / ".chroma", "test_collection")
    assert store._collection.metadata["hnsw:space"] == "cosine"


def test_reopening_a_non_cosine_collection_logs_a_warning(tmp_path, caplog):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    persist_dir = tmp_path / ".chroma"
    client = chromadb.PersistentClient(
        path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )
    client.get_or_create_collection(name="legacy_collection")  # defaults to l2

    with caplog.at_level(logging.WARNING, logger="cricllm.vectorstore"):
        VectorStore(persist_dir, "legacy_collection")

    assert any("distance" in record.message for record in caplog.records)


def test_query_ranks_by_cosine_similarity(tmp_path):
    from langchain_core.documents import Document

    store = VectorStore(tmp_path / ".chroma", "test_collection")
    store.upsert(
        documents=[Document(page_content="same direction", metadata={"chunk_hash": "a"})],
        ids=["a"],
        embeddings=[[2.0, 0.0]],  # same direction as query, different magnitude
    )
    store.upsert(
        documents=[Document(page_content="orthogonal", metadata={"chunk_hash": "b"})],
        ids=["b"],
        embeddings=[[0.0, 1.0]],
    )

    results = store.query([1.0, 0.0], n_results=2)
    assert results[0]["id"] == "a"
    assert results[0]["distance"] < results[1]["distance"]
