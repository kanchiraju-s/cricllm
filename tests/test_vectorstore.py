"""Tests for the Pinecone-backed VectorStore, using a fake client double.

No real network/API key involved — we patch `cricllm.vectorstore.Pinecone`
with a small in-memory fake that mimics just the surface area we use.
"""

from __future__ import annotations

from langchain_core.documents import Document

from cricllm.vectorstore import VectorStore, _sanitize_metadata


class _FakeIndex:
    def __init__(self) -> None:
        self._vectors: dict[str, dict] = {}
        self.upsert_calls: list[list[dict]] = []
        self.query_response: list = []

    def upsert(self, *, vectors, batch_size=None, show_progress=True):
        self.upsert_calls.append(list(vectors))
        for v in vectors:
            self._vectors[v["id"]] = v

    def fetch(self, *, ids):
        class _FetchResult:
            def __init__(self, vectors: dict) -> None:
                self.vectors = vectors

        return _FetchResult({i: self._vectors[i] for i in ids if i in self._vectors})

    def query(self, *, vector, top_k, include_metadata=False):
        class _QueryResult:
            def __init__(self, matches: list) -> None:
                self.matches = matches

        return _QueryResult(self.query_response[:top_k])

    def describe_index_stats(self):
        class _Stats:
            def __init__(self, count: int) -> None:
                self.total_vector_count = count

        return _Stats(len(self._vectors))


class _FakeMatch:
    def __init__(self, id: str, score: float, metadata: dict) -> None:
        self.id = id
        self.score = score
        self.metadata = metadata


class _FakePinecone:
    """Drop-in fake for `pinecone.Pinecone`, tracking created indexes."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self._indexes: dict[str, _FakeIndex] = {}
        self.create_index_calls: list[dict] = []

    def has_index(self, name: str) -> bool:
        return name in self._indexes

    def create_index(self, *, name, dimension, metric, spec):
        self.create_index_calls.append({"name": name, "dimension": dimension, "metric": metric})
        self._indexes[name] = _FakeIndex()

    def Index(self, name: str) -> _FakeIndex:
        return self._indexes.setdefault(name, _FakeIndex())


def _make_store(monkeypatch, fake_pinecone: _FakePinecone | None = None) -> tuple[VectorStore, _FakePinecone]:
    fake_pinecone = fake_pinecone or _FakePinecone()
    monkeypatch.setattr("cricllm.vectorstore.Pinecone", lambda api_key=None: fake_pinecone)
    store = VectorStore(api_key="dummy-key", index_name="test-index", dimension=8)
    return store, fake_pinecone


def test_creates_index_if_missing(monkeypatch):
    store, fake = _make_store(monkeypatch)
    assert fake.create_index_calls == [{"name": "test-index", "dimension": 8, "metric": "cosine"}]
    assert store.count() == 0


def test_does_not_recreate_an_existing_index(monkeypatch):
    fake = _FakePinecone()
    fake._indexes["test-index"] = _FakeIndex()  # pre-existing
    _make_store(monkeypatch, fake)
    assert fake.create_index_calls == []


def test_upsert_packs_chunk_content_into_metadata(monkeypatch):
    store, fake = _make_store(monkeypatch)
    doc = Document(page_content="the actual chunk text", metadata={"header_path": "Law 21"})

    store.upsert(documents=[doc], ids=["abc123"], embeddings=[[0.1] * 8])

    [vectors] = fake.Index("test-index").upsert_calls
    assert vectors[0]["id"] == "abc123"
    assert vectors[0]["values"] == [0.1] * 8
    assert vectors[0]["metadata"]["content"] == "the actual chunk text"
    assert vectors[0]["metadata"]["header_path"] == "Law 21"


def test_query_converts_similarity_to_distance_and_unpacks_content(monkeypatch):
    store, fake = _make_store(monkeypatch)
    index = fake.Index("test-index")
    index.query_response = [
        _FakeMatch(id="a", score=1.0, metadata={"content": "same direction", "header_path": "Law 1"}),
        _FakeMatch(id="b", score=0.0, metadata={"content": "orthogonal", "header_path": "Law 2"}),
    ]

    results = store.query([1.0, 0.0], n_results=2)

    assert results[0]["id"] == "a"
    assert results[0]["distance"] == 0.0  # perfect similarity -> zero distance
    assert results[0]["content"] == "same direction"
    assert "content" not in results[0]["metadata"]  # unpacked out, not duplicated
    assert results[1]["distance"] == 1.0


def test_existing_ids_only_returns_ids_actually_present(monkeypatch):
    store, fake = _make_store(monkeypatch)
    store.upsert(
        documents=[Document(page_content="x", metadata={})],
        ids=["present"],
        embeddings=[[0.0] * 8],
    )

    found = store.existing_ids(["present", "missing"])

    assert found == {"present"}


def test_sanitize_metadata_drops_none_and_flattens_unsupported_types():
    sanitized = _sanitize_metadata(
        {
            "keep_str": "hello",
            "keep_int": 5,
            "keep_bool": True,
            "keep_list_of_str": ["a", "b"],
            "drop_none": None,
            "flatten_dict": {"nested": 1},
        }
    )
    assert sanitized == {
        "keep_str": "hello",
        "keep_int": 5,
        "keep_bool": True,
        "keep_list_of_str": ["a", "b"],
        "flatten_dict": "{'nested': 1}",
    }
    assert "drop_none" not in sanitized
