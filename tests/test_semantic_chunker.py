from cricllm.semantic_chunker import chunk_markdown
from cricllm.token_utils import count_tokens

DOC = """# Payments API

## Create a Payment

Creates a new payment. Use this endpoint to charge a customer.

### Request

```bash
curl -X POST https://api.example.com/v1/payments \\
  -u key:secret \\
  -d amount=50000 \\
  -d currency=INR
```

### Response

```json
{
  "id": "pay_123",
  "amount": 50000,
  "currency": "INR",
  "status": "captured"
}
```

### Error Codes

| Code | Meaning |
| ---- | ------- |
| 400  | Bad request |
| 401  | Unauthorized |

## Notes

- Amounts are in the smallest currency unit (paise for INR).
- Idempotency keys are recommended for retries.
"""


def test_chunk_markdown_never_splits_code_blocks():
    documents = chunk_markdown(
        DOC,
        source="test_doc.md",
        min_tokens=10,
        max_tokens=40,
        hard_max_tokens=2000,
        use_semantic_chunking=False,
        embeddings=None,
    )
    joined_json_block = '"id": "pay_123"'
    matches = [d for d in documents if joined_json_block in d.page_content]
    assert len(matches) == 1
    assert '"status": "captured"' in matches[0].page_content


def test_chunk_markdown_never_splits_curl_command():
    documents = chunk_markdown(
        DOC,
        source="test_doc.md",
        min_tokens=10,
        max_tokens=40,
        hard_max_tokens=2000,
        use_semantic_chunking=False,
        embeddings=None,
    )
    matches = [d for d in documents if "curl -X POST" in d.page_content]
    assert len(matches) == 1
    assert "-d currency=INR" in matches[0].page_content


def test_chunk_markdown_never_splits_table():
    documents = chunk_markdown(
        DOC,
        source="test_doc.md",
        min_tokens=10,
        max_tokens=40,
        hard_max_tokens=2000,
        use_semantic_chunking=False,
        embeddings=None,
    )
    matches = [d for d in documents if "| Code | Meaning |" in d.page_content]
    assert len(matches) == 1
    assert "| 401  | Unauthorized |" in matches[0].page_content


def test_chunk_metadata_has_header_path_and_hash():
    documents = chunk_markdown(
        DOC,
        source="test_doc.md",
        min_tokens=10,
        max_tokens=40,
        hard_max_tokens=2000,
        use_semantic_chunking=False,
        embeddings=None,
    )
    assert all("chunk_hash" in d.metadata for d in documents)
    assert all("header_path" in d.metadata for d in documents)
    hashes = [d.metadata["chunk_hash"] for d in documents]
    assert len(hashes) == len(set(hashes))
    response_doc = next(d for d in documents if "pay_123" in d.page_content)
    assert response_doc.metadata["header_path"] == "Payments API > Create a Payment > Response"


def test_oversized_code_block_is_force_split_to_fit_embedding_limit():
    huge_body = "\n".join(f"line_{i}: some filler payload data here" for i in range(400))
    doc_text = f"# Docs\n\n## Huge Schema\n\n```json\n{huge_body}\n```\n"

    documents = chunk_markdown(
        doc_text,
        source="test_doc.md",
        min_tokens=10,
        max_tokens=40,
        hard_max_tokens=100,
        max_embedding_input_tokens=500,
        use_semantic_chunking=False,
        embeddings=None,
    )

    assert len(documents) > 1
    forced_docs = [d for d in documents if d.metadata["forced_split"]]
    assert len(forced_docs) > 1  # the oversized code block had to be split into multiple pieces
    assert all(count_tokens(d.page_content) <= 500 for d in documents)
    # Content survives the split (allowing for whitespace normalization at split points).
    rejoined = "".join(d.page_content for d in documents)
    assert "line_0:" in rejoined
    assert "line_399:" in rejoined


def test_normal_sized_atomic_block_is_not_marked_forced_split():
    documents = chunk_markdown(
        DOC,
        source="test_doc.md",
        min_tokens=10,
        max_tokens=40,
        hard_max_tokens=2000,
        use_semantic_chunking=False,
        embeddings=None,
    )
    assert all(d.metadata["forced_split"] is False for d in documents)
