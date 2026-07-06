from cricllm.markdown_structure import split_by_headers

SAMPLE = """# Payments

Overview text.

## Create a Payment

Create a payment via the API.

### Request Body

Body details.

### Response

Response details.

## Fetch a Payment

Fetch details.
"""


def test_headers_preserved_as_metadata():
    sections = split_by_headers(SAMPLE)
    assert len(sections) >= 4

    request_body_section = next(s for s in sections if s.headers.get("h3") == "Request Body")
    assert request_body_section.headers["h1"] == "Payments"
    assert request_body_section.headers["h2"] == "Create a Payment"
    assert request_body_section.header_path == "Payments > Create a Payment > Request Body"


def test_sections_do_not_leak_content_across_headers():
    sections = split_by_headers(SAMPLE)
    fetch_section = next(s for s in sections if s.headers.get("h2") == "Fetch a Payment")
    assert "Fetch details" in fetch_section.content
    assert "Request Body" not in fetch_section.content
