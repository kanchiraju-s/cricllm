from cricllm.block_parser import split_into_segments


def test_code_block_is_atomic_and_not_split():
    text = "Some intro text.\n\n```json\n{\n  \"a\": 1\n}\n```\n\nMore text after.\n"
    segments = split_into_segments(text)
    code_segments = [s for s in segments if s.type == "code"]
    assert len(code_segments) == 1
    assert "```json" in code_segments[0].content
    assert "```" in code_segments[0].content.strip().splitlines()[-1]


def test_table_is_detected_as_atomic():
    text = (
        "Intro\n\n"
        "| Field | Type |\n"
        "| ----- | ---- |\n"
        "| id    | str  |\n"
        "| amount | int |\n\n"
        "Outro\n"
    )
    segments = split_into_segments(text)
    table_segments = [s for s in segments if s.type == "table"]
    assert len(table_segments) == 1
    assert "| Field | Type |" in table_segments[0].content
    assert "| amount | int |" in table_segments[0].content


def test_list_is_kept_together():
    text = "Notes:\n\n- point one\n- point two\n- point three\n\nDone.\n"
    segments = split_into_segments(text)
    list_segments = [s for s in segments if s.type == "list"]
    assert len(list_segments) == 1
    assert list_segments[0].content.count("- point") == 3


def test_curl_command_is_atomic():
    text = (
        "Example:\n\n"
        "curl -X POST https://api.example.com/v1/payments \\\n"
        "  -u key:secret \\\n"
        "  -d amount=100\n\n"
        "That's it.\n"
    )
    segments = split_into_segments(text)
    curl_segments = [s for s in segments if s.type == "curl"]
    assert len(curl_segments) == 1
    assert "curl -X POST" in curl_segments[0].content
    assert "-d amount=100" in curl_segments[0].content


def test_prose_segments_capture_plain_paragraphs():
    text = "Just a plain paragraph with no special formatting.\n"
    segments = split_into_segments(text)
    assert len(segments) == 1
    assert segments[0].type == "prose"
