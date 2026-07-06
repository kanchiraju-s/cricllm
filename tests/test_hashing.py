from pathlib import Path

from cricllm.hashing import sha256_of_file, sha256_of_text


def test_sha256_of_text_is_deterministic():
    assert sha256_of_text("hello") == sha256_of_text("hello")


def test_sha256_of_text_differs_for_different_input():
    assert sha256_of_text("hello") != sha256_of_text("world")


def test_sha256_of_file(tmp_path: Path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Title\ncontent", encoding="utf-8")
    assert sha256_of_file(file_path) == sha256_of_text("# Title\ncontent")
