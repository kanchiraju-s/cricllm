#!/usr/bin/env python3
"""The browser version of scripts/ask.py.

Run:
    pip install flask
    python app.py
Then open http://localhost:5000

One JSON route (POST /api/ask) doing the same retrieval + generation as the
CLI script — both go through cricllm.qa.QAEngine so there's no duplicate
logic to keep in sync. The page itself is just plain HTML/CSS/JS, no
framework, no build step, nothing to install beyond Flask.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import bleach  # noqa: E402
import markdown as markdown_lib  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402

from cricllm.config import load_settings  # noqa: E402
from cricllm.exceptions import CricLLMError  # noqa: E402
from cricllm.logging_config import get_logger, setup_logging  # noqa: E402
from cricllm.qa import QAEngine  # noqa: E402

app = Flask(__name__)

_settings = load_settings()
setup_logging(_settings.log_dir)
logger = get_logger("webapp")
_engine = QAEngine(_settings)

# The model writes its answer in Markdown (it's told to, in the system
# prompt) — this turns that into HTML for the browser. We sanitize with
# bleach afterward since it's still LLM output landing in innerHTML; no
# reason to trust it more than any other user-facing content.
_ALLOWED_TAGS = [
    "p", "br", "strong", "em", "ul", "ol", "li",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "code", "pre", "hr", "a",
]
_ALLOWED_ATTRS = {"a": ["href", "title"]}


def render_answer_html(markdown_text: str) -> str:
    """Turn the model's Markdown answer into safe HTML for the browser."""
    raw_html = markdown_lib.markdown(markdown_text, extensions=["extra", "sane_lists"])
    return bleach.clean(raw_html, tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRS, strip=True)


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/api/ask")
def ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    top_k = payload.get("top_k")

    if not question:
        return jsonify({"error": "Question must not be empty."}), 400

    if not _engine.is_ready():
        return jsonify(
            {
                "error": (
                    "The vector store is empty. Run ingestion first: "
                    "python scripts/run_ingestion.py --input data/icc_rulebook.md"
                )
            }
        ), 503

    try:
        result = _engine.answer(question, top_k=top_k)
    except CricLLMError as exc:
        logger.error("Failed to answer %r: %s", question, exc)
        return jsonify({"error": str(exc)}), 502

    return jsonify(
        {
            "answer": result.answer,
            "answer_html": render_answer_html(result.answer),
            "sources": [
                {"header_path": s.header_path, "distance": s.distance} for s in result.sources
            ],
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
