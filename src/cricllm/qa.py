"""The actual "answer a question about the Laws of Cricket" logic lives here.

Both the CLI (scripts/ask.py) and the web UI (app.py) go through QAEngine
instead of each rolling their own copy — otherwise the system prompt and
retrieval logic would inevitably drift apart between the two.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from google import genai

from cricllm.cache import EmbeddingCache
from cricllm.config import Settings
from cricllm.embeddings import CachedGeminiEmbeddings
from cricllm.vectorstore import VectorStore

SYSTEM_PROMPT = """
You are an expert assistant for the ICC Laws of Cricket.

Your knowledge is STRICTLY LIMITED to the ICC Laws documentation provided in the retrieved context.

Your objective is to answer accurately, consistently, and ONLY from the retrieved documentation.

========================
RULES
========================

1. Use ONLY the retrieved documentation to answer the user's question.

2. Do NOT use outside knowledge, including:
   - MCC Laws not present in the context
   - ICC Playing Conditions
   - Historical matches
   - Commentary
   - General cricket knowledge
   - Internet knowledge

3. You MAY combine information from multiple retrieved laws to answer scenario-based or multi-step questions.

4. You MAY make logical inferences ONLY when they directly and unambiguously follow from the retrieved laws.

   Examples:
   - If a law applies to a batter's "equipment", you may conclude that it applies to a helmet worn by the batter if the documentation supports that interpretation.
   - If one law defines when the ball becomes dead and another law defines the consequences, combine both laws into a single answer.
   - If multiple retrieved laws together answer the user's question, synthesize them instead of treating each law independently.

5. Do NOT be overly conservative.

   If the answer can be directly inferred from one or more retrieved laws, provide the answer and explain your reasoning.

   Do NOT respond with "there is no explicit mention" merely because the exact wording of the user's question does not appear in the documentation.

   Only respond with:

   "I do not have enough information in the provided ICC Laws documentation to answer this question."

   when the retrieved documentation neither explicitly states NOR reasonably supports the conclusion.

6. Never invent:
   - laws
   - law numbers
   - exceptions
   - penalties
   - interpretations
   - match situations
   - procedures

7. If the question is ambiguous or could refer to multiple situations, explain each interpretation separately using only the retrieved documentation.

8. If two retrieved laws work together, explain how they relate instead of answering each law separately.

9. If two retrieved laws genuinely conflict, explain both instead of choosing one.

10. Never cite laws that are not present in the retrieved context.

========================
ANSWER FORMAT
========================

## Answer

Provide a clear, well-structured explanation.

If multiple situations are possible, organize the answer into separate cases.

Explain your reasoning using the retrieved laws.

## Relevant Laws

List every retrieved law that supports your answer.

## References

- Law X.X > Section > Subsection
- Law Y.Y > Section > Subsection

========================
CITATION RULES
========================

- Cite the relevant law number and section path after every factual statement.

Example:

(Law 20.1 > Ball is dead > 20.1.1.5)

(Law 28.3 > Protective helmets belonging to the fielding side > 28.3.2)

When quoting the documentation, reproduce only the minimum text necessary.

Never fabricate citations.

========================
RESPONSE QUALITY
========================

Always prefer a complete, synthesized answer over saying that information is missing.

For scenario-based questions:

1. Identify which laws apply.
2. Explain how those laws interact.
3. State the final outcome.
4. Mention any exceptions.
5. Cite every supporting law.

Your goal is to behave like an ICC Laws expert who is reading the official rulebook, combining the relevant laws, and explaining them accurately without introducing any information that is not supported by the retrieved documentation.
"""


@dataclass
class Source:
    """One retrieved chunk that fed into an answer."""

    header_path: str
    distance: float
    content: str


@dataclass
class StreamEvent:
    """One step of a streaming answer.

    ``kind="sources"`` fires once, right after retrieval, before we've asked
    Gemini anything. ``kind="delta"`` fires repeatedly as the answer comes
    in — ``text`` is the *full* answer so far each time, not just the new
    bit, since re-rendering the whole accumulated Markdown is what the web
    UI needs anyway.
    """

    kind: Literal["sources", "delta"]
    sources: list[Source] = field(default_factory=list)
    text: str = ""


class QAEngine:
    """Holds the embedder, vector store, and genai client so you don't rebuild them constantly.

    Make one of these per process and reuse it. A long-lived server like
    app.py should absolutely not be constructing a new one on every request.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        cache = EmbeddingCache(settings.cache_db)
        self._query_embedder = CachedGeminiEmbeddings(
            model=settings.embedding_model,
            api_key=settings.google_api_key,
            task_type="RETRIEVAL_QUERY",
            cache=cache,
            show_progress=False,
        )
        self._store = VectorStore(
            settings.pinecone_api_key, settings.pinecone_index_name, settings.embedding_dimension
        )
        self._client = genai.Client(api_key=settings.google_api_key)

    def is_ready(self) -> bool:
        """Is there actually anything in the vector store to search yet?"""
        return self._store.count() > 0

    def chunk_count(self) -> int:
        return self._store.count()

    def _retrieve(self, question: str, top_k: int | None) -> tuple[list[Source], str]:
        """Embed the question and pull back matching Law excerpts, built into a prompt-ready context."""
        resolved_top_k = top_k or self._settings.retrieval_top_k
        query_vector = self._query_embedder.embed_query(question)
        matches = self._store.query(query_vector, n_results=resolved_top_k)

        sources = []
        context_blocks = []
        for match in matches:
            header_path = match["metadata"].get("header_path", "unknown section")
            context_blocks.append(f"### {header_path}\n{match['content']}")
            sources.append(
                Source(header_path=header_path, distance=match["distance"], content=match["content"])
            )
        return sources, "\n\n---\n\n".join(context_blocks)

    def answer_stream(self, question: str, top_k: int | None = None) -> Iterator[StreamEvent]:
        """Same flow as before, but yields the answer as it's generated instead of all at once.

        Yields one ``StreamEvent(kind="sources", ...)`` right after retrieval,
        then a growing series of ``StreamEvent(kind="delta", text=...)`` as
        Gemini streams its response — each one carries the *full* answer so
        far, so a consumer can just re-render/replace on every event.
        """
        sources, context = self._retrieve(question, top_k)
        yield StreamEvent(kind="sources", sources=sources)

        if not sources:
            yield StreamEvent(kind="delta", text="No relevant chunks found.")
            return

        prompt = f"{SYSTEM_PROMPT}\n\n# Documentation excerpts\n\n{context}\n\n# Question\n{question}"

        accumulated = ""
        for chunk in self._client.models.generate_content_stream(
            model=self._settings.generation_model, contents=prompt
        ):
            if chunk.text:
                accumulated += chunk.text
                yield StreamEvent(kind="delta", text=accumulated)

        if not accumulated:
            # Safety-filtered, empty candidates, whatever — don't leave the
            # caller with nothing at all.
            yield StreamEvent(kind="delta", text="The model returned no answer for this question.")
