"""First step of chunking: split the rulebook on its own header hierarchy.

Before we even think about token counts, we want the document broken into
sections that each know their full header path (h1 > h2 > h3 > h4) — that's
what lets a chunk later say "I'm from Law 21 > No ball" instead of just
being an anonymous blob of text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import MarkdownHeaderTextSplitter

_HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


@dataclass
class MarkdownSection:
    """One chunk of Markdown that lives under a specific header path."""

    content: str
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def header_path(self) -> str:
        """Breadcrumb string, e.g. 'Law 21 > No ball > 21.17 No ball not to count'."""
        ordered = [self.headers[key] for key in ("h1", "h2", "h3", "h4") if key in self.headers]
        return " > ".join(ordered)


def split_by_headers(markdown_text: str) -> list[MarkdownSection]:
    """Split raw Markdown into sections, one per header, using LangChain under the hood.

    If there's a preamble before the very first ``#``, we still keep it —
    it just ends up with an empty ``headers`` dict.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    documents = splitter.split_text(markdown_text)

    sections = [
        MarkdownSection(content=doc.page_content, headers=dict(doc.metadata))
        for doc in documents
        if doc.page_content.strip()
    ]
    return sections
