"""This is where the actual chunking happens. Roughly, in order:

1. Split on headers first (see markdown_structure.py) so every chunk knows
   which Law/Section it came from.
2. Within each section, separate out the stuff that shouldn't be split —
   code blocks, tables, lists, curl commands — from regular prose
   (see block_parser.py).
3. If a prose block is still too big on its own, split it again — ideally
   using LangChain's embedding-driven SemanticChunker so the breakpoints
   follow the actual meaning of the text, not just a character count. Falls
   back to a plain recursive splitter if that's turned off or errors out.
4. Pack everything into ~400-700 token chunks. We never split an atomic
   block up — if one's bigger than our "hard max," we just let the chunk
   run over rather than break it apart.
5. ...except for one case we can't avoid: Gemini won't accept anything over
   ~2048 tokens in a single embed call (max_embedding_input_tokens). If a
   block is genuinely bigger than that, there's no choice — we force-split
   it and mark those pieces with `forced_split` in the metadata so you can
   tell them apart from a normal chunk later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cricllm.block_parser import Segment, split_into_segments
from cricllm.config import BreakpointThresholdType
from cricllm.hashing import sha256_of_text
from cricllm.logging_config import get_logger
from cricllm.markdown_structure import split_by_headers
from cricllm.token_utils import count_tokens

logger = get_logger("semantic_chunker")


def chunk_markdown(
    text: str,
    source: str,
    *,
    min_tokens: int = 400,
    max_tokens: int = 700,
    hard_max_tokens: int = 1400,
    max_embedding_input_tokens: int = 2048,
    use_semantic_chunking: bool = True,
    embeddings: Optional[Embeddings] = None,
    semantic_breakpoint_threshold_type: BreakpointThresholdType = "percentile",
) -> list[Document]:
    """Turn a whole Markdown doc into a list of chunks ready to embed."""
    sections = split_by_headers(text)
    documents: list[Document] = []
    chunk_index = 0

    for section in sections:
        segments = split_into_segments(section.content)
        segments = _expand_oversized_prose(
            segments,
            max_tokens=max_tokens,
            use_semantic_chunking=use_semantic_chunking,
            embeddings=embeddings,
            breakpoint_threshold_type=semantic_breakpoint_threshold_type,
        )
        packed = _pack_segments(
            segments,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            hard_max_tokens=hard_max_tokens,
            max_embedding_input_tokens=max_embedding_input_tokens,
        )

        for chunk in packed:
            if not chunk.content.strip():
                continue
            metadata = {
                "source": source,
                "header_path": section.header_path,
                **section.headers,
                "block_types": sorted(chunk.block_types),
                "chunk_index": chunk_index,
                "token_count": count_tokens(chunk.content),
                "chunk_hash": sha256_of_text(chunk.content),
                "forced_split": chunk.forced_split,
            }
            documents.append(Document(page_content=chunk.content, metadata=metadata))
            chunk_index += 1

    return documents


def _expand_oversized_prose(
    segments: list[Segment],
    *,
    max_tokens: int,
    use_semantic_chunking: bool,
    embeddings: Optional[Embeddings],
    breakpoint_threshold_type: BreakpointThresholdType,
) -> list[Segment]:
    """If a prose segment is bigger than our max on its own, split it further."""
    expanded: list[Segment] = []
    for seg in segments:
        if seg.type != "prose" or count_tokens(seg.content) <= max_tokens:
            expanded.append(seg)
            continue
        sub_texts = _split_prose(
            seg.content,
            max_tokens=max_tokens,
            use_semantic_chunking=use_semantic_chunking,
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
        )
        expanded.extend(Segment("prose", sub) for sub in sub_texts if sub.strip())
    return expanded


def _split_prose(
    text: str,
    *,
    max_tokens: int,
    use_semantic_chunking: bool,
    embeddings: Optional[Embeddings],
    breakpoint_threshold_type: BreakpointThresholdType,
) -> list[str]:
    """Split a prose block — try to split on meaning first, fall back to brute force."""
    if use_semantic_chunking and embeddings is not None:
        try:
            from langchain_experimental.text_splitter import SemanticChunker

            chunker = SemanticChunker(
                embeddings, breakpoint_threshold_type=breakpoint_threshold_type
            )
            return chunker.split_text(text)
        except Exception:
            logger.warning(
                "Semantic chunking failed for a %d-token prose block; falling back to "
                "recursive character splitting",
                count_tokens(text),
                exc_info=True,
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=min(50, max_tokens // 10),
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


@dataclass
class _PackedChunk:
    """One finished chunk, plus a note on how it ended up this way."""

    content: str
    block_types: set[str]
    forced_split: bool = False


def _pack_segments(
    segments: list[Segment],
    *,
    min_tokens: int,
    max_tokens: int,
    hard_max_tokens: int,
    max_embedding_input_tokens: int,
) -> list[_PackedChunk]:
    """Greedily glue segments together into chunks, without ever cutting one in half."""
    packed: list[_PackedChunk] = []
    current_parts: list[str] = []
    current_types: set[str] = set()
    current_tokens = 0

    def flush() -> None:
        nonlocal current_parts, current_types, current_tokens
        if current_parts:
            packed.append(_PackedChunk("".join(current_parts), set(current_types)))
        current_parts = []
        current_types = set()
        current_tokens = 0

    for seg in segments:
        seg_tokens = count_tokens(seg.content)

        if seg_tokens > hard_max_tokens:
            flush()
            if seg_tokens > max_embedding_input_tokens:
                logger.warning(
                    "%s segment of %d tokens exceeds the embedding API's input limit (%d); "
                    "force-splitting it, which breaks the 'never split atomic blocks' rule "
                    "but is the only way to make it embeddable at all",
                    seg.type,
                    seg_tokens,
                    max_embedding_input_tokens,
                )
                for piece in _force_split_oversized(seg.content, max_embedding_input_tokens):
                    packed.append(_PackedChunk(piece, {seg.type}, forced_split=True))
            else:
                logger.warning(
                    "%s segment of %d tokens exceeds hard max %d; keeping it whole rather "
                    "than splitting it",
                    seg.type,
                    seg_tokens,
                    hard_max_tokens,
                )
                packed.append(_PackedChunk(seg.content, {seg.type}))
            continue

        if current_parts and current_tokens + seg_tokens > max_tokens and current_tokens >= min_tokens:
            flush()
        elif current_parts and current_tokens + seg_tokens > hard_max_tokens:
            flush()

        current_parts.append(seg.content)
        current_types.add(seg.type)
        current_tokens += seg_tokens

    flush()
    return packed


def _force_split_oversized(text: str, limit_tokens: int) -> list[str]:
    """Last resort: this segment is too big to ever embed whole, so cut it up.

    We leave a bit of headroom below ``limit_tokens`` since our token count
    is just an estimate and won't match Gemini's tokenizer exactly.
    """
    safe_limit = max(1, int(limit_tokens * 0.9))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=safe_limit,
        chunk_overlap=0,
        length_function=count_tokens,
        separators=["\n\n", "\n", ", ", " ", ""],
    )
    return [piece for piece in splitter.split_text(text) if piece.strip()]
