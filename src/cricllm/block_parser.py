"""Walks through a section of Markdown and figures out what's "atomic" vs. prose.

Atomic stuff — fenced code blocks, tables, lists, curl one-liners — never
gets split up later on, no matter how the chunker packs things together.
Everything else is just prose, and that's fair game for further splitting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SegmentType = Literal["code", "table", "list", "curl", "prose"]

_LIST_ITEM_RE = re.compile(r"^\s*([-*+]\s+|\d+[.)]\s+)")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_CURL_START_RE = re.compile(r"^\s*curl\b")


@dataclass
class Segment:
    """One piece of a section — either an atomic block or a prose paragraph."""

    type: SegmentType
    content: str


def split_into_segments(text: str) -> list[Segment]:
    """Walk ``text`` line by line and break it into atomic/prose segments, in order."""
    lines = text.splitlines(keepends=True)
    segments: list[Segment] = []
    prose_buffer: list[str] = []
    i = 0
    n = len(lines)

    def flush_prose() -> None:
        if prose_buffer:
            joined = "".join(prose_buffer)
            if joined.strip():
                segments.append(Segment("prose", joined))
            prose_buffer.clear()

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush_prose()
            fence_marker = stripped[:3]
            block = [line]
            i += 1
            while i < n and not lines[i].strip().startswith(fence_marker):
                block.append(lines[i])
                i += 1
            if i < n:
                block.append(lines[i])  # closing fence
                i += 1
            segments.append(Segment("code", "".join(block)))
            continue

        if _is_table_start(lines, i):
            flush_prose()
            block = []
            while i < n and _is_table_line(lines[i]):
                block.append(lines[i])
                i += 1
            segments.append(Segment("table", "".join(block)))
            continue

        if _CURL_START_RE.match(line):
            flush_prose()
            block = [line]
            i += 1
            while i < n and block[-1].rstrip("\n").rstrip().endswith("\\"):
                block.append(lines[i])
                i += 1
            segments.append(Segment("curl", "".join(block)))
            continue

        if _LIST_ITEM_RE.match(line):
            flush_prose()
            block = []
            while i < n and (_LIST_ITEM_RE.match(lines[i]) or _is_list_continuation(lines, i)):
                block.append(lines[i])
                i += 1
            segments.append(Segment("list", "".join(block)))
            continue

        prose_buffer.append(line)
        i += 1

    flush_prose()
    return segments


def _is_table_start(lines: list[str], i: int) -> bool:
    if "|" not in lines[i]:
        return False
    if i + 1 >= len(lines):
        return False
    return bool(_TABLE_SEPARATOR_RE.match(lines[i + 1].strip()))


def _is_table_line(line: str) -> bool:
    return "|" in line and line.strip() != ""


def _is_list_continuation(lines: list[str], i: int) -> bool:
    line = lines[i]
    if line.strip() == "":
        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j < len(lines) and (_LIST_ITEM_RE.match(lines[j]) or lines[j].startswith((" ", "\t"))):
            return True
        return False
    return line.startswith(("  ", "\t"))
