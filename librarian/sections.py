"""Sections — a node body cut along its own headings (2026-10-13).

WHY. The idea taken from PageIndex (VectifyAI's "vectorless" RAG): a long
document already carries its own table of contents, and retrieval should
navigate it rather than read the opening and stop. PageIndex has an LLM walk
an LLM-built tree at query time; OKF needs neither — the markdown headings ARE
the tree, and scoring a section is deterministic.

Measured on the live corpus the day this was built (550 live nodes): median
body 5,008 chars, p90 13,721, max 142,957; 483 nodes are longer than the
2,400-char pack excerpt and 335 longer than the 4,000 chars the node embedding
reads — 53% of all body text was beyond the embedding, and an excerpt showed
the opening of a node however deep the match sat.

WHAT A CHUNK IS. A run of one section's text, at most CHUNK_CHARS long, cut
at paragraph boundaries where it can be, carrying the heading path above it
("Decision › v2") and its exact [start, end) offsets in the body, so an excerpt
slices the body itself and never a re-rendering of it. Headings inside fenced
code blocks are text, not structure.

Deterministic and model-free: the same body always yields the same chunks.
"""
from __future__ import annotations

import re

# ~300 bge tokens: inside the encoder's 512-token window with room for the
# title and heading path that ride along, and small enough that one chunk is
# one idea rather than half a session.
CHUNK_CHARS = 1200

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_FENCE = re.compile(r"^[ \t]*(```|~~~)")


def _sections(body: str) -> list[tuple[str, str, int, int]]:
    """(display path, full path, start, end) per heading-delimited section, in
    order. The text before the first heading is a section with empty paths."""
    out, stack = [], []          # stack: [(level, heading, is the opening H1)]
    start, path, full, fence, pos, first = 0, "", "", None, 0, True
    for line in body.splitlines(keepends=True):
        f = _FENCE.match(line)
        if f:
            fence = None if fence == f.group(1) else (fence or f.group(1))
        h = None if fence or f else _HEADING.match(line.rstrip("\n"))
        if h:
            if pos > start:
                out.append((path, full, start, pos))
            level = len(h.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, h.group(2).strip(), first and level == 1))
            first = False
            # an opening H1 is the node's title restated: the DISPLAY path
            # leaves it out; the full path keeps it, because the encoder was
            # measured with it (dropping it cost two benchmark top-1s)
            path = " › ".join(t for _, t, title in stack if not title)
            full = " › ".join(t for _, t, _ in stack)
            start = pos
        pos += len(line)
    if pos > start:
        out.append((path, full, start, pos))
    return out


def _cut(body: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split [start, end) into spans of at most CHUNK_CHARS, preferring a
    paragraph break, then a line break, then a hard cut."""
    spans = []
    while end - start > CHUNK_CHARS:
        window = body[start:start + CHUNK_CHARS]
        cut = window.rfind("\n\n") + 2            # past the blank line
        if cut < CHUNK_CHARS // 3:
            cut = window.rfind("\n") + 1          # past the line break
        if cut < CHUNK_CHARS // 3:
            cut = CHUNK_CHARS
        spans.append((start, start + cut))
        start += cut
    spans.append((start, end))
    return spans


def chunks(body: str) -> list[dict]:
    """Every chunk of `body`, in document order, covering it exactly:
    {"i", "path", "full", "start", "end"} — `path` for display, `full` with an
    opening H1 kept, for the encoder. Empty body → []."""
    body = body or ""
    out = []
    for path, full, s, e in _sections(body):
        for a, b in _cut(body, s, e):
            out.append({"i": len(out), "path": path, "full": full, "start": a, "end": b})
    return out


def headings(body: str) -> list[tuple[str, str]]:
    """(display path, full path) of every section in document order; the text
    before the first heading is ("", "")."""
    return [(path, full) for path, full, _, _ in _sections(body or "")]


def embed_text(title: str, chunk: dict, body: str) -> str:
    """What a chunk MEANS for the encoder: the node title and the heading path
    first (the context a bare chunk loses — Anthropic's "contextual retrieval"
    in its cheapest form), then the chunk's own text."""
    head = " › ".join(p for p in (str(title or "").strip(), chunk["full"]) if p)
    return head + "\n" + body[chunk["start"]:chunk["end"]]


def substantive(chunk: dict, body: str, floor: int = 80) -> bool:
    """Worth embedding: more than a heading line and a few words."""
    text = body[chunk["start"]:chunk["end"]]
    return len(re.sub(r"^#{1,6}[ \t].*$|\s+", "", text, flags=re.M)) >= floor
