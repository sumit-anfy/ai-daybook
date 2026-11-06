"""The bullet summary (2026-10-22) — the current state of a node, over a long
form that only ever grows.

THE DESIGN (2026-10-22). A node is three layers: metadata (title, tags,
asks), a SUMMARY that is rewritten whenever the node changes and covers every
point the node currently makes, and the long-form body, which keeps the full
history and is never overwritten. Fetching reads the summary first; the body
is opened for detail.

THE FORMAT. `summary` stays a text field, written as one "- " line per bullet:
  - bullet 1 says in one sentence what the node is: the one-line description
  - bullets 2..MAX_BULLETS each state one current point and end with
    " → § <heading>", the section of the body that holds its detail — the
    index. "(opening)" names the text before the first heading; a heading that
    occurs more than once takes " #N" (its Nth occurrence) or its full path
  - a point that was changed or overruled says so inside its bullet:
    "(changed 2026-08-25: was Claude Inbox)" or "(overruled 2026-10-15: …)";
    "undated" stands in for a date the node does not give
A summary in any other shape is legacy prose: still valid, never checked here.

WHY IT IS MEASURED. Written for all 555 live nodes on a corpus copy and
measured before adoption: a question about a passage deep in a long node found
it in the default pack of 8 for 23 of 40, against 13 — but only with each
bullet matched on its own (embed._chunk_plan); folded into the one node
vector, the bullets moved it to 16. 199 of the 555 needed more than 8 bullets:
those nodes hold more than one idea, and the warning says so — except a
session, which records a whole sitting (SPLIT_EXEMPT).
Measured on the author's private corpus; not reproducible from the example corpus.
"""
from __future__ import annotations

import re

try:
    import sections
except ImportError:
    from librarian import sections

MAX_BULLETS = 8
MAX_WORDS = 25
# Types that may need more than MAX_BULLETS without holding more than one idea
# (2026-10-25). A session records a whole sitting — what changed, what
# was measured, what went wrong, what is open — and its rulings already live in
# the nodes it cites, so splitting one would cut a sitting in half. The other
# checks still apply.
SPLIT_EXEMPT = ("session",)

_POINTER = re.compile(r"\s*→\s*§\s*(.+?)\s*$")
_NTH = re.compile(r"^(.*?)\s+#(\d+)$")
_MARKER = re.compile(r"\((changed|overruled)\b([^)]*)\)", re.IGNORECASE)
_MARKER_OK = re.compile(r"^ (\d{4}-\d{2}-\d{2}|undated): \S")


def bullets(summary) -> list[str] | None:
    """The bullets, or None when the summary is legacy prose (or empty)."""
    lines = [ln.strip() for ln in str(summary or "").splitlines() if ln.strip()]
    if not lines or not all(ln.startswith("- ") for ln in lines):
        return None
    return [ln[2:].strip() for ln in lines]


def split(bullet: str) -> tuple[str, str | None]:
    """(the point, its section pointer or None)."""
    m = _POINTER.search(bullet)
    return (bullet[:m.start()].rstrip(), m.group(1)) if m else (bullet.strip(), None)


def search_texts(summary) -> list[str]:
    """Each bullet's point without its pointer — what gets its own vector."""
    return [t for t in (split(b)[0] for b in bullets(summary) or []) if t]


def first_line(summary) -> str:
    """The one-line description: bullet 1, or legacy prose unchanged."""
    bs = bullets(summary)
    return bs[0] if bs else str(summary or "")


def _resolves(pointer: str, heads: list[tuple[str, str]]) -> str | None:
    """None when the pointer names exactly one section, else why not."""
    if pointer == "(opening)":
        return None
    m = _NTH.match(pointer)
    name, nth = (m.group(1), int(m.group(2))) if m else (pointer, None)
    hits = [k for k, (path, full) in enumerate(heads)
            if name in (path, full) or (path and name == path.split(" › ")[-1])]
    if not hits:
        return f"points to no heading in the body: '{pointer}'"
    if nth is not None:
        return None if 1 <= nth <= len(hits) else \
            f"'{pointer}': the heading occurs {len(hits)} time(s)"
    if len(hits) > 1:
        return (f"'{pointer}' names {len(hits)} sections; add ' #N' for the Nth "
                f"or use the full heading path")
    return None


def check(summary, body, node_type=None) -> list[str]:
    """Format warnings for a bullet summary ([] for legacy prose). Warnings,
    never refusals: a writer may have a reason, and the receipt says why."""
    bs = bullets(summary)
    if bs is None:
        return []
    out = []
    if len(bs) > MAX_BULLETS and node_type not in SPLIT_EXEMPT:
        out.append(f"summary has {len(bs)} bullets (max {MAX_BULLETS}) — a node whose "
                   f"current state needs more holds more than one idea; consider "
                   f"splitting it into linked nodes")
    heads = sections.headings(body)
    for k, b in enumerate(bs, 1):
        point, pointer = split(b)
        words = len(point.split())
        if words > MAX_WORDS:
            out.append(f"summary bullet {k} is {words} words (max {MAX_WORDS})")
        if k > 1 and pointer is None:
            out.append(f"summary bullet {k} has no section pointer (end it with "
                       f"' → § <heading>', or '(opening)')")
        elif pointer is not None:
            why = _resolves(pointer, heads)
            if why:
                out.append(f"summary bullet {k} {why}")
        for m in _MARKER.finditer(point):
            if not _MARKER_OK.match(m.group(2)):
                out.append(f"summary bullet {k}: write a change as '({m.group(1).lower()} "
                           f"YYYY-MM-DD: …)', or 'undated' for the date")
    return out
