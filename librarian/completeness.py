"""OKF completeness check — the unfiled-ruling detector.

`validate` tests **validity**: are the nodes that exist well-formed. Nothing tested
**completeness**: did everything that should have been recorded, get recorded. A
2026-07-27 audit found 16 ratified rulings reachable only inside session bodies —
written under `## Decisions locked` as prose, correctly classified, never promoted.
Session nodes are long and keyword-dense, so a buried ruling both hides *and*
outranks the decision nodes that should carry it.

The rule (the completeness-check spec §2):

> In a node where `type == session`, every bullet under `## Decisions locked` or
> `## Principles surfaced` must contain at least one OKF node ID.

A bullet with no ID is an **unfiled ruling**.

Deliberately blunt (§5): this cannot tell a bullet stating a rule from narrative
scaffolding, and will over-trigger. That is accepted, not a defect — having a model
judge which bullets are rulings reintroduces the very judgment call that failed, and
makes the check non-deterministic. Keep it mechanical; use the override for genuine
exceptions.

Pure functions over (frontmatter, body) — no I/O, no corpus load — so the write path
(which already holds the body) and the audit path share one implementation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Match the ID in ANY form — the corpus uses all three: [[okf_...]], `okf_...`, bare.
# Match the id pattern only; do not require wiki-link syntax (§2).
_ID_RE = re.compile(r"okf_\d{8}_\d{6}_[0-9a-f]{4}")

# Sections whose bullets must carry an ID. Matched on leading phrase, case- and
# whitespace-insensitively, at any heading level, so "## Decisions Locked"
# passes. `Rulings` is an accepted synonym (§3).
#
# `transferable lessons` / `lessons` were added 2026-07-27 when the owner widened the
# check to notes: the bakery case notes each end with a lessons section stating
# generalisable rules, and 69 such bullets cited nothing. Carrying an ID does NOT
# mean each needs a new node — where the rule already lives in the behaviour-plan
# playbook, citing that playbook is the fix, which makes the case→rule relationship
# explicit instead of implied by prose.
_WATCHED = ("decisions locked", "principles surfaced", "rulings",
            "transferable lessons", "lessons")

# Node types the check applies to. v1 was sessions only; notes joined 2026-07-27 (a
# ruling was found buried in a case note, invisible to a session-scoped check);
# pattern and playbook joined 2026-07-29.
#
# The trigger is a WATCHED SECTION, not the type — a node writing "## Decisions locked"
# is making a pointer list whatever its type is. Measured before widening: 0 of 23
# patterns and 0 of 7 playbooks use one today, so this fires on nothing and exists to
# close the forgotten-type hole rather than to work a backlog.
# Measured on the author's private corpus; not reproducible from the example corpus.
#
# `decision` stays OUT deliberately, and it is the only unscanned type that DOES use
# watched sections (3 nodes: a rubric note, the rulings-ownership decision, the review-queue note). A decision node is where a ruling is
# FILED — its "Rulings" section is its own content, and requiring it to cite another
# node would be circular. Same reasoning would exclude pattern if a pattern ever stated
# its rule under one of these headings; patterns state theirs under their own.
CHECKED_TYPES = {"session", "note", "pattern", "playbook"}

# The STRAY scan (`scan_stray_rulings`) keeps the narrower scope, and the split is the
# whole point: the two scanners ask different questions.
#
#   check_completeness  — "is this POINTER LIST complete?"  fires only inside a watched
#                          section, so a pattern that has one is making a pointer list.
#   scan_stray_rulings  — "is a rule hiding OUTSIDE one?"   fires on a self-labelled
#                          marker (`**Rule.**`, `Invariant:`) ANYWHERE in the body.
#
# Widening the stray scan to patterns was tried on 2026-07-29 and immediately produced 7
# findings, every one of them wrong: a `pattern` node writing "**Rule.** …" outside a
# rulings section is STATING ITS OWN RULE, which is the entire purpose of the type. The
# same holds for playbooks. A marker means "a rule lives here" — in a session that is a
# leak, in a pattern it is the content.
STRAY_CHECKED_TYPES = {"session", "note"}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
# A bullet opens with -, * or an ordinal (1. / 1)) as its first non-space char.
_BULLET_RE = re.compile(r"^(\s*)(?:[-*]|\d+[.)])\s+(.*)$")


@dataclass
class Unfiled:
    """One bullet under a watched section that carries no node ID."""
    section: str
    line: int          # 1-indexed, relative to the body text passed in
    text: str          # first 80 chars, per the spec's report format

    def __str__(self) -> str:
        return f"{self.section} · line {self.line} · {self.text}"


# A line the AUTHOR self-labelled as a rule — "Ruling:", "**Invariant:**", "Rule:".
# The marker must OPEN the line or bullet (optionally bolded), so the word appearing
# mid-prose ("...a ruling we made earlier") does not match. No judgement is involved:
# the check reads a label the author wrote, it does not decide what is rule-shaped.
_SELF_LABEL_RE = re.compile(
    r"^\s*(?:[-*]|\d+[.)])?\s*\*{0,2}"
    r"(?:THE KEY RULING|Ruling|Standing rule|Invariant|Rule)\b\s*\*{0,2}\s*[:.(—-]", re.I)


def find_stray_rulings(body: str) -> list[Unfiled]:
    """Self-labelled rulings sitting OUTSIDE any watched section, carrying no node id.

    Closes part of the section-scoping gap: `find_unfiled_rulings` only ever looks
    inside a rulings section, so a rule stated in a node's narrative — an "Internal
    notes" block, a design discussion — is invisible to it however plainly it is a rule.

    Deliberately narrow. Two wider designs were measured against the live corpus and
    rejected:
    - flagging every bullet with a bold lead-in (the house style for stating a rule)
      produced **375** uncited hits, nearly all list items and labels. Unusable.
    - matching the word "ruling" anywhere produced 15, at roughly one-third precision.
    Requiring the marker to OPEN the line gives 4 hits at ~75% precision, which is a
    warning volume worth reading.
    Measured on the author's private corpus; not reproducible from the example corpus.

    **Known limit, by construction:** this catches rules the author LABELLED. An
    unlabelled rule in narrative prose — house style is a bold lead-in, which is
    indistinguishable from a bold label — is not deterministically detectable, and
    belongs to the Layer-4 content audit (the content-audit decision), which reads
    bodies semantically. Do not widen this to close that; the measurement above is why.
    """
    out: list[Unfiled] = []
    watched: str | None = None
    prev_blank = True          # a wrapped continuation line is never a new statement
    for i, line in enumerate(body.splitlines(), 1):
        h = _HEADING_RE.match(line)
        if h:
            watched = _watched_section(h.group(2))
            prev_blank = True
            continue
        stripped = line.strip()
        starts_statement = bool(_BULLET_RE.match(line)) or (prev_blank and stripped)
        # An indented wrap of the previous bullet can begin with "ruling — ..." purely
        # because that is where the text broke. Only consider lines that START something.
        if not (watched or _ID_RE.search(line)) and starts_statement \
                and _SELF_LABEL_RE.match(line):
            out.append(Unfiled("stray ruling", i, stripped[:80]))
        prev_blank = not stripped
    return out


def _watched_section(title: str) -> str | None:
    """Return the canonical section name if this heading is watched, else None."""
    t = re.sub(r"\s+", " ", title).strip().lower().lstrip("#").strip()
    for name in _WATCHED:
        if t.startswith(name):
            return name
    return None


def find_unfiled_rulings(body: str) -> list[Unfiled]:
    """Bullets under a watched section that contain no okf_ id.

    Bullet parsing (§4):
    - continuation lines belong to their bullet, so an ID on line 3 of a 3-line
      bullet counts;
    - sub-bullets evaluate WITH their parent — a filed parent with three
      explanatory sub-points is one filed ruling, not three violations;
    - a section ends at the next heading of equal or shallower level, or EOF.
    """
    lines = (body or "").splitlines()
    out: list[Unfiled] = []

    section: str | None = None       # canonical name of the watched section we're in
    section_level = 0
    # the currently-open TOP-LEVEL bullet: [start_line, indent, text_parts, has_id]
    open_bullet: list | None = None

    def close_bullet():
        nonlocal open_bullet
        if open_bullet and not open_bullet[3]:
            start, _indent, parts, _ = open_bullet
            text = re.sub(r"\s+", " ", " ".join(parts)).strip()
            out.append(Unfiled(section=section or "", line=start, text=text[:80]))
        open_bullet = None

    for i, raw in enumerate(lines, start=1):
        h = _HEADING_RE.match(raw)
        if h:
            level = len(h.group(1))
            # a heading of equal/shallower level ends the current section
            if section and level <= section_level:
                close_bullet()
                section = None
            name = _watched_section(h.group(2))
            if name:
                close_bullet()
                section, section_level = name, level
            continue

        if not section:
            continue

        b = _BULLET_RE.match(raw)
        if b:
            indent = len(b.group(1).expandtabs(4))
            if open_bullet is not None and indent > open_bullet[1]:
                # sub-bullet: folds into its parent — an ID here files the parent
                open_bullet[2].append(b.group(2))
                open_bullet[3] = open_bullet[3] or bool(_ID_RE.search(raw))
            else:
                close_bullet()   # equal/shallower indent starts a new top-level bullet
                open_bullet = [i, indent, [b.group(2)], bool(_ID_RE.search(raw))]
        elif open_bullet is not None:
            if raw.strip():
                # continuation line — belongs to the open bullet
                open_bullet[2].append(raw.strip())
                open_bullet[3] = open_bullet[3] or bool(_ID_RE.search(raw))
            else:
                pass  # blank lines don't terminate a bullet (sub-lists are spaced)

    close_bullet()
    return out


def check_completeness(fm: dict, body: str) -> list[Unfiled]:
    """Unfiled rulings for a node. Empty for any type outside CHECKED_TYPES, and for
    nodes with no watched section — most produce no rulings, and that absence is
    correct (§3), never a violation."""
    node_type = (fm or {}).get("type")
    # isinstance guard first: a malformed `type: []` must be a validator finding,
    # never an unhashable-membership crash that kills the whole run
    if not isinstance(node_type, str) or node_type not in CHECKED_TYPES:
        return []
    return find_unfiled_rulings(body)


def scan_corpus(root, nodes) -> dict:
    """{node_id: [Unfiled]} for every session node with unfiled rulings.

    `load_corpus` keeps frontmatter and drops bodies, so the audit path re-reads the
    session files (a few dozen, only where `type == session`). Legacy-flagged nodes
    ARE included — on this path they are the backlog, and hiding them would defeat
    the point (§7). Unreadable files are skipped, never raised: the audit is a
    read-only report and must not fail the corpus.
    """
    from pathlib import Path

    out = {}
    for n in nodes:
        node_type = n.fm.get("type")
        if not isinstance(node_type, str) or node_type not in CHECKED_TYPES:
            continue
        try:
            text = (Path(root) / n.path).read_text(encoding="utf-8")
        except OSError:
            continue
        body, offset = _split_body(text)
        found = find_unfiled_rulings(body)
        if found:
            # report FILE-relative lines — a body-relative number sends the reader
            # to the wrong line in the editor
            out[n.id] = [Unfiled(f.section, f.line + offset, f.text) for f in found]
    return out


def scan_stray_rulings(root, nodes) -> dict:
    """{node_id: [Unfiled]} for self-labelled rulings outside any rulings section.

    Same read pattern and same never-fail contract as `scan_corpus`. Kept a separate
    function, and a separate finding, because it answers a different question: not
    "is this rulings section complete?" but "is there a rule hiding somewhere else?"
    """
    from pathlib import Path

    out = {}
    for n in nodes:
        node_type = n.fm.get("type")
        if not isinstance(node_type, str) or node_type not in STRAY_CHECKED_TYPES:
            continue
        try:
            text = (Path(root) / n.path).read_text(encoding="utf-8")
        except OSError:
            continue
        body, offset = _split_body(text)
        found = find_stray_rulings(body)
        if found:
            out[n.id] = [Unfiled(f.section, f.line + offset, f.text) for f in found]
    return out


def _split_body(text: str):
    """(body, line_offset) — body after the YAML frontmatter, and how many lines
    preceded it, so reported line numbers point into the real file."""
    m = re.match(r"^---\s*\n.*?\n---\s*(?:\n|$)", text, re.DOTALL)
    if not m:
        return text, 0
    return text[m.end():], text[:m.end()].count("\n")


def is_legacy(fm: dict) -> bool:
    """Sessions written before this rule carry `unfiled_rulings_legacy: true`.

    The WRITE path skips them (otherwise editing an old session becomes impossible);
    the AUDIT path still reports them — they are the promotion backlog and must stay
    visible (§7). Clearing the flag is a manual act and is that node's completion
    signal.
    """
    return bool((fm or {}).get("unfiled_rulings_legacy"))
