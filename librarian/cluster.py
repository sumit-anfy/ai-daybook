"""Layer 2 — the write-time supersession / contradiction check (the five-layer audit decision).

The five-layer audit decision specced this on 2026-07-21, called it *"the highest-leverage place"*, and it went
unbuilt for a week while the content audit (now Layer 4) found ~150 issues — most of
them write-induced drift.

## Why this is not an LLM call inside the tool

The five-layer audit decision rejected baking LLM work into the server: *"the MCP server can't spawn a 13-agent
fan-out from inside a tool call."* That constraint stands, and it does not need to be
broken — **every writer of an OKF node is already an LLM**, because the MCP is only
reachable by one. So the server's job at write time is not to *do* the reasoning; it is
to put the right neighbours and the right question in front of the caller already there.
This module picks the neighbours deterministically. The caller answers.

## The direction that matters

The intuitive check — *"is this new node self-consistent?"* — is worthless: a node is
always consistent with itself. Staleness is **not a property of the node that goes
stale**. When `propose_link` shipped on 2026-07-25 it made SIX other nodes wrong;
nothing about those six changed. The write that created the staleness was the new node.
So the question handed back is deliberately reversed: **"what does this node make FALSE?"**

## How the cluster is picked — and what was measured and REJECTED
Measured on the author's private corpus; not reproducible from the example corpus.

Backtested against ground truth from the 2026-07-28 content audit (now Layer 4): five
real writes and the thirteen nodes they were found to have made false.

| selection signal                        | recall  |
|-----------------------------------------|---------|
| IDF lexical similarity (tags+title+summary) | **46%** — rejected |
| + body-cited node ids                   | no gain — rejected |
| declared edges, outbound only (create)  | **69%** |
| declared edges, out + inbound (update)  | **92%** |

**Lexical similarity is the wrong instrument for selection and was dropped.** The signal
that works is the author's own `relates_to`: you link to what you are engaging with, and
what you are engaging with is what you are about to make stale. The purge-hardening session (which shipped
`delete_node`) has *no tags and zero shared tokens* with the MCP tool-design reference (which rules on tool
count) — the relationship is semantic and no lexical measure sees it.

The lexical scorer survives in a smaller role: **ranking within the declared set** when it
must be capped. Selection, no; ordering, yes.

**Known ceiling: ~69% at create.** The remaining misses are nodes the author never
referenced — which is precisely why they went stale. Layer 2 is a large improvement on
the zero that ran before it, not a guarantee; Layer 4 (the content audit) stays the backstop (the five-layer audit decision).
Measured on the author's private corpus; not reproducible from the example corpus.

## Warn only

Ruled 2026-07-29. This never blocks a write. A supersession check that refuses writes
would be wrong often enough to train the author to override by reflex, and an override
reached for every time is worse than no check. It also matches the five-layer audit decision's standing rule that
**detection is not declaration** — layers 2 and 3 only ever propose.
"""

from __future__ import annotations

import math
from collections import Counter

try:
    from audit import _tags, _tokens
except ImportError:
    from librarian.audit import _tags, _tokens

# Holds full backtest recall (12/13); cap=8 drops it to 85%. Only 11% of active nodes
# have more declared neighbours than this, so it truncates rarely.
# Measured on the author's private corpus; not reproducible from the example corpus.
DEFAULT_CAP = 12

# Fields whose change can make another node false. A pure `relates_to` backlink or an
# `updated_at` touch cannot, so those updates skip the check entirely and stay silent.
CONTENT_FIELDS = {"title", "summary", "tags", "type", "status", "project"}


def declared_neighbours(fm: dict, node_id: str | None, nodes: list, by_id: dict) -> set:
    """Active nodes the author has declared a relationship with, both directions.

    At CREATE there are no inbound edges yet (nothing points at a node that did not
    exist), so this is effectively outbound-only and recall is the lower 69% figure.
    Measured on the author's private corpus; not reproducible from the example corpus.
    """
    out = {r for r in (fm.get("relates_to") or []) if isinstance(r, str)}
    if node_id:
        out |= {n.id for n in nodes if node_id in (n.fm.get("relates_to") or [])}
    return {i for i in out
            if i in by_id and by_id[i].fm.get("status") == "active"} - {node_id}


def _lexical_scores(fm: dict, candidates: list) -> dict:
    """IDF-weighted overlap, used ONLY to order a capped cluster. Mirrors the weighting
    in audit.compute_suggestions so the corpus keeps one notion of 'topically close'."""
    if not candidates:
        return {}

    class _Shim:
        def __init__(self, fm):
            self.fm = fm

    me = _Shim(fm)
    my_tags, my_toks = _tags(me), _tokens(me)
    total = max(len(candidates), 1)
    tag_df, tok_df = Counter(), Counter()
    for n in candidates:
        for t in _tags(n):
            tag_df[t] += 1
        for t in _tokens(n):
            tok_df[t] += 1

    def idf(df):
        return math.log((total + 1) / (df + 0.5))

    scores = {}
    for n in candidates:
        shared_tags, shared_tok = _tags(n) & my_tags, _tokens(n) & my_toks
        scores[n.id] = round(sum(idf(tag_df[t]) for t in shared_tags)
                             + 0.4 * sum(idf(tok_df[t]) for t in shared_tok), 2)
    return scores


def topical_cluster(fm: dict, node_id: str | None, nodes: list, by_id: dict,
                    cap: int = DEFAULT_CAP) -> list[dict]:
    """The nodes this write is most likely to have made false, most-likely first."""
    ids = declared_neighbours(fm, node_id, nodes, by_id)
    if not ids:
        return []
    cands = [by_id[i] for i in ids]
    scores = _lexical_scores(fm, cands)
    cands.sort(key=lambda n: (-scores.get(n.id, 0), str(n.fm.get("updated_at") or "")),
               reverse=False)
    return [{"id": n.id, "title": n.fm.get("title", ""), "type": n.fm.get("type"),
             "summary": n.fm.get("summary") or "", "updated_at": n.fm.get("updated_at")}
            for n in cands[:cap]]


QUESTION = (
    "LAYER 2 — write-time supersession/contradiction check. WARN ONLY: nothing was blocked "
    "and no further call is required to complete the write.\n"
    "\n"
    "Ask this about the node you just wrote, in THIS direction:\n"
    "\n"
    "    What does this node make FALSE?\n"
    "\n"
    "Not 'is it consistent' — it is always consistent with itself. Staleness is caused by "
    "the NEW write and lands on the OLD node. For each neighbour below, decide:\n"
    "  - CONTRADICTS: the neighbour now asserts something untrue. Append a DATED correction "
    "block to the NEIGHBOUR. Never a silent rewrite, never a deletion.\n"
    "  - SUPERSEDES: the neighbour is WHOLLY replaced. Record it — `supersedes` on the new "
    "node, old node flipped to status:superseded. Declared, never inferred.\n"
    "  - FINE: no action, say nothing.\n"
    "\n"
    "Detection is not declaration (the five-layer audit decision): propose, then record only what "
    "you confirm. Neighbours are the author's own declared edges; ~3 in 10 stale nodes are "
    "NOT in this list, so a clean pass here is not proof the corpus is clean."
)


def layer2(fm: dict, node_id: str | None, nodes: list, by_id: dict,
           cap: int = DEFAULT_CAP) -> dict | None:
    """The advisory block returned to the writing agent, or None if there is nothing
    to check (a node with no declared edges)."""
    cluster = topical_cluster(fm, node_id, nodes, by_id, cap)
    if not cluster:
        return None
    return {"check": "layer2_supersession", "blocking": False,
            "question": QUESTION, "cluster": cluster}


def should_check_update(changes: dict) -> bool:
    """An update needs Layer 2 only if it changed something that can make another node
    false. A backlink write or a date touch cannot, and must stay silent."""
    return bool(set(changes or {}) & CONTENT_FIELDS)
