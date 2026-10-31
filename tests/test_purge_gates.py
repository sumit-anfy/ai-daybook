"""Fixtures for the delete gates (librarian.audit.purge_blockers, 2026-08-02).

Deletion became agent-callable on the owner's ruling, replacing a human-approval step
that in practice never ran. The gates are the ENTIRE safety story now, so each one
is pinned by a case that must be refused. Git recovery is real — a node deleted in
July was restored months later through a repo split that rewrote every hash — but
recoverable is not NOTICED, which is why the gates matter more than the undo.

The asymmetry to keep in mind while reading these: a false "blocked" costs one
retire call; a false "safe" silently removes knowledge. Every gate fails closed.

Run: python3 -m pytest tests/test_purge_gates.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian.audit import purge_blockers  # noqa: E402


class N:
    def __init__(self, nid, **fm):
        self.id = nid
        self.fm = {"title": nid, "type": "note", "status": "retired",
                   "canonical": False, "project": "global-os", **fm}


def _corpus(*nodes):
    return list(nodes), {n.id: n for n in nodes}


# ── the one case that must be ALLOWED ─────────────────────────────────────────
def test_retired_orphan_is_deletable():
    nodes, by_id = _corpus(N("okf_junk"), N("okf_other", status="active"))
    assert purge_blockers("okf_junk", nodes, by_id) == []


# ── one test per gate, each must REFUSE ───────────────────────────────────────
def test_active_node_is_refused_and_told_to_retire_first():
    nodes, by_id = _corpus(N("okf_live", status="active"))
    b = purge_blockers("okf_live", nodes, by_id)
    assert b, "an active node must never be deletable"
    assert "retire it first" in " ".join(b), "the refusal must say what to do"


def test_canonical_node_is_refused():
    nodes, by_id = _corpus(N("okf_canon", canonical=True))
    assert any("canonical" in x for x in purge_blockers("okf_canon", nodes, by_id))


def test_node_with_live_outbound_edge_is_refused():
    nodes, by_id = _corpus(N("okf_a", relates_to=["okf_b"]), N("okf_b"))
    assert any("points at" in x for x in purge_blockers("okf_a", nodes, by_id))


def test_node_with_inbound_edge_is_refused():
    """The hard one — nothing on the node itself shows this."""
    nodes, by_id = _corpus(N("okf_target"), N("okf_src", relates_to=["okf_target"]))
    b = purge_blockers("okf_target", nodes, by_id)
    assert any("link TO it" in x for x in b), b


def test_supersession_target_is_refused():
    nodes, by_id = _corpus(N("okf_old", superseded_by="okf_new"),
                           N("okf_new", supersedes="okf_old"))
    assert any("supersession" in x for x in purge_blockers("okf_old", nodes, by_id))


def test_superseding_node_is_also_refused():
    """Both ends of the chain: deleting the successor strands the predecessor."""
    nodes, by_id = _corpus(N("okf_old", superseded_by="okf_new"),
                           N("okf_new", supersedes="okf_old"))
    assert any("supersession" in x for x in purge_blockers("okf_new", nodes, by_id))


def test_unknown_id_is_refused():
    nodes, by_id = _corpus(N("okf_a"))
    assert purge_blockers("okf_nope", nodes, by_id) == ["no node with id 'okf_nope'"]


# ── gates are ANDed: several failures are all reported, not just the first ────
def test_every_failing_gate_is_reported():
    """One at a time would mean N retries to learn why something cannot go."""
    nodes, by_id = _corpus(N("okf_bad", status="active", canonical=True,
                             relates_to=["okf_b"]), N("okf_b"))
    b = purge_blockers("okf_bad", nodes, by_id)
    assert len(b) >= 3, f"expected every blocker, got {b}"


def test_a_dangling_outbound_ref_does_not_block():
    """relates_to pointing at an id that no longer exists is not a live edge —
    blocking on it would make already-deleted neighbours permanently un-cleanable."""
    nodes, by_id = _corpus(N("okf_a", relates_to=["okf_gone"]))
    assert purge_blockers("okf_a", nodes, by_id) == []
