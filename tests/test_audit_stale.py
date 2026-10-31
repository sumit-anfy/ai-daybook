"""Fixtures for the stale-neighbourhood flag and its created_at guard.

The flag means: every node this one points at is retired, so it may have drifted
too. On 2026-08-02 it fired on the body-replace incident — a post-mortem written
that day, whose single link is to the node retired in the incident it documents.
It was the NEWEST node in the corpus and was reported as possibly stale.

The guard: a node written at or after its neighbours were retired is pointing at
them deliberately. Staleness is drift, and drift takes time.

The asymmetry that sets the direction: a missed advisory costs a glance, a false
one trains the reader to skip the whole flag (the fail-loudly rule rule 3).
So the guard errs toward silence — but only for nodes that provably knew.

Run: python3 -m pytest tests/test_audit_stale.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian.audit import compute_health  # noqa: E402


class N:
    def __init__(self, nid, created, updated=None, status="active", links=()):
        self.id = nid
        self.fm = {"title": nid, "type": "note", "status": status,
                   "project": "global-os", "created_at": created,
                   "updated_at": updated or created, "relates_to": list(links)}


def _stale(*nodes):
    by_id = {n.id: n for n in nodes}
    health = compute_health(list(nodes), by_id)
    flag = next((f for f in health["flags"] if f["key"] == "stale"), None)
    return {n["id"] for n in flag["nodes"]} if flag else set()


# ── the real 2026-08-02 false positive ────────────────────────────────────────
def test_a_postmortem_written_after_the_retirement_is_not_stale():
    """The body-replace incident's exact shape: born the same day its only neighbour was retired."""
    subject = N("okf_subject", "2026-07-13", updated="2026-08-02", status="retired")
    postmortem = N("okf_postmortem", "2026-08-02", links=["okf_subject"])
    assert _stale(subject, postmortem) == set()


def test_written_strictly_after_is_not_stale():
    subject = N("okf_subject", "2026-07-13", updated="2026-08-01", status="retired")
    later = N("okf_later", "2026-08-05", links=["okf_subject"])
    assert _stale(subject, later) == set()


# ── the flag must still catch what it exists for ──────────────────────────────
def test_a_node_that_predates_the_retirement_is_still_flagged():
    """The real case: written when the neighbour was live, and never revisited."""
    subject = N("okf_subject", "2026-01-01", updated="2026-06-01", status="retired")
    old = N("okf_old", "2026-02-01", links=["okf_subject"])
    assert _stale(subject, old) == {"okf_old"}


def test_flag_survives_across_several_dead_neighbours():
    a = N("okf_a", "2026-01-01", updated="2026-06-01", status="retired")
    b = N("okf_b", "2026-01-01", updated="2026-07-01", status="superseded")
    old = N("okf_old", "2026-02-01", links=["okf_a", "okf_b"])
    assert _stale(a, b, old) == {"okf_old"}


def test_the_newest_dead_neighbour_sets_the_bar():
    """Knowing about one retirement is not knowing about a later one."""
    a = N("okf_a", "2026-01-01", updated="2026-03-01", status="retired")
    b = N("okf_b", "2026-01-01", updated="2026-09-01", status="retired")
    mid = N("okf_mid", "2026-05-01", links=["okf_a", "okf_b"])
    assert _stale(a, b, mid) == {"okf_mid"}, "b retired after mid was written"


# ── unaffected cases ──────────────────────────────────────────────────────────
def test_one_live_neighbour_means_never_stale():
    dead = N("okf_dead", "2026-01-01", updated="2026-06-01", status="retired")
    live = N("okf_live", "2026-01-01")
    n = N("okf_n", "2026-02-01", links=["okf_dead", "okf_live"])
    assert _stale(dead, live, n) == set()


def test_a_node_with_no_links_is_not_stale():
    """That is orphanhood, a different flag with a different remedy."""
    lone = N("okf_lone", "2026-01-01")
    assert _stale(lone) == set()


def test_missing_dates_do_not_suppress_the_flag():
    """A node with no created_at must NOT be silently skipped — absent data is
    not evidence of deliberate linking."""
    subject = N("okf_subject", "2026-01-01", updated="2026-06-01", status="retired")
    undated = N("okf_undated", "2026-02-01", links=["okf_subject"])
    undated.fm["created_at"] = None
    assert _stale(subject, undated) == {"okf_undated"}
