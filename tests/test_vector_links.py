"""The vector-link trial (2026-10-15): a second suggestion channel for pairs the
word-based suggester can never propose — related in meaning, sharing no tags and
almost no words. Model-free: the vectors are fabricated."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian import audit, embed                          # noqa: E402

np = pytest.importorskip("numpy")


class N:
    def __init__(self, id, title, type="note", tags=(), summary=""):
        self.id = id
        self.fm = {"id": id, "title": title, "type": type, "status": "active",
                   "project": "global-os", "tags": list(tags), "summary": summary}


def corpus():
    nodes = [N("a", "selling puts as a way into a stock"),
             N("b", "lending shares for passive yield"),
             N("c", "kitchen garden watering schedule"),
             N("d", "orchid repotting season", tags=["plants"]),
             N("e", "basil propagation from cuttings", tags=["plants"])]
    vec = {"a": [1, 0, 0, 0], "b": [0.9, 0.1, 0, 0], "c": [0, 0, 1, 0],
           "d": [0, 0, 0.9, 0.1], "e": [0, 0, 0.8, 0.2]}
    return nodes, {k: np.array(v, dtype="float32") for k, v in vec.items()}


def test_only_word_invisible_unlinked_undismissed_pairs_one_per_node():
    nodes, vec = corpus()
    out = audit.compute_vector_suggestions(nodes, set(), set(), k=3, vectors=vec)
    pairs = [frozenset((o["a"]["id"], o["b"]["id"])) for o in out]
    assert frozenset(("d", "e")) not in pairs, "a shared specific tag: the word channel's job"
    assert frozenset(("a", "b")) in pairs
    ids = [i for p in pairs for i in p]
    assert len(ids) == len(set(ids)), "at most one suggestion per node"
    assert all(o["source"] == "vector" and "trial" in o["reason"] for o in out)
    linked = audit.compute_vector_suggestions(nodes, {frozenset(("a", "b"))}, set(),
                                              k=3, vectors=vec)
    assert frozenset(("a", "b")) not in [frozenset((o["a"]["id"], o["b"]["id"])) for o in linked]
    gone = audit.compute_vector_suggestions(nodes, set(), {frozenset(("a", "b"))},
                                            k=3, vectors=vec)
    assert frozenset(("a", "b")) not in [frozenset((o["a"]["id"], o["b"]["id"])) for o in gone]


def test_only_a_warmed_service_computes_it(monkeypatch):
    nodes, _ = corpus()
    monkeypatch.setattr(embed, "_STATE", {})
    called = []
    monkeypatch.setattr(embed, "node_vectors", lambda live: called.append(1) or {})
    assert audit.compute_vector_suggestions(nodes, set(), set()) == []
    assert called == [], "a test or script never loads the model for the trial"


def test_rulings_on_word_invisible_pairs_score_the_trial(tmp_path):
    nodes, _ = corpus()
    assert audit.pair_is_word_invisible(nodes, "a", "b")
    assert not audit.pair_is_word_invisible(nodes, "d", "e")
    assert audit.vector_trial_line(tmp_path).startswith("no rulings yet")
    audit.record_vector_trial(tmp_path, "a", "b", "accept")
    audit.record_vector_trial(tmp_path, "a", "c", "dismiss")
    assert audit.vector_trial_line(tmp_path) == "1 accepted, 1 dismissed of 2 ruled (50%)"
