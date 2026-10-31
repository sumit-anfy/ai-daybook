"""The weekly findability run (librarian/findability.py, 2026-10-14) — the
scheduling, the flags audit_corpus raises, and the filing discipline, all
model-free and corpus-free. The run itself is measured on a corpus-bearing
machine through tests/retrieval_report.py."""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian import findability as fz                     # noqa: E402

UTC = dt.timezone.utc


def result(at, s8, h8, ranks=None):
    ranks = ranks or {"okf_a": {"stranger": 3, "short": 9}, "okf_b": {"stranger": None, "short": 1}}
    return {"at": at, "live": 551, "semantic": True, "sections": "ready",
            "kinds": {"stranger": {"r1": 1, "in8": s8, "in15": 30, "of": 40},
                      "short": {"r1": 1, "in8": h8, "in15": 30, "of": 40}},
            "ranks": ranks, "groups": {"type:note": [3, 4]},
            "unreachable": {"stranger": ["okf_b"], "short": []}}


@pytest.fixture
def marker(tmp_path, monkeypatch):
    p = tmp_path / "findability.json"
    monkeypatch.setenv("OKF_FINDABILITY", str(p))
    return p


def test_due_on_first_run_then_weekly_at_monday_0030_utc():
    assert fz.due({}) is True, "never run: the deploy establishes its own baseline"
    m = {"last": {"at": "2026-10-14T08:00:00Z"}}                    # a Wednesday
    assert not fz.due(m, dt.datetime(2026, 10, 19, 0, 29, tzinfo=UTC))
    assert fz.due(m, dt.datetime(2026, 10, 19, 0, 31, tzinfo=UTC))   # Monday 00:31
    m2 = {"last": {"at": "2026-10-19T00:31:00Z"}}
    assert not fz.due(m2, dt.datetime(2026, 10, 25, 23, 0, tzinfo=UTC))


def test_render_names_ids_and_counts_never_a_query():
    bank_queries = ["how do i find my old notes again", "lost notes"]
    block = fz.render(result("2026-10-21T00:31:00Z", 20, 18),
                      result("2026-10-14T08:00:00Z", 22, 18))
    assert "okf_b" in block and "| stranger | 1 | 20 (-2) | 30 |" in block
    assert not any(q in block for q in bank_queries)


def test_movers_cross_the_8_and_15_lines_only():
    prev = result("x", 0, 0, {"n": {"stranger": 3, "short": 9}})
    cur = result("y", 0, 0, {"n": {"stranger": 5, "short": None}})
    assert fz.movers(prev, cur) == [("n", "short", 9, None)]
    assert fz.movers(None, cur) == []


def test_flag_none_when_never_run_or_healthy(marker):
    assert fz.compute_flag() is None, "no marker on the Mac: nothing to say"
    now = time.time()
    at = dt.datetime.fromtimestamp(now - 3600, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    marker.write_text(json.dumps({"last": result(at, 22, 20), "prev": result(at, 22, 20)}))
    assert fz.compute_flag(now) is None


def test_flag_on_regression_failure_staleness_and_an_unreadable_marker(marker):
    now = time.time()
    at = dt.datetime.fromtimestamp(now - 3600, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    prev = result(at, 22, 20, {"okf_a": {"stranger": 3, "short": 9}})
    cur = result(at, 18, 20, {"okf_a": {"stranger": None, "short": 9}})
    marker.write_text(json.dumps({"last": cur, "prev": prev, "log_id": "okf_log"}))
    f = fz.compute_flag(now)
    assert f["label"] == "Findability regressed"
    assert "stranger 22→18 in 8" in f["desc"] and "okf_a stranger" in f["desc"]

    marker.write_text(json.dumps({"last": cur, "error": "MemoryError: x", "error_at": at}))
    assert fz.compute_flag(now)["label"] == "Findability run failed"

    old = dt.datetime.fromtimestamp(now - 9 * 86400, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    marker.write_text(json.dumps({"last": result(old, 22, 20)}))
    assert fz.compute_flag(now)["label"] == "Findability run stopped"

    marker.write_text("{not json")
    assert fz.compute_flag(now)["label"] == "Findability run failed", \
        "a verdict that cannot be verified is a failure, not a pass"


def test_tick_postpones_while_sections_embed_then_runs(marker, monkeypatch):
    calls = []
    monkeypatch.setattr(fz.embed, "chunk_status", lambda: {"state": "embedding"})
    monkeypatch.setattr(fz, "run", lambda *a, **k: calls.append("run") or result(
        "2026-10-14T08:00:00Z", 22, 20))
    monkeypatch.setattr(fz, "record", lambda root, nodes, cur: {"log_id": "okf_log"})
    load = lambda: ([], {})                                          # noqa: E731
    assert fz.tick(load, Path("/x"), log=lambda m: None).startswith("postponed")
    assert calls == []
    m = json.loads(marker.read_text())
    m["postponed_since"] = time.time() - (fz.MAX_POSTPONE_HOURS + 1) * 3600
    marker.write_text(json.dumps(m))
    assert fz.tick(load, Path("/x"), log=lambda m: None) == "ran", \
        "a postponement is bounded: a layer stuck embedding never stops the run"


def test_tick_records_a_failure_for_the_audit(marker, monkeypatch):
    monkeypatch.setattr(fz.embed, "chunk_status", lambda: {"state": "ready"})
    def boom(*a, **k):
        raise RuntimeError("corpus unreadable")
    monkeypatch.setattr(fz, "run", boom)
    assert fz.tick(lambda: ([], {}), Path("/x"), log=lambda m: None) == "failed"
    assert fz.compute_flag()["label"] == "Findability run failed"


def test_the_bank_the_service_reads_is_the_one_in_the_repo():
    assert fz.bank_path().is_file() and len(fz.load_bank()) == 20


def test_audit_flags_live_nodes_without_asks():
    """The guard for the two-week silent drop: every live node carries asks."""
    from librarian import audit

    class N:
        def __init__(self, id, status="active", asks=None):
            self.id = id
            self.fm = {"id": id, "title": id, "type": "note", "status": status,
                       "project": "global-os", "asks": asks}
    nodes = [N("a", asks=["one phrasing"]), N("b"), N("c", status="parked"),
             N("d", status="retired")]
    flags = audit.compute_health(nodes, {n.id: n for n in nodes})["flags"]
    f = [f for f in flags if f["key"] == "asks"]
    assert f and {x["id"] for x in f[0]["nodes"]} == {"b", "c"}, \
        "live nodes only — a retired node is out of the pool and needs none"
    nodes[1].fm["asks"], nodes[2].fm["asks"] = ["x"], ["y"]
    assert not [f for f in audit.compute_health(nodes, {n.id: n for n in nodes})["flags"]
                if f["key"] == "asks"]


def test_a_bank_that_matches_no_live_node_fails_loudly(marker, monkeypatch, tmp_path):
    """A bank written for another corpus measures nothing here. "0 of 0" in the
    log would read as a run that happened (public audit, 2026-11-01)."""
    monkeypatch.setattr(fz.embed, "chunk_status", lambda: {"state": "ready"})
    monkeypatch.setattr(fz, "load_bank",
                        lambda: [{"node": "okf_elsewhere", "stranger": "a b c", "short": "a b"}])
    assert fz.tick(lambda: ([], {}), tmp_path, log=lambda m: None) == "failed"
    flag = fz.compute_flag()
    assert flag["label"] == "Findability run failed" and "nothing was measured" in flag["desc"]


def test_the_log_files_under_the_configured_home(marker, monkeypatch, tmp_path):
    """A corpus without `global-os` names its own home for the log."""
    seen = {}
    monkeypatch.setattr(fz.store, "create_node",
                        lambda root, *a, **k: seen.update(k) or {"id": "okf_x"})
    monkeypatch.setenv("OKF_FINDABILITY_PROJECT", "workspace")
    fz.record(tmp_path, [], result("2026-11-02T00:31:00Z", 20, 18))
    assert seen["project"] == "workspace"


def test_the_bank_path_follows_the_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OKF_FINDABILITY_BANK", str(tmp_path / "bank.yaml"))
    assert fz.bank_path() == tmp_path / "bank.yaml"
