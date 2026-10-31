"""Fixtures for the wedge detector (librarian.audit.compute_sync).

Built 2026-08-01 after the wedge it exists to catch: 7 commits stranded on the
VPS for five hours behind a push that could never succeed, found only by hand
(the change-the-input retry pattern, the repo-split session).

The negatives matter as much as the positive here. A flag that fires during
normal Mac work, or on a repo with no upstream, is a flag nobody reads
(the fail-loudly rule §3) — so those are pinned, not assumed.

Run: python3 -m pytest tests/test_audit_sync.py
"""
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian.audit import compute_sync  # noqa: E402


def _git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"git {args} failed: {r.stderr}")
    return r


def _commit(repo, name, body="x"):
    (Path(repo) / name).write_text(body)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", f"okf: {name}")


@pytest.fixture
def repos(tmp_path):
    """A bare remote plus two clones — the real Mac/VPS topology in miniature."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "master", str(remote)],
                   capture_output=True, check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(remote), str(seed)], capture_output=True)
    _git(seed, "config", "user.email", "t@t.local")
    _git(seed, "config", "user.name", "t")
    _commit(seed, "base.md")
    _git(seed, "push", "-u", "origin", "master")

    clones = []
    for name in ("vps", "mac"):
        c = tmp_path / name
        subprocess.run(["git", "clone", str(remote), str(c)], capture_output=True)
        _git(c, "config", "user.email", "t@t.local")
        _git(c, "config", "user.name", "t")
        clones.append(c)
    return (*clones, remote)


@pytest.fixture
def autopush(monkeypatch):
    """Stand on the box that is contractually supposed to push every write."""
    monkeypatch.setenv("OKF_GIT_PUSH", "1")


# ── the positive: the 2026-08-01 wedge, reproduced ────────────────────────────
def test_diverged_box_is_flagged(repos, autopush):
    """The real shape: both boxes commit onto one parent, the other pushes first."""
    vps, mac, _ = repos
    _commit(vps, "node_a.md")                    # VPS write, push will be rejected
    _commit(mac, "librarian_change.md")
    _git(mac, "push", "origin", "master")
    _git(vps, "fetch", "origin")                 # what a rejected push leaves behind

    flag = compute_sync(vps)
    assert flag is not None, "a diverged box with unpushed writes must flag"
    assert flag["key"] == "sync"
    assert flag["count"] == 1
    assert flag["diverged"] is True
    assert flag["behind"] == 1
    assert "strand" in flag["desc"]
    assert any("node_a" in c for c in flag["commits"]), flag["commits"]


def test_unpushed_but_not_diverged_is_still_flagged(repos, autopush):
    """A failed push with a quiet remote stalls just as silently."""
    vps, _, _ = repos
    _commit(vps, "node_a.md")
    flag = compute_sync(vps)
    assert flag is not None and flag["count"] == 1
    assert flag["diverged"] is False


def test_count_tracks_the_backlog(repos, autopush):
    vps, _, _ = repos
    for i in range(7):                            # the actual 2026-08-01 depth
        _commit(vps, f"node_{i}.md")
    flag = compute_sync(vps)
    assert flag["count"] == 7
    assert len(flag["commits"]) == 5, "the digest gets a sample, not a firehose"


# ── the negatives: every way this could cry wolf ──────────────────────────────
def test_clean_box_is_silent(repos, autopush):
    vps, _, _ = repos
    _commit(vps, "node_a.md")
    _git(vps, "push", "origin", "master")
    assert compute_sync(vps) is None


def test_behind_only_is_not_a_wedge(repos, autopush):
    """Being behind costs no durability — flagging it would be noise."""
    vps, mac, _ = repos
    _commit(mac, "other.md")
    _git(mac, "push", "origin", "master")
    _git(vps, "fetch", "origin")
    assert compute_sync(vps) is None


def test_silent_where_autopush_is_off(repos, monkeypatch):
    """The Mac. Local commits before a push are normal work, not a wedge."""
    vps, _, _ = repos
    monkeypatch.delenv("OKF_GIT_PUSH", raising=False)
    _commit(vps, "node_a.md")
    assert compute_sync(vps) is None


def test_no_upstream_is_absent_not_wedged(tmp_path, autopush):
    repo = tmp_path / "solo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "master", str(repo)], capture_output=True)
    _git(repo, "config", "user.email", "t@t.local")
    _git(repo, "config", "user.name", "t")
    _commit(repo, "node_a.md")
    assert compute_sync(repo) is None


def test_not_a_repo_does_not_raise(tmp_path, autopush):
    """An audit that dies because git is unusual is worse than a quiet one."""
    assert compute_sync(tmp_path) is None


def test_missing_path_does_not_raise(tmp_path, autopush):
    assert compute_sync(tmp_path / "nope") is None


# ── wiring: the flag actually reaches the audit payload, and reads first ──────
def test_wedge_sorts_above_tidiness_flags(repos, autopush, monkeypatch):
    """Durability outranks tidiness — orphans must not push the wedge down."""
    from librarian import audit as audit_mod

    vps, mac, _ = repos
    _commit(vps, "node_a.md")
    _commit(mac, "librarian_change.md")
    _git(mac, "push", "origin", "master")
    _git(vps, "fetch", "origin")

    class FakeNode:
        def __init__(self, nid):
            self.id = nid
            self.fm = {"title": f"orphan {nid}", "type": "note",
                       "status": "active", "project": "p", "relates_to": []}

    nodes = [FakeNode("okf_a"), FakeNode("okf_b")]
    by_id = {n.id: n for n in nodes}
    health = audit_mod.compute_health(nodes, by_id, root=vps)

    keys = [f["key"] for f in health["flags"]]
    assert "sync" in keys, keys
    assert keys[0] == "sync", f"the wedge must read first, got {keys}"
    assert "orphans" in keys, "the other flags must still compute"
    assert health["summary"]["flags"] == len(health["flags"])


# ── a QUEUED push is not a wedge (2026-08-02) ─────────────────────────────────
# The background pusher opened a ~1s window after every write where HEAD is ahead
# of the remote for a healthy reason. compute_sync fired there, announcing "the
# last push did not land" about a push not yet attempted — an in-flight state
# reported as a broken one, and the cries-wolf failure of the fail-loudly rule.
def test_queued_push_does_not_trigger_the_wedge_flag(repos, autopush, monkeypatch):
    from librarian import store
    import queue as _q
    vps, _, _ = repos
    monkeypatch.setenv("OKF_GIT_PUSH_ASYNC", "1")
    monkeypatch.setenv("OKF_GIT_COMMIT", "1")
    store._push_started = False
    store._push_q = _q.Queue()

    (vps / "node_a.md").write_text("x")
    out = store._git_commit(vps, [Path("node_a.md")], "okf: create node_a")
    assert out["push"] == "queued", "precondition: the push must be deferred"

    assert compute_sync(vps) is None, \
        "a queued push must NOT read as a wedge — it is in flight, not stranded"

    for _ in range(200):
        if store._push_q.empty() and store._push_q.unfinished_tasks == 0:
            break
        time.sleep(0.05)
    assert compute_sync(vps) is None, "and still silent once it lands"


def test_a_real_wedge_still_fires_once_the_queue_is_empty(repos, autopush, monkeypatch):
    """Suppression must not swallow the thing the detector exists for."""
    from librarian import store
    import queue as _q
    vps, _, _ = repos
    monkeypatch.delenv("OKF_GIT_PUSH_ASYNC", raising=False)
    store._push_started = False
    store._push_q = _q.Queue()
    assert not store.push_pending(), "precondition: nothing in flight"

    _commit(vps, "stranded.md")          # committed, never pushed
    flag = compute_sync(vps)
    assert flag is not None and flag["count"] == 1, \
        "with no push pending, an unpushed commit is a genuine wedge"


# ── deploy health (2026-08-08, built on the owner's ruling) ────────────────────────
# The probe itself is shell on the VPS; what is testable everywhere is the
# CONTRACT of the flag: absent marker is silence, a valid marker is a loud
# self-describing flag, and an UNREADABLE marker is still a flag — a verdict
# that cannot be verified is a failure, never a quiet pass (the fail-loudly rule).

from librarian.audit import compute_deploy_gate, compute_deploy_health, compute_health  # noqa: E402


def test_deploy_health_absent_is_silent(monkeypatch, tmp_path):
    monkeypatch.setenv("OKF_DEPLOY_HEALTH", str(tmp_path / "nope.json"))
    assert compute_deploy_health() is None, \
        "no marker = healthy box (and every Mac) — a flag here is one nobody reads"


def test_deploy_health_marker_is_a_flag(monkeypatch, tmp_path):
    import json as _json
    import time as _time
    m = tmp_path / "deploy_health.json"
    m.write_text(_json.dumps({"when": int(_time.time()) - 300,
                              "revision": "8d2f5cc",
                              "failed": ["okf-web (http 000)"]}))
    monkeypatch.setenv("OKF_DEPLOY_HEALTH", str(m))
    f = compute_deploy_health()
    assert f and f["key"] == "deploy"
    assert f["failed"] == ["okf-web (http 000)"]
    assert "8d2f5cc" in f["desc"], "the desc must name the revision that broke"
    assert "5 min ago" in f["desc"], "…and how stale the verdict is"


def test_deploy_health_unreadable_marker_still_flags(monkeypatch, tmp_path):
    m = tmp_path / "deploy_health.json"
    m.write_text("{not json")
    monkeypatch.setenv("OKF_DEPLOY_HEALTH", str(m))
    f = compute_deploy_health()
    assert f and f["key"] == "deploy"
    assert "could not be read" in f["desc"], \
        "unreadable is a FAILURE verdict, not silence (the fail-loudly rule)"


def test_deploy_health_reaches_compute_health_flags(monkeypatch, tmp_path):
    """The wiring: the flag must surface through compute_health — the shared
    engine both audit_corpus and the web health panel read — not only when
    someone calls the helper directly. tmp_path is not a git repo and has no
    purge file, so the deploy flag is the only one a bare corpus can raise."""
    import json as _json
    m = tmp_path / "deploy_health.json"
    m.write_text(_json.dumps({"when": 0, "revision": "abc1234",
                              "failed": ["okf (no HTTP answer)"]}))
    monkeypatch.setenv("OKF_DEPLOY_HEALTH", str(m))
    monkeypatch.setenv("OKF_DEPLOY_GATE", str(tmp_path / "no_gate.json"))
    out = compute_health([], {}, root=tmp_path)
    assert [f["key"] for f in out["flags"]] == ["deploy"]


# ── the deploy TEST GATE marker (2026-09-07) ──────────────────────────────────
# okf-deploy.sh refuses a revision whose suite is red and leaves this marker.
# It is NOT the probe marker: the services never restarted, so they answer,
# and the probe's self-clear must not erase the fact that master has moved on
# and the box has not.

def test_deploy_gate_absent_is_silent(monkeypatch, tmp_path):
    monkeypatch.setenv("OKF_DEPLOY_GATE", str(tmp_path / "nope.json"))
    assert compute_deploy_gate() is None


def test_deploy_gate_marker_names_both_revisions(monkeypatch, tmp_path):
    import json as _json
    import time as _time
    m = tmp_path / "deploy_gate.json"
    m.write_text(_json.dumps({"when": int(_time.time()) - 600, "revision": "bbf22b3",
                              "running": "1fa2449",
                              "summary": "2 failed, 270 passed in 41.0s"}))
    monkeypatch.setenv("OKF_DEPLOY_GATE", str(m))
    f = compute_deploy_gate()
    assert f and f["key"] == "deploy_gate"
    assert "bbf22b3" in f["desc"] and "1fa2449" in f["desc"], \
        "must say what was refused AND what is still running"
    assert "2 failed" in f["desc"], "the pytest summary is the one line that says why"
    assert "10 min ago" in f["desc"]


def test_deploy_gate_unreadable_marker_still_flags(monkeypatch, tmp_path):
    m = tmp_path / "deploy_gate.json"
    m.write_text("{not json")
    monkeypatch.setenv("OKF_DEPLOY_GATE", str(m))
    f = compute_deploy_gate()
    assert f and f["key"] == "deploy_gate" and "could not be read" in f["desc"]


def test_deploy_gate_reaches_compute_health_below_a_dead_service(monkeypatch, tmp_path):
    """Both markers at once: a service that does not answer outranks code that
    merely did not ship, and both must surface."""
    import json as _json
    g = tmp_path / "deploy_gate.json"
    g.write_text(_json.dumps({"when": 0, "revision": "b", "running": "a", "summary": ""}))
    h = tmp_path / "deploy_health.json"
    h.write_text(_json.dumps({"when": 0, "revision": "a", "failed": ["okf (no HTTP answer)"]}))
    monkeypatch.setenv("OKF_DEPLOY_GATE", str(g))
    monkeypatch.setenv("OKF_DEPLOY_HEALTH", str(h))
    out = compute_health([], {}, root=tmp_path)
    assert [f["key"] for f in out["flags"]] == ["deploy", "deploy_gate"]
