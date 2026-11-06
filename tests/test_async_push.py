"""Fixtures for the background push (librarian.store, 2026-08-02).

Measured on the author's server; not reproducible here.
Deferring `git push` off the write path took a write from ~965ms to ~165ms — push
was 83% of it. The risk it introduces is not a wider stranding window (a failed
push already stranded the commit) but a LATER-NOTICED one, so the two safeguards
are what make it acceptable and are what these fixtures pin:

  1. SERIALISATION — one worker, so two writes never push concurrently.
  2. STARTUP DRAIN — a restart kills a daemon thread mid-push; boot re-pushes.

The sync path is pinned too: async is opt-in, and the default must still report a
real True/False.

Run: python3 -m pytest tests/test_async_push.py
"""
import queue
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian import store  # noqa: E402


def _git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"git {args}: {r.stderr}")
    return r


@pytest.fixture
def repo(tmp_path):
    """A clone with a bare upstream, so pushes are real."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "master", str(remote)],
                   capture_output=True, check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", str(remote), str(work)], capture_output=True)
    _git(work, "config", "user.email", "t@t.local")
    _git(work, "config", "user.name", "t")
    (work / "base.md").write_text("base\n")
    _git(work, "add", "-A"); _git(work, "commit", "-m", "base")
    _git(work, "push", "-u", "origin", "master")
    return work, remote


@pytest.fixture(autouse=True)
def clean_push_state(monkeypatch):
    """Each test gets a fresh queue and worker.

    The previous worker is a daemon blocked on an orphaned queue; it never runs
    again and dies with the process. Acceptable in tests, and the alternative
    (sharing one queue) would defeat the serialisation fixture.
    """
    store._push_started = False
    store._push_q = queue.Queue()
    monkeypatch.setenv("OKF_GIT_COMMIT", "1")
    monkeypatch.setenv("OKF_GIT_PUSH", "1")
    yield


def _remote_head(remote):
    return _git(remote, "rev-parse", "master").stdout.strip()


def _local_head(work):
    return _git(work, "rev-parse", "HEAD").stdout.strip()


def _wait_drained(timeout=30.0):
    """Wait for the worker to finish everything queued."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if store._push_q.empty() and store._push_q.unfinished_tasks == 0:
            return True
        time.sleep(0.05)
    return False


def _write(work, name, body="x"):
    (work / name).write_text(body)
    return store._git_commit(work, [Path(name)], f"okf: {name}")


# ── the sync path must be unchanged and opt-out-able ──────────────────────────
def test_sync_is_the_default_and_reports_a_real_result(repo, monkeypatch):
    work, remote = repo
    monkeypatch.delenv("OKF_GIT_PUSH_ASYNC", raising=False)
    out = _write(work, "node_a.md")
    assert out["committed"] is True
    assert out["pushed"] is True, "sync must confirm, not defer"
    assert _remote_head(remote) == _local_head(work)


# ── async: returns immediately, lands eventually ──────────────────────────────
def test_async_defers_the_push_but_it_still_lands(repo, monkeypatch):
    work, remote = repo
    monkeypatch.setenv("OKF_GIT_PUSH_ASYNC", "1")
    before = _remote_head(remote)
    out = _write(work, "node_a.md")

    assert out["committed"] is True
    assert out["pushed"] is None, "queued must be None, never False"
    assert _local_head(work) != before, "the commit is immediate"

    assert _wait_drained(), "worker never drained the queue"
    assert _remote_head(remote) == _local_head(work), "deferred push never landed"


def test_queued_is_not_failure(repo, monkeypatch):
    """None must be distinguishable from False, or a caller reads a queued write
    as a lost one — the exact misreading the tri-state exists to prevent."""
    work, _ = repo
    monkeypatch.setenv("OKF_GIT_PUSH_ASYNC", "1")
    out = _write(work, "node_a.md")
    assert out["pushed"] is not False
    assert out["pushed"] is None


# ── SAFEGUARD 1: serialisation ────────────────────────────────────────────────
def test_rapid_writes_all_land_without_racing(repo, monkeypatch):
    work, remote = repo
    monkeypatch.setenv("OKF_GIT_PUSH_ASYNC", "1")
    for i in range(8):                      # faster than a push can complete
        out = _write(work, f"node_{i}.md")
        assert out["committed"] is True
    assert _wait_drained(), "worker never drained under burst"
    assert _remote_head(remote) == _local_head(work)
    listed = _git(remote, "ls-tree", "-r", "--name-only", "master").stdout.split()
    for i in range(8):
        assert f"node_{i}.md" in listed, f"node_{i}.md never reached the remote"


def test_many_writes_start_exactly_one_worker(repo, monkeypatch):
    """Two workers draining one queue would push concurrently and lose one to a
    ref lock — the failure serialisation exists to prevent. Counted as a delta:
    daemon workers from earlier tests linger on orphaned queues."""
    import threading
    work, _ = repo
    monkeypatch.setenv("OKF_GIT_PUSH_ASYNC", "1")
    before = sum(1 for t in threading.enumerate() if t.name == "okf-push")
    for i in range(3):
        _write(work, f"node_{i}.md")
    _wait_drained()
    after = sum(1 for t in threading.enumerate() if t.name == "okf-push")
    assert after - before == 1, "three writes must share ONE worker, not spawn several"


# ── SAFEGUARD 2: startup drain ────────────────────────────────────────────────
def test_startup_drain_pushes_what_a_restart_left_behind(repo, monkeypatch):
    """Simulates the real case: the service committed, was restarted before the
    push ran, and the commit is sitting local-only."""
    work, remote = repo
    monkeypatch.setenv("OKF_GIT_PUSH_ASYNC", "1")

    # commit WITHOUT any push — as if the process died between the two
    (work / "stranded.md").write_text("stranded\n")
    _git(work, "add", "-A"); _git(work, "commit", "-m", "okf: stranded write")
    assert _remote_head(remote) != _local_head(work), "precondition: unpushed"

    store.start_background_push(work)        # what boot does
    assert _wait_drained(), "startup drain never ran"
    assert _remote_head(remote) == _local_head(work), "startup drain did not push"


def test_startup_drain_is_a_noop_when_nothing_is_pending(repo, monkeypatch):
    work, remote = repo
    monkeypatch.setenv("OKF_GIT_PUSH_ASYNC", "1")
    head = _remote_head(remote)
    store.start_background_push(work)
    assert _wait_drained()
    assert _remote_head(remote) == head


def test_startup_drain_does_nothing_when_async_is_off(repo, monkeypatch):
    """A box on the sync path must not start a worker.

    Counted as a DELTA, not an absolute: workers from earlier tests are daemons
    blocked on orphaned queues and still show up in threading.enumerate().
    """
    import threading
    work, _ = repo
    monkeypatch.delenv("OKF_GIT_PUSH_ASYNC", raising=False)
    before = sum(1 for t in threading.enumerate() if t.name == "okf-push")
    store.start_background_push(work)
    assert store._push_started is False
    after = sum(1 for t in threading.enumerate() if t.name == "okf-push")
    assert after == before, "sync path must not start a push worker"


# ── the recovery logic must survive the move into _push_now ───────────────────
def test_background_push_still_recovers_from_a_diverged_remote(repo, monkeypatch, tmp_path):
    work, remote = repo
    monkeypatch.setenv("OKF_GIT_PUSH_ASYNC", "1")

    # a second clone pushes first, so `work` diverges
    other = tmp_path / "other"
    subprocess.run(["git", "clone", str(remote), str(other)], capture_output=True)
    _git(other, "config", "user.email", "t@t.local")
    _git(other, "config", "user.name", "t")
    (other / "theirs.md").write_text("theirs\n")
    _git(other, "add", "-A"); _git(other, "commit", "-m", "theirs")
    _git(other, "push", "origin", "master")

    _write(work, "ours.md")
    assert _wait_drained(), "worker never drained after divergence"

    listed = _git(remote, "ls-tree", "-r", "--name-only", "master").stdout.split()
    assert "ours.md" in listed and "theirs.md" in listed, \
        "background push did not rebase-and-recover; one side was lost"


# ── the payload must explain itself (2026-08-02) ──────────────────────────────
# A bare `pushed: null` caused a chat session to report "unconfirmed push status"
# after two clean writes. It was right to, given what it had: nothing in the result
# said null meant queued. The caller reads the PAYLOAD, not the docstring.
def test_queued_result_says_it_is_normal(repo, monkeypatch):
    work, _ = repo
    monkeypatch.setenv("OKF_GIT_PUSH_ASYNC", "1")
    out = _write(work, "node_a.md")
    assert out["pushed"] is None
    assert out["push"] == "queued"
    note = out.get("push_note", "")
    assert "NORMAL" in note, "a queued push must say it is normal, in the payload"
    assert "not a failure" in note.lower()
    _wait_drained()


def test_confirmed_result_is_quiet(repo, monkeypatch):
    """No note on the happy path — advice nobody needs is noise."""
    work, _ = repo
    monkeypatch.delenv("OKF_GIT_PUSH_ASYNC", raising=False)
    out = _write(work, "node_a.md")
    assert out["pushed"] is True and out["push"] == "confirmed"
    assert "push_note" not in out


def test_failed_result_says_it_is_worth_reporting(repo, monkeypatch, tmp_path):
    """The one case a caller SHOULD surface must say so."""
    work, _ = repo
    monkeypatch.delenv("OKF_GIT_PUSH_ASYNC", raising=False)
    _git(work, "remote", "set-url", "origin", str(tmp_path / "gone.git"))   # unreachable
    out = _write(work, "node_a.md")
    assert out["committed"] is True, "the write itself still succeeds"
    assert out["pushed"] is False and out["push"] == "failed"
    assert "WORTH REPORTING" in out.get("push_note", "")


def test_the_three_states_are_distinguishable(repo, monkeypatch):
    """queued vs failed must never be confusable — that is the whole point."""
    work, _ = repo
    monkeypatch.setenv("OKF_GIT_PUSH_ASYNC", "1")
    queued = _write(work, "node_a.md")
    _wait_drained()
    assert queued["push"] != "failed" and queued["pushed"] is not False


# ── the rebase must not race a writer (2026-09-08, external review) ───────────
# The worker's `pull --rebase` rewrites HEAD and the tree while a create/update
# may be mid load→write→commit under the same tree. It now takes the writer's
# lock — and ONLY the worker does, because the sync caller already holds it.
def test_background_rebase_takes_the_repo_lock(repo, monkeypatch, tmp_path):
    import threading
    work, remote = repo
    monkeypatch.setenv("OKF_GIT_PUSH_ASYNC", "1")
    taken = []
    real = store._repo_lock

    def recording(root, shared=False):
        taken.append((threading.current_thread().name, shared))
        return real(root, shared=shared)
    monkeypatch.setattr(store, "_repo_lock", recording)

    other = tmp_path / "other"
    subprocess.run(["git", "clone", str(remote), str(other)], capture_output=True)
    _git(other, "config", "user.email", "t@t.local")
    _git(other, "config", "user.name", "t")
    (other / "theirs.md").write_text("theirs\n")
    _git(other, "add", "-A"); _git(other, "commit", "-m", "theirs")
    _git(other, "push", "origin", "master")

    _write(work, "ours.md")
    assert _wait_drained(), "worker never drained after divergence"
    exclusive_off_main = [t for t, sh in taken
                          if t != threading.main_thread().name and not sh]
    assert exclusive_off_main, f"the worker rebased without the exclusive lock: {taken}"
    listed = _git(remote, "ls-tree", "-r", "--name-only", "master").stdout.split()
    assert "ours.md" in listed and "theirs.md" in listed


def test_sync_rebase_under_the_writer_lock_does_not_deadlock(repo, monkeypatch, tmp_path):
    """create_node holds the exclusive lock while _git_commit pushes synchronously.
    If _push_now re-took the lock on that path, the rebase branch would block
    forever against its own caller. Run the diverged sync write in a thread with
    a deadline so a regression fails instead of hanging the suite."""
    import threading
    work, remote = repo
    monkeypatch.delenv("OKF_GIT_PUSH_ASYNC", raising=False)

    other = tmp_path / "other"
    subprocess.run(["git", "clone", str(remote), str(other)], capture_output=True)
    _git(other, "config", "user.email", "t@t.local")
    _git(other, "config", "user.name", "t")
    (other / "theirs.md").write_text("theirs\n")
    _git(other, "add", "-A"); _git(other, "commit", "-m", "theirs")
    _git(other, "push", "origin", "master")

    result = {}

    def write_under_lock():
        with store._repo_lock(work):          # exactly what create_node does
            result.update(_write(work, "ours.md"))
    t = threading.Thread(target=write_under_lock, daemon=True)
    t.start(); t.join(timeout=60)
    assert not t.is_alive(), "sync push deadlocked against the writer's own lock"
    assert result.get("pushed") is True and result.get("push") == "confirmed", result
    listed = _git(remote, "ls-tree", "-r", "--name-only", "master").stdout.split()
    assert "ours.md" in listed and "theirs.md" in listed
