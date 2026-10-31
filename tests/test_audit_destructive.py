"""Fixtures for the destructive-write detector (librarian.audit).

Built 2026-08-10, after seven nodes were found with their bodies destroyed and
nobody noticed for twelve days across 309 commits (the destructive-write pattern,
The emptied-nodes session).

WHY THIS SHAPE AND NOT PEAK-VS-HEAD. Comparing each node's largest-ever size
against its current size was the first instrument tried, and it declared the
corpus clean while two JSON-encoded health nodes were still broken — because a
later write had grown them back past their old size. Asking which COMMITS removed
far more than they added found both classes, because it looks at the write rather
than at the surviving file. That is the difference these fixtures pin.

The negatives matter as much as the positive. This runs on every audit_corpus
call, so a flag that fires on ordinary editing is one nobody reads
(the fail-loudly rule §3) — the silence cases are pinned, not assumed.

Run: python3 -m pytest tests/test_audit_destructive.py
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian.audit import compute_destructive_writes  # noqa: E402

LONG = "\n".join(f"line {i} of a node with real content in it" for i in range(60))
SHORT_APPEND_BLOCK = "## APPEND\n\n---\n\n## Additions — later\n\nOne short block."


def _git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(f"git {args} failed: {r.stderr}")
    return r


def _commit(repo, name, body, date="2026-08-15T12:00:00", msg=None):
    (Path(repo) / name).write_text(body)
    _git(repo, "add", "-A")
    env_args = ["-c", f"user.name=t", "-c", "user.email=t@t.local"]
    subprocess.run(
        ["git", "-C", str(repo), *env_args, "commit", "-m", msg or f"okf: write {name}"],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin",
             "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
             "HOME": str(repo)},
        check=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "corpus"
    r.mkdir()
    subprocess.run(["git", "init", "-b", "master", str(r)], capture_output=True, check=True)
    _git(r, "config", "user.email", "t@t.local")
    _git(r, "config", "user.name", "t")
    _commit(r, "node.md", LONG, date="2026-08-11T09:00:00")
    return r


BASE = "2026-08-01"


# ── the positive: the shape that cost 119,000 characters ─────────────────────
def test_a_body_replaced_by_a_short_block_is_flagged(repo):
    """The exact July 2026 write: a long body replaced by a short append block."""
    _commit(repo, "node.md", SHORT_APPEND_BLOCK, date="2026-08-12T09:00:00",
            msg="okf: update okf_x (relates_to, summary)")
    flag = compute_destructive_writes(repo, since=BASE)
    assert flag is not None, "the whole point of the check"
    assert flag["key"] == "destructive"
    assert flag["count"] == 1
    hit = flag["commits"][0]
    assert hit["file"] == "node.md" and hit["removed"] > hit["added"]


def test_the_flag_reports_a_shape_not_a_verdict(repo):
    """It cannot know whether a rewrite was intended, and must not pretend to."""
    _commit(repo, "node.md", SHORT_APPEND_BLOCK, date="2026-08-12T09:00:00")
    desc = compute_destructive_writes(repo, since=BASE)["desc"]
    assert "SHAPE, not a verdict" in desc
    assert "git show" in desc, "must say how to adjudicate a hit"


def test_it_catches_loss_that_peak_vs_head_would_MISS(repo):
    """The two health nodes: content dropped, then the file grown back bigger.

    A peak-vs-HEAD sweep sees a node larger than it has ever been and says
    nothing. The removal still happened, and this is the instrument that sees it.
    """
    _commit(repo, "node.md", "tiny", date="2026-08-12T09:00:00")
    _commit(repo, "node.md", LONG + "\n" + LONG, date="2026-08-13T09:00:00")
    flag = compute_destructive_writes(repo, since=BASE)
    assert flag is not None and flag["count"] == 1
    assert flag["commits"][0]["date"] == "2026-08-12"


# ── the negatives: it must be SILENT on ordinary work ────────────────────────
def test_growth_is_not_flagged(repo):
    _commit(repo, "node.md", LONG + "\n" + LONG, date="2026-08-12T09:00:00")
    assert compute_destructive_writes(repo, since=BASE) is None


def test_a_small_edit_is_not_flagged(repo):
    """Trimming a few lines is editing, not loss."""
    trimmed = "\n".join(LONG.split("\n")[:-5])
    _commit(repo, "node.md", trimmed, date="2026-08-12T09:00:00")
    assert compute_destructive_writes(repo, since=BASE) is None


def test_a_rewrite_of_similar_length_is_not_flagged(repo):
    """Honest limit, pinned so nobody mistakes this for total coverage: a swap of
    comparable size does not trip the ratio and never will."""
    other = "\n".join(f"entirely different line {i}" for i in range(60))
    _commit(repo, "node.md", other, date="2026-08-12T09:00:00")
    assert compute_destructive_writes(repo, since=BASE) is None


def test_a_clean_repo_says_nothing(repo):
    assert compute_destructive_writes(repo, since=BASE) is None


# ── scoping: the baseline, deletions, and hostile environments ───────────────
def test_commits_before_the_baseline_are_not_reported(repo):
    """Adjudicated history must age out, or the flag becomes furniture."""
    _commit(repo, "node.md", SHORT_APPEND_BLOCK, date="2026-08-12T09:00:00")
    assert compute_destructive_writes(repo, since=BASE) is not None
    assert compute_destructive_writes(repo, since="2026-09-01") is None


def test_a_deleted_file_is_not_reported(repo):
    """delete_node is gated four ways and leaves an obvious commit — including
    deletions here put nine scratch probes in the output and nothing true."""
    _git(repo, "rm", "-q", "node.md")
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "okf: delete node"],
                   capture_output=True, check=True)
    assert compute_destructive_writes(repo, since=BASE) is None


def test_a_directory_that_is_not_a_repo_returns_none(tmp_path):
    """Never raises: an audit that dies because git is unusual is worse than one
    that stays quiet. This runs on the read path."""
    assert compute_destructive_writes(tmp_path, since=BASE) is None


def test_a_nonexistent_path_returns_none():
    assert compute_destructive_writes(Path("/nonexistent/nope"), since=BASE) is None


# ── it must reach audit_corpus, not just exist ───────────────────────────────
def test_the_flag_is_wired_into_the_health_report(repo, monkeypatch):
    """A check nobody calls is the class-3 failure this corpus keeps relearning
    (the redundancy-check handover)."""
    from librarian import audit
    _commit(repo, "node.md", SHORT_APPEND_BLOCK, date="2026-08-12T09:00:00")
    monkeypatch.setattr(audit, "DESTRUCTIVE_BASELINE", BASE)
    monkeypatch.setattr(audit, "compute_sync", lambda *a, **k: None)
    monkeypatch.setattr(audit, "compute_deploy_health", lambda *a, **k: None)
    monkeypatch.setattr(audit, "compute_deploy_gate", lambda *a, **k: None)
    health = audit.compute_health([], {}, root=repo)
    keys = [f["key"] for f in health["flags"]]
    assert "destructive" in keys, f"not surfaced by compute_health: {keys}"


def test_it_outranks_deploy_health_in_the_flag_order(repo, monkeypatch):
    """Ordering is a real decision: a service that is down is visible and will be
    fixed; content already removed is invisible and decays."""
    from librarian import audit
    _commit(repo, "node.md", SHORT_APPEND_BLOCK, date="2026-08-12T09:00:00")
    monkeypatch.setattr(audit, "DESTRUCTIVE_BASELINE", BASE)
    monkeypatch.setattr(audit, "compute_sync", lambda *a, **k: None)
    monkeypatch.setattr(audit, "compute_deploy_health",
                        lambda *a, **k: {"key": "deploy", "label": "d", "desc": "d"})
    monkeypatch.setattr(audit, "compute_deploy_gate", lambda *a, **k: None)
    keys = [f["key"] for f in audit.compute_health([], {}, root=repo)["flags"]]
    assert keys.index("destructive") < keys.index("deploy")
