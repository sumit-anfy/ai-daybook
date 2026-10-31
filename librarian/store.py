"""OKF node store — the Librarian's write path (MCP-agnostic, fully testable).

Every mutation is transactional: the candidate (and any node it supersedes) is
assembled in memory, the whole set is validated against node.schema, and files
are written only if there are zero errors. IDs, defaults, folder placement, and
the supersession side-effect are the Librarian's deterministic responsibility.
"""
from __future__ import annotations

import fcntl
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager, nullcontext
from datetime import date
from pathlib import Path

import yaml

try:  # dual import: `python server.py` (root) or `python -m librarian.store`
    from yamlio import safe_load as _yload, safe_dump as _ydump
    from ids import new_id
    import summary as _summary
    from transitions import (check_birth_status, check_status_transition,
                             check_maturity_transition)
    from validate import (Node, SEVERITY_ERROR, SEVERITY_WARN, load_corpus, load_folder_map,
                          load_registry_projects, run_checks,
                          _first_yaml_block_with_key)
except ImportError:
    from librarian.yamlio import safe_load as _yload, safe_dump as _ydump
    from librarian.ids import new_id
    from librarian import summary as _summary
    from librarian.transitions import (check_birth_status, check_status_transition,
                                       check_maturity_transition)
    from librarian.validate import (Node, SEVERITY_ERROR, SEVERITY_WARN, load_corpus,
                                    load_folder_map, load_registry_projects,
                                    run_checks, _first_yaml_block_with_key)

# Frontmatter emit order (node.schema §2). Unknown keys appended after.
FIELD_ORDER = ["id", "type", "title", "summary", "status", "maturity", "canonical",
               "created_at", "updated_at", "project", "tags", "asks", "source",
               # work_state and superseded_in_part sit beside status on purpose:
               # all three answer "how much of this node still holds?", and a reader
               # scanning the frontmatter should meet them together.
               "work_state", "superseded_in_part",
               "relates_to", "supersedes", "superseded_by", "confidence", "resource",
               "unfiled_rulings_legacy", "allow_dangling_refs"]

_FM_SPLIT = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


class StoreError(ValueError):
    """Raised when a mutation would produce an invalid corpus (write refused)."""


# ── serialization ─────────────────────────────────────────────────────────────
def split_doc(text: str):
    """Return (frontmatter_dict, body_str)."""
    m = _FM_SPLIT.match(text)
    if not m:
        return None, text
    return _yload(m.group(1)), m.group(2)


def render_node(fm: dict, body: str) -> str:
    ordered = {k: fm[k] for k in FIELD_ORDER if k in fm and fm[k] is not None}
    ordered.update({k: v for k, v in fm.items()
                    if k not in FIELD_ORDER and v is not None})
    front = _ydump(ordered, sort_keys=False, allow_unicode=True,
                           default_flow_style=False).rstrip("\n")
    return f"---\n{front}\n---\n\n{body.strip()}\n"


def slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:60] or "node"


def _norm_title(title: str) -> str:
    """Normalized title for near-duplicate comparison (case/punctuation-insensitive)."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


_CODA_KEY_RE = re.compile(r"coda://docs/[^/\s]+/pages/[^\s()]+")


def _coda_key(source):
    """Extract the canonical Coda page URI from a free-text `source`, or None.
    The reliable duplicate signal is the page URI (section-XXXX), NOT the whole
    source string — migrated sources append hand-written descriptions that vary
    ('— original: X' vs '| orig: X') and Coda HTML-escapes '>' to '&gt;', so
    exact-string matching misses real duplicates. Free-text sources
    (e.g. 'session-2026-07-13', which legitimately repeats) yield None and are
    never hard-blocked."""
    if not isinstance(source, str):
        return None
    m = _CODA_KEY_RE.search(source)
    return m.group(0) if m else None


@contextmanager
def _repo_lock(root: Path, shared: bool = False):
    """Repository-level lock. Writers take it exclusive: the load->validate->
    write->commit sequence must not interleave across concurrent HTTP calls
    (FastMCP runs sync tools in a threadpool). Readers take it shared so they
    never observe a half-applied multi-file mutation. flock is advisory but
    every store path goes through here."""
    lock = open(root / ".okf.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _contained(root: Path, rel: Path) -> None:
    """Refuse any node path that resolves outside the corpus root — covers both
    hostile type names and hostile FOLDER_MAP values."""
    if not (root / rel).resolve().is_relative_to(root.resolve()):
        raise StoreError(f"path '{rel}' escapes the corpus root")


def _ensure_clean_tree(root: Path) -> None:
    """When auto-commit is on, refuse to write over an already-dirty corpus:
    dirty .md files mean a previous mutation did not complete (the atomic-write
    window is between os.replace calls). Loud beats silent — recover with
    `git status` / `git restore`, then retry."""
    if os.environ.get("OKF_GIT_COMMIT") != "1":
        return
    r = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "*.md"],
                       capture_output=True, text=True, timeout=30)
    if r.returncode == 0 and r.stdout.strip():
        raise StoreError("corpus has uncommitted .md changes — a previous mutation "
                         "may not have completed; inspect `git status`, recover "
                         "(git restore / commit), then retry:\n" + r.stdout.strip())


# ── internal: validate an in-memory set, refuse on any error ───────────────────
def _assert_valid(nodes, by_id, valid_types, valid_projects=None) -> None:
    findings = run_checks(nodes, by_id, valid_types, valid_projects=valid_projects)
    errs = [f for f in findings if f.severity == SEVERITY_ERROR]
    if errs:
        raise StoreError("write refused — validation failed:\n" +
                         "\n".join(str(e) for e in errs))
    return findings   # so a caller can surface WARNs without a second full pass


# A record's FACTS. Everything else — relates_to, supersedes/superseded_by, status,
# resource, tags, updated_at — is linking machinery and stays editable, because a record
# that cannot gain a backlink is unusable in a linked corpus.
RECORD_FROZEN_FIELDS = ("title", "summary", "created_at")

# The heading `body_append` writes ahead of every appended block. Stamped by the
# TOOL, not left to the caller: this whole defect began with "## APPEND" being a
# convention that lived only in prose, so the one place it now appears is code.
APPEND_HEADING = "## APPEND ({date})"


def _assert_record_immutable(before: dict, after: dict, old_body: str, new_body: str,
                             allow_record_amendment: bool) -> None:
    """A `record` is a dated snapshot; its recorded facts are frozen.

    The type registry states this as the type's entire purpose — *"IMMUTABILITY IS THE WHOLE
    DISCIPLINE: it is never edited to reflect a new state … if you find yourself editing a
    record to keep it true, it should have been a resource pointer"* — and until 2026-08-01
    nothing enforced it. It was pure prose inside a document the validator otherwise reads
    live, which is the sharpest form of the redundancy-check handover's class 3: the
    surrounding document IS enforced, so the prose inherits an authority it does not have.

    What is actually forbidden is narrower than "no edits", and the registry's own wording
    says so — *never edited to reflect a NEW STATE*. So:

    - **frozen**: title, summary, created_at — the claim and its date
    - **append-only**: the body. Existing text may not change; new text may be added.
    - **free**: relates_to, supersedes/superseded_by, status, resource, tags

    Append-only rather than byte-frozen because the corpus's standing remediation rule
    (the content-audit decision) is *append a dated correction, never rewrite* — a record
    must be able to carry one. Measured against the two real records: of the two
    post-creation edits, the pure append passes and the one that REWROTE a provenance line
    is refused, which is the correct bite.

    Override `allow_record_amendment=True` lands in the record, per the house style of
    `allow_unlink` / `allow_unfiled_rulings` — an override belongs in the record, not in
    silence.
    """
    if before.get("type") != "record" and after.get("type") != "record":
        return
    if allow_record_amendment:
        return
    changed = [f for f in RECORD_FROZEN_FIELDS if before.get(f) != after.get(f)]
    if changed:
        raise StoreError(
            f"write refused — a `record` is a dated snapshot and {changed} are frozen. "
            "A record is never edited to reflect a new state: file a NEW record that "
            "`supersedes` this one. Override: allow_record_amendment=true.")
    if not (new_body or "").startswith((old_body or "").rstrip()) and (old_body or "").strip():
        raise StoreError(
            "write refused — a `record`'s body is APPEND-ONLY; this edit changes text that "
            "is already there. Append a dated note instead (the content-audit decision), or "
            "file a new record that supersedes this one. Override: allow_record_amendment=true.")


_GDRIVE_ID_RE = re.compile(r"/d/([A-Za-z0-9_-]{20,})|[?&]id=([A-Za-z0-9_-]{20,})")


def _gdrive_id(uri) -> str | None:
    m = _GDRIVE_ID_RE.search(str(uri or ""))
    return (m.group(1) or m.group(2)) if m else None


def _assert_pointer_stable(before: dict, after: dict, allow_repoint: bool) -> None:
    """A Drive pointer's file ID must not change under an existing node.

    `node.schema` §2 states it: *"update the file in place, never delete-and-reupload —
    a new ID breaks the pointer."* It was prose and nothing checked it, which matters
    more now that `record` nodes are frozen: a record's whole value is that its pointer
    still resolves, and delete-and-reupload silently breaks that while every other
    check stays green.

    The rule is safe to enforce strictly because a Drive file ID **survives being moved
    between folders** — verified 2026-07-28. So reorganising Drive never changes an ID;
    a changed ID means a different file, which is the violation itself.

    Override `allow_repoint=True` for the genuine cases — the original was deleted
    outside OKF and re-created, or the pointer was wrong when first written.
    """
    if allow_repoint:
        return
    old_res, new_res = before.get("resource"), after.get("resource")
    if not isinstance(old_res, dict) or not isinstance(new_res, dict):
        return
    if not str(old_res.get("kind", "")).startswith("gdrive"):
        return
    old_id, new_id = _gdrive_id(old_res.get("uri")), _gdrive_id(new_res.get("uri"))
    if old_id and new_id and old_id != new_id:
        raise StoreError(
            f"write refused — this node's Drive file ID would change ({old_id[:12]}… -> "
            f"{new_id[:12]}…). A file ID survives being MOVED, so a new ID means the file "
            "was deleted and re-uploaded, which breaks every pointer to it (node.schema §2). "
            "Update the file in place. Override: allow_repoint=true.")


def _assert_complete(fm: dict, body, allow_unfiled_rulings: bool) -> None:
    """Refuse a session whose rulings are stated in prose and filed nowhere.

    The preventive half of the completeness spec (the completeness-check spec §6). It
    inverts the writing order on purpose (§10): a session can no longer be written
    before its decision nodes exist, because it needs their IDs to point at.
    Promotion stops being an optional follow-up and becomes a prerequisite — capture
    will feel slower, and that cost IS the control.

    Escapes, both deliberate and visible: `allow_unfiled_rulings` in the call (the
    `allow_unlink` precedent — an override belongs in the record, not in silence),
    and the `unfiled_rulings_legacy` flag on sessions predating the rule.
    """
    try:
        from completeness import check_completeness, is_legacy
    except ImportError:
        from librarian.completeness import check_completeness, is_legacy
    if allow_unfiled_rulings or is_legacy(fm):
        return
    unfiled = check_completeness(fm, body)
    if unfiled:
        listed = "\n".join(f"    - [{u.section}] line {u.line}: {u.text}" for u in unfiled)
        raise StoreError(
            f"write refused — {len(unfiled)} unfiled ruling(s): a bullet under "
            f"'Decisions locked'/'Principles surfaced' states a ruling but points at "
            f"no node.\n{listed}\n"
            "  File each as its own decision/pattern node and cite its id in the "
            "bullet (any form: [[okf_...]], `okf_...`, bare), or pass "
            "allow_unfiled_rulings=true to override deliberately.")


def _prepare_supersession(by_id: dict, successor_id: str, target_id: str, today: str):
    """Return an updated Node for the superseded target (in-memory), or raise."""
    target = by_id.get(target_id)
    if target is None:
        raise StoreError(f"supersedes points to unknown id '{target_id}'")
    check_status_transition(target.fm.get("status"), "superseded")
    tfm = dict(target.fm)
    tfm.update(status="superseded", canonical=False,
               superseded_by=successor_id, updated_at=today)
    return Node(id=target.id, path=target.path, fm=tfm)


def _prepare_unsupersession(by_id: dict, target_id: str, today: str):
    """Restore a target whose successor just dropped/changed its `supersedes`
    pointer: status -> active, clear superseded_by. canonical is NOT auto-restored
    (its pre-supersession value was never recorded) — the human re-asserts it.
    Returns None if the target no longer exists."""
    target = by_id.get(target_id)
    if target is None:
        return None
    tfm = dict(target.fm)
    check_status_transition(tfm.get("status"), "active", allow_unsupersede=True)
    tfm["status"] = "active"
    tfm.pop("superseded_by", None)
    tfm["updated_at"] = today
    return Node(id=target.id, path=target.path, fm=tfm)


# ── background push ────────────────────────────────────────────────────────────
# Measured 2026-08-02: a write costs ~965ms and ~800ms of that is `git push`, which
# is GitHub's SSH auth on a 1-vCPU box (already halved by connection reuse; it will
# not get much better). The caller is made to wait on a network round trip that
# tells it nothing it can act on. Deferring it takes a write to ~165ms.
#
# This does NOT widen the window where a node exists on one machine only: if a push
# fails, the commit was already stranded until the next write or the wedge detector
# (the change-the-input retry pattern). What moves is WHEN a failure is noticed, not whether
# one can happen. Two safeguards below make that acceptable.
_push_q: queue.Queue = queue.Queue()
_push_started = False
_push_inflight = 0
_push_lock = threading.Lock()


def _async_push_enabled() -> bool:
    return os.environ.get("OKF_GIT_PUSH_ASYNC") == "1"


def _push_now(root: Path, *, lock: bool = False) -> bool:
    """Push, recovering from a diverged remote. Never raises. Returns success.

    `lock` — take the repository lock (exclusive) around the REBASE. The push
    itself only talks to the remote; `pull --rebase` rewrites HEAD and the
    working tree, and until 2026-09-08 the background worker did that with no
    lock while a writer could be mid load→validate→write→commit under the
    same tree (the external review's finding). The worker passes lock=True.
    The SYNC caller (`_git_commit` inside create/update/delete) must NOT: it
    already runs under the writer's exclusive lock, and flock on a fresh
    descriptor blocks against the same process — re-taking it would deadlock.
    """
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args],
                                  capture_output=True, text=True, timeout=120)

        r = git("push")
        if r.returncode != 0:
            # A diverged remote is the one push failure that does NOT ride the
            # next push: once the remote holds a commit we lack, every later push
            # is rejected too and every write after it strands on this box.
            # Rebase onto the remote and retry once.
            with (_repo_lock(root) if lock else nullcontext()):
                if git("pull", "--rebase").returncode == 0:
                    r = git("push")
                else:
                    # Never leave a half-finished rebase behind: a dirty tree would
                    # fail the NEXT write too, far from its cause.
                    git("rebase", "--abort")
        if r.returncode == 0:
            return True
        print(f"okf: ERROR git push failed and rebase did not recover it. This box "
              f"is DIVERGED from the remote — writes will keep stranding here until "
              f"it is resolved by hand: {r.stderr.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"okf: ERROR background push raised: {e}", file=sys.stderr)
    return False


def push_pending() -> bool:
    """True while any push is queued or mid-flight.

    The wedge detector must not read a QUEUED write as a stranded one. Between the
    commit and the worker's push there is a window (~1s) in which HEAD is ahead of
    the remote-tracking ref for an entirely healthy reason — and before this existed
    `compute_sync` fired there, announcing "the last push did not land" about a push
    that had not yet been attempted. Same defect as the bare `pushed: null`: an
    IN-FLIGHT state reported as a BROKEN one.
    """
    return _push_inflight > 0 or not _push_q.empty()


def _push_worker(q: queue.Queue):
    """SAFEGUARD 1 — serialisation. One worker, so two writes in quick succession
    never push concurrently and lose one to a ref lock.

    The queue is passed in rather than read from the module global: a worker that
    resolves `_push_q` at call time would follow the global if it were ever
    rebound, so two workers could end up draining one queue. Production only ever
    has one queue, which is exactly why that would go unnoticed.
    """
    global _push_inflight
    while True:
        root = q.get()
        _push_inflight += 1          # set BEFORE the push, cleared after, so the
        try:                         # window is never momentarily "not pending"
            _push_now(root, lock=True)   # a rebase here must not race a writer
        finally:
            _push_inflight -= 1
            q.task_done()


def start_background_push(root: Path) -> None:
    """SAFEGUARD 2 — drain on startup. Call once at server boot.

    A daemon thread dies with the process, so a push in flight when systemd
    restarts the service (which the deploy timer does on every code change) is
    lost. The commit is safe on disk, but unpushed. Enqueuing one push at boot
    clears whatever the previous process left behind — and costs nothing when
    there is nothing to send.

    Deliberately not a SIGTERM handler: intercepting shutdown to flush a queue is
    more invasive than re-pushing at boot, and boot also covers crashes and
    reboots, which no handler would.
    """
    global _push_started
    if not _async_push_enabled():
        return
    with _push_lock:
        if _push_started:
            return
        threading.Thread(target=_push_worker, args=(_push_q,), daemon=True,
                         name="okf-push").start()
        _push_started = True
    _push_q.put(Path(root))


def _git_commit(root: Path, rel_paths, message: str) -> dict:
    """Librarian responsibility (spec §6.1): commit corpus writes to git.
    Gated by OKF_GIT_COMMIT=1 (off by default — local dev commits by hand);
    OKF_GIT_PUSH=1 additionally pushes, rebasing onto the remote and retrying
    once if the push is rejected — a diverged remote never heals itself. The
    node write has already succeeded when this runs, so NOTHING here may raise
    — commit/push/timeout failures WARN to stderr; a git hiccup must never look
    like a failed write. The commit is pathspec-scoped so unrelated staged
    changes are never absorbed.

    Returns {committed, pushed, push[, push_note]}.

    `pushed` is TRI-STATE and `push` is its plain-language twin, because a caller
    that sees only `null` cannot tell "queued" from "went wrong":
      True  / "confirmed" — on the remote
      False / "failed"    — attempted and did not land; this box is the only copy
      None  / "queued"    — handed to the background worker; not yet known
              also "disabled" (OKF_GIT_PUSH off) and "not-attempted" (commit failed)

    `push_note` appears ONLY when there is something to say — it tells the caller
    whether the state is worth surfacing to a human. Silence on the happy path.
    """
    # `push` and `push_note` exist because a bare `pushed: null` MISREPORTS ITSELF.
    # On 2026-08-02 a chat session saw null after two `true`s and flagged "unconfirmed
    # push status" — correctly, given what it had to work with. The caller reads the
    # payload, not this docstring (the MCP tool-design reference applied to a return value
    # rather than a tool name), so the payload has to carry its own meaning.
    out = {"committed": False, "pushed": False, "push": "not-attempted"}
    if os.environ.get("OKF_GIT_COMMIT") != "1":
        out["push"] = "disabled"
        return out
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args],
                                  capture_output=True, text=True, timeout=60)

        paths = [str(p) for p in rel_paths]
        r = git("add", "-A", "--", *paths)
        if r.returncode == 0:
            r = git("commit", "-m", message, "--", *paths)
        if r.returncode != 0:
            print(f"okf: WARN git commit failed: {r.stderr.strip()}", file=sys.stderr)
            return out
        out["committed"] = True
        if os.environ.get("OKF_GIT_PUSH") != "1":
            out["push"] = "disabled"
        elif _async_push_enabled():
            # Hand the network round trip to the worker and return. `pushed`
            # is None — QUEUED, not "failed": the caller must not read a
            # missing True as a lost write.
            start_background_push(root)
            _push_q.put(Path(root))
            out["pushed"] = None
            out["push"] = "queued"
            out["push_note"] = (
                "NORMAL — not a failure and not worth flagging to the user. The node is "
                "committed on the VPS and a background worker pushes it to GitHub within "
                "about a second. Only push='failed' means a push was attempted and did "
                "not land.")
        elif _push_now(root):
            out["pushed"] = True
            out["push"] = "confirmed"
        else:
            out["push"] = "failed"
            out["push_note"] = (
                "WORTH REPORTING — the push was attempted and did not land, so this node "
                "currently exists on one machine only. It is committed, so nothing is "
                "lost, and the next write retries; but say so rather than passing it over.")
    except Exception as e:  # timeout, missing git binary, anything
        print(f"okf: WARN git step failed after successful write: {e}", file=sys.stderr)
    return out


def _atomic_write_all(writes, deletes=()) -> None:
    """Write (Path, content) pairs via temp file + os.replace, then remove the
    `deletes` paths — all as ONE transaction. Temps are staged first (a failure
    during prep touches nothing); each replace is atomic per file (temp lives in
    the target's directory = same filesystem). If any replace OR delete fails
    partway, every already-applied step is ROLLED BACK from in-memory snapshots,
    so multi-file mutations (supersession, type refiling) land all-or-nothing up
    to a rollback-write failure."""
    staged = []
    try:
        for path, content in writes:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            staged.append((Path(tmp), path))
    except Exception:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
        raise

    replaced = []  # (path, prior bytes | None if the file did not exist)
    removed = []   # (path, prior bytes) — deletions that already happened
    try:
        for tmp, path in staged:
            prior = path.read_bytes() if path.exists() else None
            os.replace(tmp, path)
            replaced.append((path, prior))
        for path in deletes:
            if path.exists():
                prior = path.read_bytes()
                os.remove(path)
                removed.append((path, prior))
    except Exception:
        for path, prior in reversed(removed):
            try:
                path.write_bytes(prior)
            except OSError as e:
                print(f"okf: WARN rollback (restore {path}) failed: {e} — "
                      "recover via git", file=sys.stderr)
        for path, prior in reversed(replaced):
            try:
                if prior is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(prior)
            except OSError as e:
                print(f"okf: WARN rollback of {path} failed: {e} — "
                      "recover via git", file=sys.stderr)
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
        raise


# ── public API ────────────────────────────────────────────────────────────────
def read_node(root, node_id: str) -> dict:
    root = Path(root)
    with _repo_lock(root, shared=True):   # never observe a half-applied mutation
        _, by_id, *_ = load_corpus(root)
        node = by_id.get(node_id)
        if node is None:
            raise KeyError(f"no node with id '{node_id}'")
        _, body = split_doc((root / node.path).read_text(encoding="utf-8"))
    return {"id": node_id, "path": str(node.path), "frontmatter": node.fm, "body": body}


def list_nodes(root, node_type: str | None = None, status: str | None = None) -> list[dict]:
    root = Path(root)
    with _repo_lock(root, shared=True):
        nodes, *_ = load_corpus(root)
    out = []
    for n in nodes:
        if node_type and n.fm.get("type") != node_type:
            continue
        if status and n.fm.get("status") != status:
            continue
        out.append({"id": n.id, "type": n.fm.get("type"), "title": n.fm.get("title"),
                    "status": n.fm.get("status"), "source": n.fm.get("source"),
                    "path": str(n.path)})
    return out


def create_node(root, type: str, title: str, body: str = "", **fields) -> dict:
    root = Path(root)
    with _repo_lock(root):
        return _create_node(root, type, title, body, **fields)


def _create_node(root: Path, type: str, title: str, body: str, **fields) -> dict:
    _ensure_clean_tree(root)
    nodes, by_id, _, valid_types, dup = load_corpus(root)
    if dup:
        raise StoreError("corpus has duplicate ids; refusing to write until resolved")
    if not valid_types:  # fail closed: no registry means a broken corpus, not a permissive one
        raise StoreError("type registry missing or unparseable — refusing all writes")
    if type not in valid_types:
        raise StoreError(f"type '{type}' not in vocabulary {sorted(valid_types)}")

    # project (the world axis) is a closed, registry-owned vocabulary — a typo
    # (`bakry`, `Personal`) would silently fragment a world. Fail closed on a
    # missing PROJECT_REGISTRY, hard-reject an unknown project; adding a new world
    # is a one-line edit to PROJECT_REGISTRY, so the registry stays the single
    # point of change (matches how `type` is handled).
    valid_projects = load_registry_projects(root)
    if not valid_projects:
        raise StoreError("project registry (PROJECT_REGISTRY) missing or unparseable "
                         "— refusing all writes")
    project = fields.get("project")
    # REQUIRED since node.schema v1.7 (2026-09-06): a node has exactly one home and
    # the write door is where that is decided. No default — `global-os` is a home,
    # not a fallback — and the validator's REQUIRED_FIELDS would refuse the write
    # anyway; refusing here names the vocabulary instead of a bare "missing field".
    if project is None or (isinstance(project, str) and not project.strip()):
        raise StoreError("project is required — every node has exactly one home; "
                         f"pass one of {sorted(valid_projects)}")
    if isinstance(project, str) and project not in valid_projects:
        raise StoreError(f"project '{project}' not in vocabulary {sorted(valid_projects)}")

    # Duplicate detection (PRD §6.1) — the gap that let six duplicate nodes
    # through. Compared only against LIVE nodes: a retired/superseded duplicate
    # must not block a legitimate re-creation. HARD signals (block unless
    # allow_duplicate=true): the same Coda page URI, or a byte-identical source
    # that is a real citation URI. SOFT signals (warn, never block): a
    # near-identical title of the same type, or a shared free-text source — both
    # legitimately recur (slug-disambiguated same titles; sources like
    # 'build-spec-v1' or 'session-2026-07-13' cited by many nodes).
    source = fields.get("source")
    src_key = _coda_key(source)
    src_is_uri = isinstance(source, str) and bool(
        re.search(r"[a-z][a-z0-9+.\-]*://", source, re.IGNORECASE))
    tnorm = _norm_title(title)
    live = [n for n in nodes if n.fm.get("status") not in ("retired", "superseded")]
    warnings: list[str] = []
    hard_hits: list[str] = []
    for n in live:
        ns = n.fm.get("source")
        if src_key and _coda_key(ns) == src_key:
            hard_hits.append(f"{n.id} ('{n.fm.get('title')}') — same Coda page ({src_key})")
        elif src_is_uri and isinstance(ns, str) and ns == source:
            hard_hits.append(f"{n.id} ('{n.fm.get('title')}') — identical source URI")
        elif n.fm.get("type") == type and _norm_title(str(n.fm.get("title", ""))) == tnorm:
            warnings.append(f"near-identical title to {n.id} ('{n.fm.get('title')}')")
        elif source and isinstance(ns, str) and ns == source:
            warnings.append(f"shares source with {n.id} ('{n.fm.get('title')}')")
    if hard_hits and not fields.get("allow_duplicate"):
        raise StoreError(
            "refusing a likely duplicate — collides with " + "; ".join(hard_hits) +
            ". An equivalent node already exists; pass allow_duplicate=true to "
            "override (e.g. a deliberate re-read, or a genuinely distinct node).")

    today = date.today().isoformat()
    # pass the suffixes already in use so the last-4 shorthand the corpus and its
    # agents actually use stays unambiguous — see ids.new_id
    node_id = fields.get("id") or new_id(taken_suffixes={i[-4:] for i in by_id})
    if node_id in by_id:
        raise StoreError(f"id collision '{node_id}'")

    fm: dict = {"id": node_id, "type": type, "title": title}
    if fields.get("summary"):
        fm["summary"] = fields["summary"]
    fm["status"] = fields.get("status", "active")
    fm["maturity"] = fields.get("maturity", "captured")
    fm["canonical"] = bool(fields.get("canonical", False))
    fm["created_at"] = today
    fm["updated_at"] = today
    # `asks` and `work_state` were missing from this list until 2026-10-14, so
    # both were DROPPED at the door without a word: the v1.9 docs offered asks
    # at create, every node created after the 2026-09-29 migration (13 by
    # 10-14) was born without them, and logs lost their work_state.
    for k in ("project", "source", "confidence", "tags", "relates_to",
              "supersedes", "resource", "asks", "work_state"):
        if fields.get(k) is not None:
            fm[k] = fields[k]

    check_birth_status(fm["status"])       # birth may not be retired/superseded

    folder = load_folder_map(root).get(type, f"{type}s")
    rel = Path(folder) / (slugify(title) + ".md")
    _contained(root, rel)                   # no escape via type name or FOLDER_MAP value
    if (root / rel).exists():               # never overwrite a different node's file
        rel = Path(folder) / f"{slugify(title)}-{node_id.rsplit('_', 1)[-1]}.md"
    if (root / rel).exists():
        raise StoreError(f"path collision could not be resolved for '{rel}'")
    candidate = Node(id=node_id, path=rel, fm=fm, body=body)

    # a NEW session is never legacy — it is written under the rule, so it must
    # point at the nodes its rulings live in
    _assert_complete(fm, body, bool(fields.get("allow_unfiled_rulings")))

    # transactional prep: candidate + optional superseded target
    updated_target = None
    if fm.get("supersedes"):
        updated_target = _prepare_supersession(by_id, node_id, fm["supersedes"], today)

    val_nodes, val_by_id = [], {}
    for n in nodes:
        use = updated_target if (updated_target and n.id == updated_target.id) else n
        val_nodes.append(use)
        val_by_id[use.id] = use
    val_nodes.append(candidate)
    val_by_id[node_id] = candidate
    findings = _assert_valid(val_nodes, val_by_id, valid_types, valid_projects)

    # Surface the NEW node's own WARN-level findings in the create response.
    # ERRORS already refuse the write; warnings were computed and then thrown
    # away, so a node created with no summary — "structurally under-retrievable",
    # the audit's own words — sailed through silently and was only caught if the
    # author happened to read the node back. That miss happened twice in one day
    # (the full-body-retrieval acceptance, a second note), by the same author: the rule diet's build-on-friction bar,
    # met exactly. Scoped to the candidate's own findings — corpus-wide warnings
    # are the audit's job, not a write receipt's. No second validation pass:
    # _assert_valid already computed these.
    warnings.extend(f.message for f in findings
                    if f.severity == SEVERITY_WARN and str(f.path) == str(rel))

    # commit to disk (atomic: temp files staged, then replaced)
    writes = [(root / rel, render_node(fm, body))]
    if updated_target is not None:
        tgt_body = split_doc((root / updated_target.path).read_text(encoding="utf-8"))[1]
        writes.append((root / updated_target.path, render_node(updated_target.fm, tgt_body)))
    _atomic_write_all(writes)
    git_info = _git_commit(root, [rel] + ([updated_target.path] if updated_target else []),
                           f"okf: create {node_id} ({type}) — {title}")

    return {"id": node_id, "path": str(rel),
            "superseded": updated_target.id if updated_target else None,
            "warnings": warnings,
            **git_info}


def update_node(root, node_id: str, changes: dict, body: str | None = None,
                *, body_append: str | None = None,
                allow_unsupersede: bool = False, allow_unlink: bool = False,
                allow_unfiled_rulings: bool = False,
                allow_record_amendment: bool = False,
                allow_repoint: bool = False,
                allow_body_replacement: bool = False) -> dict:
    root = Path(root)
    with _repo_lock(root):
        return _update_node(root, node_id, changes, body,
                            body_append=body_append,
                            allow_unsupersede=allow_unsupersede, allow_unlink=allow_unlink,
                            allow_unfiled_rulings=allow_unfiled_rulings,
                            allow_record_amendment=allow_record_amendment,
                            allow_repoint=allow_repoint,
                            allow_body_replacement=allow_body_replacement)


def _update_node(root: Path, node_id: str, changes: dict, body,
                  *, allow_unsupersede: bool, allow_unlink: bool,
                  body_append: str | None = None,
                  allow_unfiled_rulings: bool = False,
                  allow_record_amendment: bool = False,
                  allow_body_replacement: bool = False,
                  allow_repoint: bool = False) -> dict:
    _ensure_clean_tree(root)
    nodes, by_id, _, valid_types, dup = load_corpus(root)
    if dup:
        raise StoreError("corpus has duplicate ids; refusing to write until resolved")
    if not valid_types:
        raise StoreError("type registry missing or unparseable — refusing all writes")
    valid_projects = load_registry_projects(root)
    if not valid_projects:
        raise StoreError("project registry (PROJECT_REGISTRY) missing or unparseable "
                         "— refusing all writes")
    node = by_id.get(node_id)
    if node is None:
        raise KeyError(f"no node with id '{node_id}'")

    before = node.fm
    for immutable in ("id", "created_at"):      # P0/P1: identity + birth time are fixed
        if immutable in changes and changes[immutable] != before.get(immutable):
            raise StoreError(f"'{immutable}' is immutable and cannot be changed")

    # relates_to is REPLACE-semantics: a `changes` list that omits an existing link
    # silently drops it (the caller re-specifies a partial list and loses the rest).
    # Guard the silent-loss footgun — dropping a link must be deliberate (allow_unlink),
    # the same speed-bump pattern as allow_duplicate. Adding links stays free.
    if not allow_unlink and isinstance(changes.get("relates_to"), list):
        dropped = [r for r in (before.get("relates_to") or [])
                   if r not in set(changes["relates_to"])]
        if dropped:
            raise StoreError(
                f"update would drop {len(dropped)} existing relates_to link(s): "
                f"{dropped}. Pass a complete relates_to (existing + new), or set "
                f"allow_unlink=true to remove them deliberately.")

    # The registry's machine blocks are runtime-critical: a body update that
    # breaks REGISTRY/FOLDER_MAP would brick every subsequent write (fail-closed
    # everywhere else). Refuse it at the door instead.
    if body is not None and str(node.path) == "schemas/type-registry.md":
        reg = _first_yaml_block_with_key(body, "REGISTRY")
        if not isinstance(reg, dict) or not isinstance(reg.get("node_types"), dict) \
                or not reg["node_types"]:
            raise StoreError("registry update rejected: REGISTRY.node_types must "
                             "remain a non-empty mapping")
        fmap = _first_yaml_block_with_key(body, "FOLDER_MAP")
        if fmap is not None and not isinstance(fmap, dict):
            raise StoreError("registry update rejected: FOLDER_MAP must be a mapping")
        preg = _first_yaml_block_with_key(body, "PROJECT_REGISTRY")
        if preg is not None and (not isinstance(preg, dict)
                                 or not isinstance(preg.get("projects"), dict)
                                 or not preg["projects"]):
            raise StoreError("registry update rejected: PROJECT_REGISTRY.projects "
                             "must be a non-empty mapping")

    # BODY WRITES — two parameters, two intents, and that separation is the whole
    # design. `body` REPLACES; it never appends. The corpus's own "## APPEND /
    # Original text retained above" convention was a HUMAN READING convention the
    # write path never implemented, and on 2026-08-02 a caller reasonably inferred
    # otherwise and destroyed a 2,400-word node — recovered only because an earlier
    # read_node happened still to be in context (the body-replace incident).
    #
    # The first fix (2026-08-02) refused only a DESTRUCTIVE replacement — not
    # containing the old text AND dropping past BODY_SHRINK_FLOOR = 0.5 — because
    # a blanket flag on every body write would be required 100% of the time, and a
    # flag always passed is not a control (the gated-deletion decision).
    #
    # That reasoning was right and its conclusion was a compromise, because `body`
    # was carrying BOTH intents. The 51% hole was real and measured: replacing a
    # 1000-character body with 510 characters of entirely different text passed
    # silently. `body_append` now carries the append intent on its own, so every
    # use of `body` genuinely means "I intend to replace" — and the floor drops to
    # zero. ANY non-containing replacement needs the flag (2026-08-04).
    #
    # The flag is still not required 100% of the time, which is what keeps the gated-deletion decision
    # satisfied: appending does not use `body` at all, and growing a body by
    # concatenation still contains the original and passes untouched.
    if body is not None and body_append is not None:
        raise StoreError(
            "pass `body` OR `body_append`, not both — they are opposite intents "
            "(replace the whole body vs add a dated block to the end). Passing "
            "both is a mistake about which one wins, and guessing is how the "
            "2026-08-02 loss happened.")

    _old_body_len = 0
    if body_append is not None:
        block = str(body_append).strip()
        if not block:
            raise StoreError("body_append is empty — nothing to append.")
        old_body = split_doc((root / node.path).read_text(encoding="utf-8"))[1]
        old_s = old_body.rstrip()
        _old_body_len = len(old_body.strip())
        heading = APPEND_HEADING.format(date=date.today().isoformat())
        # The tool stamps the heading so the convention is enforced rather than
        # remembered. A caller that writes its own heading anyway does not get a
        # duplicate one.
        if block.lstrip().startswith("## APPEND"):
            body = f"{old_s}\n\n{block}\n" if old_s else f"{block}\n"
        else:
            body = f"{old_s}\n\n{heading}\n\n{block}\n" if old_s else \
                   f"{heading}\n\n{block}\n"

    elif body is not None:
        old_body = split_doc((root / node.path).read_text(encoding="utf-8"))[1]
        old_s, new_s = old_body.strip(), body.strip()
        _old_body_len = len(old_s)
        # An APPEND-HEADED replacement is refused for EVERY caller, including one
        # holding allow_body_replacement. Ruled 2026-08-10 after seven
        # nodes were found destroyed this way between 2026-07-24 and 2026-07-29 —
        # ~119,000 characters, undetected for twelve days (the destructive-write pattern).
        #
        # Why the existing containment guard is not enough: it refuses this shape,
        # but it names allow_body_replacement as the way through, and a caller who
        # believes the heading makes `body` append will set that flag and destroy
        # the node anyway. Every one of the seven callers believed exactly that.
        # The flag cannot be the control here, because reaching for it IS the error.
        #
        # There is no legitimate replacement of this shape: a body whose first line
        # is an append heading is a node with no original content left, which is
        # the damage itself rather than a way of writing it. So this one has no
        # override — deliberately, and alone among the guards in this file.
        if new_s.lstrip().startswith("## APPEND"):
            raise StoreError(
                "refused — the replacement body STARTS with an '## APPEND' "
                "heading, which is the signature of an append written as a "
                "replacement. `body` REPLACES: writing that heading yourself does "
                "not make it append, it just names the block that survives. This "
                "destroyed seven nodes in July 2026. Use `body_append` and pass "
                "ONLY your new text — it stamps the dated heading for you and "
                "cannot lose anything. If you genuinely mean to replace, include "
                "the text you intend to keep. allow_body_replacement does NOT "
                "override this.")
        # A `record` is exempt from THIS guard because it has a STRICTER one of its
        # own: _assert_record_immutable makes its body append-only for every caller,
        # with allow_record_amendment as the deliberate override. Running both would
        # (a) mask the specific error with the generic one — the record refusal is
        # the one that tells you to file a new snapshot instead — and (b) demand two
        # flags for one intent, which is how a flag becomes reflex.
        is_record = before.get("type") == "record"
        # Containment is the safe pattern (read, concatenate, write) and is always
        # allowed, at any size ratio. Everything else is a replacement.
        if old_s and old_s not in new_s and not allow_body_replacement and not is_record:
            delta = len(new_s) - len(old_s)
            shape = (f"DROP {-delta} of {len(old_s)} characters "
                     f"({-delta * 100 // len(old_s)}%)" if delta < 0 else
                     f"REWRITE all {len(old_s)} characters (new length {len(new_s)})")
            raise StoreError(
                f"update would {shape} of the body, and the existing body does not "
                f"appear inside the replacement. `body` REPLACES — it never appends, "
                f"and a '## APPEND' heading does not make it append. To ADD to this "
                f"node use `body_append`, which appends a dated block and cannot "
                f"lose anything. To rewrite it deliberately, set "
                f"allow_body_replacement=true.")

    today = date.today().isoformat()
    after = dict(before)
    after.update(changes)
    after["id"] = node_id                        # identity never moves
    after["updated_at"] = today

    # transition checks (the whole reason update lives here, not in the CLI)
    if "status" in changes and changes["status"] != before.get("status"):
        check_status_transition(before.get("status"), after["status"],
                                allow_unsupersede=allow_unsupersede)
        # RULE (node.schema §6): retiring/superseding FORCES canonical=false —
        # the Librarian applies it, it doesn't refuse the retirement
        if after["status"] in ("retired", "superseded"):
            after["canonical"] = False
        # un-supersession clears the stale back-pointer (schema §4 guard)
        if before.get("status") == "superseded" and after["status"] == "active":
            after.pop("superseded_by", None)
    if "maturity" in changes and changes["maturity"] != before.get("maturity"):
        check_maturity_transition(before.get("maturity"), after["maturity"])

    # a type change re-files the node into its type's folder (path is not identity)
    new_path = node.path
    if "type" in changes and changes["type"] != before.get("type"):
        folder = load_folder_map(root).get(after["type"], f"{after['type']}s")
        moved = Path(folder) / node.path.name
        _contained(root, moved)
        if moved != node.path and (root / moved).exists():
            raise StoreError(f"cannot re-file node: '{moved}' already exists")
        new_path = moved

    updated = Node(id=node_id, path=new_path, fm=after)
    overrides = {node_id: updated}              # id -> in-memory node to write

    old_sup, new_sup = before.get("supersedes"), after.get("supersedes")
    if old_sup != new_sup:
        # a retired node's pointers are historical: it may DROP one, never mint
        # one (the validator exempts retired carriers, so this is the only gate)
        if new_sup and after.get("status") == "retired":
            raise StoreError("a retired node cannot establish a new supersession "
                             "(removing its historical pointer is allowed)")
        if old_sup:                             # successor stopped superseding old target -> restore it
            restored = _prepare_unsupersession(by_id, old_sup, today)
            if restored is not None:
                overrides[restored.id] = restored
        if new_sup:                             # supersede the new target
            sup = _prepare_supersession(by_id, node_id, new_sup, today)
            if sup.id == node_id:
                raise StoreError("a node cannot supersede itself")
            overrides[sup.id] = sup

    # COMPLETENESS gate — a session may not be written with rulings stated in prose
    # and filed nowhere. Uses the body that will actually land (existing body when
    # the caller passes none), so an update that only touches frontmatter is still
    # judged on what the node will say.
    effective_body = (split_doc((root / node.path).read_text(encoding="utf-8"))[1]
                      if body is None else body)
    _assert_complete(after, effective_body, allow_unfiled_rulings)

    # RECORD IMMUTABILITY — the discipline the `record` type exists for.
    _assert_record_immutable(before, after,
                             split_doc((root / node.path).read_text(encoding="utf-8"))[1],
                             effective_body, allow_record_amendment)

    # A Drive file ID survives a MOVE, so a changed id means delete-and-reupload —
    # which silently breaks every pointer while all other checks stay green.
    _assert_pointer_stable(before, after, allow_repoint)

    val_nodes, val_by_id = [], {}
    for n in nodes:
        use = overrides.get(n.id, n)
        val_nodes.append(use)
        val_by_id[use.id] = use
    _assert_valid(val_nodes, val_by_id, valid_types, valid_projects)

    # commit (atomic across all touched files)
    writes = []
    for oid, onode in overrides.items():
        if oid == node_id:
            keep_body = split_doc((root / node.path).read_text(encoding="utf-8"))[1]
            content = render_node(onode.fm, keep_body if body is None else body)
        else:
            obody = split_doc((root / onode.path).read_text(encoding="utf-8"))[1]
            content = render_node(onode.fm, obody)
        writes.append((root / onode.path, content))
    deletes = [root / node.path] if new_path != node.path else []
    _atomic_write_all(writes, deletes)          # refile deletion is transactional too
    git_paths = [n.path for n in overrides.values()]
    if deletes:
        git_paths.append(node.path)             # pathspec stages the deletion too
    # The commit message names the BODY DELTA, not just the frontmatter keys the
    # caller declared. Ruled 2026-08-10. Every one of the seven destroying
    # writes of July 2026 logged as `okf: update <id> (relates_to, summary)` while
    # replacing the whole body — so `git log` described a metadata edit and the
    # damage stayed invisible for twelve days and 309 commits. A destructive write
    # must not be able to present itself as a link change.
    _fields = sorted(changes)
    if body is not None:
        _fields.append(f"body {len(body.strip()) - _old_body_len:+d}")
    git_info = _git_commit(root, git_paths,
                           f"okf: update {node_id} ({', '.join(_fields)})")

    out = {"id": node_id, "path": str(new_path),
           "superseded": new_sup if (old_sup != new_sup and new_sup) else None,
           "unsuperseded": old_sup if (old_sup != new_sup and old_sup) else None,
           **git_info}
    if body is not None:
        # Report the size change even when the guard allowed it. The 2026-08-02
        # loss was silent precisely because `committed: true` looked identical
        # whether the call appended a paragraph or deleted 2,400 words.
        out["body_bytes"] = {"before": _old_body_len, "after": len(body.strip()),
                             "delta": len(body.strip()) - _old_body_len}
    warnings = _summary_warnings(after, changes, body, effective_body)
    if warnings:
        out["warnings"] = warnings
    return out


# Types whose summary is not a current state to keep in step with the body: a
# log grows by design and a record's summary is frozen with its facts.
SUMMARY_STEP_EXEMPT = ("log", "record")


def _summary_warnings(after: dict, changes: dict, body, effective_body) -> list:
    """The update receipt's summary warnings (2026-10-22). The summary is the
    node's CURRENT STATE and the body its history (librarian/summary.py), so a
    body write that leaves the summary as it was is how the two drift apart —
    the 2026-08-25 pattern (the foot-of-node correction pattern) of a correction filed
    where nobody reads it. Warned, never refused."""
    out = []
    if body is not None or "summary" in changes:
        out += _summary.check(after.get("summary"), effective_body, after.get("type"))
    if (body is not None and "summary" not in changes
            and after.get("type") not in SUMMARY_STEP_EXEMPT):
        out.append("the body changed but the summary did not — the summary is the "
                   "node's current state: rewrite it in the same call "
                   "(changes={'summary': ...}), saying in the bullet what changed or "
                   "was overruled")
    return out


def delete_node(root, node_id: str) -> dict:
    """Hard-delete a node's file. NOT an OKF norm (prefer retire/supersede) — but
    legitimate for junk, and safe because git keeps the history. References are
    cleaned up transactionally so no dangling links remain: the id is stripped
    from every relates_to; a node that superseded the deleted one has its
    supersedes cleared; a node that WAS superseded by it is restored to active."""
    root = Path(root)
    with _repo_lock(root):
        return _delete_node(root, node_id)


def _delete_node(root: Path, node_id: str) -> dict:
    _ensure_clean_tree(root)
    nodes, by_id, _, valid_types, dup = load_corpus(root)
    if dup:
        raise StoreError("corpus has duplicate ids; refusing to write until resolved")
    target = by_id.get(node_id)
    if target is None:
        raise KeyError(f"no node with id '{node_id}'")

    today = date.today().isoformat()
    overrides = {}                                   # id -> referrer Node with the ref removed
    for n in nodes:
        if n.id == node_id:
            continue
        nf = dict(n.fm)
        changed = False
        rel = nf.get("relates_to")
        if isinstance(rel, list) and node_id in rel:
            nf["relates_to"] = [r for r in rel if r != node_id]
            changed = True
        if nf.get("supersedes") == node_id:          # referrer superseded the deleted node
            nf.pop("supersedes", None)
            changed = True
        if nf.get("superseded_by") == node_id:       # referrer was superseded BY it -> restore
            nf.pop("superseded_by", None)
            if nf.get("status") == "superseded":
                nf["status"] = "active"
            changed = True
        if changed:
            nf["updated_at"] = today
            overrides[n.id] = Node(id=n.id, path=n.path, fm=nf)

    val_nodes, val_by_id = [], {}
    for n in nodes:
        if n.id == node_id:
            continue
        use = overrides.get(n.id, n)
        val_nodes.append(use)
        val_by_id[use.id] = use
    _assert_valid(val_nodes, val_by_id, valid_types, load_registry_projects(root))

    writes = []
    for onode in overrides.values():
        body = split_doc((root / onode.path).read_text(encoding="utf-8"))[1]
        writes.append((root / onode.path, render_node(onode.fm, body)))
    _atomic_write_all(writes, deletes=[root / target.path])
    git_paths = [n.path for n in overrides.values()] + [target.path]
    git_info = _git_commit(root, git_paths,
                           f"okf: delete {node_id} ({target.fm.get('title', '')})")
    return {"id": node_id, "deleted": True, "cleaned_refs": len(overrides), **git_info}
