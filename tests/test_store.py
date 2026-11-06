"""Write-path tests for the Librarian store. Dependency-free:

    python tests/test_store.py

Each test runs against a FRESH throwaway copy of the real corpus (schemas +
decisions), so the real corpus is never touched and tests don't leak into each
other.
"""
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for _corpus

from librarian import context, store, validate    # noqa: E402
from librarian.transitions import TransitionError  # noqa: E402
from _corpus import skip_if_no_corpus             # noqa: E402

CORPUS_SRC = skip_if_no_corpus()

NON_CORPUS_DIRS = {".git", "librarian", "tests", "deploy", "__pycache__",
                   ".venv", "venv", ".pytest_cache", ".claude"}

# A denylist that happens to match everything yields an EMPTY corpus, and every
# test then passes against nothing — a green suite that verifies nothing, which is
# strictly worse than a red one. Guard the floor.
MIN_FIXTURE_NODES = 20


def fresh_corpus() -> Path:
    """Copy every corpus folder (anything that isn't code/tooling) — a hardcoded
    subset breaks the moment a node in one folder links into an uncopied one."""
    tmp = Path(tempfile.mkdtemp(prefix="okf_test_"))
    for sub in CORPUS_SRC.iterdir():
        if sub.is_dir() and sub.name not in NON_CORPUS_DIRS:
            shutil.copytree(sub, tmp / sub.name)
    found = len(list(tmp.rglob("*.md")))
    if found < MIN_FIXTURE_NODES:
        raise RuntimeError(
            f"refusing to test against an empty corpus: copied {found} node file(s) "
            f"from {CORPUS_SRC} (need >= {MIN_FIXTURE_NODES}). Set OKF_TEST_CORPUS "
            "to the OKF-corpus checkout.")
    return tmp


def _validate_clean(root):
    findings, nodes, _, _ = validate.validate_corpus(root)
    errs = [str(f) for f in findings if f.severity == validate.SEVERITY_ERROR]
    assert not errs, "corpus not clean:\n" + "\n".join(errs)
    return nodes


# pytest support: expose `root` as a fixture so `python -m pytest` works too.
# Guarded so the file still runs dependency-free via its own runner (main() below).
try:
    import pytest

    @pytest.fixture(name="root")
    def _root_fixture():
        corpus = fresh_corpus()
        try:
            yield corpus
        finally:
            shutil.rmtree(corpus, ignore_errors=True)
except ImportError:
    pass


# ── create / read ─────────────────────────────────────────────────────────────
def test_create_and_read(root):
    r = store.create_node(root, project="global-os", type="note", title="Note about caching",
                          body="Body.", summary="one line")
    assert r["id"].startswith("okf_")
    assert r["path"].startswith("knowledge/")
    rd = store.read_node(root, r["id"])
    assert rd["frontmatter"]["title"] == "Note about caching"
    assert rd["frontmatter"]["maturity"] == "captured"     # default
    assert "Body." in rd["body"]


def test_slug_collision_no_overwrite(root):  # P0 #1
    a = store.create_node(root, project="global-os", type="note", title="Same Title")
    b = store.create_node(root, project="global-os", type="note", title="Same Title")
    assert a["path"] != b["path"], "second create must not reuse the path"
    assert (root / a["path"]).exists() and (root / b["path"]).exists()
    assert store.read_node(root, a["id"])["id"] == a["id"]
    assert store.read_node(root, b["id"])["id"] == b["id"]


# ── birth rules ───────────────────────────────────────────────────────────────
def test_born_established_allowed(root):       # judgment call A
    r = store.create_node(root, project="global-os", type="decision", title="A born-established decision",
                          status="active", maturity="established", canonical=True, body="x")
    assert store.read_node(root, r["id"])["frontmatter"]["maturity"] == "established"


def test_born_retired_refused(root):           # P1 birth-status
    try:
        store.create_node(root, project="global-os", type="note", title="Born dead", status="retired")
        assert False, "born-retired should be refused"
    except TransitionError:
        pass


def test_canonical_guard(root):
    try:
        store.create_node(root, project="global-os", type="note", title="Bad canon", canonical=True)
        assert False, "canonical on a captured note should be refused"
    except store.StoreError:
        pass


def test_resource_requires_uri(root):
    try:
        store.create_node(root, project="global-os", type="resource", title="Dangling pointer",
                          resource={"kind": "todoist_project"})   # no uri
        assert False, "resource without uri should be refused"
    except store.StoreError as e:
        assert "uri" in str(e), "must fail on the MISSING URI, not the kind"


def test_task_stub_is_identity_only(root):
    """Was test_coda_row_stub. The substrate moved to Todoist on 2026-08-03
    (the tasks-live-in-the-task-app decision), but the contract it pins did not: a task pointer
    carries identity and nothing else, because a pointer holding state is a copy
    of live data by another name (the knowledge-not-live-data decision)."""
    r = store.create_node(root, project="global-os", type="resource", title="Task 47 — quality-audit pipeline",
                          summary="A pointer to one live Todoist task, kept as "
                                  "identity only so it can never drift from source.",
                          resource={"kind": "todoist_task",
                                    "uri": "https://app.todoist.com/app/task/47"})
    assert r["path"].startswith("resources/")
    fm = store.read_node(root, r["id"])["frontmatter"]
    assert fm["resource"]["kind"] == "todoist_task"
    assert set(fm["resource"].keys()) == {"kind", "uri"}   # identity only, no state


# ── transitions ───────────────────────────────────────────────────────────────
def test_legal_transitions(root):
    r = store.create_node(root, project="global-os", type="note", title="Transitions")
    store.update_node(root, r["id"], {"status": "parked"})
    assert store.read_node(root, r["id"])["frontmatter"]["status"] == "parked"
    store.update_node(root, r["id"], {"maturity": "candidate"})
    assert store.read_node(root, r["id"])["frontmatter"]["maturity"] == "candidate"


def test_maturity_regression_refused(root):
    r = store.create_node(root, project="global-os", type="note", title="No regress", maturity="candidate")
    try:
        store.update_node(root, r["id"], {"maturity": "captured"})
        assert False
    except TransitionError:
        pass


def test_terminal_retired(root):
    r = store.create_node(root, project="global-os", type="note", title="Terminal")
    store.update_node(root, r["id"], {"status": "retired"})
    try:
        store.update_node(root, r["id"], {"status": "active"})
        assert False
    except TransitionError:
        pass


# ── immutable fields ──────────────────────────────────────────────────────────
def test_id_immutable(root):                   # P0 #2
    r = store.create_node(root, project="global-os", type="note", title="Fixed id")
    try:
        store.update_node(root, r["id"], {"id": "okf_20000101_000000_dead"})
        assert False, "id change should be refused"
    except store.StoreError:
        pass


def test_created_at_immutable(root):
    r = store.create_node(root, project="global-os", type="note", title="Fixed birth")
    try:
        store.update_node(root, r["id"], {"created_at": "2000-01-01"})
        assert False
    except store.StoreError:
        pass


# ── supersession ──────────────────────────────────────────────────────────────
def _canonical_target(root):
    """Mint a canonical decision to supersede, instead of borrowing a live one.

    These two tests used to hardcode the first task-substrate decision and assert it was
    active+canonical. On 2026-08-03 a parallel session superseded it ("Tasks are
    canonical in Coda" -> "... in Todoist") and both tests failed — not because
    the code broke, but because the corpus legitimately moved on. A fixture that
    breaks when the knowledge base does its job is testing the corpus, not the
    store. The throwaway corpus is writable; use it.
    """
    return store.create_node(root, project="global-os", type="decision", title="Supersession target",
                             status="active", maturity="established",
                             canonical=True,
                             summary="A canonical decision minted purely as a "
                                     "supersession target for the store tests.",
                             body="x")["id"]


def test_supersession_side_effect(root):
    target = _canonical_target(root)
    before = store.read_node(root, target)["frontmatter"]
    assert before["status"] == "active" and before["canonical"] is True
    r = store.create_node(root, project="global-os", type="decision", title="Supersede v2",
                          status="active", maturity="established", canonical=True,
                          supersedes=target, body="x")
    after = store.read_node(root, target)["frontmatter"]
    assert after["status"] == "superseded"
    assert after["canonical"] is False
    assert after["superseded_by"] == r["id"]


def test_unsupersession_cleanup(root):         # P1 un-supersession
    target = _canonical_target(root)
    r = store.create_node(root, project="global-os", type="decision", title="Supersede then undo",
                          status="active", maturity="established", canonical=True,
                          supersedes=target, body="x")
    assert store.read_node(root, target)["frontmatter"]["status"] == "superseded"
    res = store.update_node(root, r["id"], {"supersedes": None})   # drop the pointer
    assert res["unsuperseded"] == target
    restored = store.read_node(root, target)["frontmatter"]
    assert restored["status"] == "active"
    assert "superseded_by" not in restored
    assert restored["canonical"] is False       # NOT auto-restored (documented)


# ── review-2 adversarial cases ────────────────────────────────────────────────
def test_update_sets_supersedes(root):         # P0 regression: args were reversed
    a = store.create_node(root, project="global-os", type="note", title="Old approach")
    b = store.create_node(root, project="global-os", type="note", title="New approach")
    r = store.update_node(root, b["id"], {"supersedes": a["id"], "summary": "kept"})
    assert r["superseded"] == a["id"]
    target = store.read_node(root, a["id"])["frontmatter"]
    successor = store.read_node(root, b["id"])["frontmatter"]
    assert target["status"] == "superseded" and target["superseded_by"] == b["id"]
    assert successor["status"] == "active", "successor must NOT be the one superseded"
    assert successor["summary"] == "kept", "successor's changes must not be discarded"
    assert successor["supersedes"] == a["id"]


def test_missing_registry_fails_closed(root):  # P0: no vocabulary = broken corpus
    (root / "schemas" / "type-registry.md").unlink()
    try:
        store.create_node(root, project="global-os", type="note", title="No registry")
        assert False, "write without a registry should be refused"
    except store.StoreError as e:
        assert "registry" in str(e)


def test_path_traversal_refused(root):         # P0: hostile FOLDER_MAP value
    reg = root / "schemas" / "type-registry.md"
    reg.write_text(reg.read_text().replace("note:      /knowledge",
                                           "note:      /../evil"))
    try:
        store.create_node(root, project="global-os", type="note", title="Escape attempt")
        assert False, "path escaping the corpus root should be refused"
    except store.StoreError as e:
        assert "escapes" in str(e)


def test_concurrent_creates_no_loss(root):     # P1: repo write lock
    import threading
    results, errors = [], []

    def create(i):
        try:
            results.append(store.create_node(root, project="global-os", type="note", title="Same Race Title"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=create, args=(i,)) for i in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    paths = {r["path"] for r in results}
    assert len(paths) == 2, f"concurrent creates must not share a path: {paths}"
    assert all((root / p).exists() for p in paths)


def test_hand_superseded_without_successor_flagged(root):  # P1 reverse invariant
    r = store.create_node(root, project="global-os", type="note", title="Orphan supersession")
    p = root / r["path"]
    p.write_text(p.read_text().replace("status: active", "status: superseded"))
    findings, *_ = validate.validate_corpus(root)
    errs = [f for f in findings if f.severity == validate.SEVERITY_ERROR]
    assert any("superseded_by" in str(e) for e in errs), errs


def test_resource_identity_only(root):         # the first task-substrate decision amendment enforced
    try:
        store.create_node(root, project="global-os", type="resource", title="Stateful stub",
                          resource={"kind": "todoist_task", "uri": "https://app.todoist.com/app/task/1",
                                    "due_date": "2026-08-01"})
        assert False, "resource block carrying state should be refused"
    except store.StoreError as e:
        assert "identity only" in str(e)


def test_malformed_yaml_is_error_not_crash(root):
    (root / "decisions" / "broken.md").write_text("---\ntitle: [unclosed\n---\nbody\n")
    findings, *_ = validate.validate_corpus(root)
    assert any("malformed YAML" in str(f) for f in findings)


def test_frontmatter_without_id_is_error(root):
    (root / "decisions" / "no-id.md").write_text("---\ntitle: Lost node\n---\nbody\n")
    findings, *_ = validate.validate_corpus(root)
    assert any("missing 'id'" in str(f) for f in findings)


def test_type_change_refiles(root):            # P2: filing follows type
    r = store.create_node(root, project="global-os", type="note", title="Actually a decision")
    old_path = r["path"]
    assert old_path.startswith("knowledge/")
    r2 = store.update_node(root, r["id"], {"type": "decision"})
    assert r2["path"].startswith("decisions/")
    assert not (root / old_path).exists(), "old location must be removed"
    assert store.read_node(root, r["id"])["frontmatter"]["type"] == "decision"


def test_schema_code_drift(root):              # P2: constants must mirror node.schema
    from librarian.transitions import STATUS_TRANSITIONS, MATURITY_ORDER
    text = (CORPUS_SRC / "schemas" / "node.schema.md").read_text(encoding="utf-8")
    spec = validate._first_yaml_block_with_key(text, "SPEC")
    trans = validate._first_yaml_block_with_key(text, "TRANSITIONS")
    assert set(spec["fields"]["status"]["values"]) == validate.STATUSES
    assert set(spec["fields"]["maturity"]["values"]) == validate.MATURITIES
    assert set(spec["fields"]["resource"]["fields"]["kind"]["enum"]) == validate.RESOURCE_KINDS
    assert set(spec["required_in_valid_node"]) == set(validate.REQUIRED_FIELDS)
    assert set(spec["fields"].keys()) == validate.KNOWN_FIELDS
    assert set(store.FIELD_ORDER) == validate.KNOWN_FIELDS
    schema_status = {(None if k == "_birth_" else k): set(v) for k, v in trans["status"].items()}
    assert schema_status == {k: set(v) for k, v in STATUS_TRANSITIONS.items()}
    assert trans["maturity"]["_birth_"] == MATURITY_ORDER  # birth allows all, in order


# ── review-3 cases ────────────────────────────────────────────────────────────
def test_unhashable_values_are_findings_not_crashes(root):   # P1
    (root / "decisions" / "hostile.md").write_text(
        "---\nid: okf_20260713_000001_beef\ntype: []\ntitle: Hostile\n"
        "status: active\nmaturity: captured\ncanonical: false\n"
        "created_at: 2026-07-13\nupdated_at: 2026-07-13\n"
        "relates_to: [{a: 1}]\nsupersedes: [x]\n---\nbody\n")
    findings, *_ = validate.validate_corpus(root)   # must not raise
    msgs = [str(f) for f in findings]
    assert any("type must be a string" in m for m in msgs), msgs
    assert any("relates_to entry must be an id string" in m for m in msgs)
    assert any("supersedes must be an id string" in m for m in msgs)


def test_asks_field_is_validated(root):                     # v1.9 (2026-09-29)
    """`asks` is a list of short non-empty strings — retrieval metadata, never a
    second summary. A string, an empty entry, or nine entries are refused; a
    proper list validates green."""
    fm = ("---\nid: okf_20260929_000001_a5c5\ntype: note\ntitle: Asks fixture\n"
          "status: active\nmaturity: captured\ncanonical: false\nproject: global-os\n"
          "created_at: 2026-09-29\nupdated_at: 2026-09-29\n"
          "summary: a fixture node whose only purpose is to exercise the validator on the new field\n")
    def run(asks_yaml):
        (root / "knowledge" / "phrasings-fixture.md").write_text(fm + asks_yaml + "---\nbody\n")
        findings, *_ = validate.validate_corpus(root)
        return [str(f) for f in findings if "asks must" in str(f) or "asks holds" in str(f)]
    assert run("asks: not a list\n"), "a bare string is refused"
    assert run("asks: ['fine', '']\n"), "an empty phrasing is refused"
    assert run("asks: [" + ", ".join(f"'q{i}'" for i in range(9)) + "]\n"), "nine entries are refused"
    assert run("asks: ['what to do when a loaf over-proofs', 'delete a note by itself']\n") == []


def test_resource_top_level_state_refused(root):             # P1
    r = store.create_node(root, project="global-os", type="resource", title="Clean stub",
                          resource={"kind": "todoist_task", "uri": "https://app.todoist.com/app/task/9"})
    try:
        store.update_node(root, r["id"], {"due_date": "2026-08-01"})
        assert False, "top-level state on a resource node should be refused"
    except store.StoreError as e:
        assert "unknown top-level" in str(e)


def test_resource_uri_must_be_string(root):                  # P1
    try:
        store.create_node(root, project="global-os", type="resource", title="Numeric uri",
                          resource={"kind": "todoist_task", "uri": 12345})
        assert False
    except store.StoreError as e:
        assert "non-empty string" in str(e)


def test_atomic_rollback_on_partial_failure(root):           # P1
    a = store.create_node(root, project="global-os", type="note", title="Rollback target")
    before = (root / a["path"]).read_text()
    real_replace, calls = store.os.replace, []

    def failing_replace(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise OSError("injected failure on second replace")
        return real_replace(src, dst)

    store.os.replace = failing_replace
    try:
        store.create_node(root, project="global-os", type="note", title="Successor",
                          status="active", maturity="established",
                          supersedes=a["id"])
        assert False, "injected failure should propagate"
    except OSError:
        pass
    finally:
        store.os.replace = real_replace
    assert (root / a["path"]).read_text() == before, "target must be rolled back"
    assert not list((root / "knowledge").glob("successor*")), "new file must be rolled back"
    findings, *_ = validate.validate_corpus(root)
    assert not [f for f in findings if f.severity == validate.SEVERITY_ERROR]


def test_retire_canonical_forces_flag(root):                 # P2: rule applied, not refused
    r = store.create_node(root, project="global-os", type="note", title="Canonical then retired",
                          status="active", maturity="established", canonical=True)
    store.update_node(root, r["id"], {"status": "retired"})
    fm = store.read_node(root, r["id"])["frontmatter"]
    assert fm["status"] == "retired" and fm["canonical"] is False


def test_unsupersede_after_successor_retired(root):          # P2: schema route usable
    a = store.create_node(root, project="global-os", type="note", title="Route A")
    b = store.create_node(root, project="global-os", type="note", title="Route B", supersedes=a["id"])
    store.update_node(root, b["id"], {"status": "retired"})
    store.update_node(root, a["id"], {"status": "active"}, allow_unsupersede=True)
    fm = store.read_node(root, a["id"])["frontmatter"]
    assert fm["status"] == "active" and "superseded_by" not in fm


def test_duplicate_canonical_title_warns(root):              # P2
    store.create_node(root, project="global-os", type="note", title="One True Doctrine",
                      status="active", maturity="established", canonical=True)
    store.create_node(root, project="global-os", type="note", title="One  True Doctrine!",
                      status="active", maturity="established", canonical=True)
    findings, *_ = validate.validate_corpus(root)
    warns = [f for f in findings if f.severity == validate.SEVERITY_WARN]
    assert any("duplicate canonical claim" in str(w) for w in warns), warns


# ── review-4 cases ────────────────────────────────────────────────────────────
def test_refile_rollback_on_delete_failure(root):            # P1
    r = store.create_node(root, project="global-os", type="note", title="Refile rollback")
    old_path, before = r["path"], (root / r["path"]).read_text()
    real_remove = store.os.remove

    def failing_remove(p):
        raise OSError("injected unlink failure")

    store.os.remove = failing_remove
    try:
        store.update_node(root, r["id"], {"type": "decision"})
        assert False, "injected failure should propagate"
    except OSError:
        pass
    finally:
        store.os.remove = real_remove
    assert (root / old_path).read_text() == before, "old file must survive rollback"
    assert not (root / "decisions" / Path(old_path).name).exists(), \
        "new location must be rolled back (no duplicate id)"
    findings, *_ = validate.validate_corpus(root)
    assert not [f for f in findings if f.severity == validate.SEVERITY_ERROR]


def test_retired_cannot_mint_supersession(root):             # P1
    a = store.create_node(root, project="global-os", type="note", title="Victim C")
    b = store.create_node(root, project="global-os", type="note", title="Retired actor")
    store.update_node(root, b["id"], {"status": "retired"})
    try:
        store.update_node(root, b["id"], {"supersedes": a["id"]})
        assert False, "retired node must not establish a new supersession"
    except store.StoreError as e:
        assert "retired" in str(e)
    # …but DROPPING a historical pointer stays allowed
    c = store.create_node(root, project="global-os", type="note", title="Old truth")
    d = store.create_node(root, project="global-os", type="note", title="New truth", supersedes=c["id"])
    store.update_node(root, d["id"], {"status": "retired"})
    store.update_node(root, d["id"], {"supersedes": None})
    assert store.read_node(root, c["id"])["frontmatter"]["status"] == "active"


def test_broken_registry_block_fails_closed_not_crash(root):  # P1
    reg = root / "schemas" / "type-registry.md"
    reg.write_text("---\nid: okf_20260706_090500_0f02\ntype: reference\n"
                   "title: OKF Type Registry\nstatus: active\nmaturity: established\n"
                   "canonical: true\ncreated_at: 2026-07-06\nupdated_at: 2026-07-13\n"
                   "---\n\n```yaml\nREGISTRY:\n  node_types: []\n```\n")
    findings, *_ = validate.validate_corpus(root)   # must not raise
    assert any("registry missing or unparseable" in str(f) for f in findings)
    try:
        store.create_node(root, project="global-os", type="note", title="Should refuse")
        assert False
    except store.StoreError as e:
        assert "registry" in str(e)


def test_registry_body_update_guarded(root):                 # P1
    reg_id = "okf_20260706_090500_0f02"
    try:
        store.update_node(root, reg_id, {},
                          body="```yaml\nREGISTRY:\n  node_types: []\n```\n")
        assert False, "vocabulary-breaking registry body should be refused"
    except store.StoreError as e:
        assert "non-empty mapping" in str(e)


def test_scalar_and_ownership_findings(root):                # P2
    (root / "decisions" / "sloppy.md").write_text(
        "---\nid: okf_20260713_000002_cafe\ntype: note\ntitle: Sloppy\n"
        "summary: [not, a, string]\nproject: {x: 1}\nsource: [2]\n"
        "status: active\nmaturity: captured\ncanonical: false\n"
        "created_at: 2026-07-13\nupdated_at: '2026-99-99junk'\n"
        "superseded_by: okf_20260713_000001_beef\n---\nbody\n")
    findings, *_ = validate.validate_corpus(root)
    msgs = [str(f) for f in findings]
    for expect in ("summary must be a string", "project must be a string",
                   "source must be a string", "updated_at must be a date",
                   "superseded_by present on a node"):
        assert any(expect in m for m in msgs), (expect, msgs)


# ── duplicate detection (PRD §6.1 — the gap that let 6 dups through) ───────────
def test_duplicate_source_refused_despite_phrasing(root):
    # SAME Coda page, DIFFERENT hand-written suffix + &gt; vs > — the real dup
    # shape the exact-string check missed. Must still be refused.
    store.create_node(root, project="global-os", type="reference", title="RAG Architectures",
                      source="coda://docs/X/pages/section-abc (Learning &gt; AI) — original: @x")
    try:
        store.create_node(root, project="global-os", type="reference", title="Top 5 RAG Architectures",
                          source="coda://docs/X/pages/section-abc (Learning > AI) | orig: @x")
        assert False, "same Coda page must be refused regardless of source phrasing"
    except store.StoreError as e:
        assert "already exists" in str(e) and "allow_duplicate" in str(e)


def test_duplicate_source_override(root):
    src = "coda://docs/X/pages/section-p2 (a page)"
    store.create_node(root, project="global-os", type="reference", title="Re-read source", source=src)
    r = store.create_node(root, project="global-os", type="reference", title="Deliberate re-read",
                          source=src, allow_duplicate=True)
    assert r["id"].startswith("okf_")   # override lets it through


def test_distinct_coda_pages_not_blocked(root):   # migration must not self-trip
    for i in range(3):
        r = store.create_node(root, project="global-os", type="reference", title=f"Page {i}",
                             source=f"coda://docs/X/pages/section-p{i} (path)")
        assert r["id"].startswith("okf_")


def test_freetext_source_repeats_allowed(root):   # 'session-2026-07-13' repeats legitimately
    store.create_node(root, project="global-os", type="decision", title="Decision one", source="session-2026-07-13")
    r = store.create_node(root, project="global-os", type="decision", title="Decision two", source="session-2026-07-13")
    assert r["id"].startswith("okf_"), "non-Coda free-text sources must not hard-block"


def test_near_title_warns_not_blocks(root):
    store.create_node(root, project="global-os", type="note", title="Customer System")
    r = store.create_node(root, project="global-os", type="note", title="customer  system!")   # same normalized
    assert r["id"].startswith("okf_"), "near-title must not block"
    assert any("near-identical title" in w for w in r["warnings"])


def test_duplicate_identical_uri_source_blocks(root):   # byte-identical citation URI = hard
    url = "https://example.com/post/42"
    store.create_node(root, project="global-os", type="reference", title="Cited once", source=url)
    try:
        store.create_node(root, project="global-os", type="reference", title="Cited again, different title", source=url)
        assert False, "identical citation URI should be refused"
    except store.StoreError as e:
        assert "allow_duplicate" in str(e)
    r = store.create_node(root, project="global-os", type="reference", title="Deliberate re-cite",
                          source=url, allow_duplicate=True)
    assert r["id"].startswith("okf_")   # override lets it through


def test_duplicate_excludes_retired(root):   # a retired dup must not block re-creation
    src = "coda://docs/Y/pages/section-ret (a page)"
    a = store.create_node(root, project="global-os", type="reference", title="First cut", source=src)
    store.update_node(root, a["id"], {"status": "retired"})
    r = store.create_node(root, project="global-os", type="reference", title="Fresh cut", source=src)
    assert r["id"].startswith("okf_"), "a retired duplicate must not block a legitimate re-creation"


def test_shared_freetext_source_warns_not_blocks(root):   # 'build-spec' cited by many nodes
    store.create_node(root, project="global-os", type="note", title="Cites the spec A", source="build-spec-v1-2026-07-05")
    r = store.create_node(root, project="global-os", type="note", title="Cites the spec B", source="build-spec-v1-2026-07-05")
    assert r["id"].startswith("okf_"), "shared free-text source must not block"
    assert any("shares source" in w for w in r["warnings"])


# ── project vocabulary (the world axis, PROJECT_REGISTRY) ──────────────────────
def test_create_rejects_unknown_project(root):
    try:
        store.create_node(root, type="note", title="Typo world", project="bakry")
        assert False, "unknown project should be refused"
    except store.StoreError as e:
        assert "not in vocabulary" in str(e) and "bakery" in str(e)


def test_create_accepts_valid_project(root):
    r = store.create_node(root, type="note", title="Garden note", project="garden")
    assert store.read_node(root, r["id"])["frontmatter"]["project"] == "garden"


def test_create_keeps_asks_and_work_state(root):
    """Both were dropped at the door until 2026-10-14: the v1.9 docs offered
    `asks` at create, and every node created after the 2026-09-29 migration
    was born without them."""
    asks = ["how often do we check search quality", "weekly search check"]
    r = store.create_node(root, type="log", title="A weekly record", project="global-os",
                          asks=asks, work_state="open")
    fm = store.read_node(root, r["id"])["frontmatter"]
    assert fm["asks"] == asks and fm["work_state"] == "open"


def test_no_project_refused(root):             # project is REQUIRED (node.schema v1.7)
    for bad in ({}, {"project": None}, {"project": ""}, {"project": "  "}):
        try:
            store.create_node(root, type="note", title="Worldless note", **bad)
            assert False, f"a node with no home must be refused (fields={bad})"
        except store.StoreError as e:
            assert "project is required" in str(e) and "global-os" in str(e)


def test_update_cannot_strip_project(root):     # the field cannot be removed later either
    r = store.create_node(root, type="note", title="Homed then unhomed", project="personal")
    try:
        store.update_node(root, r["id"], {"project": None})
        assert False, "clearing project on update must be refused"
    except store.StoreError as e:
        assert "missing required field 'project'" in str(e)
    assert store.read_node(root, r["id"])["frontmatter"]["project"] == "personal"


def test_validator_flags_node_without_project(root):   # the read path agrees with the write door
    (root / "knowledge" / "homeless.md").write_text(
        "---\nid: okf_20260906_000001_beef\ntype: note\ntitle: Homeless\n"
        "status: active\nmaturity: captured\ncanonical: false\n"
        "created_at: 2026-09-06\nupdated_at: 2026-09-06\n---\nbody\n")
    findings, *_ = validate.validate_corpus(root)
    assert any("missing required field 'project'" in str(f) for f in findings), \
        [str(f) for f in findings]


def test_update_rejects_unknown_project(root):
    r = store.create_node(root, type="note", title="Valid then invalid", project="personal")
    try:
        store.update_node(root, r["id"], {"project": "Personal"})   # capitalised typo
        assert False, "unknown project on update should be refused"
    except store.StoreError as e:
        assert "not in registry vocabulary" in str(e)


def test_new_project_via_registry(root):       # adding a world is a one-line registry edit
    try:
        store.create_node(root, type="note", title="New world node", project="testworld")
        assert False, "unknown project should be rejected before it's registered"
    except store.StoreError as e:
        assert "not in vocabulary" in str(e)
    reg = root / "schemas" / "type-registry.md"
    reg.write_text(reg.read_text().replace(
        "  projects:\n", "  projects:\n    testworld: \"a throwaway world for the e2e test.\"\n"))
    r = store.create_node(root, type="note", title="New world node", project="testworld")
    assert store.read_node(root, r["id"])["frontmatter"]["project"] == "testworld"
    _validate_clean(root)


def test_missing_project_registry_fails_closed(root):   # PROJECT_REGISTRY is load-bearing
    reg = root / "schemas" / "type-registry.md"
    reg.write_text(reg.read_text().replace("PROJECT_REGISTRY:", "PROJECT_REGISTRY_GONE:"))
    findings, *_ = validate.validate_corpus(root)   # must not crash
    assert any("PROJECT_REGISTRY" in str(f) and "missing or unparseable" in str(f)
               for f in findings)
    try:
        store.create_node(root, type="note", title="No project vocab", project="global-os")
        assert False, "write without a project registry should be refused"
    except store.StoreError as e:
        assert "project registry" in str(e)


def test_registry_body_update_project_guarded(root):   # can't brick project vocab via body edit
    reg_id = "okf_20260706_090500_0f02"
    body = ("```yaml\nREGISTRY:\n  node_types:\n    note: \"a note\"\n```\n"
            "```yaml\nPROJECT_REGISTRY:\n  projects: []\n```\n")   # projects not a mapping
    try:
        store.update_node(root, reg_id, {}, body=body)
        assert False, "a broken PROJECT_REGISTRY.projects should be refused"
    except store.StoreError as e:
        assert "PROJECT_REGISTRY" in str(e)


# ── context assembler (briefing reads) ────────────────────────────────────────
def test_brief_finds_relevant_seed_with_body(root):
    a = store.create_node(root, project="global-os", type="note", title="Widget caching strategy",
                          summary="how we cache widgets", tags=["caching", "widgets"],
                          body="Details about widget caching and eviction.")
    nodes, by_id, *_ = validate.load_corpus(root)
    r = context.assemble_context(nodes, by_id, "widget caching", root, limit=5)
    seed = next((s for s in r["seeds"] if s["id"] == a["id"]), None)
    assert seed is not None, "the on-topic node must be a seed"
    assert "widget caching" in seed["body"].lower(), "seed carries its body"


def test_brief_excludes_superseded(root):
    a = store.create_node(root, project="global-os", type="note", title="Zorptastic protocol v1",
                          tags=["zorptastic"], body="old")
    b = store.create_node(root, project="global-os", type="note", title="Zorptastic protocol v2",
                          status="active", maturity="established", tags=["zorptastic"],
                          supersedes=a["id"], relates_to=[a["id"]], body="new")
    nodes, by_id, *_ = validate.load_corpus(root)
    r = context.assemble_context(nodes, by_id, "zorptastic protocol", root, limit=5)
    seed_ids = {s["id"] for s in r["seeds"]}
    assert b["id"] in seed_ids and a["id"] not in seed_ids, "superseded node must not seed a brief"
    assert any(x["id"] == a["id"] for x in r["excluded_stale"]), "it surfaces as excluded history"


def test_brief_finds_a_word_that_appears_only_in_a_body(root):
    """The corpus's knowledge is in the bodies, and they were not indexed.

    Until 2026-08-04 retrieval read title, summary and tags only, so a word
    appearing solely in a node's prose returned ZERO hits — verified against the
    live corpus before the fix. A half-remembered phrase from inside a document
    was unfindable, which is most of what anyone actually searches for.
    """
    a = store.create_node(root, project="global-os", type="note", title="Deployment notes",
                          summary="how the box is set up",
                          body="The scheduler takes a quorbulated lock before writing.")
    nodes, by_id, *_ = validate.load_corpus(root)
    r = context.assemble_context(nodes, by_id, "quorbulated", root, limit=5)
    assert a["id"] in {s["id"] for s in r["seeds"]}, \
        "a term only in the body must be findable"


def test_brief_does_not_let_a_body_mention_outrank_the_real_subject(root):
    """Body terms are weighted BELOW title/summary/tags on purpose — otherwise a
    passing mention buries the node the subject is actually about."""
    subject = store.create_node(root, project="global-os", type="note", title="Quorbulation protocol",
                                summary="the canonical quorbulation reference",
                                tags=["quorbulation"], body="details")
    mention = store.create_node(root, project="global-os", type="note", title="Unrelated meeting notes",
                                summary="a standup",
                                body="Someone mentioned quorbulation once, in passing.")
    nodes, by_id, *_ = validate.load_corpus(root)
    seeds = context.assemble_context(nodes, by_id, "quorbulation", root, limit=5)["seeds"]
    ids = [s["id"] for s in seeds]
    assert ids[0] == subject["id"], "the node ABOUT the term must rank first"
    assert mention["id"] in ids, "but the passing mention is still findable"


def test_brief_index_cache_sees_a_same_length_body_edit(root):
    """The index is cached per corpus STATE, and the key must be the content.

    A cache keyed on node count, mtime or `updated_at` (day resolution) would
    serve a stale index after a same-day, same-length edit — an invalidation
    coarser than the thing it caches is a correctness bug, not a tuning choice.
    """
    a = store.create_node(root, project="global-os", type="note", title="Cache probe",
                          summary="pins index invalidation", body="AAAAAAAA zorptastic")
    nodes, by_id, *_ = validate.load_corpus(root)

    # This test pins the LEXICAL index, so it reads lexical evidence: on a box
    # with the embedding model installed, the semantic recall net may add seeds
    # with score_lexical == 0, and a seed-count assertion conflates the layers
    # (it did — the VPS suite failed here the first time it ran with the model).
    def lex_ids(query):
        return [s["id"] for s in context.assemble_context(
            nodes, by_id, query, root)["seeds"] if s["score_lexical"] > 0]

    assert lex_ids("zorptastic") == [a["id"]]

    # same length, different content — the case a coarse key misses
    store.update_node(root, a["id"], {}, body="AAAAAAAA quorbulate",
                      allow_body_replacement=True)
    nodes, by_id, *_ = validate.load_corpus(root)
    assert lex_ids("zorptastic") == [], "the stale term must be gone from the index"
    assert lex_ids("quorbulate") == [a["id"]], "and the new one present"


def test_load_corpus_carries_the_body(root):
    """context.py indexes Node.body, so load_corpus must actually populate it."""
    a = store.create_node(root, project="global-os", type="note", title="Body carrier",
                          summary="s " * 20, body="# H\n\nDistinctive prose here.")
    nodes, by_id, *_ = validate.load_corpus(root)
    assert "Distinctive prose here." in by_id[a["id"]].body


def test_brief_matches_a_partial_word(root):
    """A half-typed term must find the word it starts.

    The owner searched the graph for "internation" and got "Nothing matches", while a
    node titled "Bakery international launch" sat in the corpus: retrieval was
    a set intersection over whole words, so a prefix matched nothing at all
    (2026-08-03). People type prefixes constantly — the search box is used while
    still typing — so this is the common case, not an edge case.
    """
    a = store.create_node(root, project="global-os", type="note", title="Quorbulation protocol review",
                          body="how quorbulation is reviewed")
    nodes, by_id, *_ = validate.load_corpus(root)
    full = context.assemble_context(nodes, by_id, "quorbulation", root, limit=5)
    part = context.assemble_context(nodes, by_id, "quorbul", root, limit=5)
    assert a["id"] in {s["id"] for s in full["seeds"]}
    assert a["id"] in {s["id"] for s in part["seeds"]}, "a prefix must find the word"


def test_brief_prefix_never_overrides_a_real_word(root):
    """Expansion fires ONLY for terms the corpus does not know.

    "cap" is a word in its own right; it must keep meaning "cap" and not quietly
    become "cap*" and drag in every capital/capture/caption. This is the guard
    that keeps prefix matching from changing any query that already worked.
    """
    hit = store.create_node(root, project="global-os", type="note", title="Cap table snapshot",
                            tags=["cap-table"], body="ownership")
    miss = store.create_node(root, project="global-os", type="note", title="Capitalisation and capture rules",
                             body="unrelated prose")
    nodes, by_id, *_ = validate.load_corpus(root)
    r = context.assemble_context(nodes, by_id, "cap table", root, limit=10)
    lex = {s["id"]: s.get("score_lexical") for s in r["seeds"]}
    assert hit["id"] in lex
    assert not lex.get(miss["id"]), "a known word must not expand to its prefixes"


def test_brief_ignores_a_uselessly_short_prefix(root):
    """Below 4 characters a prefix means too little to EXPAND: "quo" must not
    reach "Quorbulation". Pinned on the expansion contract, not on empty
    results — the live corpus now contains the literal word "quo" (a session
    node quoting the fragment-guard story), so exact-matching it is CORRECT
    behaviour and `seeds == []` was hostage to corpus content: the
    live-data-as-fixture trap, again. Semantic admissions are also barred for
    fragment queries (the guard has its own model-free pins), so with the
    model present every seed here must be a lexical exact match."""
    a = store.create_node(root, project="global-os", type="note", title="Quorbulation protocol review")
    nodes, by_id, *_ = validate.load_corpus(root)
    pack = context.assemble_context(nodes, by_id, "quo", root)
    assert a["id"] not in {s["id"] for s in pack["seeds"]}, \
        "a sub-PREFIX_MIN prefix must never expand to longer words"
    assert all(s["score_lexical"] > 0 for s in pack["seeds"]), \
        "fragment queries admit exact lexical matches only — never semantic"


def test_brief_empty_query(root):
    nodes, by_id, *_ = validate.load_corpus(root)
    r = context.assemble_context(nodes, by_id, "   ", root)
    assert r["seeds"] == [] and r["counts"]["seeds"] == 0


# ── work_state + superseded_in_part (schema v1.6, ruled 2026-08-04) ──────────
def test_work_state_accepted_on_a_log(root):
    """The lifecycle of a piece of work as a FIELD, not a title in English —
    'OPEN WORK' in a title passed green after the work shipped (the content-audit findings, live)."""
    r = store.create_node(root, project="global-os", type="log", title="Work item fixture", summary="s " * 15)
    store.update_node(root, r["id"], {"work_state": "shipped"})
    assert store.read_node(root, r["id"])["frontmatter"]["work_state"] == "shipped"
    _validate_clean(root)


def test_work_state_refused_outside_log(root):
    """Scoped to log ON PURPOSE: sessions/notes use these words as vocabulary —
    the text-only alternative measured 80% false positives on exactly that.
    The write path refuses ERROR findings at write time, so the wrong write
    never lands (class-1 enforcement, not an audit flag).
    Measured on the author's private corpus; not reproducible from the example corpus."""
    r = store.create_node(root, project="global-os", type="note", title="Note with a gap in its title",
                          summary="s " * 15)
    with pytest.raises(store.StoreError) as e:
        store.update_node(root, r["id"], {"work_state": "open"})
    assert "log" in str(e.value)
    assert "work_state" not in store.read_node(root, r["id"])["frontmatter"]


def test_work_state_enum_enforced(root):
    r = store.create_node(root, project="global-os", type="log", title="Enum fixture", summary="s " * 15)
    with pytest.raises(store.StoreError) as e:
        store.update_node(root, r["id"], {"work_state": "in-progress"})
    # "must be one of" discriminates the enum check from the generic
    # unknown-field refusal, which also happens to contain the field name
    assert "must be one of" in str(e.value)


def test_superseded_in_part_is_a_date_not_a_link(root):
    """Chosen on measurement: of 29 live partial-supersession cases, only 6 named
    a target node — a link would have fitted a fifth of the subjects.
    Measured on the author's private corpus; not reproducible from the example corpus."""
    from datetime import date as _date
    r = store.create_node(root, project="global-os", type="decision", title="Partly corrected ruling",
                          summary="s " * 15)
    # today, not a literal: a hardcoded date trips the predates-birth guard the
    # moment midnight passes (which is exactly how this fixture first failed)
    today = _date.today().isoformat()
    store.update_node(root, r["id"], {"superseded_in_part": today})
    assert str(store.read_node(root, r["id"])["frontmatter"]["superseded_in_part"]) \
        == today
    _validate_clean(root)
    # a node id in the field is the LINK shape, deliberately rejected
    with pytest.raises(store.StoreError) as e:
        store.update_node(root, r["id"], {"superseded_in_part": "okf_20260101_000000_dead"})
    assert "not a link" in str(e.value)


def test_superseded_in_part_excluded_on_wholly_superseded(root):
    """Whole supersession subsumes partial — carrying both is not a state."""
    a = store.create_node(root, project="global-os", type="note", title="Old truth fixture", summary="s " * 15)
    store.create_node(root, project="global-os", type="note", title="New truth fixture", summary="s " * 15,
                      status="active", maturity="established", supersedes=a["id"])
    with pytest.raises(store.StoreError) as e:
        store.update_node(root, a["id"], {"superseded_in_part": "2026-08-04"})
    assert "drop the partial marker" in str(e.value)


def test_superseded_in_part_cannot_predate_birth(root):
    r = store.create_node(root, project="global-os", type="note", title="Chronology fixture", summary="s " * 15)
    with pytest.raises(store.StoreError) as e:
        store.update_node(root, r["id"], {"superseded_in_part": "2020-01-01"})
    assert "predates" in str(e.value)


# ── delete ────────────────────────────────────────────────────────────────────
def test_delete_cleans_references(root):
    a = store.create_node(root, project="global-os", type="note", title="Delete target")
    b = store.create_node(root, project="global-os", type="note", title="Points at target", relates_to=[a["id"]])
    res = store.delete_node(root, a["id"])
    assert res["deleted"] and res["cleaned_refs"] == 1
    ids = {n.id for n in validate.load_corpus(root)[0]}
    assert a["id"] not in ids, "target must be gone"
    assert store.read_node(root, b["id"])["frontmatter"].get("relates_to", []) == [], \
        "dangling ref must be cleaned"
    _validate_clean(root)


def test_delete_restores_superseded(root):
    a = store.create_node(root, project="global-os", type="note", title="Old truth")
    b = store.create_node(root, project="global-os", type="note", title="New truth", status="active",
                          maturity="established", supersedes=a["id"])
    assert store.read_node(root, a["id"])["frontmatter"]["status"] == "superseded"
    store.delete_node(root, b["id"])   # delete the successor
    fm = store.read_node(root, a["id"])["frontmatter"]
    assert fm["status"] == "active" and "superseded_by" not in fm, "predecessor restored"
    _validate_clean(root)


def test_delete_unknown_refused(root):
    try:
        store.delete_node(root, "okf_20000101_000000_dead")
        assert False
    except KeyError:
        pass


# ── git auto-commit (VPS write path) ──────────────────────────────────────────
def _git(root, *args):
    import subprocess
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True)


def _init_git(root):
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@okf.local")
    _git(root, "config", "user.name", "okf-test")
    (root / ".gitignore").write_text(".okf.lock\n")   # mirrors the real repo
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "seed")


def test_git_autocommit_when_enabled(root):
    import os
    _init_git(root)
    os.environ["OKF_GIT_COMMIT"] = "1"
    try:
        r = store.create_node(root, project="global-os", type="note", title="Committed by librarian")
    finally:
        os.environ.pop("OKF_GIT_COMMIT", None)
    assert r["committed"] is True and r["pushed"] is False
    log = _git(root, "log", "--oneline", "-1").stdout
    assert "okf: create" in log and r["id"] in log
    assert _git(root, "status", "--porcelain").stdout.strip() == ""


def test_no_autocommit_by_default(root):
    _init_git(root)
    r = store.create_node(root, project="global-os", type="note", title="Left uncommitted")
    assert r["committed"] is False
    assert _git(root, "status", "--porcelain").stdout.strip() != ""


def test_commit_excludes_unrelated_staged(root):   # P1: pathspec-scoped commit
    import os
    _init_git(root)
    (root / "unrelated.txt").write_text("staged but not the librarian's business")
    _git(root, "add", "unrelated.txt")
    os.environ["OKF_GIT_COMMIT"] = "1"
    try:
        r = store.create_node(root, project="global-os", type="note", title="Scoped commit")
    finally:
        os.environ.pop("OKF_GIT_COMMIT", None)
    assert r["committed"] is True
    committed = _git(root, "show", "--name-only", "--format=", "HEAD").stdout
    assert "unrelated.txt" not in committed, "librarian commit absorbed a stranger"
    assert _git(root, "status", "--porcelain").stdout.strip() != "", \
        "unrelated file should still be pending"


def test_dirty_tree_guard(root):               # P1: loud, not silent, on partial state
    import os
    _init_git(root)
    node = next((root / "decisions").glob("*.md"))
    node.write_text(node.read_text() + "\n<!-- interrupted mutation residue -->\n")
    os.environ["OKF_GIT_COMMIT"] = "1"
    try:
        store.create_node(root, project="global-os", type="note", title="Should refuse")
        assert False, "write over a dirty corpus should be refused"
    except store.StoreError as e:
        assert "uncommitted" in str(e)
    finally:
        os.environ.pop("OKF_GIT_COMMIT", None)


# ── durability + invariants ───────────────────────────────────────────────────
def test_no_temp_files_left(root):             # P1 atomic write
    store.create_node(root, project="global-os", type="note", title="Leaves no temp")
    leftovers = list(root.rglob("*.tmp"))
    assert not leftovers, f"temp files leaked: {leftovers}"


def test_corpus_stays_valid(root):
    store.create_node(root, project="global-os", type="note", title="Keeps corpus valid", body="x")
    _validate_clean(root)


def test_relates_to_unlink_guard(root):   # silent relates_to-drop footgun
    a = store.create_node(root, project="global-os", type="note", title="Unlink target A")
    b = store.create_node(root, project="global-os", type="note", title="Unlink target B")
    c = store.create_node(root, project="global-os", type="note", title="Unlink target C")
    n = store.create_node(root, project="global-os", type="note", title="Unlink linker",
                          relates_to=[a["id"], b["id"]])
    # adding a link (superset) is free
    store.update_node(root, n["id"], {"relates_to": [a["id"], b["id"], c["id"]]})
    assert set(store.read_node(root, n["id"])["frontmatter"]["relates_to"]) == \
        {a["id"], b["id"], c["id"]}
    # dropping a link WITHOUT allow_unlink is refused
    try:
        store.update_node(root, n["id"], {"relates_to": [a["id"]]})
        assert False, "partial relates_to that drops a link should be refused"
    except store.StoreError:
        pass
    # an unrelated update (no relates_to in changes) preserves the links
    store.update_node(root, n["id"], {"summary": "still fully linked"})
    assert set(store.read_node(root, n["id"])["frontmatter"]["relates_to"]) == \
        {a["id"], b["id"], c["id"]}
    # dropping WITH allow_unlink succeeds — a deliberate unlink
    store.update_node(root, n["id"], {"relates_to": [a["id"]]}, allow_unlink=True)
    assert store.read_node(root, n["id"])["frontmatter"]["relates_to"] == [a["id"]]


def test_purge_candidates_gates(root):   # the purge design auto-gated purge proposer
    from librarian import audit
    a = store.create_node(root, project="global-os", type="note", title="Junk stub")
    store.update_node(root, a["id"], {"status": "retired"})           # retired + orphaned + non-canon
    b = store.create_node(root, project="global-os", type="note", title="Retired but referenced")
    store.create_node(root, project="global-os", type="note", title="Keeper", relates_to=[b["id"]])
    store.update_node(root, b["id"], {"status": "retired"})           # retired but pointed-at
    c = store.create_node(root, project="global-os", type="note", title="Active node")     # active
    nodes, by_id, *_ = validate.load_corpus(root)
    cands = {d["id"] for d in audit.purge_candidates(nodes, by_id)}
    assert a["id"] in cands, "retired + orphaned + non-canonical must be a purge candidate"
    assert b["id"] not in cands, "retired-but-referenced must NOT be a candidate"
    assert c["id"] not in cands, "active must never be a candidate"


UNFILED = "## Decisions locked\n- **A ruling stated in prose** with no pointer anywhere\n"


def test_completeness_refuses_unfiled_session(root):   # the completeness-check spec write path
    try:
        store.create_node(root, project="global-os", type="session", title="Leaky session", body=UNFILED)
    except store.StoreError as e:
        assert "unfiled ruling" in str(e), e
    else:
        raise AssertionError("a session with an unfiled ruling must be refused")


def test_completeness_override_and_filed_pointer(root):
    ok = store.create_node(root, project="global-os", type="session", title="Overridden session",
                           body=UNFILED, allow_unfiled_rulings=True)
    assert ok["id"]
    assert "allow_unfiled_rulings" not in store.read_node(root, ok["id"])["frontmatter"]
    target = store.create_node(root, project="global-os", type="decision", title="The ruling itself")
    filed = store.create_node(root, project="global-os", type="session", title="Compliant session",
                              body=f"## Decisions locked\n- **The ruling** · [[{target['id']}]]\n")
    assert filed["id"]


def test_completeness_scoped_to_types_and_sections(root):
    # notes are checked too (widened 2026-07-27 after a ruling was found
    # buried in a case note); decisions and playbooks are not
    try:
        store.create_node(root, project="global-os", type="note", title="A note with an unfiled ruling", body=UNFILED)
    except store.StoreError as e:
        assert "unfiled ruling" in str(e), e
    else:
        raise AssertionError("a note with an unfiled ruling must be refused")
    assert store.create_node(root, project="global-os", type="decision", title="A decision with bullets",
                             body=UNFILED)["id"]
    assert store.create_node(root, project="global-os", type="session", title="Unwatched section only",
                             body="## What changed\n- a bullet with no id\n")["id"]


def test_completeness_legacy_flag_lets_old_sessions_be_edited(root):
    a = store.create_node(root, project="global-os", type="session", title="Old leaky session",
                          body=UNFILED, allow_unfiled_rulings=True)
    store.update_node(root, a["id"], {"unfiled_rulings_legacy": True})
    store.update_node(root, a["id"], {"summary": "edited after the rule existed"})
    b = store.create_node(root, project="global-os", type="session", title="Unflagged leaky session",
                          body=UNFILED, allow_unfiled_rulings=True)
    try:
        store.update_node(root, b["id"], {"summary": "frontmatter-only edit"})
    except store.StoreError as e:
        assert "unfiled ruling" in str(e), e
    else:
        raise AssertionError("an unflagged session must be re-checked on update")


def test_unknown_frontmatter_field_is_refused(root):
    # An unknown key is never intentional: it is a typo, or an update-INSTRUCTION
    # object passed as the changes payload and serialised verbatim into the node.
    # Observed live 2026-07-28 on a reference node (`frontmatter:` + `body_patch:`), which sat
    # green for six days while the summary update inside it never applied.
    r = store.create_node(root, project="global-os", type="note", title="A note", summary="x " * 20)
    try:
        store.update_node(root, r["id"], {"body_patch": [{"find": "a", "replace": "b"}]})
    except store.StoreError as e:
        assert "unknown frontmatter field" in str(e), e
    else:
        raise AssertionError("update_node accepted an unknown frontmatter field")


def test_unknown_field_check_covers_every_type_not_just_resource(root):
    # the check began life inside the `resource` branch; the corruption it actually
    # caught was on a reference node and a session node
    for t in ("reference", "session", "decision"):
        findings = []
        node = validate.Node(id="okf_20260101_000000_aaaa", path=Path(f"x/{t}.md"),
                             fm={"id": "okf_20260101_000000_aaaa", "type": t,
                                 "title": "T", "summary": "s " * 20,
                                 "status": "active", "maturity": "captured",
                                 "typo_field": 1})
        validate.check_node(node, {t}, findings, {}, None)
        assert any("unknown frontmatter field" in f.message for f in findings), \
            f"unknown field not caught on type={t}"


def test_in_body_dangling_ref_warns_and_opt_out_silences(root):
    # `relates_to` was always checked; an id written in PROSE never was
    r = store.create_node(root, project="global-os", type="note", title="Cites a ghost", summary="s " * 20,
                          body="See `okf_20991231_235959_ffff` for the rest.")

    def mine(root):  # the fixture copies the LIVE corpus — never assert global counts,
        nodes, by_id, *_ = validate.load_corpus(root)   # or any new real node breaks this
        return [f for f in validate.scan_body_refs(root, nodes, by_id)
                if f.path == r["path"]]

    assert any("does not exist" in f.message for f in mine(root)), mine(root)
    store.update_node(root, r["id"], {"allow_dangling_refs": True})
    assert not mine(root), "opt-out did not silence"


def test_truncated_id_warns_but_a_template_does_not(root):
    # `okf_20260728_...` is an unfinished reference; `okf_...` is a legitimate template
    bad = store.create_node(root, project="global-os", type="note", title="Truncated", summary="s " * 20,
                            body="route it by `okf_20260728_...` (the filing axis)")
    ok = store.create_node(root, project="global-os", type="note", title="Template", summary="s " * 20,
                           body="ids look like `okf_...` / `okf_<YYYYMMDD>_<HHMMSS>_<4char>`")
    nodes, by_id, *_ = validate.load_corpus(root)
    # scope to the two fixtures: the fixture copies the LIVE corpus, so a global count
    # would break as soon as any real node legitimately quotes a truncated id
    found = {f.path for f in validate.scan_body_refs(root, nodes, by_id)
             if "truncated node id" in f.message}
    assert bad["path"] in found, "truncated id not flagged"
    assert ok["path"] not in found, "a template must not be flagged"


def test_layer2_cluster_is_declared_edges_both_directions(root):
    from librarian import cluster
    a = store.create_node(root, project="global-os", type="note", title="Alpha", summary="s " * 20)
    b = store.create_node(root, project="global-os", type="note", title="Beta", summary="s " * 20,
                          relates_to=[a["id"]])                     # b -> a
    c = store.create_node(root, project="global-os", type="note", title="Gamma", summary="s " * 20)
    store.update_node(root, a["id"], {"relates_to": [c["id"]]})      # a -> c
    nodes, by_id, *_ = validate.load_corpus(root)
    ids = {x["id"] for x in cluster.topical_cluster(by_id[a["id"]].fm, a["id"], nodes, by_id)}
    assert ids == {b["id"], c["id"]}, ids   # inbound (b) AND outbound (c), never itself


def test_layer2_never_blocks_and_survives_a_broken_corpus(root):
    # advisory by ruling (2026-07-29): a Layer 2 failure must never fail a good write
    from librarian import cluster
    assert cluster.layer2({"relates_to": []}, None, [], {}) is None  # nothing to check
    r = store.create_node(root, project="global-os", type="note", title="Solo", summary="s " * 20)
    assert r["id"], "a write with no neighbours must still succeed"


def test_layer2_skips_updates_that_cannot_make_anything_false(root):
    from librarian import cluster
    assert not cluster.should_check_update({"relates_to": ["x"], "updated_at": "2026-01-01"})
    assert cluster.should_check_update({"summary": "changed"})
    assert cluster.should_check_update({"status": "superseded"})


def test_layer2_cap_holds_the_backtested_recall(root):
    # cap=12 was chosen because it preserved 12/13 recall on the 2026-07-28 ground
    # truth while cap=8 dropped it to 85%. Lock the constant against silent tuning.
    # Measured on the author's private corpus; not reproducible from the example corpus.
    from librarian import cluster
    assert cluster.DEFAULT_CAP == 12
    a = store.create_node(root, project="global-os", type="note", title="Hub", summary="s " * 20)
    others = [store.create_node(root, project="global-os", type="note", title=f"N{i}", summary="s " * 20,
                                relates_to=[a["id"]])["id"] for i in range(20)]
    nodes, by_id, *_ = validate.load_corpus(root)
    got = cluster.topical_cluster(by_id[a["id"]].fm, a["id"], nodes, by_id)
    assert len(got) == 12, f"cap not applied: {len(got)}"
    assert all(g["id"] in others for g in got)


def test_new_id_avoids_a_suffix_already_in_use(root):
    """The last-4 shorthand is only an identifier while suffixes are distinct.
    First real collision 2026-07-28: two nodes a week apart, both ending ea8a."""
    from librarian.ids import new_id
    # leave exactly one suffix free, then mint 200 ids and demand it every time
    taken = {f"{i:04x}" for i in range(0x10000)} - {"beef"}
    for _ in range(200):
        assert new_id(taken_suffixes=taken).endswith("_beef")


def test_created_nodes_do_not_reuse_a_corpus_suffix(root):
    """A create must not ADD a suffix collision.

    Deliberately not "the corpus has no duplicate suffixes": one pre-existing pair is
    grandfathered — a pair ending ea8a in the original corpus, minted before
    this guard and before the parallel session that produced the second. Ids are immutable
    (see test_id_immutable), so that pair cannot be re-minted and the corpus carries it
    permanently. What IS enforceable is that no new one appears."""
    from collections import Counter
    from librarian import validate

    def dupes(r):
        nodes, _, *_ = validate.load_corpus(r)
        return {s for s, c in Counter(n.id[-4:] for n in nodes).items() if c > 1}

    before = dupes(root)
    made = [store.create_node(root, project="global-os", type="note", title=f"S{i}", summary="s " * 20)["id"]
            for i in range(12)]
    after = dupes(root)
    assert after <= before, f"create introduced suffix collisions: {sorted(after - before)}"
    nodes, _, *_ = validate.load_corpus(root)
    assert all(m in {n.id for n in nodes} for m in made)


def test_new_id_still_returns_a_valid_id_when_suffixes_are_exhausted(root):
    """Uniqueness comes from the TIMESTAMP, not the suffix — exhaustion must degrade
    the shorthand, never produce an invalid or duplicate id."""
    from librarian.ids import new_id, is_valid_id
    every = {f"{i:04x}" for i in range(0x10000)}
    got = new_id(taken_suffixes=every)
    assert is_valid_id(got), got


def test_completeness_covers_pattern_and_playbook_but_not_decision(root):
    """Widened 2026-07-29. The trigger is the SECTION, not the type — a node writing
    '## Decisions locked' is making a pointer list whatever its type. Measured before
    widening: 0 of 23 patterns and 0 of 7 playbooks used a watched section, so this
    closes the forgotten-type hole rather than working a backlog.
    Measured on the author's private corpus; not reproducible from the example corpus.

    `decision` stays out on purpose and is pinned here: it is the only unscanned type
    that DOES use watched sections, because a decision node is where a ruling is FILED.
    Requiring it to cite another node would be circular."""
    for t_ in ("pattern", "playbook"):
        try:
            store.create_node(root, project="global-os", type=t_, title=f"A {t_} with an unfiled ruling",
                              body=UNFILED)
        except store.StoreError as e:
            assert "unfiled ruling" in str(e), e
        else:
            raise AssertionError(f"a {t_} with an unfiled ruling must be refused")
    # the deliberate gap, asserted so it stays a choice rather than becoming an oversight
    assert store.create_node(root, project="global-os", type="decision", title="A decision states its own rule",
                             body=UNFILED)["id"]


def test_tool_registry_drift(root):
    """Every @mcp.tool must carry a recorded justification, and vice versa.

    The MCP tool-design reference has a standing rule that a new tool must justify itself against the
    selection-degradation curve. It was PROSE, and three tools shipped without one —
    delete_node most starkly, since the purge design had ruled it should not be an MCP tool at all
    and the reversal was never written down. Prose is not a control (the verification-discipline guardrail), so the rule
    now fails a test instead of going unnoticed. Mirrors test_schema_code_drift."""
    import re
    reg = validate._first_yaml_block_with_key(
        (CORPUS_SRC / "references" / "okf-mcp-tool-registry-every-tool-and-the-justification-that-.md")
        .read_text(encoding="utf-8"), "TOOL_REGISTRY")
    assert reg and isinstance(reg.get("tools"), dict), "TOOL_REGISTRY block missing/broken"
    registered = set(reg["tools"])
    exposed = set(re.findall(r"@mcp\.tool\s*\ndef\s+(\w+)",
                             (REPO / "server.py").read_text(encoding="utf-8")))
    assert exposed, "sanity: found no @mcp.tool decorators"
    assert exposed == registered, (
        f"tool registry drift — only in server.py: {sorted(exposed - registered)}; "
        f"only in the registry: {sorted(registered - exposed)}")
    for name, spec in reg["tools"].items():
        assert str(spec.get("justification", "")).strip(), f"{name} has no justification"
    assert len(exposed) <= reg["count_threshold"], (
        f"{len(exposed)} tools exceeds the degradation threshold "
        f"{reg['count_threshold']} — argue what to REMOVE, not what to add")


def test_ratified_scope_warns_when_the_library_outgrows_it(root):
    """The ratified-scope node's risk acceptance was argued against a case count and drifted twice —
    5-7 -> 15 unnoticed, then 15 -> 17 within two days of being re-ratified. Nothing
    counted. WARN not ERROR: filing a case is legitimate work; what needs a human is
    the risk acceptance."""
    # state-INDEPENDENT: drive the count past the ratified figure inside the fixture.
    # An earlier version asserted "the live corpus is over the figure", which passed only
    # until the owner ratified at 17 — a test coupled to corpus state, the second time in this
    # session. Assert the BEHAVIOUR, never the current numbers.
    #
    # Third instance (2026-09-07): the rewrite above still ASSUMED the fixture starts at
    # or under the figure, and the live library has stood at 19-20 against 17 since
    # 2026-08-17 — a deferral the owner RULED (the ratified-scope node, 2026-08-18), so the standing warning is
    # correct and this test was red on every box with a corpus. The fixture now pins its
    # own ratified figure to whatever it holds, so the test starts at parity by
    # construction and only the behaviour (growing past it warns) is asserted.
    nodes, by_id, *_ = validate.load_corpus(root)
    scope_node = by_id[validate._RATIFIED_SCOPE_NODE]
    held = sum(1 for n in nodes if n.fm.get("status") == "active"
               and validate._CASE_TITLE_RE.match(str(n.fm.get("title") or "")))
    p = root / scope_node.path
    src = p.read_text(encoding="utf-8")
    patched, n_sub = re.subn(r"(?m)^(\s*cases:\s*)\d+\s*$", rf"\g<1>{held}", src, count=1)
    assert n_sub == 1, "RATIFIED_SCOPE block's `cases:` line not found in the fixture"
    # This test pins the plain rule (growth past `cases` warns); the live block may
    # carry `re_ratify_at`, which defers that — strip it here, test it below.
    # `.*` not `\d+\s*`: the live line carries a trailing comment, and a strip that
    # missed it left the key in place — the deploy gate's first real refusal, 085bdd5.
    patched = re.sub(r"(?m)^\s*re_ratify_at:.*\n", "", patched)
    p.write_text(patched, encoding="utf-8")
    nodes, by_id, *_ = validate.load_corpus(root)
    assert not validate.scan_ratified_scope(root, nodes, by_id), \
        f"fixture pinned at {held} must start at parity"
    for i in range(3):
        store.create_node(root, project="global-os", type="note", title=f"Case Z{i} — synthetic overflow",
                          summary="s " * 20)
    nodes, by_id, *_ = validate.load_corpus(root)
    found = validate.scan_ratified_scope(root, nodes, by_id)
    assert found, "growing past the ratified figure must warn"
    assert "ratified at" in found[0].message
    assert found[0].severity == validate.SEVERITY_WARN, "must never refuse a write"


def test_ratified_scope_re_ratify_at_defers_the_warning_until_it_is_reached(root):
    """The owner, 2026-09-08: "set the next trigger at 35". A trigger the validator does
    not read is prose, and this node's prose trigger ("re-ratify at 25") was reached
    without anything firing. With `re_ratify_at` in the block, growth past `cases`
    is silent until the count REACHES it; at it, the warning names both figures."""
    nodes, by_id, *_ = validate.load_corpus(root)
    scope_node = by_id[validate._RATIFIED_SCOPE_NODE]
    held = sum(1 for n in nodes if n.fm.get("status") == "active"
               and validate._CASE_TITLE_RE.match(str(n.fm.get("title") or "")))
    p = root / scope_node.path
    src = re.sub(r"(?m)^\s*re_ratify_at:.*\n", "", p.read_text(encoding="utf-8"))
    src, n = re.subn(r"(?m)^(\s*cases:\s*)\d+\s*$",
                     rf"\g<1>{held}\n  re_ratify_at: {held + 2}", src, count=1)
    assert n == 1
    p.write_text(src, encoding="utf-8")
    store.create_node(root, project="bakery", type="note", title="Case Z0 — one past ratified",
                      summary="s " * 20)
    nodes, by_id, *_ = validate.load_corpus(root)
    assert not validate.scan_ratified_scope(root, nodes, by_id), \
        "one past `cases` but under `re_ratify_at` must stay silent"
    store.create_node(root, project="bakery", type="note", title="Case Z1 — reaches the trigger",
                      summary="s " * 20)
    nodes, by_id, *_ = validate.load_corpus(root)
    found = validate.scan_ratified_scope(root, nodes, by_id)
    assert found and f"due at {held + 2}" in found[0].message and f"ratified at {held}" in found[0].message
    assert found[0].severity == validate.SEVERITY_WARN


def test_ratified_scope_counts_the_unlettered_case_form(root):
    """Case letters were retired 2026-09-03; new cases are "Case — <name>". The
    counter saw only "Case X" and reported 20 while the library held 25 — caught
    on 2026-09-08 with a re-ratification about to be written against the wrong
    figure. An undercounting guard is worse than none (the regex's own comment)."""
    nodes, by_id, *_ = validate.load_corpus(root)
    before = sum(1 for n in nodes if n.fm.get("status") == "active"
                 and validate._CASE_TITLE_RE.match(str(n.fm.get("title") or "")))
    store.create_node(root, project="bakery", type="note",
                      title="Case — Synthetic Person: the unlettered form", summary="s " * 20)
    store.create_node(root, project="bakery", type="note",
                      title="Case-study notes that are not a case", summary="s " * 20)
    nodes, by_id, *_ = validate.load_corpus(root)
    after = sum(1 for n in nodes if n.fm.get("status") == "active"
                and validate._CASE_TITLE_RE.match(str(n.fm.get("title") or "")))
    assert after == before + 1, "the unlettered form must count once; a hyphenated non-case never"


def test_ratified_scope_fails_loud_when_its_own_block_is_broken(root):
    """The fail-loudly rule: a check that cannot verify must FAIL, never pass quietly. A missing or
    unparseable RATIFIED_SCOPE block means the guard stopped guarding — that is an
    unverified state, not a passing one."""
    nodes, by_id, *_ = validate.load_corpus(root)
    node = by_id[validate._RATIFIED_SCOPE_NODE]
    p = root / node.path
    p.write_text(p.read_text(encoding="utf-8").replace("RATIFIED_SCOPE:", "SCOPE_GONE:"),
                 encoding="utf-8")
    nodes, by_id, *_ = validate.load_corpus(root)
    found = validate.scan_ratified_scope(root, nodes, by_id)
    assert found and "NOT passing" in found[0].message, \
        "a broken scope block must report unverified, not silently pass"


def test_record_body_is_append_only_and_facts_are_frozen(root):
    """`record` exists FOR immutability and nothing enforced it until 2026-08-01 — it was
    prose inside a document the validator otherwise reads live, which is why it read as
    binding. Append-only rather than byte-frozen so a record can still carry the corpus's
    standing dated-correction remedy (the content-audit decision)."""
    r = store.create_node(root, project="global-os", type="record", title="Snapshot as of 2026-01-01",
                          summary="A dated snapshot for the immutability test. " * 3,
                          body="# Snapshot\n\nOpening balance 100.\n")
    body = store.read_node(root, r["id"])["body"]

    store.update_node(root, r["id"], {}, body=body + "\n> Note appended.\n")   # allowed

    for label, kw in (
        ("body rewrite", dict(body=body.replace("100", "200"))),
        ("summary change", dict(changes={"summary": "a different claim " * 6})),
        ("title change", dict(changes={"title": "Snapshot as of 2026-02-02"})),
    ):
        try:
            store.update_node(root, r["id"], kw.get("changes", {}), kw.get("body"))
        except store.StoreError as e:
            assert "record" in str(e), (label, e)
        else:
            raise AssertionError(f"a record must refuse a {label}")


def test_record_can_still_be_linked_and_superseded(root):
    """The discipline is 'a new snapshot supersedes the old', so that path must WORK.
    A record that cannot gain a backlink or be superseded is unusable, and a guard that
    blocked those would be overridden by reflex."""
    old = store.create_node(root, project="global-os", type="record", title="Snapshot A",
                            summary="First dated snapshot for the supersession test. " * 2,
                            body="# A\n\nvalue 1\n")
    other = store.create_node(root, project="global-os", type="note", title="Cites the snapshot", summary="s " * 20)
    fm = store.read_node(root, old["id"])["frontmatter"]
    store.update_node(root, old["id"],
                      {"relates_to": sorted(set(list(fm.get("relates_to") or []) + [other["id"]]))})
    new = store.create_node(root, project="global-os", type="record", title="Snapshot B",
                            summary="Second dated snapshot, superseding the first. " * 2,
                            supersedes=old["id"], body="# B\n\nvalue 2\n")
    after = store.read_node(root, old["id"])["frontmatter"]
    assert after["status"] == "superseded" and after["superseded_by"] == new["id"]


def test_record_amendment_override_is_available_and_lands_in_the_record(root):
    r = store.create_node(root, project="global-os", type="record", title="Snapshot with an amendment",
                          summary="A dated snapshot used for the override test. " * 3,
                          body="# S\n\noriginal line\n")
    body = store.read_node(root, r["id"])["body"]
    store.update_node(root, r["id"], {}, body=body.replace("original", "amended"),
                      allow_record_amendment=True)
    assert "amended" in store.read_node(root, r["id"])["body"]


def test_gdrive_pointer_id_cannot_change(root):
    """node.schema §2: "update the file in place, never delete-and-reupload — a new ID
    breaks the pointer." Prose until 2026-08-01. Safe to enforce strictly because a Drive
    file ID SURVIVES a move (verified 2026-07-28), so reorganising never changes one — a
    changed ID means a different file, which is the violation."""
    r = store.create_node(root, project="global-os", type="resource", title="A Drive-backed artifact",
                          resource={"kind": "gdrive_file",
                                    "uri": "https://drive.google.com/file/d/" + "A"*33 + "/view"})
    same_id_new_path = {"kind": "gdrive_file",
                        "uri": "https://drive.google.com/file/d/" + "A"*33 + "/edit"}
    store.update_node(root, r["id"], {"resource": same_id_new_path})      # allowed

    changed = {"kind": "gdrive_file",
               "uri": "https://drive.google.com/file/d/" + "B"*33 + "/view"}
    try:
        store.update_node(root, r["id"], {"resource": changed})
    except store.StoreError as e:
        assert "file ID would change" in str(e), e
    else:
        raise AssertionError("a changed Drive file ID must be refused")

    store.update_node(root, r["id"], {"resource": changed}, allow_repoint=True)  # override


def test_layer3_fires_only_for_rule_bearing_types(root):
    """Layer 2 asks what a write makes false; layer 3 asks whether it already exists
    (the redundancy-check handover). A session or note RECORDS; a decision/pattern/playbook ASSERTS, so only the
    latter can be redundant with a governing document."""
    from librarian import redundancy
    nodes, _, *_ = validate.load_corpus(root)
    base = {"project": "global-os", "title": "Some rule about the librarian write path",
            "summary": "A rule about validation and the librarian. " * 3,
            "tags": ["librarian", "okf", "second-brain"]}
    for t_ in ("decision", "pattern", "playbook"):
        assert redundancy.layer3({**base, "type": t_}, None, nodes), f"{t_} must be checked"
    for t_ in ("session", "note", "log", "record", "reference", "resource"):
        assert redundancy.layer3({**base, "type": t_}, None, nodes) is None, \
            f"{t_} must stay silent"


def test_layer3_returns_governing_docs_and_never_blocks(root):
    from librarian import redundancy
    nodes, _, *_ = validate.load_corpus(root)
    b = redundancy.layer3({"type": "decision", "project": "global-os",
                           "title": "A rule about node validation",
                           "summary": "Concerns the librarian, validation and the corpus. " * 2,
                           "tags": ["librarian", "okf"]}, None, nodes)
    assert b["blocking"] is False, "layer 3 must never block a write"
    docs = b["governing_documents"]
    assert docs, "expected at least one governing document"
    for d in docs:                      # only canonical docs or playbooks qualify
        n = next(x for x in nodes if x.id == d["id"])
        assert n.fm.get("canonical") is True or n.fm.get("type") == "playbook"
        assert n.fm.get("project") == "global-os"
    assert "false COVERED loses a rule permanently" in b["question"]
    assert "not proof the rule is new" in b["question"]      # states its own scope limit


def test_layer3_is_silent_when_nothing_governs_the_domain(root):
    """A project with no canonical documents has nothing for a new rule to duplicate —
    silence is the correct output, not an empty warning."""
    from librarian import redundancy
    nodes, _, *_ = validate.load_corpus(root)
    assert redundancy.layer3({"type": "decision", "project": "garden",
                              "title": "A first rule for a domain with no canon",
                              "summary": "Nothing governs this domain yet at all. " * 3,
                              "tags": ["zzz-no-such-tag"]}, None, nodes) is None


def test_layer3_reports_truncation_rather_than_hiding_it(root):
    from librarian import redundancy
    nodes, _, *_ = validate.load_corpus(root)
    fm = {"type": "decision", "project": "global-os", "title": "Broad rule",
          "summary": "Touches the librarian, okf, corpus and second brain. " * 2,
          "tags": ["librarian", "okf", "second-brain", "corpus-integrity"]}
    docs, truncated = redundancy.governing_docs(fm, None, nodes, cap=2)
    assert len(docs) == 2
    if truncated:
        assert "were not shown" in redundancy.layer3(fm, None, nodes, cap=2)["question"]


# ── runner ────────────────────────────────────────────────────────────────────
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    print("=== store write-path tests ===")
    for name, fn in tests:
        root = fresh_corpus()
        try:
            fn(root)
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {(str(e).splitlines() or [''])[0][:90]}")
            failed += 1
        finally:
            shutil.rmtree(root, ignore_errors=True)
    print(f"\n{'ALL PASS' if not failed else str(failed) + ' FAILED'} — {passed + failed} tests")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())


# ── task substrate: Coda kinds retired 2026-08-03 (the tasks-live-in-the-task-app decision) ──
def test_coda_resource_kinds_are_no_longer_mintable(root):
    """The owner ruled Coda "is not the fallback and not the archive". The enum
    enforces that rather than documenting it — a Coda task pointer cannot be
    created at all. Safe because no node carries one; the first task-substrate decision mentions the kinds
    in prose only, and prose is not validated."""
    for dead in ("coda_table", "coda_row"):
        try:
            store.create_node(root, project="global-os", type="resource", title=f"Dead {dead} pointer",
                              summary="A pointer to the retired Coda task table, "
                                      "which the schema must now refuse outright.",
                              resource={"kind": dead, "uri": "coda://docs/x/tables/y"})
            assert False, f"{dead} should no longer be a valid resource kind"
        except store.StoreError as e:
            assert "resource.kind" in str(e)


def test_todoist_resource_kinds_replace_them(root):
    """The decision keeps the knowledge-not-live-data decision's shape — tasks are live data, OKF holds a
    pointer — so the capability has to exist on the new substrate."""
    for live in ("todoist_project", "todoist_task"):
        r = store.create_node(root, project="global-os", type="resource", title=f"Bakery {live}",
                              summary="A pointer to live task data in Todoist, "
                                      "which is canonical for tasks since 2026-08-03.",
                              resource={"kind": live, "uri": f"https://app.todoist.com/{live}"})
        assert store.read_node(root, r["id"])["frontmatter"]["resource"]["kind"] == live


# ── create_node surfaces the new node's own warnings (2026-08-05) ─────────────
def test_create_surfaces_own_warnings(root):
    """A node created with no summary is 'structurally under-retrievable' — the
    audit always said so, but the create response threw the warning away, and the
    miss happened twice in one day (the full-body-retrieval acceptance, a second note), caught only by read-back. The
    write receipt now carries the candidate's own WARN findings."""
    r = store.create_node(root, project="global-os", type="note", title="Warning surfacing fixture")
    assert any("summary" in w for w in r["warnings"]), \
        f"empty-summary warning must reach the writer at write time, got {r['warnings']}"


def test_create_with_good_summary_stays_quiet(root):
    """The receipt must not become noise — a clean create carries no warnings
    (the fail-loudly rule §3: a check that cries wolf gets ignored)."""
    r = store.create_node(root, project="global-os", type="note", title="Quiet fixture",
                          summary="a perfectly adequate summary of reasonable length "
                                  "that says what the node is about")
    assert r["warnings"] == []


def test_create_warnings_are_scoped_to_the_new_node(root):
    """Corpus-wide warnings are the audit's job; the receipt reports only the
    candidate's own findings."""
    store.create_node(root, project="global-os", type="note", title="Pre-existing thin node")  # warns, but not ours
    r = store.create_node(root, project="global-os", type="note", title="Scoped fixture",
                          summary="a perfectly adequate summary of reasonable length "
                                  "that says what this node is about")
    assert r["warnings"] == [], "another node's warning must not leak into this receipt"


# ── the summary is the current state, and moves with the body (2026-10-22) ────
_BULLETS = ("- A fixture note about one thing, for the summary receipts.\n"
            "- The point it makes → § Detail")
_BODY = "## Detail\nThe point, at length.\n"


def test_create_with_a_well_formed_bullet_summary_is_quiet(root):
    r = store.create_node(root, project="global-os", type="note", title="Bullet fixture",
                          summary=_BULLETS, body=_BODY)
    assert r["warnings"] == [], r["warnings"]


def test_create_surfaces_a_bullet_pointing_nowhere(root):
    r = store.create_node(root, project="global-os", type="note", title="Bad pointer fixture",
                          summary=_BULLETS.replace("§ Detail", "§ Elsewhere"), body=_BODY)
    assert any("points to no heading" in w for w in r["warnings"]), r["warnings"]


def test_a_body_write_without_the_summary_warns(root):
    r = store.create_node(root, project="global-os", type="note", title="Drift fixture",
                          summary=_BULLETS, body=_BODY)
    u = store.update_node(root, r["id"], {}, body_append="A later correction.")
    assert any("the body changed but the summary did not" in w for w in u.get("warnings", [])), u


def test_a_body_write_with_the_summary_is_quiet(root):
    r = store.create_node(root, project="global-os", type="note", title="In step fixture",
                          summary=_BULLETS, body=_BODY)
    new = _BULLETS.replace("The point it makes", "The point it now makes (changed 2026-10-22: was the old point)")
    u = store.update_node(root, r["id"], {"summary": new}, body_append="A later correction.")
    assert "warnings" not in u, u


def test_a_log_grows_without_a_summary_warning(root):
    r = store.create_node(root, project="global-os", type="log", title="Log fixture",
                          summary="a running log that grows by appended blocks over the weeks it is kept",
                          body=_BODY)
    u = store.update_node(root, r["id"], {}, body_append="Another run.")
    assert "warnings" not in u, u
