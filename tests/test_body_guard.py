"""Fixtures for the body-replacement guard (librarian.store, 2026-08-02).

THE BUG (the body-replace incident, observed live): `update_node`'s `body`
parameter REPLACES. A caller appending a correction block passed only the block,
and a 2,400-word node was destroyed — silently, because the success payload
reports `committed: true` identically whether you added a paragraph or deleted
everything. It was recovered only because an earlier read_node happened to still
be in context.

THE FIX, and why it is not the one the pattern preferred. That node proposed a
blanket `allow_body_replacement` on every body write. Rejected: EVERY body write
is a replacement (there is no append path), so the flag would be required 100% of
the time — and a flag that is always passed is not a control, which is
The gated-deletion decision's own closing principle. The guard instead refuses only
a DESTRUCTIVE replacement, exactly as allow_unlink refuses only a link-DROPPING
change. It must therefore be silent on ordinary work, and these fixtures pin both
halves.

Run: python3 -m pytest tests/test_body_guard.py
"""
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _corpus import skip_if_no_corpus                     # noqa: E402
from librarian import store                               # noqa: E402
from librarian.store import StoreError                    # noqa: E402

CORPUS_SRC = skip_if_no_corpus()

from test_store import fresh_corpus, _validate_clean      # noqa: E402


LONG = "\n\n".join(f"## Section {i}\n\n" + ("body text here. " * 40) for i in range(8))
BLOCK = "## APPEND\n\n---\n\n## RETIRED — 2026-08-02\n**Original text retained above.**"
# A deliberate replacement that is NOT append-headed. Needed since 2026-08-10:
# an APPEND-headed body is refused for every caller, so it can no longer stand in
# for "some other body" in tests about the override or the reported delta.
REWRITE = "## RETIRED — 2026-08-02\n**This node was rewritten in full, on purpose.**"


@pytest.fixture
def node(tmp_path):
    """A node with a substantial body, in a throwaway corpus."""
    root = fresh_corpus()
    res = store.create_node(root, type="note", title="Body guard fixture node",
                            summary="A node with a long body, used to pin the "
                                    "body-replacement guard against real loss.",
                            project="global-os", body=LONG)
    return root, res["id"]


# ── the bug, reproduced: it must now be REFUSED ───────────────────────────────
def test_the_real_2026_08_02_loss_is_refused(node):
    """The exact call that destroyed a model note: a correction block passed alone."""
    root, nid = node
    with pytest.raises(StoreError) as e:
        store.update_node(root, nid, {"status": "retired"}, body=BLOCK)
    msg = str(e.value)
    assert "REPLACES" in msg, "the refusal must state the semantics"
    assert "body_append" in msg, "and must name the tool that does what was meant"
    assert "allow_body_replacement" in msg, "and the deliberate override"


def test_the_body_survives_a_refused_write(node):
    """A refusal that still wrote would be worse than no guard."""
    root, nid = node
    with pytest.raises(StoreError):
        store.update_node(root, nid, {}, body=BLOCK)
    assert store.read_node(root, nid)["body"].strip() == LONG.strip()


# ── it must be SILENT on ordinary work, or it will be reflexively overridden ──
def test_appending_the_full_body_passes_untouched(node):
    """The safe pattern: read, concatenate, write. No flag, no friction."""
    root, nid = node
    full = LONG + "\n\n" + BLOCK
    out = store.update_node(root, nid, {}, body=full)
    assert out["id"] == nid
    assert store.read_node(root, nid)["body"].strip().endswith("retained above.**")


def test_a_much_shorter_body_passes_if_it_CONTAINS_the_original(node):
    """Containment is the append case and is allowed at any size ratio."""
    root, nid = node
    out = store.update_node(root, nid, {}, body=LONG + "\n\nx")
    assert out["id"] == nid


def test_a_modest_rewrite_now_NEEDS_the_flag(node):
    """The 51% hole, closed 2026-08-04.

    The original floor allowed any replacement keeping >=50% of the length, so
    overwriting a 1000-character body with 510 characters of entirely different
    text passed silently — measured, not theorised. That compromise existed only
    because `body` carried both intents; now that `body_append` carries the append
    intent, every use of `body` means "replace", and the floor is zero.
    """
    root, nid = node
    kept = LONG[:int(len(LONG) * 0.75)]          # 75% — comfortably past the old floor
    with pytest.raises(StoreError):
        store.update_node(root, nid, {}, body=kept)
    assert store.read_node(root, nid)["body"].strip() == LONG.strip()
    store.update_node(root, nid, {}, body=kept, allow_body_replacement=True)


def test_an_equally_long_rewrite_is_refused_too(node):
    """The guard is about CONTAINMENT, not size — a same-length swap loses just
    as much as a shrinking one, and the old length-based floor never saw it."""
    root, nid = node
    with pytest.raises(StoreError) as e:
        store.update_node(root, nid, {}, body="Z" * len(LONG))
    assert "REWRITE" in str(e.value), "the message must not claim a DROP"


def test_frontmatter_only_updates_are_unaffected(node):
    """The overwhelmingly common write must not touch this path at all."""
    root, nid = node
    out = store.update_node(root, nid, {"tags": ["a", "b"]})
    assert out["id"] == nid and "body_bytes" not in out


# ── the override, and the diff that makes even an allowed loss visible ────────
# AMENDED 2026-08-10: these two used BLOCK — which is APPEND-headed — merely as a
# convenient short replacement. That shape is now refused outright, flag or no
# flag, so they use REWRITE instead. What they pin is unchanged: the override
# still works, and the delta is still reported.
def test_allow_body_replacement_permits_a_deliberate_rewrite(node):
    root, nid = node
    out = store.update_node(root, nid, {}, body=REWRITE, allow_body_replacement=True)
    assert out["id"] == nid
    assert store.read_node(root, nid)["body"].strip() == REWRITE.strip()


def test_body_bytes_reports_the_delta(node):
    """The 2026-08-02 loss was silent because committed:true looked the same
    whether a paragraph was added or 2,400 words deleted."""
    root, nid = node
    out = store.update_node(root, nid, {}, body=REWRITE, allow_body_replacement=True)
    b = out["body_bytes"]
    assert b["before"] == len(LONG.strip())
    assert b["after"] == len(REWRITE.strip())
    assert b["delta"] == b["after"] - b["before"] < 0, "a loss must show as negative"


def test_growth_also_reported(node):
    root, nid = node
    out = store.update_node(root, nid, {}, body=LONG + "\n\n" + BLOCK)
    assert out["body_bytes"]["delta"] > 0


# ── body_append: the intent `body` was being asked to carry ──────────────────
# Built 2026-08-04. The point is not convenience: separating the two intents is
# what let the replacement floor drop to zero without firing on ordinary work.
def test_body_append_keeps_everything_and_adds_the_block(node):
    root, nid = node
    out = store.update_node(root, nid, {}, body_append="A correction, filed later.")
    after = store.read_node(root, nid)["body"]
    assert LONG.strip() in after, "the original body must survive verbatim"
    assert after.rstrip().endswith("A correction, filed later.")
    assert out["body_bytes"]["delta"] > 0


def test_body_append_stamps_a_dated_heading(node):
    """The convention moves out of prose and into the tool — this whole defect
    began with '## APPEND' existing only as something people remembered."""
    root, nid = node
    store.update_node(root, nid, {}, body_append="text")
    after = store.read_node(root, nid)["body"]
    assert f"## APPEND ({date.today().isoformat()})" in after


def test_body_append_does_not_double_the_heading(node):
    """A caller that writes its own '## APPEND' heading gets one, not two."""
    root, nid = node
    store.update_node(root, nid, {}, body_append="## APPEND (2026-01-01)\n\nmine")
    assert store.read_node(root, nid)["body"].count("## APPEND") == 1


def test_body_append_needs_no_flag_at_any_size(node):
    """It cannot lose data by construction, so it must never ask for permission —
    a guard on the safe path is what trains people to override the real one."""
    root, nid = node
    store.update_node(root, nid, {}, body_append="x")
    assert LONG.strip() in store.read_node(root, nid)["body"]


def test_body_and_body_append_together_is_refused(node):
    """Opposite intents. Guessing which wins is how the 2026-08-02 loss happened."""
    root, nid = node
    with pytest.raises(StoreError) as e:
        store.update_node(root, nid, {}, body="replacement", body_append="addition")
    assert "not both" in str(e.value)
    assert store.read_node(root, nid)["body"].strip() == LONG.strip()


def test_empty_body_append_is_refused(node):
    root, nid = node
    with pytest.raises(StoreError):
        store.update_node(root, nid, {}, body_append="   ")


def test_body_append_satisfies_record_immutability(node):
    """A `record` body is APPEND-ONLY, which is exactly what this parameter does —
    so the safe path must not collide with the immutability rule."""
    root = fresh_corpus()
    rec = store.create_node(root, type="record", title="Dated snapshot fixture",
                            summary="A record used to pin append-only behaviour.",
                            project="global-os", body="Original recorded facts.")
    store.update_node(root, rec["id"], {}, body_append="A later observation.")
    after = store.read_node(root, rec["id"])["body"]
    assert "Original recorded facts." in after and "A later observation." in after


# ── the corpus must still be valid afterwards ────────────────────────────────
def test_corpus_stays_clean_through_a_guarded_write(node):
    root, nid = node
    store.update_node(root, nid, {}, body=LONG + "\n\n" + BLOCK)
    _validate_clean(root)


def test_corpus_stays_clean_after_an_append(node):
    root, nid = node
    store.update_node(root, nid, {}, body_append="Appended, then validated.")
    _validate_clean(root)


# ── the APPEND-headed replacement: the July 2026 corpus loss ─────────────────
# Built 2026-08-10 (the owner ruled it). Seven nodes lost ~119,000 characters between
# 2026-07-24 and 2026-07-29 because a caller hand-wrote a '## APPEND' heading and
# passed it as `body`. The containment guard above already refuses that shape —
# but it names allow_body_replacement as the way through, and a caller who thinks
# the heading makes `body` append will set the flag and destroy the node anyway.
# Reaching for the flag IS the error, so the flag cannot be the control.
def test_append_headed_replacement_is_refused_even_WITH_the_flag(node):
    """The hole the 2026-08-04 guard left open, and the whole point of this one."""
    root, nid = node
    with pytest.raises(StoreError) as e:
        store.update_node(root, nid, {}, body=BLOCK, allow_body_replacement=True)
    msg = str(e.value)
    assert "body_append" in msg, "must name the tool that does what was meant"
    assert "allow_body_replacement does NOT override" in msg, \
        "must say plainly that the usual escape hatch is closed here"


def test_the_body_survives_an_append_headed_refusal(node):
    """A refusal that still wrote would be the bug wearing a warning label."""
    root, nid = node
    with pytest.raises(StoreError):
        store.update_node(root, nid, {}, body=BLOCK, allow_body_replacement=True)
    assert store.read_node(root, nid)["body"].strip() == LONG.strip()


def test_a_genuine_restoration_carrying_append_blocks_is_ALLOWED(node):
    """The 2026-08-10 recovery shape: original body + the append blocks written
    after it. It CONTAINS '## APPEND' but does not START with it, and refusing it
    would block the very repair this guard exists to make unnecessary."""
    root, nid = node
    out = store.update_node(root, nid, {}, body=LONG + "\n\n" + BLOCK)
    assert out["id"] == nid
    assert out["body_bytes"]["delta"] > 0


def test_body_append_is_untouched_even_when_it_produces_an_append_headed_body(node):
    """The edge case that would break the safe path: appending a self-headed block
    to an EMPTY body yields a body starting with '## APPEND'. The guard must not
    fire — it governs `body` (replace), never `body_append`."""
    root = fresh_corpus()
    res = store.create_node(root, type="note", title="Empty body append fixture",
                            summary="A node with no body, used to pin that the "
                                    "append-headed guard never fires on the safe path.",
                            project="global-os", body="")
    store.update_node(root, res["id"], {}, body_append="## APPEND (2026-01-01)\n\nmine")
    after = store.read_node(root, res["id"])["body"]
    assert after.lstrip().startswith("## APPEND"), "this is the shape being pinned"
    assert "mine" in after


# ── the commit message must describe the DAMAGE, not the declared fields ──────
# Every destroying write of July 2026 logged as `okf: update <id> (relates_to,
# summary)` while replacing the whole body. git log described a metadata edit, so
# the loss survived 309 later commits unnoticed.
def _captured_message(monkeypatch):
    seen = {}

    def fake(root, rel_paths, message):
        seen["msg"] = message
        return {"committed": True, "pushed": None, "push": "queued"}

    monkeypatch.setattr(store, "_git_commit", fake)
    return seen


def test_commit_message_names_the_body_delta(node, monkeypatch):
    root, nid = node
    seen = _captured_message(monkeypatch)
    store.update_node(root, nid, {}, body=LONG + "\n\n" + BLOCK)
    assert "body +" in seen["msg"], f"growth must be visible: {seen['msg']}"


def test_commit_message_shows_a_LOSS_as_negative(node, monkeypatch):
    """The case that mattered: a big deletion must be legible in `git log`."""
    root, nid = node
    seen = _captured_message(monkeypatch)
    store.update_node(root, nid, {"status": "retired"},
                      body="a deliberately much shorter body",
                      allow_body_replacement=True)
    assert "body -" in seen["msg"], f"a loss must show as negative: {seen['msg']}"
    assert "status" in seen["msg"], "declared fields are still reported"


def test_frontmatter_only_commit_message_has_no_body_token(node, monkeypatch):
    """The common write must not gain noise, or the token stops being a signal."""
    root, nid = node
    seen = _captured_message(monkeypatch)
    store.update_node(root, nid, {"tags": ["a", "b"]})
    assert "body" not in seen["msg"] and "tags" in seen["msg"]
