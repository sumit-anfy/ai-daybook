"""Fixtures for the unfiled-ruling detector (the completeness-check spec §8).

Positives are the real sessions remediated on 2026-07-27; the discriminator negative
is the pre-remediation text of a real session, taken from git (commit b09f6bf) — three bullets
under one heading, one filed, two not. It must produce exactly 2 violations, proving
the check separates WITHIN a section rather than flagging the whole section.

Run: python3 tests/test_completeness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian.completeness import check_completeness, find_unfiled_rulings  # noqa: E402

SESSION = {"type": "session"}


# ── 1-2. real remediated sessions — every bullet carries an ID ────────────────
REMEDIATED_84F3 = """
## Decisions locked

Pointer list — each ruling is stated in full in its own node.

- **Pending-not-done for financial records** — unprocessed records stay explicit PENDING states, never marked done or silently skipped · [[okf_20250110_090000_c147]]
- **Guardrails gate on raw character length, not a token estimate** · [[okf_20250110_090100_98f3]]
- **Progress is read from the NDJSON, not the store alone** · [[okf_20250110_090200_5490]]

## What changed / was built
- something with no id at all, in an unwatched section
"""


def test_remediated_session_passes():
    assert find_unfiled_rulings(REMEDIATED_84F3) == []


def test_unwatched_sections_are_out_of_scope():
    """Scope containment (§3): only the two named sections are checked. The rest of a
    session is prose and must never be flagged."""
    body = """
## What changed / was built
- a bullet with no id
- another one

## Failures / corrections (recorded honestly)
- **Called the run "hung"** when the store froze — no id here either
- reinforces [[guardrail-verify-idempotency-safety-dedup-claims-empirically]]
"""
    assert find_unfiled_rulings(body) == []


# ── 3. no watched section at all → pass (most sessions produce no rulings) ────
def test_session_without_decisions_section_passes():
    assert check_completeness(SESSION, "# Session\n\nJust prose.\n\n## Purpose\n- did a thing") == []


# ── 4. every link form counts — bare, backticked, wiki ───────────────────────
def test_all_id_forms_match():
    body = """
## Decisions locked
- bare form okf_20250110_090000_c147
- backticked form `okf_20250110_090100_98f3`
- wiki form [[okf_20250110_090200_5490]]
"""
    assert find_unfiled_rulings(body) == []


# ── 5. sub-bullets evaluate with their parent ────────────────────────────────
def test_sub_bullets_fold_into_filed_parent():
    """One filed ruling with three explanatory sub-points is ONE filed ruling, not
    three violations."""
    body = """
## Decisions locked
- **The ruling** · [[okf_20250110_090000_c147]]
  - explanatory detail with no id
  - second detail
    - deeper detail
"""
    assert find_unfiled_rulings(body) == []


def test_continuation_line_carries_the_id():
    """An ID on line 3 of a 3-line bullet still files it (§4)."""
    body = """
## Decisions locked
- **A ruling stated across several lines** — the reasoning runs on
  and continues here, with the pointer landing last:
  [[okf_20250110_090000_c147]]
"""
    assert find_unfiled_rulings(body) == []


# ── 6. THE DISCRIMINATOR — pre-remediation a real session, verbatim from commit b09f6bf ─
PRE_REMEDIATION_84F3 = """## Decisions locked
- **Pending-not-done for financial records** → `okf_20250110_090000_c147` (its own node;
  The owner corrected a data-integrity slip of mine).
- **Guardrail gates on raw character length, not a token estimate** — `len//4`
  under-counts newline/number-dense financial text ~3× so a token-est gate let the giant
  TRX-* trade files (170k–660k chars) slip. Chars cleanly separate normal (<50k) from
  giant. (Got this wrong twice before switching to chars.)
- **Progress must be read from the NDJSON, not the store alone** — the store holds only
  successful extractions; errors/too-large go to NDJSON only, so a store that looks
  "stalled" can actually be fast-erroring. The NDJSON is the true throughput signal.

## Failures / corrections (recorded honestly)
- **Marked unprocessed records "done"** to save a re-check → reverted on the owner's rule.
"""


def test_discriminator_separates_within_a_section():
    found = find_unfiled_rulings(PRE_REMEDIATION_84F3)
    assert len(found) == 2, f"expected exactly 2 unfiled, got {len(found)}: {[f.text for f in found]}"
    assert all(f.section == "decisions locked" for f in found)
    assert found[0].text.startswith("**Guardrail gates on raw character length")
    assert found[1].text.startswith("**Progress must be read from the NDJSON")


# ── section-matching tolerance (§3) ──────────────────────────────────────────
def test_heading_variants_are_matched():
    for heading in ("## Decisions Locked", "### decisions locked",
                    "## Principles surfaced", "## Rulings", "##   Decisions locked  "):
        body = f"{heading}\n- an unfiled ruling with no pointer\n"
        assert len(find_unfiled_rulings(body)) == 1, f"heading not matched: {heading!r}"


def test_section_ends_at_equal_or_shallower_heading():
    body = """
## Decisions locked
- filed · [[okf_20250110_090000_c147]]

### A deeper subsection still inside
- also unfiled here

## Next actions
- this one is outside and must not count
"""
    found = find_unfiled_rulings(body)
    assert len(found) == 1, [f.text for f in found]
    assert found[0].text.startswith("also unfiled")


# ── type scoping — only sessions are checked ─────────────────────────────────
def test_only_checked_types_are_scanned():
    """Any type that writes a POINTER-LIST section is checked; the rest are out.

    Widened 2026-07-29 to pattern and playbook — the trigger is the section, not the
    type. `decision` stays out on purpose: a decision node is where a ruling is FILED,
    so its own "Rulings" section is its content and citing another node would be
    circular. That exclusion is asserted, not assumed, so it stays a choice."""
    body = "## Decisions locked\n- an unfiled ruling\n"
    assert find_unfiled_rulings(body), "sanity: body does contain a violation"
    for t in ("decision", "log", "reference", "project", "resource", "record"):
        assert check_completeness({"type": t}, body) == [], f"{t} must not be checked"
    for t in ("session", "note", "pattern", "playbook"):
        assert len(check_completeness({"type": t}, body)) == 1, f"{t} must be checked"


def test_stray_scan_stays_narrower_than_the_section_scan():
    """A `**Rule.**` marker means "a rule lives here" — a LEAK in a session, the
    CONTENT in a pattern. Widening the stray scan to patterns on 2026-07-29 produced
    7 findings, all wrong, so the two scanners keep different scopes."""
    from librarian.completeness import CHECKED_TYPES, STRAY_CHECKED_TYPES
    assert {"pattern", "playbook"} <= CHECKED_TYPES
    assert {"pattern", "playbook"}.isdisjoint(STRAY_CHECKED_TYPES)
    assert STRAY_CHECKED_TYPES < CHECKED_TYPES


def test_transferable_lessons_in_a_note_are_checked():
    """The shape: a case note whose 'Transferable lessons' state generalisable
    rules. Citing where the rule lives — a new node OR the playbook it already sits
    in — is what files the bullet."""
    body = """
## Transferable lessons — this case produced the two-oven forks
1. **Two ovens → two schedules, not one.** Router test: did both get a full load?
2. **THE KEY RULING — schedule each oven from its OWN readings.** · [[okf_20250110_090300_00d4]]
3. **The mirror-schedule pattern.** When loads run opposite, schedules can invert.
"""
    found = check_completeness({"type": "note"}, body)
    assert len(found) == 2, [f.text for f in found]
    assert all(f.section == "transferable lessons" for f in found)
    assert found[0].text.startswith("**Two ovens")
    assert found[1].text.startswith("**The mirror-schedule pattern")


def test_ordinal_bullets_count():
    body = "## Decisions locked\n1. first unfiled\n2. second, filed · [[okf_20250110_090000_c147]]\n"
    found = find_unfiled_rulings(body)
    assert len(found) == 1 and found[0].text.startswith("first unfiled"), [f.text for f in found]


def test_reports_line_number_and_truncates_text():
    body = "\n\n## Decisions locked\n- " + ("x" * 200) + "\n"
    found = find_unfiled_rulings(body)
    assert len(found) == 1
    assert found[0].line == 4, found[0].line
    assert len(found[0].text) == 80


def test_stray_ruling_found_outside_a_watched_section():
    """A self-labelled rule in a node's narrative is invisible to the section scan."""
    from librarian.completeness import find_stray_rulings
    body = """
## Internal notes for leadership
- **Invariant:** syncing only writes new rows, never rewrites existing ones.
- Ruling: never set a reasoning param globally.
- Just an ordinary observation about the client.
- **Ruling.** This one is already filed. [[okf_20250110_090400_45d6]]
"""
    found = find_stray_rulings(body)
    assert len(found) == 2, [f.text for f in found]
    assert all(f.section == "stray ruling" for f in found)


def test_stray_scan_ignores_watched_sections():
    """Inside a rulings section the section scan owns it — don't double-report."""
    from librarian.completeness import find_stray_rulings, find_unfiled_rulings
    body = "## Decisions locked\n- Ruling: something unfiled.\n"
    assert find_unfiled_rulings(body), "sanity: the section scan sees it"
    assert find_stray_rulings(body) == [], "stray scan must not double-report it"


def test_stray_scan_ignores_wrapped_continuation_lines():
    """A wrapped bullet can break so that 'ruling' lands first on the next physical
    line. That is not a new statement. (Caught live on a rulings session.)"""
    from librarian.completeness import find_stray_rulings
    body = ("## What changed\n"
            "- recipes v1.2 (new: rye-loaf R-1; seeded-loaf R-2 merged the\n"
            "  ruling — no duplicate recipe_id).\n")
    assert find_stray_rulings(body) == []


def test_stray_scan_self_citation_marks_the_home():
    """A node that IS the rule's home says so by citing its own id — the scan skips any
    line carrying an id, so this is how a procedure document declares a rule terminal."""
    from librarian.completeness import find_stray_rulings
    body = ("## The sync workflow\n"
            "**Invariant** (this node is the rule's home — [[okf_20250110_090500_ba7a]]): "
            "syncing only writes new rows.\n")
    assert find_stray_rulings(body) == []


def test_stray_scan_known_limit_unlabelled_rules():
    """PINS A DELIBERATE GAP. An unlabelled rule stated in house style (a bold lead-in)
    is NOT detected, and must not be — flagging every bold lead-in measured 375 uncited
    hits on the live corpus, nearly all list items. That class belongs to the Layer-4
    content audit. If this test starts failing, someone widened the heuristic; read
    find_stray_rulings' docstring before 'fixing' it.
    Measured on the author's private corpus; not reproducible from the example corpus."""
    from librarian.completeness import find_stray_rulings
    body = ("## Internal notes for leadership\n"
            "- **Exact delivery times land very differently on different cafés.** "
            "Useful for a busy kitchen; a nuisance for a quiet one.\n")
    assert find_stray_rulings(body) == []


def test_malformed_type_does_not_crash():
    """A hostile `type: []` must be a validator finding, not an unhashable-membership
    TypeError that kills the whole run (caught by test_unhashable_values... in the
    store suite when notes were added to CHECKED_TYPES)."""
    body = "## Decisions locked\n- an unfiled ruling\n"
    for bad in ([], {"a": 1}, 7, None):
        assert check_completeness({"type": bad}, body) == []


def test_empty_and_missing_body_are_safe():
    assert find_unfiled_rulings("") == []
    assert check_completeness(SESSION, None) == []
    assert check_completeness({}, "## Decisions locked\n- unfiled\n") == []


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    print("=== completeness (unfiled-ruling detector) tests ===")
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {(str(e).splitlines() or [''])[0][:100]}")
            failed += 1
    print(f"\n{'ALL PASS' if not failed else str(failed) + ' FAILED'} — {passed + failed} tests")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
