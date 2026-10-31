"""The bullet summary (librarian/summary.py) — pure, no corpus, no model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian import summary   # noqa: E402

BODY = """# Title restated

Opening text.

## Rulings
The ruling.

### Detail
More.

## APPEND (2026-09-14)
First append.

## APPEND (2026-09-14)
Second append.
"""


def bs(*lines):
    return "\n".join("- " + ln for ln in lines)


def test_legacy_prose_is_not_a_bullet_summary_and_is_never_checked():
    prose = "A paragraph summary - with a dash inside it, as many have."
    assert summary.bullets(prose) is None
    assert summary.check(prose, BODY) == []
    assert summary.bullets("") is None and summary.bullets(None) is None
    assert summary.first_line(prose) == prose


def test_bullets_split_into_point_and_pointer():
    s = bs("What this is.", "The point stands → § Rulings")
    assert summary.bullets(s) == ["What this is.", "The point stands → § Rulings"]
    assert summary.split("The point stands → § Rulings") == ("The point stands", "Rulings")
    assert summary.search_texts(s) == ["What this is.", "The point stands"]
    assert summary.first_line(s) == "What this is."


def test_a_well_formed_summary_is_quiet():
    s = bs("A ruling note about one thing.",
           "The ruling holds (changed 2026-08-25: was the old rule) → § Rulings",
           "Nested detail → § Rulings › Detail",
           "Inner heading alone → § Detail",
           "The first append → § APPEND (2026-09-14) #1",
           "Undated change (overruled undated: the earlier view) → § (opening)")
    assert summary.check(s, BODY) == []


def test_too_many_bullets_says_split():
    s = bs("What.", *[f"Point {k} → § Rulings" for k in range(8)])
    assert any("consider splitting" in w for w in summary.check(s, BODY))
    assert any("consider splitting" in w for w in summary.check(s, BODY, "decision"))


def test_a_session_may_run_past_eight_bullets_but_keeps_the_other_checks():
    s = bs("A sitting.", *[f"Point {k} → § Rulings" for k in range(8)])
    assert summary.check(s, BODY, "session") == []
    s = bs("A sitting.", *[f"Point {k} → § Nowhere" for k in range(8)])
    assert any("points to no heading" in w for w in summary.check(s, BODY, "session"))


def test_long_bullet_and_missing_pointer_warn():
    s = bs("What.", " ".join(["word"] * 26) + " → § Rulings", "No pointer here")
    w = summary.check(s, BODY)
    assert any("26 words" in x for x in w) and any("no section pointer" in x for x in w)


def test_pointer_must_name_exactly_one_section():
    w = summary.check(bs("What.", "Gone → § Nowhere"), BODY)
    assert any("points to no heading" in x for x in w)
    w = summary.check(bs("What.", "Which one → § APPEND (2026-09-14)"), BODY)
    assert any("add ' #N'" in x for x in w)
    w = summary.check(bs("What.", "Third → § APPEND (2026-09-14) #3"), BODY)
    assert any("occurs 2 time" in x for x in w)


def test_change_markers_need_a_date_or_undated():
    w = summary.check(bs("What.", "Moved (changed: was X) → § Rulings"), BODY)
    assert any("YYYY-MM-DD" in x for x in w)
