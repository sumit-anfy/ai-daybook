"""Fixtures for the search log (librarian/searchlog, 2026-08-04; tri-state 2026-08-07).

WHAT THIS INSTRUMENT IS FOR. Built as the evidence stream behind the amended
v2-embeddings trigger; v2 shipped 2026-08-07 and the role inverted to the
hybrid engine's regression evidence. The same day the miss labels were caught
fabricating failure — 27 of 27 MCP queries read as "misses" while several were
verifiably succeeding — so the owner ruled the semantics SURFACE-CORRECT.

THE DESIGN CONSTRAINTS THESE FIXTURES PIN. (1) A score threshold cannot detect
failure: on the eight founding paraphrase probes the WRONG node scored 4.2-7.4.
(2) The behavioural signal differs by surface: web has an open signal and keeps
the miss inference; MCP does not — brief() returns bodies, so "no open" means
nothing, and outcomes are TRI-STATE (suspected via rewording / unknown /
engaged-observational). (3) A read of a seed is ENGAGEMENT, never success —
read-as-open was explicitly rejected because a perfect pack needs no follow-up
read, and that proxy would rebuild the same instrument error mirrored.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian import searchlog                            # noqa: E402


@pytest.fixture
def log(tmp_path, monkeypatch):
    p = tmp_path / "nested" / "search.jsonl"
    monkeypatch.setenv("OKF_SEARCH_LOG", str(p))
    return p


# ── web: the behavioural miss inference stands, unchanged ─────────────────────

# ── clicks belong to attempts, not to query strings (2026-09-12) ──────────────
def _stamp(p, offsets):
    """Rewrite the log's timestamps to the given second offsets, in file order —
    the fixture writes `now` on every record, and attribution is by time."""
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)
    lines = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert len(lines) == len(offsets)
    for rec, off in zip(lines, offsets):
        rec["at"] = (base + timedelta(seconds=off)).isoformat(timespec="seconds")
    p.write_text("".join(json.dumps(r) + "\n" for r in lines))


def test_an_old_click_does_not_clear_a_later_search(log):
    """The exact reproduction from the review: a click on "cap table" in the
    past made a later ZERO-result "cap table" search register no miss."""
    searchlog.record_search("cap table", 8, 8.92, "web")
    searchlog.record_open("cap table", "okf_20250110_090600_9366", 1, "web")
    searchlog.record_search("cap table", 0, None, "web")
    _stamp(log, [0, 5, 3600])                     # the second search an hour later
    o = searchlog.outcomes()["web"]
    assert o["attempts"] == 2 and o["misses"] == 1 and o["zero_hit"] == 1


def test_a_click_ends_a_run_even_inside_the_rewording_window(log):
    """search → click → same query again within 90 s is TWO attempts, one found
    and one not — a click is the end of an attempt, not a keystroke in it."""
    searchlog.record_search("cap table", 8, 8.92, "web")
    searchlog.record_open("cap table", "okf_20250110_090600_9366", 1, "web")
    searchlog.record_search("cap table", 8, 8.92, "web")
    _stamp(log, [0, 5, 20])
    o = searchlog.outcomes()["web"]
    assert o["attempts"] == 2 and o["misses"] == 1


def test_a_click_before_the_search_does_not_count(log):
    searchlog.record_open("cap table", "okf_20250110_090600_9366", 1, "web")
    searchlog.record_search("cap table", 8, 8.92, "web")
    _stamp(log, [0, 30])
    assert searchlog.outcomes()["web"]["misses"] == 1


def test_query_case_does_not_split_a_click_from_its_search(log):
    """The web list now lowercases before sending; older records may not be —
    attribution compares case-insensitively either way."""
    searchlog.record_search("Cap Table", 8, 8.92, "web")
    searchlog.record_open("cap table", "okf_20250110_090600_9366", 1, "web")
    _stamp(log, [0, 4])
    assert searchlog.outcomes()["web"]["misses"] == 0


def test_a_web_search_nobody_opened_is_a_miss(log):
    searchlog.record_search("who owns how much of the company", 5, 4.24, "web")
    o = searchlog.outcomes()
    assert o["web"]["misses"] == 1
    assert o["web"]["recent"][0]["q"].startswith("who owns")


def test_a_web_search_followed_by_an_open_is_not_a_miss(log):
    """The founding point: a good score does not prove a find, and an open does."""
    searchlog.record_search("cap table", 8, 8.92, "web")
    searchlog.record_open("cap table", "okf_20250110_090600_9366", 1, "web")
    assert searchlog.outcomes()["web"]["misses"] == 0


def test_a_high_score_alone_does_not_clear_a_web_miss(log):
    """Pins the reason a threshold was rejected — 7.4 here, still a failure."""
    searchlog.record_search("what happens when the server cannot reach github", 8, 7.36, "web")
    assert searchlog.outcomes()["web"]["misses"] == 1


def test_web_rewording_collapses_to_one_attempt(log):
    """One frustrated search must not read as three failures — that would
    inflate the very signal this log exists to measure honestly."""
    for q in ("cap", "cap table", "cap table equity"):
        searchlog.record_search(q, 3, 3.0, "web")
    o = searchlog.outcomes()
    assert o["searches"] == 3, "every search is still recorded"
    assert o["web"]["attempts"] == 1, "but they collapse to one attempt"
    assert o["web"]["misses"] == 1


def test_unrelated_consecutive_searches_do_not_collapse(log):
    searchlog.record_search("cap table", 3, 3.0, "web")
    searchlog.record_search("todoist routing", 3, 3.0, "web")
    assert searchlog.outcomes()["web"]["attempts"] == 2


# ── mcp: tri-state — no fabricated binary ─────────────────────────────────────
def test_an_untouched_mcp_query_is_unknown_not_a_miss(log):
    """THE 27/27 defect: brief() returns the bodies, so a perfect MCP result
    often needs no further action at all. Absence of signal is UNKNOWN."""
    searchlog.record_search("can we delete a node", 8, 9.1, "mcp",
                            seed_ids=["okf_a", "okf_b"])
    o = searchlog.outcomes()["mcp"]
    assert (o["queries"], o["attempts"]) == (1, 1)
    assert o["suspected_misses"] == 0
    assert o["unknown"] == 1


def test_a_quickly_reworded_mcp_run_is_a_suspected_miss(log):
    searchlog.record_search("privacy retrieval rule", 6, 4.0, "mcp")
    searchlog.record_search("privacy retrieval ruling sensitive nodes", 6, 5.0, "mcp")
    o = searchlog.outcomes()["mcp"]
    assert o["attempts"] == 1, "the rewording run collapses"
    assert o["suspected_misses"] == 1 and o["unknown"] == 0
    assert o["recent_suspected"][0]["q"].endswith("sensitive nodes")


def test_a_read_of_a_seed_is_engaged_but_never_success(log):
    """Engagement is observational: it must NOT convert unknown into success
    or cancel a suspected miss — read-as-open was rejected by ruling."""
    searchlog.record_search("deletion gates", 5, 6.0, "mcp",
                            seed_ids=["okf_bb78", "okf_x"])
    searchlog.record_read("okf_bb78")
    o = searchlog.outcomes()["mcp"]
    assert o["engaged"] == 1
    assert o["unknown"] == 1, "engaged does not reclassify the outcome"


def test_a_read_of_a_non_seed_is_not_engagement(log):
    searchlog.record_search("deletion gates", 5, 6.0, "mcp", seed_ids=["okf_bb78"])
    searchlog.record_read("okf_unrelated")
    assert searchlog.outcomes()["mcp"]["engaged"] == 0


def test_web_and_mcp_streams_do_not_mix(log):
    """A web search and an MCP query in the same window must not collapse into
    one run or leak into each other's counts."""
    searchlog.record_search("cap table", 3, 3.0, "web")
    searchlog.record_search("cap table shareholding", 8, 7.0, "mcp")
    o = searchlog.outcomes()
    assert o["web"]["attempts"] == 1 and o["mcp"]["attempts"] == 1
    assert o["web"]["misses"] == 1 and o["mcp"]["unknown"] == 1


def test_zero_hit_counted_per_surface(log):
    searchlog.record_search("quorbulation", 0, None, "mcp")
    searchlog.record_search("zorptastic", 0, None, "web")
    o = searchlog.outcomes()
    assert o["mcp"]["zero_hit"] == 1 and o["web"]["zero_hit"] == 1


# ── plumbing: never raise, never lie about absence ────────────────────────────
def test_an_empty_query_is_not_logged(log):
    searchlog.record_search("   ", 0, None, "web")
    searchlog.record_open("", "okf_x", 1, "web")
    searchlog.record_read("")
    assert searchlog.outcomes()["searches"] == 0


def test_logging_never_raises_even_when_the_path_is_unwritable(log, monkeypatch):
    """Telemetry must never break the thing it measures. Losing a line costs
    nothing; a 500 on the search box costs the surface the owner actually uses."""
    monkeypatch.setenv("OKF_SEARCH_LOG", "/proc/definitely/not/writable/x.jsonl")
    searchlog.record_search("anything", 1, 1.0, "web")     # must not raise
    assert searchlog.outcomes() == searchlog.empty_outcomes(), \
        "no data reads as the one canonical empty shape, shared with audit"


def test_a_torn_line_does_not_break_the_reader(log):
    searchlog.record_search("good one", 2, 2.0, "web")
    with log.open("a", encoding="utf-8") as f:
        f.write('{"kind": "search", "q": "trunc\n')      # a partial write
    searchlog.record_search("another", 2, 2.0, "web")
    assert searchlog.outcomes()["searches"] == 2


def test_missing_log_reads_as_empty_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("OKF_SEARCH_LOG", str(tmp_path / "never-written.jsonl"))
    assert searchlog.outcomes() == searchlog.empty_outcomes()


def test_the_log_is_jsonl_and_timestamped_with_capped_ids(log):
    searchlog.record_search("shape check", 1, 1.5, "mcp",
                            seed_ids=[f"okf_{i}" for i in range(20)])
    rec = json.loads(log.read_text(encoding="utf-8").strip())
    assert rec["kind"] == "search" and rec["q"] == "shape check"
    assert rec["surface"] == "mcp" and rec["top"] == 1.5
    assert rec["at"].startswith("20"), "every record carries a timestamp"
    assert len(rec["ids"]) == 8, "seed ids are capped — a log line stays a line"


def test_default_path_is_outside_the_repos(monkeypatch):
    """A query log is live data (the knowledge-not-live-data decision), grows without bound, and would make every
    search a commit if it lived in the corpus."""
    monkeypatch.delenv("OKF_SEARCH_LOG", raising=False)
    p = searchlog.log_path()
    repo = Path(__file__).resolve().parent.parent
    assert repo not in p.parents and p != repo
