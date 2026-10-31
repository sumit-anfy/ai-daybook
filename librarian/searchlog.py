"""Search telemetry — surface-correct outcomes for the hybrid engine.

LINEAGE. Built 2026-08-04 as the instrument behind the staged semantic-retrieval decision's
amended v2-embeddings trigger. v2 SHIPPED 2026-08-07 and the role inverted —
and was then refined the same day: the WEB miss trend is regression-readable,
while the MCP suspected queue is a failure-DISCOVERY instrument (adjudication
candidates, contaminated by diagnostic traffic by design), never a quality
metric. The owner ruled the semantics SURFACE-CORRECT after the miss labels were
caught fabricating failure (27 of 27 MCP queries read as "misses" while
several were verifiably succeeding, correct node at rank 1 — and the refined
role was proven the same way: a rank-1-correct query sat in the suspected
queue the day it shipped).

WHY A SCORE THRESHOLD IS NOT ENOUGH (the founding design, unchanged): on the
eight failing paraphrase probes the WRONG node scored 4.2–7.4,
indistinguishable from a good hit. Score cannot separate "found it" from
"returned something plausible and useless". The signal must be BEHAVIOURAL —
but behaviour differs by surface, which is what the first design missed:

- WEB has an open signal. A search from which nothing was opened, reworded
  runs collapsed, IS a miss. That inference stands, unchanged.
- MCP does not. `brief()` returns the bodies, so there is usually nothing to
  open, and "no open" means NOTHING. MCP outcomes are TRI-STATE: a run that
  was immediately reworded is a SUSPECTED miss; a run with no further action
  is UNKNOWN — not a miss and not a success. A `read_node` of a returned
  seed is counted as ENGAGED, observational only, never a success criterion:
  the rejected alternative (read-as-open) would have rebuilt the same
  instrument error mirrored, since a perfect pack needs no follow-up read.
  A fabricated binary metric is worse than an honest partial one.

`outcomes()` derives all of this at read time, so the thresholds can be
re-tuned later without having thrown data away.

WHERE IT WRITES. Outside both git repos by default. A query log is live data, not
knowledge (the knowledge-not-live-data decision), it grows without bound, and putting it in the corpus would make
every search a commit. Override with OKF_SEARCH_LOG.

NEVER RAISES. A telemetry failure must not break a search. Every entry point
swallows its own errors — the cost of losing a log line is nothing; the cost of a
500 on the search box is the surface the owner actually uses.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# A reworded query within this many seconds counts as the same attempt, so the
# pair reads as one failed search rather than two searches.
REWORD_WINDOW_S = 90

# A read_node of a returned seed within this window counts as ENGAGED on the
# MCP surface — observational evidence the pack was used, never success.
ENGAGE_WINDOW_S = 600


def log_path() -> Path:
    env = os.environ.get("OKF_SEARCH_LOG")
    return Path(env).expanduser() if env else Path.home() / ".okf" / "search.jsonl"


def _append(rec: dict) -> None:
    try:
        p = log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        rec["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass        # telemetry must never break the thing it measures


def record_search(query: str, hits: int, top_score, surface: str,
                  seed_ids=None) -> None:
    """One search. `top_score` is the best relevance score, or None for no hits.
    `seed_ids` (MCP surface) lets outcomes() correlate later read_node calls
    into the observational ENGAGED count."""
    q = (query or "").strip()
    if not q:
        return
    rec = {"kind": "search", "q": q, "hits": int(hits),
           "top": round(float(top_score), 2) if top_score is not None else None,
           "surface": surface}
    if seed_ids:
        rec["ids"] = [str(i) for i in list(seed_ids)[:8]]
    _append(rec)


def record_read(node_id) -> None:
    """Observational: a read_node call. On the MCP surface a follow-up read of
    a seed is ENGAGEMENT evidence — never a success criterion, and its absence
    is never failure (the tri-state ruling, 2026-08-07)."""
    if node_id:
        _append({"kind": "read", "id": str(node_id)})


def record_open(query: str, node_id: str, rank, surface: str) -> None:
    """A node opened FROM a search result — the signal that the search worked.

    This is the half a score threshold cannot provide. A search followed by an
    open succeeded whatever it scored; a search followed by nothing did not."""
    q = (query or "").strip()
    if not q or not node_id:
        return
    _append({"kind": "open", "q": q, "id": node_id,
             "rank": int(rank) if rank is not None else None, "surface": surface})


def _read(path: Path | None = None) -> list[dict]:
    p = path or log_path()
    out = []
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue        # a torn final line is not a reason to fail
    except FileNotFoundError:
        return []
    except Exception:
        return []
    return out


def _ts(rec):
    try:
        return datetime.fromisoformat(rec["at"])
    except Exception:
        return None


def _runs(searches: list[dict]) -> list[tuple[dict, int]]:
    """Collapse a rewording run — consecutive same-surface searches within
    REWORD_WINDOW_S where one query contains, or shares a word with, the
    next — into (final_record, run_length). Otherwise one frustrated search
    reads as three, inflating whatever the run is evidence of."""
    out: list[tuple[dict, int]] = []
    for r in searches:
        if out:
            prev, n = out[-1]
            t0, t1 = _ts(prev), _ts(r)
            if t0 and t1 and 0 <= (t1 - t0).total_seconds() <= REWORD_WINDOW_S \
                    and prev.get("surface") == r.get("surface"):
                a, b = prev["q"].lower(), r["q"].lower()
                if a in b or b in a or (set(a.split()) & set(b.split())):
                    out[-1] = (r, n + 1)   # the later attempt supersedes
                    continue
        out.append((r, 1))
    return out


def _norm_q(q) -> str:
    return str(q or "").strip().lower()


def _web_runs(searches: list[dict], opens: list[dict]) -> list[tuple[dict, int, bool]]:
    """Web attempts with their opens attributed BY TIME, not by query text.

    Until 2026-09-12 a web search counted as not-a-miss if its exact query
    string appeared in ANY open record in the whole log — so one click on
    "cap table" in August cleared every later "cap table" search forever,
    including a zero-result one (found by the fifth external review,
    reproduced). An open now belongs to the attempt it followed: events are
    walked in time order; an open whose query matches the current run marks
    that run opened; a search that would extend the run by the rewording rule
    starts a NEW run instead if the current one has already been opened,
    because a click ends an attempt. Returns (final_record, run_length,
    opened). Queries compare case-insensitively — the web list normalises
    them the same way since the same day.
    """
    events = sorted(
        [(_ts(r), 0, r) for r in searches] + [(_ts(r), 1, r) for r in opens],
        key=lambda t: (t[0] is None, t[0] or datetime.min.replace(tzinfo=timezone.utc)))
    out: list[list] = []            # [final_record, n, opened, first_ts]
    for ts, kind, r in events:
        if kind == 1:
            if out and _norm_q(r.get("q")) == _norm_q(out[-1][0].get("q")) \
                    and not out[-1][2]:
                out[-1][2] = True
            elif out and not out[-1][2]:
                # a click on a different query still belongs to the latest
                # attempt when the run was reworded through it (rare); keep the
                # strict rule — text mismatch, no credit — so a miss stays a miss
                pass
            continue
        if out and not out[-1][2]:
            prev, n, _, t0 = out[-1]
            t1 = ts
            if t0 is not None and t1 is not None and 0 <= (t1 - t0).total_seconds() <= REWORD_WINDOW_S:
                a, b = _norm_q(prev.get("q")), _norm_q(r.get("q"))
                if a in b or b in a or (set(a.split()) & set(b.split())):
                    out[-1] = [r, n + 1, False, t0]
                    continue
        out.append([r, 1, False, ts])
    return [(r, n, opened) for r, n, opened, _ in out]


def outcomes(path: Path | None = None, limit: int = 12) -> dict:
    """Surface-correct search outcomes (2026-08-07 ruling; see module doc).

    WEB: the behavioural miss inference stands — a collapsed run from which
    nothing was opened is a miss. MCP: tri-state — a reworded run is a
    SUSPECTED miss, an untouched run is UNKNOWN, and a read_node of a
    returned seed within ENGAGE_WINDOW_S is ENGAGED (observational only).
    """
    recs = _read(path)
    searches = [r for r in recs if r.get("kind") == "search"]
    web = [r for r in searches if r.get("surface") != "mcp"]
    mcp = [r for r in searches if r.get("surface") == "mcp"]
    opens = [r for r in recs if r.get("kind") == "open" and r.get("surface") != "mcp"]
    reads = [(r.get("id"), _ts(r)) for r in recs if r.get("kind") == "read"]

    def _recent(rows):
        return [{"q": r["q"], "hits": r.get("hits"), "top": r.get("top"),
                 "surface": r.get("surface"), "at": r.get("at")}
                for r in rows[-limit:][::-1]]

    web_runs = _web_runs(web, opens)
    web_miss = [r for r, _, opened in web_runs if not opened]

    mcp_runs = _runs(mcp)
    suspected = [r for r, n in mcp_runs if n > 1]
    engaged = 0
    for r, _ in mcp_runs:
        ids, t0 = set(r.get("ids") or []), _ts(r)
        if ids and t0 and any(
                i in ids and t is not None
                and 0 <= (t - t0).total_seconds() <= ENGAGE_WINDOW_S
                for i, t in reads):
            engaged += 1

    return {"searches": len(searches),
            "web": {"searches": len(web), "attempts": len(web_runs),
                    "misses": len(web_miss),
                    "zero_hit": sum(1 for r in web_miss if r.get("hits", 0) == 0),
                    "recent": _recent(web_miss)},
            "mcp": {"queries": len(mcp), "attempts": len(mcp_runs),
                    "suspected_misses": len(suspected),
                    "unknown": sum(1 for _, n in mcp_runs if n == 1),
                    "engaged": engaged,
                    "zero_hit": sum(1 for r, _ in mcp_runs if r.get("hits", 0) == 0),
                    "recent_suspected": _recent(suspected)}}


def empty_outcomes() -> dict:
    """The shape outcomes() returns over no data — shared with audit's
    fail-soft path so the two can never drift."""
    return {"searches": 0,
            "web": {"searches": 0, "attempts": 0, "misses": 0, "zero_hit": 0,
                    "recent": []},
            "mcp": {"queries": 0, "attempts": 0, "suspected_misses": 0,
                    "unknown": 0, "engaged": 0, "zero_hit": 0,
                    "recent_suspected": []}}
