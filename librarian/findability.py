"""The weekly findability run (2026-10-14) — the the asks-and-findability decision approved and left unbuilt.

WHAT IT ASKS. Can a person who has forgotten a node's wording still reach it?
The held-out seed bank (tests/findability_seed.yaml) holds, per node, a
stranger question and a short phrase written from the body alone with the
node's own vocabulary banned, by writers who never saw its `asks`. Each week
every query runs through the same engine `brief` uses, and the rank of the node
it belongs to is recorded. No judge: the answer is a rank, not an opinion.

WHERE IT RUNS. Inside the okf MCP service, in the thread that warms it — the
model and the section vectors are already resident there, so the run costs no
memory on the 961 MB droplet, and a new systemd timer would need a hand on the
box that a git-push deploy does not have. Weekly, aligned to Monday 00:30 UTC
(06:00 IST) so the result is in place before the weekly digest reads
audit_corpus; and once immediately when no run has ever been recorded, so a
deploy establishes its own baseline.

WHAT IT WRITES. One LIVING log node, a dated block appended per run — not a
node per week, which would put fifty-two near-identical number sheets a year
into the retrieval pool. Blocks cite node ids and counts only, never a query:
the benchmark's filing discipline (a quoted query is the demonstrated
contamination path). The full per-node detail stays in a marker file outside
both repos (~/.okf/findability.json), which audit_corpus reads to FLAG a
regression, a failed run, or a run that has stopped happening.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from collections import Counter
from pathlib import Path

try:  # dual import: root (`python server.py`) or package
    import context
    import embed
    import store
except ImportError:
    from librarian import context, embed, store

KINDS = ("stranger", "short")
LOG_TAG = "findability-audit"
LOG_TITLE = "Findability audit — the weekly record of held-out stranger questions"

# a run is due once per week, at this point in the week (UTC)
DUE_WEEKDAY, DUE_HOUR, DUE_MINUTE = 0, 0, 30          # Monday 00:30 UTC = 06:00 IST
# audit_corpus flags the run as stopped when the last one is older than this
STALE_DAYS = 8
# a fall of this many queries in the default pack, in either kind, is a flag
DROP_FLAG = 3
# the section layer still embedding postpones a run, but never past this
MAX_POSTPONE_HOURS = 6
# The home the log node is filed under. It must be one of the corpus's
# PROJECT_REGISTRY homes; a corpus without `global-os` sets its own.
LOG_PROJECT_DEFAULT = "global-os"


def bank_path() -> Path:
    """The seed bank: OKF_FINDABILITY_BANK, else the one beside the code. The
    bank names node ids, so it measures only the corpus it was written for."""
    env = os.environ.get("OKF_FINDABILITY_BANK")
    return Path(env).expanduser() if env else \
        Path(__file__).resolve().parent.parent / "tests" / "findability_seed.yaml"


def marker_path() -> Path:
    env = os.environ.get("OKF_FINDABILITY")
    return Path(env).expanduser() if env else Path.home() / ".okf" / "findability.json"


def load_bank() -> list:
    import yaml
    return yaml.safe_load(bank_path().read_text(encoding="utf-8"))["findability"]


def load_marker() -> dict:
    try:
        return json.loads(marker_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_marker(m: dict) -> None:
    p = marker_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(m, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


# ── the measurement ──────────────────────────────────────────────────────────
def run(nodes, by_id, root, bank=None) -> dict:
    """Rank every held-out query against the live corpus. Pure: reads only."""
    bank = load_bank() if bank is None else bank
    live = {n.id for n in nodes if n.fm.get("status") in ("active", "parked")}
    bank = [b for b in bank if b["node"] in live]
    ranks, semantic = {}, False
    for b in bank:
        ranks[b["node"]] = {}
        for kind in KINDS:
            pack = context.assemble_context(nodes, by_id, b[kind], root,
                                            limit=15, body_chars=0)
            semantic = semantic or pack.get("semantic", False)
            ids = [s["id"] for s in pack["seeds"]]
            ranks[b["node"]][kind] = ids.index(b["node"]) + 1 if b["node"] in ids else None
    kinds = {}
    for kind in KINDS:
        rs = [r[kind] for r in ranks.values()]
        kinds[kind] = {"r1": sum(r == 1 for r in rs), "in8": sum(bool(r and r <= 8) for r in rs),
                       "in15": sum(r is not None for r in rs), "of": len(rs)}
    groups: Counter = Counter()
    totals: Counter = Counter()
    for b in bank:
        n = by_id[b["node"]]
        for g in (f"type:{n.fm.get('type')}", f"project:{n.fm.get('project')}"):
            for kind in KINDS:
                r = ranks[b["node"]][kind]
                groups[g] += bool(r and r <= 8)
                totals[g] += 1
    return {"at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "live": len(live), "semantic": semantic,
            "sections": embed.chunk_status().get("state"),
            "kinds": kinds, "ranks": ranks,
            "groups": {g: [groups[g], totals[g]] for g in sorted(totals)},
            "unreachable": {k: sorted(i for i, r in ranks.items() if r[k] is None)
                            for k in KINDS}}


def movers(prev: dict | None, cur: dict) -> list:
    """(node, kind, old rank, new rank) wherever a query crossed the default
    pack (8) or the fifteen — aggregate is the alarm, movers are the diagnosis
    (the benchmark's bracketing ruling, 2026-08-07)."""
    if not prev:
        return []
    def band(r):
        return 0 if r is None else 1 if r > 8 else 2
    out = []
    for nid, r in cur["ranks"].items():
        old = (prev.get("ranks") or {}).get(nid)
        if not old:
            continue
        for kind in KINDS:
            if band(old.get(kind)) != band(r.get(kind)):
                out.append((nid, kind, old.get(kind), r.get(kind)))
    return out


def render(cur: dict, prev: dict | None) -> str:
    """The dated block for the log node: counts and ids, never a query."""
    k = cur["kinds"]
    lines = [f"**Run {cur['at'][:10]}** — {cur['live']} live nodes, "
             f"{k['stranger']['of']} in the bank; semantic {'on' if cur['semantic'] else 'OFF'}, "
             f"sections {cur['sections']}.", "",
             "| | top-1 | in 8 | in 15 |", "|---|---|---|---|"]
    for kind in KINDS:
        d = k[kind]
        was = ""
        if prev and kind in (prev.get("kinds") or {}):
            delta = d["in8"] - prev["kinds"][kind]["in8"]
            was = f" ({delta:+d})" if delta else ""
        lines.append(f"| {kind} | {d['r1']} | {d['in8']}{was} | {d['in15']} |")
    lines += ["", "In 8, both kinds: " + "; ".join(
        f"{g} {h}/{t}" for g, (h, t) in cur["groups"].items()) + "."]
    unreach = sorted(set(cur["unreachable"]["stranger"]) | set(cur["unreachable"]["short"]))
    lines.append(f"Unreachable in 15 by either query ({len(unreach)}): "
                 + (", ".join(unreach) if unreach else "none") + ".")
    mv = movers(prev, cur)
    if prev:
        lines.append("Movers across 8 or 15 since the last run: " + (
            "; ".join(f"{n} {kind} {o or '—'}→{r or '—'}" for n, kind, o, r in mv)
            if mv else "none") + ".")
    return "\n".join(lines)


# ── filing ───────────────────────────────────────────────────────────────────
def _find_log(nodes) -> str | None:
    for n in nodes:
        if (n.fm.get("type") == "log" and n.fm.get("status") in ("active", "parked")
                and LOG_TAG in (n.fm.get("tags") or [])):
            return n.id
    return None


def record(root, nodes, cur: dict) -> dict:
    """Append this run to the living log node (creating it on the first run)
    and write the marker audit_corpus reads."""
    m = load_marker()
    prev = m.get("last")
    block = render(cur, prev)
    log_id = _find_log(nodes)
    if log_id:
        store.update_node(root, log_id, {}, body_append=block)
    else:
        made = store.create_node(
            root, "log", LOG_TITLE,
            body="# " + LOG_TITLE + "\n\nWritten by the okf service once a week "
                 "(librarian/findability.py) from the held-out seed bank in the code "
                 "repo (tests/findability_seed.yaml). Each block is one run: how many "
                 "of the bank's queries reached their node at rank 1, in the default "
                 "pack of 8 and in 15; the same per type and project; the nodes no "
                 "query reached; and the movers since the previous run. Ids and "
                 "counts only — the queries stay in the code repo, never here.\n\n"
                 + block,
            project=os.environ.get("OKF_FINDABILITY_PROJECT") or LOG_PROJECT_DEFAULT,
            work_state="open", maturity="captured",
            summary="The weekly findability run's record: per run, how many held-out "
                    "stranger questions and short phrases (written blind to each "
                    "node's asks, its own vocabulary banned) reach their node in the "
                    "default pack of 8 and in 15, by type and project, with every "
                    "unreachable node and every mover named by id; regressions "
                    "surface in audit_corpus.",
            tags=["okf", "retrieval", "findability", LOG_TAG, "log"],
            asks=["how well does search find notes from vague questions each week",
                  "weekly score for whether notes can be found",
                  "which notes can nobody find with everyday words",
                  "is search getting better or worse over time"],
            source="librarian/findability.py — the weekly findability run")
        log_id = made.get("id")
    _save_marker({"last": cur, "prev": prev, "log_id": log_id, "error": None,
                  "error_at": None, "postponed_since": None})
    return {"log_id": log_id, "block": block}


def _record_error(e: Exception) -> None:
    m = load_marker()
    m.update(error=f"{type(e).__name__}: {e}"[:300],
             error_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    _save_marker(m)


# ── scheduling ───────────────────────────────────────────────────────────────
def _last_due(now: dt.datetime) -> dt.datetime:
    """The most recent Monday 00:30 UTC at or before `now`."""
    d = now - dt.timedelta(days=(now.weekday() - DUE_WEEKDAY) % 7)
    d = d.replace(hour=DUE_HOUR, minute=DUE_MINUTE, second=0, microsecond=0)
    return d if d <= now else d - dt.timedelta(days=7)


def due(marker: dict, now: dt.datetime | None = None) -> bool:
    now = now or dt.datetime.now(dt.timezone.utc)
    last = (marker.get("last") or {}).get("at")
    if not last:
        return True                                    # never run: establish the baseline
    last_at = dt.datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    return last_at < _last_due(now)


def tick(load_corpus, root, log=None, now=None) -> str:
    """One scheduler step: run and record if due. Returns what it did."""
    say = log or (lambda m: print(m, flush=True))
    m = load_marker()
    if not due(m, now):
        return "not due"
    status = embed.chunk_status()
    state = status.get("state")
    if status.get("node_refresh"):         # node vectors stale after a corpus-wide change
        state = "node vectors refreshing"
    if state in ("embedding", "missing", "not loaded", "node vectors refreshing"):
        since = m.get("postponed_since") or time.time()
        if time.time() - since < MAX_POSTPONE_HOURS * 3600:
            if not m.get("postponed_since"):
                m["postponed_since"] = since
                _save_marker(m)
            return f"postponed (sections {state})"
    try:
        nodes, by_id, *_ = load_corpus()
        cur = run(nodes, by_id, root)
        if not cur["kinds"]["stranger"]["of"]:
            # A bank written for another corpus matches nothing here. Recording
            # "0 of 0" would read as a run that happened; it measured nothing.
            raise RuntimeError(
                f"no seed-bank entry names a live node in this corpus ({bank_path()}) "
                "— nothing was measured. Write a bank for this corpus and point "
                "OKF_FINDABILITY_BANK at it")
        out = record(root, nodes, cur)
        k = cur["kinds"]
        say(f"findability: stranger {k['stranger']['in8']}/{k['stranger']['of']} in 8, "
            f"short {k['short']['in8']}/{k['short']['of']} in 8 → {out['log_id']}")
        return "ran"
    except Exception as e:
        _record_error(e)
        say(f"findability: run failed ({e})")
        return "failed"


def loop(load_corpus, root, log=None, every=1800) -> None:
    """The service's weekly scheduler: a tick every half hour, forever. Every
    failure is recorded in the marker (audit_corpus flags it) and retried at
    the next tick — a run that cannot happen must say so, never pass quietly."""
    while True:
        try:
            tick(load_corpus, root, log)
        except Exception:
            pass
        time.sleep(every)


# ── what audit_corpus shows ──────────────────────────────────────────────────
def compute_flag(now: float | None = None) -> dict | None:
    """A flag when the run failed, stopped, or regressed; None when it is
    healthy or has never run on this machine (the Mac). Never raises."""
    p = marker_path()
    try:
        if not p.is_file():
            return None
        m = json.loads(p.read_text(encoding="utf-8"))
        now = now or time.time()
        last = m.get("last") or {}
        base = {"key": "findability", "log_id": m.get("log_id")}
        if m.get("error"):
            return {**base, "label": "Findability run failed",
                    "desc": f"The weekly findability run failed at {m.get('error_at')}: "
                            f"{m['error']}. It retries every half hour; the last good "
                            f"run was {last.get('at') or 'never'}."}
        if last.get("at"):
            at = dt.datetime.strptime(last["at"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=dt.timezone.utc).timestamp()
            if now - at > STALE_DAYS * 86400:
                return {**base, "label": "Findability run stopped",
                        "desc": f"No findability run since {last['at']} — the weekly "
                                "run in the okf service has not happened; check that "
                                "the service is up and warmed."}
        prev = m.get("prev") or {}
        drops = [f"{k} {prev['kinds'][k]['in8']}→{last['kinds'][k]['in8']} in 8"
                 for k in KINDS if prev.get("kinds") and last.get("kinds")
                 and prev["kinds"][k]["in8"] - last["kinds"][k]["in8"] >= DROP_FLAG]
        lost = [f"{n} {kind}" for n, kind, o, r in movers(prev, last) if o and r is None]
        if drops or lost:
            return {**base, "label": "Findability regressed",
                    "desc": "The weekly findability run fell since the previous run: "
                            + "; ".join(drops + ([f"newly unreachable in 15: {', '.join(lost)}"]
                                                 if lost else []))
                            + f". Movers and detail are in the log node {m.get('log_id')}."}
        return None
    except Exception:
        return {"key": "findability", "label": "Findability run failed",
                "desc": f"The findability marker at {p} exists but could not be read — "
                        "a verdict that cannot be verified is a failure, not a pass."}


def summary_line() -> str | None:
    """'stranger 22/40, short 20/40 in 8 (2026-10-14)' for the audit summary."""
    last = (load_marker().get("last") or {})
    k = last.get("kinds")
    if not k:
        return None
    return (f"stranger {k['stranger']['in8']}/{k['stranger']['of']}, "
            f"short {k['short']['in8']}/{k['short']['of']} in 8 ({last['at'][:10]})")
