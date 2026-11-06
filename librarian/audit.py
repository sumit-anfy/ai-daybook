"""Deterministic corpus audit — health flags + link suggestions.

Pure functions over an already-loaded corpus, shared by the web app (okf_web) and
the MCP server so both surfaces compute the IDENTICAL triage. No LLM:

- health flags are structural — orphans (no links), same-source duplicates,
  stale neighbourhoods (every neighbour retired), near-identical titles, and
  unpushed corpus writes (the wedge detector — the only flag about durability
  rather than tidiness, so it sorts first).
- link suggestions are unlinked *active* pairs scored by shared-tag rarity (IDF)
  so a project-wide tag like `bakery` counts for almost nothing while a narrow
  one like `behaviour-plan` ranks a pair high, plus a lighter content-word
  overlap on title+summary.

It only proposes — a human clicks Connect / Skip in the web panel, or reads the
weekly digest. Dismissed pairs (.okf/dismissed.json) are suppressed; curated
pairs (.okf/suggestions.json, human/LLM-authored) override the auto reason and
rank first.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

try:  # dual import: root (`python server.py`) or package (`-m okf_web.server`)
    from store import _coda_key, _norm_title
    import store as _store
except ImportError:
    from librarian.store import _coda_key, _norm_title
    from librarian import store as _store

_STOP = set("""the and for with that this from into your you our are was were will would
should shall must have has had not but they them their its it as at by an of to in on or a
is be been being can via per vs etc about over under more most less out up down off which
when what who how why than then also just only some any all one two three does did done use
using used make made need needs get got given give more first second same other such very
between across within under over into onto because while where whom whose here there""".split())


def _tags(n):
    return {str(t).lower().strip() for t in (n.fm.get("tags") or []) if str(t).strip()}


def _tokens(n):
    text = (str(n.fm.get("title", "")) + " " + str(n.fm.get("summary") or "")).lower()
    return {t for t in re.findall(r"[a-z][a-z0-9-]{3,}", text) if t not in _STOP}


def _brief(n):
    return {"id": n.id, "title": n.fm.get("title", ""),
            "type": n.fm.get("type"), "project": n.fm.get("project", "—")}


# ── sync (wedge detector) ──────────────────────────────────────────────────────
def compute_sync(root):
    """Unpushed corpus writes on a box that is supposed to auto-push.

    The Librarian pushes every write (`OKF_GIT_PUSH=1`), so on that box ANY
    unpushed commit means a push failed and the corpus now exists in exactly one
    place. `store._git_commit` recovers on the NEXT write, which leaves the hole
    this closes: a box that diverges and then goes idle stays stranded, silently
    and indefinitely. That is the 2026-08-01 wedge — 7 commits, five hours, found
    only by hand (the change-the-input retry pattern).

    Deliberately network-free. `ahead` compares HEAD to the remote-tracking ref,
    which git advances only on a CONFIRMED push, so a stale ref can never hide a
    wedge — it can only make us miss being *behind*, which costs no durability
    and is not flagged. No fetch means this stays safe in the read path.

    Returns a flag dict, or None when there is nothing to say. Never raises: an
    audit that dies because git is unusual is worse than one that stays quiet.
    """
    # Gate on the auto-push contract. On the Mac, committing before pushing is
    # normal work, and a flag that fires there every day is one nobody reads
    # (the fail-loudly rule §3).
    if os.environ.get("OKF_GIT_PUSH") != "1":
        return None
    # A push that is QUEUED is not a wedge. Since the background pusher landed
    # (2026-08-02) there is a ~1s window after every write where HEAD is ahead of
    # the remote-tracking ref for a completely healthy reason, and firing there
    # announces "the last push did not land" about a push not yet attempted.
    # Suppressing is safe: this is a periodic check, not a real-time monitor, so a
    # genuine wedge is caught by the next call once the queue drains.
    try:
        if _store.push_pending():
            return None
    except Exception:
        pass
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args],
                                  capture_output=True, text=True, timeout=10)

        r = git("rev-list", "--left-right", "--count", "@{upstream}...HEAD")
        if r.returncode != 0:
            return None  # no upstream, not a repo — absent, not wedged
        behind, ahead = (int(x) for x in r.stdout.split())
        if ahead == 0:
            return None
        log = git("log", "--format=%h %s", "-6", "@{upstream}..HEAD")
        commits = log.stdout.strip().splitlines() if log.returncode == 0 else []
    except Exception:
        return None

    diverged = behind > 0
    what = ("the remote has moved on and this box has NOT — the push is being "
            "rejected and every further write will strand here"
            if diverged else
            "the last push did not land")
    return {"key": "sync", "label": "Unpushed corpus writes",
            "count": ahead, "behind": behind, "diverged": diverged,
            "commits": commits[:5],
            "desc": f"{ahead} commit(s) exist only on this machine — {what}. "
                    "These nodes are not on GitHub and not on any other box. "
                    "Fix: `git pull --rebase` then `git push` as the repo owner."}


# ── destructive writes ─────────────────────────────────────────────────────────
# Everything before this date was adjudicated by hand on 2026-08-10 (session
# The emptied-nodes session): 11 flagged modifications, all explained — seven
# append-headed body replacements, two JSON-encoded bodies, two deliberate audit
# commits of the owner's own. Reporting them forever would make this flag noise, and
# a flag nobody reads is the failure mode this whole check exists to avoid
# (the fail-loudly rule §3). MOVE THIS DATE when a later batch is adjudicated
# — never widen the thresholds to silence something instead.
DESTRUCTIVE_BASELINE = "2026-08-10"
# Tuned against the real corpus, not guessed: at these values the sweep returns
# all 11 known incidents across 777 commits and nothing else. Loosening to
# (>=6, 2x) adds only already-explained commits; tightening past 15 lines starts
# dropping real ones (the smallest true positive removed 20 lines).
DESTRUCTIVE_MIN_LINES = 15
DESTRUCTIVE_RATIO = 3


def compute_destructive_writes(root, since: str | None = None):
    """Commits that removed far more than they added — content loss, in general.

    THE GAP THIS CLOSES. On 2026-08-10 seven nodes were found with their bodies
    destroyed, ~119,000 characters, undetected for twelve days and 309 commits
    (the destructive-write pattern). Two instruments were tried and only one worked:

      - comparing each node's PEAK size against HEAD misses any loss a later
        write grows back past — which is exactly how two JSON-encoded health
        nodes stayed hidden after the first sweep declared the corpus clean;
      - asking which COMMITS removed far more than they added found both classes,
        because it looks at the write rather than at the surviving file.

    So this is the second one. It is deliberately not specific to that bug: the
    append-headed shape now cannot happen at all (store.py refuses it), and the
    next loss will look like something nobody has thought of yet.

    Cheap enough for the read path: ONE `git log --numstat` over all history
    measures at ~0.1s on the live corpus, because the cost is one process rather
    than one per revision.
    Measured on the author's private corpus; not reproducible from the example corpus.

    `--diff-filter=M` restricts this to MODIFICATIONS. A deleted node is not a
    silent loss — delete_node is gated four ways and leaves an obvious commit —
    and including deletions put nine scratch-probe removals in the output.

    Returns a flag dict, or None when there is nothing to say. Never raises: an
    audit that dies because git is unusual is worse than one that stays quiet.
    """
    # Resolved at CALL time, not bound as a default: a module constant used as a
    # default argument freezes at import and cannot be moved without a restart —
    # and this one is meant to be moved every time a batch is adjudicated.
    since = since or DESTRUCTIVE_BASELINE
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "log", f"--since={since}", "--numstat",
             "--diff-filter=M", "--format=@@%h|%ad|%s", "--date=short",
             "--", "*.md"],
            capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            return None                      # not a repo, or no history — absent
    except Exception:
        return None

    cur, hits = None, []
    for line in r.stdout.splitlines():
        if line.startswith("@@"):
            cur = line[2:].split("|", 2)
            continue
        parts = line.split("\t")
        if cur is None or len(parts) != 3:
            continue
        ins, dele, path = parts
        if not (ins.isdigit() and dele.isdigit()):
            continue                          # binary, or a rename line
        ins, dele = int(ins), int(dele)
        if dele >= DESTRUCTIVE_MIN_LINES and dele >= DESTRUCTIVE_RATIO * max(ins, 1):
            hits.append({"commit": cur[0], "date": cur[1],
                         "file": path.split("/")[-1],
                         "removed": dele, "added": ins,
                         "subject": cur[2][:80]})
    if not hits:
        return None

    hits.sort(key=lambda h: h["removed"] - h["added"], reverse=True)
    return {"key": "destructive", "label": "Writes that removed far more than they added",
            "count": len(hits), "since": since, "commits": hits[:8],
            "desc": f"{len(hits)} commit(s) since {since} removed at least "
                    f"{DESTRUCTIVE_MIN_LINES} lines while adding under a "
                    f"{DESTRUCTIVE_RATIO}rd as many. Some are legitimate rewrites "
                    "— this reports a SHAPE, not a verdict. Check each against "
                    "`git show <commit>`, and once a batch is explained move "
                    "audit.DESTRUCTIVE_BASELINE forward rather than loosening "
                    "the thresholds."}


# ── deploy health ──────────────────────────────────────────────────────────────
def compute_deploy_health():
    """The last deploy's post-restart probe verdict, when it failed.

    okf-deploy.sh probes both services over loopback after every restart —
    behavioural (an HTTP answer), not systemd unit state, because a unit can
    be `active` around a frozen app (the 2026-08-07 event-loop stall). On
    failure it leaves a marker file; any later healthy probe removes it. This
    surfaces the marker as a flag, because the probe's own journal is a place
    nobody reads — both 2026-08 crash-loops were caught only because a human
    happened to be watching the deploy (the fail-soft-is-two-programs pattern, fourth
    instance; built on the owner's 2026-08-08 ruling).

    A marker that exists but cannot be parsed is STILL a flag, phrased as
    unreadable — a verdict that cannot be verified is a failure, never a
    quiet pass (the fail-loudly rule). Absent marker → None (the Mac and
    every healthy box). Never raises.
    """
    path = Path(os.environ.get("OKF_DEPLOY_HEALTH", "")
                or Path.home() / ".okf" / "deploy_health.json")
    try:
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        failed = [str(f) for f in (raw.get("failed") or [])]
        when = int(raw.get("when") or 0)
        ago = f"{max(0, int(time.time()) - when) // 60} min ago" if when \
            else "at an unknown time"
        return {"key": "deploy", "label": "Last deploy unhealthy",
                "failed": failed, "revision": raw.get("revision"), "when": when,
                "desc": f"The post-deploy probe ({ago}, revision "
                        f"{raw.get('revision') or '?'}) got no HTTP answer "
                        f"from: {', '.join(failed) or 'unrecorded'}. The "
                        "service may be crash-looping or frozen — check "
                        "`journalctl -u okf -u okf-web` on the VPS. The "
                        "marker clears itself on the next healthy probe."}
    except Exception:
        return {"key": "deploy", "label": "Last deploy unhealthy",
                "failed": [], "revision": None, "when": None,
                "desc": f"A deploy-health marker exists at {path} but could "
                        "not be read — treat as a failed deploy until "
                        "checked; a probe verdict that cannot be verified is "
                        "a failure, not a pass."}


def compute_deploy_gate():
    """A revision the deploy TEST GATE refused, while it stands refused.

    Since 2026-09-07 okf-deploy.sh runs the test suite against the incoming
    revision (in a throwaway worktree, against the corpus this box holds) and
    only then fast-forwards and restarts. A red suite leaves this marker and the
    box keeps running the previous revision — which is the safe outcome, but a
    silent one: master has moved, the VPS has not, and nothing else says so.
    The external review (2026-09-06) found four failing tests and no CI on the
    path between `git push` and a restarted service.

    Distinct from the probe marker on purpose: the probe self-clears whenever
    the services answer, and here the services are healthy BY DESIGN (they never
    restarted). This clears only when a later revision passes the gate.
    """
    path = Path(os.environ.get("OKF_DEPLOY_GATE", "")
                or Path.home() / ".okf" / "deploy_gate.json")
    try:
        if not path.is_file():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        when = int(raw.get("when") or 0)
        ago = f"{max(0, int(time.time()) - when) // 60} min ago" if when \
            else "at an unknown time"
        rev = raw.get("revision") or "?"
        return {"key": "deploy_gate", "label": "Deploy refused by the test gate",
                "revision": rev, "running": raw.get("running"), "when": when,
                "summary": raw.get("summary") or "",
                "desc": f"Revision {rev} failed the test suite on the VPS ({ago}); "
                        f"the box is still running {raw.get('running') or '?'} and "
                        f"will not deploy this revision again. {raw.get('summary') or ''} "
                        "Fix the tests and push: the next commit is re-tested. Log: "
                        "/home/okf/.okf/deploy_gate.log on the VPS."}
    except Exception:
        return {"key": "deploy_gate", "label": "Deploy refused by the test gate",
                "revision": None, "running": None, "when": None, "summary": "",
                "desc": f"A deploy-gate marker exists at {path} but could not be "
                        "read — treat as a refused deploy until checked; a gate "
                        "verdict that cannot be verified is a failure, not a pass."}


# ── health ─────────────────────────────────────────────────────────────────────
def compute_health(nodes, by_id, root=None):
    """Structural health flags. Returns {"summary": {...}, "flags": [...]}."""
    deg = {n.id: 0 for n in nodes}
    for n in nodes:
        for r in (n.fm.get("relates_to") or []):
            if r in by_id:
                deg[n.id] += 1
                deg[r] += 1
    active = [n for n in nodes if n.fm.get("status") == "active"]

    def neighbours(n):
        s = set()
        for r in (n.fm.get("relates_to") or []):
            if r in by_id:
                s.add(r)
        for m in nodes:
            if n.id in (m.fm.get("relates_to") or []):
                s.add(m.id)
        return s

    orphans = [_brief(n) for n in active if deg[n.id] == 0]

    by_src = defaultdict(list)
    for n in active:
        k = _coda_key(n.fm.get("source"))
        if k:
            by_src[k].append(n)
    dup_groups = [[_brief(m) for m in g] for g in by_src.values() if len(g) > 1]

    stale = []
    for n in active:
        nb = neighbours(n)
        if not nb or any(by_id[i].fm.get("status") == "active" for i in nb):
            continue
        # A node WRITTEN AFTER its neighbours were retired is not drifting — it is
        # pointing at them deliberately. The archetype is a post-mortem: a node
        # documenting an incident links to the thing retired in that incident, so
        # its only edge is dead the moment it is born, and it was flagged as
        # possibly-stale while being the newest node in the corpus (2026-08-02,
        # The body-replace incident). Staleness is drift, and drift takes time.
        #
        # `created_at` rather than `updated_at`: created_at is stable, while
        # updated_at moves on any backlink write, which would let an automatic
        # edge silently mark a node as "reviewed since". Day granularity means a
        # node created the same day as the retirement is treated as knowing —
        # deliberately, since a missed advisory costs a glance and a false one
        # trains the reader to skip the whole flag (the fail-loudly rule rule 3).
        retired_at = max((str(by_id[i].fm.get("updated_at") or "") for i in nb),
                         default="")
        born = str(n.fm.get("created_at") or "")
        if born and retired_at and born >= retired_at:
            continue
        stale.append(_brief(n))

    by_title = defaultdict(list)
    for n in active:
        by_title[_norm_title(n.fm.get("title", ""))].append(n)
    title_groups = [[_brief(m) for m in g] for g in by_title.values() if len(g) > 1]

    # Live nodes with no `asks` (2026-10-14). The the asks-and-findability decision
    # puts asks on every node — the measured lever for a stranger's wording,
    # roughly doubling findability on held-out questions — and create_node
    # silently dropped the field for two weeks: 13 nodes were born without it
    # and nothing said so. This is the something that says so.
    no_asks = [_brief(n) for n in nodes if n.fm.get("status") in ("active", "parked")
               and not n.fm.get("asks")]

    flags = []
    if no_asks:
        flags.append({"key": "asks", "label": "No asks", "nodes": no_asks,
                      "desc": "Live nodes with no `asks` — a stranger's wording cannot "
                              "reach them. Add six to eight everyday phrasings with "
                              "update_node (changes: {asks: [...]}), written from the "
                              "body in the asker's words, not the node's."})
    if orphans:
        flags.append({"key": "orphans", "label": "Orphans", "nodes": orphans,
                      "desc": "Active nodes with no links — connect them, or delete if not worth keeping."})
    if dup_groups:
        flags.append({"key": "dup", "label": "Same source", "groups": dup_groups,
                      "desc": "Multiple active nodes from the same Coda page — likely redundant; merge or supersede."})
    if stale:
        flags.append({"key": "stale", "label": "Stale neighbourhood", "nodes": stale,
                      "desc": "Active nodes whose every link points to a retired/superseded node — may be stale too."})
    if title_groups:
        flags.append({"key": "title", "label": "Near-identical titles", "groups": title_groups,
                      "desc": "Active nodes sharing a normalised title — possible duplicates."})
    if root is not None:
        pend = [_brief(by_id[c["id"]]) for c in load_purge(root) if c.get("id") in by_id]
        if pend:
            flags.append({"key": "purge", "label": "Proposed for deletion", "nodes": pend,
                          "desc": "Agents proposed these for deletion because delete_node's gates "
                          "blocked them (live, linked, canonical, or in a supersession chain); "
                          "a human reviews and deletes them in the graph editor, or clears the proposal."})
        # the weekly findability run (2026-10-14): failed, stopped or regressed.
        # Appended, not inserted first — a retrieval regression is worth reading,
        # but it is not a durability or a deploy problem. Imported here because
        # findability imports context, which imports this module.
        try:
            try:
                import findability as _fz
            except ImportError:
                from librarian import findability as _fz
            fz = _fz.compute_flag()
        except Exception:
            fz = None
        if fz:
            flags.append(fz)
        gate = compute_deploy_gate()
        if gate:
            flags.insert(0, gate)
        deploy = compute_deploy_health()
        if deploy:
            # Above the gate: a service that does not ANSWER outranks code that
            # merely did not ship.
            flags.insert(0, deploy)
        destructive = compute_destructive_writes(root)
        if destructive:
            # Above deploy health: a service that is down is visible and will be
            # fixed, whereas content already removed is invisible by construction
            # and gets harder to recover the longer it sits. Below the wedge only
            # because the wedge says the corpus exists in exactly one place.
            flags.insert(0, destructive)
        wedge = compute_sync(root)
        if wedge:
            # Durability outranks tidiness: every other flag says the corpus is
            # untidy, this one says part of it exists in only one place. It reads
            # first or it gets skimmed past. (And ahead of deploy health — data
            # in one place outranks a service that is down.)
            flags.insert(0, wedge)
    retired = sum(1 for n in nodes if n.fm.get("status") in ("retired", "superseded"))
    summary = {"total": len(nodes), "active": len(active),
               "retired": retired, "flags": len(flags)}
    try:
        try:
            import findability as _fz
        except ImportError:
            from librarian import findability as _fz
        line = _fz.summary_line()
        if line:
            summary["findability"] = line
    except Exception:
        pass
    return {"summary": summary, "flags": flags}


# ── link suggestions ───────────────────────────────────────────────────────────
def existing_links(active, aset):
    s = set()
    for n in active:
        for r in (n.fm.get("relates_to") or []):
            if r in aset:
                s.add(frozenset((n.id, r)))
    return s


def compute_suggestions(active, existing, dismissed, curated, limit=14):
    total = max(len(active), 1)
    # a tag that merely names a node type (session, note, playbook, …) is
    # structural, not topical — two nodes "both being sessions" is no reason to
    # link them. Exclude these from the tag signal entirely (score + reason).
    type_tags = {str(n.fm.get("type", "")).lower() for n in active}
    tagmap = {n.id: _tags(n) for n in active}
    tokmap = {n.id: _tokens(n) for n in active}
    tag_df, tok_df = Counter(), Counter()
    for n in active:
        for t in tagmap[n.id]:
            tag_df[t] += 1
        for t in tokmap[n.id]:
            tok_df[t] += 1

    def idf(df):
        return math.log((total + 1) / (df + 0.5))

    briefs = {n.id: _brief(n) for n in active}
    ids = [n.id for n in active]
    cand = {}
    for i in range(len(ids)):
        a = ids[i]
        for j in range(i + 1, len(ids)):
            b = ids[j]
            key = frozenset((a, b))
            if key in existing or key in dismissed:
                continue
            shared_tags = (tagmap[a] & tagmap[b]) - type_tags  # structural type tags carry no signal
            spec = [t for t in shared_tags if tag_df[t] < 0.5 * total]  # drop ~project-wide tags
            shared_tok = tokmap[a] & tokmap[b]
            if not spec and len(shared_tok) < 3:
                continue
            score = (sum(idf(tag_df[t]) for t in shared_tags)
                     + 0.4 * sum(idf(tok_df[t]) for t in shared_tok))
            if score < 2.4:
                continue
            if spec:
                spec.sort(key=lambda t: tag_df[t])  # rarest (most specific) first
                reason = "Both tagged " + ", ".join(spec[:3])
            else:
                top = sorted(shared_tok, key=lambda t: tok_df[t])[:3]
                reason = "Overlapping topics: " + ", ".join(top)
            cand[key] = {"a": briefs[a], "b": briefs[b], "reason": reason,
                         "score": round(score, 2), "source": "auto"}

    for c in curated:  # human/LLM-curated pairs override auto reason and rank first
        a, b = c["a"], c["b"]
        if a not in briefs or b not in briefs:
            continue
        key = frozenset((a, b))
        if key in existing or key in dismissed:
            continue
        base = cand.get(key, {}).get("score", 0)
        cand[key] = {"a": briefs[a], "b": briefs[b], "reason": c.get("reason", "Related."),
                     "score": round(base + 6, 2), "source": "curated"}

    # curated (LLM-verified via propose_link) always rank above raw auto suggestions,
    # so human-vetted links surface first instead of being buried under high-IDF noise.
    ranked = sorted(cand.values(), key=lambda d: (d.get("source") != "curated", -d["score"]))
    per_node, out = Counter(), []
    for d in ranked:  # spread suggestions so one hub node can't dominate the list
        ia, ib = d["a"]["id"], d["b"]["id"]
        if per_node[ia] >= 3 or per_node[ib] >= 3:
            continue
        per_node[ia] += 1
        per_node[ib] += 1
        out.append(d)
        if len(out) >= limit:
            break
    return out


# ── the vector-link trial (2026-10-15) ─────────────────────────────────────────
# compute_suggestions above sees only shared tags and title/summary words, so a
# pair of nodes that share neither can never be proposed. Measured 2026-10-15
# (the section-search session) on 2,579 existing links: 145 of them are
# word-invisible in exactly that sense. Node vectors, with each node TYPE's mean
# vector subtracted (raw vectors matched document genre — 11 of their top 12
# candidates were session↔session pairs), put 14% of those in a node's top 10
# and 24% in its top 25, where the word channel has 0%. On all links the word
# channel is better (41% vs 36% in the top 10), so vectors are a SECOND channel,
# not a replacement, and only for the pairs the first cannot see.
# Measured on the author's private corpus; not reproducible from the example corpus.
#
# The owner ruled a TRIAL, not a feature: up to VECTOR_SLOTS such pairs ride in the
# suggestion list, labelled, and his accept/dismiss rulings score it. A
# word-invisible pair can only have come from this channel, so resolve_suggestion
# records every ruling on one in .okf/vector_trial.json; the audit summary
# reports the running score. Computed only in a warmed service, from vectors it
# already holds — a test or script never loads the model for it.
VECTOR_SLOTS = 3
VECTOR_REASON = ("Related in meaning, but no shared tags or words — the vector-link "
                 "trial: accept or dismiss to score it")


def _word_maps(active):
    total = max(len(active), 1)
    type_tags = {str(n.fm.get("type", "")).lower() for n in active}
    tagmap = {n.id: _tags(n) for n in active}
    tokmap = {n.id: _tokens(n) for n in active}
    tag_df = Counter()
    for n in active:
        tag_df.update(tagmap[n.id])
    return total, type_tags, tagmap, tokmap, tag_df


def _invisible(a, b, maps) -> bool:
    """No specific shared tag and fewer than 3 shared title/summary words — the
    exact condition under which compute_suggestions can never propose a pair."""
    total, type_tags, tagmap, tokmap, tag_df = maps
    spec = [t for t in (tagmap[a] & tagmap[b]) - type_tags if tag_df[t] < 0.5 * total]
    return not spec and len(tokmap[a] & tokmap[b]) < 3


def pair_is_word_invisible(nodes, a, b) -> bool:
    active = [n for n in nodes if n.fm.get("status") == "active"]
    ids = {n.id for n in active}
    return a in ids and b in ids and _invisible(a, b, _word_maps(active))


def compute_vector_suggestions(active, existing, dismissed, k=VECTOR_SLOTS, vectors=None):
    """Up to k unlinked, undismissed, word-invisible pairs, ranked by type-centred
    node-vector cosine, at most one per node. [] whenever vectors are unavailable."""
    try:
        import numpy as np
        if vectors is None:
            try:
                import embed
            except ImportError:
                from librarian import embed
            if not embed._STATE.get("serving"):
                return []
            vectors = embed.node_vectors(active)
        if not vectors or k <= 0:
            return []
        act = [n for n in active if n.id in vectors]
        if len(act) < 2:
            return []
        V = np.stack([np.asarray(vectors[n.id], dtype="float32") for n in act])
        types = [n.fm.get("type") for n in act]
        for t in set(types):
            m = [i for i, x in enumerate(types) if x == t]
            V[m] -= V[m].mean(axis=0)
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        S = V @ V.T
        iu = np.triu_indices(len(act), 1)
        maps = _word_maps(active)
        briefs, used, out = {n.id: _brief(n) for n in act}, set(), []
        for j in np.argsort(-S[iu]):
            a, b = act[iu[0][j]].id, act[iu[1][j]].id
            if a in used or b in used:
                continue
            key = frozenset((a, b))
            if key in existing or key in dismissed or not _invisible(a, b, maps):
                continue
            used |= {a, b}
            out.append({"a": briefs[a], "b": briefs[b], "reason": VECTOR_REASON,
                        "score": round(float(S[iu[0][j], iu[1][j]]), 3), "source": "vector"})
            if len(out) >= k:
                break
        return out
    except Exception:
        return []                       # the trial must never break the audit


def load_vector_trial(root) -> list:
    try:
        d = json.loads(_meta_path(root, "vector_trial.json").read_text(encoding="utf-8"))
        return [r for r in d.get("rulings", []) if isinstance(r, dict)]
    except Exception:
        return []


def record_vector_trial(root, a, b, outcome) -> Path:
    """Append one ruling on a word-invisible pair. Caller commits."""
    rows = load_vector_trial(root)
    rows.append({"a": a, "b": b, "outcome": outcome,
                 "at": time.strftime("%Y-%m-%d", time.gmtime())})
    meta = Path(root) / ".okf"
    meta.mkdir(exist_ok=True)
    path = meta / "vector_trial.json"
    path.write_text(json.dumps({"rulings": rows}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return path


def vector_trial_line(root) -> str:
    rows = load_vector_trial(root)
    acc = sum(r.get("outcome") == "accept" for r in rows)
    dis = sum(r.get("outcome") == "dismiss" for r in rows)
    if not rows:
        return "no rulings yet (started 2026-10-15)"
    return f"{acc} accepted, {dis} dismissed of {len(rows)} ruled ({acc / len(rows):.0%})"


# ── .okf meta (dismissed / curated pairs) ──────────────────────────────────────
def _meta_path(root, name):
    return Path(root) / ".okf" / name


def load_dismissed(root):
    try:
        d = json.loads(_meta_path(root, "dismissed.json").read_text(encoding="utf-8"))
        return {tuple(p) for p in d.get("pairs", []) if isinstance(p, list) and len(p) == 2}
    except Exception:
        return set()


def load_curated(root):
    try:
        d = json.loads(_meta_path(root, "suggestions.json").read_text(encoding="utf-8"))
        return [p for p in d.get("pairs", []) if isinstance(p, dict) and p.get("a") and p.get("b")]
    except Exception:
        return []


def save_dismissed(root, pairs):
    """Persist dismissed pairs. Caller commits (git side-effects stay out of here)."""
    meta = Path(root) / ".okf"
    meta.mkdir(exist_ok=True)
    (meta / "dismissed.json").write_text(
        json.dumps({"pairs": [list(p) for p in sorted(pairs)]}, ensure_ascii=False),
        encoding="utf-8")


def load_purge(root):
    """Nodes an agent PROPOSED for deletion — awaiting a human's approval in the editor.
    Agents delete only through delete_node's four gates (since 2026-08-02); this list is
    the queue of nodes those gates blocked, which only a human may act on."""
    try:
        d = json.loads(_meta_path(root, "purge_candidates.json").read_text(encoding="utf-8"))
        return [c for c in d.get("candidates", []) if isinstance(c, dict) and c.get("id")]
    except Exception:
        return []


def propose_purge(root, node_id, reason=""):
    """Record a delete PROPOSAL for a node the delete gates block. Idempotent per id. Caller commits."""
    cands = load_purge(root)
    if not any(c.get("id") == node_id for c in cands):
        cands.append({"id": node_id, "reason": reason})
    meta = Path(root) / ".okf"
    meta.mkdir(exist_ok=True)
    (meta / "purge_candidates.json").write_text(
        json.dumps({"candidates": cands}, ensure_ascii=False), encoding="utf-8")
    return cands


def purge_blockers(node_id, nodes, by_id) -> list[str]:
    """Which purge gates a specific node FAILS — empty list means safe to delete.

    Same gates as purge_candidates, but per-node and stated as reasons, because a
    refusal has to tell the caller what to do about it. `delete_node` is agent-
    callable (2026-08-02) and these gates are the whole safety story: an
    agent cannot reach a live, linked, canonical or supersession-target node, so
    a misread instruction cannot remove load-bearing knowledge. Deletion is
    recoverable from git — verified by restoring a deleted node months later, through a
    repo split that rewrote every hash — but recoverable is not the same as
    NOTICED, which is why the gates matter rather than the undo.
    """
    n = by_id.get(node_id)
    if n is None:
        return [f"no node with id '{node_id}'"]
    blockers = []
    if n.fm.get("status") != "retired":
        blockers.append(
            f"status is '{n.fm.get('status')}', not 'retired' — retire it first "
            "(update_node status=retired); retirement is reversible, deletion needs a gate")
    if n.fm.get("canonical"):
        blockers.append("node is canonical — canonical knowledge is never agent-deletable")
    live_out = [r for r in (n.fm.get("relates_to") or []) if r in by_id]
    if live_out:
        blockers.append(f"still points at {len(live_out)} live node(s): {', '.join(live_out[:3])}")
    inbound = [m.id for m in nodes
               if node_id in (m.fm.get("relates_to") or []) and m.id != node_id]
    if inbound:
        blockers.append(f"{len(inbound)} node(s) still link TO it: {', '.join(inbound[:3])}")
    # BOTH ends of a chain are protected. Deleting the predecessor orphans the
    # explanation of why the successor exists; deleting the SUCCESSOR silently
    # restores the predecessor to active (store.delete_node does that cleanup by
    # design), which resurrects superseded knowledge without anyone asking.
    sup = [m.id for m in nodes if m.fm.get("supersedes") == node_id]
    if sup or n.fm.get("superseded_by") or n.fm.get("supersedes"):
        blockers.append("it is part of a supersession chain — deleting either end "
                        "either orphans the explanation of why the successor exists, "
                        "or silently restores the superseded node to active")
    return blockers


def purge_candidates(nodes, by_id):
    """Retired nodes structurally safe to REMOVE from the corpus: orphaned (no live
    in- or out-edge), non-canonical, and not a supersession target. Read-only proposer
    — the same four gates delete_node enforces, so an agent may delete these directly
    (since 2026-08-02, the gated-deletion decision) or a human may in the graph editor. Conservative by
    construction: a *false* (not a candidate) is always
    safe; every gate below must pass. Retirement stays the default for once-true
    knowledge — this only surfaces nodes created in error that have no reader and break
    no chain."""
    inbound, sup_targets = set(), set()
    for n in nodes:
        for r in (n.fm.get("relates_to") or []):
            inbound.add(r)
        if n.fm.get("supersedes"):
            sup_targets.add(n.fm["supersedes"])          # this node's target
        if n.fm.get("superseded_by"):
            sup_targets.add(n.id)                        # this node IS a target
    out = []
    for n in nodes:
        if n.fm.get("status") != "retired" or n.fm.get("canonical"):
            continue                                     # never live/parked or canonical
        if [r for r in (n.fm.get("relates_to") or []) if r in by_id]:
            continue                                     # still points somewhere
        if n.id in inbound or n.id in sup_targets:
            continue                                     # pointed at, or explains a successor
        d = _brief(n)
        d["reason"] = "retired · orphaned · non-canonical · not a supersession target"
        out.append(d)
    return out


# ── convenience: full triage in one call (used by the MCP audit_corpus tool) ────
def audit(nodes, by_id, root, limit=14):
    active = [n for n in nodes if n.fm.get("status") == "active"]
    aset = {n.id for n in active}
    existing = existing_links(active, aset)
    dismissed = {frozenset(p) for p in load_dismissed(root) if p[0] in aset and p[1] in aset}
    sug = compute_suggestions(active, existing, dismissed, load_curated(root), limit=limit)
    sug += compute_vector_suggestions(active, existing, dismissed,
                                      k=min(VECTOR_SLOTS, max(limit, 0)))
    health = compute_health(nodes, by_id, root)
    purge = purge_candidates(nodes, by_id)
    # Search outcomes, surface-correct (ruled 2026-08-07, refined same day).
    # Carried in the WEEKLY digest rather than a log file nobody opens. Roles:
    # the WEB miss trend is regression-readable (that surface has an open
    # signal); the MCP suspected queue is a DISCOVERY QUEUE — adjudication
    # candidates, contaminated by diagnostic traffic by design — never a
    # quality metric. Regression evidence is the benchmark, adjudicated
    # samples, and semantic:false incidence (the fail-loudly rule: an
    # instrument that cannot say "unknown" lies). Best-effort — a missing or
    # unreadable log must never break the audit.
    from . import searchlog
    try:
        searches = searchlog.outcomes()
    except Exception:
        searches = searchlog.empty_outcomes()

    return {"summary": {**health["summary"], "purge_candidates": len(purge),
                        "web_search_misses": searches["web"]["misses"],
                        "mcp_suspected_misses": searches["mcp"]["suspected_misses"],
                        "vector_link_trial": vector_trial_line(root)},
            "flags": health["flags"], "suggestions": sug, "suggestion_count": len(sug),
            "purge_candidates": purge, "searches": searches}
