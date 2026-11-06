"""Context Assembler (PRD §6) — deterministic retrieval for the Compiler's
briefing reads: "what do we know about X" / "brief me before Y", answered from
the corpus rather than one chat's memory.

Given a query it finds the most relevant LIVE nodes (seeds, with their bodies),
pulls in their one-hop neighbourhood (related, as briefs), and reports any
retired/superseded nodes it deliberately excluded. No LLM: it assembles the
*material*; the caller (a Claude in chat, or the ambient runtime) writes the
prose brief. Mirrors librarian.audit — the deterministic core hands back
structured data, the LLM presents it. Retrieval is IDF-weighted term overlap over
title, summary, tags AND body; no vectors (PRD defers RAG, deterministic is the
ethos).

Bodies were NOT indexed until 2026-08-04, on the reasoning that nodes are
well-titled and well-summarised so precision would be high. That was true and
beside the point: the failure mode was recall, not precision. A word appearing
only in a node's prose returned zero hits — verified against the live corpus —
so a half-remembered phrase from inside a document was unfindable, which is
most of what anyone actually searches for. Body terms carry BODY_WEIGHT rather
than full weight, so a passing mention does not outrank the node the subject is
about.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

try:  # dual import: root (`python server.py`) or package
    from audit import _STOP
    from store import split_doc
    import embed
    import sections
    import summary
except ImportError:
    from librarian.audit import _STOP
    from librarian.store import split_doc
    from librarian import embed, sections, summary

_WORD = re.compile(r"[a-z][a-z0-9-]{2,}")

# Measured on the author's private corpus; not reproducible from the example corpus.
# How much a term found only in a node's BODY counts, relative to one in its
# summary. This interacts with the 1.5 relevance floor to decide which body words
# are findable at all, so it was MEASURED over the live corpus (285 live nodes)
# rather than chosen — 40 body-only words sampled per document-frequency band:
#
#   weight     df=1     df 2-4   df 5-15   df 16-40      verdict
#   0.30      39/40      4/40      0/40      0/12    only unique strings; useless
#   0.40      39/40     40/40     23/40      0/12    patchy in the middle band
#   0.50      39/40     40/40     40/40      1/12    ← saturates recall, rejects noise
#   0.65      39/40     40/40     40/40     11/12    admits words in 6-14% of nodes
#
# 0.5 is the only value that finds every reasonably specific body word while still
# refusing the common ones. Note the first probe run used df=1 terms ONLY and
# scored 40/40 at EVERY weight — an incomplete population that would have made any
# choice look correct (the verification-discipline guardrail failure mode 3).
BODY_WEIGHT = 0.5

# The relevance floor. It is relative in effect: a body-only word clears it while
# it sits in roughly 5% of live nodes or fewer (BODY_WEIGHT × idf ≥ 1.5). Below
# about 50 live nodes that bound falls under two nodes, so a word in just two
# bodies — the df 2-4 band the table above finds in full — dropped out of every
# result whenever the model was off, and a new corpus is always that small. The
# floor never exceeds what a two-node body match scores; from 50 live nodes up
# that is above 1.5, so the floor is the plain 1.5 and ranking is unchanged.
LEX_FLOOR = 1.5


def lex_floor(total: int) -> float:
    return min(LEX_FLOOR, BODY_WEIGHT * math.log((total + 1) / 2.5))

# ── semantic blend (v2 embeddings, 2026-08-06) — values MEASURED, see below ──
# SEM_BASE: cosine below this contributes nothing and cannot admit a candidate.
# SEM_WEIGHT: worth of the per-query-strongest semantic match, relative to the
#             per-query-strongest lexical match (components are normalised to
#             their own max before blending — see the ranking block for why).
# SEM_TOPK: at most this many semantic-only candidates join the pool per query.
# LEX_REF: absolute floor for the lexical normaliser (see the ranking block).
#
# MEASURED on the 57-question dev set, 310 live nodes, 2026-08-06 (top-1 /
# top-5 / recall@15 / paraphrase-subset top-5 of 12). Lexical-only baseline
# with the doorway fix: 67% / 75% / 82% / 3.
# Measured on the author's private corpus; not reproducible from the example corpus.
#
#   weight    0.4      0.6      0.8      1.0      1.25     1.5
#   top-1     68       67-68    70-72    72       68       68
#   top-5     77       79-81    81-82    86       88       86
#   r@15      88-89    89       89-91    91       91       91
#   para t5   3        4-5      5-6      7        7        7
#
# 1.0 is the knee: past it, top-5 buys nothing further and top-1 pays (the
# record intent starts losing exact matches to semantically-plausible
# neighbours). LEX_REF swept 4-8 at w=1.0: flat on aggregates (72/86/91
# throughout), kept at 5.0 for the weak-lexical case it provably protects.
# SEM_BASE swept 0.45-0.60: 0.5 dominant or tied everywhere. Latency 10ms/q
# lexical → ~19ms/q hybrid (query embed included).
SEM_BASE = 0.5
SEM_WEIGHT = 1.0
SEM_TOPK = 10
# The cap was absolute, and the corpus grew past it (2026-09-12 diagnosis): the
# cap-table record for a paraphrase query sat at semantic rank 4 of 318 live in
# August and rank 16 of 490 in September with an UNCHANGED cosine — twelve newer
# nodes scored fractionally higher — so it never entered the pool at all. A
# fixed ten admits a shrinking fraction of the corpus; this admits the larger
# of the fixed floor and a fraction of the live count (25 at 490). Measured on
# both corpus states: the vanished answer returns, nothing else moves.
# Measured on the author's private corpus; not reproducible from the example corpus.
SEM_TOPK_FRAC = 0.05
# Measured on the author's private corpus; not reproducible from the example corpus.
# RAISED 5 → 8 on 2026-10-14, by the first instrument that could see it. The
# August sweep (above) was flat because the benchmark's queries share words
# with their targets. The findability bank's stranger questions do not, by
# construction, and diagnosing its 33 misses outside fifteen found the
# largest cause was this blend: 18 targets ranked 1-15 by meaning (several at
# semantic rank 1-2, the strongest cosine in the corpus) were outranked by
# nodes holding a few incidental words, because a weak best lexical match was
# still normalised to full confidence. 10 more were never in the pool, and 5
# had words but no meaning. Swept on a read-only clone (552 live), tuning on
# half the bank and holding the other half out:
#
#   LEX_REF   bench t1/t5/t15   deep t8/t15   bank-A in8/in15   bank-B HELD
#   5         57/66/67          24/32         18/23             23/24
#   6.5       56/66/67          —             —                 —
#   8         56/66/67          23/31         22/29             25/27
#   12        55/66/67          23/31         23/30             25/27
#   (SEM_WEIGHT 1.25-2.0 bought less on the bank and cost more top-1)
#
# At 8: 27 of 227 queries up, 13 down; 7 enter the default pack of 8, 2 leave
# it; one benchmark rank-1 becomes rank 2 (and 6.5 costs that same one).
LEX_REF = 8.0

# ── section-level search (2026-10-13) — see librarian/sections.py ─────────────
# CHUNK_DISCOUNT: a chunk's cosine counts this much less than a node vector's
#   before the two compete for a node's semantic score (MEASURED, see below).
# HEAD_CHARS: when a seed's best-matching section lies beyond the opening, the
#   excerpt keeps this much of the opening (the node's own framing) and spends
#   the rest of body_chars on the matching sections instead of more opening.
#
# Measured on the author's private corpus; not reproducible from the example corpus.
# MEASURED 2026-10-13 on a read-only clone (550 live nodes, 6,048 chunks): the
# 67-entry benchmark (top-1/top-5/r@15, adjudicated) and the deep set
# (tests/deep_benchmark.yaml — 40 passages starting past char 4,000 of their
# node, a plain and a stranger question each; top-1/top-8/r@15, and "shown" =
# the passage visible in a top-8 excerpt):
#
#                 bench      plain        shown   stranger     shown   ms/q
#   chunks off    55/65/67   7/17/22      16      1/4/7        1       33
#   d = 0.00      56/64/67   8/20/25      18      2/5/10       2       42
#   d = 0.02      56/66/67   8/19/25      17      2/5/9        2       42
#   d = 0.03      57/66/67   8/19/24      17      2/5/8        2       42
#   d = 0.04      57/66/67   8/19/24      17      2/5/7        2       41
#   d = 0.06      57/66/67   7/18/23      16      1/5/8        2       —
#
# "shown" was 0 for every row before section excerpts — every target starts
# past the old 2,400-char cut. d = 0 lets long nodes' many chunks outbid (a
# benchmark top-5 lost); 0.03 is the best benchmark with most of the deep
# gain. Movers at 0.03 vs off: 6→1, 2→1 up; 2→3, 2→4, 7→8 down, all in the top 8.
CHUNK_DISCOUNT = 0.03
HEAD_CHARS = 600

# ── intent-aware ranking (2026-08-05, the intent-ranking work item next-cycle item 2) ─────────────────
# A query is not just words — it wants a KIND of answer. Measured on the
# 57-question benchmark before building: the engine's worst misses were
# wrong-TYPE winners, not wrong-topic ones — a session outranked the capture
# playbook on "how do I capture a session" (sessions are long and keyword-dense,
# the documented anti-pattern from the completeness-check spec), and an audit log outranked the type
# registry on a which-keys-are-legal question. Classification is DETERMINISTIC
# keyword rules — no LLM in the read path (the deterministic-read-path decision's ethos; the scheduled-runtime gate decision's no-key-on-the-
# hot-path property). The calling LLM reranks the shortlist itself; this makes
# the shortlist worth reranking.
# Measured on the author's private corpus; not reproducible from the example corpus.
#
# STRUCTURAL RULE, inherited from the reorder-never-delete pattern: priors REORDER,
# they never gate. The relevance floor is applied to the RAW score; the prior
# multiplies the ranking key only. A wrong intent guess can therefore demote a
# right answer, never delete it — recall is untouched by construction. (One
# honest caveat, measured 2026-08-07: the pack's LIMIT re-introduces a gate at
# the pack boundary — a lost prior can push a correct node out of the returned
# top-N even though it stays in the candidate pool.)
# Measured on the author's private corpus; not reproducible from the example corpus.
#
# MODAL CUES STAY — measured, 2026-08-07. The precision-first proposal (remove
# "can we"/"should i" etc. as sufficient evidence; abstain without concept
# words) was run as a three-variant sweep on the 57-entry benchmark and
# REFUTED: top-1 flat, top-5 48→47, r@15 52→51, paraphrase top-5 7→6, and
# misroutes UP 6→7 — because in an ORDERED table a deleted cue does not
# produce abstention, it produces FALL-THROUGH to a later cue (a "which …
# should I keep …" question fell from a correct current_ruling to a wrong
# reference via "which "). The one observed overfire ("what checker should I
# run…" → current_ruling) is accepted and pinned in the benchmark as a known
# misroute; the semantic layer ranks the right node first regardless
# (verified live). The regression gate is the misroute CEILING in
# test_intent_misroute_ceiling: cue edits must not add misroutes; abstention
# on a labeled query is free.
# Measured on the author's private corpus; not reproducible from the example corpus.
_INTENT_CUES = [
    # order matters: first match wins; more specific phrasings first. The order
    # and two cues below were corrected by MEASUREMENT, not intuition: v1 put
    # record before project_state ("how much ... actually built" routed to
    # record), and sent "which session shipped X" to reference off the word
    # "which" while the query literally said "session".
    # Measured on the author's private corpus; not reproducible from the example corpus.
    ("procedure",      ("how do i ", "how to ", "what has to pass", "the steps",
                        "the procedure", "checklist", "playbook for", "how should i ")),
    ("project_state",  ("actually run", "as actually", "actually built", "actually in",
                        "current state", "as built", "in production", "what is built",
                        "state of the", "where does", "where do the")),
    ("record",         ("who owns", "how much", "as of ", "the numbers", "figures",
                        "spreadsheet", "the source workbook", "list of everything",
                        "inventory of", "cap table", "share register")),
    ("history",        ("session", "what happened when", "what went wrong",
                        "what was the", "before this", "the story of", "what shipped",
                        "how did we end up", "what happened in", "what did we do",
                        "originally", "why did we", "why was", "how was it decided")),
    ("open_question",  ("what happened to the", "is it still parked", "still open",
                        "unresolved", "not yet decided", "been decided",
                        "decided yet", "ruled yet")),
    ("current_ruling", ("can the ", "can we ", "can i ", "should we", "should i",
                        "is it allowed", "why can't", "do we allow", "what's the rule",
                        "what is the rule", "are we allowed", "is there a rule",
                        "the policy on", "how do we stop", "who decides", "does the owner")),
    ("reference",      ("which ", "where is ", "where's ", "what holds",
                        "the definition of", "vocabulary", "schema for")),
]

# type multipliers per intent, applied to the RANKING key only. Coarse and
# rational on purpose — tuned priors would just memorise the 57-question
# benchmark (the verification-discipline guardrail: training on the test). The shape encodes two facts: the
# answer-type follows the question-type, and sessions/logs are RESIDUE — they
# cite the durable nodes, so for any non-history question the durable node
# should outrank the session that mentions it.
_TYPE_PRIORS = {
    "procedure":      {"playbook": 1.5, "pattern": 1.15, "session": 0.75, "log": 0.8},
    "record":         {"record": 1.6, "resource": 1.4, "session": 0.75, "log": 0.85},
    "history":        {"session": 1.4, "log": 1.2, "record": 1.1},
    "project_state":  {"project": 1.45, "reference": 1.2, "note": 1.1, "session": 0.85},
    "open_question":  {"note": 1.2, "log": 1.2, "decision": 1.1, "session": 0.85},
    "current_ruling": {"decision": 1.45, "pattern": 1.25, "playbook": 1.1,
                       "session": 0.7, "log": 0.8},
    "reference":      {"reference": 1.5, "playbook": 1.1, "record": 1.1,
                       "session": 0.75, "log": 0.75},
    # No intent detected: NO prior at all. v1 applied a "residue below durable
    # nodes" penalty here on the argument that it was universally true — the
    # benchmark refuted it (history top-1 fell 5 to 3): unclassified natural
    # questions skew toward "what did we do / the session where...", whose
    # correct answers ARE sessions. A prior that cannot tell which case it is
    # in must not bet.
    # Measured on the author's private corpus; not reproducible from the example corpus.
    None:             {},
}


# The decision DOORWAY (2026-08-06). Measured miss, not a hypothetical: "What is
# the current ruling on whether sensitive OKF nodes may be returned in broad
# retrieval results?" classified as None, so no prior fired and the canonical
# decision (the full-body-retrieval acceptance) sat FOURTH under two logs and a session — reproduced live
# before this was written. The cue table's current_ruling entries were all
# yes/no-question shapes ("can we", "is it allowed"); nothing recognised a query
# ABOUT a ruling by name. Rather than whack-a-mole phrases (the failure mode the
# second external review warns against), this is ONE combination rule: a
# decision-subject word AND a question structure, checked only AFTER the ordered
# cue table has had its say — so every measured routing above keeps its
# precedence, and "why did we decide" (history) and "been decided yet"
# (open_question) are claimed by their tables before this rule can see them.
# Deliberately not a general grammar: one intent had a doorway problem, one
# intent gets a combination rule. Generalise only if a second intent measurably
# shows the same failure (earned complexity — the rule-diet decision).
# Measured on the author's private corpus; not reproducible from the example corpus.
_RULING_SUBJECTS = ("ruling", "policy", "decided", "decide", "decision",
                    "allowed", "allow", "accepted", "accept", "rejected",
                    "reject", "prohibited", "permitted", "governs", "stance")
_RULING_STRUCTURES = ("what is", "what's", "whether", "is it", "is there",
                      "are we", "do we", "did we", "may ", "should ", "was ",
                      "currently")


def classify_intent_ex(query) -> tuple[str | None, list[str]]:
    """Deterministic, order-sensitive keyword classification, returning the
    intent AND the cue strings that fired — the signals are surfaced in the
    pack so a misclassification is diagnosable from the output alone, without
    reconstructing the cue table by hand."""
    q = " " + str(query).lower().strip() + " "
    for intent, cues in _INTENT_CUES:
        hit = [c for c in cues if c in q]
        if hit:
            return intent, [c.strip() for c in hit]
    subj = [s for s in _RULING_SUBJECTS if s in q]
    if subj:
        struct = [s for s in _RULING_STRUCTURES if s in q]
        if struct:
            return "current_ruling", [s.strip() for s in subj + struct]
    return None, []


def classify_intent(query) -> str | None:
    """Returns None when no cue matches — an unclassified query gets no prior."""
    return classify_intent_ex(query)[0]


def _tok(text) -> set[str]:
    return {t for t in _WORD.findall(str(text).lower()) if t not in _STOP}


def _tags(n) -> set[str]:
    return {str(t).lower().strip() for t in (n.fm.get("tags") or []) if str(t).strip()}


def _brief(n, *, full=True) -> dict:
    """Identity plus AUTHORITY (canonical/maturity/freshness — 2026-08-07,
    The owner's ruling on the third external review's strongest finding): the
    caller LLM is the judgment half of retrieval, and it was judging packs
    with the authority fields withheld. EXPOSED, never ranked on — a
    specialist checker legitimately outranks a canonical project map, so
    authority informs the reader of the pack, not the order of it."""
    u = n.fm.get("updated_at")
    return {"id": n.id, "title": n.fm.get("title", ""), "type": n.fm.get("type"),
            "project": n.fm.get("project", "—"),
            # a seed carries the whole summary; a related brief only its first
            # bullet, the one-line description (librarian/summary.py)
            "summary": (n.fm.get("summary", "") if full
                        else summary.first_line(n.fm.get("summary", ""))),
            "canonical": bool(n.fm.get("canonical")),
            "maturity": n.fm.get("maturity"),
            "updated_at": str(u) if u is not None else None}


def _neighbours(n, by_id, nodes):
    """Undirected relates_to neighbourhood (both stored directions)."""
    s = set()
    for r in (n.fm.get("relates_to") or []):
        if r in by_id:
            s.add(r)
    for m in nodes:
        if n.id in (m.fm.get("relates_to") or []):
            s.add(m.id)
    return s


# ── the term index, built once per corpus STATE ───────────────────────────────
# Tokenising 1.5 MB of bodies costs ~55ms and produces the identical result on
# every query, so it is cached. The key is a hash of the indexed text itself
# rather than a node count, an mtime or a date: `updated_at` has day resolution,
# so two writes to one node on the same day would leave a count-and-date key
# unchanged and serve a stale index — a cache whose invalidation is coarser than
# the thing it caches is a correctness bug, not a performance trade-off.
# Hashing the same bytes costs ~2ms, which is the price of never being wrong.
_INDEX: dict = {}


def _fingerprint(live) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for n in live:
        for part in (n.id, str(n.fm.get("title", "")), str(n.fm.get("summary") or ""),
                     repr(n.fm.get("tags") or []), repr(n.fm.get("asks") or []), n.body):
            h.update(part.encode("utf-8", "replace"))
            h.update(b"\x00")
    return h.digest()


def _index(live):
    """Return (field, df, total) for the live set, from cache when unchanged.

    `field[id]` is (title, summary, tags, body) token sets, kept APART so the same
    term scores by WHERE it appears. Merged, title and summary tied: "cap table"
    scored the actual cap-table record identically to a node that merely mentions
    one in prose, and tie order is arbitrary (2026-08-03).

    BODIES ARE INDEXED since 2026-08-04. Until then only title/summary/tags were,
    so a word appearing solely in a node's prose returned literally zero hits —
    measured against the live corpus, not inferred. The corpus exists to answer
    "what do we know about X" and the knowledge is in the bodies; a search that
    reads only the labels on the boxes is a catalogue, not a brain.
    Measured on the author's private corpus; not reproducible from the example corpus.
    """
    fp = _fingerprint(live)
    cached = _INDEX.get("live")
    if cached and cached[0] == fp:
        return cached[1]

    field, df = {}, Counter()
    for n in live:
        ti = _tok(n.fm.get("title", ""))
        # `asks` (v1.9) is indexed WITH the summary, weight 1.0 — the exact shape
        # the adoption measurement used (asks appended to the summary text).
        su = _tok(n.fm.get("summary") or "") | _tok(" ".join(str(a) for a in (n.fm.get("asks") or [])))
        tg = _tags(n)
        bo = _tok(n.body) - ti - su - tg
        field[n.id] = (ti, su, tg, bo)
        # df counts a term ONCE per node across ALL fields — document frequency,
        # not term frequency. Including bodies makes it honest: words like
        # "session" or "corpus" are rare in titles and everywhere in prose, so a
        # metadata-only df would score them as specific and let a passing mention
        # outrank the node actually about the subject.
        for t in (ti | su | tg | bo):
            df[t] += 1

    built = (field, df, max(len(live), 1))
    _INDEX["live"] = (fp, built)
    return built


# ── retrieval aliases — the corpus-owned thesaurus (2026-09-21) ──
# Any live node may carry a ```yaml block keyed RETRIEVAL_ALIASES, a list of
# {phrase, adds, ...}. When `phrase` occurs in the query, the ADDS words are
# searched as a SECOND lexical query and the two candidate pools are UNIONED
# (raw = max per node). Measured 2026-09-19: appending aliases to the query
# string displaced 8–9 of 15 results per query through the coverage
# multiplier, so preservation of what the original words found is structural
# here, not observed. Semantic admission stays on the original query: an
# alias is a lexical bridge ("throw away" → delete), never a new meaning.
# The registry node itself explains each line (scope, why, a positive and a
# negative example); the engine reads only phrase + adds. Cached on the same
# fingerprint as the index, so it costs one scan per corpus change.
# Measured on the author's private corpus; not reproducible from the example corpus.
_ALIASES: dict = {}
_ALIAS_KEY = "RETRIEVAL_ALIASES"


def _aliases(live) -> list:
    fp = _fingerprint(live)
    cached = _ALIASES.get("live")
    if cached and cached[0] == fp:
        return cached[1]
    out = []
    for n in live:
        body = n.body or ""
        if _ALIAS_KEY not in body:
            continue
        try:
            import validate as _v
        except ImportError:
            from librarian import validate as _v
        block = _v._first_yaml_block_with_key(body, _ALIAS_KEY)
        for e in (block or []):
            if not isinstance(e, dict):
                continue
            phrase = re.sub(r"[^a-z0-9]+", " ", str(e.get("phrase", "")).lower()).strip()
            adds = _tok(" ".join(str(a) for a in (e.get("adds") or [])))
            if phrase and adds:
                out.append((phrase, adds, n.id))
    _ALIASES["live"] = (fp, out)
    return out


def fired_aliases(query, aliases) -> list:
    """The alias entries whose phrase occurs in `query`, as whole words."""
    q = " " + re.sub(r"[^a-z0-9]+", " ", str(query).lower()) + " "
    return [(p, adds, nid) for p, adds, nid in aliases if " " + p + " " in q]


def _read_body(root, path) -> str:
    try:
        _, body = split_doc((root / path).read_text(encoding="utf-8"))
        return body or ""
    except Exception:
        return ""


def _section_path(body, i):
    if i is None:
        return None
    if isinstance(i, str):        # a retrieval key (embed._keys), not a section
        return "(summary)" if i.startswith("b") else "(asks)"
    cs = sections.chunks(body)
    return (cs[i]["path"] or "(opening)") if i < len(cs) else None


def _excerpt(body, body_chars, words, idf, chunk_cos, opening=True):
    """(excerpt, sections shown) for one seed. The opening of the body, as
    before, unless the best-matching section lies beyond it: then the first
    HEAD_CHARS of the opening and, in document order, the best-scoring chunks
    that fit the rest of the budget, each gap marked with the heading the next
    part sits under. A chunk scores like a node does — query words it contains
    (IDF-weighted) plus its cosine above SEM_BASE, each normalised to its best
    chunk in this node — so a section found by meaning alone can be shown.

    opening=False (a node with a bullet summary, 2026-10-22): the summary
    already states the current point of every section, so the excerpt is the
    matching sections only — "" when no section matches the query."""
    if len(body) <= body_chars:
        return body, []
    cs = sections.chunks(body)
    lex = {c["i"]: sum(idf(w) for w in _tok(body[c["start"]:c["end"]]) & words)
           for c in cs}
    sem = {k: max(0.0, v - SEM_BASE) for k, v in (chunk_cos or {}).items()}
    ml, ms = max(lex.values(), default=0.0) or 1.0, max(sem.values(), default=0.0) or 1.0
    score = {c["i"]: lex[c["i"]] / ml + sem.get(c["i"], 0.0) / ms for c in cs}
    best = max(cs, key=lambda c: (score[c["i"]], -c["i"]), default=None)
    if not opening:
        if best is None or score[best["i"]] <= 0:
            return "", []
        head_end = 0
    elif best is None or score[best["i"]] <= 0 or best["end"] <= body_chars:
        return body[:body_chars], []
    else:
        # the opening keeps at most a quarter of the budget, cut at a line break
        head_cap = min(HEAD_CHARS, body_chars // 4)
        cut = body.rfind("\n", 0, head_cap)
        head_end = cut + 1 if cut > head_cap // 2 else head_cap
    room = budget = body_chars - head_end
    spans = []                           # (start, end, path) beyond the opening
    # When the best section alone overflows the room, a weaker one that fits must
    # not take its place (review, 2026-10-23): fall straight to a window of it.
    fits = best["end"] - max(best["start"], head_end) <= budget
    for c in (sorted(cs, key=lambda c: (-score[c["i"]], c["i"])) if fits else []):
        s, e = max(c["start"], head_end), c["end"]
        if score[c["i"]] <= 0 or e <= s or e - s > budget:
            continue
        spans.append((s, e, c["path"]))
        budget -= e - s
    if not spans:
        # the best section alone overflows the room: show a window of it that
        # starts at the line holding its first matched word
        s, e = max(best["start"], head_end), best["end"]
        low = body[s:e].lower()
        hits = [m.start() for w in words if (m := re.search(rf"\b{re.escape(w)}\b", low))]
        if hits:
            at = s + min(hits)
            line = body.rfind("\n", s, at) + 1
            # the line's start, unless that is so far back the word would not
            # fit — then the word boundary a little before it
            s = max(s, line if at - line < room // 2
                    else body.rfind(" ", s, max(s, at - room // 8)) + 1)
        spans = [(s, min(e, s + room), best["path"])]
    parts, shown, pos = ([body[:head_end].rstrip()] if head_end else []), [], head_end
    for s, e, path in sorted(spans):
        if s > pos:
            parts.append(f"[… § {path}]" if path else "[…]")
        parts.append(body[s:e].strip("\n"))
        if path not in shown:
            shown.append(path)
        pos = e
    return "\n\n".join(parts), shown


def assemble_context(nodes, by_id, query, root, *, limit=8, body_chars=2400,
                     related_cap=16, body_seeds=None):
    """Return a structured context pack for `query`. Seeds carry (truncated)
    bodies — a brief needs the knowledge, not the whole file; related nodes are
    briefs only; stale relatives are surfaced as deliberately excluded."""
    live = [n for n in nodes if n.fm.get("status") in ("active", "parked")]
    qtok = _tok(query)
    if not qtok:
        return {"query": query, "seeds": [], "related": [], "excluded_stale": [],
                "counts": {"seeds": 0, "related": 0, "searched": len(live)}}

    field, df, total = _index(live)

    def idf(t):
        return math.log((total + 1) / (df.get(t, 0) + 0.5))

    floor = lex_floor(total)

    # A query term matches EXACTLY when the corpus knows the word, and by PREFIX
    # when it does not. The owner searched "internation" and got nothing while a node
    # titled "Bakery international launch" sat in the corpus: set-intersection
    # has no notion of a partial word, so a half-typed term matched zero fields
    # and the relevance floor did the rest (2026-08-03). A typed query is a
    # prefix far more often than it is a typo — and the web list was already
    # substring-matching locally, so the ranked engine was contradicting the box
    # it sits under.
    #
    # Deliberately conservative: a term the corpus knows is NEVER expanded (so
    # every existing query scores exactly as before), prefixes shorter than 4
    # characters are not expanded (too vague to mean anything), and a prefix that
    # hits more than PREFIX_MAX distinct words is treated as no match at all
    # rather than dragging the whole corpus in behind it.
    PREFIX_MIN, PREFIX_MAX = 4, 25
    def expansion(qtok):
        expand = {}
        for t in qtok:
            if t in df or len(t) < PREFIX_MIN:
                expand[t] = {t}
                continue
            pre = {v for v in df if v.startswith(t)}
            expand[t] = pre if 0 < len(pre) <= PREFIX_MAX else {t}
        return expand

    def lexical_pool(qtok):
        """id -> {raw, n, body_only} for one token set — the v1 lexical engine,
        unchanged; called once for the query and once more per alias union."""
        expand = expansion(qtok)

        cand = {}    # id -> {"raw": lexical score, "n": node, "body_only": matched body words}
        for n in live:
            ti, su, tg, bo = field[n.id]
            pool = ti | su | tg | bo

            # Weight by WHERE the term appears, strongest signal first: an author
            # chose the tag and the title deliberately; a summary word may be
            # incidental; a body word may be an aside in a 6,000-character document.
            def weight(t):
                if t in tg:
                    return 1.6
                if t in ti:
                    return 1.4
                if t in su:
                    return 1.0
                return BODY_WEIGHT

            # Score per QUERY term, not per matched word: "internation" expanding to
            # {international, internationally} is still one term of the query, so it
            # must not score twice or inflate coverage. Best matched word wins.
            hits, score, tag_hit = 0, 0.0, False
            body_only = set()
            for t in qtok:
                got = expand[t] & pool
                if not got:
                    continue
                hits += 1
                tag_hit = tag_hit or bool(got & tg)
                score += max(idf(m) * weight(m) for m in got)
                if not (got - bo):     # every match for this term lives ONLY in the body
                    body_only |= got
            if not hits:
                continue

            score *= 0.6 + 0.4 * (hits / len(qtok))   # reward covering more of the query
            # a node is "about" the query only on a substantive match (a tag, or ≥2
            # terms) — don't let the canonical boost promote an incidental one-word
            # mention (e.g. a vocabulary list that merely names the term).
            if n.fm.get("canonical") is True and (tag_hit or hits >= 2):
                score *= 1.15
            if score >= floor:  # one semi-specific term, not a single common word
                cand[n.id] = {"raw": score, "n": n, "body_only": body_only}
        return cand

    cand = lexical_pool(qtok)
    fired = fired_aliases(query, _aliases(live))
    if fired:
        extra = set().union(*(adds for _, adds, _ in fired))
        for i, d in lexical_pool(qtok | extra).items():
            if i in cand:
                if d["raw"] > cand[i]["raw"]:
                    cand[i] = {**d, "body_only": cand[i]["body_only"]}
            else:
                cand[i] = d
        # The registry that carries the fired line is infrastructure, not an
        # answer: it necessarily contains the phrase and its bridge words, so it
        # outranked the deletion ruling on the first live probe (2026-09-21).
        # When one of its lines fires, the registry itself leaves the pool; a
        # query ABOUT the registry fires no line and finds it normally.
        for _, _, src in fired:
            cand.pop(src, None)
    fired_sources = {src for _, _, src in fired}
    # every corpus word the query (and any fired alias) matched, for choosing
    # which sections of a seed to show
    match_words = set().union(*expansion(
        qtok | set().union(*(adds for _, adds, _ in fired))).values())

    # ── the semantic layer (v2 embeddings, 2026-08-06) ────────────────────────
    # Bridges the vocabulary gap lexical scoring cannot: a query in the asker's
    # words, not the node's. Two roles, both ADDITIVE (the reorder-never-delete pattern applied twice over):
    # semantic similarity RE-RANKS every candidate, and nodes with no lexical
    # hit at all are GATED IN as recall-net candidates — nothing lexical is ever
    # removed or demoted below where lexical alone would have put it, because
    # the bonus is non-negative. {} when the model is unavailable → this whole
    # block is a no-op and retrieval is exactly v1 lexical.
    # The semantic layer engages only when some query token is >= PREFIX_MIN
    # characters. A shorter fragment ("quo") means too little to search on
    # lexically, and its embedding means just as little semantically — but bge
    # cosines live in a compressed band, so a noise vector still clears
    # SEM_BASE somewhere and filled the pack with arbitrary admissions. Found
    # by the VPS suite the first time it ran WITH the model installed; the
    # Mac's green (no model, fail-soft to lexical) never exercises this path —
    # The verification-discipline guardrail mode 3, an incomplete population in the shape of a machine.
    #
    # The first version of this guard also engaged on a corpus-KNOWN short
    # token (mirroring the lexical no-expansion rule) and the live corpus
    # falsified it the same day: "status quo" in one body makes "quo" a known
    # word, and the fragment query was back to eight semantic admissions.
    # Knownness makes a token exact-MATCHABLE (the lexical path still scores
    # it, floor and all); it does not make a three-letter query MEAN anything.
    sem, csem, sem_section = {}, {}, {}
    if any(len(t) >= PREFIX_MIN for t in qtok):
        sem = embed.similarities(live, query)
    if sem:
        # SECTION VECTORS (2026-10-13): a node's semantic score is the better
        # of its node vector and its best chunk, the chunk discounted by
        # CHUNK_DISCOUNT so a long node's many draws do not outbid a short
        # node's one. {} when the chunk layer cannot answer completely —
        # then this is the node-vector ranking exactly.
        csem = embed.chunk_similarities(live, query)
        for i, cs in csem.items():
            best, c = max(cs.items(), key=lambda kv: kv[1], default=(None, None))
            if c is not None and c - CHUNK_DISCOUNT > sem.get(i, 0.0):
                sem[i] = c - CHUNK_DISCOUNT
                sem_section[i] = best
    if sem:
        topk = max(SEM_TOPK, int(len(live) * SEM_TOPK_FRAC))
        # the fired registry stays out here too: lexical exclusion alone let the
        # semantic step admit it straight back (review, 2026-10-23)
        extra = sorted(((c, i) for i, c in sem.items()
                        if c >= SEM_BASE and i not in cand and i not in fired_sources),
                       reverse=True)[:topk]
        by_live = {n.id: n for n in live}
        for c, i in extra:
            cand[i] = {"raw": 0.0, "n": by_live[i], "body_only": set()}

    # Rank by intent-weighted hybrid score; the lexical floor (above) gated on
    # the RAW score and the semantic floor gates candidates IN, so the prior and
    # the bonus can only reorder / add — never remove (the reorder-never-delete pattern).
    #
    # Both components are normalised to THEIR OWN per-query maximum before
    # blending. Measured first without this: bge cosines live in a compressed
    # band (~0.50-0.73 on real queries), so an absolute bonus SEM_SCALE·(cos −
    # base) tops out near 1.5 points while ordinary lexical noise scores 4-10 —
    # the paraphrase subset stayed at 3/12 top-1 at EVERY absolute weight even
    # though the targets sat at semantic rank 1-9. The information is in the
    # relative ordering, not the cosine magnitude; normalising makes the blend
    # scale-free, so the strongest semantic match competes at SEM_WEIGHT of the
    # strongest lexical match, whatever raw numbers either produced.
    # Measured on the author's private corpus; not reproducible from the example corpus.
    intent, intent_signals = classify_intent_ex(query)
    prior = _TYPE_PRIORS.get(intent, _TYPE_PRIORS[None])
    # The lexical normaliser has an ABSOLUTE floor: on a query where the best
    # lexical score is weak (a couple of incidental words), pure per-query
    # normalisation inflates that noise to 1.0 and it stacks over a semantic
    # rank-1 match capped at SEM_WEIGHT — measured: the file-pointer decision at semantic rank 2 of
    # 310 was crowded out of the top 15 by mid-lexical stacking. Below LEX_REF
    # points, lexical evidence is not allowed to claim full confidence.
    # Measured on the author's private corpus; not reproducible from the example corpus.
    max_raw = max(max((d["raw"] for d in cand.values()), default=0.0), LEX_REF)
    max_sem = max((max(0.0, c - SEM_BASE) for c in sem.values()), default=0.0) or 1.0
    for d in cand.values():
        cos = sem.get(d["n"].id, 0.0)
        d["sem"] = cos if sem else None
        d["prior"] = prior.get(d["n"].fm.get("type"), 1.0)
        d["final"] = (d["raw"] / max_raw
                      + SEM_WEIGHT * max(0.0, cos - SEM_BASE) / max_sem) * d["prior"]
    ranked = sorted(cand.values(), key=lambda d: -d["final"])
    seeds = ranked[:limit]
    seed_ids = {d["n"].id for d in seeds}

    # one-hop neighbourhood: live relatives become `related` (briefs); dead ones
    # are surfaced as deliberately excluded so the brief knows history exists.
    related, excluded, seen_rel, seen_ex = [], [], set(), set()
    for d in seeds:
        n = d["n"]
        for nid in _neighbours(n, by_id, nodes):
            if nid in seed_ids or nid not in by_id:
                continue
            m = by_id[nid]
            st = m.fm.get("status")
            if st in ("active", "parked"):
                if nid not in seen_rel:
                    seen_rel.add(nid)
                    related.append(_brief(m, full=False))
            elif nid not in seen_ex:
                seen_ex.add(nid)
                excluded.append({"id": nid, "title": m.fm.get("title", ""),
                                 "status": st, "superseded_by": m.fm.get("superseded_by")})

    seed_out = []
    for rank, d in enumerate(seeds, 1):
        n = d["n"]
        # body_seeds (2026-09-21): the pack widened to 15 so a
        # rank-9-to-15 answer is visible, but bodies stay on the top ranks only —
        # the tail carries summary, tags and authority, and `body_omitted` says
        # so; read_node opens one. None = every seed carries a body (web list).
        omit = body_seeds is not None and rank > body_seeds
        # load_corpus now carries the body, so the seeds no longer cost a second
        # read each; _read_body remains the fallback for a Node built elsewhere.
        b = n.body or _read_body(root, n.path)
        # a bullet summary states the node's current points itself, so its
        # seed carries only the sections that match, in half the budget
        bulleted = summary.bullets(n.fm.get("summary")) is not None
        budget = body_chars // 2 if bulleted else body_chars
        sec_cos = {k: v for k, v in (csem.get(n.id) or {}).items() if isinstance(k, int)}
        excerpt, shown = (("", []) if omit or not body_chars else
                          _excerpt(b, budget, match_words, idf, sec_cos,
                                   opening=not bulleted))
        # DECOMPOSITION (2026-08-06, second external review): `score` is the
        # value the order actually sorts on, and the components reconstruct it —
        # final = (lexical + SEM_SCALE·max(0, semantic−SEM_BASE)) · prior.
        # Before this, the displayed score was raw while the order was
        # prior-weighted, so a record at 8.63 outranking a project at 10.26 was
        # correct and UNEXPLAINABLE from the output — which forfeits the main
        # reason retrieval is deterministic at all.
        #
        # match_in_excerpt guards the SILENT failure of truncation: a query term
        # matched deep in the body while the excerpt shows only the opening — a
        # caller who answers from that excerpt answers from the wrong part of
        # the node without knowing it. False means: call read_node before
        # answering. It speaks only for lexical body matches; a semantic-only
        # seed has no term to locate, so it is True by vacuity there.
        vis = _tok(excerpt)
        sip = n.fm.get("superseded_in_part")
        seed_out.append({**_brief(n), "score": round(d["final"], 2),
                         # tags are the strongest lexical field (weight 1.6) and
                         # the caller could not see them: a reformulation had
                         # to guess the corpus's vocabulary. Exposed 2026-09-18
                         # so a near-miss seed hands over the words that score.
                         "tags": [str(t) for t in (n.fm.get("tags") or [])],
                         # seed-only authority: log-lifecycle and partial-
                         # supersession, the two fields a caller needs before
                         # treating a seed's claim as current
                         "work_state": n.fm.get("work_state"),
                         "superseded_in_part": str(sip) if sip is not None else None,
                         "score_lexical": round(d["raw"], 2),
                         "score_semantic": (round(d["sem"], 3)
                                            if d["sem"] is not None else None),
                         "prior": d["prior"],
                         "body_truncated": bool(b) if omit else len(b) > budget,
                         "body_omitted": omit,
                         "match_in_excerpt": not (d["body_only"] - vis),
                         # the headings of the sections the excerpt shows
                         # beyond the opening ([] = the opening only), and the
                         # section whose vector carried the semantic score
                         # (None = the node vector did)
                         "excerpt_sections": shown,
                         "semantic_section": _section_path(b, sem_section.get(n.id)),
                         "body": "" if omit or not excerpt else
                                 excerpt + ("…" if len(b) > budget else "")})
    return {"query": query, "seeds": seed_out, "related": related[:related_cap],
            "excluded_stale": excluded,
            # the detected intent is REPORTED so the caller can see (and second-
            # guess) the prior that shaped the order; None means no prior
            # applied. The signals are the cue strings that fired — a wrong
            # intent is diagnosable from the output alone.
            "intent": intent, "intent_signals": intent_signals,
            # the alias phrases that fired, so a pack is explainable from its
            # output alone; [] means the query used its own words only
            "aliases_fired": [p for p, _, _ in fired],
            # False = the model is unavailable and this pack is pure lexical
            "semantic": bool(sem),
            # the section layer's state (2026-10-13): ready, embedding with
            # progress, missing, failed with its error — a layer that is
            # quietly off is visible here, not only in a journal nobody reads
            "sections": embed.chunk_status(),
            "counts": {"seeds": len(seed_out), "related": min(len(related), related_cap),
                       "searched": len(live)}}
