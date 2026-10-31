"""OKF MCP server — the single interface both interactive chat and the scheduled
task speak through (build-spec §3, §5).

This is a THIN wrapper. All determinism (id assignment, validation, transitions,
supersession) lives in librarian/; this file only exposes it as MCP tools. The
scheduled-task path is gated behind the scheduled-runtime gate decision until the
reachability spike passes — running this server interactively does not lift that.

Run (stdio, default):  python server.py
Run (http, for VPS):   OKF_TRANSPORT=http OKF_AUTH_TOKEN=<secret> python server.py
Env (also read from a .env beside this file; see .env.example): OKF_ROOT
(corpus root, default: examples/corpus), OKF_HOST (default
127.0.0.1 — keep loopback and put TLS/proxy in front), OKF_PORT (default 8765),
OKF_AUTH_TOKEN (bearer token; REQUIRED for http), and see librarian/store.py for
OKF_GIT_COMMIT / OKF_GIT_PUSH (auto-commit corpus writes — turn on for the VPS).
"""
import json
import os
import sys
from pathlib import Path

from fastmcp import FastMCP

from librarian import config

config.load_env()

from librarian import (audit, cluster, context, embed, findability, redundancy,  # noqa: E402
                       searchlog, store, validate)

CORPUS_ROOT = validate.assert_corpus_root(config.corpus_root())

# Safeguard 2 for the background push: a restart (the deploy timer does one on
# every code change) can kill a push in flight, leaving a committed-but-unpushed
# node. Draining at boot clears whatever the previous process left behind.
store.start_background_push(CORPUS_ROOT)


def _attach_layer2(result: dict) -> dict:
    """Layer 2 (the five-layer audit decision): after a successful write, hand the
    calling agent its declared neighbours and the question 'what does this node make
    FALSE?'. The server does no LLM work — the caller IS the LLM. Advisory only: it
    never blocks, and a failure here must never fail a write that already succeeded.
    """
    # store returns {id, path, ...} on success and RAISES on refusal — there is no
    # "ok" key to test. Keying on `id` is the contract that actually exists.
    if not isinstance(result, dict) or not result.get("id"):
        return result
    try:
        nodes, by_id, *_ = validate.load_corpus(CORPUS_ROOT)
        node = by_id.get(result.get("id"))
        if node is None:
            return result
        block = cluster.layer2(node.fm, node.id, nodes, by_id)
        if block:
            result["layer2"] = block
        # Layer 3 is layer 2's COMPLEMENT — "does this already exist?" against the
        # canonical documents governing this project (the redundancy-check handover). Only fires for the types
        # that state a durable rule, so ordinary session/note writes stay quiet.
        red = redundancy.layer3(node.fm, node.id, nodes)
        if red:
            result["layer3"] = red
    except Exception as e:                      # advisory; never breaks the write
        result["layer2_error"] = f"{type(e).__name__}: {e}"
    return result


def _auth():
    """Auth layers, composed:
    - OKF_AUTH_TOKEN            -> static bearer token (scripts, Claude Code CLI)
    - OKF_GITHUB_CLIENT_ID/     -> OAuth via a GitHub OAuth App (claude.ai custom
      _SECRET + _ALLOWED_USERS     connectors need OAuth + dynamic client
                                   registration; bearer headers aren't offered
                                   there). Fails closed: only the GitHub logins
                                   in OKF_GITHUB_ALLOWED_USERS are accepted.
    Both may be active at once (MultiAuth) — OAuth for claude.ai, token for CLI.
    """
    token = os.environ.get("OKF_AUTH_TOKEN")
    verifier = None
    if token:
        from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
        # scopes must cover GitHubProvider.required_scopes ("user") or MultiAuth
        # 403s the token with insufficient_scope when OAuth is also enabled
        verifier = StaticTokenVerifier(
            tokens={token: {"client_id": "okf-client", "scopes": ["user"]}})

    gh_id = os.environ.get("OKF_GITHUB_CLIENT_ID")
    gh_secret = os.environ.get("OKF_GITHUB_CLIENT_SECRET")
    if gh_id and gh_secret:
        from fastmcp.server.auth import MultiAuth
        from fastmcp.server.auth.providers.github import GitHubProvider

        allowed = {u.strip() for u in
                   os.environ.get("OKF_GITHUB_ALLOWED_USERS", "").split(",") if u.strip()}
        if not allowed:
            sys.exit("refusing OAuth without OKF_GITHUB_ALLOWED_USERS — any GitHub "
                     "account could otherwise authorize itself into the corpus")

        from librarian.auth_grace import GraceCache
        grace = GraceCache()

        class RestrictedGitHub(GitHubProvider):
            async def verify_token(self, token):  # fail closed on unknown logins
                cached = grace.fresh(token)
                if cached is not None:
                    return cached
                result = await super().verify_token(token)
                if result is not None:
                    if result.claims.get("login") not in allowed:
                        # A gate that rejects must say who it rejected (the fail-loudly rule
                        # rule 5) — a bare invalid_token 401 is indistinguishable
                        # from a broken server.
                        print(f"AUTH: GitHub login {result.claims.get('login')!r} "
                              f"rejected — not in OKF_GITHUB_ALLOWED_USERS "
                              f"({sorted(allowed)})", flush=True)
                        return None
                    grace.record(token, result)
                    return result
                # Verification FAILED — which upstream reports identically for
                # "revoked" and "GitHub is down" (any non-200 -> None). On
                # 2026-08-17 GitHub's outage 503'd the identity check and every
                # grant died as "invalid_token" within seconds of issuance. A
                # token this process has recently seen verify is served from the
                # grace cache instead; one that never verified stays refused.
                held = grace.grace(token)
                if held is not None:
                    stale, age = held
                    print(f"AUTH: upstream verification FAILED for a token that "
                          f"verified OK {age:.0f}s ago — serving grace-cached "
                          f"identity (window {int(grace.grace_ttl)}s); GitHub is "
                          f"likely degraded", flush=True)
                    return stale
                return result

        gh = RestrictedGitHub(
            client_id=gh_id, client_secret=gh_secret,
            base_url=os.environ.get("OKF_PUBLIC_URL", "http://127.0.0.1:8765"))
        # fastmcp builds the private_key_jwt client-auth audience as
        # f"{base_url}/token" (oauth_proxy/proxy.py:2061, unfixed through
        # 3.4.6) and pydantic renders a bare origin WITH a trailing slash,
        # so a ChatGPT connector's client JWTs were rejected against the
        # impossible audience 'https://okf.example.com//token' (2026-09-03).
        # claude.ai never trips it: public client + PKCE, no client JWT.
        # A plain str here crash-loops the service instead: the mcp SDK's
        # validate_issuer_url reads .scheme off this value at route-build
        # time (learned live, 2026-09-03). So: still a real AnyHttpUrl —
        # attributes intact, isinstance passes — that merely renders
        # without pydantic's trailing slash, making every f-string
        # construction downstream match the metadata already advertised.
        from pydantic import AnyHttpUrl

        class _SlashlessUrl(AnyHttpUrl):
            def __str__(self):
                return super().__str__().rstrip("/")

        gh.base_url = _SlashlessUrl(str(gh.base_url))
        return MultiAuth(server=gh, verifiers=[verifier] if verifier else None)

    return verifier


mcp = FastMCP("okf", auth=_auth())

# The tool CONTRACT version, stamped into the update_node docstring so a client's
# cached schema betrays its own vintage. MCP clients cache tool schemas at connect
# time: after body_append shipped (2026-08-04), the session that built it could not
# call it, and an external review filed the same staleness as a P0 "parity defect"
# when the server was fine. There is no way to force a remote client to refresh, so
# the cheapest honest fix is a version string a human can compare — if the schema
# you see does not mention v1.7, your client predates it: reconnect.
# v1.7 (2026-09-06): `project` became a REQUIRED create_node parameter (node.schema
# v1.7, the required-project decision) — a client holding the v1.6 schema still offers it
# as optional and gets a refusal it cannot explain.
# v1.8 (2026-09-15): close_out gains a REQUIRED `tasks` list for sessions — the
# TASKS line of the close-out contract (the session close-out playbook) was "soft on every surface" because
# nothing asked for it; the week-38 audit run named it, the owner ruled the field in.
# The brief pack: fifteen seeds, bodies on the top eight (2026-09-21, ruled "2a").
BRIEF_BODY_SEEDS = 8

CONTRACT_VERSION = "v1.8 (2026-09-15)"

# The close-out contract (the session close-out playbook), restated in the results
# of the two tools a session actually thinks with. A document is read once at session
# start and decays under load; a tool response re-enters context at the moment of use,
# which is a different decay profile entirely — that asymmetry is the whole design.
# Deliberately NOT on read_node/list_nodes: those fire constantly, and a contract
# repeated on every call is skimmed, not read.
CLOSE_OUT_PROTOCOL = (
    "Session close-out (the session close-out playbook): end every session with Keep / "
    "Conflicts / Your call. Say 'nothing to keep' explicitly rather than ending "
    "silently — most sessions produce nothing durable and saying so is correct. "
    "Claude proposes; the owner rules. A ruling stated in a session body and filed "
    "nowhere is lost: promote it to its own node and cite the id. Cross-project "
    "guardrails go to `pattern` in global-os, not a decision in the project they "
    "arose in. After the owner rules, state whether you are still unconvinced — "
    "'agreed' is valid but must be said.")


@mcp.tool
def validate_corpus() -> dict:
    """Validate every node against node.schema. Returns ok plus any errors/warnings."""
    with store._repo_lock(CORPUS_ROOT, shared=True):
        findings, nodes, skipped, valid_types = validate.validate_corpus(CORPUS_ROOT)
    errors = [str(f) for f in findings if f.severity == validate.SEVERITY_ERROR]
    warnings = [str(f) for f in findings if f.severity == validate.SEVERITY_WARN]
    return {"ok": not errors, "node_count": len(nodes), "skipped": skipped,
            "vocabulary": sorted(valid_types), "errors": errors, "warnings": warnings}


@mcp.tool
def read_node(node_id: str) -> dict:
    """Read a node's frontmatter and body by id.

    VERIFY THE RESPONSE IS THE NODE YOU ASKED FOR: check `frontmatter.id` equals
    your `node_id` before using the content. On 2026-08-04, under nine concurrent
    callers sharing one client session, three reads received a DIFFERENT node's
    content (correct on retry). The server and proxy were exonerated by
    reproduction — 468 concurrent calls with induced aborts, zero mismatches — so
    the misrouting sits in the shared CLIENT layer after a timeout, which this
    server cannot fix. The id check is one comparison and it caught all three
    live instances."""
    searchlog.record_read(node_id)   # observational engagement signal, never success
    return store.read_node(CORPUS_ROOT, node_id)


@mcp.tool
def list_nodes(type: str | None = None, status: str | None = None) -> list:
    """List nodes, optionally filtered by type and/or status."""
    return store.list_nodes(CORPUS_ROOT, type, status)


@mcp.tool
def audit_corpus(limit: int = 14) -> dict:
    """Corpus triage for the weekly review. Read-only; proposes, never writes.
    Returns {summary, flags, suggestions, purge_candidates}:
    - summary: counts (total / active / retired / open flags / purge_candidates).
    - flags: structural health issues — orphans (no links), same-source
      duplicates, stale neighbourhoods (every neighbour retired), near-identical
      titles. Each lists the node(s) involved.
    - suggestions: active node pairs that share specific tags/topics but aren't
      linked yet, each with a reason and the two node briefs (id/title/type/
      project), ranked. Act on each with `resolve_suggestion` (accept / dismiss);
      accepted pairs become edges and dismissed pairs are already excluded.
    - purge_candidates: RETIRED nodes structurally safe to remove — orphaned (no
      live in/out edge), non-canonical, not a supersession target. Auto-detected;
      these are exactly the nodes `delete_node`'s four gates admit, so an agent
      may delete them (agent-callable since 2026-08-02) or a human may in /graph.
    - searches: search outcomes, SURFACE-CORRECT (2026-08-07 ruling). WEB keeps
      the behavioural miss inference (nothing opened, reworded runs collapsed =
      miss). MCP is TRI-STATE, because brief() returns bodies and "no open"
      means nothing there: a quickly-reworded run is a SUSPECTED miss, an
      untouched run is UNKNOWN (not a miss, not a success), and a read_node of
      a returned seed counts as ENGAGED — observational only. Roles (ruled
      2026-08-07): WEB misses are a readable trend — that surface has a real
      open signal. MCP suspected misses are a DISCOVERY QUEUE — queries worth
      adjudicating — never a retrieval-quality metric: diagnostic and
      evaluation traffic lands there by design (a query returning the correct
      node at rank 1 sat in the queue the day this shipped), so trends in the
      count are not evidence of regression or improvement without
      adjudication. Regression evidence is the benchmark, adjudicated
      real-query samples, and `semantic: false` incidence. Surface suspected
      MCP queries as candidates to investigate; never quote an MCP "miss
      rate" (unknown is not a miss). A low relevance score does NOT mean a
      failed search — measured and rejected as a signal."""
    with store._repo_lock(CORPUS_ROOT, shared=True):
        nodes, by_id, *_ = validate.load_corpus(CORPUS_ROOT)
    return {**audit.audit(nodes, by_id, CORPUS_ROOT, limit=limit),
            "protocol": CLOSE_OUT_PROTOCOL}


@mcp.tool
def brief(query: str, limit: int = 15) -> dict:
    """Assemble a context pack for "what do we know about X" / "brief me before Y",
    answered from the corpus rather than one chat's memory. Deterministic
    retrieval over the LIVE nodes (read-only, no LLM) — returns the *material*,
    you write the brief from it:
    - `seeds`: the nodes most relevant to `query`, ranked, with a score. The
      TOP EIGHT carry a (truncated) body — the actual knowledge to brief from.
      Ranks 9–15 carry title, summary, tags and authority only, with
      `body_omitted: true` (2026-09-21): a near miss is now VISIBLE instead of
      cut off at eight — the cap-table record sat at rank 9–12 through a whole
      regression — and read_node opens any of them in one call;
    - `related`: their one-hop neighbourhood as briefs (id/title/type/project/summary);
    - `excluded_stale`: retired/superseded relatives deliberately left out (with any
      `superseded_by`), so you know history exists without pulling it in.

    RANKING AND YOUR HALF OF IT (2026-08-06). The engine is HYBRID: lexical
    keyword/tag IDF over title/summary/tags/bodies, blended with self-hosted
    semantic embeddings that bridge the paraphrase gap — a query in YOUR words,
    not the node's, now matches on meaning (still deterministic, still no LLM,
    text never leaves the box). Semantics only ADDS: it reorders and admits
    extra candidates, never removes a lexical match. Each seed's score
    decomposes as score_lexical / score_semantic / prior, so the order is
    explainable from the output alone; `semantic: false` on the pack means the
    embedding model was unavailable and this pack is pure lexical. The order
    is also intent-aware: a deterministic classifier reads the query's shape
    (procedure / record / history / current_ruling / project_state /
    open_question / reference) and weights node TYPES accordingly — a playbook
    outranks a session for a how-do-I question, a decision outranks a session
    for a can-we question. The detected intent is returned as `intent` (null =
    no cue matched, neutral ranking), with `intent_signals` naming the cues
    that fired, so a misroute is diagnosable from the output.
    PHRASING — YOUR FIRST ATTEMPT DECIDES MOST OF THE OUTCOME (2026-09-18,
    from adjudicating every query in the suspected-miss queue): write ONE
    plain question or subject phrase of about 6–12 words that names the
    subject in the corpus's own terms. Never paste a paragraph or a sentence
    from elsewhere, and never send a bare list of keywords. The lexical score
    pays a node for every rare word it contains, so a long word list ranks the
    longest session summary above the ruling it summarises (measured: a
    nine-word keyword list put a session at 16.8 over the decision at 8.7).
    The word "session" is an intent cue that promotes sessions — leave it
    out unless sessions are what you want. Each seed carries its `tags`: the
    field the ranking weights most and the vocabulary the corpus actually
    uses, so a reformulation borrows a tag or title word from the nearest
    miss rather than guessing.
    ALIASES (2026-09-21): a corpus-owned thesaurus node bridges phrasings the
    corpus never uses ("throw away" → delete, deletion, purge). When a phrase
    fires, the alias words run as a second search and the candidate pools are
    unioned, so nothing your own words found can drop out; `aliases_fired`
    lists what fired ([] = none). To add a bridge, add a line to that node —
    no code.
    SECTIONS (2026-10-13): a long node is searched section by section. Each
    heading-delimited chunk has its own embedding, and a node's semantic score
    is the better of its node vector and its best chunk; `semantic_section`
    names that chunk's heading path (null = the node vector carried it). When
    the best-matching section lies beyond the opening, the excerpt is the
    opening's first lines plus the matching sections, each gap marked
    `[… § heading]`, and `excerpt_sections` lists what was stitched in ([] =
    the plain opening). Quote a stitched excerpt knowing it skips text.
    `sections` on the pack is the layer's state — ready, embedding (done of
    total), missing, or failed with its error; until ready, the ranking uses
    node vectors only and `semantic_section` stays null.
    BULLET SUMMARIES (2026-10-22): a node's `summary` may be "- " bullets —
    its CURRENT state, every point, each ending "→ § <heading>" (the section
    holding the detail), with "(changed DATE: was …)" / "(overruled DATE: …)"
    where a point moved. Read it as the answer of record: the body is the
    history. Such a seed's `body` is only the sections that match the query
    (possibly ""), and read_node opens the section a bullet points to. Every
    bullet and every `asks` entry is matched on its own; `semantic_section`
    reads "(summary)" or "(asks)" when one of those carried the score. A
    `related` brief carries only bullet 1, the one-line description.
    You are the judgment half: (1) RERANK the seeds against what the question
    actually asks before answering — the right node is USUALLY but not always
    first; (2) a seed with `match_in_excerpt: false` matched BEYOND its
    truncated body — read_node before answering from it; (3) if the seeds look
    off-topic, REFORMULATE ONCE with a tag or title word from a near-miss seed
    and call again (e.g. "who owns how much" → "cap table shareholding") —
    above all when `semantic: false` says the paraphrase bridge is down and
    lexical limits are back."""
    with store._repo_lock(CORPUS_ROOT, shared=True):
        nodes, by_id, *_ = validate.load_corpus(CORPUS_ROOT)
    pack = context.assemble_context(nodes, by_id, query, CORPUS_ROOT, limit=limit,
                                    body_seeds=BRIEF_BODY_SEEDS)
    seeds = pack.get("seeds") or []
    searchlog.record_search(query, len(seeds),
                            seeds[0].get("score") if seeds else None, "mcp",
                            seed_ids=[s["id"] for s in seeds])
    return {**pack, "protocol": CLOSE_OUT_PROTOCOL}


@mcp.tool
def create_node(type: str, title: str, project: str, body: str = "",
                summary: str | None = None,
                status: str = "active", maturity: str = "captured",
                canonical: bool = False,
                tags: list | None = None, source: str | None = None,
                relates_to: list | None = None, supersedes: str | None = None,
                resource: dict | None = None, asks: list | None = None,
                work_state: str | None = None, allow_duplicate: bool = False,
                allow_unfiled_rulings: bool = False,
                close_out: dict | str | None = None) -> dict:
    """Create a node: assign an id, apply defaults, validate against node.schema,
    and write it to the folder implied by its type. Refuses to write if invalid.
    Refuses a node whose `source` duplicates an existing one unless
    allow_duplicate=true; near-identical titles are returned as `warnings`.

    `asks` (node.schema v1.9, 2026-09-29; actually accepted here only since
    2026-10-14 — before that it was silently dropped): SIX TO EIGHT everyday
    phrasings a stranger would type when this node is the answer — retrieval
    metadata, indexed and embedded like the summary, never shown as knowledge.
    Write it in the asker's words, not the node's; a plain question or subject
    phrase of 4–12 words each, covering different angles (the problem before
    the answer is known, the why, a concrete situation, the consequence).
    Every live node carries eight since 2026-10-15; a node born with fewer
    is outranked by its neighbours. Measured both times on every live node
    before adoption: four lifted held-out stranger findability by about half,
    and eight lifted it again, 47 → 56 of 80 in the default pack.
    `summary` (2026-10-22) is written as BULLETS, one "- " line each — the
    node's current state, which search reads first:
      - bullet 1: one sentence on what this node is (its one-line description)
      - bullets 2–8: EVERY point the node currently makes, ≤25 words each,
        each ending " → § <heading>" — the body section holding the detail
        ("(opening)" for text before the first heading; " #N" for the Nth of
        a repeated heading)
      - a point that changed or was overruled says so in its bullet:
        "(changed 2026-08-25: was X)" / "(overruled 2026-10-15: X)",
        "undated" when no date is known
    Every point in the summary must also be in the body: the summary is
    rewritten as things change, the body is the history and is never lost.
    Format problems come back as `warnings`; more than 8 bullets means the node
    holds more than one idea — split it into linked nodes. A session is exempt
    from that one: it records a whole sitting. Measured on all 555
    live nodes before adoption: a deep passage reached the default pack 13 →
    23 of 40.
    `project` is REQUIRED (node.schema v1.7, 2026-09-06): every node has exactly
    one home, and it is one of the type registry's PROJECT_REGISTRY values —
    the life areas the registry lists, or global-os (the capability
    home for anything that serves every life area: OKF itself, MCP servers,
    cross-project guardrails). There is no default: global-os is a home, not a
    fallback. A node with no home is refused at the door.

    ON THE `push` FIELD IN THE RESULT — read this before reporting anything about it.
    `push: "queued"` (with `pushed: null`) is the NORMAL, HEALTHY outcome: the node is
    committed and a background worker sends it to GitHub a moment later. Do NOT report
    it as an unconfirmed, failed, or partial write — it is neither. Only
    `push: "failed"` is worth telling the user about, and `push_note` says so
    explicitly whenever there is anything to say.

    ON SUCCESS the result carries a `layer2` block: the node's declared neighbours
    and the question "what does this node make FALSE?". It is ADVISORY — nothing was
    blocked and no follow-up call is needed to complete the write — but you are
    expected to act on it: staleness is caused by the NEW write and lands on the OLD
    node, so a neighbour that now asserts something untrue gets a dated correction
    block appended to IT (never a silent rewrite). The five-layer audit decision.

    A `layer3` block also comes back for `decision` / `pattern` / `playbook` writes: the
    canonical documents and playbooks governing that project, and the complementary
    question — "does one of these already STATE this rule?" (COVERED / PARTIAL / MISSING
    / SCOPE-SPECIFIC, quoting the line). Layer 2 catches what a write makes FALSE; layer 3
    catches what it DUPLICATES. Also advisory. The redundancy-check handover.

    For `type=session` two extra contracts apply:
    - COMPLETENESS: every bullet under `## Decisions locked` / `## Principles
      surfaced` must cite the id of the node that ruling lives in. A bullet stating
      a rule and pointing nowhere is refused — file it as its own decision/pattern
      node first and cite the id (any form). Override: allow_unfiled_rulings=true.
    - CLOSE-OUT: pass `close_out` = {keep: [...], conflicts: [...],
      nothing_to_keep: bool, tasks: [...]} — the report the owner gets at session end.
      "Nothing to keep" is a normal, correct outcome, but it must be SAID
      (nothing_to_keep=true with an empty keep list); silence and forgetting are
      indistinguishable. `tasks` (REQUIRED since v1.8, 2026-09-15) lists what
      should go to Todoist, one line each — an empty list is a valid answer and
      means "nothing to do", said rather than forgotten; "is there something now
      TRUE" (keep) and "is there something to DO" (tasks) are independent
      questions (the session close-out playbook). Cross-project guardrails belong in
      `keep` as `pattern` in global-os, not as a decision in the project they
      arose in."""
    # Some MCP clients serialise object-typed params as a JSON string; accept both,
    # or the gate becomes unsatisfiable through the very interface it guards.
    if isinstance(close_out, str):
        try:
            close_out = json.loads(close_out)
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"close_out is not valid JSON: {e}"}
    if type == "session" and not isinstance(close_out, dict):
        return {"ok": False, "error":
                "a session node requires close_out={keep:[...], conflicts:[...], "
                "nothing_to_keep:bool} — report what is worth keeping, what it "
                "contradicts, and say 'nothing to keep' explicitly when that is the "
                "answer (the session close-out playbook). Claude proposes; the owner rules."}
    if type == "session":
        missing = [k for k in ("keep", "conflicts", "nothing_to_keep", "tasks") if k not in close_out]
        if missing:
            return {"ok": False, "error": f"close_out is missing {missing} — all four "
                    "fields are required; 'nothing to keep' and 'nothing to do' must be "
                    "said, not implied (tasks: [] is a valid answer). Contract v1.8."}
        if not isinstance(close_out.get("tasks"), list) or \
                not all(isinstance(t, str) and t.strip() for t in close_out["tasks"]):
            return {"ok": False, "error": "close_out.tasks must be a list of non-empty "
                    "strings — one line per thing that should go to Todoist; [] means none."}
        if not close_out.get("keep") and not close_out.get("nothing_to_keep"):
            return {"ok": False, "error": "close_out.keep is empty but nothing_to_keep is "
                    "false — say which it is. Silence and forgetting are indistinguishable."}
    result = store.create_node(
        CORPUS_ROOT, type=type, title=title, body=body, summary=summary,
        status=status, maturity=maturity, canonical=canonical, project=project,
        tags=tags, source=source, relates_to=relates_to, supersedes=supersedes,
        resource=resource, asks=asks, work_state=work_state,
        allow_duplicate=allow_duplicate,
        allow_unfiled_rulings=allow_unfiled_rulings)
    return _attach_layer2(result)


@mcp.tool
def update_node(node_id: str, changes: dict, body: str | None = None,
                body_append: str | None = None,
                allow_unsupersede: bool = False, allow_unlink: bool = False,
                allow_unfiled_rulings: bool = False,
                allow_record_amendment: bool = False,
                allow_repoint: bool = False,
                allow_body_replacement: bool = False) -> dict:
    """[contract v1.7, 2026-09-06 — if the schema you are reading does not say this, your client cached an older contract: reconnect.]
    Update a node's frontmatter (and optionally body). Enforces the status /
    maturity state machine (node.schema §4) and re-validates before writing.
    NOTE: `changes.relates_to` REPLACES the whole list — pass the complete set
    (existing + new). A change that would DROP an existing link is refused unless
    allow_unlink=true, so links are never lost by a partial update.
    For `type=session`, the completeness check also applies: a bullet under
    `## Decisions locked` / `## Principles surfaced` that cites no node id is
    refused (override: allow_unfiled_rulings=true). Sessions predating the rule
    carry `unfiled_rulings_legacy: true` and are skipped.
    Returns a `layer2` advisory when the change could make another node false — see
    create_node. Backlink-only and date-only updates skip it.

    For `type=record` the IMMUTABILITY discipline applies: `title`, `summary` and
    `created_at` are frozen and the body is APPEND-ONLY. A record is a dated snapshot
    and is never edited to reflect a new state — file a NEW record that `supersedes`
    this one (that path works, and is the intended one). Linking, supersession and
    status changes stay open. Override: allow_record_amendment=true.

    A `gdrive_*` pointer's FILE ID is also frozen: a Drive id survives being moved, so
    a changed id means the file was deleted and re-uploaded, which breaks every pointer
    to it (node.schema §2). Update the file in place. Override: allow_repoint=true.

    TWO WAYS TO WRITE A BODY, AND YOU ALMOST ALWAYS WANT THE FIRST.

    `body_append` ADDS to the end of the node. Pass ONLY your new text — do not
    read the node first, do not repeat what is already there, and do not write a
    "## APPEND" heading: this tool stamps a dated one for you. It cannot lose
    anything, it needs no flag, and it is the correct call for a correction block,
    a dated update, or any "add this to that node". It is also far cheaper than a
    replacement on a large node, because you send only the new block.

    `body` REPLACES THE WHOLE BODY. It does not append, and a "## APPEND" heading
    in your text does NOT make it append. ANY replacement that does not contain the
    existing body is REFUSED unless you set allow_body_replacement=true — not just
    a shrinking one. Use it only when you genuinely mean to rewrite the node, and
    expect to pass the flag when you do. Passing BOTH `body` and `body_append` is
    an error, because they are opposite intents.

    The result reports `body_bytes` (before/after/delta) whenever a body was
    written, so even an intended rewrite is visible rather than silent.

    THE SUMMARY MOVES WITH THE BODY (2026-10-22). The body is the history and
    only grows; the summary is the CURRENT state and is rewritten. So a body
    write comes with `changes={"summary": ...}` in the same call: the bullets
    restated as they now stand, the one that moved saying "(changed DATE: was
    …)" or "(overruled DATE: …)" (format in create_node). A body write that
    leaves the summary as it was returns a `warnings` entry; so does a summary
    in the wrong shape. Logs and records are exempt — a log grows by design
    and a record's summary is frozen.

    ON THE `push` FIELD IN THE RESULT — read this before reporting anything about it.
    `push: "queued"` (with `pushed: null`) is the NORMAL, HEALTHY outcome: the node is
    committed and a background worker sends it to GitHub a moment later. Do NOT report
    it as an unconfirmed, failed, or partial write — it is neither. Only
    `push: "failed"` is worth telling the user about, and `push_note` says so
    explicitly whenever there is anything to say."""
    if not changes and body is None and body_append is None:
        return {"ok": False, "error": "nothing to change — pass `changes`, `body` or "
                "`body_append`. Refused rather than committed as a no-op: an empty "
                "update used to bump updated_at and add a commit (2026-09-18)."}
    result = store.update_node(CORPUS_ROOT, node_id, changes, body,
                               body_append=body_append,
                               allow_unsupersede=allow_unsupersede, allow_unlink=allow_unlink,
                               allow_unfiled_rulings=allow_unfiled_rulings,
                               allow_record_amendment=allow_record_amendment,
                               allow_repoint=allow_repoint,
                               allow_body_replacement=allow_body_replacement)
    # a backlink or a date touch cannot make another node false — stay silent on those.
    # a BODY write can and does, and it arrives as its own parameter rather than
    # inside `changes`, so it has to be tested separately or every prose correction
    # would slip past the check. body_append counts: an appended correction block is
    # the single most likely thing to make a NEIGHBOUR stale.
    if body is not None or body_append is not None or cluster.should_check_update(changes):
        return _attach_layer2(result)
    return result


@mcp.tool
def resolve_suggestion(a: str, b: str, action: str, reason: str = "") -> dict:
    """ACCEPT or DISMISS one link suggestion from `audit_corpus`. This is the action
    surface the weekly digest points at. The graph's 'Audit & suggest links' panel
    was removed on 2026-08-03 and until this tool (2026-09-18) a suggestion could be
    read but never answered, so the same curated pairs led every digest for seven
    weeks and the digest stopped being read.

    - action="accept": adds `b` to `a`'s relates_to through the store — validated,
      committed, pushed like any edit; the graph treats relates_to as undirected, so
      one direction makes the edge. Idempotent: an edge that already exists in either
      direction answers {ok: true, already: true} and writes nothing.
    - action="dismiss": records the pair in .okf/dismissed.json so the engine stops
      proposing it. No node changes; reversible by editing that file.

    Suggestions with source "vector" are the vector-link TRIAL (2026-10-15): pairs
    related in meaning that share no tags or words. Rule them like any other; each
    ruling on such a pair is recorded and scores the trial (the response and the
    audit summary carry the running count).

    Both ids must be real nodes. The pair need not be in the current suggestion
    list. The owner rules per pair in one message and a session calls this per ruling.
    The weekly task must NOT call it: it proposes, a human rules
    (deploy/weekly-audit-task.md)."""
    action = (action or "").strip().lower()
    if action not in ("accept", "dismiss"):
        return {"ok": False, "error": "action must be 'accept' or 'dismiss'"}
    if not a or not b or a == b:
        return {"ok": False, "error": "a and b must be two different node ids"}
    with store._repo_lock(CORPUS_ROOT, shared=True):
        nodes, by_id, *_ = validate.load_corpus(CORPUS_ROOT)
    missing = [x for x in (a, b) if x not in by_id]
    if missing:
        return {"ok": False, "error": f"no node with id {missing}"}
    # The vector-link trial (2026-10-15): a word-invisible pair can only have
    # come from the trial channel, so every ruling on one is its score.
    trial = audit.pair_is_word_invisible(nodes, a, b)

    def _score_trial():
        if not trial:
            return None
        try:
            audit.record_vector_trial(CORPUS_ROOT, a, b, action)
            store._git_commit(CORPUS_ROOT, [Path(".okf/vector_trial.json")],
                              f"okf: vector-link trial — {action} {a} <-> {b}")
        except Exception:
            pass
        return audit.vector_trial_line(CORPUS_ROOT)
    if action == "accept":
        rel = list(by_id[a].fm.get("relates_to") or [])
        if b in rel or a in (by_id[b].fm.get("relates_to") or []):
            return {"ok": True, "action": "accept", "a": a, "b": b, "already": True}
        rel.append(b)
        try:
            res = store.update_node(CORPUS_ROOT, a, {"relates_to": rel})
        except (store.StoreError, KeyError, ValueError) as e:
            return {"ok": False, "error": str(e)}
        out = {"ok": True, "action": "accept", "a": a, "b": b, "already": False,
               "committed": res.get("committed"), "pushed": res.get("pushed"),
               "push": res.get("push")}
        score = _score_trial()
        return {**out, "vector_link_trial": score} if score else out
    pairs = audit.load_dismissed(CORPUS_ROOT)
    key = tuple(sorted((a, b)))
    already = key in pairs
    pairs.add(key)
    audit.save_dismissed(CORPUS_ROOT, pairs)
    try:  # keep the meta file synced (git steps gated on OKF_GIT_COMMIT)
        store._git_commit(CORPUS_ROOT, [Path(".okf/dismissed.json")],
                          f"okf: dismiss link suggestion {a} <-> {b}"
                          + (f" — {reason}" if reason else ""))
    except Exception:
        pass
    out = {"ok": True, "action": "dismiss", "a": a, "b": b, "already": already,
           "note": "the engine will not propose this pair again; undo by removing it "
                   "from .okf/dismissed.json in the corpus repo."}
    score = None if already else _score_trial()
    return {**out, "vector_link_trial": score} if score else out

@mcp.tool
def propose_deletion(node_id: str, reason: str = "") -> dict:
    """PROPOSE a node for deletion — for anything `delete_node`'s gates BLOCK. Agents CAN
    delete (since 2026-08-02, the gated-deletion decision), but only a node that is retired,
    non-canonical, unlinked and outside any supersession chain. A node that fails a gate
    yet should still go — live, linked, canonical, or a chain member — is a human's call:
    this records it as a purge candidate, surfaced by audit_corpus and the graph, and
    The owner deletes it in the graph editor, which does not apply the gates. Prefer retiring
    (update_node status=retired) for obsolete-but-real knowledge; propose deletion only
    for junk / a genuine mistake.

    History, because a tool description is the only thing an LLM caller reads
    (the MCP tool-design reference): this was `delete_node` until 2026-07-30, when the owner
    ruled agents cannot delete and it became a proposer; on 2026-08-02 a genuinely
    deleting `delete_node` returned behind four structural gates. From then on the
    gates, not a human step, are the control — and this tool is the path around them."""
    with store._repo_lock(CORPUS_ROOT, shared=True):
        _, by_id, *_ = validate.load_corpus(CORPUS_ROOT)
    if node_id not in by_id:
        return {"ok": False, "error": f"no node with id '{node_id}'"}
    audit.propose_purge(CORPUS_ROOT, node_id, reason)
    try:  # keep the meta file synced (git steps gated on OKF_GIT_COMMIT)
        store._git_commit(CORPUS_ROOT, [Path(".okf/purge_candidates.json")],
                          f"okf: propose purge {node_id}")
    except Exception:
        pass
    return {"ok": True, "proposed": node_id, "note": "recorded as a purge candidate — "
            "a human decides in the graph editor (delete_node's gates do not apply there); "
            "if the node is retired and unlinked, delete_node can remove it directly."}


@mcp.tool
def delete_node(node_id: str, reason: str = "") -> dict:
    """DELETE a node — permitted only for a node that is already RETIRED and structurally
    disconnected. Prefer `update_node status=retired` for obsolete-but-real knowledge:
    retirement preserves the reasoning trail, which is most of what the corpus is for.
    Delete only junk and genuine mistakes.

    Four gates, ALL required, and none of them overridable:
      - status is `retired` (retire it first — that step is reversible, this one needs a gate)
      - `canonical: false`
      - no live edges in EITHER direction
      - not part of a supersession chain

    A refusal names the gate that failed and what to do about it. Nothing here can remove
    live, linked, or canonical knowledge, which is the entire safety story: an agent that
    misreads an instruction cannot reach load-bearing nodes.

    Removal is from the CORPUS, not from history — the file stays in git and is
    recoverable (verified 2026-08-02 by restoring a node deleted months earlier, through
    a repo split that rewrote every hash). It is NOT erasure: for a secret or
    client-identifying material, a history rewrite is a separate and deliberately heavier
    procedure. Agent-callable since 2026-08-02; `propose_deletion` remains for
    anything these gates block."""
    with store._repo_lock(CORPUS_ROOT, shared=True):
        nodes, by_id, *_ = validate.load_corpus(CORPUS_ROOT)
    blockers = audit.purge_blockers(node_id, nodes, by_id)
    if blockers:
        return {"ok": False, "deleted": None, "blocked_by": blockers,
                "note": "refused — every gate must pass. Retire it first if it is still "
                        "active, or use propose_deletion to hand it to a human."}
    title = by_id[node_id].fm.get("title", "")
    result = store.delete_node(CORPUS_ROOT, node_id)
    # The store answers `deleted: True`; this tool promises `deleted: <id>`, and
    # the spread used to let the store's boolean overwrite the id — caught the
    # day the stdio round-trip was first collected by pytest (2026-09-07).
    return {"ok": True, "deleted": node_id, "title": title, "reason": reason,
            "recoverable": "yes — the file remains in git history",
            **{k: v for k, v in result.items() if k not in ("id", "deleted")}}


if __name__ == "__main__":
    transport = os.environ.get("OKF_TRANSPORT", "stdio")
    if transport == "http":
        if not (os.environ.get("OKF_AUTH_TOKEN") or
                (os.environ.get("OKF_GITHUB_CLIENT_ID")
                 and os.environ.get("OKF_GITHUB_CLIENT_SECRET"))):
            sys.exit("refusing to serve http without auth (OKF_AUTH_TOKEN and/or "
                     "OKF_GITHUB_CLIENT_ID+SECRET) — the corpus write path must "
                     "never be exposed unauthenticated")
        # Warm the semantic layer off-path so the first brief() after a deploy
        # never pays the ~10-20s model load interactively. HTTP branch ONLY:
        # stdio serves the test harness, which spawns many short-lived server
        # processes — a per-process model load there is pure waste (and, on the
        # 961 MB droplet, an OOM). Lazy loading remains the stdio behaviour.
        import threading

        def _load_corpus():
            with store._repo_lock(CORPUS_ROOT, shared=True):
                return validate.load_corpus(CORPUS_ROOT)

        def _warm_then_weekly():
            embed.warm(lambda: [n for n in _load_corpus()[0]
                                if n.fm.get("status") in ("active", "parked")])
            # The weekly findability run (2026-10-14) lives HERE, after the warm,
            # in this service only: the model and section vectors are resident,
            # so it costs no memory, and a new systemd timer would need a hand on
            # a box that deploys by git push. okf-web never runs it.
            findability.loop(_load_corpus, CORPUS_ROOT)

        threading.Thread(target=_warm_then_weekly, daemon=True).start()
        mcp.run(transport="http",
                host=os.environ.get("OKF_HOST", "127.0.0.1"),
                port=int(os.environ.get("OKF_PORT", "8765")))
    else:
        mcp.run()
