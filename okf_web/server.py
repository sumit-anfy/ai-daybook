"""OKF web server (Starlette). Serves the graph UI + a JSON API over the corpus.

Run:  uvicorn okf_web.server:app --host 127.0.0.1 --port 8790
Env:  OKF_ROOT (corpus root), OKF_WEB_PASSWORD (login), OKF_WEB_SECRET (cookie
      signing — REQUIRED, the server refuses to start without it unless
      OKF_WEB_DEV=1), and OKF_GIT_COMMIT/OKF_GIT_PUSH (so human edits version + push).
Mounted behind Caddy at /graph on okf.example.com.
"""
import contextlib
import hashlib
import hmac
import os
import re
import sys
import threading
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from starlette.routing import Route

# Code and corpus are separate repos, so the import path and the corpus root are
# no longer the same directory: `librarian` is always imported from beside this
# file, while OKF_ROOT says where the NODES live. Deriving sys.path from OKF_ROOT
# would look for the code inside the corpus repo.
CODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_ROOT))
from librarian import config                               # noqa: E402

config.load_env()
ROOT = config.corpus_root()

from librarian import audit, context, embed, searchlog, store, validate   # noqa: E402
from librarian.transitions import TransitionError          # noqa: E402

validate.assert_corpus_root(ROOT)   # fail loudly, not over an empty corpus
store.start_background_push(ROOT)   # drain any push a restart interrupted

HERE = Path(__file__).resolve().parent
# Fail closed: no secret, no server (okf_web/config.py explains). A SystemExit
# here, not a raise, so the journal shows one line with the fix instead of a
# traceback ending in a name nobody will search for.
try:
    from okf_web.config import web_secret          # noqa: E402
    SECRET = web_secret()
except Exception as _e:                            # ConfigError, or a broken import
    raise SystemExit(f"okf-web: {_e}") from None
PASSWORD = os.environ.get("OKF_WEB_PASSWORD", "")

# frontmatter fields a human may edit from the UI (id/created_at/etc. stay off-limits;
# supersedes/status transitions are still enforced by the store)
EDITABLE = {"title", "summary", "tags", "status", "canonical", "project", "source"}


# Code graphs (graphify): optional, DERIVED artifacts generated elsewhere and
# copied into OKF_CODEGRAPH_DIR; this server never builds them. Served behind
# the same login as the corpus, because a code graph can describe private
# systems: the access boundary is identity, as it is for the corpus itself.
CODEGRAPH_DIR = Path(os.environ.get("OKF_CODEGRAPH_DIR", "/home/okf/graphs"))


def codegraph_file(name: str) -> Path | None:
    """Whitelist-validated, containment-safe resolution of a graph name.
    Pure so the contract is testable on any machine: only [a-z0-9_-] names,
    path built from the validated stem alone — traversal is unrepresentable."""
    if not re.fullmatch(r"[a-z0-9_-]{1,64}", name or ""):
        return None
    f = CODEGRAPH_DIR / f"{name}.html"
    return f if f.is_file() else None


def _token() -> str:
    return hmac.new(SECRET, b"okf-web-v1", hashlib.sha256).hexdigest()


def _authed(req) -> bool:
    return bool(PASSWORD) and hmac.compare_digest(req.cookies.get("okf_web", ""), _token())


async def _json(req):
    try:
        return await req.json()
    except Exception:
        return None


async def index(req):
    return HTMLResponse((HERE / "index.html").read_text(encoding="utf-8"))


async def login(req):
    data = await _json(req) or {}
    if PASSWORD and hmac.compare_digest(str(data.get("password", "")), PASSWORD):
        r = JSONResponse({"ok": True})
        r.set_cookie("okf_web", _token(), httponly=True, secure=True,
                     samesite="lax", max_age=60 * 60 * 24 * 30, path="/")
        return r
    return JSONResponse({"ok": False, "error": "Wrong password."}, status_code=401)


async def graph(req):
    if not _authed(req):
        return JSONResponse({"error": "auth"}, status_code=401)
    nodes, by_id, _skipped, _vt, _ff = validate.load_corpus(ROOT)
    out = []
    for n in nodes:
        fm = n.fm
        out.append({
            "id": n.id, "title": fm.get("title", ""), "type": fm.get("type"),
            "project": fm.get("project", "—"), "status": fm.get("status"),
            "canonical": bool(fm.get("canonical")),
            "relates_to": [r for r in (fm.get("relates_to") or []) if r in by_id],
            # summary/tags/updated_at are for the LIST view and its instant local
            # filter. Until 2026-08-03 the payload carried neither, so the client's
            # search could only ever match `title` — it was not a tuning problem,
            # the fields were not there. Summaries are truncated because the list
            # shows two lines; full-text relevance is /search's job, not the
            # client's.
            "summary": (fm.get("summary") or "")[:400],
            "tags": [str(t) for t in (fm.get("tags") or [])],
            "updated_at": str(fm.get("updated_at") or ""),
        })
    # Hand the UI the live vocabulary from the corpus registry so it never
    # carries a stale hardcoded copy (a missing project silently mis-rendered
    # nodes and, worse, let the editor overwrite an out-of-list project on save).
    return JSONResponse({"nodes": out, "count": len(out),
                         "projects": sorted(validate.load_registry_projects(ROOT)),
                         "types": sorted(_vt)})


def search(req):
    """Ranked search, using the SAME engine as the MCP's `brief`.

    The web app used to match `title.includes(query)` client-side while
    `librarian/context.py` sat unused beside it — IDF-weighted over title +
    summary + tags, tag hits boosted, query coverage rewarded, canonical nodes
    nudged, and a relevance floor so one common word does not return the corpus.
    Two implementations of one idea, and the worse one was the one the owner used.

    This is the `librarian/audit.py` arrangement applied to retrieval: one engine,
    two surfaces, so the graph and the MCP can never disagree about what is
    relevant.

    DELIBERATELY SYNC (`def`, not `async def`) since the engine went hybrid:
    Starlette runs sync endpoints in its threadpool, so a slow search — above
    all an embedding-model load — cannot block the event loop. As `async`,
    the first post-deploy search froze the ENTIRE app for the ~15s model load
    (measured live, 2026-08-07: every surface stuck, not just the search).
    Measured on the author's server; not reproducible here.
    """
    if not _authed(req):
        return JSONResponse({"error": "auth"}, status_code=401)
    q = (req.query_params.get("q") or "").strip()
    if not q:
        return JSONResponse({"query": "", "hits": [], "count": 0})
    nodes, by_id, *_ = validate.load_corpus(ROOT)
    # body_chars=0 — the list renders its own summary and the drawer fetches the
    # body on click, so shipping bodies here would be payload for nothing.
    pack = context.assemble_context(nodes, by_id, q, ROOT, limit=40, body_chars=0)
    hits = [{"id": s["id"], "title": s["title"], "type": s["type"],
             "project": s["project"], "score": s.get("score")}
            for s in pack.get("seeds", [])]
    searchlog.record_search(q, len(hits), hits[0]["score"] if hits else None, "web")
    return JSONResponse({"query": q, "hits": hits, "count": len(hits),
                         # intent passthrough: which prior shaped this order
                         "intent": pack.get("intent"),
                         "searched": pack.get("counts", {}).get("searched", 0)})


async def search_open(req):
    """A node opened FROM a search result.

    This is the half a relevance threshold cannot supply: on the eight failing
    paraphrase probes the WRONG node scored 4.2-7.4, indistinguishable from a good
    hit, so score alone cannot tell a find from a plausible miss. A search followed
    by an open succeeded whatever it scored; one followed by nothing did not."""
    if not _authed(req):
        return JSONResponse({"error": "auth"}, status_code=401)
    d = await _json(req) or {}
    searchlog.record_open(d.get("query"), d.get("id"), d.get("rank"), "web")
    return JSONResponse({"ok": True})


async def health(req):
    """Deterministic corpus-health flags (no LLM) — orphans, same-Coda-page
    duplicates, stale neighbourhoods, near-identical titles. Shared engine in
    librarian.audit so the web panel and the MCP audit tool agree exactly."""
    if not _authed(req):
        return JSONResponse({"error": "auth"}, status_code=401)
    nodes, by_id, *_ = validate.load_corpus(ROOT)
    return JSONResponse(audit.compute_health(nodes, by_id, ROOT))


# ── link suggestions (Layer 2) — engine lives in librarian.audit ───────────────
async def suggest(req):
    if not _authed(req):
        return JSONResponse({"error": "auth"}, status_code=401)
    nodes, by_id, *_ = validate.load_corpus(ROOT)
    active = [n for n in nodes if n.fm.get("status") == "active"]
    aset = {n.id for n in active}
    existing = audit.existing_links(active, aset)
    dismissed = {frozenset(p) for p in audit.load_dismissed(ROOT) if p[0] in aset and p[1] in aset}
    sug = audit.compute_suggestions(active, existing, dismissed, audit.load_curated(ROOT))
    return JSONResponse({"suggestions": sug, "count": len(sug)})


async def link(req):
    """Accept a suggested connection: add b to a's relates_to (the graph treats
    relates_to as undirected, so one direction makes the edge). Goes through the
    store so it's validated + committed + pushed like any other edit."""
    if not _authed(req):
        return JSONResponse({"error": "auth"}, status_code=401)
    data = await _json(req) or {}
    a, b = data.get("a"), data.get("b")
    if not a or not b or a == b:
        return JSONResponse({"ok": False, "error": "a and b required"}, status_code=400)
    try:
        cur = store.read_node(ROOT, a)
    except KeyError:
        return JSONResponse({"ok": False, "error": "node not found"}, status_code=404)
    rel = list(cur["frontmatter"].get("relates_to") or [])
    if b in rel:
        return JSONResponse({"ok": True, "already": True})
    rel.append(b)
    try:
        res = store.update_node(ROOT, a, {"relates_to": rel})
    except (TransitionError, store.StoreError, KeyError, ValueError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "committed": res.get("committed"),
                         "pushed": res.get("pushed")})


async def dismiss(req):
    if not _authed(req):
        return JSONResponse({"error": "auth"}, status_code=401)
    data = await _json(req) or {}
    a, b = data.get("a"), data.get("b")
    if not a or not b:
        return JSONResponse({"ok": False, "error": "a and b required"}, status_code=400)
    pairs = audit.load_dismissed(ROOT)
    pairs.add(tuple(sorted((a, b))))
    audit.save_dismissed(ROOT, pairs)
    try:  # keep the VPS working tree clean (git steps gated on OKF_GIT_COMMIT)
        store._git_commit(ROOT, [Path(".okf/dismissed.json")], "okf: dismiss link suggestion")
    except Exception:
        pass
    return JSONResponse({"ok": True})


async def node(req):
    if not _authed(req):
        return JSONResponse({"error": "auth"}, status_code=401)
    nid = req.path_params["nid"]
    if req.method == "GET":
        try:
            return JSONResponse(store.read_node(ROOT, nid))
        except KeyError:
            return JSONResponse({"error": "not found"}, status_code=404)
    if req.method == "DELETE":
        try:
            res = store.delete_node(ROOT, nid)
        except (store.StoreError, KeyError) as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        return JSONResponse({"ok": True, "cleaned_refs": res.get("cleaned_refs", 0),
                             "committed": res.get("committed"), "pushed": res.get("pushed")})
    # POST — save
    data = await _json(req)
    if data is None:
        return JSONResponse({"ok": False, "error": "invalid JSON body"}, status_code=400)
    changes = {k: v for k, v in data.items() if k in EDITABLE}
    try:
        # allow_body_replacement is passed UNCONDITIONALLY here, and only here.
        # The guard exists because an agent cannot see what it is about to
        # overwrite; this editor is a textarea holding the full body, edited by a
        # human who is looking at it. Fixing one typo produces a body that does not
        # contain the old one, so under the tightened floor (2026-08-04) every save
        # would otherwise be refused. The control on this path is the human reading
        # the text, not a flag — which is why the gated-deletion decision's "a flag always passed is not a
        # control" does not bite: the gated-deletion decision is about the AGENT passing its own confirm.
        res = store.update_node(ROOT, nid, changes, body=data.get("body"),
                                allow_body_replacement=True)
    except (TransitionError, store.StoreError, KeyError, ValueError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    findings, *_ = validate.validate_corpus(ROOT)
    errs = [str(f) for f in findings if f.severity == validate.SEVERITY_ERROR]
    return JSONResponse({"ok": True, "errors": errs,
                         "committed": res.get("committed"), "pushed": res.get("pushed")})


async def create(req):
    if not _authed(req):
        return JSONResponse({"error": "auth"}, status_code=401)
    data = await _json(req)
    if not data or not data.get("type") or not data.get("title"):
        return JSONResponse({"ok": False, "error": "type and title are required."}, status_code=400)
    kw = {k: data[k] for k in ("summary", "project", "source", "tags", "status",
                               "maturity", "canonical", "relates_to") if data.get(k) not in (None, "", [])}
    try:
        res = store.create_node(ROOT, type=data["type"], title=data["title"],
                                body=data.get("body", ""), **kw)
    except (TransitionError, store.StoreError, ValueError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "id": res["id"], "warnings": res.get("warnings", [])})


def codegraphs(req):
    """List hosted code graphs — the Code tab's picker. Sync on purpose:
    directory stat()s belong in Starlette's threadpool, not on the loop."""
    if not _authed(req):
        return JSONResponse({"error": "auth"}, status_code=401)
    out = []
    try:
        for p in sorted(CODEGRAPH_DIR.glob("*.html")):
            out.append({"name": p.stem, "updated": int(p.stat().st_mtime)})
    except OSError:
        pass
    return JSONResponse({"graphs": out})


def codegraph(req):
    """Serve one hosted code graph (a self-contained graphify HTML page).
    Sync on purpose: these files are megabytes, and a loop-blocking read is
    the same class of stall the search endpoint already paid for."""
    if not _authed(req):
        return JSONResponse({"error": "auth"}, status_code=401)
    f = codegraph_file(req.path_params["name"])
    if f is None:
        return JSONResponse({"error": "no such graph"}, status_code=404)
    return HTMLResponse(f.read_text(encoding="utf-8"))


routes = [
    Route("/graph", index),
    Route("/graph/", index),
    Route("/graph/api/login", login, methods=["POST"]),
    Route("/graph/api/graph", graph),
    Route("/graph/api/search", search),
    Route("/graph/api/search/open", search_open, methods=["POST"]),
    Route("/graph/api/codegraphs", codegraphs),
    Route("/graph/codegraph/{name}", codegraph),
    Route("/graph/api/health", health),
    Route("/graph/api/suggest", suggest),
    Route("/graph/api/suggest/dismiss", dismiss, methods=["POST"]),
    Route("/graph/api/link", link, methods=["POST"]),
    Route("/graph/api/node", create, methods=["POST"]),
    Route("/graph/api/node/{nid}", node, methods=["GET", "POST", "DELETE"]),
]


def _warm_semantics():
    """Startup warm in a daemon thread: the model load costs ~10-20s on the
    droplet and must never be paid inside an interactive search. Until it
    finishes, searches degrade to lexical (embed's non-blocking single-flight),
    which is invisible next to the alternative — the pre-warm behaviour was the
    whole app frozen on the first post-deploy keystroke."""
    embed.warm(lambda: [n for n in validate.load_corpus(ROOT)[0]
                        if n.fm.get("status") in ("active", "parked")])


@contextlib.asynccontextmanager
async def _lifespan(app):
    # The deprecated on_startup kwarg is GONE in current Starlette — passing it
    # is a TypeError at import, which crash-looped this service on 2026-08-07
    # (and the Mac could not catch it: its import dies lines EARLIER, on the
    # corpus assert). lifespan is the supported form.
    threading.Thread(target=_warm_semantics, daemon=True).start()
    yield


app = Starlette(routes=routes, lifespan=_lifespan)
