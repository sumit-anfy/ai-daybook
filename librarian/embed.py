"""Semantic embeddings (v2) — the query-side vocabulary bridge.

LINEAGE. The staged semantic-retrieval decision staged retrieval: v1 (LLM judges a
deterministic shortlist, shipped 2026-07-25), v2 (self-hosted embeddings)
deferred behind a trigger. The trigger was re-pointed at QUERIES on 2026-08-04
after the pair form was measured unfireable, and on 2026-08-06 the owner ruled
"scope plus embeddings-v2" — pulling v2 forward on the probe evidence (queries
phrased in a person's own words, avoiding the target's vocabulary, ranked the
right node #1 in 0 of 8) rather than waiting for the miss log to accumulate.
The eval sample now VALIDATES this build instead of justifying it.

WHAT IT IS. bge-small-en-v1.5 int8-quantised via fastembed/ONNX — a real
transformer encoder, no torch, ~34 MB on disk, CPU-only. The model TIER was
chosen by measurement, weakest-first, on the benchmark's 12 paraphrase queries
(the exact gap this layer exists to close):

    model                       paraphrase-target semantic ranks
    potion-base-8M   (static)   stuck — subset 3/12 at every blend weight
    potion-retr-32M  (static)   1,1,5,5,12,18,31,50,94,153,243 — tail unusable
    bge-small-en-v1.5 (ONNX)    1,1,2,3,3,4,5,7,7,9 (+1 hard at 107)

Static embeddings were the preferred fit for the 961 MB VPS (508 MB available,
no swap, measured 2026-08-06) and they FAILED the measurement; bge-small is the
smallest model that passed, and it fits: ONNX runtime + int8 weights cost
~150-250 MB resident. Queries use query_embed (bge's asymmetric instruction
prefix), documents passage_embed — mixing them up costs recall silently.

WHAT IT IS NOT. Not an LLM on the read path (the deterministic-read-path decision/the scheduled-runtime gate decision): encoding is a fixed
matrix lookup — deterministic, offline, key-free. Same input, same vector,
every time.

FAIL-SOFT, EVERYWHERE. Retrieval must never break because semantics is
unavailable — no model2vec on the box, no network for the first model download,
a corrupt cache: every path degrades to {} and the caller runs pure lexical.
The deploy script does not pip install (pull + restart only), so the code MUST
tolerate arriving before its dependency does.

CACHING. Node vectors are cached per node keyed on a hash of the EMBEDDED TEXT
(same rule as context._index: content, not count/mtime/date — a cache whose
invalidation is coarser than what it caches is a correctness bug). Persisted to
~/.okf/embeddings.npz — live data, not knowledge (the knowledge-not-live-data decision), so it lives outside
both git repos, like the search log.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

MODEL_NAME = os.environ.get("OKF_EMB_MODEL", "BAAI/bge-small-en-v1.5")

# fastembed's DEFAULT batch is 256 documents per ONNX run. At this corpus's
# ~512-token node texts that is one ~3.0 GB attention buffer (measured live,
# 2026-08-07: `Failed to allocate ... 3221225472` — 256·12·512²·4 bytes), which
# OOM-killed the first VPS embed outright. The "~150-250 MB resident" sizing
# was the WEIGHTS; the missing term was the batch-sized activation, which
# scales linearly with this number. 8 keeps that buffer under ~100 MB for a
# few extra seconds of wall clock on a full re-embed — the right trade on a
# 961 MB box, and invisible on any larger one.
EMB_BATCH = int(os.environ.get("OKF_EMB_BATCH", "8"))

# process-lifetime state: "model" is None (untried) / False (unavailable) /
# a loaded model; "vecs" is {node_id: (text_hash, unit_vector)}
_STATE: dict = {}

# Callers run in threads since 2026-08-07 (the web search moved off the event
# loop, and each service warms in a background thread at start), so the slow
# operations need single-flight discipline: _MODEL_LOCK guards the ~10-20s ONNX
# session load, _VEC_LOCK serialises corpus embedding and the cache file write.
# Two locks, never nested, because warm() holds them in sequence.
_MODEL_LOCK = threading.Lock()
_VEC_LOCK = threading.Lock()


def _model():
    m = _STATE.get("model")
    if m is None:
        # Single-flight and NON-blocking: loading costs seconds, and a caller
        # arriving while another thread is mid-load must degrade to lexical NOW
        # (the fail-soft contract) rather than queue behind the load. Losing
        # this race returns None exactly once per loser; the next query finds
        # the loaded model in _STATE and never touches the lock again.
        if not _MODEL_LOCK.acquire(blocking=False):
            return None
        try:
            m = _STATE.get("model")     # re-check: the race winner may have won
            if m is None:
                if os.environ.get("OKF_SEMANTIC") == "0":
                    # Off on purpose: the lexical-only path, chosen rather than
                    # stumbled into, so it can be tested and run deliberately.
                    m = False
                else:
                    try:
                        from fastembed import TextEmbedding
                        m = TextEmbedding(MODEL_NAME)
                    except Exception:
                        m = False   # remember the failure: don't retry a download per query
                _STATE["model"] = m
        finally:
            _MODEL_LOCK.release()
    return m or None


def available() -> bool:
    return _model() is not None


def _node_text(n) -> str:
    """What a node MEANS, for embedding: metadata first (deliberate words),
    then the opening of the body. 4000 chars ≈ the summary-bearing head of even
    long nodes; appending 20 KB of session prose would dilute, not sharpen."""
    tags = " ".join(str(t) for t in (n.fm.get("tags") or []))
    asks = " | ".join(str(a) for a in (n.fm.get("asks") or []))
    summary = str(n.fm.get("summary") or "") + (" Asks: " + asks if asks else "")
    return "\n".join([str(n.fm.get("title", "")), summary, tags, (n.body or "")[:4000]])


def _h(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=12).hexdigest()


def cache_path() -> Path:
    env = os.environ.get("OKF_EMB_CACHE")
    return Path(env).expanduser() if env else Path.home() / ".okf" / "embeddings.npz"


def _load_disk() -> dict:
    try:
        import numpy as np
        z = np.load(cache_path())
        return {str(i): (str(h), v) for i, h, v in zip(z["ids"], z["hashes"], z["vecs"])}
    except Exception:
        return {}


def _save_disk(mem: dict) -> None:
    try:
        import numpy as np
        p = cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        ids = list(mem)
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, ids=np.array(ids),
                            hashes=np.array([mem[i][0] for i in ids]),
                            vecs=np.stack([mem[i][1] for i in ids]))
        tmp.replace(p)
    except Exception:
        pass    # a lost cache is a re-embed (~1s), never an error


# A CORPUS-WIDE CHANGE IS RE-EMBEDDED IN THE BACKGROUND (2026-10-14). The
# 2026-09-29 asks migration changed every node's text, and the first query
# after it re-embedded all of them INSIDE that request, holding _VEC_LOCK: two
# MCP calls timed out and the server's resident size went 232 → 587 MB. In a
# service, more than NODE_INLINE_MAX changed nodes now serve their PREVIOUS
# vector (a node whose asks grew is still about what it was about) while a
# background thread re-embeds them, taking the lock only to merge each step.
# A node with no vector at all is simply absent from the semantic layer until
# its turn. Tests and scripts (no warm) keep the inline path: a measurement
# must see exact vectors.
NODE_INLINE_MAX = int(os.environ.get("OKF_NODE_INLINE_MAX", "32"))


def _refresh_nodes(model, todo_texts) -> None:
    """Background re-embed of [(node_id, text_hash, text)], merged per step."""
    import numpy as np
    try:
        for k in range(0, len(todo_texts), 64):
            part = todo_texts[k:k + 64]
            vecs = np.stack(list(model.passage_embed([t for _, _, t in part],
                                                     batch_size=EMB_BATCH)))
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
            with _VEC_LOCK:
                mem = _STATE.setdefault("vecs", {})
                for (i, h, _), v in zip(part, vecs):
                    mem[i] = (h, v.astype("float32"))
            _STATE["node_refresh"] = {"done": k + len(part), "total": len(todo_texts)}
        with _VEC_LOCK:
            _save_disk(dict(_STATE.get("vecs") or {}))
    finally:
        _STATE.pop("node_refresh", None)


def node_vectors(live):
    """{node_id: unit vector} for the live set, embedding only what changed.
    None when the model is unavailable."""
    model = _model()
    if model is None:
        return None
    import numpy as np
    want = {n.id: _h(_node_text(n)) for n in live}
    # Serialised: concurrent threadpool searches must not embed the same nodes
    # twice (two batch-sized activations at once) or interleave writes to the
    # cache file's tmp path. Post-warm steady state never enters the todo
    # branch, so the lock costs a dict lookup.
    start = None
    with _VEC_LOCK:
        mem = _STATE.get("vecs")
        if mem is None:
            mem = _load_disk()
            _STATE["vecs"] = mem
        todo = [n for n in live if mem.get(n.id, ("",))[0] != want[n.id]]
        if len(todo) > NODE_INLINE_MAX and _STATE.get("serving"):
            bg = _STATE.get("node_bg")
            if not (bg and bg.is_alive()):
                start = [(n.id, want[n.id], _node_text(n)) for n in todo]
            todo = []                      # serve what is cached, stale or not
        if todo:
            vecs = np.stack(list(model.passage_embed([_node_text(n) for n in todo],
                                                     batch_size=EMB_BATCH)))
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
            for n, v in zip(todo, vecs):
                mem[n.id] = (want[n.id], v.astype("float32"))
            _save_disk({i: mem[i] for i in want if i in mem})
        out = {i: mem[i][1] for i in want if i in mem}
    if start:
        _STATE["node_refresh"] = {"done": 0, "total": len(start)}
        t = threading.Thread(target=_refresh_nodes, args=(model, start), daemon=True)
        _STATE["node_bg"] = t
        t.start()
    return out


def warm(load_live, log=None) -> None:
    """Best-effort warm-up for a service start: load the model and the corpus
    vectors NOW, in whatever (daemon) thread this runs in, so no interactive
    query ever pays the ~10-20s session load — the stall that froze the graph
    UI on the first post-deploy search (2026-08-07). `load_live` returns the
    live node list; called lazily so corpus loading also happens off-path.
    Every failure is swallowed and logged: a service that cannot warm serves
    lexical, which is the standing contract, and the next query retries."""
    say = log or (lambda m: print(m, flush=True))
    _STATE["serving"] = True
    try:
        import time
        t0 = time.time()
        if _model() is None:
            say("semantic warm: model unavailable — retrieval stays lexical")
            return
        live = load_live()
        vecs = node_vectors(live)
        say(f"semantic warm: {len(vecs or {})} node vectors ready "
            f"in {time.time() - t0:.1f}s")
        t1 = time.time()
        k = _embed_chunks_blocking(live)
        say(f"semantic warm: chunk vectors ready ({k} embedded) "
            f"in {time.time() - t1:.1f}s")
    except Exception as e:
        say(f"semantic warm: failed ({e}) — lexical until a later query succeeds")


def similarities(live, query) -> dict:
    """{node_id: cosine} against `query` for every live node; {} whenever the
    semantic layer cannot answer — the caller then runs pure lexical, which is
    the entire contract."""
    try:
        model = _model()
        if model is None:
            return {}
        import numpy as np
        vecs = node_vectors(live)
        if not vecs:
            return {}
        q = _query_vector(model, query)
        if q is None:
            return {}
        return {i: float(v @ q) for i, v in vecs.items()}
    except Exception:
        return {}


def _query_vector(model, query):
    """Unit query vector, remembered for the last query: the node pass and the
    chunk pass of one search ask for the same vector, and an embed is ~5ms."""
    import numpy as np
    last = _STATE.get("qvec")
    if last and last[0] == str(query):
        return last[1]
    q = np.asarray(list(model.query_embed([str(query)]))[0])
    qn = float(np.linalg.norm(q))
    if qn == 0.0:
        return None
    q = q / qn
    _STATE["qvec"] = (str(query), q)
    return q


# ── chunk vectors (2026-10-13): the body below the first 4,000 chars ─────────
# The node vector reads title, summary, asks, tags and the OPENING of the body
# (_node_text). Measured the day this was built: 335 of 550 live nodes run past
# that window and 53% of all body text sat beyond it, so a question about
# something deep in a session or a long decision could only ever match on
# words. Every substantive heading-delimited chunk (librarian/sections.py) now
# gets its own vector, and the ranking reads a node's best chunk beside its
# node vector.
#
# CACHE. Content-addressed: {hash of the embedded text: vector}, in a file
# beside the node cache (so the deploy gate's OKF_EMB_CACHE moves both). A
# chunk whose text is unchanged is never re-embedded wherever it moves, and
# the file is pruned to the live chunk set on every save.
#
# NEVER A CORPUS-WIDE EMBED INSIDE A REQUEST. 6,048 chunks on the droplet's
# one core take minutes, and the 2026-09-29 asks migration showed what an
# in-request corpus re-embed does there (two MCP timeouts, RSS 232 → 587 MB).
# So the layer is ALL OR NOTHING per query: when more than CHUNK_INLINE_MAX
# chunks are missing, the query gets None (node vectors only — the pre-chunk
# ranking, exactly) and the missing chunks are embedded in a background
# thread. A write that changes one node embeds its handful inline. Two
# services share the cache file, so the background embed takes a cross-process
# flock: the second process waits, then finds the work done on disk.
CHUNK_INLINE_MAX = int(os.environ.get("OKF_CHUNK_INLINE_MAX", "64"))
_CHUNK_LOCK = threading.Lock()


def chunk_cache_path() -> Path:
    p = cache_path()
    return p.with_name(p.stem + ".chunks.npz")


def _keys(n) -> list:
    """[(key, text)] for a node's retrieval KEYS (2026-10-22): each `asks` entry
    ("a0", "a1", …) and each bullet of a bullet summary ("b0", …), each its own
    vector beside the section chunks. Folded into the one node vector, a
    summary's bullets dilute each other: measured on all 555 live nodes, a
    deep passage reached the default pack 16 of 40 that way and 23 of 40 with
    every bullet and ask matched on its own (librarian/summary.py)."""
    try:
        import summary
    except ImportError:
        from librarian import summary
    out = [(f"a{k}", str(a).strip()) for k, a in enumerate(n.fm.get("asks") or [])]
    out += [(f"b{k}", t) for k, t in enumerate(summary.search_texts(n.fm.get("summary")))]
    return [(k, t) for k, t in out if t]


def _chunk_plan(live) -> list:
    """[(node_id, chunk_index, text_hash, text)] for every substantive chunk —
    an int index for a body section, "a<k>"/"b<k>" for an ask or a summary
    bullet (_keys). Cached on a hash of what it is built from: rebuilding it
    (chunking and hashing ~4 MB of bodies) cost ~130ms a query on the Mac,
    measured — four times the rest of the search together."""
    fp = hashlib.blake2b(digest_size=16)
    for n in live:
        for part in (n.id, str(n.fm.get("title", "")), n.body or "",
                     str(n.fm.get("summary") or ""),
                     "\x01".join(str(a) for a in (n.fm.get("asks") or []))):
            fp.update(part.encode("utf-8", "replace"))
            fp.update(b"\x00")
    key = fp.digest()
    cached = _STATE.get("chunk_plan")
    if cached and cached[0] == key:
        return cached[1]
    try:
        import sections
    except ImportError:
        from librarian import sections
    plan = []
    for n in live:
        body = n.body or ""
        title = str(n.fm.get("title", ""))
        for c in sections.chunks(body):
            if sections.substantive(c, body):
                t = sections.embed_text(title, c, body)
                plan.append((n.id, c["i"], _h(t), t))
        for k, t in _keys(n):
            plan.append((n.id, k, _h(t), t))
    _STATE["chunk_plan"] = (key, plan)
    return plan


def _load_chunks() -> dict:
    try:
        import numpy as np
        z = np.load(chunk_cache_path())
        return {str(h): v for h, v in zip(z["hashes"], z["vecs"])}
    except Exception:
        return {}


def _save_chunks(mem: dict) -> None:
    try:
        import numpy as np
        p = chunk_cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        hs = list(mem)
        # per-process temp name: both services save this file, and a shared
        # temp path would let two concurrent writes interleave before replace
        tmp = p.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, hashes=np.array(hs),
                            vecs=np.stack([mem[h] for h in hs]) if hs
                            else np.zeros((0, 384), dtype="float32"))
        tmp.replace(p)
    except Exception:
        pass


# A corpus-wide embed SAVES AS IT GOES. The first deploy (2026-10-13) left the
# layer off for over 40 minutes with no way to see why from outside the box:
# 6,048 chunks at ~13/s on one Mac core is ~8 minutes, slower on the droplet,
# and with a save only at the end, a service restarted or killed part-way
# (memory on a 961 MB box) lost everything and began again from zero.
CHUNK_SAVE_EVERY = 512


def _status(**kw) -> None:
    _STATE["chunk_status"] = {**(_STATE.get("chunk_status") or {}), **kw}


def chunk_status() -> dict:
    """The chunk layer's state, for the pack: ready / embedding (done of
    total) / missing (n chunks unembedded, a background embed due) / failed
    (with the error) / off (no model). A layer that is quietly off must be
    visible from the output, like `semantic: false`."""
    st = dict(_STATE.get("chunk_status") or {"state": "off" if _STATE.get("model") is False
                                             else "not loaded"})
    if _STATE.get("node_refresh"):
        # node vectors being refreshed after a corpus-wide change: the ranking
        # is using each changed node's previous vector until this finishes
        st["node_refresh"] = dict(_STATE["node_refresh"])
    return st


def _embed_into(model, mem: dict, todo: list, save=None) -> None:
    import numpy as np
    since = 0
    for k in range(0, len(todo), 64):     # 64 texts per call; EMB_BATCH bounds the activations
        part = todo[k:k + 64]
        vecs = np.stack(list(model.passage_embed([t for *_, t in part],
                                                 batch_size=EMB_BATCH)))
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        for (_, _, h, _), v in zip(part, vecs):
            mem[h] = v.astype("float32")
        since += len(part)
        _status(done=k + len(part))
        if save and since >= CHUNK_SAVE_EVERY:
            save(mem)
            since = 0


def _embed_chunks_blocking(live, log=None) -> int:
    """Embed every missing chunk, holding both locks — the warm path and the
    background thread. Returns how many were embedded."""
    model = _model()
    if model is None:
        return 0
    plan = _chunk_plan(live)
    with _CHUNK_LOCK:
        lockf = None
        try:
            import fcntl
            p = chunk_cache_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            lockf = open(p.with_suffix(".lock"), "w")
            fcntl.flock(lockf, fcntl.LOCK_EX)
        except Exception:
            lockf = None                  # no flock: single-process behaviour
        try:
            mem = {**_load_chunks(), **(_STATE.get("chunks") or {})}
            todo = [p for p in plan if p[2] not in mem]
            if todo:
                import time
                _status(state="embedding", done=0, total=len(todo),
                        started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        error=None)
                _embed_into(model, mem, todo, save=_save_chunks)
            want = {h for _, _, h, _ in plan}
            mem = {h: v for h, v in mem.items() if h in want}
            if todo:
                _save_chunks(mem)
            _STATE["chunks"] = mem
            _STATE.pop("chunk_matrix", None)
            _STATE["chunk_status"] = {"state": "ready", "chunks": len(plan)}
            return len(todo)
        except Exception as e:
            _status(state="failed", error=f"{type(e).__name__}: {e}"[:300])
            raise
        finally:
            if lockf is not None:
                lockf.close()


def _background(live) -> None:
    # Only a process that warmed — a long-lived service — embeds in the
    # background. A test run or a stdio server never does: the deploy gate's
    # suite runs on the droplet's one core under a 900s limit, and a minutes-
    # long embed started inside it would compete for that core with the tests.
    if not _STATE.get("serving"):
        return
    if _STATE.get("chunk_bg") and _STATE["chunk_bg"].is_alive():
        return
    def run():
        try:
            _embed_chunks_blocking(live)
        except Exception:
            pass                          # recorded in chunk_status; the next query retries
    t = threading.Thread(target=run, daemon=True)
    _STATE["chunk_bg"] = t
    t.start()


def chunk_matrix(live):
    """(keys, matrix): keys[k] = (node_id, chunk_index), matrix rows are unit
    vectors. None whenever the chunk layer cannot answer completely — model
    unavailable, an embed in progress, or more than CHUNK_INLINE_MAX chunks
    missing (which starts the background embed)."""
    model = _model()
    if model is None:
        return None
    import numpy as np
    plan = _chunk_plan(live)
    key = _STATE["chunk_plan"][0]         # the content fingerprint the plan was built from
    cached = _STATE.get("chunk_matrix")
    if cached and cached[0] == key:
        return cached[1], cached[2]
    if not _CHUNK_LOCK.acquire(blocking=False):
        return None                       # warm or background embed in progress
    start_bg = False
    try:
        mem = _STATE.get("chunks")
        if mem is None:
            mem = _load_chunks()
            _STATE["chunks"] = mem
        todo = [p for p in plan if p[2] not in mem]
        if len(todo) > CHUNK_INLINE_MAX:
            start_bg = True
            if (_STATE.get("chunk_status") or {}).get("state") != "failed":
                _status(state="missing", missing=len(todo))
            return None
        if todo:
            _embed_into(model, mem, todo)
            want = {h for _, _, h, _ in plan}
            for h in [h for h in mem if h not in want]:
                del mem[h]
            _save_chunks(mem)
        keys = [(i, c) for i, c, _, _ in plan]
        mat = (np.stack([mem[h] for _, _, h, _ in plan]) if plan
               else np.zeros((0, 384), dtype="float32"))
        _STATE["chunk_matrix"] = (key, keys, mat)
        _STATE["chunk_status"] = {"state": "ready", "chunks": len(keys)}
        return keys, mat
    finally:
        _CHUNK_LOCK.release()
        if start_bg:
            _background(live)


def chunk_similarities(live, query) -> dict:
    """{node_id: {chunk_index: cosine}} for every substantive chunk; {} whenever
    the chunk layer cannot answer completely — the caller then ranks on node
    vectors alone, which is the pre-chunk behaviour exactly."""
    try:
        model = _model()
        if model is None:
            return {}
        got = chunk_matrix(live)
        if not got:
            return {}
        keys, mat = got
        q = _query_vector(model, query)
        if q is None or not keys:
            return {}
        cos = mat @ q
        out: dict = {}
        for (i, c), v in zip(keys, cos.tolist()):
            out.setdefault(i, {})[c] = v
        return out
    except Exception:
        return {}
