"""The semantic layer (v2 embeddings, 2026-08-06) — the blend's contract, model-free.

Every test here fabricates nodes and stubs `embed.similarities`: what is under
test is the BLEND's contract, not the encoder's — fail-soft to pure lexical,
additive-never-remove (the reorder-never-delete pattern applied twice over), the SEM_BASE admission floor,
the SEM_TOPK cap, and the pack's own explainability (decomposed scores, the
`semantic` flag, match_in_excerpt, the current_ruling doorway). None of it
needs fastembed, numpy, or a corpus, so it runs on the Mac — the machine that
deliberately holds no nodes — where the encoder itself cannot.

Measured at build time (2026-08-06, 57 questions, 310 live nodes):
  lexical (doorway fix)   top-1 67%  top-5 75%  r@15 82%  paraphrase top-5 3/12
  hybrid  (shipped)       top-1 72%  top-5 86%  r@15 91%  paraphrase top-5 7/12
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian import context, embed                       # noqa: E402


class N:
    """The four attributes assemble_context actually reads — a list of these is
    a corpus as far as the ranking is concerned."""
    def __init__(self, id, title, *, type="note", summary="", tags=(), body="…",
                 status="active"):
        self.id = id
        self.fm = {"type": type, "title": title, "summary": summary,
                   "tags": list(tags), "status": status}
        self.body = body
        self.path = f"notes/{id}.md"


def filler():
    """Background nodes with disjoint vocabulary: idf in a 2-node corpus is so
    flat that even a full title match scores under the 1.5 relevance floor —
    these exist to make document frequency mean something."""
    words = ["granite countertop maintenance", "orchid watering timetable",
             "bicycle chain lubrication", "sourdough starter feeding",
             "telescope mirror collimation", "kayak storage hoist"]
    return [N(f"f{i}", w) for i, w in enumerate(words)]


@pytest.fixture
def model_loads(monkeypatch):
    """For tests of the loader itself: they inject a fake fastembed and must
    reach it even when the suite runs with OKF_SEMANTIC=0."""
    monkeypatch.delenv("OKF_SEMANTIC", raising=False)


def corpus(*nodes):
    return list(nodes), {n.id: n for n in nodes}


def pack_for(nodes, by_id, query, sims, monkeypatch, chunks=None, **kw):
    monkeypatch.setattr(context.embed, "similarities", lambda live, q: dict(sims))
    monkeypatch.setattr(context.embed, "chunk_similarities",
                        lambda live, q: dict(chunks or {}))
    return context.assemble_context(nodes, by_id, query, Path("/nonexistent"), **kw)


# ── fail-soft: {} from the encoder must mean EXACTLY v1 lexical ───────────────
def test_unavailable_encoder_is_pure_lexical(monkeypatch):
    nodes, by_id = corpus(N("a", "quorbulation runbook steps"),
                          N("b", "unrelated grocery errands"), *filler())
    p = pack_for(nodes, by_id, "quorbulation runbook", {}, monkeypatch)
    assert p["semantic"] is False, "the pack must LABEL a pure-lexical answer"
    assert [s["id"] for s in p["seeds"]] == ["a"]
    assert all(s["score_semantic"] is None for s in p["seeds"])


# ── additive, twice over (the reorder-never-delete pattern): rerank and gate IN, never remove ─────────────
def test_semantic_reorders_but_never_removes(monkeypatch):
    nodes, by_id = corpus(N("strong", "flux capacitor calibration guide"),
                          N("weak", "flux calibration troubleshooting"), *filler())
    q = "flux capacitor calibration"
    lex = pack_for(nodes, by_id, q, {}, monkeypatch, limit=50)
    hyb = pack_for(nodes, by_id, q, {"weak": 0.95, "strong": 0.0}, monkeypatch,
                   limit=50)
    assert {s["id"] for s in lex["seeds"]} <= {s["id"] for s in hyb["seeds"]}, \
        "a lexically-qualified node must survive any semantic opinion of it"
    ids = [s["id"] for s in hyb["seeds"]]
    assert ids.index("weak") < ids.index("strong"), \
        "…and a strong cosine is allowed to REORDER the pair"


def test_paraphrase_gate_in(monkeypatch):
    """The founding case in miniature: zero shared vocabulary, high cosine —
    the exact query shape that ranked the right node #1 in 0 of 8."""
    nodes, by_id = corpus(
        N("target", "circuit breaker patterns for the payment path"), *filler())
    q = "how do we stop cascading failures downstream"
    lex = pack_for(nodes, by_id, q, {}, monkeypatch)
    assert "target" not in {s["id"] for s in lex["seeds"]}, \
        "sanity: lexical alone cannot see the paraphrase target"
    hyb = pack_for(nodes, by_id, q, {"target": 0.82}, monkeypatch)
    s = next((s for s in hyb["seeds"] if s["id"] == "target"), None)
    assert s is not None, "a strong-enough cosine gates the node IN"
    assert s["score_lexical"] == 0.0 and s["score_semantic"] == 0.82
    assert s["match_in_excerpt"] is True, \
        "no lexical body term to locate — True by vacuity, not False by accident"


def test_fragment_queries_never_engage_semantics(monkeypatch):
    """"quo" means too little to search on — lexically (the prefix floor) AND
    semantically: bge cosines are compressed, so a noise vector clears SEM_BASE
    somewhere and filled packs with arbitrary admissions. Found by the VPS
    suite the first time it ran WITH the model. The first guard exempted
    corpus-KNOWN short tokens and the live corpus falsified that the same day
    ("status quo" in one body made "quo" known); knownness makes a token
    exact-matchable, not meaningful — so the rule is now unconditional."""
    called = []

    def spy(live, q):
        called.append(q)
        return {}

    monkeypatch.setattr(context.embed, "similarities", spy)
    # …even when the corpus KNOWS the word — the exact falsifying case
    nodes, by_id = corpus(N("y", "the quo files"), *filler())
    p = context.assemble_context(nodes, by_id, "quo", Path("/nonexistent"))
    assert not called, \
        "a sub-PREFIX_MIN fragment must not consult the encoder, known or not"
    assert all(s["score_semantic"] is None for s in p["seeds"]), \
        "…while lexical exact-matching of the known word is untouched"
    # a query with one meaningful token engages normally
    context.assemble_context(nodes, by_id, "quo files", Path("/nonexistent"))
    assert called, "a >= PREFIX_MIN token makes the query semantic-eligible"


def test_sem_base_is_an_admission_floor(monkeypatch):
    nodes, by_id = corpus(N("meh", "granite countertop maintenance"))
    p = pack_for(nodes, by_id, "cascading failures downstream",
                 {"meh": context.SEM_BASE - 0.01}, monkeypatch)
    assert p["seeds"] == [], "below SEM_BASE, semantics cannot admit a candidate"


def test_sem_topk_caps_semantic_only_admission(monkeypatch):
    extras = [N(f"s{i}", f"padding node number{i}")
              for i in range(context.SEM_TOPK + 4)]
    nodes, by_id = corpus(*extras)
    sims = {f"s{i}": 0.9 - i * 0.001 for i in range(len(extras))}
    p = pack_for(nodes, by_id, "cascading failures downstream", sims, monkeypatch,
                 limit=50)
    assert len(p["seeds"]) == context.SEM_TOPK


def test_sem_topk_scales_with_the_live_corpus(monkeypatch):
    """2026-09-12: an absolute cap of ten admitted a shrinking fraction of a
    growing corpus — the cap-table record for a paraphrase query dropped from
    semantic rank 4 of 318 to 16 of 490 with an unchanged cosine and left the
    pool entirely. Admission is now the larger of SEM_TOPK and SEM_TOPK_FRAC
    of the live count: 10 stays 10 at 14 nodes (the test above), 400 nodes
    admit 20."""
    n = 400
    extras = [N(f"s{i}", f"padding node number{i}") for i in range(n)]
    nodes, by_id = corpus(*extras)
    sims = {f"s{i}": 0.9 - i * 0.0005 for i in range(n)}
    p = pack_for(nodes, by_id, "cascading failures downstream", sims, monkeypatch,
                 limit=500)
    assert len(p["seeds"]) == int(n * context.SEM_TOPK_FRAC) == 20
    assert int(n * context.SEM_TOPK_FRAC) > context.SEM_TOPK, "the fraction must be what binds here"


# ── the pack explains itself: the components reconstruct the sort key ─────────
def test_score_decomposition_reconstructs_the_order(monkeypatch):
    nodes, by_id = corpus(N("lex", "flux capacitor calibration guide"),
                          N("both", "flux calibration troubleshooting"), *filler())
    sims = {"both": 0.71, "lex": 0.55, "f0": 0.52}
    p = pack_for(nodes, by_id, "flux capacitor calibration", sims, monkeypatch,
                 limit=50)
    assert p["intent"] is None, "sanity: no prior in play, every prior is 1.0"
    max_raw = max(max(s["score_lexical"] for s in p["seeds"]), context.LEX_REF)
    max_sem = max(max(0.0, c - context.SEM_BASE) for c in sims.values())
    for s in p["seeds"]:
        want = (s["score_lexical"] / max_raw
                + context.SEM_WEIGHT
                * max(0.0, (s["score_semantic"] or 0.0) - context.SEM_BASE)
                / max_sem) * s["prior"]
        assert abs(s["score"] - want) < 0.02, \
            f'{s["id"]}: displayed {s["score"]} vs reconstructed {want:.3f}'
    scores = [s["score"] for s in p["seeds"]]
    assert scores == sorted(scores, reverse=True), \
        "`score` is the value the order actually sorts on"


def test_match_in_excerpt_guards_silent_truncation(monkeypatch):
    deep = N("deep", "doorknob protocol registry",
             body=("filler words " * 40) + "zanzibar appears here only")
    head = N("head", "doorknob protocol handbook", body="zanzibar up front")
    nodes, by_id = corpus(deep, head, *filler())
    p = pack_for(nodes, by_id, "doorknob protocol zanzibar", {}, monkeypatch,
                 body_chars=60, limit=50)
    by = {s["id"]: s for s in p["seeds"]}
    assert by["deep"]["body_truncated"] is True
    assert "zanzibar" in by["deep"]["body"], \
        "section-level excerpts (2026-10-13): the window moves to the match"
    assert by["deep"]["match_in_excerpt"] is True
    assert by["head"]["match_in_excerpt"] is True


def test_match_in_excerpt_still_fires_when_matches_outgrow_the_budget(monkeypatch):
    """Two matched words in two far-apart sections and room for one: the
    excerpt shows the better one and the flag says the other is unseen."""
    body = ("## Opening\n" + "preamble words here\n" * 10 +
            "## Alpha\n" + "zanzibar " + "padding text " * 30 + "\n" +
            "## Middle\n" + "middle filler line\n" * 40 +
            "## Omega\n" + "quixotry " + "padding text " * 30 + "\n")
    deep = N("deep", "doorknob protocol registry", body=body)
    nodes, by_id = corpus(deep, *filler())
    p = pack_for(nodes, by_id, "doorknob protocol zanzibar quixotry", {}, monkeypatch,
                 body_chars=500, limit=50)
    s = {s["id"]: s for s in p["seeds"]}["deep"]
    assert s["excerpt_sections"], "a deep section was stitched in"
    assert s["match_in_excerpt"] is False, "one matched section did not fit"


def test_excerpt_keeps_the_opening_when_the_match_is_up_front(monkeypatch):
    body = "## Opening\nzanzibar is right here\n" + "## Later\n" + "padding text\n" * 300
    n = N("n", "doorknob protocol registry", body=body)
    nodes, by_id = corpus(n, *filler())
    p = pack_for(nodes, by_id, "doorknob zanzibar", {}, monkeypatch, body_chars=400)
    s = p["seeds"][0]
    assert s["body"] == body[:400] + "…" and s["excerpt_sections"] == []


def test_chunk_vectors_lift_a_node_and_name_the_section(monkeypatch):
    """The chunk layer: a node whose best chunk beats its node vector takes the
    chunk's cosine (less CHUNK_DISCOUNT), and the pack names that section."""
    body = "## Opening\nintro\n\n## The deep part\n" + "content line\n" * 20
    a = N("a", "doorknob protocol alpha", body=body)
    b = N("b", "doorknob protocol beta")
    nodes, by_id = corpus(a, b, *filler())
    p = pack_for(nodes, by_id, "doorknob protocol", {"a": 0.60, "b": 0.70}, monkeypatch,
                 chunks={"a": {0: 0.55, 1: 0.90}})
    by = {s["id"]: s for s in p["seeds"]}
    assert p["seeds"][0]["id"] == "a"
    assert by["a"]["score_semantic"] == round(0.90 - context.CHUNK_DISCOUNT, 3)
    assert by["a"]["semantic_section"] == "The deep part"
    assert by["b"]["semantic_section"] is None


def test_chunk_layer_unavailable_is_the_node_ranking(monkeypatch):
    a = N("a", "doorknob protocol alpha")
    b = N("b", "doorknob protocol beta")
    nodes, by_id = corpus(a, b, *filler())
    p = pack_for(nodes, by_id, "doorknob protocol", {"a": 0.60, "b": 0.70}, monkeypatch)
    assert [s["id"] for s in p["seeds"]][:2] == ["b", "a"]
    assert {s["score_semantic"] for s in p["seeds"][:2]} == {0.6, 0.7}


# ── the current_ruling doorway (2026-08-06): one combination rule ─────────────
def test_doorway_classifies_a_query_about_a_ruling():
    intent, signals = context.classify_intent_ex(
        "What is the current ruling on whether sensitive OKF nodes may be "
        "returned in broad retrieval results?")
    assert intent == "current_ruling", "the measured miss that motivated the rule"
    assert "ruling" in signals


def test_doorway_defers_to_the_ordered_cue_table():
    """'decide' words appear in both queries; the table's own intents must claim
    them BEFORE the doorway can see them — precedence is the design."""
    assert context.classify_intent("why did we decide to split the repos") == "history"
    assert context.classify_intent(
        "has the frontend framework been decided yet") == "open_question"


def test_doorway_needs_both_halves():
    assert context.classify_intent("ruling politics annoys everyone") is None
    assert context.classify_intent("what is the fastest bird") is None


def test_intent_signals_surface_in_the_pack(monkeypatch):
    nodes, by_id = corpus(N("d", "repo split decision", type="decision"), *filler())
    p = pack_for(nodes, by_id, "what is the rule on the repo split", {}, monkeypatch)
    assert p["intent"] == "current_ruling"
    assert p["intent_signals"], "a misroute must be diagnosable from the output"


# ── authority metadata (2026-08-07): exposed, never ranked on ─────────────────
def test_pack_carries_authority_metadata(monkeypatch):
    """The caller LLM is the judgment half of retrieval; the owner's 2026-08-07
    ruling: authority is EXPOSED, never turned into a prior — a specialist
    checker legitimately outranks a canonical project map. Seeds carry all
    five fields; related carry canonical/maturity/updated_at only."""
    a = N("a", "flux capacitor calibration guide")
    a.fm.update(canonical=True, maturity="established", updated_at="2026-08-07",
                work_state="open", superseded_in_part="2026-08-01",
                relates_to=["b"])
    b = N("b", "adjacent operational note")
    b.fm.update(maturity="captured", updated_at="2026-08-02")
    nodes, by_id = corpus(a, b, *filler())
    p = pack_for(nodes, by_id, "flux capacitor calibration", {}, monkeypatch)
    s = next(x for x in p["seeds"] if x["id"] == "a")
    assert (s["canonical"], s["maturity"], s["updated_at"]) == \
        (True, "established", "2026-08-07")
    assert (s["work_state"], s["superseded_in_part"]) == ("open", "2026-08-01")
    r = next(x for x in p["related"] if x["id"] == "b")
    assert (r["canonical"], r["maturity"], r["updated_at"]) == \
        (False, "captured", "2026-08-02")
    assert "work_state" not in r, \
        "related stay briefs — the two log-lifecycle fields are seed-only"


# ── concurrency: searches run in threads since 2026-08-07 ─────────────────────
def test_model_load_is_single_flight_and_nonblocking(model_loads, monkeypatch):
    """The web endpoint is threadpooled and both services warm in a background
    thread, so a cold model must be loaded by exactly ONE thread while every
    other caller degrades to lexical INSTANTLY. The pre-fix behaviour was
    either a double load (2 × ~250 MB on a 961 MB box) or callers queued
    behind a ~15s load — the 'why is it getting stuck' of 2026-08-07."""
    import threading as th
    import time
    import types

    calls = []

    class SlowTE:
        def __init__(self, name):
            calls.append(name)
            time.sleep(0.4)

    mod = types.ModuleType("fastembed")
    mod.TextEmbedding = SlowTE
    monkeypatch.setitem(sys.modules, "fastembed", mod)
    monkeypatch.setattr(embed, "_STATE", {})

    loader = th.Thread(target=embed._model)
    loader.start()
    time.sleep(0.1)     # loader is mid-load, holding the single-flight lock
    assert embed._model() is None, "a caller during the load degrades, never waits"
    assert embed.similarities([N("a", "anything")], "query") == {}, \
        "…and similarities fail-softs to the lexical contract"
    loader.join()
    assert embed._model() is not None
    assert len(calls) == 1, "exactly one load, however many callers raced"


def test_warm_preloads_model_and_vectors(model_loads, tmp_path, monkeypatch):
    """warm() is what each service runs in a daemon thread at start: after it
    returns, a query pays neither the model load nor the corpus embed."""
    np = pytest.importorskip("numpy")
    import types

    class FakeTE:
        def __init__(self, name):
            pass

        def passage_embed(self, texts, **kw):
            for _ in texts:
                yield np.ones(4, dtype="float32")

        def query_embed(self, texts):
            for _ in texts:
                yield np.ones(4, dtype="float32")

    mod = types.ModuleType("fastembed")
    mod.TextEmbedding = FakeTE
    monkeypatch.setitem(sys.modules, "fastembed", mod)
    monkeypatch.setenv("OKF_EMB_CACHE", str(tmp_path / "warm.npz"))
    monkeypatch.setattr(embed, "_STATE", {})

    live = [N("a", "alpha node"), N("b", "beta node")]
    msgs = []
    embed.warm(lambda: live, log=msgs.append)
    assert any("2 node vectors ready" in m for m in msgs), msgs
    assert set(embed.similarities(live, "gamma")) == {"a", "b"}, \
        "post-warm queries are served without any further loading"


def test_chunk_layer_is_all_or_nothing_and_never_embeds_the_corpus_in_a_request(model_loads,
        tmp_path, monkeypatch):
    """Chunk vectors (2026-10-13): a few missing chunks are embedded inline; more
    than CHUNK_INLINE_MAX and the query gets {} (the node-vector ranking) — and
    only a warmed service starts the background embed, never a test or stdio."""
    np = pytest.importorskip("numpy")
    import types
    calls = []

    class FakeTE:
        def __init__(self, name):
            pass

        def passage_embed(self, texts, **kw):
            calls.append(len(texts))
            for _ in texts:
                yield np.ones(4, dtype="float32")

        def query_embed(self, texts):
            for _ in texts:
                yield np.ones(4, dtype="float32")

    mod = types.ModuleType("fastembed")
    mod.TextEmbedding = FakeTE
    monkeypatch.setitem(sys.modules, "fastembed", mod)
    monkeypatch.setenv("OKF_EMB_CACHE", str(tmp_path / "e.npz"))
    monkeypatch.setattr(embed, "_STATE", {})
    sec = "\n".join(f"## Part {k}\n" + "a sentence of real content here. " * 4
                    for k in range(5))
    live = [N(f"n{i}", f"node {i}", body=sec) for i in range(3)]      # 15 chunks
    monkeypatch.setattr(embed, "CHUNK_INLINE_MAX", 10)
    assert embed.chunk_similarities(live, "q") == {}, "15 missing > 10: no partial layer"
    assert "chunk_bg" not in embed._STATE, "an unwarmed process never embeds in the background"
    assert calls == []
    monkeypatch.setattr(embed, "CHUNK_INLINE_MAX", 64)
    got = embed.chunk_similarities(live, "q")
    assert set(got) == {"n0", "n1", "n2"} and all(len(v) == 5 for v in got.values())
    assert embed.chunk_cache_path().exists()
    # an unchanged corpus is served from memory: nothing embedded again
    n = sum(calls)
    embed.chunk_similarities(live, "another")
    assert sum(calls) == n
    assert embed.chunk_status() == {"state": "ready", "chunks": 15}


def test_a_corpus_wide_chunk_embed_saves_as_it_goes_and_reports_failure(model_loads,
        tmp_path, monkeypatch):
    """The first deploy left the layer off with nothing visible from outside:
    a killed or failing embed must keep its progress on disk, and say so in
    chunk_status (which every pack carries as `sections`)."""
    np = pytest.importorskip("numpy")
    import types
    batches = []

    class Flaky:
        def __init__(self, name):
            pass

        def passage_embed(self, texts, **kw):
            batches.append(len(texts))
            if len(batches) == 3:
                raise MemoryError("arena could not grow")
            for _ in texts:
                yield np.ones(4, dtype="float32")

    mod = types.ModuleType("fastembed")
    mod.TextEmbedding = Flaky
    monkeypatch.setitem(sys.modules, "fastembed", mod)
    monkeypatch.setenv("OKF_EMB_CACHE", str(tmp_path / "e.npz"))
    monkeypatch.setattr(embed, "_STATE", {})
    monkeypatch.setattr(embed, "CHUNK_SAVE_EVERY", 64)
    sec = "\n".join(f"## Part {k}\n" + "a sentence of real content here. " * 4
                    for k in range(10))
    live = [N(f"n{i}", f"node {i}", body=sec) for i in range(20)]      # 200 chunks
    with pytest.raises(MemoryError):
        embed._embed_chunks_blocking(live)
    st = embed.chunk_status()
    assert st["state"] == "failed" and "MemoryError" in st["error"]
    assert len(embed._load_chunks()) == 128, "two completed steps survived on disk"
    # a restart resumes from the saved progress instead of from zero
    monkeypatch.setattr(embed, "_STATE", {})
    batches.clear()
    mod.TextEmbedding = type("Ok", (Flaky,), {"passage_embed": lambda self, t, **k:
                                              (np.ones(4, dtype="float32") for _ in t)})
    assert embed._embed_chunks_blocking(live) == 72
    assert embed.chunk_status() == {"state": "ready", "chunks": 200}


def test_web_search_endpoint_is_threadpooled():
    """Starlette runs SYNC endpoints in its threadpool; `async def` runs ON the
    event loop, where the model load froze the entire app (2026-08-07 — every
    surface stuck, not just search). Pinned from SOURCE TEXT, not import:
    importing okf_web needs a corpus (so the pin would skip on the Mac AND in
    a bare VPS pytest run, where OKF_ROOT is unset) and starts the background
    pusher as a side effect. Reading the file pins the one-word regression
    (`def` back to `async def`) on every machine, side-effect-free."""
    import re
    src = (Path(__file__).resolve().parent.parent / "okf_web" / "server.py")\
        .read_text(encoding="utf-8")
    assert re.search(r"^def search\(", src, re.M), "search endpoint missing?"
    assert not re.search(r"^async def search\(", src, re.M), \
        "search must stay sync so Starlette threadpools it"


# ── the classifier, measured separately from ranking (2026-08-07 batch) ───────
def test_intent_misroute_ceiling():
    """Precision-first, encoded as the regression gate the ruling asked for:
    classification is scored on its own, abstention on a labeled query is
    FREE, and a firing that contradicts the label is a MISROUTE. The
    three-variant sweep (2026-08-07) is why this gate exists INSTEAD of cue
    deletion: removing modal cues left top-1 flat, cost top-5/r@15/paraphrase,
    and INCREASED misroutes via fall-through — in an ordered table a deleted
    cue yields the next matching intent, not null. Measured on the 64-entry
    set: 7 misroutes (6 pre-existing and non-modal, plus the accepted checker
    overfire, marked in the YAML). Ceiling 9 = measured + headroom, so cue
    edits that add misroutes fail here while ordinary benchmark growth does
    not; the strict floor catches a classifier collapse."""
    import yaml
    bench = yaml.safe_load((Path(__file__).resolve().parent /
                            "retrieval_benchmark.yaml").read_text())["benchmark"]
    strict = abstained = 0
    misroutes = []
    for e in bench:
        labels = e.get("intent")
        labels = labels if isinstance(labels, list) else [labels]
        pred = context.classify_intent(e["query"])
        if pred in labels:
            strict += 1
        elif pred is None:
            abstained += 1
        else:
            misroutes.append((e["query"][:44], labels, pred))
    assert len(misroutes) <= 9, misroutes
    assert strict >= 30, f"classifier collapse: only {strict} strict hits " \
                         f"({abstained} abstentions, {len(misroutes)} misroutes)"


# ── embed.py's own fail-soft, without the model ───────────────────────────────
def test_embed_unavailable_degrades_to_empty(monkeypatch):
    monkeypatch.setattr(embed, "_STATE", {"model": False})
    live = [N("a", "anything")]
    assert embed.available() is False
    assert embed.node_vectors(live) is None
    assert embed.similarities(live, "query") == {}, \
        "the entire contract: {} means the caller runs pure lexical"


def test_node_text_is_metadata_first_and_bounded():
    n = N("a", "Title Words", summary="the summary", tags=["tag-one"],
          body="b" * 9000)
    t = embed._node_text(n)
    head = "Title Words\nthe summary\ntag-one\n"
    assert t.startswith(head), "deliberate words lead; prose follows"
    assert len(t) <= len(head) + 4000
    assert embed._h(t) == embed._h(t) and embed._h(t) != embed._h(t + "x")


def test_node_vectors_embed_in_bounded_batches(tmp_path, monkeypatch):
    """fastembed's DEFAULT batch (256 docs) allocates a ~3.0 GB attention
    buffer at this corpus's ~512-token node texts — measured live 2026-08-07,
    it OOM-killed the first VPS embed. The bounded batch size must actually
    reach the encoder; dropping the kwarg re-breaks the box silently, because
    similarities() fail-softs the crash into {} and retrieval just goes
    quietly lexical."""
    np = pytest.importorskip("numpy")
    seen = {}

    class FakeModel:
        def passage_embed(self, texts, **kw):
            seen["batch_size"] = kw.get("batch_size")
            for _ in texts:
                yield np.ones(4, dtype="float32")

    monkeypatch.setenv("OKF_EMB_CACHE", str(tmp_path / "emb.npz"))
    monkeypatch.setattr(embed, "_STATE", {"model": FakeModel(), "vecs": {}})
    out = embed.node_vectors([N("a", "t1"), N("b", "t2")])
    assert set(out) == {"a", "b"}
    assert seen["batch_size"] == embed.EMB_BATCH
    assert embed.EMB_BATCH <= 32, "the whole point is a SMALL bound"


def test_vector_cache_roundtrip(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setenv("OKF_EMB_CACHE", str(tmp_path / "emb.npz"))
    mem = {"okf_x": ("hash1", np.ones(4, dtype="float32")),
           "okf_y": ("hash2", np.zeros(4, dtype="float32"))}
    embed._save_disk(mem)
    back = embed._load_disk()
    assert set(back) == {"okf_x", "okf_y"}
    assert back["okf_x"][0] == "hash1" and np.allclose(back["okf_x"][1], 1.0)


def test_missing_cache_is_empty_not_error(tmp_path, monkeypatch):
    monkeypatch.setenv("OKF_EMB_CACHE", str(tmp_path / "absent.npz"))
    assert embed._load_disk() == {}


# ── hosted code graphs (2026-08-08): name containment, pure ───────────────────
def test_codegraph_name_containment(tmp_path):
    """The Code tab serves files by NAME from a fixed directory, and the pure
    resolver is the entire security contract: whitelist alphabet, path built
    from the validated stem alone (traversal unrepresentable), missing file is
    None not error. Extracted from SOURCE by AST for the same reason the
    threadpool pin reads source text: importing okf_web needs a corpus and
    starts the pusher. AST (not a text-split prelude) because the module runs
    `validate.assert_corpus_root` at import — any extraction that executes
    surrounding module code re-imports that requirement through the back door.
    The one seam, CODEGRAPH_DIR, is injected as the function's global."""
    import ast
    src = (Path(__file__).resolve().parent.parent / "okf_web" / "server.py") \
        .read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "codegraph_file"),
              None)
    assert fn is not None, "codegraph_file gone from server.py — route unguarded?"
    import re
    ns = {"re": re, "Path": Path, "CODEGRAPH_DIR": tmp_path}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "server.codegraph_file",
                 "exec"), ns)
    resolver = ns["codegraph_file"]
    (tmp_path / "ledger.html").write_text("<b>ok</b>")
    assert resolver("ledger") == tmp_path / "ledger.html"
    assert resolver("missing") is None, "unknown name is None, not an exception"
    for bad in ("../etc/passwd", "a/b", "a finance app", "ledger.html", "x" * 65, "", None):
        assert resolver(bad) is None, f"must reject {bad!r}"


def test_routes_reference_only_earlier_definitions():
    """server.py's import is untestable on the Mac (corpus assert), so a
    handler registered in `routes` but defined BELOW it is a NameError the
    local suite cannot see — it crash-loops the VPS instead, which is exactly
    how the codegraph handlers shipped on 2026-08-08 (and the on_startup
    kwarg before them). This walks the module AST in order: every name the
    `routes` assignment loads must be bound earlier at module level."""
    import ast
    import builtins
    src = (Path(__file__).resolve().parent.parent / "okf_web" / "server.py") \
        .read_text(encoding="utf-8")
    tree = ast.parse(src)
    bound = set(dir(builtins))
    for node in tree.body:
        is_routes = (isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "routes"
                             for t in node.targets))
        if is_routes:
            loaded = {n.id for n in ast.walk(node.value)
                      if isinstance(n, ast.Name)}
            missing = loaded - bound
            assert not missing, \
                f"routes references names not yet defined: {sorted(missing)}"
            return
        for t in ast.walk(node):
            if isinstance(t, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(t.name)
            elif isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                bound.add(t.id)
            elif isinstance(t, ast.alias):
                bound.add((t.asname or t.name).split(".")[0])
    raise AssertionError("no `routes = [...]` assignment found in server.py")


def test_seeds_carry_tags(monkeypatch):
    """Tags are the strongest lexical field (weight 1.6) and until 2026-09-18 the
    pack withheld them, so a caller reformulating after a miss had to guess the
    corpus's vocabulary. Adjudicating the 25-query suspected-miss queue found the
    misses were phrasing, not ranking — the fix is to hand the caller the words
    that score. Exposed, never ranked on differently."""
    nodes, by_id = corpus(
        N("target", "circuit breaker patterns", tags=("resilience", "payments")),
        *filler())
    p = pack_for(nodes, by_id, "circuit breaker patterns", {}, monkeypatch)
    s = next(s for s in p["seeds"] if s["id"] == "target")
    assert s["tags"] == ["resilience", "payments"]
    assert all(isinstance(x["tags"], list) for x in p["seeds"])


ALIAS_NODE_BODY = """# Retrieval aliases

```yaml
RETRIEVAL_ALIASES:
  - phrase: throw away
    adds: [delete, deletion, purge]
    scope: removing a node from the corpus
```
"""


def test_alias_unions_pools_and_keeps_every_original_candidate(monkeypatch):
    """The corpus-owned thesaurus (2026-09-21): an alias phrase in the
    query runs its ADDS words as a second lexical search and the pools are
    UNIONED. Appending aliases to the query was measured to displace 8–9 of 15
    results (the coverage multiplier shrinks when the query grows), so this
    test pins the structural contract: every candidate the original words found
    is still a candidate, and the alias words admit the target the original
    words could not reach."""
    context._ALIASES.clear()
    thesaurus = N("thes", "Retrieval aliases — the corpus-owned thesaurus", body=ALIAS_NODE_BODY)
    target = N("target", "Deletion is agent-callable behind four gates", tags=("delete", "deletion"))
    bystander = N("notes", "Notes on the assistant and the owner", tags=("assistant", "notes"))
    nodes, by_id = corpus(thesaurus, target, bystander, *filler())
    q = "can the assistant throw away notes on its own"
    plain = pack_for([n for n in nodes if n.id != "thes"], {i: n for i, n in by_id.items() if i != "thes"},
                     q, {}, monkeypatch, limit=50)
    assert "target" not in {s["id"] for s in plain["seeds"]}, "sanity: no alias, no target"
    assert plain["aliases_fired"] == []
    with_alias = pack_for(nodes, by_id, q, {}, monkeypatch, limit=50)
    assert with_alias["aliases_fired"] == ["throw away"]
    after = {s["id"] for s in with_alias["seeds"]}
    assert "target" in after, "the alias words admit the node the original words could not reach"
    assert {s["id"] for s in plain["seeds"]} <= after, "no original candidate may drop out"
    assert "thes" not in after, "the registry whose line fired is infrastructure, not an answer"
    assert "thes" in {s["id"] for s in pack_for(nodes, by_id, "retrieval aliases thesaurus", {},
                                                 monkeypatch, limit=50)["seeds"]}, \
        "a query about the registry fires no line and finds it"
    # a query without the phrase is untouched by the thesaurus
    p2 = pack_for(nodes, by_id, "notes on the assistant", {}, monkeypatch, limit=50)
    assert p2["aliases_fired"] == [] and "target" not in {s["id"] for s in p2["seeds"]}


def test_body_seeds_omits_bodies_below_the_line(monkeypatch):
    """Pack width (2026-09-21): fifteen seeds, bodies on the top
    eight. Below the line a seed says body_omitted and carries no text, so the
    pack grows by summaries, not by bodies."""
    nodes, by_id = corpus(*[N(f"w{i}", f"widget caching note {i}", body="body text " * 50) for i in range(6)],
                          *filler())
    p = pack_for(nodes, by_id, "widget caching", {}, monkeypatch, limit=6, body_seeds=2)
    assert len(p["seeds"]) == 6
    assert all(s["body_omitted"] is False and s["body"] for s in p["seeds"][:2])
    assert all(s["body_omitted"] is True and s["body"] == "" for s in p["seeds"][2:])
    p_all = pack_for(nodes, by_id, "widget caching", {}, monkeypatch, limit=6)
    assert all(s["body_omitted"] is False for s in p_all["seeds"]), "None keeps every body (web list)"


def test_asks_are_indexed_and_fingerprinted(monkeypatch):
    """`asks` (node.schema v1.9, 2026-09-29): everyday phrasings a stranger would type,
    indexed with the summary so the word layer has question-shaped text to match.
    The adoption measurement on 532 nodes: stranger findability 9–11 → 14–16 of 20,
    benchmark 51/61/66 → 56/66/67, no regression. The fingerprint must change when
    asks change, or the cached index would keep serving the old vocabulary."""
    target = N("target", "Backtest is not live edge — pilot small before sizing up")
    target.fm["asks"] = ["should we bet big on something that worked on old data"]
    nodes, by_id = corpus(target, *filler())
    p = pack_for(nodes, by_id, "worked on old data, bet big", {}, monkeypatch)
    assert p["seeds"] and p["seeds"][0]["id"] == "target", "the asks words carry the match"
    fp_before = context._fingerprint(nodes)
    target.fm["asks"] = ["a different phrasing entirely"]
    assert context._fingerprint(nodes) != fp_before, "asks are part of the index fingerprint"
    assert "bet big" in embed._node_text(target) or "different phrasing" in embed._node_text(target)


def test_a_corpus_wide_node_change_is_reembedded_in_the_background(model_loads, tmp_path, monkeypatch):
    """The 2026-09-29 migration re-embedded every node inside one request (two
    MCP timeouts, RSS 232 → 587 MB). A service now serves the previous vectors
    at once and refreshes in a background thread; a non-service process (tests,
    scripts, the gate) still embeds inline, because a measurement needs exact
    vectors."""
    np = pytest.importorskip("numpy")
    import types
    gen = {"v": 1.0}

    class FakeTE:
        def __init__(self, name):
            pass

        def passage_embed(self, texts, **kw):
            for _ in texts:
                yield np.full(4, gen["v"], dtype="float32") + np.arange(4, dtype="float32")

        def query_embed(self, texts):
            for _ in texts:
                yield np.ones(4, dtype="float32")

    mod = types.ModuleType("fastembed")
    mod.TextEmbedding = FakeTE
    monkeypatch.setitem(sys.modules, "fastembed", mod)
    monkeypatch.setenv("OKF_EMB_CACHE", str(tmp_path / "e.npz"))
    monkeypatch.setattr(embed, "_STATE", {})
    monkeypatch.setattr(embed, "NODE_INLINE_MAX", 5)
    live = [N(f"n{i}", f"node {i}") for i in range(12)]
    old = embed.node_vectors(live)                      # first embed: inline (not serving)
    assert len(old) == 12

    for n in live:
        n.fm["summary"] = "every node changed at once"
    gen["v"] = 5.0
    embed._STATE["serving"] = True
    stale = embed.node_vectors(live)
    assert all(np.allclose(stale[i], old[i]) for i in old), "previous vectors served at once"
    assert "node_refresh" in embed.chunk_status(), "the pack says a refresh is running"
    embed._STATE["node_bg"].join(timeout=10)
    fresh = embed.node_vectors(live)
    assert not any(np.allclose(fresh[i], old[i]) for i in old), "refreshed in the background"
    assert "node_refresh" not in embed.chunk_status()

    for n in live:
        n.fm["summary"] = "changed again"
    embed._STATE["serving"] = False                     # a script: inline, exact
    gen["v"] = 9.0
    inline = embed.node_vectors(live)
    assert not any(np.allclose(inline[i], fresh[i]) for i in fresh)


# ── bullet summaries and retrieval keys (2026-10-22) ─────────────────────────
BULLETED = ("- Doorknob protocol, the ruling on hinges.\n"
            "- Hinges are oiled monthly → § Opening\n"
            "- Zanzibar brass needs no oil → § Deep part")


def test_a_bullet_or_ask_carries_the_score_and_says_so(monkeypatch):
    """Each bullet and each ask is matched on its own (embed._keys): a key whose
    cosine beats the node vector lifts the node exactly as a section does, and
    the pack names the layer rather than a heading."""
    a = N("a", "doorknob protocol alpha", summary=BULLETED)
    b = N("b", "doorknob protocol beta")
    nodes, by_id = corpus(a, b, *filler())
    p = pack_for(nodes, by_id, "doorknob protocol", {"a": 0.60, "b": 0.70}, monkeypatch,
                 chunks={"a": {"b2": 0.90, "a0": 0.80}})
    by = {s["id"]: s for s in p["seeds"]}
    assert p["seeds"][0]["id"] == "a"
    assert by["a"]["score_semantic"] == round(0.90 - context.CHUNK_DISCOUNT, 3)
    assert by["a"]["semantic_section"] == "(summary)"
    p = pack_for(nodes, by_id, "doorknob protocol", {"a": 0.60, "b": 0.70}, monkeypatch,
                 chunks={"a": {0: 0.55, "a3": 0.90}})
    assert {s["id"]: s for s in p["seeds"]}["a"]["semantic_section"] == "(asks)"


def test_a_bulleted_seed_shows_matching_sections_not_the_opening(monkeypatch):
    body = ("## Opening\nintro words about hinges\n\n## Deep part\n"
            + "padding line\n" * 150 + "zanzibar brass detail here\n")
    n = N("n", "doorknob protocol registry", summary=BULLETED, body=body)
    nodes, by_id = corpus(n, *filler())
    s = pack_for(nodes, by_id, "doorknob zanzibar", {}, monkeypatch, body_chars=800)["seeds"][0]
    assert "intro words" not in s["body"] and "zanzibar" in s["body"]
    assert s["summary"] == BULLETED, "a seed carries every bullet"
    # nothing in the body matches: the bullets are the whole answer
    s = pack_for(nodes, by_id, "doorknob registry", {}, monkeypatch, body_chars=800)["seeds"][0]
    assert s["body"] == ""


def test_a_prose_seed_keeps_its_opening(monkeypatch):
    body = "## Opening\nintro words\n\n## Later\n" + "padding line\n" * 150
    n = N("n", "doorknob protocol registry", summary="a prose summary of it", body=body)
    nodes, by_id = corpus(n, *filler())
    s = pack_for(nodes, by_id, "doorknob registry", {}, monkeypatch, body_chars=400)["seeds"][0]
    assert s["body"].startswith("## Opening\nintro words")


def test_a_related_brief_carries_only_the_first_bullet():
    n = N("n", "t", summary=BULLETED)
    assert context._brief(n, full=False)["summary"] == "Doorknob protocol, the ruling on hinges."
    assert context._brief(n)["summary"] == BULLETED
    p = N("p", "t", summary="prose stays whole")
    assert context._brief(p, full=False)["summary"] == "prose stays whole"


def test_the_chunk_plan_carries_asks_and_bullets_and_tracks_them(monkeypatch):
    monkeypatch.setattr(embed, "_STATE", {})
    n = N("n", "t", summary=BULLETED, body="")
    n.fm["asks"] = ["how often are hinges oiled", "does brass need oil"]
    keys = {(i, c): t for i, c, _, t in embed._chunk_plan([n])}
    assert keys == {("n", "a0"): "how often are hinges oiled",
                    ("n", "a1"): "does brass need oil",
                    ("n", "b0"): "Doorknob protocol, the ruling on hinges.",
                    ("n", "b1"): "Hinges are oiled monthly",
                    ("n", "b2"): "Zanzibar brass needs no oil"}
    n.fm["summary"] = "prose now"
    assert {c for _, c, _, _ in embed._chunk_plan([n])} == {"a0", "a1"}, \
        "a summary change rebuilds the plan; prose gets no bullet keys"


# ── review fixes (2026-10-23) ─────────────────────────────────────────────────
def test_a_fired_registry_is_not_readmitted_by_meaning(monkeypatch):
    """The registry that carries a fired alias leaves the lexical pool; the
    semantic step used to admit it straight back when its cosine cleared
    SEM_BASE, and it ranked first above the answer it exists to bridge to."""
    context._ALIASES.clear()
    thesaurus = N("thes", "Retrieval aliases — the corpus-owned thesaurus", body=ALIAS_NODE_BODY)
    target = N("target", "Deletion is agent-callable behind four gates", tags=("delete", "deletion"))
    nodes, by_id = corpus(thesaurus, target, *filler())
    p = pack_for(nodes, by_id, "throw away", {"thes": 0.95, "target": 0.60}, monkeypatch)
    assert p["aliases_fired"] == ["throw away"]
    ids = [s["id"] for s in p["seeds"]]
    assert "thes" not in ids and ids[0] == "target", ids


def test_the_best_section_is_shown_even_when_it_overflows_the_budget(monkeypatch):
    """A weaker section that fits must not stand in for the best one that does
    not: the excerpt shows a window of the best section instead."""
    body = ("## Opening\n" + "padding line\n" * 60 + "## Aside\nzanzibar aside\n\n"
            "## Main answer\n" + "zanzibar doorknob main answer text. " * 30)
    n = N("n", "registry of hinges", summary="a prose summary of it", body=body)
    nodes, by_id = corpus(n, *filler())
    s = pack_for(nodes, by_id, "zanzibar doorknob", {}, monkeypatch, body_chars=500)["seeds"][0]
    assert s["excerpt_sections"] == ["Main answer"], s["excerpt_sections"]
    assert "main answer text" in s["body"] and "zanzibar aside" not in s["body"]


# ── the lexical floor on a small corpus (2026-11-01) ─────────────────────────
def test_lex_floor_is_unchanged_from_fifty_live_nodes():
    """The fix for small corpora must not move ranking on a grown one."""
    assert context.lex_floor(50) == context.lex_floor(600) == context.LEX_FLOOR
    assert context.lex_floor(39) < context.LEX_FLOOR


def test_a_word_in_two_bodies_is_found_without_the_model_on_a_small_corpus(monkeypatch):
    """The public audit's failing case (2026-11-01): forty notes, a word in two
    of them, one only in a body. With the model off the fixed 1.5 floor dropped
    the body mention — a word in 5% of a 40-node corpus clears no fixed bar —
    and every new corpus starts that small."""
    subject = N("subject", "Quorbulation protocol", tags=("quorbulation",))
    mention = N("mention", "Unrelated meeting notes",
                body="Someone mentioned quorbulation once, in passing.")
    others = [N(f"g{i}", f"ledger page {'abcdefghijklmnopqrstuvwxyzabcdefghijkl'[i]}x")
              for i in range(38)]
    nodes, by_id = corpus(subject, mention, *others)
    ids = [s["id"] for s in pack_for(nodes, by_id, "quorbulation", {}, monkeypatch)["seeds"]]
    assert ids[0] == "subject" and "mention" in ids


def test_semantic_off_switch_never_loads_the_model(monkeypatch):
    """OKF_SEMANTIC=0 runs the lexical path on purpose — CI runs the whole suite
    that way — and must not even try the import."""
    import types
    calls = []
    mod = types.ModuleType("fastembed")
    mod.TextEmbedding = lambda name: calls.append(name)
    monkeypatch.setitem(sys.modules, "fastembed", mod)
    monkeypatch.setattr(embed, "_STATE", {})
    monkeypatch.setenv("OKF_SEMANTIC", "0")
    assert embed.available() is False and calls == []
