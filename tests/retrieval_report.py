"""Retrieval report — the full benchmark picture, on a corpus-bearing machine.

Usage:  python tests/retrieval_report.py        (VPS, or OKF_TEST_CORPUS set)

Prints the two measurements the 2026-08-07 batch ruled must never be conflated
again:

  1. CLASSIFICATION — did the intent layer understand the query? Scored on its
     own: strict hits, accepted abstentions (labels may be lists with null =
     abstention-accepted), and MISROUTES listed individually. The regression
     gate lives in test_intent_misroute_ceiling; this is the diagnostic view.
  2. RETRIEVAL — per-intent rank distribution (1 / 2-5 / 6-15 / absent), so a
     weak intent is visible instead of averaged away, plus the paraphrase
     subset that measures the semantic layer's founding job.

The floors in the test suite are collapse detectors; this report is the
inspection surface. Nothing here writes anything.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml                                                # noqa: E402

from _corpus import find_corpus                            # noqa: E402
from librarian import context, validate                    # noqa: E402


def accepted(e: dict) -> list[str]:
    """The original label first, then adjudicated additions (see the yaml header)."""
    return [e["expected"], *(e.get("also_accept") or [])]


def best_rank(ids: list[str], e: dict):
    """1-based rank of the best-placed accepted node, or None."""
    ranks = [ids.index(a) + 1 for a in accepted(e) if a in ids]
    return min(ranks) if ranks else None


def main() -> None:
    root = find_corpus(required=True)
    nodes, by_id, *_ = validate.load_corpus(root)
    bench = yaml.safe_load((Path(__file__).parent / "retrieval_benchmark.yaml")
                           .read_text())["benchmark"]
    entries = [e for e in bench if e["expected"] in by_id]
    dropped = len(bench) - len(entries)
    superseded = sum(1 for e in entries
                     if by_id[e["expected"]].fm.get("status") == "superseded")
    adjudicated = sum(1 for e in entries if e.get("also_accept"))
    para = [e for e in entries if "paraphrase" in (e.get("note") or "")]

    # ── classification, scored on its own ────────────────────────────────────
    strict = abstained = 0
    misroutes = []
    for e in entries:
        labels = e.get("intent")
        labels = labels if isinstance(labels, list) else [labels]
        pred = context.classify_intent(e["query"])
        if pred in labels:
            strict += 1
        elif pred is None:
            abstained += 1
        else:
            misroutes.append((e["query"], labels, pred))

    # ── retrieval, per intent ────────────────────────────────────────────────
    BUCKETS = ("1", "2-5", "6-15", "absent")
    dist: dict = defaultdict(Counter)
    pack_semantic = None
    t1 = t5 = r15 = pt5 = 0
    o1 = o5 = o15 = op5 = 0          # the ORIGINAL-label series, for comparison
    for e in entries:
        pack = context.assemble_context(nodes, by_id, e["query"], root,
                                        limit=15, body_chars=0)
        pack_semantic = pack.get("semantic") if pack_semantic is None else pack_semantic
        ids = [s["id"] for s in pack["seeds"]]
        o = (ids.index(e["expected"]) + 1) if e["expected"] in ids else None
        o1 += (o == 1); o5 += bool(o and o <= 5); o15 += bool(o)
        if e in para:
            op5 += bool(o and o <= 5)
        r = best_rank(ids, e)
        bucket = ("1" if r == 1 else "2-5" if r and r <= 5
                  else "6-15" if r else "absent")
        label = e.get("intent")
        label = label[0] if isinstance(label, list) else label
        dist[label or "(unlabeled)"][bucket] += 1
        t1 += (r == 1)
        t5 += bool(r and r <= 5)
        r15 += bool(r)
        if e in para:
            pt5 += bool(r and r <= 5)

    n = len(entries)
    print(f"retrieval report — {n} entries ({dropped} dropped: target absent from corpus; "
          f"{superseded} superseded targets; {adjudicated} adjudicated), "
          f"{len(by_id)} nodes, semantic={'ON' if pack_semantic else 'OFF (lexical only)'}")
    print(f"\nCLASSIFICATION  strict {strict}/{n}  abstained {abstained}  "
          f"misroutes {len(misroutes)}")
    for q, labels, pred in misroutes:
        want = "/".join(str(x) for x in labels)
        print(f"  misroute  [{want} -> {pred}]  {q[:70]}")

    print(f"\nRETRIEVAL (adjudicated)  top-1 {t1}/{n} ({t1 / n:.0%})   top-5 {t5}/{n} "
          f"({t5 / n:.0%})   r@15 {r15}/{n} ({r15 / n:.0%})   "
          f"paraphrase top-5 {pt5}/{len(para)}")
    print(f"RETRIEVAL (original)     top-1 {o1}/{n} ({o1 / n:.0%})   top-5 {o5}/{n} "
          f"({o5 / n:.0%})   r@15 {o15}/{n} ({o15 / n:.0%})   "
          f"paraphrase top-5 {op5}/{len(para)}")
    print(f"\n{'intent':<16}" + "".join(f"{b:>8}" for b in BUCKETS) + f"{'n':>6}")
    for label in sorted(dist, key=str):
        row = dist[label]
        total = sum(row.values())
        print(f"{str(label):<16}"
              + "".join(f"{row.get(b, 0):>8}" for b in BUCKETS) + f"{total:>6}")

    findability_report(nodes, by_id, root)


def findability_report(nodes, by_id, root) -> None:
    """The findability seed bank (tests/findability_seed.yaml): can a stranger
    who forgot a node's words still reach it — in the default pack of eight
    (bodies), and anywhere in fifteen. Scored per question kind, then per type
    and per project, and every node unreachable in fifteen is NAMED: the
    The asks-and-findability decision approved, with no judge."""
    bank = yaml.safe_load((Path(__file__).parent / "findability_seed.yaml")
                          .read_text())["findability"]
    bank = [b for b in bank if b["node"] in by_id
            and by_id[b["node"]].fm.get("status") in ("active", "parked")]
    print(f"\nFINDABILITY  {len(bank)} live nodes, held-out stranger questions")
    by_group: dict = defaultdict(lambda: [0, 0])
    unreachable = []
    for kind in ("stranger", "short"):
        r1 = r8 = r15 = 0
        for b in bank:
            ids = [s["id"] for s in context.assemble_context(
                nodes, by_id, b[kind], root, limit=15, body_chars=0)["seeds"]]
            r = ids.index(b["node"]) + 1 if b["node"] in ids else None
            r1 += r == 1; r8 += bool(r and r <= 8); r15 += bool(r)
            n = by_id[b["node"]]
            for g in (f"type:{n.fm.get('type')}", f"project:{n.fm.get('project')}"):
                by_group[g][0] += bool(r and r <= 8)
                by_group[g][1] += 1
            if not r:
                unreachable.append((kind, b["node"], n.fm.get("title", "")[:70]))
        print(f"  {kind:<9} top-1 {r1}  in 8 {r8}  in 15 {r15}   of {len(bank)}")
    print("  in 8, both kinds:  " + "  ".join(
        f"{g} {h}/{t}" for g, (h, t) in sorted(by_group.items())))
    for kind, nid, title in unreachable:
        print(f"  unreachable in 15 [{kind}]  {nid}  {title}")


if __name__ == "__main__":
    main()
