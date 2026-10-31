"""Layer 3 — the write-time REDUNDANCY check (the redundancy-check handover).

Layer 2 asks *"what does this node make FALSE?"* and fires unprompted on every write.
This is its complement — *"does a canonical document already STATE this?"* — and until
now it fired only when someone remembered the backlog-reconciliation playbook, the playbook for
reconciling a backlog against a canonical document. The redundancy-check handover put it exactly:

> The write path has an automatic STALENESS check and no automatic REDUNDANCY check.
> They are complements. Only one is wired in.

The backlog-reconciliation playbook's first run found **34 of 71 case lessons already filed**; blanket promotion would
have created 34 duplicates of correctly-filed rules. That is the cost this prevents.

## Why the candidate set is "governing documents", not a similarity cluster

The redundancy-check handover §3 proposed a cluster of same-project nodes intersected on tags. Measured against the
live corpus, that is unusable: **median 17, p90 31, 67% over twelve** — within a project,
rule-bearing nodes nearly all share tags. Capping it truncated 67–87% of clusters, which
would have hidden candidates rather than ranking them. An IDF-scored version at a high
threshold still gave median 11.

The question is not "what is nearby" but **"what already governs this domain"** — which is
precisely what the backlog-reconciliation playbook assumes the reader already knows. So the set is: **canonical decisions
and playbooks in the same project that share a topic.** Measured at **median 12, p90 15,
max 17** — and stable, because it does not grow every time a note is filed.

## Warn only, and the asymmetry that sets the bar

Never blocks — consistent with *detection is not declaration* (the five-layer audit decision).
The verdict vocabulary is the backlog-reconciliation playbook's, and so is the tie-breaker, which matters more than the
cluster:

> **A false COVERED loses a rule permanently and silently; a false MISSING merely creates
> a duplicate you will notice.** When unsure, say PARTIAL.

## Honest about what it cannot see

Project-scoped, so it will miss a rule already stated in another project's canonical doc —
the cross-domain duplicate. The returned question says so, rather than letting a clean pass
imply the corpus is clean. Same discipline as Layer 2 stating its ~69% ceiling.
"""

from __future__ import annotations

try:
    from audit import _tags, _tokens
except ImportError:
    from librarian.audit import _tags, _tokens

# p90 of the measured distribution, so truncation is rare and reported when it happens.
DEFAULT_CAP = 15

# Types whose writes state a rule that could already exist. A session/note/log records
# what happened; a decision/pattern/playbook asserts something durable.
RULE_TYPES = {"decision", "pattern", "playbook"}

# Minimum shared free-text tokens when no specific tag is shared.
_MIN_SHARED_TOKENS = 2


def governing_docs(fm: dict, node_id: str | None, nodes: list, cap: int = DEFAULT_CAP):
    """Canonical decisions and playbooks in the same project that share this topic.

    Returns (docs, truncated_count). A node with no project, or a project with no
    canonical documents yet, yields an empty list — the check then stays silent, which
    is correct: there is nothing for the new rule to be redundant WITH.
    """
    project = fm.get("project")
    if not project:
        return [], 0
    active = [n for n in nodes if n.fm.get("status") == "active" and n.id != node_id]
    type_tags = {str(n.fm.get("type", "")).lower() for n in active}

    class _Shim:
        def __init__(self, fm):
            self.fm = fm

    me = _Shim(fm)
    my_tags, my_toks = _tags(me), _tokens(me)

    hits = []
    for n in active:
        if n.fm.get("project") != project:
            continue
        if not (n.fm.get("canonical") is True or n.fm.get("type") == "playbook"):
            continue
        shared_tags = (_tags(n) & my_tags) - type_tags
        shared_toks = _tokens(n) & my_toks
        if not shared_tags and len(shared_toks) < _MIN_SHARED_TOKENS:
            continue
        # rarest shared tag first — a specific overlap is a likelier duplicate than a
        # project-wide one
        hits.append((len(shared_tags), len(shared_toks), n))
    hits.sort(key=lambda r: (-r[0], -r[1]))
    kept = hits[:cap]
    return ([{"id": n.id, "type": n.fm.get("type"), "title": n.fm.get("title", ""),
              "canonical": n.fm.get("canonical") is True,
              "summary": n.fm.get("summary") or ""} for _, _, n in kept],
            max(0, len(hits) - len(kept)))


def _question(truncated: int) -> str:
    q = (
        "LAYER 3 — write-time REDUNDANCY check. WARN ONLY: nothing was blocked and no "
        "further call is required.\n"
        "\n"
        "Layer 2 asks what this node makes FALSE. This asks the complement:\n"
        "\n"
        "    Does one of these documents already STATE this rule?\n"
        "\n"
        "For each, decide:\n"
        "  - COVERED — quote the line. Cite it from this node and do NOT keep the new rule.\n"
        "  - PARTIAL — quote it, state precisely what this adds, and sharpen the EXISTING "
        "rule in place rather than filing a second one.\n"
        "  - MISSING — proceed.\n"
        "  - SCOPE-SPECIFIC — cite the parent and leave it.\n"
        "\n"
        "**An unquoted COVERED is an assertion**, which is what the verification-discipline guardrail "
        "forbids. And the asymmetry that sets the bar: a false COVERED loses a rule "
        "permanently and silently; a false MISSING merely creates a duplicate you will "
        "notice. **When unsure, say PARTIAL.**\n"
        "\n"
        "These are the canonical documents and playbooks in this project that share its "
        "topic — NOT every document that could cover it. The check is project-scoped, so a "
        "rule already stated in another project's canonical doc will not appear here. A "
        "clean pass is not proof the rule is new."
    )
    if truncated:
        q += (f"\n\n⚠ {truncated} further governing document(s) matched and were not shown "
              "(cap reached) — this list is incomplete even within its own scope.")
    return q


def should_check(node_type: str | None) -> bool:
    return isinstance(node_type, str) and node_type in RULE_TYPES


def layer3(fm: dict, node_id: str | None, nodes: list, cap: int = DEFAULT_CAP):
    """The advisory block, or None when there is nothing to check."""
    if not should_check(fm.get("type")):
        return None
    docs, truncated = governing_docs(fm, node_id, nodes, cap)
    if not docs:
        return None
    return {"check": "layer3_redundancy", "blocking": False,
            "question": _question(truncated), "governing_documents": docs}
