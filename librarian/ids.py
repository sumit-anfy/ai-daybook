"""Node ID assignment — a Librarian responsibility (build-spec §8, node.schema §2).

IDs are stable across renames and edits and are never derived from title,
filename, or content. Born-in-OKF ids encode a readable creation time plus 4 hex
chars of collision resistance. Migrated-from-Coda ids use the coda_ prefix.
"""
from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone

# Mirrors node.schema §3 SPEC.fields.id.pattern
ID_PATTERN = re.compile(r"^(okf_\d{8}_\d{6}_[0-9a-f]{4}|coda_.+)$")


# How many suffixes to try before giving up on uniqueness. 4 hex = 65,536 values, so
# this only exhausts on an absurdly large corpus — and the fallback is still a VALID,
# unique id, because uniqueness comes from the timestamp, not the suffix.
_SUFFIX_ATTEMPTS = 64


def new_id(now: datetime | None = None, taken_suffixes=frozenset()) -> str:
    """Return a fresh born-in-OKF id: okf_<YYYYMMDD>_<HHMMSS>_<4hex>.

    `taken_suffixes` is the set of last-4s already used in the corpus; a suffix that
    collides is re-rolled. Full ids never collided — the timestamp guarantees that —
    but the corpus, its commit messages and every agent working in it refer to nodes
    by their LAST FOUR characters, and that shorthand is only an identifier while the
    suffixes are distinct.

    This is not defensive coding against a rare event. With a 4-hex suffix the
    birthday bound gives ~43% chance of at least one collision at 272 nodes and ~7.6
    expected colliding pairs at 1,000 — collisions are the EXPECTED case as the corpus
    grows. The first one appeared 2026-07-28 (two nodes minted a week apart, both ending
    `ea8a`) and made seven existing prose references ambiguous.

    Callers that pass nothing keep the old behaviour, so this is safe to adopt
    incrementally.
    """
    now = now or datetime.now(timezone.utc)
    stamp = f"okf_{now:%Y%m%d}_{now:%H%M%S}"
    for _ in range(_SUFFIX_ATTEMPTS):
        suffix = secrets.token_hex(2)
        if suffix not in taken_suffixes:
            return f"{stamp}_{suffix}"
    # Random retries thin out as the space fills, so fall back to picking from the
    # free set directly. This makes the guarantee hold at ANY fill level instead of
    # only at low ones — the retry loop is the fast path, not the promise.
    free = [f"{i:04x}" for i in range(0x10000) if f"{i:04x}" not in taken_suffixes]
    if free:
        return f"{stamp}_{secrets.choice(free)}"
    return f"{stamp}_{secrets.token_hex(2)}"      # still unique; shorthand just isn't


def is_valid_id(value: str | None) -> bool:
    return bool(value) and bool(ID_PATTERN.match(value))
