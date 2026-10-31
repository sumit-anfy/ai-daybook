"""Status + maturity state machine — mirrors node.schema §4 TRANSITIONS.

This is the enforcement the CLI validator cannot do: a transition needs a
before-state. It lives here so the write path (store.update_node) can call it.
"""
from __future__ import annotations


class TransitionError(ValueError):
    """Raised when a requested status/maturity change is not permitted."""


# node.schema §4 TRANSITIONS.status. None = birth. retired = terminal.
STATUS_TRANSITIONS: dict = {
    None: {"active", "parked"},
    "active": {"parked", "superseded", "retired"},
    "parked": {"active", "superseded", "retired"},
    "superseded": {"active"},   # gated: requires_un_supersession
    "retired": set(),           # terminal
}

# node.schema §4 TRANSITIONS.maturity — forward-only.
MATURITY_ORDER = ["captured", "candidate", "established"]


def check_birth_status(status) -> None:
    """A node may only be born active or parked — never retired or superseded
    (born-dead / born-replaced-by-nothing are meaningless). Maturity, by contrast,
    may be born at any level (see node.schema §4: `captured` is the default, not
    the ceiling); only *changes* to maturity are forward-only."""
    if status not in STATUS_TRANSITIONS[None]:
        raise TransitionError(
            f"a node cannot be born with status {status!r}; "
            f"birth allows {sorted(STATUS_TRANSITIONS[None])}")


def check_status_transition(old, new, *, allow_unsupersede: bool = False) -> None:
    if old == new:
        return
    allowed = STATUS_TRANSITIONS.get(old, set())
    if new not in allowed:
        pretty = sorted(allowed) if allowed else "none — terminal state"
        raise TransitionError(
            f"illegal status transition {old!r} -> {new!r} (allowed from {old!r}: {pretty})")
    if old == "superseded" and new == "active" and not allow_unsupersede:
        raise TransitionError(
            "superseded -> active requires un-supersession: pass allow_unsupersede=True "
            "and clear the successor's supersedes pointer first")


def check_maturity_transition(old, new) -> None:
    if old == new:
        return
    if old not in MATURITY_ORDER or new not in MATURITY_ORDER:
        raise TransitionError(f"unknown maturity value in transition {old!r} -> {new!r}")
    if MATURITY_ORDER.index(new) < MATURITY_ORDER.index(old):
        raise TransitionError(
            f"maturity is forward-only; cannot go {old!r} -> {new!r} "
            "(express trust loss via status: retire or supersede)")
