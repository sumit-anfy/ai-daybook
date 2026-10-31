"""Locate the corpus the tests run against.

OKF_TEST_CORPUS if set, else the bundled example corpus (examples/corpus). The
corpus-dependent tests copy it to a throwaway directory before writing, so the
example is never modified by a test run.

The probe is `schemas/type-registry.md` for the same reason
`validate.assert_corpus_root` uses it: without the registry the type vocabulary
is empty and every node fails, so a directory lacking it is not a corpus no
matter what else it contains. Never return a directory that is not a corpus:
an empty corpus makes the whole suite pass while verifying nothing.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def candidates() -> list[Path]:
    env = os.environ.get("OKF_TEST_CORPUS")
    return ([Path(env).expanduser()] if env else []) + [REPO / "examples" / "corpus"]


def find_corpus(required: bool = True) -> Path | None:
    """Return a corpus, or None when there is none and required=False."""
    for c in candidates():
        if (c / "schemas" / "type-registry.md").exists():
            return c
    if not required:
        return None
    raise RuntimeError(
        "cannot find an OKF corpus. Set OKF_TEST_CORPUS to one (looked in: "
        + ", ".join(str(c) for c in candidates()) + ")")


def skip_if_no_corpus():
    """Module-level guard for corpus-dependent test files. With the example
    corpus in the repo this only skips if someone deleted it — and says so."""
    found = find_corpus(required=False)
    if found is None:
        import pytest
        pytest.skip("no OKF corpus found — examples/corpus is missing; set "
                    "OKF_TEST_CORPUS to a corpus", allow_module_level=True)
    return found
