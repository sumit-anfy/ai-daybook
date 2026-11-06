"""YAML load/dump that uses libyaml (the C implementation) when it is available.

Not a test-only optimisation. `validate.load_corpus` runs on every read, and every WRITE
re-validates the whole corpus, so frontmatter parsing is the hottest path in the Librarian —
it sits under `validate_corpus`, `audit_corpus`, `brief`, and both write tools.

Measured on the author's private corpus; not reproducible from the example corpus.
Measured 2026-07-29 on the 272-node corpus:

    pure-python SafeLoader   228.0 ms   for all frontmatter
    C CSafeLoader             19.6 ms   -> 11.7x

That was ~90% of `load_corpus`'s 255 ms. The measurement also corrected a wrong guess: the
test suite's slowness had been blamed on its fixture copying the corpus per test, which is
only 6.8 s of a ~60 s run. Parsing was the cost.

Verified before adopting, on the real corpus rather than in principle:
  - both loaders return EQUAL objects for all 272 nodes (0 mismatches)
  - both dumpers produce BYTE-IDENTICAL output for all 272 nodes (0 mismatches)

Falls back silently to the pure-Python implementation when libyaml is absent, so a machine
without it still works — just slower. `ACCELERATED` exposes which one is live so a
diagnostic can report it instead of guessing.
"""

import yaml

try:                                                    # libyaml present
    from yaml import CSafeLoader as _Loader, CSafeDumper as _Dumper
    ACCELERATED = True
except ImportError:                                     # pure-python fallback
    from yaml import SafeLoader as _Loader, SafeDumper as _Dumper
    ACCELERATED = False


def safe_load(stream):
    """Drop-in for yaml.safe_load."""
    return yaml.load(stream, Loader=_Loader)


def safe_dump(data, **kwargs):
    """Drop-in for yaml.safe_dump."""
    return yaml.dump(data, Dumper=_Dumper, **kwargs)
