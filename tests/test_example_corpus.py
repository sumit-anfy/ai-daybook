"""The bundled example corpus is the quickstart and the suite's fixture, so it
must stay a clean corpus: zero validation errors, zero warnings, every node
type in the registry represented, and the v2.0 shapes the README promises."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian import summary, validate                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "examples" / "corpus"


def test_example_corpus_validates_clean():
    findings, nodes, skipped, types = validate.validate_corpus(ROOT)
    assert not findings, "\n".join(str(f) for f in findings)
    assert not skipped and len(nodes) >= 30


def test_every_registry_type_has_an_example():
    _, nodes, _, types = validate.validate_corpus(ROOT)
    assert types <= {n.fm["type"] for n in nodes}


def test_example_shows_the_v2_shapes():
    _, nodes, *_ = validate.validate_corpus(ROOT)
    live = [n for n in nodes if n.fm["status"] in ("active", "parked")]
    assert all(summary.bullets(n.fm.get("summary")) for n in nodes), "bullet summaries"
    assert all(n.fm.get("asks") for n in live), "asks on every live node"
    assert any(n.fm.get("supersedes") for n in nodes), "a supersession"
    assert any(n.fm["type"] == "session" and "## Close-out" in n.body for n in nodes)
