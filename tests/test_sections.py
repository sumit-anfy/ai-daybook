"""librarian/sections.py — a body cut along its own headings (2026-10-13).

Model-free and corpus-free: the splitter's contract is that chunks cover the
body exactly, respect CHUNK_CHARS, ignore headings inside code fences, and keep
the opening H1 out of the display path while the encoder's path keeps it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian import sections                             # noqa: E402

BODY = """# Node title

Opening words.

## First
alpha text

```
## not a heading
```

### Deeper
beta text

## Second
gamma text
"""


def test_chunks_cover_the_body_exactly():
    cs = sections.chunks(BODY)
    assert "".join(BODY[c["start"]:c["end"]] for c in cs) == BODY
    assert [c["i"] for c in cs] == list(range(len(cs)))


def test_paths_follow_the_heading_tree_and_skip_fenced_headings():
    paths = [c["path"] for c in sections.chunks(BODY)]
    assert paths == ["", "First", "First › Deeper", "Second"]


def test_opening_h1_is_kept_only_for_the_encoder():
    cs = sections.chunks(BODY)
    assert cs[1]["full"] == "Node title › First"
    assert sections.embed_text("Node title", cs[1], BODY).startswith(
        "Node title › Node title › First\n")


def test_long_sections_split_at_paragraphs_within_the_cap():
    para = "word " * 100 + "\n\n"
    body = "## Long\n" + para * 8
    cs = sections.chunks(body)
    assert len(cs) > 1 and all(c["end"] - c["start"] <= sections.CHUNK_CHARS for c in cs)
    assert all(body[c["end"] - 2:c["end"]] == "\n\n" for c in cs[:-1]), "cut at a paragraph"
    assert {c["path"] for c in cs} == {"Long"}


def test_empty_and_heading_only_bodies():
    assert sections.chunks("") == []
    c = sections.chunks("## Just a heading\n")[0]
    assert not sections.substantive(c, "## Just a heading\n")
