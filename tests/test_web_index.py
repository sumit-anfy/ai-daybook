"""Source-text pins on okf_web/index.html — the one surface no test on any
machine can drive (the Mac cannot import okf_web; the VPS suite runs no
browser). The pattern test_retrieval_semantic uses for server.py, applied to
the page: pin the exact regression, from the file, side-effect-free.
"""
import re
from pathlib import Path

SRC = (Path(__file__).resolve().parent.parent / "okf_web" / "index.html").read_text(encoding="utf-8")


def test_search_query_is_normalised_on_every_side():
    """2026-09-12: `ranked.q===lq` compared a typed-case query with a lowercased
    one, so any capitalised query never showed the ranked union. Every side of
    that comparison, and every request carrying the query, must pass through
    the one normaliser."""
    assert re.search(r"^const norm=s=>String\(s\|\|''\)\.trim\(\)\.toLowerCase\(\);", SRC, re.M)
    assert "ranked&&ranked.q===lq" in SRC, "the comparison the bug lived in still exists"
    assert "lq=norm(e.target.value)" in SRC, "local state goes through norm"
    assert "runSearch(lq)" in SRC, "the ranked request sends the normalised query"
    assert "ranked={q:norm(q)" in SRC, "the ranked result stores the normalised query"
    assert "query:lq,id:n.id" in SRC, "the open record carries the normalised query"
    assert "$('lq').value.trim()" not in SRC and "e.target.value.trim()" not in SRC, \
        "no second, ad-hoc normalisation of the SEARCH query may reappear (editor forms trim their own fields)"
