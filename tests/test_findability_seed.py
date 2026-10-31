"""The findability seed bank (tests/findability_seed.yaml, rebuilt 2026-10-14)
keeps its shape: the report reads it on a corpus-bearing machine, but its
form is checked everywhere — a bank that silently lost entries or fields
would measure less than it claims (the fail-loudly rule rule 4)."""
import re
from pathlib import Path

import yaml

BANK = yaml.safe_load((Path(__file__).parent / "findability_seed.yaml").read_text())["findability"]


def test_bank_has_twenty_distinct_nodes_with_both_query_kinds():
    assert len(BANK) == 20
    assert len({b["node"] for b in BANK}) == 20
    for b in BANK:
        assert re.fullmatch(r"okf_\d{8}_\d{6}_[0-9a-f]{4}", b["node"])
        assert {"node", "type", "project", "stranger", "short"} <= set(b)


def test_query_lengths_match_their_kind():
    for b in BANK:
        assert 6 <= len(b["stranger"].split()) <= 12, b["stranger"]
        assert 2 <= len(b["short"].split()) <= 5, b["short"]
