"""okf-web must not start with a forgeable cookie secret (2026-09-07).

Before this, `okf_web/server.py` fell back to the literal "dev-only-secret" —
a string in a public repo — whenever OKF_WEB_SECRET was unset, and a cookie
signed with it was accepted with a 200 (reproduced by the 2026-09-06 external
review against a throwaway corpus). Tested through `okf_web.config` because
importing the app validates a corpus and starts the background pusher.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from okf_web.config import ConfigError, web_secret      # noqa: E402


def test_missing_secret_refuses_to_start():
    with pytest.raises(ConfigError) as e:
        web_secret({})
    assert "OKF_WEB_SECRET" in str(e.value) and "okf-web.env" in str(e.value), \
        "the refusal must carry the fix — a bare exception in a journal is a mystery"


def test_blank_secret_is_missing():
    with pytest.raises(ConfigError):
        web_secret({"OKF_WEB_SECRET": "   "})


def test_set_secret_is_used_verbatim():
    assert web_secret({"OKF_WEB_SECRET": "abc123"}) == b"abc123"


def test_dev_escape_is_explicit_and_only_then():
    assert web_secret({"OKF_WEB_DEV": "1"}) == b"dev-only-secret"
    with pytest.raises(ConfigError):
        web_secret({"OKF_WEB_DEV": "yes"})     # not a truthy string: exactly "1"


def test_real_secret_beats_dev_flag():
    assert web_secret({"OKF_WEB_DEV": "1", "OKF_WEB_SECRET": "real"}) == b"real"


def test_server_module_holds_no_fallback_literal():
    """Source-text pin (the pattern test_retrieval_semantic uses for okf_web): the
    fallback string must live ONLY behind the dev gate in config.py. If it reappears
    in server.py, the fail-closed contract has been re-opened."""
    src = (Path(__file__).resolve().parent.parent / "okf_web" / "server.py") \
        .read_text(encoding="utf-8")
    assert "dev-only-secret" not in src
    assert "web_secret()" in src, "server.py must obtain SECRET through config.web_secret"
