"""Suite-wide hooks.

The deploy gate (deploy/okf-deploy.sh) keeps only the LAST line of a run as
its verdict, and audit_corpus shows that line — the only part of a refused
deploy the Mac can read, since the laptop cannot reach the droplet's log.
"3 failed, 340 passed" said that something broke but not what (2026-10-22,
revision 9041067). So a run with failures ends on a line that names them.

And no test reads or writes the live service's state in ~/.okf. Two tests
assumed a bare corpus raises only the flags they set up, while compute_health
also read the box's real findability marker; once the 2026-10-19 weekly run
recorded a regression, every revision failed the gate on the droplet and
passed on the Mac, which has no marker. The store tests appended their reads
to the real search log the same way. A test that wants a marker sets its own.
"""
import os

import pytest

# The example corpus carries a ratified-scope node; the check reads its id from
# the environment (off when unset), so the suite turns it on here.
os.environ.setdefault("OKF_RATIFIED_SCOPE_NODE", "okf_20260815_200000_c0de")

_LIVE_STATE = {"OKF_FINDABILITY": "findability.json", "OKF_SEARCH_LOG": "search.jsonl",
               "OKF_DEPLOY_HEALTH": "deploy_health.json", "OKF_DEPLOY_GATE": "deploy_gate.json"}


@pytest.fixture(autouse=True)
def _no_live_state(monkeypatch, tmp_path_factory):
    d = tmp_path_factory.mktemp("okf_state")
    for var, name in _LIVE_STATE.items():
        monkeypatch.setenv(var, str(d / name))


_FAILED: list[str] = []
_PASSED = 0


def pytest_runtest_logreport(report):
    global _PASSED
    if report.failed:
        name = report.nodeid.rsplit("/", 1)[-1]
        _FAILED.append(name if report.when == "call" else f"{name} [{report.when}]")
    elif report.passed and report.when == "call":
        _PASSED += 1


def pytest_unconfigure(config):
    if _FAILED:
        print(f"{len(_FAILED)} failed: {'; '.join(_FAILED)} ({_PASSED} passed)", flush=True)
