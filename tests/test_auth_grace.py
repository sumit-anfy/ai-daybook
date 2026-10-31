"""The grace cache must bridge upstream outages ONLY for identities it has
already seen proven — and must never widen what the allowlist admits.

Origin: 2026-08-17, GitHub's platform outage. Its API 503'd the per-request
identity check, the verifier mapped any non-200 to None, and the client
permanently dropped its grant as "invalid_token" — four times in one afternoon.
"""

from librarian.auth_grace import GraceCache


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make():
    clock = FakeClock()
    return GraceCache(positive_ttl=60, grace_ttl=900, max_size=3, clock=clock), clock


def test_never_verified_token_gets_nothing():
    # The fail-closed property: an invalid token has no entry, so an upstream
    # outage must not admit it.
    cache, clock = make()
    assert cache.fresh("tok") is None
    assert cache.grace("tok") is None


def test_fresh_within_positive_ttl_then_expires():
    cache, clock = make()
    cache.record("tok", "identity")
    assert cache.fresh("tok") == "identity"
    clock.now += 61
    assert cache.fresh("tok") is None  # must re-verify upstream


def test_grace_bridges_failure_within_window():
    cache, clock = make()
    cache.record("tok", "identity")
    clock.now += 600  # upstream now failing; last success 10 min ago
    result, age = cache.grace("tok")
    assert result == "identity"
    assert age == 600


def test_grace_refuses_after_window():
    cache, clock = make()
    cache.record("tok", "identity")
    clock.now += 901
    assert cache.grace("tok") is None


def test_success_resets_the_clock():
    cache, clock = make()
    cache.record("tok", "identity")
    clock.now += 800
    cache.record("tok", "identity")  # a later successful verification
    clock.now += 800  # 1600s after first, 800s after second
    result, age = cache.grace("tok")
    assert result == "identity"
    assert age == 800


def test_expired_entries_are_pruned_on_record():
    cache, clock = make()
    cache.record("old", "a")
    clock.now += 901
    cache.record("new", "b")
    assert "old" not in cache._entries
    assert cache.grace("old") is None


def test_size_bound_evicts_oldest_first():
    cache, clock = make()
    for i, tok in enumerate(["t1", "t2", "t3", "t4"]):
        clock.now += 1
        cache.record(tok, i)
    assert "t1" not in cache._entries  # oldest evicted at max_size=3
    assert cache.grace("t2") is not None
    assert cache.grace("t4") is not None
