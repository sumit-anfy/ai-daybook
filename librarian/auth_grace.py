"""Grace cache for upstream identity verification.

The GitHub verifier treats ANY non-200 from api.github.com as "bad token", so a
transient GitHub 503 was indistinguishable from a revocation: the client saw
invalid_token and permanently dropped its grant. On 2026-08-17 GitHub's platform
outage did exactly this — every MCP call re-verifies the token against GitHub
(twice, cache off), so during their ~20% error window each call was a coin-flip
that killed the connection, and reconnects died within seconds all afternoon.

This cache remembers each token's last SUCCESSFUL verification so the server can
tell "cannot verify right now" apart from "verified and refused":

- fresh(): a success younger than positive_ttl is served without asking GitHub
  at all. Side effect: the happy path stops paying two GitHub round-trips per
  MCP call, which also shrinks the surface exposed to upstream failures.
- grace(): when verification FAILS, a success younger than grace_ttl is served
  instead of refusing — GitHub being down no longer means being locked out.

A token that never verified successfully has no entry, so an actually-invalid
token is still refused: the cache only ever bridges outages for identities it
has already seen proven. The accepted risk (2026-08-17): a genuinely
revoked token can outlive its revocation by up to grace_ttl — taken knowingly
on a single-user allowlisted server.
"""

import time

POSITIVE_TTL_SECONDS = 60.0
GRACE_TTL_SECONDS = 900.0
MAX_SIZE = 50


class GraceCache:
    def __init__(self, positive_ttl=POSITIVE_TTL_SECONDS, grace_ttl=GRACE_TTL_SECONDS,
                 max_size=MAX_SIZE, clock=time.monotonic):
        self.positive_ttl = positive_ttl
        self.grace_ttl = grace_ttl
        self.max_size = max_size
        self._clock = clock
        self._entries = {}  # token -> (result, verified_at)

    def record(self, token, result):
        """Store a verification that succeeded AND passed the allowlist."""
        self._entries[token] = (result, self._clock())
        self._prune()

    def fresh(self, token):
        """The result, if this token verified successfully within positive_ttl."""
        entry = self._entries.get(token)
        if entry is not None and self._clock() - entry[1] <= self.positive_ttl:
            return entry[0]
        return None

    def grace(self, token):
        """(result, age_seconds) if this token verified within grace_ttl, else None.

        Only consulted after a live verification has FAILED — the caller decides
        whether to bridge, and logs that it did.
        """
        entry = self._entries.get(token)
        if entry is not None:
            age = self._clock() - entry[1]
            if age <= self.grace_ttl:
                return entry[0], age
        return None

    def _prune(self):
        now = self._clock()
        for token in [t for t, (_, at) in self._entries.items()
                      if now - at > self.grace_ttl]:
            del self._entries[token]
        while len(self._entries) > self.max_size:
            oldest = min(self._entries, key=lambda t: self._entries[t][1])
            del self._entries[oldest]
