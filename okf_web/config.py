"""Startup configuration for okf-web that must be checkable WITHOUT importing
the app — importing `okf_web.server` validates a corpus and starts the
background pusher, so anything that needs a unit test lives here instead.
"""
import os

# The only value the cookie signer ever accepted when OKF_WEB_SECRET was unset.
# It is in a public repo, so any cookie signed with it is forgeable by anyone
# who can read GitHub. Kept as an EXPLICIT dev escape only.
_DEV_SECRET = "dev-only-secret"


class ConfigError(RuntimeError):
    pass


def web_secret(env: dict | None = None) -> bytes:
    """The cookie-signing secret, or refuse to start.

    Fail CLOSED (2026-09-07, from the external review): before this, an unset
    OKF_WEB_SECRET silently fell back to the string above, and a cookie forged
    from it was accepted with a 200 — the login was a formality on any box where
    the env file lost the line. A missing secret is now a startup failure with
    the fix in the message; systemd's Restart=on-failure turns that into a
    crash-loop the deploy probe reports, which is the loud outcome we want.

    The dev fallback survives only behind OKF_WEB_DEV=1, so a laptop can run the
    UI with no secret but a production unit can never drift into it by omission.
    """
    env = os.environ if env is None else env
    secret = (env.get("OKF_WEB_SECRET") or "").strip()
    if secret:
        return secret.encode()
    if env.get("OKF_WEB_DEV") == "1":
        return _DEV_SECRET.encode()
    raise ConfigError(
        "okf-web refuses to start: OKF_WEB_SECRET is not set. Add "
        "OKF_WEB_SECRET=$(openssl rand -hex 32) to /home/okf/okf-web.env "
        "(or export OKF_WEB_DEV=1 for a local, insecure dev run).")
