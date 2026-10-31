"""Where settings come from: the process environment, then a `.env` file.

Everything a deployment varies lives in the environment, never in the code:
the corpus path (OKF_ROOT), the public URL (OKF_PUBLIC_URL), the GitHub logins
allowed through OAuth (OKF_GITHUB_ALLOWED_USERS), the auth secrets. The project
homes are not here either: they are the PROJECT_REGISTRY block of the corpus's
own `schemas/type-registry.md`, so a corpus carries its vocabulary with it.
`.env.example` lists every variable.

`load_env()` reads `KEY=value` lines from `.env` beside the code. A variable
already set in the environment wins, so a systemd unit or an MCP client config
overrides the file. No dependency: the format is deliberately the plain subset
(no interpolation, no multi-line values).
"""
from __future__ import annotations

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CORPUS = CODE_ROOT / "examples" / "corpus"


def load_env(path: Path | None = None) -> None:
    path = path or CODE_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value


def corpus_root() -> Path:
    """OKF_ROOT, else the bundled example corpus — so a fresh clone runs as is."""
    env = os.environ.get("OKF_ROOT", "").strip()
    return Path(env).expanduser().resolve() if env else EXAMPLE_CORPUS
