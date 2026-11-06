# Deploying OKF to a server

You do not need this for the quickstart: over stdio, Claude Desktop or Claude
Code start the server themselves. Deploy when you want the corpus reachable from
claude.ai, a phone, or several machines.

The files here are templates for one small Linux box (1 GB of RAM is enough):

| File | What it is |
|---|---|
| `okf.service` | systemd unit for the MCP server over HTTP |
| `okf-web.service` | systemd unit for the graph explorer and editor |
| `Caddyfile` | TLS reverse proxy: `/graph` → the web app, everything else → MCP |
| `okf-deploy.sh` + `.service` + `.timer` | every 2 minutes: check the corpus reached GitHub, then pull new code, run the test suite on it, and restart only if it is green |

The layout they assume: a user `okf`, the code at `/home/okf/OKF`, the corpus
(its own git repo) at `/home/okf/OKF-corpus`. Edit the paths in the units if
yours differ.

## 1. Prepare the box

```bash
sudo adduser --disabled-password okf
sudo apt update && sudo apt install -y git python3.11 python3.11-venv caddy
```

## 2. Clone and install

```bash
sudo -iu okf
git clone https://github.com/sumit-anfy/ai-daybook.git ~/OKF
cd ~/OKF && python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt   # includes pytest, which the deploy gate runs
```

The corpus is its own git repo. Clone yours if it already exists on GitHub:

```bash
git clone <your corpus repo> ~/OKF-corpus
```

Or start one from the schema notes (see "Your own corpus" in the main README),
and commit and push it before the services start. The Librarian refuses to write
while any note is uncommitted or untracked, and the units turn commit and push on:

```bash
mkdir ~/OKF-corpus && cp -r ~/OKF/examples/corpus/schemas ~/OKF-corpus/
cd ~/OKF-corpus && git init -b main && echo .okf.lock > .gitignore
# keep the global-os home, or move both schema notes to one of yours (main README)
git add .gitignore schemas && git -c user.name=okf-librarian -c user.email=librarian@okf.local commit -m "OKF schema"
git remote add origin <your corpus repo> && git push -u origin main
```

Then, for either route:

```bash
git -C ~/OKF-corpus config user.name "okf-librarian"
git -C ~/OKF-corpus config user.email "librarian@okf.local"
~/OKF/.venv/bin/python -m librarian.validate ~/OKF-corpus   # must be green
```

For the Librarian to push corpus writes, give the corpus clone a deploy key or a
fine-grained token scoped to that one repo (contents: read and write).

## 3. Secrets and services

Secrets go in env files outside git, readable only by `okf`:

```bash
echo "OKF_AUTH_TOKEN=$(openssl rand -hex 32)" | sudo tee /home/okf/okf.env
printf 'OKF_WEB_PASSWORD=<choose one>\nOKF_WEB_SECRET=%s\n' "$(openssl rand -hex 32)" \
  | sudo tee /home/okf/okf-web.env
sudo chown okf:okf /home/okf/okf*.env && sudo chmod 600 /home/okf/okf*.env
sudo cp ~okf/OKF/deploy/okf.service ~okf/OKF/deploy/okf-web.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now okf okf-web
```

`okf-web` refuses to start without `OKF_WEB_SECRET`, and the MCP server refuses
HTTP without `OKF_AUTH_TOKEN` or GitHub OAuth. For claude.ai custom connectors,
which need OAuth, create a GitHub OAuth App and add `OKF_GITHUB_CLIENT_ID`,
`OKF_GITHUB_CLIENT_SECRET`, `OKF_PUBLIC_URL` and `OKF_GITHUB_ALLOWED_USERS` (your
GitHub login) to `okf.env`. Only the listed logins get in.

## 4. TLS

Point a DNS record at the box, copy `deploy/Caddyfile` to `/etc/caddy/Caddyfile`,
set `OKF_DOMAIN=okf.example.com` in Caddy's environment (for example with
`sudo systemctl edit caddy` → `Environment=OKF_DOMAIN=…`), and reload Caddy.
The MCP endpoint is then `https://okf.example.com/mcp`, and the graph explorer
is at `https://okf.example.com/graph`.

## 5. Smoke test

```bash
claude mcp add okf --transport http https://okf.example.com/mcp \
  --header "Authorization: Bearer <token>"
```

In a session, call `validate_corpus` (expect `ok: true`), then create a
throwaway note and confirm it reaches the corpus repo on GitHub.

## 6. Auto-deploy (optional)

```bash
sudo cp ~okf/OKF/deploy/okf-deploy.service ~okf/OKF/deploy/okf-deploy.timer /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now okf-deploy.timer
```

Every two minutes it checks that the Librarian's corpus commits reached GitHub,
then fetches new code. A new revision is checked out into a scratch worktree and
the suite runs there, against the bundled example corpus; only a green run
fast-forwards and restarts the services, and a post-restart probe records
whether they came back. Both results surface in `audit_corpus`.

It follows `main` in both repos. If either uses another branch, set
`OKF_CODE_BRANCH` / `OKF_CORPUS_BRANCH` with `sudo systemctl edit okf-deploy`
(`[Service]` then `Environment=OKF_CORPUS_BRANCH=master`).

## Rules that save a bad afternoon

- Run every git command against the repos as `okf`, never as root:
  `sudo -u okf git -C /home/okf/OKF pull --ff-only`. Root-owned files in the
  tree block the Librarian's next write.
- The server is the only unattended writer of the corpus. If you also edit by
  hand elsewhere, pull first; a rejected push makes the Librarian rebase and
  retry once, and log an error if that fails.
