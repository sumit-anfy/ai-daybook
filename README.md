# AI Daybook

An AI memory engine inspired by Google's Open Knowledge Format.

A personal knowledge base that an AI assistant reads and writes through one MCP
server. The notes are plain markdown files with YAML frontmatter, in a git repo
you own. Between the assistant and those files sits the **Librarian**: ordinary,
deterministic Python with no LLM in it. It assigns ids, files notes, validates
every write against a schema, enforces each note's lifecycle, and searches.

The assistant does the judgment: what to write, what a new note makes false,
how to rerank what search returns. The Librarian does the bookkeeping and refuses
bad writes. Search runs on your machine: word scoring plus a small local
embedding model (`bge-small`), so your text never leaves the box.

Built by Sumit for his own notes, and run daily against several hundred of them.
This repo has the code and a fictional example corpus; no real notes are included.

The file format is inspired by Google's
[Open Knowledge Format](https://github.com/GoogleCloudPlatform/open-knowledge-format),
also "OKF". This project was first named after it, which is why the code still says
`OKF` and `okf_`. It is a separate project, not affiliated with Google. See [Credits](#credits).

**Status:** shared as a reference implementation, as is. I can't offer support, so
issues and pull requests are off. Ideas are welcome in Discussions, though I may not
reply to all of them. MIT-licensed, so fork it and make it yours.

## The note design: three layers

Every note (a *node*) has three layers, read in this order:

1. **Metadata.** Title, type, project, tags, and `asks`: up to eight questions a
   stranger would type when this note is the answer. Search indexes `asks` like
   the summary, so a note can be found by someone who has forgotten its words.
2. **Summary: the current state.** Up to eight `- ` bullets. Bullet 1 says what
   the note is. Each other bullet states one point the note *currently* makes and
   ends with `→ § <heading>`, the section of the body that holds the detail. When
   a point changes, its bullet says so: `(changed 2026-09-02: was 2pm)`. The summary
   is rewritten whenever the note changes.
3. **Body: the long form.** It is never overwritten. Corrections are appended as
   dated blocks, and a body replacement that would drop existing text is refused
   unless the caller passes an explicit flag. The body is the history; the summary
   is the answer of record.

```markdown
---
id: okf_20260902_064500_d002
type: decision
title: Wholesale orders close at noon the day before
summary: |-
  - The current wholesale cut-off: cafés order by noon for the next morning's delivery.
  - Noon gives the evening shift the full mix before it starts → § Why
  - Replaces the 2pm cut-off (changed 2026-09-02: was 2pm) → § Why
status: active
maturity: established
canonical: true
project: bakery
asks:
- what time do cafés have to order by
- can a café order after lunch
supersedes: okf_20260714_070000_d001
---
# Wholesale orders close at noon the day before
## Why
…
```

Replacing a note does not delete it. The new note `supersedes` the old one; the
Librarian marks the old note `superseded` and links both ways. Search then leaves
superseded notes out and tells you history exists.

## Quickstart (5 minutes, over stdio)

You need Python 3.11 or 3.12, and git.

```bash
git clone https://github.com/sumit-anfy/ai-daybook.git
cd ai-daybook
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m librarian.validate examples/corpus     # OK — 0 error(s)
```

The server runs against the bundled example corpus by default: 37 notes from a
fictional baker, Rowan, covering every note type. To keep the repo clean while
you experiment, work on a copy:

```bash
cp -r examples/corpus ~/okf-demo
```

**Claude Code:**

```bash
claude mcp add --transport stdio --env OKF_ROOT=$HOME/okf-demo okf -- "$PWD/.venv/bin/python" "$PWD/server.py"
```

**Claude Desktop:** add this to `claude_desktop_config.json` (Settings →
Developer → Edit Config), with absolute paths, then restart Claude Desktop:

```json
{
  "mcpServers": {
    "okf": {
      "command": "/absolute/path/to/ai-daybook/.venv/bin/python",
      "args": ["/absolute/path/to/ai-daybook/server.py"],
      "env": { "OKF_ROOT": "/Users/you/okf-demo" }
    }
  }
}
```

Then ask Claude things like:

- *"Brief me: what time do the cafés have to order by, and why did it change?"*
- *"Who owns the bakery?"*
- *"Create a note in garden: the beans failed because they went in too early."*
- *"Run the corpus audit."*

The first search downloads the embedding model (about 64 MB) into a local cache.
Until it is ready, and wherever `fastembed` is not installed, search falls back
to words only and says so (`semantic: false`). `OKF_SEMANTIC=0` chooses that
mode on purpose.

### Your own corpus

Start from the two schema notes, in a git repo of its own:

```bash
OKF_CODE=$PWD                   # run from the ai-daybook folder
mkdir ~/my-corpus && cp -r examples/corpus/schemas ~/my-corpus/
cd ~/my-corpus
git init -b main
git config user.name "Your Name"
git config user.email "you@example.com"
echo .okf.lock > .gitignore     # the Librarian's write lock
```

Edit `PROJECT_REGISTRY` in `schemas/type-registry.md` to list your own homes.
The two schema notes are themselves filed under `global-os`, so either keep that
home or change the `project:` line in both to one of yours. Then commit, and
point `OKF_ROOT` at the folder:

```bash
"$OKF_CODE/.venv/bin/python" "$OKF_CODE/librarian/validate.py" .    # must be green
git add .gitignore schemas && git commit -m "OKF schema"
```

Commit before you turn on `OKF_GIT_COMMIT=1`: the Librarian refuses to write
while any `.md` file in the corpus is uncommitted or untracked, so it never
commits your half-finished edits along with its own. To push every write too,
add a remote and push once (`git remote add origin <url> && git push -u origin
main`) before setting `OKF_GIT_PUSH=1`. All settings are in
[`.env.example`](.env.example).

If your homes do not include `global-os`, set `OKF_FINDABILITY_PROJECT` to the
home the weekly findability log should be filed under. That run (HTTP mode only)
measures a bank of held-out questions that name note ids, and the bundled bank
names the example corpus's. On your own corpus it reports that nothing was
measured until you write a bank and set `OKF_FINDABILITY_BANK`; the format is in
`tests/findability_seed.yaml`.

### Tests

About 30 seconds. The write tests run on throwaway copies of the example corpus.
Run them both ways: with the embedding model, and words only.

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
OKF_SEMANTIC=0 .venv/bin/python -m pytest
```

CI runs both on every push.

## The tools

| Tool | What it does |
|---|---|
| `brief` | Search. Returns the best-matching notes with bodies, their neighbours, and what was left out as stale |
| `read_node` / `list_nodes` | Read one note by id; list by type and status |
| `create_node` / `update_node` | The write path: validated, filed, state machine enforced |
| `validate_corpus` | Check every note against the schema |
| `audit_corpus` | Weekly triage: orphans, stale links, near-duplicates, suggested links |
| `resolve_suggestion` | Accept or dismiss a suggested link |
| `propose_deletion` / `delete_node` | Delete only retired, unlinked notes; hand anything else to a human |

A session note must end with a close-out (Keep / Conflicts / Your call /
Tasks), and every ruling it lists must cite the note it is filed in. The server
refuses a session note that skips either.

## Architecture

```
Claude (Desktop, Code, claude.ai)
   │  MCP: stdio, or HTTP behind TLS with a bearer token or GitHub OAuth
server.py           thin FastMCP wrapper: ten tools, auth, no logic
   │
librarian/          the deterministic core (no LLM, no network)
   validate.py      schema + registry, read live from the corpus's own schema notes
   store.py         the write path: lock, validate as a set, atomic replace,
                    state machine, supersession, body guard, git commit/push
   context.py       brief(): hybrid ranking, intent-aware type priors, aliases
   embed.py         local bge-small embeddings (fastembed, ONNX), cached in ~/.okf
   sections.py      long notes searched section by section
   summary.py       the bullet-summary format and its checks
   cluster.py       after a write: "which neighbours does this make false?"
   redundancy.py    after a rule is written: "does a canonical note already say this?"
   completeness.py  session rulings must cite the note they are filed in
   audit.py         weekly triage, link suggestions, purge candidates
   findability.py   weekly run of held-out stranger questions, logged as a note
   │
your corpus         markdown + YAML frontmatter, its own git repo
   schemas/         node.schema.md and type-registry.md are themselves notes
okf_web/            Starlette graph explorer and markdown editor, password login
deploy/             systemd units, Caddy, a pull-and-test auto-deploy timer
```

Some design choices:

- **The vocabulary lives in the corpus.** Note types, folders and project homes
  are YAML blocks inside `schemas/type-registry.md`. The validator reads them
  live; changing your homes is an edit to a note, not to the code.
- **Fail closed.** No registry means no writes. HTTP without auth refuses to
  start. A check that cannot run reports a failure, not a pass.
- **Search only adds.** The embedding layer and the aliases can reorder results
  or add matches, but never remove a word match. Every score is broken down in
  the output (`score_lexical`, `score_semantic`, `prior`), so you can see why a
  note ranked where it did.
- **Live data stays in its own app.** Tasks, spreadsheets and calendars are
  `resource` notes that point at them, never copies that go stale.

## Retrieval numbers (example corpus only)

`python tests/retrieval_report.py` prints these. They are measured on the
37-note example corpus with the benchmark in `tests/retrieval_benchmark.yaml`
(48 questions) and the held-out bank in `tests/findability_seed.yaml` (20 notes,
two questions each, written without reusing the note's own words).

| | Words + embeddings | Words only |
|---|---|---|
| Benchmark top-1 | 46/48 | 43/48 |
| Benchmark top-5 | 47/48 | 48/48 |
| Held-out questions, top-1 | 15/20 | 5/20 |
| Held-out questions, in the default pack of 8 | 20/20 | 14/20 |
| Held-out short phrases, in the default pack of 8 | 20/20 | 16/20 |

Read these as a smoke test, not a quality claim: the same person wrote the corpus
and the questions, and 37 notes is an easy haystack. The useful signal is the
gap between the two columns on held-out wording. That gap is what the embedding
layer is for.

## What is not built

- **No LLM inside.** The server never calls a model. Summaries, `asks`, and the
  "what does this make false?" review are written by the assistant that calls
  the tools.
- **No importers.** Nothing reads Notion, Obsidian or Apple Notes.
- **Single owner.** One corpus, one owner. Auth decides who gets in, with no
  per-note permissions, no sharing, and no multi-user editing.
- **No packaging.** Run from a checkout; there is no `pip install okf` and no
  Docker image.
- **No encryption at rest.** Notes are plain files. Protect the disk and the repo.
- **Edges are only links.** `relates_to` links are plain lists. The typed-edge
  vocabulary in the schema (supports, contradicts, …) is not used yet.
- **Tested on macOS and Linux only.** It has not been run on Windows.
- **Scheduled work is external.** The weekly audit is meant to be run by a
  scheduled assistant task calling `audit_corpus`; this repo ships no scheduler
  beyond the findability run inside the HTTP service.

## Credits

- **[Google's Open Knowledge Format](https://github.com/GoogleCloudPlatform/open-knowledge-format)**,
  this project's inspiration, and where the code's `OKF` name comes from. Knowledge as cross-linked markdown files with YAML
  frontmatter, versioned in git, comes from there. Google's OKF is a format, and it
  deliberately leaves storage, serving and search open; this project is one answer
  to those for a single owner. It has also drifted from the spec: notes use their
  own status values and link by id rather than by markdown link, and they have not
  been tested against an OKF v0.2 consumer.
- **[Andrej Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)**:
  an LLM maintaining a persistent wiki under a schema, rather than retrieving and
  forgetting. This design was reached separately and later found to match it closely.
  The difference is that here the schema is enforced by code that refuses bad writes.
- **Tiago Forte's *Building a Second Brain***, where the idea of a second brain
  started, and the capture-everything inbox this project chose not to build.
- Built on [FastMCP](https://github.com/jlowin/fastmcp) and
  [fastembed](https://github.com/qdrant/fastembed), with BAAI's `bge-small-en-v1.5`
  embedding model.

## Licence

[MIT](LICENSE).
