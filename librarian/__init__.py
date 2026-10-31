"""The Librarian — deterministic core of the OKF system (build-spec §6.1).

No LLM lives here. It assigns ids, validates frontmatter against the schema,
checks link integrity, and (later) regenerates indexes and commits to git.
"""
