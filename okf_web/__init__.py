"""OKF web — a login-gated graph explorer + markdown editor for the corpus.

Runs on the VPS against the SAME corpus files the MCP server reads. Human edits
go through the same librarian.store (validation, repo lock, git commit+push), so
the web app and the okf MCP are two front-ends over one Librarian — no sync, no
second source of truth. This is the deferred "graph UI" from the build spec.
"""
