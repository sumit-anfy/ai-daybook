---
id: okf_20260707_101500_7a10
type: reference
title: OKF MCP tool registry — every tool and the justification that earns it
summary: '- The list of MCP tools the server exposes, each with the reason it earns
  a slot.

  - Ten tools; a new one must argue what it adds against the cost of a longer tool
  list → § Why a registry

  - Reads: validate_corpus, read_node, list_nodes, brief, audit_corpus → § Tools

  - Writes: create_node, update_node, resolve_suggestion, propose_deletion, delete_node
  → § Tools'
status: active
maturity: established
canonical: false
created_at: '2026-07-07'
updated_at: '2026-07-07'
project: global-os
tags:
- mcp
- tools
- librarian
- okf
asks:
- which tools does the knowledge server expose
- why is there no bulk edit tool
- how many MCP tools are too many
- what does each server tool do
relates_to:
- okf_20260706_090000_0f01
- okf_20260706_100000_b001
---

# OKF MCP tool registry

## Why a registry

Every tool an agent can see makes every other tool slightly harder to choose. A
new tool earns its slot by naming what it does that the others cannot. The test
suite checks that this list and the `@mcp.tool` functions in `server.py` match.

## Tools

```yaml
TOOL_REGISTRY:
  count_threshold: 15
  tools:
    validate_corpus:
      justification: The read half of the Librarian; every other guarantee rests on it.
    read_node:
      justification: Fetch one node by id, so an agent can follow a citation.
    list_nodes:
      justification: Retrieval by frontmatter without reading every file.
    brief:
      justification: Answers "what do we know about X" in one call instead of N reads.
    audit_corpus:
      justification: Weekly triage as data — orphans, stale links, suggested links.
    create_node:
      justification: The write path; id, filing, validation and supersession live here.
    update_node:
      justification: Mutation with the state machine and the body guard enforced.
    resolve_suggestion:
      justification: The only way to answer a link suggestion from the audit.
    propose_deletion:
      justification: Hands a node the deletion gates refuse to a human.
    delete_node:
      justification: Removes junk that is retired, unlinked and outside any supersession chain.
```
