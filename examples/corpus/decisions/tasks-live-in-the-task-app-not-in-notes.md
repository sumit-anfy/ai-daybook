---
id: okf_20260706_113000_d005
type: decision
title: Tasks live in the task app, not in notes
summary: '- The knowledge base holds knowledge; tasks and other live data stay in
  their own apps.

  - A task is live data: it changes daily, so a copy in a note goes stale → § Why

  - The corpus holds a resource pointer to the task project, never the tasks themselves
  → § How'
status: active
maturity: established
canonical: true
created_at: '2026-07-06'
updated_at: '2026-07-06'
project: global-os
tags:
- tasks
- live-data
- architecture
- okf
asks:
- can I keep my to-do list in the notes
- where do tasks go
- why not copy tasks into the knowledge base
- what is a resource pointer for
relates_to:
- okf_20260706_100000_b001
- okf_20260710_090000_f001
---

# Tasks live in the task app, not in notes

## Why

A task changes daily. A copy in a note is wrong by the next morning, and a wrong
note is worse than no note.

## How

The corpus holds a `resource` node pointing at the task project. Sessions list
new tasks in their close-out; they are added to the task app, not to a note.
