---
id: okf_20260722_210000_c003
type: note
title: How search works in this knowledge base
summary: '- How brief finds notes: words and meaning together, all on this machine,
  with no AI model judging.

  - Lexical scoring weights tags and titles, then summary, then body → § Two signals

  - A small local embedding model adds matches by meaning, never removes a word match
  → § Two signals

  - Write asks in a stranger''s words so a note can be found by people who do not
  know its title → § Asks'
status: active
maturity: captured
canonical: false
created_at: '2026-07-22'
updated_at: '2026-07-22'
project: global-os
tags:
- retrieval
- search
- embeddings
- okf
asks:
- how does search find my notes
- does search use AI
- why add asks to a note
- does my text leave the computer when I search
relates_to:
- okf_20260708_093000_a11a
- okf_20260706_100000_b001
---

# How search works in this knowledge base

## Two signals

1. **Words.** Each query word scores by how rare it is, and more in tags and
   titles than in the body.
2. **Meaning.** A small embedding model (bge-small) runs locally and adds notes
   that mean the same thing in other words. It only adds or reorders.

## Asks

`asks` are the questions a stranger would type when this note is the answer.
They are searched like the summary, so write them in the asker's words.
