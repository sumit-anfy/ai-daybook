---
id: okf_20260708_093000_a11a
type: reference
title: Retrieval aliases — words the corpus never uses
summary: '- A thesaurus for search: everyday phrasings mapped to the words the corpus
  actually uses.

  - When a phrase appears in a query, its words run as a second search and the results
  are merged → § How it works

  - Each line names the phrase and the words it adds; nothing else is read → § Aliases'
status: active
maturity: candidate
canonical: false
created_at: '2026-07-08'
updated_at: '2026-07-08'
project: global-os
tags:
- retrieval
- search
- aliases
- okf
asks:
- why does search miss my wording
- how do I teach search a synonym
- search found nothing for a phrase I use
relates_to:
- okf_20260722_210000_c003
---

# Retrieval aliases

## How it works

`brief` reads the `RETRIEVAL_ALIASES` block below. When a query contains a
phrase, the listed words are searched as a second query and the two candidate
pools are merged, so nothing the original words found can drop out.

## Aliases

```yaml
RETRIEVAL_ALIASES:
  - phrase: throw away
    adds: [delete, deletion, purge]
  - phrase: deadline
    adds: [cut-off, cutoff]
  - phrase: shareholders
    adds: [cap table, shares, equity]
  - phrase: veg patch
    adds: [allotment, plot, beds]
```
