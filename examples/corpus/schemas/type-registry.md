---
id: okf_20260706_090500_0f02
type: reference
title: OKF Type Registry
summary: '- The controlled vocabulary: the ten node types, the folder each files under,
  and the project homes.

  - Ten types; note is the default and the validator rejects any type not listed →
  § 1. Node types

  - Each type files under its own folder, playbooks under /skills → § 2. Folder map

  - Four homes: personal, bakery and garden are life areas, global-os holds the tools
  that serve them all → § 3. Project homes

  - A node has exactly one home; anything cross-cutting is a relates_to link, never
  a second home → § 3. Project homes'
status: active
maturity: established
canonical: true
created_at: '2026-07-06'
updated_at: '2026-07-06'
project: global-os
tags:
- schema
- taxonomy
- vocabulary
- librarian
- foundational
asks:
- what kinds of note can I create
- which folder does a playbook go in
- what are the allowed project homes
- can I invent a new note type
- where do cross-cutting tools belong
- my note was refused for an unknown type
- difference between a decision and a pattern
- how do I add a new home
source: the OKF build spec
relates_to:
- okf_20260706_090000_0f01
---

# OKF Type Registry

`node.schema` declares `type: {source: type-registry}`; this node is that source.
The Librarian reads the three blocks below live from the corpus, so changing the
vocabulary is an edit to this node — no code.

Grow on friction: add a type or a home only when a real node does not fit one
that exists. If choosing a type takes more than ten seconds, the taxonomy is too
fine.

## 1. Node types

```yaml
REGISTRY:
  node_types:
    note:      "a unit of knowledge or thinking you produced. THE DEFAULT when nothing more specific fits."
    project:   "an ongoing body of work with a goal and a lifespan."
    decision:  "a choice that was made, with alternatives and rationale."
    reference: "external or definitional knowledge you are citing rather than authoring."
    resource:  "a POINTER to live data in another system. Carries a resource: block. Never holds a copied value."
    session:   "the structured residue of one working session."
    playbook:  "a codified, runnable procedure."
    pattern:   "a reusable principle or recurring shape that is not itself a procedure."
    log:       "an append-only register, or the record of one bounded piece of work."
    record:    "a DATED, IMMUTABLE snapshot of authoritative data. Superseded by a new snapshot, never edited."
```

## 2. Folder map

```yaml
FOLDER_MAP:
  note:      /knowledge
  reference: /references     # the schema nodes live in /schemas
  decision:  /decisions
  project:   /projects
  resource:  /resources
  session:   /sessions
  playbook:  /skills
  pattern:   /patterns
  log:       /logs
  record:    /records
```

## 3. Project homes

`project` is a node's home: whose world it belongs to. `type` says what a node
is; `project` says where it lives. Topics are tags; worlds are projects.

```yaml
PROJECT_REGISTRY:
  projects:
    personal:  "Rowan's own life: health, reading, money that is not the bakery's."
    bakery:    "Juniper Street Bakery: the shop, wholesale, recipes, suppliers, staff."
    garden:    "Allotment plot 14: beds, sowing, soil, pests."
    global-os: "The capability home: this knowledge base and the habits that serve every life area."
```
