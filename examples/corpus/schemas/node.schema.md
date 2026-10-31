---
id: okf_20260706_090000_0f01
type: reference
title: OKF Node Schema
summary: '- The frontmatter contract, state machine and Librarian-enforced rules every
  node satisfies; the machine blocks win over the prose.

  - status (active, parked, superseded, retired) and maturity (captured, candidate,
  established) are independent axes → § 1. Two axes and one flag

  - canonical marks the one trusted node per topic, and only an active, established
  node may carry it → § 1. Two axes and one flag

  - A node is three layers: metadata, a bullet summary of its current state, and a
  body that only grows → § 2. Three layers

  - type, title and project come from the writer; the Librarian assigns id and dates
  and rejects anything outside the registry → § 3. Machine block — SPEC

  - Status moves only along TRANSITIONS; maturity only moves forward → § 4. Transitions

  - Superseding a node marks it superseded and links both ways; nothing is deleted
  to replace it → § 5. Rules'
status: active
maturity: established
canonical: true
created_at: '2026-07-06'
updated_at: '2026-07-06'
project: global-os
tags:
- schema
- frontmatter
- librarian
- foundational
asks:
- which fields does a note need
- what does canonical mean on a note
- how do statuses change over time
- why was my note refused by the librarian
- what is the difference between status and maturity
- how should a summary be written
- how does replacing an old note work
- what frontmatter keys are allowed
source: the OKF build spec
relates_to:
- okf_20260706_090500_0f02
allow_dangling_refs: true
---

# OKF Node Schema

Every node is a markdown file with YAML frontmatter. This node defines what that
frontmatter may hold. The fenced `yaml` blocks below are read by the code and by
`tests/test_store.py::test_schema_code_drift`, which fails when they and the
constants in `librarian/` disagree. Where this prose and a block differ, the
block wins.

## 1. Two axes and one flag

- **status** — is this node still in force? `active`, `parked`, `superseded`,
  `retired`.
- **maturity** — how much has it been tested? `captured` (written down),
  `candidate` (seen to hold more than once), `established` (relied on).
- **canonical** — the one node to trust on its topic. Only an `active`,
  `established` node may be canonical.

The axes are independent: an established decision can be retired, and a captured
idea can be active.

## 2. Three layers

1. **Metadata** — title, tags, `asks` (the questions a stranger would type when
   this node is the answer).
2. **Summary** — the node's *current state*, as up to eight `- ` bullets. Bullet 1
   says what the node is. Every other bullet states one point and ends with
   `→ § <heading>`, the body section that holds its detail. A point that moved
   says so: `(changed 2026-09-02: was 2pm)`.
3. **Body** — the long form. It only grows: corrections are appended as dated
   blocks, never written over the old text.

The summary is rewritten as things change; the body keeps the history.

## 3. Machine block — SPEC

Ids look like `okf_<YYYYMMDD>_<HHMMSS>_<4 hex>` and are assigned by the Librarian.

```yaml
SPEC:
  required_from_human: [type, title, project]
  required_in_valid_node: [id, type, title, status, maturity, canonical, created_at, updated_at, project]
  fields:
    id:            {type: string, pattern: "^okf_\\d{8}_\\d{6}_[0-9a-f]{4}$", assigned_by: librarian}
    type:          {type: enum, source: type-registry, defaulted: false}
    title:         {type: string, defaulted: false}
    summary:       {type: string, optional: true}
    status:        {type: enum, values: [active, parked, superseded, retired], default: active}
    maturity:      {type: enum, values: [captured, candidate, established], default: captured, monotonic: true}
    canonical:     {type: bool, default: false}
    created_at:    {type: date, assigned_by: librarian, immutable: true}
    updated_at:    {type: date, assigned_by: librarian}
    project:       {type: enum, source: type-registry, defaulted: false}
    tags:          {type: list, optional: true}
    asks:          {type: list, of: string, max: 8, max_len: 160, optional: true}
    source:        {type: string, optional: true}
    confidence:    {type: number, optional: true}
    relates_to:    {type: list, of: id, optional: true}
    supersedes:    {type: id, optional: true}
    superseded_by: {type: id, assigned_by: librarian, optional: true}
    work_state:    {type: enum, values: [open, shipped, closed],
                    applies_to: "type == log", optional: true}
    superseded_in_part: {type: date, optional: true}
    resource:      {type: object, required_when: "type == resource",
                    fields: {kind: {enum: [todoist_project, todoist_task, supabase_table, calendar, api, file, gdrive_file, gdrive_folder]}, uri: {type: string}}}
    unfiled_rulings_legacy: {type: boolean, applies_to: "type in [session, note]", optional: true}
    allow_dangling_refs: {type: boolean, optional: true}
```

## 4. Transitions

```yaml
TRANSITIONS:
  status:
    _birth_:    [active, parked]
    active:     [parked, superseded, retired]
    parked:     [active, superseded, retired]
    superseded: [active]     # only by removing the successor's supersedes pointer
    retired:    []           # terminal
  maturity:                  # forward-only; birth may start anywhere
    _birth_:   [captured, candidate, established]
    captured:  [candidate, established]
    candidate: [established]
    established: []
```

## 5. Rules

```yaml
EDGE_SPEC:
  edge_types: [supports, contradicts, depends_on, inspired_by, relevant_to, actioned_by]
```

```yaml
RULES:
  - reject if any of required_from_human is missing
  - reject a type or project outside the type registry
  - assign id if absent; reject if id is malformed or collides
  - default status=active, maturity=captured, canonical=false when absent
  - set created_at on birth (immutable thereafter); rewrite updated_at on every mutation
  - reject any status transition not in TRANSITIONS.status
  - reject any maturity transition that is not forward
  - on `supersedes: X`: set X to status=superseded, canonical=false, superseded_by=<this.id>
  - canonical=true only when status=active and maturity=established
  - resource nodes hold only resource.kind and resource.uri — a pointer, never a copy
```
