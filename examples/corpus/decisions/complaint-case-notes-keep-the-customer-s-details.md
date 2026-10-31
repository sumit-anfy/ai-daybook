---
id: okf_20260815_200000_c0de
type: decision
title: Complaint case notes keep the customer's details
summary: '- A ratified risk: case notes about customer complaints keep names and details,
  under controls argued for a small library.

  - Case notes keep full detail so the lesson stays usable → § The acceptance

  - The controls were argued for a library of two cases; at five it is re-argued →
  § Scope'
status: active
maturity: established
canonical: false
created_at: '2026-08-15'
updated_at: '2026-08-15'
project: bakery
tags:
- complaints
- privacy
- risk
- cases
asks:
- can complaint notes include the customer's name
- how many complaint cases before we review privacy
relates_to:
- okf_20260816_093000_ca5e
- okf_20260822_161500_ca5f
---

# Complaint case notes keep the customer's details

## The acceptance

A complaint case note keeps the customer's name and the details of what went
wrong, because the lesson is in the specifics. The corpus is private and behind
a login.

## Scope

The validator counts the active notes titled `Case — …` and warns when the count
reaches `re_ratify_at`: the controls were argued for this many cases, and more
cases need the argument made again.

```yaml
RATIFIED_SCOPE:
  cases: 2
  ratified_on: 2026-08-15
  re_ratify_at: 5
```
