---
id: okf_20260723_090000_8001
type: pattern
title: A check that cannot verify must fail loudly
summary: '- A guardrail for every automated check: if it cannot do its job, it reports
  failure, never a pass.

  - A test that finds no data to test must fail, not pass empty → § The rule

  - Skips are acceptable only when the reason is printed → § The rule'
status: active
maturity: established
canonical: false
created_at: '2026-07-23'
updated_at: '2026-07-23'
project: global-os
tags:
- testing
- guardrail
- okf
asks:
- why did the tests fail when nothing was there to test
- should a check pass when it has nothing to check
- is a silent skip ok
relates_to:
- okf_20260706_100000_b001
---

# A check that cannot verify must fail loudly

## The rule

A check that finds nothing to check has not passed. An empty corpus, a missing
file, an unreachable service: each is a failure with a reason, or a skip that
prints why. A green result must mean something was verified.
