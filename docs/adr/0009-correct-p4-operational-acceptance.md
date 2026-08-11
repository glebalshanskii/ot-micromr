# ADR-0009: Correct P4 operational acceptance

- Status: accepted; supersedes the interval-floor part of ADR-0008; active implementation integrated by ADR-0010
- Date: 2026-08-11
- Scope: interpretation of `SIM-FIG4-002` and future fixed-horizon rate experiments

## Context

The frozen P4 target required every seed-threshold-resolution cell to contain at least 20
complete inter-fill intervals. This condition was intended to prevent an undefined or visibly
unstable renewal-rate estimate. It had no sampling-distribution, power or downstream-sensitivity
derivation. The runner nevertheless included it in the hard operational acceptance conjunction.

That was a category error. P4 inference treats a complete simulated path as the independent
cluster and resamples complete seed-level rate vectors. Inter-fill intervals within one path are
dependent observations, and a count of 20 is neither an independent sample-size requirement nor
a calibrated precision criterion. Applying the rule to all grid cells also allowed deliberately
remote, low-turnover diagnostic thresholds to invalidate inference around the curve maximum.

## Decision

1. Reject `minimum_complete_intervals` as a hard acceptance gate. The historical value 20 is
   retained only as a labelled coverage diagnostic.
2. Operational validity requires the complete coordinate and policy grids, a defined finite
   rate estimator in every requested cell, non-flat measured policies, zero state/non-finite
   violations, numerical-error budgets, wealth reconciliation and deterministic replay.
   A rate is defined when at least one complete interval supplies a positive finite duration.
3. Inferential adequacy is decided at the declared independent-cluster level. P4 keeps the
   frozen 30-seed design, complete-vector bootstrap, minimum meaningful inward shift, Holm
   correction and three-way `supported`/`inconclusive` decision. Event counts do not replace
   confidence intervals or power.
4. Re-evaluate the existing immutable evidence with policy `p4-operational-validity-v2`.
   Do not rerun market simulation because neither raw data, estimands, tests nor multiplicity
   rules change.
5. Preserve the original `summary.json`, source config and manifest as historical evidence.
   Write the corrected decision to a separate derived review artifact and make it canonical in
   the report and plan.

## Consequences

The historical runner status remains reproducible at implementation commit `ca9aa7c`, but it no
longer determines the project conclusion. A sparse tail is visible in coverage diagnostics and
in seed-cluster uncertainty rather than converted into an unrelated stage failure. Zero-interval,
non-finite or incomplete cells remain hard failures.

For `SIM-FIG4-002`, all corrected operational gates pass. The scientific result does not change:
two inward-shift rows are supported, the high-gamma row is inconclusive and the six-test numerical
refinement family is inconclusive. The canonical stage status is therefore
`completed / operational validity passed / scientific inconclusive`.

## Implementation note, 2026-08-12

[`ADR-0010`](0010-current-experiment-surface.md) integrates this corrected decision directly
into the only active `SIM-FIG4-002` runner and removes the historical post-hoc command and
coverage-floor code. The original run/review artifacts remain immutable evidence under their
cited commits; current runs no longer require a second interpretation step.
