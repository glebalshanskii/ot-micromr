# Current experiment code review

- Date: 2026-08-12
- Branch: `refactor/current-experiments-only`
- Decision: [`ADR-0010`](../adr/0010-current-experiment-surface.md)

## Outcome

The repository now exposes one current implementation for each completed experiment. More than
four thousand lines of superseded code, configs and tests were removed. Historical scientific
evidence remains in Git, ADRs and reports; it is no longer mixed into the executable surface.

Removed components:

- P3 `SIM-*001` configs and v1 summary/statistical branches;
- Figure 4 legacy and pilot configs, scalar policy engine and pilot-selection branches;
- all-cell interval coverage gate and the post-hoc acceptance-review command;
- one-off P3 sensitivity and CPU/GPU benchmark scripts;
- unused fault-injection, vectorized-band and frozen-cost counterfactual implementations;
- step-quantile and binned-drift diagnostics that did not enter current decisions.

## Review findings fixed

- the stale `event_step_fraction` metric and empty event-log path were removed after review found
  that the old value depended on whether records were persisted rather than on the simulated path.
- P4 non-flat initialization is checked explicitly at measurement start instead of being
  mislabeled as an invariant violation or checked at the terminal state.
- Brownian probabilities below the configured cutoff are now actually omitted before sampling;
  their summed bound remains part of operational acceptance.
- P4 replication metrics no longer contain hard-coded zero violation counters.
- Current config validation accepts only the five active experiment IDs and rejects fields from
  removed implementations.

## Verification

The complete unit/integration suite passes, including the compiled CUDA accounting test. A clean
canonical `SIM-FIG4-002` rerun is required after the cleanup commit; its run ID, runtime and
scientific/operational decisions will be added here and to the main paper report.

## Remaining risks

- Figure 4 is still an independent partial reconstruction because the paper does not publish
  primitive parameters, code, seeds or raw Monte Carlo outputs.
- CPU market generation and CUDA crossing evaluation intentionally use different RNG families.
- The two-boundary crossing treatment remains an explicitly bounded approximation; its bound is
  enforced by the P4 numerical-error gate.
- Historical run artifacts use older schemas and must be interpreted with their cited commits.
