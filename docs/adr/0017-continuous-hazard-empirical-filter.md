# ADR-0017: continuous-hazard empirical marked filter

- **Status:** accepted
- **Date:** 2026-08-12
- **Stage:** P6C

## Context

P6M used real irregular event durations but approximated the point-process survival term by
freezing the hazard at the interval start. Its descriptive rollout likewise sampled an
exponential waiting time from the current hazard. This differs from equations (2.5)--(2.6)
of the paper because Brownian $X$ changes $G$ and every intensity while the book is silent.

## Decision

1. Preserve P6M as the frozen-hazard baseline; do not reinterpret its result.
2. Add `EMP-MARK-CT-001` with endpoint mark intensity and path-integrated survival.
3. Use vectorized compiled PyTorch CUDA kernels for fit, filtering and rollout.
4. Use nested four/eight-substep Brownian paths and a multiplicity-adjusted numerical
   equivalence audit before interpreting scientific gates.
5. Refit every rolling-origin fold. Applying a new clock to old P6M parameters is allowed
   only as an ablation, not as the P6C result.
6. Keep orders and P&L disabled unless P6C passes the unchanged usability and calibration
   families.

## Consequences and limitations

P6C tests a materially closer continuous-time likelihood and makes silence depend on the
whole latent path. Runtime and random-number volume increase. The training stage still
conditions on a causal proxy path rather than optimizing a joint latent marginal likelihood;
the held-out particle filter does integrate sampled Brownian paths. Therefore P6C is closer
to the paper's event clock but remains an empirical filtering extension, not an exact
implementation of an estimator supplied by the paper.
