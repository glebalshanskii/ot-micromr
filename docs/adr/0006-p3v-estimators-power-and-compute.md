# ADR-0006: P3V estimators, power design and compute backend

- Status: accepted
- Date: 2026-08-11
- Scope: P3V only; historical `SIM-*001` contracts and artifacts are immutable

## Context

The P3 run validated pathwise invariants and most moment claims, but rare open parity made
the conditional-flow ratio and realised open drift too noisy. Increasing an arbitrary point
threshold or blindly appending seeds would not answer whether the remaining uncertainty can
change the Figure 4 decision.

## Decision

1. Preserve legacy metrics and add integrated hazards, transition counts and compensators.
2. Use exact generator drift as an algebraic gate and realised finite-h drift as a stochastic
   sampler gate; neither is allowed to impersonate the other.
3. Use a joint Holm-adjusted primary family and equivalence/superiority tests defined in the
   P3V protocol. Point estimates alone cannot pass.
4. Use new seed labels, a fixed `20000`-reversion-time horizon and no optional extension.
5. Process-parallelise independent paths after a worker-count benchmark. Keep post-processing
   vectorised and cap nested numerical-library threads at one per worker.
6. Do not add PyTorch merely to port a sequential branch-heavy event loop. A GPU backend
   requires an end-to-end `torch.compile` benchmark advantage and identical scientific
   semantics.
7. After the preregistered pilot sensitivity, reduce the flow margin to `0.05`, keep the
   open-drift theorem as an exact-generator gate, and use a jump-compensator stochastic
   negative control. This dated change precedes all `SIM-*002` target runs.

## Consequences

The confirmatory run is more expensive but has a bounded compute budget and an explicit
three-way decision. Header-only event logs replace large raw logs for `SIM-*002`; sufficient
seed-level audit quantities and replay digests remain. P4 stays blocked until the global P3V
family and downstream sensitivity condition both pass.
