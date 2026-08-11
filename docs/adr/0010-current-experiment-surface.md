# ADR-0010: One current executable implementation per experiment

- Status: accepted
- Date: 2026-08-12
- Scope: active configs, simulator implementations, P4 acceptance and artifact schema

## Context

P2--P4 development left several executable generations in the default tree: failed P3
estimators, two Figure 4 pilots, a scalar policy engine, one-off sensitivity/benchmark tools
and a post-hoc acceptance reviewer. They were useful while decisions were being made, but they
made it unclear which implementation was authoritative and kept an invalid interval-count gate
reachable from current code.

The immutable Git history, ADRs and reports already preserve those results. Keeping their code
active is not required for scientific provenance.

## Decision

1. The executable surface contains exactly five configs: `ANA-SMOKE-001`, `ANA-FIG3-001`,
   `SIM-MOMENTS-002`, `SIM-UNBALANCED-002` and `SIM-FIG4-002`.
2. P3 uses only the integrated-hazard/jump-compensator estimators and powered statistical gates.
   Binned drift, step-quantile collection, fault injection and v1 summary branches are removed.
3. P4 uses only CPU `float64` adaptive market generation plus compiled CUDA `float32`
   crossing/policy evaluation. The scalar policy implementation and backend-selection branches
   are removed.
4. Correct operational acceptance is evaluated directly by `SIM-FIG4-002`. The invalid
   all-cell interval floor and the separate post-hoc reviewer are removed. A rate cell is valid
   when it has at least one complete positive-duration interval and finite outputs; inference
   remains at the independent seed-cluster level.
5. Remove the unused frozen-half-spread counterfactual, fake hard-coded diagnostic counters and
   historical benchmark/sensitivity scripts. Keep numerical-error bounds, accounting checks,
   replay and raw replication metrics because they validate the experiment actually executed.
6. Historical reports and ADRs remain unchanged evidence. Old executable code is recoverable
   from the commits cited there, but it is not supported by the current CLI.

## Consequences

The active code path and documentation now identify one implementation per experiment. Current
artifact names are not promised to be backward-compatible with old run directories; old reports
remain readable and cite their original commits. `SIM-FIG4-002` must be rerun from a clean commit
to produce a direct current-run acceptance result and confirm that removal of unreachable code
did not change the scientific conclusion.
