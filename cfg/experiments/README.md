# Experiment configuration contract

Executable experiment inputs use strict TOML `RunSpec v1`, defined by
[`ADR-0001`](../../docs/adr/0001-research-modes-package-and-run-contract.md).
Every result-affecting value must be present in the source config; runtime provenance is
written separately to `manifest.json` and never mutates the preregistered TOML.

## P1 configs

| Experiment | Config | Purpose | Run status |
|---|---|---|---|
| `ANA-SMOKE-001` | `ana_smoke_001.toml` | One deterministic Dawson-root contract | passed in P2 |
| `ANA-FIG3-001` | `ana_fig3_001.toml` | Deterministic Figure 3 reconstruction | reproduced in P2 |
| `SIM-MOMENTS-001` | `sim_moments_001.toml` | Jump-model invariants and stationary theorem checks | historical P3 acceptance failed; immutable |
| `SIM-UNBALANCED-001` | `sim_unbalanced_001.toml` | One-factor parity-drift negative control | historical P3 acceptance failed; immutable |
| `SIM-FIG4-001` | `sim_fig4_001.toml` | Legacy independent partial Figure 4 contract | blocked and superseded before run |
| `SIM-MOMENTS-002` | `sim_moments_002.toml` | Powered integrated-flow and exact-generator validation | passed; global P3V supported |
| `SIM-UNBALANCED-002` | `sim_unbalanced_002.toml` | Powered jump-compensator negative control | passed; global P3V supported |
| `SIM-FIG4-PILOT-001` | `sim_fig4_pilot_001.toml` | Non-claim crossing, variance and compute pilot | failed before simulation: observation scheduler bug |
| `SIM-FIG4-PILOT-002` | `sim_fig4_pilot_002.toml` | Immutable replacement for the infrastructure pilot | preregistered; not run |

The source paper does not disclose the primitive parameters, simulator, seeds or raw
outputs used for Figure 4. Parameters in all three `SIM-*` configs are therefore
explicitly project-chosen. `SIM-UNBALANCED-001` is a labelled one-factor extension and
cannot replace the balanced baseline. None may be described as recovered author settings.

## Validation policy

P1 validates only that each TOML file parses, has the common required tables/fields,
uses unique experiment IDs and agrees with the preregistered protocol. Scientific runs
begin in P2/P3 after the strict typed validator and runner exist.

Missing or unknown fields, non-finite floats, scientific CLI overrides and implicit
seeds/defaults are errors. New result-affecting values require a new config or a dated
protocol amendment before the affected target output is inspected.

The strict typed validator and runner support both `ANA-*` contracts plus both generations
of `SIM-MOMENTS-*` and `SIM-UNBALANCED-*`. `SIM-FIG4-001` remains a preregistered P4
input and is superseded by the planned `SIM-FIG4-002`. Existing failed
run directories are immutable; any replacement requires a new experiment ID, justified
statistical contract and dated amendment rather than editing these configs.

## Statistical gate policy

[`statistical-gates-v1`](../../docs/protocols/common/statistical-gates.md) applies to
every future stochastic config. The old `SIM-*001` configs intentionally retain their
historical point/refinement gates for exact replay; they are not templates for new
experiments.

A future executable stochastic RunSpec must declare each gate class, estimand, target,
SESOI/equivalence margin with an external or downstream-sensitivity justification,
independent unit, familywise alpha/correction, target power and three-way decision rule.
Equality uses equivalence tests; superiority uses one-sided inference over a minimum
effect; refinement uses paired or independent equivalence and never `tolerance OR SE`.

Reference statistical primitives live in `ot_micromr.statistical_gates`. The `*002`
contracts follow the dated sensitivity amendment in the P3V protocol; their global Holm
decision is evaluated only after both immutable target runs exist.
