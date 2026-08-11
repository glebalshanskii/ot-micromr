# Experiment configuration contract

Executable experiment inputs use strict TOML `RunSpec v1`, defined by
[`ADR-0001`](../../docs/adr/0001-research-modes-package-and-run-contract.md).
Every result-affecting value must be present in the source config; runtime provenance is
written separately to `manifest.json` and never mutates the preregistered TOML.

## P1 configs

| Experiment | Config | Purpose | Run status |
|---|---|---|---|
| `ANA-SMOKE-001` | `ana_smoke_001.toml` | One deterministic Dawson-root contract | preregistered; not run |
| `ANA-FIG3-001` | `ana_fig3_001.toml` | Deterministic Figure 3 reconstruction | preregistered; not run |
| `SIM-MOMENTS-001` | `sim_moments_001.toml` | Jump-model invariants and stationary theorem checks | preregistered; not run |
| `SIM-UNBALANCED-001` | `sim_unbalanced_001.toml` | One-factor parity-drift negative control | preregistered; not run |
| `SIM-FIG4-001` | `sim_fig4_001.toml` | Independent partial Figure 4 reconstruction | preregistered; not run |

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
