# Experiment configuration contract

Executable inputs use strict TOML `RunSpec v1`. Every result-affecting value is explicit;
unknown fields, implicit defaults, non-finite values and CLI overrides are rejected.
Runtime provenance is written to the immutable run directory and never mutates the source
config.

## Current experiments

| Experiment | Config | Purpose | Status |
|---|---|---|---|
| `ANA-SMOKE-001` | `ana_smoke_001.toml` | Dawson-root executable smoke check | passed |
| `ANA-FIG3-001` | `ana_fig3_001.toml` | Deterministic Figure 3 reconstruction | reproduced |
| `SIM-MOMENTS-002` | `sim_moments_002.toml` | Integrated-flow and balanced-generator validation | passed |
| `SIM-UNBALANCED-002` | `sim_unbalanced_002.toml` | One-factor jump-compensator negative control | passed |
| `SIM-FIG4-002` | `sim_fig4_002.toml` | Powered Figure 4 reconstruction on the current CPU/CUDA implementation | operationally passed; scientific family inconclusive |
| `EMP-DATA-001` | `emp_data_001.toml` | OKX raw integrity, large-tick eligibility and chronological split freeze | passed; `20260811T232210534423Z-45f5a299b7ff-det` |
| `FILTER-SYN-001` | `filter_syn_001.toml` | Causal six-event particle-filter identification against naive/Kalman controls | passed; `20260811T234700354892Z-9e7f2939b506-det` |
| `EMP-FILTER-001` | `emp_filter_001.toml` | Train-only OKX parameter fit and causal filter diagnostics | completed negative; `20260812T000514761846Z-7075bc32601b-det` |
| `FILTER-MARK-SYN-001` | `filter_mark_syn_001.toml` | Known-$X$ marked multi-spread filter validation | passed; `20260812T061258615041Z-6daac30b7613-det` |
| `EMP-MARK-FILTER-001` | `emp_mark_filter_001.toml` | Rolling-origin marked multi-spread OKX filter evaluation | completed negative; `20260812T063536959101Z-9956cb3f2077-det` |
| `EMP-MARK-CT-001` | `emp_mark_ct_001.toml` | Continuous-hazard refit/filter with endpoint event intensity and nested path quadrature | completed negative; refinement passed; `20260812T100151852237Z-c8a620999b93-det` |

`emp_data_001_sources.toml` is the strict acquisition contract for the P5 target. It contains
only official OKX URLs selected from train dates. Acquisition is always explicit:

```bash
uv run ot-micromr fetch-data cfg/experiments/emp_data_001_sources.toml
```

The command writes ignored raw files and a content-addressed manifest below `data/`; it does
not execute an experiment or inspect validation/test data.

The source paper does not disclose the primitive parameters, simulator, seeds or raw Figure 4
outputs. All `SIM-*` parameters are therefore project-chosen and must not be described as
recovered author settings.

Historical configs and executable variants were removed after P4. Their immutable commits,
ADRs and reports preserve the failed and superseded evidence without keeping obsolete code in
the active experiment surface.

## Statistical policy

[`statistical-gates-v1`](../../docs/protocols/common/statistical-gates.md) applies to every
stochastic experiment. Equality claims use equivalence tests; directional claims use
superiority over a justified minimum effect; refinement uses paired or independent equivalence.
Event-count heuristics are not acceptance gates. Decisions are made at the declared independent
seed-cluster level with the preregistered multiplicity correction and power target.

Reference statistical primitives live in `ot_micromr.statistical_gates`. The joint P3V Holm
decision is evaluated by `scripts/evaluate_p3v_family.py` after both current P3 outputs exist.
