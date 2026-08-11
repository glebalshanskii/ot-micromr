# P3V sensitivity, estimators and power design

- Date: 2026-08-11
- Status: pilot complete; confirmatory runs not yet executed
- Protocol: [`p3v-powered-validation.md`](../protocols/synthetic/p3v-powered-validation.md)
- Decision: [`ADR-0006`](../adr/0006-p3v-estimators-power-and-compute.md)

## What was measured

Historical `SIM-*001` seed labels were reused only for pilot variance and controlled fault
injection. Confirmatory `20260821xx` seeds were not touched. The pilot used the balanced
project parameters, `epsilon=0.01`, 20 seeds, horizon `10000`, six scenarios and 5000
paired seed bootstraps.

The diagnostic strategy monitor evaluates the flip band at equal-spaced observation
endpoints. It vectorises threshold comparisons, fill extraction and reward arithmetic but
does not implement continuous Brownian first hits. Its outputs are sensitivity evidence,
not a Figure 4 reproduction.

Raw local artifacts (ignored by Git):

- `outputs/p3v/pilot-sensitivity.json`;
- `outputs/p3v/pilot-sensitivity-curves.csv`;
- `outputs/p3v/p3v-compute-benchmark.json`.

## Fault sensitivity

| Fault | Measured validation signal | Peak grid change | Normalised peak-rate change | Paired 95% interval |
|---|---:|---:|---:|---:|
| opening sampler, nominal `-0.10` | flow `-0.0820` | `-0.05 theta_D` | `+0.19%` | `[-0.89%, +1.26%]` |
| opening sampler, nominal `+0.10` | flow `+0.1108` | `-0.05 theta_D` | `+0.25%` | `[-0.55%, +1.04%]` |
| open drift `0.85x` | finite-h ratio `0.8394` | `-0.05 theta_D` | `-0.33%` | `[-1.21%, +0.49%]` |
| open drift `1.15x` | finite-h ratio `1.1569` | `-0.05 theta_D` | `-1.20%` | `[-2.14%, -0.34%]` |

Thus the original provisional margins `0.10` for flow and `0.15` for stochastic open
drift were not demonstrated immaterial at the one-percentage-point downstream resolution.
They were not carried into target configs.

The amended design uses flow margin `0.05`. The drift theorem is tested where it is exact:
the nominal generator residual must be below `1e-12` in each parity. Realised finite-h and
jump-compensator slopes remain reported compatibility diagnostics. A separate planted
`alpha_c=1.25` control provides the stochastic check that the event sampler exposes the
parity mechanism.

## Estimator changes

Each path now records:

- nominal integrated channel hazards and measured channel counts;
- channel compensators and variance-stabilised compensator values;
- pathwise opening/closing transition conservation;
- integrated-hazard flow residual;
- time-integrated gap-square exposure and realised jump-drift slope by parity;
- replay digest independent of whether raw events are retained.

Raw event logs are disabled in `SIM-*002`; the required CSV remains as a header-only audit
artifact. This removes about 112 MiB of historical-style event output without deleting
seed-level sufficient diagnostics. Per-step quantile arrays are also disabled for the long
powered runs; exact streaming step counts and maximum event probability remain, avoiding
an otherwise multi-gigabyte memory cost per concurrent batch.

## Power design after sensitivity

Normal approximations use worst-case Holm local alpha `0.025` and target power `0.90`.
They are planning approximations, not achieved-power claims.

| Component | Projected per-seed SD | Boundary distance | Approx. required seeds | Planned |
|---|---:|---:|---:|---:|
| flow equivalence, horizon `40000` | `0.0461` | `0.05` | `9` | `20` |
| unbalanced superiority, horizon `20000` | `0.1227` | `0.15` | `8` | `20` |
| flow refinement, conservative historical scaling | `0.0635` | `0.05` | `17` | `20` |
| control refinement, conservative historical scaling | `0.1807` | `0.15` | `16` | `20` |

No optional sample extension is allowed after target inspection.

## Compute benchmark

Machine: Intel i9-12900H class host, 20 logical CPUs; NumPy OpenBLAS native threads were
capped at one per worker through `threadpoolctl 3.6.0`.

| Workers | Seconds | Speedup | Digest match |
|---:|---:|---:|---|
| 1 | `17.329` | `1.00x` | yes |
| 4 | `5.465` | `3.17x` | yes |
| 10 | `4.114` | `4.21x` | yes |
| 20 | `4.084` | `4.24x` | yes |

Ten workers were selected: twenty improved elapsed time by less than one percent while
roughly doubling simultaneous process memory. The event-driven path is sequential,
branch-heavy and tied to NumPy PCG64DXSM stream semantics. PyTorch is absent from the
environment; no equivalent `torch.compile` implementation with a credible transfer/
compilation-inclusive advantage exists, so a heavy GPU dependency was not added.

## Remaining limitation

The sensitivity band monitor misses intra-observation Brownian crossings. It is sufficient
to reject unsafe validation margins but cannot establish Figure 4 itself. P4 still requires
the continuous-crossing monitor and its own rate/peak refinement inference.
