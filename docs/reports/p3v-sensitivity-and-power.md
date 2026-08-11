# P3V sensitivity, estimators and power design

- Date: 2026-08-11
- Status: confirmatory validation supported
- Protocol: [`p3v-powered-validation.md`](../protocols/synthetic/p3v-powered-validation.md)
- Decision: [`ADR-0006`](../adr/0006-p3v-estimators-power-and-compute.md)
- P4 backend decision: [`ADR-0007`](../adr/0007-p4-hybrid-cpu-cuda-backend.md)

## What was measured

Historical `SIM-*001` seed labels were reused only for pilot variance and controlled fault
injection. At the pilot freeze, confirmatory `20260821xx` seeds had not been touched. The
pilot used the balanced project parameters, `epsilon=0.01`, 20 seeds, horizon `10000`, six
scenarios and 5000 paired seed bootstraps.

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
roughly doubling simultaneous process memory. This choice applies to the adaptive
event-path generator, not automatically to vectorisable strategy evaluation.

## Confirmatory results

Both immutable target runs used Python `3.14.0`, NumPy `2.5.2`, SciPy `1.18.0`, the
Intel i9-12900H host, 10 processes with one native numerical-library thread each, and
clean commit `9dedb6b44d87933147420a316e445b19cf4c5080`.

| Experiment / component | Estimate | Preregistered boundary | Raw p-value | Holm adjusted p-value | Decision |
|---|---:|---:|---:|---:|---|
| `SIM-MOMENTS-002` flow | `0.01567` | equivalence `[-0.05, 0.05]` | `0.00629` | `0.00629` | supported |
| `SIM-UNBALANCED-002` jump contrast | `0.23015` | superiority over `0.10` | `0.00148` | `0.00297` | supported |
| flow primary-vs-fine | `0.00563` | equivalence `[-0.05, 0.05]` | `0.01170` | `0.01508` | supported |
| control primary-vs-fine | `-0.03620` | equivalence `[-0.15, 0.15]` | `0.00754` | `0.01508` | supported |

The flow TOST 90% interval was `[-0.00587, 0.03722]`. The unbalanced one-sided lower
confidence bound was `0.16407`, above the minimum effect. All deterministic replays,
generator residual checks, transition conservation rules and simulator invariants passed.
The moments run took `1189.46 s`; the control took `583.86 s`. No new seeds or horizon
extensions were added after inspection.

Local ignored artifacts:

- `outputs/SIM-MOMENTS-002/20260811T184531842286Z-4cb501542645-det/`;
- `outputs/SIM-UNBALANCED-002/20260811T190621510298Z-f3c0ff8a3b29-det/`;
- `outputs/p3v/global-gate.json`.

The global status is `supported`; P4 is no longer blocked by P3V.

## GPU feasibility before P4

After the confirmatory runs, PyTorch `2.13.0+cu130` was installed as the optional `gpu`
extra and tested on an NVIDIA RTX 3080 Ti Laptop GPU. The benchmark implements the exact
endpoint band state machine in vectorised NumPy and compiled Torch, including alternating
fills, spread-aware rewards and host-to-device transfer. A unit regression test compares
the vectorised CPU implementation with the historical scalar reference.

Representative workload: 20 paths, 50000 observations per path and 21 thresholds
(21 million path-threshold observations), median of five steady-state repetitions.

| Backend | End-to-end median | Speedup vs NumPy | Maximum rate error | Counts |
|---|---:|---:|---:|---|
| NumPy `float64` | `0.69324 s` | `1.0x` | reference | reference |
| compiled CUDA `float32` | `0.00445 s` | `155.6x` | `6.81e-6` | exact |
| compiled CUDA `float64` | `0.00563 s` | `123.2x` | `2.91e-13` | exact |

Cold compilation cost was `3.22 s` for `float32` and `0.94 s` for `float64`; it is
amortised by the planned repeated P4 evaluations. The P4 policy/threshold post-processing
backend was initially compiled CUDA `float64`. A subsequent 84-million-element benchmark
with ten repetitions measured `0.01516 s` in compiled `float32` and `0.01912 s` in
compiled `float64`: `float32` was `1.26x` faster and reduced end-to-end runtime by 20.7%.
Counts remained exact and maximum absolute rate error was `7.19e-6`, or `1.82e-7` of
the maximum reference-rate scale. Therefore `float32`
is the primary P4 candidate, with `float64` retained as regression oracle; the final freeze
still requires the full continuous-crossing kernel to pass. This decision does not port
the adaptive event generator to GPU and does not require NumPy/Torch RNG sequence identity.
Raw engineering artifacts: `outputs/p3v/band-backends.json` and
`outputs/p3v/band-backends-large.json`.

## Remaining limitation

The sensitivity band monitor misses intra-observation Brownian crossings. It is sufficient
to reject unsafe validation margins but cannot establish Figure 4 itself. The CUDA result
applies to this exact endpoint kernel; P4 still requires the continuous-crossing monitor,
its own CPU/GPU regression and rate/peak refinement inference before a claim-eligible run.
