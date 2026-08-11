# P4 paper reproduction report

- Date: 2026-08-11
- Paper: Amaral, *Optimal Trading of Microstructure Mean Reversion*,
  `arXiv:2608.00885v1`
- Result label: `independent_partial_reproduction`
- Stage outcome: **completed / operational acceptance failed / scientific inconclusive**
- Protocol: [`p4-figure-reconstruction.md`](../protocols/synthetic/p4-figure-reconstruction.md)
- Decisions: [`ADR-0007`](../adr/0007-p4-hybrid-cpu-cuda-backend.md),
  [`ADR-0008`](../adr/0008-p4-crossing-execution-and-inference.md)

## Outcome

The structural mechanisms in Figures 2 and 5 were reconstructed, and the Figure 4
exact-jump-model experiment was completed for three realised gamma values. The target is not
an author-parameter replication: the paper does not provide primitives, seeds, horizon or
code.

The result is mixed. At gamma `0.272` and `0.342`, the discrete rate peaks moved inward by
15% and 20%. At gamma `0.469`, the peak moved only 5% and the directional test was
inconclusive. Only the low-gamma row closely matched the paper's described rate losses.
The six-test resolution-refinement family was inconclusive after Holm correction.

Formal operational acceptance failed only because 38 of 4,320 seed-threshold cells had fewer
than the preregistered 20 complete intervals; the minimum was 12. These were almost entirely
far-right thresholds (`1.45--1.60 theta_D`) in the highest-gamma row. The three primary peak
cells had at least 80, 75 and 48 intervals per seed respectively. This contextualises the
failure but does not override the preregistered all-cell gate.

## Provenance and setup

| Field | Value |
|---|---|
| Experiment | `SIM-FIG4-002` |
| Run ID | `20260811T202753134457Z-837035232ead-det` |
| Git commit | `ca9aa7c1e9841fccb35f47f41a8e0863e795d3c7` |
| RunSpec SHA-256 | `837035232ead731c04a7d0a04970831a0388c55867a40e8d606f65d0f8f28201` |
| Source config SHA-256 | `74de348d2e3f0406770590749178d284ec2570fe3dbb9c50778155a2148ddf8f` |
| Seeds | 12 calibration; 30 independent target strategy seeds |
| Resolutions | `epsilon={0.01,0.005}` |
| Measured horizon | 300 reversion times |
| Grid | `theta/theta_D=0.50:0.05:1.60`, plus `theta_star` |
| Bootstrap | 10,000 complete seed-vector resamples |
| Hardware | Intel i9-12900H, 20 logical CPUs; NVIDIA RTX 3080 Ti Laptop, 16 GB |
| Software | Python 3.14.0; NumPy 2.5.2; SciPy 1.18.0; PyTorch 2.13.0, CUDA 13.0 |
| Runtime | 34.244 s |

Adaptive market endpoints were generated in CPU `float64` with ten processes. Continuous
Brownian-bridge crossings, alternating fills and threshold reductions used compiled CUDA
`float32`; the CUDA crossing phase took 9.964 s and CPU market generation 19.156 s. The
corresponding pilot fell from 186.996 s on the old policy-inside-step CPU implementation to
34.424 s on the hybrid implementation (`5.43x`).

## Figure 4 results

| Target gamma | Realised gamma | Discrete peak | Fitted peak | Inward shift | Holm p | Decision | Loss at theta_D (95% bootstrap) | Loss at theta_star (95% bootstrap) |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 0.28 | 0.272 | 0.85 | 0.848 | 15% | 0.0242 | supported | 3.39% (1.62%, 5.71%) | 4.87% (1.89%, 8.28%) |
| 0.36 | 0.342 | 0.80 | 0.856 | 20% | 0.000300 | supported | 0.13% (0%, 2.60%) | 1.20% (-0.15%, 4.23%) |
| 0.47 | 0.469 | 0.95 | 0.914 | 5% | 0.541 | inconclusive | 1.06% (0%, 4.07%) | 2.66% (1.01%, 5.70%) |

The paper describes an approximately 15--20% inward shift, 3--4% loss at `theta_D` and
5--6% at `theta_star`. The low-gamma row is compatible with all three descriptions. The
middle row reproduces the peak displacement but not the stated losses. The high-gamma row
does not reproduce the stated displacement and remains statistically inconclusive.

Realised peak rates were `0.867`, `0.902` and `0.948` of the Gaussian-surrogate optimum.
Thus the surrogate preserves the broad curve shape but overstates the exact-model rate for
this project-chosen family.

The Figure 4 plot and data are in
[`figure4.png`](../../outputs/SIM-FIG4-002/20260811T202753134457Z-837035232ead-det/figures/figure4.png)
and
[`figure4-data.csv`](../../outputs/SIM-FIG4-002/20260811T202753134457Z-837035232ead-det/figures/figure4-data.csv).

## Statistical and operational gates

The primary family used a minimum meaningful inward shift of 5% and Holm FWER 0.05.
Thirty seeds were frozen from a pilot-SD power calculation before target execution. Two of
three row-level tests were supported; the stage-level primary family is therefore
inconclusive rather than supported.

Refinement compared `epsilon=0.01` and `0.005` rates at fixed `theta_D` and `theta_star`
using independent Welch TOST, margin 0.02 and Holm correction across six checks. Adjusted
equivalence p-values ranged from 0.124 to 0.345, so every check was inconclusive. No extra
seeds were added after inspection.

All deterministic gates passed: requested rows/seeds/thresholds were complete; every policy
was non-flat at measurement; state and non-finite violations were zero; deterministic market
replay matched; maximum omitted-crossing bound was `5.28e-11`; maximum wealth-marking identity
residual was `4.26e-14`. The all-cell 20-interval floor failed, producing the immutable
`acceptance_failed` run status.

## Figures 2 and 5

Figure 2 is a structural illustration with an explicitly elevated opening baseline so
two-tick episodes are visible. It demonstrates parity-locked bid/ask jumps and a continuous
efficient price; it is not calibrated or pixel-identical to the paper. Artifacts:
[`figure2.png`](../../outputs/SIM-FIG4-002/20260811T202753134457Z-837035232ead-det/figures/figure2.png)
and
[`figure2-data.csv`](../../outputs/SIM-FIG4-002/20260811T202753134457Z-837035232ead-det/figures/figure2-data.csv).

Figure 5 uses a parity-dependent Dawson threshold, displayed-touch fills, one-lot entry,
two-lot flips and both mid- and efficient-price-marked wealth. Its single path has positive
gross wealth after the model spread, but remains an illustrative synthetic path and is not
evidence of real-market profitability. Artifacts:
[`figure5.png`](../../outputs/SIM-FIG4-002/20260811T202753134457Z-837035232ead-det/figures/figure5.png)
and
[`figure5-data.csv`](../../outputs/SIM-FIG4-002/20260811T202753134457Z-837035232ead-det/figures/figure5-data.csv).

## Limitations and claims

- Author parameters, raw Monte Carlo output and implementation are unavailable; exact
  numeric reproduction is impossible.
- The market simulator is an adaptive frozen-left single-jump approximation. Brownian
  one-sided hit probabilities are exact conditional on endpoints, while simultaneous
  two-boundary hits are bounded and refined rather than exactly timed.
- CUDA and CPU use different bridge RNGs; pilot-level functional agreement replaces
  impossible bitwise cross-library identity.
- Fees, latency, slippage, impact, queue position and filtering of the latent efficient price
  are absent. Positive synthetic gross rates do not establish a deployable strategy.
- The primary discrete argmax is noisy and non-smooth; fitted peaks are diagnostic only.

Claim statuses:

| Claim | Status |
|---|---|
| Figure 2 structural mechanism | reproduced illustratively |
| Figure 4 inward optimum | partially reproduced; family inconclusive |
| 3--4% loss at `theta_D` | compatible only for low gamma |
| 5--6% loss at `theta_star` | compatible only for low gamma; high-gamma interval overlaps |
| Figure 5 strategy mechanics | reproduced illustratively |
| Real-market profitability | not tested; requires P5--P9 |

Canonical raw evidence is the immutable run directory
[`outputs/SIM-FIG4-002/20260811T202753134457Z-837035232ead-det`](../../outputs/SIM-FIG4-002/20260811T202753134457Z-837035232ead-det/).
