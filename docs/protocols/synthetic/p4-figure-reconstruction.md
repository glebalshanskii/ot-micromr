# P4 protocol: independent Figures 2, 4 and 5 reconstruction

- Date: 2026-08-11
- Status: pilot preregistered; target amendment pending pilot variance/runtime only
- Paper: Amaral, *Optimal Trading of Microstructure Mean Reversion*, `arXiv:2608.00885v1`
- Paper SHA-256: `fd1a0dfc0d8fc8d7feb26ee23231232ac4263e95a5bb0ef41d18e4c0a8c611ba`
- Decision: [`ADR-0008`](../../adr/0008-p4-crossing-execution-and-inference.md)
- Statistical policy: [`statistical-gates-v1`](../common/statistical-gates.md)

## 1. Scope and source limitations

P4 independently reconstructs the paper's illustrative sample paths and Monte Carlo claims:

- Figure 2: the parity-locked bid/ask, efficient price and stationary gap;
- Figure 4: exact-model band-rate curves, inward optimum shift, overshoot and rate losses;
- Figure 5: fills, two-lot flips, spread-dependent threshold and marked wealth.

The paper omits all numerical primitives, sample size, seeds, horizon and simulation code.
Figures 2 and 5 are explicitly illustrative and uncalibrated. Therefore visual similarity is
not an acceptance criterion, and Figure 4 receives the permanent label
`independent_partial_reproduction`.

## 2. Project parameter family

Common primitives are `delta=1`, `sigma_X=1`, `mu_s=1`, `mu_o=0.01`, `mu_c=2` in
synthetic units. Sweep

$$
\alpha\in\{0.20,0.30,0.40,0.50,0.65,0.80\},\qquad
\alpha_s=\alpha/2,\quad\alpha_o=0,\quad\alpha_c=\alpha.
$$

This varies the ramp slopes while holding baselines fixed, as the Figure 4 caption states.
It is project-chosen and may not match the author's family.

For each row, independent calibration paths discard 50 reversion times and pool samples
from the next 100 at interval `0.02/alpha`. The pooled population standard deviation freezes
`s_G`; `gamma=(delta/2)/s_G`, `theta_D=s_G*u_D(gamma)` and `theta_star` then follow from
the analytical implementation already reproduced in P2. The nearest row to each paper display
target `{0.28,0.36,0.47}` is selected by absolute gamma distance, with lower row index on ties.

## 3. Pilot and target separation

`SIM-FIG4-PILOT-001` is non-claim evidence. It uses historical labels `20260811xx`, six
strategy seeds, horizon 300 reversion times, epsilons `{0.01,0.005}`, threshold multipliers
`0.50:0.10:1.60` and 1000 bootstrap replications. It may establish:

- realised gamma coverage;
- fill rate and whether every cell exceeds 100 complete intervals;
- seed-level variance of rate, peak shift and loss functionals;
- CPU worker scaling and CUDA reduction break-even;
- bridge-only hit share and omitted-recrossing bound.

It may not establish a paper claim. After pilot, a dated amendment freezes the target's new
seeds, horizon, threshold grid, epsilon pair, sample size, bootstrap count and maximum compute.
Target data are never used for optional extension.

## 4. Strategy and rate

Every policy starts flat after 50 market burn-in reversion times, acquires a position during
50 strategy burn-in times and must be non-flat at measurement. It then targets

$$
q_t^\theta=+1\;\text{at}\;G_t\le-\theta,
\qquad
q_t^\theta=-1\;\text{at}\;G_t\ge+\theta,
$$

holding otherwise. First entry is one lot and every later transition is a two-lot flip.
Diffusion crossing precedes a simultaneous right-end book event. Buys execute at ask and sells
at bid, with zero latency, fee, slippage, impact and partial fills.

For measured fills `F_0,...,F_N`, the seed-level primary estimator is

$$
\widehat R(\theta)=
\frac{\sum_{k=1}^{N}2(|G_{F_k}|-S_{F_k}/2)}{t_{F_N}-t_{F_0}}.
$$

At least 100 complete intervals are operationally required. Report raw rate, rate divided by
`alpha*s_G`, rate divided by the surrogate optimum, mean inter-fill time, overshoot, open-fill
share and a frozen `delta/2` cost diagnostic.

## 5. Estimands and inference

The across-seed mean curve is primary. The discrete-grid peak uses the lowest multiplier on an
exact tie. The row inward shift is `1 - theta_peak/theta_D`. Rate loss is one minus the mean
rate at `theta_D` or `theta_star` divided by the peak mean rate. A five-point quadratic fit
around the discrete peak is diagnostic only.

The target primary family contains the inward-shift claim for each distinct row selected for
the three paper gamma displays. The target amendment must set a nonzero SESOI using pilot
precision and the paper's claim resolution; one-sided cluster-bootstrap p-values receive Holm
correction. Rate-loss intervals are compatibility comparisons with the paper's 3--4% and
5--6% descriptions, never tuning targets.

Refinement compares fixed physical thresholds under epsilon `0.01` and `0.005`. Equivalence
margins and required sample size are frozen in the amendment using pilot variance and a
one-percentage-point downstream materiality scale. `inconclusive` is valid; `tolerance OR SE`
is forbidden.

## 6. Operational gates

- all requested rows, seeds and thresholds present;
- no parity, transition, intensity or nonfinite invariant violation;
- every policy non-flat before measurement and at least 100 complete measured intervals;
- omitted bridge-probability and full-band recrossing upper bounds below frozen budgets;
- deterministic replay for declared pilot/target seeds;
- Dawson residual below `1e-10` and calibration table written before P&L evaluation;
- clean tree for target; pilot may run dirty and is never claim eligible.

## 7. Figures 2 and 5

Figure 2 uses a separate labelled illustration with elevated opening baseline so open episodes
are visible, matching the paper's stated visualisation choice. Figure 5 uses one declared seed,
parity-dependent `theta_D(S)` with `phi=delta/2` in tight and `phi=delta` in open state, and
plots touch fills, inventory and mid-marked wealth against the surrogate slope. Neither path is
calibration or statistical evidence for Figure 4.

## 8. Required artifacts

Source/resolved config, manifest, calibration table and hash, seed-threshold metrics, summary,
bootstrap/functionals table, Figure 2/4/5 data and PNGs, fill sample, log and backend benchmark.
Heavy raw market paths remain ignored under `outputs/`.
