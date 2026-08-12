# P6D factorized clock and conditional mark

- **Status:** completed; clock moment calibration passed, conditional mark and state
  usability failed
- **Date:** 2026-08-12
- **Protocol:** [`p6d-factorized-clock-mark.md`](../protocols/empirical/p6d-factorized-clock-mark.md)
- **Architecture:** [`ADR-0019`](../adr/0019-factorized-clock-mark-model.md)
- **Decision:** [`ADR-0020`](../adr/0020-p6d-clock-pass-state-negative.md)
- **Config:** [`emp_mark_fact_001.toml`](../../cfg/experiments/emp_mark_fact_001.toml)

## Result

P6D tested the P6C structural diagnosis directly by separating total BBO activity from the
conditional mark:

$$
\lambda_m(t)=\Lambda_\psi(t)p_\theta(m\mid\mathcal F_{t^-}).
$$

This repaired the two preregistered event-clock moment metrics, from `2.209/5.330` in P6C
to `1.069/1.138`. It did not make the latent state usable. The normalized gap tilt failed
to improve held-out conditional mark score and posterior uncertainty increased from `1.620`
to `8.426` option margins when silence no longer supplied the spurious gap-dependent timing
signal. P7/P8 remain blocked.

## Provenance

| Field | Value |
|---|---|
| Run | `EMP-MARK-FACT-001/20260812T105127206423Z-44416f08cb43-det` |
| Commit | `be0f33d6f014877786005f3437c63c88a8d382c5` |
| RunSpec SHA-256 | `44416f08cb43e56530571174b2460182572782dfa40a91b199595ef55afac7a3` |
| Config SHA-256 | `5186093eea2867a67f642a283560ecd6a7cbb289b8aa3ee7f38c0c03c90d9c29` |
| Data | verified P6 OKX dependency; `1,122,613` held-out healthy transitions |
| Hardware | NVIDIA GeForce RTX 3080 Ti Laptop GPU, CUDA 13.0 |
| Numerics | PyTorch 2.13 CUDA `float32`; statistics/clock reductions `float64`; `torch.compile(reduce-overhead)` |
| Runtime | `127.62 s` |
| Replay | exact tensor equality; digest `1b0ef8a7...1039da` |
| Status | `acceptance_failed`; operationally valid |

## Method

The causal clock predicts each next duration with a lognormal law. Its location and scale
use only 200 preceding healthy durations plus a 50-event prior estimated on previous
rolling-origin days. The duration currently being predicted is excluded. Analytic survival
gives the time-rescaling variable

$$
z_i=-\log S_\psi(T_i\mid\mathcal F_{i-1}).
$$

The mark baseline is the same smoothed spread-conditioned 729-bucket table as P6M/P6C.
Only direction receives a latent-gap tilt:

$$
p_\theta(m\mid g,s)=
\frac{p^0_{s,m}\exp(-\beta d_m g)}
{\sum_kp^0_{s,k}\exp(-\beta d_k g)},\qquad g=G/s_G.
$$

Normalization makes the total mark probability exactly one for every particle, so this
tilt cannot alter the event clock. The particle filter propagates Brownian $X$ over every
observed irregular interval and uses conditional mark probability, but not clock
likelihood, to update relative particle weights.

## Statistical results

| Estimand | Mean | Simultaneous interval | Required | Decision |
|---|---:|---:|---:|---|
| Time-rescaling block mean | `1.06917` | `[1.05827,1.08006]` | inside `[0.9,1.1]` | equivalent |
| Time-rescaling block SD | `1.13795` | `[1.12616,1.14974]` | inside `[0.8,1.2]` | equivalent |
| Conditional mark gain | `-0.0000834 nat/event` | `[-0.0001668,0.00000004]` | lower `>0` | inconclusive/negative center |
| $1-posterior\ SD/option\ margin$ | `-7.42566` | `[-7.53374,-7.31757]` | lower `>0` | inferior |

Clock moment calibration is a real improvement and not a redefinition of the P6C result.
However, it does not validate the distributional shape or the efficient-price estimator.
The directional coefficient shrinks from `0.0259` in the first fold to roughly
`0.011--0.012` later, and held-out gains change sign across dates.

## Visual diagnostics

### Observed BBO and causal predictions

![Observed BBO, duration and direction predictions](../../outputs/EMP-MARK-FACT-001/20260812T105127206423Z-44416f08cb43-det/figures/timeseries.png)

The top panel shows actual bid/ask and filtered efficient price for the first ten December
minutes. The duration panel compares each actual interval with the pre-event median and
10--90% predictive range. The directional expectation stays near zero, consistent with the
near-balanced actual up/down flow and the failed conditional score ablation.

### Actual and predictive distributions

![Actual and predictive distributions](../../outputs/EMP-MARK-FACT-001/20260812T105127206423Z-44416f08cb43-det/figures/predictive-distributions.png)

Aggregate category total-variation distances are:

| Distribution | TV distance |
|---|---:|
| Direction | `0.00225` |
| Spread family | `0.01808` |
| Midpoint magnitude | `0.07917` |
| Spread magnitude | `0.03708` |

The continuous lognormal duration histogram has descriptive TV distance `0.483` from the
actual histogram. The largest mismatch is the sharp actual mass near `0.01 s`, which is
consistent with exchange timestamps/batching and cannot be reproduced by a smooth
continuous density. The time-rescaling histogram is likewise not Exp(1) in full shape even
though its registered block mean and SD pass. These are descriptive findings because P6D
preregistered only the two moment gates.

### Fold stability

![Fold-level calibration](../../outputs/EMP-MARK-FACT-001/20260812T105127206423Z-44416f08cb43-det/figures/fold-calibration.png)

Five of six held-out dates have clock means within the shaded target range; the first fold
is low. Mark gain is positive only in three middle folds and negative in March, November
and December. Posterior SD is between `5.93` and `9.27` option margins on every date.

Figure data are stored next to the PNG files as `timeseries-data.csv`,
`predictive-distributions-data.csv` and `fold-calibration-data.csv`. Raw metrics, fold
parameters, distribution tables and the December tensor state are in the same immutable
run directory.

## Relation to the article

The article's gap-dependent intensities jointly determine event time and direction. P6D
keeps the article's central latent gap and mean-reversion interpretation, but factorizes the
empirical marked process to test whether that joint restriction caused the real-data
failure. It did cause the clock over-speed: separate timing is much better calibrated.
The remaining negative result is stronger, not weaker: once the false timing information is
removed, BBO marks alone provide almost no information about $X$ under this estimator.

## Limitations and decision

- The lognormal clock is a project-chosen renewal model and misses discrete timestamp mass.
- The same seven already studied dates are development evidence, not a new untouched test.
- Spot remains a causal diagnostic, not ground-truth efficient price.
- No orders, fills, costs, backtest or P&L were computed.

P6D therefore ends as a useful mixed result: clock factorization is supported at the frozen
moment margins, while book-only state identification and trading readiness are rejected.
The next empirical model must add an independently informative causal state observation;
clock tuning alone is not a reason to enter P7.
