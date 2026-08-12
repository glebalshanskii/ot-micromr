# P6D factorized clock and conditional-mark protocol

- **Status:** preregistered before target run
- **Date:** 2026-08-12
- **Experiment:** `EMP-MARK-FACT-001`
- **Architecture:** [`ADR-0019`](../../adr/0019-factorized-clock-mark-model.md)
- **Reference:** `EMP-MARK-CT-001/20260812T100151852237Z-c8a620999b93-det`

## Question and estimands

P6D tests whether the P6C clock failure comes from tying total BBO activity to the same
latent-gap correction that predicts the mark. For duration $T_i$ and mark $m_i$,

$$
\log L_i=\log f_\psi(T_i\mid\mathcal F_{i-1})
 +\log p_\theta(m_i\mid G_{t_i^-},S_{i-1}).
$$

The clock estimands are the 30-minute-block mean and standard deviation of
$z_i=-\log S_\psi(T_i\mid\mathcal F_{i-1})$. The mark estimand is the paired block mean
of full minus zero-tilt conditional log score. Downstream usability is the P6C metric
$1-\operatorname{median}(posterior\ SD)/option\ margin$.

## Frozen data and rolling origin

Use the unchanged verified P6 processed BBO dependency, OKX `BTC-USDT-SWAP`, dates
`2024-01-15, 03-15, 05-15, 07-15, 09-15, 11-15, 12-15`, and folds that train on all
earlier dates and score the next date. All healthy transitions remain supported. Spot data
remain a causal diagnostic only. Strategy, execution and P&L are disabled.

## Total-activity clock

Training supplies global mean $\mu_0$ and variance $v_0$ of log valid durations. For
held-out valid event $i$, let $H_i$ contain at most the 200 valid durations strictly before
$i$. With prior count $k=50$,

$$
\mu_i=\frac{\sum_{j\in H_i}\log T_j+k\mu_0}{|H_i|+k},
$$

$$
q_i=\frac{\sum_{j\in H_i}(\log T_j)^2+k(v_0+\mu_0^2)}{|H_i|+k},
\qquad \sigma_i=\sqrt{\max(q_i-\mu_i^2,10^{-4})}.
$$

Then $T_i\mid\mathcal F_{i-1}\sim\operatorname{LogNormal}(\mu_i,\sigma_i)$. Prefix sums
compute all parameters vectorially; the current duration is excluded by construction.
This defines a renewal hazard through $\Lambda_i(a)=f_i(a)/S_i(a)$ and an exact integrated
hazard $z_i=-\log S_i(T_i)$.

The 200/50 choice follows a disclosed design scan on the already-used development dates.
The target run does not rescan or select these values.

## Conditional mark model

Let $p^0_{s,m}$ be the Dirichlet-smoothed empirical mark table and
$d_m\in\{-1,0,1\}$ the midpoint direction. With $g=G/s_G$,

$$
p_\theta(m\mid g,s)=
\frac{p^0_{s,m}\exp(-\beta d_m g)}
{\sum_k p^0_{s,k}\exp(-\beta d_k g)},\qquad \beta\ge0.
$$

One scalar $\beta$ is fitted on the causal training proxy by conditional cross-entropy
with softplus parameterization. The zero-tilt comparator uses the identical $p^0$, clock,
folds and events. Because probabilities are normalized, $\beta$ cannot alter total event
activity.

Held-out filtering uses 256 particles, exact Brownian endpoints over observed irregular
durations, systematic resampling and common random numbers for replay. Only mark
likelihood changes relative particle weights.

## Statistical decisions

Thirty-minute blocks are the inference unit; no event is treated as an independent
replicate. The frozen decisions are:

1. **Clock calibration:** simultaneous TOST-style intervals with per-metric
   $\alpha=0.025$ must place rescaling mean inside `1.0 +/- 0.10` and rescaling SD inside
   `1.0 +/- 0.20`.
2. **Conditional mark signal:** a one-sided 95% block interval for full minus zero-tilt
   conditional log score must have lower bound above zero. No arbitrary positive effect
   threshold is used; practical usefulness is assessed separately by state uncertainty.
3. **State usability:** with one-sided 95% block inference, the lower bound of
   $1-posterior\ SD/option\ margin$ must exceed zero.

P7 is unblocked only if all three scientific decisions and every operational gate pass.
Clock improvement alone is insufficient. Outcomes are `supported`, `inferior`, or
`inconclusive`; failed calibration is not repaired by widening margins after the run.

## Required diagnostics and artifacts

The run must save:

- fold parameters, block metrics, inference table, dependency/timestamp/replay audits;
- December causal filter state and predictive clock/direction tensors;
- `timeseries.png`: observed bid/ask, filtered efficient price, actual/predicted duration,
  and observed/predicted direction;
- `predictive-distributions.png`: actual versus predictive duration, time-rescaling,
  direction, family, midpoint-magnitude and spread-magnitude distributions;
- `fold-calibration.png`: clock, mark-score and uncertainty results by held-out day;
- CSV figure data separate from PNG files.

All prediction distributions are formed before observing the scored duration/mark.
Predictive duration samples use the frozen CUDA generator seed only for visualization;
scientific clock metrics use analytic CDF/survival values.

## Operational gates and stop rule

Require dependency hashes, full healthy-transition support, zero future timestamp access,
finite tensors, positive posterior variance, deterministic December replay, frozen fold
parameters, complete figures/data and runtime below 1200 seconds. Stop on any nonfinite
value. Do not run strategy, orders, fills or P&L in P6D.
