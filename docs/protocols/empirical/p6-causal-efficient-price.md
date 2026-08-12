# P6: causal efficient-price estimation

- **Статус:** preregistered before filter target outputs
- **Дата:** 2026-08-12
- **Decision:** [`ADR-0014`](../../adr/0014-p6-causal-efficient-price-filter.md)
- **P5 dependency:** `EMP-DATA-001/20260811T232210534423Z-45f5a299b7ff-det`

## Scope and information sets

P6 оценивает latent efficient price, но не рассчитывает orders, fills или P&L. Synthetic
leg использует oracle $X$ только как labelled diagnostic. Empirical leg использует только
OKX train observations с timestamp не позже текущего decision timestamp. Validation
2025-01-01--2025-06-30 и test 2025-07-01--2025-12-31 остаются закрытыми.

## Synthetic design: FILTER-SYN-001

Balanced six-event model использует project-chosen P3 parameters:
$\delta=1$, $\sigma_X=1$, $(\mu_s,\mu_o,\mu_c)=(1,0.02,2)$ и
$(\alpha_s,\alpha_o,\alpha_c)=(0.4,0.2,1)$ per second. Fixed-step generator применяет
exact frozen-left Bernoulli occurrence probability и максимум один event на `dt=0.005 s`.
Burn-in `50 s`, measured horizon `100 s`; 64 independent sessions.

Particle filter: 256 particles/session, systematic resampling каждые 50 steps, Brownian
transition $X_{t+dt}=X_t+\sigma_X\sqrt{dt}\epsilon$. Interval likelihood:

$$
\log L_{no\ event}=-\lambda_\Sigma(G_t)dt,
$$

$$
\log L_{event\ k}=\log(1-e^{-\lambda_\Sigma(G_t)dt})
+\log\lambda_k(G_t)-\log\lambda_\Sigma(G_t).
$$

Naive baseline sets $\widehat X_t=M_t$ after every observation. Kalman control is a causal
two-state random-walk/OU filter for $(X,G)$ with observation $M=X+G$; it is secondary.

Primary session-level metrics:

1. `1 - RMSE(PF)/RMSE(naive)`, minimum useful effect `0.10`;
2. `(log_score(PF)-log_score(naive))/book_events`, minimum useful effect
   `0.01 nat/event`.

Both use one-sided normal cluster inference with Bonferroni `alpha=0.025` each, global
FWER 0.05. Planned alternatives are `0.20` and `0.03`; conservative session SDs are
`0.20` and `0.04`. Normal planning requires 43 sessions for 90% power; fixed `n=64`
provides margin. No sequential extension is allowed.

Secondary calibration estimand is session-level coverage of the PF Gaussian 90% posterior
interval. TOST-equivalent normal interval must lie inside `[0.85,0.95]` at alpha 0.05;
otherwise status is `inconclusive` or `meaningfully_different`, never «calibrated by
non-significance».

Operational gates: deterministic replay on the same CUDA/PyTorch environment; zero
non-finite weights; maximum simulated one-event probability below 0.10; oracle RMSE zero;
no strategy fields enabled; complete artifacts.

## Empirical design: EMP-FILTER-001

### Frozen chronology

| Role | UTC day | Permitted use |
|---|---:|---|
| Fit | 2024-01-15 | parameter estimation and basis initialization |
| Selection | 2024-07-15 | balanced/unbalanced likelihood comparison |
| P6 audit | 2024-12-15 | one final train-only diagnostic after model freeze |

March/May/September/November swap days supply descriptive parameter/stationarity stability
only. Spot/swap pairs exist exactly on fit, selection and audit dates.

### Normalization and transitions

Integer price ticks remain `0.1 USDT`. Mid in half-ticks is `bid_ticks + ask_ticks`.
Paper-compatible channels are exact transitions:

- tight $\to$ tight and mid change $\pm2$ half-ticks: slide up/down;
- tight $\to$ open and mid change $\pm1$: open up/down;
- open $\to$ tight and mid change $\pm1$: close up/down.

Any wider spread, other jump or recovery boundary starts a new segment. No likelihood or
signal is propagated across a reset. L2 impossible states use the P5 quarantine rule.

Spot reference at swap time uses `torch.searchsorted(..., right=true)-1`; equality is
allowed, any selected spot timestamp greater than swap timestamp is an invalid run.
Daily swap--spot basis is filtered causally; future daily median/subtraction is forbidden.

### Estimation and selection

Reduced $s_G$, $\alpha$ and $\sigma_X$ use fit-day causal spot-reference gaps and 1-second
past increments. Six-event ramp parameters use constrained Poisson likelihood with explicit
survival exposure. All rates have positive softplus parameterization; balanced and
unbalanced fits share initialization and optimization budget.

Selection estimand is paired 30-minute block difference
`NLL_balanced - NLL_unbalanced`, normalized per compatible event, on 2024-07-15. Unbalanced
is selected only if the one-sided 95% cluster lower bound of its improvement is above
`0.01 nat/event`; otherwise balanced is frozen by parsimony. This selection gate is not a
claim that the rejected model is equivalent.

Required empirical operational gates:

- zero future timestamp accesses and exact deterministic replay;
- all persisted filter values finite with posterior variance positive;
- fit/selection/audit roles match frozen dates;
- selected model and fitted parameters are frozen before audit evaluation;
- empirical posterior uncertainty median is strictly below the optimistic paper option
  margin $\theta_D-\delta/2$ on the audit day. Failure is a negative P6 feasibility result;
- synthetic dependency has supported both required primary metrics.

Stationarity, ACF, time-rescaling, parity drift, parameter stability, spot-reference error
and posterior coverage are descriptive because three paired days cannot support a general
market-distribution claim. They are saved fully and do not become post-hoc gates.

## Artifacts and decisions

Synthetic artifacts include session metrics, inference table, replay digest and summary.
Empirical artifacts include extracted daily event tables, fitted balanced/unbalanced
parameters, block likelihoods, day diagnostics, causal-filter metrics, timestamp audit and
summary. Heavy records remain local under `outputs/`.

P6 status:

- `supported`: both synthetic primary gates supported and all empirical operational gates pass;
- `negative`: synthetic signal fails materially or audit uncertainty exceeds option margin;
- `inconclusive`: required stochastic evidence lacks precision without material contradiction;
- `invalid`: data/timestamp/code contract is violated.
