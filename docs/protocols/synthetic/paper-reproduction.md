# Preregistered protocol: synthetic reproduction of Amaral (2026)

- **Статус:** version 1.1 executed for P2/P3; P3 historical gates failed; future stochastic use amended
- **Protocol version:** 1.1 historical + post-run amendment A1
- **Дата регистрации:** 2026-08-11
- **Paper:** Lucas Rabechini Amaral, *Optimal Trading of Microstructure Mean Reversion*, `arXiv:2608.00885v1`
- **Paper SHA-256:** `fd1a0dfc0d8fc8d7feb26ee23231232ac4263e95a5bb0ef41d18e4c0a8c611ba`
- **Track / mode:** `synthetic` / `paper-faithful`
- **Related decisions:** [`ADR-0001`](../../adr/0001-research-modes-package-and-run-contract.md), [`ADR-0002`](../../adr/0002-controlled-jump-simulation-semantics.md), [`ADR-0005`](../../adr/0005-statistical-decision-gates.md)
- **Configs:** [`cfg/experiments/`](../../../cfg/experiments/)
- **Results location:** `docs/reports/paper-reproduction.md` после выполнения; этот protocol не переписывается под результат

Pre-run clarification `2026-08-11`: до первого target run для bounded direct
optimizer явно зафиксированы bounds, `xatol=10^{-12}` и `maxiter=500` в обоих
analytical configs. Scientific estimands, grids и acceptance gates не менялись;
target outputs при уточнении не создавались и не просматривались.

Post-run amendment A1 `2026-08-11`: после выполнения P3 проведён
[`statistical gate audit`](../../reports/statistical-gate-audit.md). Sections 9--11 и
configs `SIM-MOMENTS-001`/`SIM-UNBALANCED-001` остаются immutable historical contract;
их decisions не пересчитываются задним числом. Для любых новых stochastic runs этот
protocol дополняется [`statistical-gates-v1`](../common/statistical-gates.md): equality
требует multiplicity-aware equivalence, directional claim — superiority over justified
minimum effect, а refinement больше не может проходить по rule `tolerance OR SE`.
`SIM-FIG4-001` в старом виде не запускается и будет заменён новым experiment ID только
после margin-sensitivity и power design.

## 1. Purpose и граница claims

Protocol заранее фиксирует analytical и Monte Carlo проверки статьи до реализации
target algorithms и просмотра target outputs. Цели:

1. независимо воспроизвести closed-form OU-surrogate results и Figure 3;
2. проверить exact identities paper jump model на самостоятельно выбранном parameter
   instance;
3. провести independent partial reconstruction Figure 4 и разложить discrepancy на
   numerical resolution, overshoot, timing и spread-state effects.

Synthetic P&L является model-conditional evidence и не поддерживает утверждение о
доходности на реальном рынке. Empirical data, filtering latent $X$, market latency,
fees и backtests относятся к отдельному future protocol.

## 2. Source facts и preregistered status

Статья предоставляет equations, proofs и готовые figures, но не предоставляет:

- simulation code или executable configs;
- primitive parameters Figures 2, 4 и 5;
- initial state, algorithm/discretization, horizon или burn-in;
- seeds, number of paths/cycles, peak estimator или raw outputs.

Из Figure 4 известны только representative
$\gamma\approx\{0.28,0.36,0.47\}$, sweep ramp slopes при fixed unnamed baselines и
bands шириной one standard error. Поэтому:

- `ANA-SMOKE-001` и `ANA-FIG3-001` могут получить status `reproduced` или
  `not-reproduced` относительно exact formulas;
- `SIM-MOMENTS-001` проверяет theorems на project-chosen instance;
- `SIM-FIG4-001` всегда маркируется `independent_partial_reproduction`, даже если
  qualitative ranges совпадут;
- неизвестные author settings не восстанавливаются подбором по опубликованной figure.

## 3. Claims и estimands

| Claim ID | Paper target | Experiment | Primary estimand |
|---|---|---|---|
| `PAPER-B.1-DAWSON-OPTIMUM` | Unique exact surrogate optimum | `ANA-SMOKE-001`, `ANA-FIG3-001` | Absolute Dawson FOC residual и agreement с direct optimizer |
| `PAPER-3.7-KRAMERS-ROOT` | $u^*$ — large-threshold approximation | `ANA-FIG3-001` | $u_D$, $u^*$ и rate loss по fixed $\gamma$ grid |
| `PAPER-FIG3-THRESHOLDS` | Threshold curves Figure 3 | `ANA-FIG3-001` | Deterministic curve/table |
| `PAPER-FIG3-RATE-CURVES` | Myopic rate zero, unique interior maximum | `ANA-FIG3-001` | Normalized rate curve and maximum |
| `PAPER-FACT1-PARITY-LOCK` | Spread является parity mid | `SIM-MOMENTS-001` | Pathwise violation count |
| `PAPER-2.9-OPEN-FLOW-BALANCE` | Stationary open/close flow identity | `SIM-MOMENTS-001` | Relative flow imbalance |
| `PAPER-2.12-BALANCED-DRIFT` | Drift $-\alpha G$ в обеих parities | `SIM-MOMENTS-001` | Parity-specific binned/regression slopes |
| `PAPER-2.14-STATIONARY-MOMENTS` | Mean, variance identity и exponential ACF | `SIM-MOMENTS-001` | Seed-level moment and ACF estimates |
| `CONTROL-UNBALANCED-PARITY-SPLIT` | One-factor negative control, не paper theorem claim | `SIM-UNBALANCED-001` | Generator slopes and finite-h parity contrast |
| `PAPER-FIG4-INWARD-OPTIMUM` | Jump optimum inside $\theta_D$ | `SIM-FIG4-001` | Discrete optimum shift fraction |
| `PAPER-FIG4-RATE-LOSS-AT-DAWSON` | 3--4% loss at $\theta_D$ | `SIM-FIG4-001` | Rate loss against project grid optimum |
| `PAPER-FIG4-RATE-LOSS-AT-KRAMERS` | 5--6% loss at $\theta^*$ | `SIM-FIG4-001` | Rate loss against project grid optimum |

Scientific claim status и stage completion различаются: completed run может дать
`not-reproduced`, `inconclusive` или numerical blocker.

## 4. Units, precision и environment

- Analytical runs используют normalization $\alpha=1\,\mathrm{s}^{-1}$,
  $s_G=1$ synthetic price unit; $u=\theta/s_G$ и
  $\gamma=\phi/s_G$ dimensionless.
- Simulation time хранится в seconds; horizon и burn-in задаются также в reversion
  times $1/\alpha$.
- Price — `synthetic_price_unit`, quantity — integer `lot`, cash —
  `synthetic_quote_currency`, timezone — UTC.
- Все вычисления — IEEE-754 `float64` на CPU. GPU и mixed precision запрещены для
  confirmatory runs.
- P1 не добавляет dependencies. P2/P3 добавляют необходимые NumPy/SciPy только через
  `uv add`; exact versions и hashes фиксирует `uv.lock` и run manifest.
- Confirmatory run требует clean worktree и full commit SHA. Smoke run может быть
  dirty только с сохранённым sanitized patch hash в manifest.

## 5. Analytical experiments

### 5.1. Common formulas

Для $u=\theta/s_G$ и $\gamma=\phi/s_G$:

$$
F(u;\gamma)=u-\gamma-\sqrt 2D\left(\frac{u}{\sqrt 2}\right),
$$

$$
u^*=\frac{\gamma+\sqrt{\gamma^2+4}}{2},
\qquad
\frac{\widetilde R(u)}{\alpha s_G}
=\frac{2}{\pi}\frac{u-\gamma}{\operatorname{erfi}(u/\sqrt 2)}.
$$

$u_D$ is the unique root $F(u_D;\gamma)=0$ with $u_D>\gamma$.

### 5.2. `ANA-SMOKE-001`

Config: `cfg/experiments/ana_smoke_001.toml`.

- Fixed point: $\alpha=1$, $s_G=1$, $\gamma=\phi=0.4$.
- Primary metric: $|F(u_D;0.4)|$.
- `brentq` bracket: $(\gamma+10^{-12},8]$; `float64`, `xtol=rtol=10^{-13}`,
  at most 200 iterations.
- Cross-check: bounded direct maximization of $\widetilde R$ on
  $[\gamma+10^{-9},8]$, `xatol=10^{-12}`, `maxiter=500`.
- No RNG operation is permitted; the stored seed is contract metadata only.

Acceptance:

- root residual $\le10^{-10}$;
- $u_D>\gamma$;
- root and direct optimizer differ by $\le10^{-7}$;
- direct and optimum-identity rates differ by $\le10^{-10}$.

This is the first executable baseline. Failure stops P2 before Figure 3.

### 5.3. `ANA-FIG3-001`

Config: `cfg/experiments/ana_fig3_001.toml`.

- Threshold grid: $\gamma=0.05,0.06,\ldots,3.00$.
- Rate curves: $\gamma\in\{0.25,0.5,1.0\}$,
  $u\in[\gamma,3]$ with step 0.005 and exact myopic endpoint.
- Direct-optimizer diagnostic uses $[\gamma+10^{-9},8]$, `xatol=10^{-12}` and
  `maxiter=500`; it does not replace the preregistered Dawson-root gates.
- Precomputed audit points from independent evaluation, not values to return blindly:

| $\gamma$ | $u_D$ | $u^*$ | Rate loss at $u^*$ |
|---:|---:|---:|---:|
| 0.05 | 0.5416928365 | 1.0253124512 | 0.0871240388 |
| 0.28 | 1.0065633048 | 1.1497524449 | 0.0129152662 |
| 0.40 | 1.1558728538 | 1.2198039027 | 0.0030007967 |
| 1.70 | 2.2649755449 | 2.1624404748 | 0.0229956694 |

Acceptance следует exact tolerances config: root residual $\le10^{-10}$, audit-point
errors не выше $5\times10^{-9}$ для $u$ и $5\times10^{-8}$ для rate-loss fraction,
myopic rate absolute value $\le10^{-12}$. Все grid rows и figure data публикуются;
поиск только maximum/выгодных cells запрещён.

На preregistered grid $\gamma=0.40,0.41,\ldots,3.00$ independent audit даёт maximum
rate-loss fraction `0.0230357036` at $\gamma=1.66$. Эти два значения и tolerances
явно записаны в config; rounded paper statements `2.3%` и “near nine percent” остаются
descriptive comparisons, а не gates с неявными targets.

## 6. Project-chosen jump-model instance

`SIM-MOMENTS-001` использует следующий normalized instance:

| Primitive | Value |
|---|---:|
| $\delta$ | 1.0 price unit |
| $\sigma_X$ | 0.5 price unit$/\sqrt{\mathrm{s}}$ |
| $\mu_s$ | 1.0 $\mathrm{s}^{-1}$ |
| $\mu_o$ | 0.01 $\mathrm{s}^{-1}$ |
| $\mu_c$ | 2.0 $\mathrm{s}^{-1}$ |
| $\alpha_s$ | 0.5 $\mathrm{s}^{-1}$ |
| $\alpha_o$ | 0.0 $\mathrm{s}^{-1}$ |
| $\alpha_c=\alpha$ | 1.0 $\mathrm{s}^{-1}$ |

Balanced response holds exactly:
$2\alpha_s+\alpha_o=\alpha_c=\alpha=1$. At $G=0$ baseline open occupancy is
$\mu_o/(\mu_o+\mu_c)\approx0.004975$; stationary occupancy remains an estimand, not
an assumed result. Choosing $\alpha_o=0$ isolates the exact balanced identity.
`SIM-UNBALANCED-001` меняет только $\alpha_c:1.0\to1.25$; этот negative control
проверяет, что tight/open generator slopes расходятся, и не заменяет baseline.

Initial state is tight and gap-neutral. Canonical representation is
`time_seconds=0`, `mid_half_ticks=1`, $X_0=\delta/2$, $q_0=W_0=W^X_0=0$; values
$B_0=0$, $S_0=\delta$, $M_0=\delta/2$ and $G_0=0$ are derived assertions, not
independent mutable fields. Author parameters are not claimed or inferred.

## 7. Seeds, horizon и random streams

Analytical configs store seed `20260811` but do not consume RNG.

Confirmatory simulation uses exactly 20 ordered master seeds:

```text
2026081101, 2026081102, 2026081103, 2026081104, 2026081105,
2026081106, 2026081107, 2026081108, 2026081109, 2026081110,
2026081111, 2026081112, 2026081113, 2026081114, 2026081115,
2026081116, 2026081117, 2026081118, 2026081119, 2026081120
```

- RNG: NumPy `PCG64DXSM`.
- Single-policy `SIM-MOMENTS-001` and `SIM-UNBALANCED-001` use fixed
  `SeedSequence.spawn` order: `brownian_increment`, `book_occurrence`,
  `book_channel`, `brownian_bridge`.
- `SIM-FIG4-001` uses zero-based `response_scale_index`. Domain tag `0` spawns the
  three endpoint/book streams; domain tag `1` plus stable `policy_id` creates a bridge
  stream. Grid IDs are `0..20`, $\theta^*$ is `21`; an aliased $\theta^*$ uses the
  matching grid policy. Policies share an endpoint/book **market skeleton**, not the
  same continuous bridge path; complete policy vectors remain one seed cluster.
- Bootstrap seed: `2026081199`, 10,000 resamples.
- `SIM-MOMENTS-001`/`SIM-UNBALANCED-001` burn-in: 100 reference reversion times.
- `SIM-FIG4-001`: 100-reversion-time calibration pre-period, then 100 strategy
  burn-in reversion times after thresholds freeze; the declared total
  `burn_in_reversion_times` is therefore 200.
- Measured horizon: 2000 reference reversion times per seed after the applicable
  burn-in sequence.
- Scheduled parallelism may reorder completed tasks but cannot change stream mapping.
- A failed replication is retained and not silently replaced by a new seed. Fewer than
  20 valid replications means confirmatory gate fails unless a dated amendment changes
  the protocol before re-run.

Horizon/seed count may be increased only to improve predeclared precision, retaining
the original runs and seed prefix; model parameters and target metrics stay fixed.

## 8. Numerical simulator semantics

ADR-0002 is normative. The simulator is an
`adaptive-frozen-intensity approximation`, never an exact sampler.

Resolution ladder:

```text
epsilon = 0.020  coarse diagnostic
epsilon = 0.010  primary
epsilon = 0.005  acceptance refinement
```

For left total intensity $\Lambda_t$, reference response $\alpha$ and next scheduled
observation/phase boundary $t_b$:

$$
h=\min\left\{T-t,t_b-t,\frac{\epsilon}{\alpha},
\frac{-\log(1-\epsilon)}{\Lambda_t}\right\}.
$$

Thus the frozen-left probability of at least one event is at most $\epsilon$; at most
one book event is generated per step. Rates are not retroactively recomputed after the
Brownian endpoint is observed. Book jump is placed at the right endpoint after
Brownian-bridge crossings; strategy execution after a jump uses post-jump gap and touch.
The precomputed regular sample grid splits steps exactly: no interpolation or additional
sample RNG is allowed, and the snapshot is recorded after the full end-state phase.

After entry, the single active opposite boundary uses exact bridge hit probability and
the conditional left-first recursion of ADR-0002; localization leaf width is at most
`epsilon * h`. The sole flat entry uses a level-order dyadic bridge polyline with depth
`ceil(log2(1/epsilon))`, resolving the two-boundary order without independent hit tests.
Every run logs step/event counts, left probability cap, right-intensity diagnostics,
bridge-only hits, flat-tree entries, rejection attempts, multiple-crossing refinements
and invariant failures.

Primary results at 0.010 are repeated at 0.005 without changing seeds, horizon,
parameters, estimators or policies. Same seed labels across adaptive resolutions do not
imply paired pathwise coupling.

## 9. `SIM-MOMENTS-001`

Config: `cfg/experiments/sim_moments_001.toml`.

### 9.1. Pathwise gates

Required violation count is zero for:

- negative active intensities or nonzero inactive intensities;
- illegal parity/event transition or off-grid mid;
- simultaneous/multiple book jumps in one numerical step;
- discontinuous $X$ at book event or independently evolved $G$;
- nonfinite state/metric;
- deterministic replay mismatch in a fixed environment.

Zero inactive intensities are valid and are not counted as negative.

### 9.2. Statistical estimands

After burn-in, $G$ is sampled every $\Delta\tau=0.01$ reversion times. Within each
seed, stationary variance is the population second central moment of this equal-spaced
sample. Jump variance rate is

$$
\widehat\sigma_M^2=T^{-1}\sum_m(\Delta M_m)^2,
$$

For each seed, the variance-identity signed relative residual is estimate minus
$(\sigma_X^2+\widehat\sigma_M^2)/(2\alpha)$ divided by that positive target. The
reported equality error is the absolute arithmetic mean of the 20 signed residuals,
not the mean of their absolute values.

For open-flow balance put $L=\widehat p/(1-\widehat p)$ and

$$
R=
  \frac{\mu_o+(\alpha_o/\delta)E[|G|\mid S=\delta]}
       {\mu_c+(\alpha_c/\delta)E[|G|\mid S=2\delta]};
$$

each seed's signed residual is $2(L-R)/(|L|+|R|)$. The reported flow error is the
absolute mean seed residual. Conditional expectations use equal-spaced observations in
their stated parity.

Drift has two separate checks. First, at every accepted left state compute the exact
generator quantity

$$
b_M(G,S)=\sum_j\lambda_j(G,S)\,\Delta M_j.
$$

The pathwise residual $b_M+\alpha G$ must be at floating-point zero in both parities.
Second, realized dynamics use one-step finite-$h$ slopes. For seed $r$ and left parity
$p$,

$$
\widehat\beta_{r,p}=
\frac{\sum_{t:S_t=p}G_t(G_{t+h}-G_t)/h}{\sum_{t:S_t=p}G_t^2},
\qquad
\beta_h=\frac{e^{-\alpha h}-1}{h},
$$

with $h=0.01/\alpha$. Thus the target is the exact finite-h conditional-mean slope,
not the infinitesimal value $-\alpha$. At least 200 observations are required in each
seed/parity. For the fixed gap-bin diagnostic, every observation is divided by its own
seed's stationary $s_G$ before pooling. The bin edges
`[-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3]` are a descriptive diagnostic: cells with fewer
than 1,000 pooled observations are `insufficient` and are not merged post hoc.

Within each seed, ACF is sample autocovariance about that seed's sample mean divided by
lag-zero sample variance, at lags
$\{0.25,0.5,1,2,3,5\}/\alpha$ versus $e^{-\alpha h}$. Also report stationary mean,
open occupancy, event counts, simulator diagnostics and effective sample size.

### 9.3. Statistical procedure и acceptance

One seed-level estimate is the independent replication unit. Report mean, standard
error and two-sided Student-$t$ 95% interval across 20 seeds. The two finite-h parity
slopes and six ACF values form one eight-coordinate vector. A centered seed-cluster
bootstrap resamples complete vectors 10,000 times, studentizes every coordinate within
each resample and uses the 0.95 quantile of the maximum absolute $t$ statistic for
simultaneous intervals. This preserves cross-metric dependence and does not treat dense
within-path observations as independent.

Predeclared gates:

- every invariant count is zero and $|b_M+\alpha G|\le10^{-12}$;
- absolute mean of the seed-level $\widehat{E G}/\widehat s_G$ ratios is $\le0.02$;
- stationary variance-identity relative error $\le0.03$;
- open/close flow relative error $\le0.03$;
- each mean finite-h parity slope differs from $\beta_h$ by at most 5% of
  $|\beta_h|$, and both targets lie inside simultaneous 95% bands;
- theoretical ACF lies inside the same family of simultaneous 95% bands at every lag.

Refinement from epsilon 0.010 to 0.005 covers the explicit config list: normalized
mean, signed variance/flow residuals, both slope/target ratios and every ACF lag. Each absolute
shift must be no more than 0.01 or the conservative difference standard error
$\sqrt{SE_{.01}^2+SE_{.005}^2}$. Equal seed labels are not treated as paired because
adaptive runs have no proved coupling.

Both raw estimate and uncertainty are reported even when a threshold passes. A gate
failure is not repaired by changing bins, burn-in, seeds or primitives.

### 9.4. `SIM-UNBALANCED-001` negative control

This labelled extension changes exactly one primitive from `SIM-MOMENTS-001`:
$\alpha_c:1.0\to1.25\,\mathrm{s}^{-1}$. Hence the generator target is $-G$ in the
tight state and $-1.25G$ in the open state; `alpha_ref_per_second=1.25` controls the
step size. All baselines, $\sigma_X$, initial state, seeds, burn-in, horizon, sampling,
resolution ladder and diagnostics remain fixed.

Generator OLS slopes must match $-1.0$ and $-1.25$ within $10^{-10}$ and the pointwise
generator residual within $10^{-12}$. For realized finite-h slopes the preregistered
directional contrast is $\widehat\beta_{tight}-\widehat\beta_{open}$; its one-sided 95%
Student-$t$ lower bound across seeds must exceed $0.05\,\mathrm{s}^{-1}$. This second
target is deliberately directional rather than an exact exponential formula because
parity may change within the finite interval. Its epsilon 0.010--0.005 shift must be no
more than $0.02\,\mathrm{s}^{-1}$ or
$\sqrt{SE_{.01}^2+SE_{.005}^2}$. Failure means the simulator does not detect the
one-factor control; it does not weaken the balanced baseline gates.

## 10. `SIM-FIG4-001`

Config: `cfg/experiments/sim_fig4_001.toml`.

### 10.1. Parameter family

Keep $\delta,\sigma_X,\mu_s,\mu_o,\mu_c$ fixed at Section 6 values. Sweep response
$\alpha\in\{0.35,0.5,0.75,1.0,1.5,2.0\}\,\mathrm{s}^{-1}$ with
$\alpha_s=\alpha/2$, $\alpha_o=0$, $\alpha_c=\alpha$.

For each response row and all 20 seeds, simulate a 100-reversion-time pre-period before
activating any policy. Discard its first 50 reversion times, then pool equal-spaced $G$
samples every 0.01 reversion times from the remaining 50 across seeds. The row-level
$s_G$ is the square root of this pooled population second central moment. A single table
of $s_G$, realized $\gamma=\delta/(2s_G)$, $\theta_D$ and $\theta^*$ is written and
hashed before any strategy rate is evaluated; every policy/seed in that row uses the
same frozen values.

For display, choose the nearest row to each paper target $\{0.28,0.36,0.47\}$ using
epsilon-0.010 calibration $\gamma$ only. All response rows remain in tables. No response
value may be added to improve agreement with P&L curves without a dated amendment.
Exact distance ties choose the lower response-grid index; one row may represent more
than one target and is not duplicated computationally.

### 10.2. Policies, fills and exact-model rate

For each response row/seed, policies share the Brownian endpoints and book events of one
market skeleton. They do **not** claim a shared continuous Brownian path: each policy
uses its own stable-ID `brownian_bridge/<policy_id>` stream. This avoids policy-order
dependence; the full vector is still one seed cluster for inference.

After the pre-period every policy starts flat and runs through a further 100-reversion-
time policy burn-in; those fills and accounting are discarded. Measurement then runs
for $2000/\alpha$, and its first fill is only the renewal left boundary. Run symmetric
flip bands at

$$
\theta/\theta_D=0.60,0.65,\ldots,1.60,
$$

where $1.00\theta_D$ is the unique $\theta_D$ policy; add $\theta^*$ unless its physical
price is within $10^{-12}$ price units of a grid policy, in which case it aliases that
policy and no bridge stream is added. First entry is one lot, subsequent flips are two
lots, and there is no flat zone.

The frozen tight value $\phi=\delta/2$ is used **only** to compute $\gamma$, $\theta_D$
and $\theta^*$. Primary execution follows paper (2.15)--(3.3): post-state-move at the
displayed touch with realized $\phi_F=S_F/2$, zero latency, fees, slippage and impact.
The synthetic contract multiplier is one quote-currency unit per price-unit lot.
Let $F_0$ be the first fill after measurement starts and $F_1,\ldots,F_N$ the later
fills ending within the measured horizon. $F_0$ establishes the left boundary and
contributes no reward interval. The seed-level primary renewal estimate is

$$
\widehat R=
\frac{\sum_{k=1}^{N}2\left(|G_{F_k}|-\phi_{F_k}\right)}
     {t_{F_N}-t_{F_0}},
$$

with at least 100 complete inter-fill intervals required per seed/policy. A frozen-cost
diagnostic replaces every $\phi_{F_k}$ by $\delta/2$ on exactly the same eligible fills;
it is not labelled exact-model P&L. Secondary ledger rates mark the open terminal
inventory to both $M_T$ and $X_T$ without a liquidation fill and verify
$W-W^X=qG$ within $10^{-10}$ price-lot units. Calibration and policy-burn-in
fills/wealth are excluded.

### 10.3. Estimation and plots

At each threshold the estimand is the arithmetic mean of 20 seed-level renewal rates;
report its SE, Student-$t$ 95% interval and paper-style mean $\pm1$ SE band. The primary
peak is the argmax of this **mean curve**, not the mean of seed-level argmaxes. Exact
ties choose the lowest config policy index and every tied cell is disclosed. The primary
inward-shift estimand is $1-\theta_{grid\ peak}/\theta_D$.

The Figure 4 ordinate divides each seed-level exact rate by deterministic surrogate
$R_D^*=\alpha s_G\sqrt{2/\pi}\exp(-u_D^2/2)$ from the frozen row calibration; raw
rates remain in the table.

Rate loss at a named threshold is one minus its seed-mean rate divided by the maximum
seed-mean grid rate; mean per-seed ratios are prohibited. For peak/loss uncertainty,
resample 20 complete seed policy vectors 10,000 times and recompute the mean curve,
argmax and both ratio-of-means losses in every replicate; report percentile 95%
intervals. This cluster bootstrap preserves the shared-skeleton dependence.

The diagnostic fitted peak is unweighted quadratic OLS on the mean-curve discrete peak
and two grid neighbors on each side. It is `not_available` at a grid edge, for
nonnegative fitted curvature or if the vertex leaves those five cells; it never replaces
the discrete primary. Report inter-fill time, overshoot, open-book fill fraction,
realized-versus-frozen cost difference and both ledger reconciliations for attribution.

At epsilon 0.005 rerun, for the response-row indices selected by epsilon 0.010, the full
multiplier grid plus $\theta^*$ using **physical threshold prices frozen from the primary
calibration table**. Separately recalibrate $s_G$ at 0.005 to measure its relative
resolution shift, but do not move the comparison thresholds. At both resolutions rate
normalization uses the frozen epsilon-0.010 denominator $\alpha s_G^{(.01)}$; dividing
the fine run by $s_G^{(.005)}$ is prohibited because it could hide numerical shift.
Refinement covers $s_G$, every normalized rate, inward shift and both losses. Each
absolute dimensionless shift
must be no more than 0.01 or
$\sqrt{SE_{.01}^2+SE_{.005}^2}$; equal seed labels are not paired.

All 20 seeds, response rows and threshold rows must be reported. Missing calibration
artifacts, fewer than 100 complete intervals or any invariant failure fails the gate.

Paper ranges (about 20% inward shift, 3--4% and 5--6% rate losses) are comparison
targets, not acceptance thresholds. `author_numeric_match_required=false`; any match,
mismatch or sign reversal is retained. The experiment cannot receive exact
`reproduced` status without author settings.

## 11. Run order и stopping rules

Mandatory sequence:

1. Validate all TOML configs without creating result directories.
2. Run `ANA-SMOKE-001`; stop analytical track on failure.
3. Run `ANA-FIG3-001` and publish deterministic table before jump simulation claims.
4. Run simulator unit/property tests and one non-claim smoke path.
5. Run `SIM-MOMENTS-001` at epsilon 0.020, then 0.010 and 0.005.
6. Run `SIM-UNBALANCED-001` as the preregistered one-factor negative control.
7. Proceed to `SIM-FIG4-001` only if pathwise invariants, moment/refinement and control
   gates do not show unresolved simulator bias.

Stop and record a failed run when:

- any pathwise invariant fails;
- any NaN/Inf appears;
- config, code, seed mapping or source data differs from manifest;
- fewer than 20 confirmatory replications complete;
- primary/refinement discrepancy exceeds both numerical tolerance and Monte Carlo
  uncertainty;
- any seed/parity lacks the 200 observations required for the finite-h slope gate;
- target output was inspected before an unregistered change.

Increasing compute for precision is allowed only through amendment retaining old
artifacts. Changing primitives, thresholds, estimator, seed replacement, peak rule or
acceptance after target inspection creates a new experiment ID.

## 12. Artifacts, provenance и reporting

Each run writes immutable
`outputs/<experiment_id>/<run_id>/` containing source config, resolved RunSpec,
manifest, log and raw/summary metrics; tables and plots include machine-readable source
data. Manifest records commit/config hashes, branch/dirty state, Python and dependency
versions, CPU/hardware, runtime, seed-to-stream mapping, resolution and artifact hashes.

Heavy outputs/events remain untracked. Compact factual results, limitations and links
to local artifacts go to `docs/reports/paper-reproduction.md`. Report must separate:

- source fact;
- exact analytical reproduction;
- statistical theorem check;
- independent partial reconstruction;
- interpretation/hypothesis.

No empirical market data, model checkpoints, optimizer/training loop or external API is
used in this protocol.

## 13. Amendments

There are no amendments at registration. After the first target run this body is
immutable. Necessary changes are appended below as dated sections containing reason,
affected experiment IDs, whether target output was already inspected and new IDs when
required. Results never replace or rewrite preregistered choices.
