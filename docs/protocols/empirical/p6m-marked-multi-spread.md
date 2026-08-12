# P6M marked multi-spread causal filtering protocol

- **Статус:** preregistered before `FILTER-MARK-SYN-001` and
  `EMP-MARK-FILTER-001` target outputs
- **Дата freeze:** 2026-08-12
- **Decision:** [`ADR-0015`](../../adr/0015-marked-multi-spread-causal-model.md)
- **Paper:** `arXiv:2608.00885v1`
- **P6 dependency:** `EMP-FILTER-001/20260812T000514761846Z-7075bc32601b-det`
- **Scope:** causal filtering only; orders, thresholds, fills and P&L are forbidden

## 1. Motivation and prior observation disclosure

The P6 implementation passed known-$X$ synthetic recovery but failed empirical usability.
On `2024-12-15`, only `7.7319%` of BBO transitions were exact paper channels;
`73.9666%` were multi-tick tight translations and `15.5492%` touched spread wider than
two ticks. These observations motivated the present model and are not hidden pilot data.

All existing P5/P6 dates, including `2024-12-15`, remain usable. P6M freezes the model,
folds and decisions before its own outputs, uses rolling-origin held-out blocks, and reports
a sensitivity aggregate excluding December. It does not call the reused days a new untouched
market sample. A distinct period remains reserved for future P9 profitability evaluation.

## 2. Observed state and fixed mark alphabet

For integer bid/ask ticks $b_t,a_t$ and tick size $\delta$:

$$
D_t=a_t-b_t,\qquad M_t=\frac{\delta(a_t+b_t)}{2},\qquad
G_t=M_t-X_t,
$$

$$
r_t=(\Delta b_t,\Delta a_t),\qquad
y_r=\Delta b_t+\Delta a_t,\qquad
d_r=\Delta a_t-\Delta b_t,\qquad
J_r=\frac{\delta y_r}{2}.
$$

Exact $r_t$, $y_r$, $d_r$ and both endpoint spreads are retained in artifacts. The modeled
observation is the following fixed coarsening:

- previous-spread bucket: `1,2,...,7,8+`;
- price direction: `down`, `zero`, `up` from $\operatorname{sign}(y_r)$;
- spread family: `narrow`, `same`, `widen` from $\operatorname{sign}(d_r)$;
- absolute midpoint half-tick bucket for $|y_r|$: `0`, `1`, `2-3`, `4-7`, `8-15`,
  `16-31`, `32-63`, `64-127`, `128+`;
- absolute spread-change bucket for $|d_r|$: the same nine buckets.

The Cartesian alphabet has `729` marks per previous-spread bucket. Impossible/unseen cells
remain in the alphabet and receive train-only Dirichlet smoothing `beta=0.01`. Every healthy
non-snapshot BBO transition therefore has finite held-out probability. Snapshot/recovery
boundaries start a new filter segment and are not scored as exchange events.

Paper slides/opens/closes are exact subsets of this schema. Coverage of the coarsened
alphabet is an operational invariant, not scientific evidence.

## 3. Marked point-process models

Let $p_{D,r}$ be a train-only mark probability and $\nu_D$ a baseline total event rate.
The gap-independent proper-score baseline is

$$
\lambda^0_r(D)=\nu_Dp^{raw}_{D,r}.
$$

It is the old six-event gap channel plus a full-support gap-independent residual in the
sense that exact paper cells are preserved, while every other mark is assigned empirical
residual mass. It replaces the invalid comparison against zero probability.

For the theory-constrained marked model, up/down counts with identical spread family and
magnitude buckets are symmetrized in $p^{sym}$. For each spread and direction, define

$$
\kappa_{D,r}=
\frac{p^{sym}_{D,r}}
{\sum_{r':\,\operatorname{sign}(J_{r'})=\operatorname{sign}(J_r)}
|J_{r'}|p^{sym}_{D,r'}}.
$$

The primary intensity is

$$
\lambda_r(G,D)=\nu_Dp^{sym}_{D,r}+
\alpha\kappa_{D,r}[-\operatorname{sign}(J_r)G]^+.
$$

Thus baseline price drift is zero and both directional corrective first moments equal
$\alpha$, giving $E[dG_t\mid G_t,D_t]=-\alpha G_tdt$. Zero-midpoint marks receive no
gap correction. Positive parameters use softplus; $(\nu_D,\alpha)$ are fit by vectorized
Poisson point-process NLL including event and survival terms.

The unconstrained diagnostic uses raw directional $p^{raw}$ with separately normalized
up/down correction distributions. It may have nonzero baseline drift. It is not eligible
to generate the primary downstream signal.

One-factor correction masks are frozen:

- `no_multi_tick`: gap correction disabled for same-spread translations with
  $|y_r|>2$;
- `no_multi_spread`: gap correction disabled whenever $D_{t-}>2$ or $D_t>2$;
- `full`: all nonzero-midpoint marks may carry correction.

Disabled marks remain in the gap-independent residual support. Correction weights are
renormalized to retain unit directional first moment wherever a direction has eligible
marks.

## 4. Causal reference, parameter fit and particle filter

The fit-only proxy for every swap day is a past-only irregular EWMA of displayed midpoint
with time constant `300 s`. It is used to estimate point-process parameters, $s_G$ and
$\sigma_X$, not as a particle-filter measurement. Same-venue OKX spot remains a causal
as-of diagnostic only on the three available paired days.

For each held-out fold, all transformations, mark tables, smoothing, parameters and reduced
moments are fit only on earlier dates. The particle state propagates as

$$
X_{t+\Delta t}=X_t+\sigma_X\sqrt{\Delta t}\epsilon.
$$

For each particle the event update is

$$
\log L_i=\log\lambda_{r_t}(G_i,D_{t-})
-\Lambda(G_i,D_{t-})\Delta t.
$$

Systematic resampling occurs at fixed chunk boundaries. Snapshot segments initialize
$X\sim N(M,s_G^2)$. Multi-tick and wide-spread transitions never reinitialize the filter.
The predictive log score is the particle-integrated normalizer increment, not a plug-in
score at the fitted EWMA proxy.

## 5. Frozen rolling-origin evaluation

Dates are ordered:

`2024-01-15, 2024-03-15, 2024-05-15, 2024-07-15, 2024-09-15,
2024-11-15, 2024-12-15`.

Six folds fit on all preceding dates and evaluate only the next date. Each held-out day is
split into 48 nonoverlapping `30 min` UTC blocks; blocks with no valid scored interval are
reported and excluded only because the estimand is undefined. Planned maximum is 288 blocks.
Thirty minutes is much longer than the seconds-scale mechanism and is the independent
inference unit; day-level estimates and the aggregate excluding December are mandatory
sensitivity diagnostics.

No filter/model choice uses strategy P&L. No fold is revisited after seeing its target
metrics. Fixed computation is one synthetic run and one empirical rolling-origin run.

## 6. Synthetic experiment and gates

`FILTER-MARK-SYN-001` uses project-chosen, not author-reported parameters:

- $\delta=1$, $\sigma_X=1$, $\alpha=1$, spreads `1..4`;
- baseline total rates `(8,10,12,14) /s`;
- translation sizes `(1,2,4)` ticks with weights `(0.60,0.25,0.10)`;
- widening base weight `0.08`, narrowing weight per available tick `0.12`, maximum
  spread change `2` ticks;
- originally `dt=0.005 s`, burn-in `30 s`, measurement `60 s`, 64 sessions, 256
  particles; the dated amendment below refines only `dt` and steps per physical chunk;
- CUDA `float32`, PyTorch `float64` final statistics, `torch.compile` reduce-overhead.

Primary family `P6M-SYN-PRIMARY-001` has two paired session metrics and Bonferroni
familywise alpha `0.05` (`0.025` per metric):

1. $1-RMSE_{PF}/RMSE_{mid}$, minimum useful effect `0.10`;
2. PF minus current-mid plug-in predictive log score, minimum `0.01 nat/event`.

The 64-session fixed design exceeds the prior P6 planned minimum 43 sessions. It reuses
the same downstream-derived margins because P6M supplies the same latent signal to the
same future threshold decision.

Calibration is a separate TOST family: 90% posterior coverage target `0.90`, margin
`0.05`, alpha `0.05`. Deterministic gates require exact mark arithmetic, positive spread,
finite positive intensities, baseline drift absolute error `<1e-6`, directional corrective
moment absolute error `<1e-6`, maximum one-step event probability `<0.10`, exact replay and
zero oracle use before metric evaluation.

Empirical P6M is not run unless this synthetic stage passes.

### Amendment 2026-08-12 after first synthetic run

The first target run
`FILTER-MARK-SYN-001/20260812T010035485774Z-737bceddd9b4-det` passed state recovery,
predictive score, posterior calibration, drift, replay and wall-time gates, but failed two
operational checks. Maximum one-step event probability was `0.12798 > 0.10`, so the
frozen-left discretization was too coarse. The mark-arithmetic check also exposed a CUDA
boundary bug: floating `log2/floor` mapped exact powers of two into the previous bucket;
CPU did not reproduce the error.

Before any empirical run, `dt` is refined from `0.005 s` to `0.0025 s`. Chunk and fixed
resampling interval increase from 50 to 100 steps, preserving the same `0.25 s` physical
interval, burn-in, horizon, sessions, particles and all scientific parameters/gates. Bucket
assignment is changed to exact vectorized integer threshold comparisons at
`1,2,4,...,128`; no mark definition or acceptance threshold changes. The failed run remains
immutable and a new synthetic run is required. Consequently, the active synthetic config
source hash and the empirical dependency hash are updated before that new run.

## 7. Empirical statistical decisions

### 7.1 Primary usability family

Two one-sided block-level metrics use Bonferroni familywise alpha `0.05`:

1. particle-integrated full-model log score minus gap-independent full-support baseline,
   minimum useful effect `0.01 nat/event`;
2. $1-\operatorname{median}(posterior\ SD)/option\ margin$, minimum `0`.

Planning alternatives are `0.04 nat/event` and `0.25`; conservative planning block SDs
are `0.10` and `0.50`. Normal planning at per-metric alpha `0.025`, power `0.90`, requires
at most 117 blocks, below the planned 288. Actual inference uses Student-$t$ block vectors;
if fewer than 117 valid blocks remain, status is `blocked-precision`, not failed.

Both lower confidence bounds must exceed their minimum. The second condition is equivalent
to an upper bound below one for posterior uncertainty divided by the optimistic Dawson
option-value margin. Fees/latency are absent, so failure is conservative against trading.

### 7.2 Calibration family

For each block, posterior-prior expected total intensity produces event time-rescaling
values. Two TOST metrics use Bonferroni familywise alpha `0.05`:

- block mean rescaling target `1`, margin `0.10`;
- block rescaling standard deviation target `1`, margin `0.20`.

Planning SDs `0.30` and `0.50` require fewer than 117 blocks at 90% power. Failure to reject
difference is not equivalence. Categorical mark calibration is reported through block
proper scores and family-frequency residual tables; it is descriptive because a second
numeric calibration margin would duplicate the primary proper-score decision.

### 7.3 Model-integrity and explanatory families

- Theory-constrained minus unconstrained paired log score is tested for non-inferiority
  with margin `0.005 nat/event`, half the primary useful improvement, one-sided alpha
  `0.05`.
- Full minus `no_multi_tick` and full minus `no_multi_spread` block scores form a Holm
  family at alpha `0.05`. Their minimum effect is zero because these are mechanism
  attribution claims, not downstream usability gates. Unsupported components receive
  `partial`; they do not erase an otherwise useful full-model result.
- Primary score and uncertainty signs must remain positive after excluding December.
  This is sensitivity evidence that the decision is not numerically carried by the day
  that motivated the model; it does not prohibit use of that day.

### 7.4 Operational gates and stage decision

Operational gates: dependency hashes, full healthy transition support, exact tensor/grid
invariants, no future timestamp access, finite positive state, frozen fold parameters,
deterministic replay and wall time below `300 s` synthetic / `600 s` empirical.

Stage `supported` requires synthetic pass, operational pass, primary usability pass,
calibration equivalence and theory-model non-inferiority. `negative` means a powered
opposite/practically inadequate result; `inconclusive` means insufficient precision;
`invalid` is reserved for contract/code/data failure. Only `supported` unlocks P7.

## 8. Artifacts

Synthetic: session metrics, inference table, replay, mark table and summary.

Empirical: dependency audit, fold/block scores, fold parameters, mark/family diagnostics,
calibration vectors, per-day state summary, December-exclusion sensitivity, replay and one
compact December filter-state tensor. Raw archives and large intermediate tensors remain
untracked.
