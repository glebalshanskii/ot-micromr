# P6C continuous-hazard marked filtering

- **Status:** completed; numerical refinement passed, empirical scientific acceptance failed
- **Date:** 2026-08-12
- **Protocol:** [`p6c-continuous-hazard.md`](../protocols/empirical/p6c-continuous-hazard.md)
- **Architecture ADR:** [`ADR-0017`](../adr/0017-continuous-hazard-empirical-filter.md)
- **Result ADR:** [`ADR-0018`](../adr/0018-p6c-continuous-hazard-negative.md)
- **Config:** [`emp_mark_ct_001.toml`](../../cfg/experiments/emp_mark_ct_001.toml)

## Result

P6C replaced the P6M left-frozen event clock consistently in parameter fit, held-out
particle filtering and free-running simulation. Event likelihood is evaluated at the
pre-event endpoint, survival uses path-integrated hazard, and the rollout locates a unit
exponential cumulative-hazard threshold on a dyadic Brownian bridge.

This correction is numerically resolved but does not repair the empirical model. Four and
eight filter substeps are statistically equivalent on all three preregistered numerical
metrics. Nevertheless, posterior uncertainty remains too wide and event time-rescaling is
far above its equivalence region. The several-event forecast still predicts BBO changes about
3.5 times too quickly and does not beat persistence for midpoint. P7/P8 remain blocked.

## Provenance

| Field | Value |
|---|---|
| Run | `EMP-MARK-CT-001/20260812T100151852237Z-c8a620999b93-det` |
| Commit | `6b2306e19eff55bba1d90033301a13b39bc5477a` |
| RunSpec SHA-256 | `c8a620999b932b8d668b2c1e8a1319f849410276f2123f632c8c80e44db618a8` |
| Config SHA-256 | `9069d3195fe34526990eca5536d9cfe97437429a087da18bde9f945376bf2bef` |
| Hardware | NVIDIA GeForce RTX 3080 Ti Laptop GPU |
| Numerics | PyTorch CUDA `float32`; statistics `float64`; `torch.compile(reduce-overhead)` |
| Runtime | `126.13 s` |
| Status | `acceptance_failed`; operationally valid |

The run used the same content-addressed P6 OKX dependency, dates and rolling-origin folds as
P6M. It processed all `1,122,613` healthy transitions, produced 288 valid 30-minute blocks,
made zero future timestamp accesses and replayed December bitwise exactly.

## Method

For an observed interval $[t_i,t_{i+1})$, midpoint and spread remain at their pre-event
values while every latent particle follows Brownian substeps. Its log weight increment is

$$
\Delta\log w_i=
\log\lambda_{m_{i+1}}(M_i-X_{t_{i+1}},S_i)
-\operatorname{trapz}_{u\in[t_i,t_{i+1}]}
\Lambda(M_i-X_u,S_i).
$$

Training uses the corresponding endpoint and integrated-hazard objective on the causal
EWMA proxy path. This is closer to the paper's continuous-time clock but is not joint latent
maximum likelihood; the paper does not provide an empirical estimator for unobserved $X$.

Primary filtering aggregates eight addressed Brownian increments into four substeps.
Refinement uses all eight increments, retaining common endpoints and random stream. This
makes primary/fine differences paired rather than Monte Carlo differences between unrelated
paths.

Free-running simulation draws $E\sim\operatorname{Exp}(1)$ and finds

$$
T=\inf\left\{t:\int_{t_0}^{t}\Lambda(G_u,S_u)du\ge E\right\}.
$$

Each four-second Brownian endpoint is recursively filled to depth eight. Integrated hazard
on each leaf is exact for the piecewise-linear bridge segment, including a zero crossing of
$G$; the event time inside the crossing leaf is solved by 16 bisections. Leaf width is
`0.015625 s`. No threshold reached the 32-second cap.

## Numerical refinement

Paired December 30-minute blocks establish 4/8-substep equivalence:

| Fine minus primary metric | Mean | Simultaneous interval | Margin | Decision |
|---|---:|---:|---:|---|
| Log score, nat/event | `0.0000713` | `[-0.0000748, 0.0002174]` | `±0.005` | equivalent |
| Time rescaling | `-0.0000253` | `[-0.0001622, 0.0001116]` | `±0.05` | equivalent |
| Uncertainty metric | `0.0006159` | `[-0.0010925, 0.0023243]` | `±0.05` | equivalent |

Therefore increasing path quadrature from four to eight substeps cannot plausibly move any
scientific decision at the frozen margins.

## Scientific results

| Estimand | Mean | Adjusted interval/bound | Required | Decision |
|---|---:|---:|---:|---|
| Full minus residual log score | `0.30398 nat/event` | lower `0.26159` | `> 0.01` | superior |
| $1-posterior\ SD/option\ margin$ | `-0.61985` | `[-0.70358,-0.53612]` | `> 0` | inferior |
| Time-rescaling block mean | `2.20883` | `[1.97659,2.44106]` | inside `[0.9,1.1]` | above region |
| Time-rescaling block SD | `5.32993` | `[4.75799,5.90187]` | inside `[0.8,1.2]` | above region |
| Constrained minus unconstrained | `0.001758 nat/event` | lower `0.001246` | `> -0.005` | non-inferior |
| Full minus no-multi-tick | `0.25302 nat/event` | lower `0.22504` | `> 0` | supported |
| Full minus no-multi-spread | `0.07205 nat/event` | lower `0.06117` | `> 0` | supported |

The posterior SD is about `1.620` times the optimistic option-value margin. Removing December
leaves log-score gain positive (`0.30192`) but uncertainty negative (`-0.63090`), so the
decision does not depend on that day.

## P6M comparison

| Metric | Frozen P6M | Continuous P6C | Change |
|---|---:|---:|---:|
| Log-score improvement | `0.30486` | `0.30398` | `-0.00089` |
| Posterior SD / margin | `1.60873` | `1.61985` | `+0.01111` |
| Rescaling mean | `2.20828` | `2.20883` | `+0.00054` |
| Rescaling SD | `5.32846` | `5.32993` | `+0.00147` |

Thus the frozen-hazard approximation was not the cause of the P6M negative result. Within an
observed interval Brownian changes are small relative to the fitted gap scale
(`s_G≈101 USDT` in the December fold), so path integration only weakly changes intensities.

The remaining clock error is structural. The same one-sided gap ramp supplies useful
directional prediction and necessarily raises total activity with $|G|$. Fitting that
directional signal can therefore overstate total event activity; rescaling mean above one is
the direct diagnostic. This interpretation is supported by the close frozen/continuous
comparison but remains a mechanism hypothesis, not a separately identified causal claim.

## Several-event forecast

Descriptive December forecast: first two minutes, 60 healthy origins, 1024 independent paths
per origin, ten future BBO events.

| Metric | P6M frozen | P6C continuous | Persistence/actual |
|---|---:|---:|---:|
| Midpoint MAE, horizon 1 | `1.7951` | `1.7914` | `1.7826 USDT` |
| Midpoint MAE, horizon 10 | `13.0333` | `13.0478` | `12.6651 USDT` |
| Mean time to 10 events | `4.2415 s` | `4.2255 s` | actual `14.7948 s` |

Continuous simulation had zero invalid-transition fallbacks, zero unresolved hazard
thresholds and minimum spread one tick. Actual future BBO and timestamps were used only for
error evaluation.

Artifacts:

- `outputs/P6C-MULTISTEP-VIZ/20241215T000000Z-2min-h10-p1024-v4/multistep-trajectories.png`;
- `outputs/P6C-MULTISTEP-VIZ/20241215T000000Z-2min-h10-p1024-v4/horizon-mae.png`;
- `outputs/P6C-MULTISTEP-VIZ/20241215T000000Z-2min-h10-p1024-v4/horizon-metrics.csv`;
- `outputs/P6C-MULTISTEP-VIZ/20241215T000000Z-2min-h10-p1024-v4/provenance.json`.

## Decision

P6C is complete as an informative negative result. Continuous-time likelihood and simulation
are now close enough to the paper for the tested numerical margins, but the book-only latent
state is still too uncertain and the empirical event clock is miscalibrated. Do not proceed
to trading/P&L using this filter. A next model must separately parameterize total activity
and conditional mark direction, or add a causal state observation; either is a new preregistered
extension rather than another quadrature refinement.
