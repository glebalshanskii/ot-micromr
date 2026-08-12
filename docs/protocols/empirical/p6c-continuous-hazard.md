# P6C continuous-hazard marked filtering protocol

- **Status:** preregistered before target run
- **Date:** 2026-08-12
- **Config:** [`emp_mark_ct_001.toml`](../../../cfg/experiments/emp_mark_ct_001.toml)
- **Predecessor:** [`P6M`](p6m-marked-multi-spread.md)

## Motivation and hypothesis

P6M used every observed interval duration but froze event intensities at its left
endpoint. It therefore scored an event with

$$
\log \lambda_m(G_{t_i},S_{t_i})-\Lambda(G_{t_i},S_{t_i})\Delta t_i.
$$

The paper defines a continuous-time point process. Between book events $M$ and $S$
are fixed while $X$ is Brownian, so $G=M-X$ and the hazard continue to move. P6C tests
whether the P6M calibration and latent-state failure survive the closer likelihood

$$
\log \lambda_m(G_{t_{i+1}^-},S_{t_i})
-\int_{t_i}^{t_{i+1}}\Lambda(G_u,S_{t_i})\,du.
$$

The hypothesis is that endpoint scoring and survival-conditioned Brownian paths materially
alter fitted rates, time-rescaling and posterior uncertainty. No direction of improvement is
assumed. In particular, continuous hazard may increase rather than reduce event activity.

## Frozen scope and data

The marked state space, rolling-origin dates, OKX BTC-USDT-SWAP processed BBO dependency,
causal spot diagnostic, baselines, scientific margins and no-P&L boundary are unchanged from
P6M. This is a numerical/likelihood correction, not a new dataset search or strategy stage.

The paper assumes observable $G$. Empirical P6C remains a project extension because it must
filter latent $X$. Its causal EWMA proxy is used only for parameter fitting; held-out particle
updates never read future BBO or spot values.

## Continuous-hazard fit

For training interval $i$, the pre-event midpoint remains $M_i$. The causal proxy supplies
start and pre-event endpoint gaps. Primary fit uses four-node-interval trapezoidal quadrature
along their linear Brownian-bridge mean path. Event intensity is evaluated at the endpoint.
This removes the left-endpoint error but is still a proxy fit rather than joint marginal
maximum likelihood over $X$; that limitation must remain explicit in the report.

## Continuous-hazard particle filter

Each particle interval consumes eight deterministic-address Brownian normals. Primary
quadrature aggregates them into four increments; refinement uses all eight, so both levels
share the same Brownian endpoint proposal. For primary nodes $u_{ij}$,

$$
H_i^{(4)}=\operatorname{trapz}
\left(\Lambda(M_i-X_{u_{ij}},S_i),u_{ij}\right),
$$

and particle weights update as

$$
\log w_{i+1}=\log w_i+\log\lambda_{m_{i+1}}
(M_i-X_{t_{i+1}},S_i)-H_i^{(4)}.
$$

The expected time-rescaling diagnostic is the prior-particle mean of $H_i$, not
$\Lambda(G_{t_i})\Delta t_i$. Brownian $X$ stays continuous at the event; only observed
$M,S$ jump afterward.

## Numerical refinement

The December full-model filter is rerun with eight substeps and the same seed/addressed
Brownian increments. Paired 30-minute block differences fine-minus-primary must establish
equivalence to zero simultaneously for:

| Quantity | Equivalence margin | Per-test alpha |
|---|---:|---:|
| log score | `0.005 nat/event` | `0.05/3` |
| time rescaling | `0.05` | `0.05/3` |
| uncertainty metric | `0.05` | `0.05/3` |

These margins are tied to at most half of the corresponding scientific decision scale.
Failure is a numerical-resolution failure, not evidence for or against the economic model.

## Scientific decisions

P6C retains the P6M powered block-level families:

1. full-minus-residual log score must be superior to `0.01 nat/event`;
2. $1-\operatorname{median}(SD[X\mid\mathcal F_t])/$ optimistic option margin must be
   superior to zero;
3. time-rescaling block mean and SD must be equivalent to `1` within `0.10` and `0.20`;
4. constrained model must remain non-inferior to unconstrained;
5. multi-tick and multi-spread components retain Holm-adjusted attribution;
6. the December-exclusion sensitivity and all operational checks remain required.

Only simultaneous scientific and numerical success can unblock P7. A better event clock
without usable latent-state uncertainty remains a negative result.

## Free-running rollout

After the immutable target run, descriptive several-event forecasts use the refitted model.
For each event a unit exponential threshold $E$ is drawn and the simulator locates

$$
T=\inf\left\{t:\int_{t_0}^{t}\Lambda(G_u,S_u)\,du\ge E\right\}.
$$

The crossing cell is refined with Brownian-bridge bisection; it is not snapped to the
numerical grid. Actual future BBO/timestamps are used only for scoring.

## Stop rule

Do not weaken margins or tune substeps after observing the target. If refinement fails,
increase resolution in a dated amendment and rerun. If refinement passes but scientific
families fail, retain P6C as a continuous-hazard negative result and keep P7/P8 blocked.
