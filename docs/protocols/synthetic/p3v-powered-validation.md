# P3V: sensitivity-informed powered validation

Status: **preregistered before `SIM-*002` target runs**

Date: 2026-08-11

Parent protocol: [`paper-reproduction.md`](paper-reproduction.md)

Statistical policy: [`statistical-gates-v1`](../common/statistical-gates.md)

## 1. Question and scope

P3V decides whether the controlled jump simulator is precise enough to support the
Figure 4 band experiment. It does not reproduce Figure 4 and does not tune a trading
strategy. The historical `SIM-*001` outputs are pilot data only and retain their failed /
inconclusive status.

Two kinds of error are kept separate:

1. **Estimator uncertainty**: the stationary open state occurs only about `0.35%` of the
   historical measured time. A noisy estimate of open flow or drift is not itself a model
   perturbation and cannot be inserted into Figure 4 as if it were a primitive.
2. **Material simulator error**: a biased event sampler or a wrong parity drift can change
   the law of $(G,S)$ and therefore band passage times, fill overshoots and rates.

The bridge between validation and the downstream decision is the set of quantities used by
Figure 4: selected peak on a `0.05 theta_D` grid, loss at `theta_D`, loss at `theta*`, fill
rate and open-fill share. Until the continuous-crossing Figure 4 monitor exists, P3V uses
the following conservative decision resolution:

- one threshold-grid cell (`0.05 theta_D`) is the smallest selectable peak change;
- one percentage point of peak-normalised rate is the smallest change that can alter the
  paper comparison (the reported losses start at roughly three percent);
- an error smaller than both resolutions is operationally immaterial for P4, but a margin
  is accepted only when a documented sensitivity or bound maps it to those outputs.

No significance test can choose these scientific resolutions. Statistical tests determine
whether the data are precise enough relative to them.

## 2. Pilot evidence and estimator choice

The already inspected `SIM-*001` seed rows may be used to estimate variance and
time-scaling, never as confirmatory observations. The historical standard deviations at
`epsilon=0.01`, horizon `2000`, are approximately:

| Estimand | Pilot SD |
|---|---:|
| conditional-flow residual | `0.3164` |
| open finite-h drift slope | `0.4957 / second` |
| unbalanced finite-h contrast | `0.6035 / second` |

P3V adds, without replacing the historical metrics:

- integrated nominal opening hazard $A_o$ and closing hazard $A_c$;
- transition counts $N_o,N_c$ and channel-wise compensators $N_j-A_j$;
- integrated-hazard flow residual

  $$r_F = \frac{2(A_c-A_o)}{A_c+A_o};$$

- realised flow conservation $2(N_c-N_o)/(N_c+N_o)$ as a deterministic/pathwise
  diagnostic, not a stochastic acceptance claim;
- the existing finite-h drift slopes plus exact generator slopes.

The integrated estimator avoids a ratio of separately estimated rare conditional means.
The exact generator check establishes the intended local drift algebraically; the finite-h
estimate remains necessary to test the realised event sampler.

## 3. Sensitivity and margins

Before interpreting a confirmatory result, the implementation report must provide either a
fault-injection sensitivity curve or a conservative bound from each primary validation
estimand to the downstream resolutions above. If a mapping is not identifiable, that
estimand cannot receive a Figure-4-derived equivalence margin and remains a diagnostic.

For the first powered run the primary scientific family is deliberately limited to the two
historical precision blockers and their negative control:

1. balanced integrated-hazard flow residual: equivalence to zero within `0.10`;
2. balanced open finite-h drift slope divided by its exact finite-h target: equivalence to
   one within `0.15`;
3. unbalanced finite-h parity contrast: superiority over `0.10 / second`.

The flow margin is a sampler-error scale, not a tolerated economic model imbalance. The
open-drift margin is conditioned on the exact-generator residual remaining below `1e-12`.
Both are provisional for P4: a sensitivity result showing that either margin can move the
selected peak by a grid cell or a normalised rate by one percentage point invalidates the
margin and blocks P4.

The numerical-refinement family compares `epsilon=0.01` with `0.005`, seed-paired by
label but tested from the empirical paired differences:

- integrated-hazard flow residual margin `0.10`;
- open drift target-ratio margin `0.15`;
- unbalanced contrast margin `0.15 / second`.

## 4. Confirmatory design

Experiments: `SIM-MOMENTS-002` and `SIM-UNBALANCED-002`.

- 20 new positive integer seeds, disjoint from all `SIM-*001` seeds;
- burn-in `100` reference reversion times;
- measured horizon `20000` reference reversion times;
- observation interval `0.01` reference reversion times;
- resolutions `epsilon in {0.01, 0.005}`;
- full deterministic replay for three preregistered primary seeds; historical full replay
  plus unit tests already cover the invariant;
- raw event logging disabled because channel counts, hazards, compensators, digests and seed
  metrics are sufficient for this gate; a header-only event-log artifact remains present;
- no optional stopping and no sample-size extension after target inspection.

The horizon follows the conservative square-root information scaling of the historical
seed SDs. At familywise alpha `0.05`, Holm worst-case local alpha `0.05/3`, power target
`0.90`, the design targets at least 20 seeds for the hardest contrast using theoretical
effect about `0.249 / second`. Exact achieved/predicted power is reported, not assumed.

## 5. Inference and decision

The independent seed is the unit of inference. Raw primary p-values are:

- TOST p-value for items 1 and 2;
- one-sided superiority p-value for item 3.

Holm adjustment is applied jointly across the three experiments/estimands. The primary
family passes only if all three adjusted decisions are supported. The refinement family is
Holm-adjusted separately. Deterministic invariants must all pass; they do not receive
p-values.

Every stochastic result has one of `supported`, `meaningfully_different`, or
`inconclusive`. Failure to establish equivalence is not evidence of difference. If the
primary or refinement family is not supported, P4 remains `blocked-precision` or
`blocked-bias` according to the compatibility interval.

## 6. Compute protocol

Independent `(epsilon, seed)` paths are process-parallel. Candidate worker counts are
benchmarked on the same pilot workload with numerical outputs compared bitwise; the chosen
count cannot exceed the machine's 20 logical CPUs. NumPy post-processing stays vectorised.
BLAS/OpenMP worker thread counts are fixed to one to avoid nested oversubscription.

The path evolution is branch-heavy, stateful and sequential, while the confirmatory batch
has only 20 paths. PyTorch is not a project dependency. A GPU/`torch.compile` backend is
added only if a representative benchmark, including transfers and compilation, beats the
selected CPU process backend without changing `float64` semantics or RNG mapping. Absence
of that evidence is recorded as a measured/reasoned rejection, not as a GPU result.

## 7. Stop and transition rule

- Passing P3V authorises implementation and execution of `SIM-FIG4-002`; it does not make
  any Figure 4 claim pass automatically.
- A failed deterministic invariant stops immediately.
- An inconclusive powered run is retained and P4 stays blocked; no extra seeds are appended.
- A material bias triggers a new simulator ADR and experiment ID.

## 8. Amendment 2026-08-11: sensitivity result before target runs

No `SIM-*002` target output had been generated when this amendment was made. A pilot-only
fault-injection run used all 20 historical P3 seed labels, horizon `10000`, endpoint-only
band monitoring and 5000 paired seed bootstraps. Raw local artifacts are
`outputs/p3v/pilot-sensitivity.json` and `pilot-sensitivity-curves.csv`.

The preregistered provisional margins were too loose:

- planted flow-sampler residuals near `+/-0.10` had paired 95% intervals for the change in
  peak-normalised rate extending to approximately `-0.89%..+1.26%`;
- a planted `+15%` open-drift change had an interval approximately
  `-2.14%..-0.34%` for that rate change.

Therefore `0.10/0.15` are not accepted as P4-safe equivalence margins. The revised design
is:

1. integrated-hazard flow equivalence margin `0.05`;
2. exact open/tight generator residual remains a deterministic gate at `1e-12`; it gets no
   fictitious p-value. The realised open finite-h and jump-compensator slopes are reported
   with compatibility intervals, not used to re-prove an algebraic identity;
3. the stochastic unbalanced control uses the realised jump-compensator parity contrast and
   must be superior to `0.10 / second`;
4. the primary Holm family has these two stochastic p-values: flow equivalence and control
   superiority;
5. the refinement Holm family contains flow paired equivalence within `0.05` and control
   contrast paired equivalence within `0.15 / second`;
6. `SIM-MOMENTS-002` measured horizon increases to `40000`; the control remains at `20000`.
7. Per-step quantile arrays are disabled in `SIM-*002`: they are not gate inputs and would
   require memory linear in tens of millions of adaptive steps. Maximum event probability,
   step count, hazards and invariant counters remain exact streaming diagnostics.

Using the horizon-10000 pilot SD and worst-case Holm local alpha `0.025`, normal power
approximations require about 18 seeds for flow at horizon `20000`; doubling the balanced
horizon provides margin for refinement. The control requires fewer than 20 seeds at horizon
`20000`. Confirmatory seeds and target outputs remain untouched.
