# Statistical gate audit

- **Дата:** 2026-08-11
- **Status:** completed; gate framework revised; no scientific rerun
- **Scope:** all existing executable gates, blocked Figure 4 contract и planned P5--P9 gates
- **Decision:** [`ADR-0005`](../adr/0005-statistical-decision-gates.md)
- **Protocol:** [`statistical-gates-v1`](../protocols/common/statistical-gates.md)

## Inputs и provenance

Audit не создавал новые Monte Carlo paths и не изменял historical artifacts. Он
использовал immutable seed-level CSV из двух P3 runs:

- `SIM-MOMENTS-001`, run
  `20260811T172240473272Z-060cfab011c3-det`, implementation commit `6aa0e48`;
- `SIM-UNBALANCED-001`, run
  `20260811T172916459381Z-5497c39af2dd-det`, implementation commit `6aa0e48`.

Configs, seeds, estimators, horizons и old decisions не изменялись. Retrospective tests
ниже используют original margins только для диагностики decision rule; это не новая
preregistration и не post-hoc перевод failed run в pass.

Reference audit primitives: commit `c34ab73c275a90aebdafbc8fd11540d3dea9d1b4`,
Python `3.14`, NumPy `2.5.2`, SciPy `1.18.0`, CPU `float64`; GPU не использовался,
gate arithmetic векторизована через NumPy. Representative eight-test gate suite занял
`0.003 s`; GPU/worker parallelism не включались для 20-value vectors и CPU-only SciPy
distribution functions, поскольку это короче scheduling/compilation scale.

## Метод аудита

Для equality metrics рассчитаны Student-$t$ TOST, ordinary two-sided point-null test и
Holm correction внутри пяти-coordinate primary family: normalized mean, variance
identity, flow identity и tight/open drift. Для old refinement margins рассчитан
independent Welch TOST, поскольку одинаковые seed labels между resolutions не имеют
доказанного common-path coupling. Для unbalanced contrast рассчитаны one-sided tests
против zero и original minimum effect `0.05 s^-1`.

`p_difference` отвечает на вопрос «обнаружено ли отличие от точного target», а
`p_equivalence` — «достаточно ли evidence, что отличие меньше margin». Большой
`p_difference` не означает equivalence.

## Existing gate inventory

| Gate class | Existing use | Audit decision |
|---|---|---|
| Deterministic numerical | P2 solver residuals, independent optimizer/checkpoints | Significance `not_applicable`; status P2 unchanged |
| Pathwise/operational | parity, event legality, replay, finite values, provenance | Significance `not_applicable`; fail-fast remains |
| Stochastic equality | P3 mean, variance, flow, drift, ACF | Replace point/compatibility rules by family-adjusted equivalence |
| Directional effect | unbalanced control, future profitability/ablation claims | One-sided superiority over justified minimum effect |
| Numerical refinement | P3/P4 primary--fine comparisons | Replace `tolerance OR SE` by paired/independent equivalence |
| Data sufficiency | minimum observations/intervals | Treat as operational floor plus explicit power, never evidence by itself |
| Descriptive paper match | Figure 4 approximate ranges | Keep descriptive; no forced author-match gate without disclosed settings |

## Retrospective P3 statistical audit

### Primary balanced metrics at epsilon `0.01`

| Metric | Estimate ± SE | Old margin | `p_difference` | Raw `p_equivalence` | Holm `p_equivalence` | Audit status |
|---|---:|---:|---:|---:|---:|---|
| $E[G]/s_G$ | `0.00629 ± 0.00488` | `0.02` | `0.213` | `0.00558` | `0.0217` | equivalent under legacy margin |
| Variance residual | `-0.00372 ± 0.00931` | `0.03` | `0.694` | `0.00543` | `0.0217` | equivalent under legacy margin |
| Open-flow residual | `-0.03338 ± 0.07074` | `0.03` | `0.642` | `0.519` | `0.951` | **inconclusive** |
| Tight drift relative residual | `-0.00732 ± 0.00897` | `0.05` | `0.425` | `6.86e-5` | `3.43e-4` | equivalent under legacy margin |
| Open drift relative residual | `-0.04306 ± 0.11140` | `0.05` | `0.703` | `0.475` | `0.951` | **inconclusive** |

Это показывает обе ошибки heuristic point gates. Flow получил historical fail без
evidence meaningful difference; open drift получил historical central-value pass без
evidence equivalence. ACF point-null p-values на шести lags лежат между `0.319` и
`0.989`, но equivalence margins для ACF не были обоснованы, поэтому корректный status —
`compatible`, не `equivalent`.

### Refinement

При original margin `0.01` и independent-resolution TOST десять из одиннадцати
refinement coordinates не устанавливают equivalence после Holm correction. Только ACF
lag `0.25` имеет adjusted `p_equivalence=0.00496`; следующий лучший lag `0.5` имеет
adjusted `0.408`. Historical rule объявил весь refinement passed, потому что разрешал
заменить tolerance большой difference SE. Этот pass не переносится в новый framework.

### Unbalanced control

Primary contrast равен `0.25734 ± 0.13494 s^-1`:

- sign above zero статистически поддержан: one-sided `p=0.0359`, lower 95% bound
  `0.0240 s^-1`;
- superiority over original minimum effect `0.05 s^-1` не подтверждена:
  `p=0.0705`;
- primary--fine equivalence within `0.02 s^-1` не подтверждена:
  `p_equivalence=0.839`.

Следовательно, planted mechanism виден по знаку и exact generator, но magnitude и
resolution stability остаются inconclusive.

## Pilot power implications

Normal-approximation calculation использует P3 only as pilot variance и original
margins only as a scale illustration. При неизменной per-seed variance и true effect at
the favorable planning point приблизительное число independent seeds для 90% power:

| Gate | Pilot SD | Distance to null/boundary | Approximate seeds |
|---|---:|---:|---:|
| Open-flow equivalence | `0.316` | `0.03` | `953` |
| Open-drift relative equivalence | `0.498` | `0.05` | `851` |
| Unbalanced superiority over `0.05`, assuming true `0.25` | `0.603` | `0.20` | `78` |

Это не authorization на run с 953 seeds. Estimates показывают, что original 20-seed
design заведомо слаб для rare-open metrics и naive seed scaling может быть
неэффективным. До нового config нужно сравнить увеличение horizon, event/compensator
estimators и seed parallelism, не меняя paper model semantics.

## Review blocked and future gates

### `SIM-FIG4-001`

- percentile cluster bootstrap peak/loss intervals — корректная основа uncertainty;
- old `tolerance OR SE` refinement rule отклонён;
- minimum 100 inter-fill intervals остаётся operational floor, но не заменяет power;
- inward-shift claim требует one-sided family-adjusted lower bound above an externally
  justified minimum shift; author approximate ranges остаются descriptive;
- старый config не запускается, новый Figure 4 experiment ID создаётся после P3
  validation и gate-margin sensitivity.

### P5--P9

- P5 eligibility cutoffs требуют uncertainty для occupancy/quality rates и power по
  independent sessions, а не только raw record counts.
- P6 filter comparison формулируется как paired superiority over a minimum useful
  recovery/log-likelihood effect; calibration claims используют equivalence.
- P7 accounting/timestamp/fill legality остаются deterministic/property gates.
- P8 freeze и no-leakage являются operational gates; validation selection использует
  nested folds и declared multiplicity family.
- P9 profitability требует family-adjusted one-sided lower bound выше zero **и** выше
  заранее обоснованного minimum economically relevant net rate. Stress comparisons
  получают non-inferiority margins; controls — paired superiority. Недостаточное число
  sessions даёт `inconclusive`, не profitable/not-profitable binary verdict.

## Итог

P2 deterministic status остаётся `reproduced`. P3 historical status остаётся
`acceptance_failed`, но scientific status — `inconclusive`, теперь по явно показанной
разнице между compatibility и equivalence. P4 остаётся blocked.

Следующий шаг — до новых simulations провести downstream sensitivity: определить,
какое отклонение каждого P3 estimand способно materially изменить Figure 4 peak/rate
claims, на этой основе зафиксировать SESOI/numerical margins, затем сделать power и
compute design на полностью новых confirmatory seeds. Старые outputs используются
только как pilot variance и сохраняются как отрицательное evidence старого gate design.
