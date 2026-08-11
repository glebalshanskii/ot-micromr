# Statistical gates protocol

- **Статус:** active policy for future confirmatory runs
- **Version:** `statistical-gates-v1`
- **Дата:** 2026-08-11
- **Scope:** synthetic и empirical stochastic experiments после P3 historical runs
- **Decision:** [`ADR-0005`](../../adr/0005-statistical-decision-gates.md)

## 1. Назначение

Protocol отделяет четыре вопроса, которые нельзя сводить к одному heuristic threshold:

1. корректен ли code/path/data contract;
2. контролируется ли numerical approximation;
3. совместимы ли stochastic observations с target;
4. достаточно ли точны данные, чтобы подтвердить practically meaningful equality или
   superiority.

Statistical significance не устраняет необходимость effect-size margin. Уровень
значимости задаёт допустимый риск false positive, margin задаёт, какое отличие имеет
научное или экономическое значение, а power контролирует риск получить
`inconclusive` при эффекте запланированного размера.

## 2. Обязательная запись gate

Каждый future stochastic gate до target output фиксирует:

- `gate_id`, claim и gate class;
- estimand, target и physical/dimensionless units;
- independent replication/cluster unit и aggregation;
- estimator и resampling/test method;
- alternative direction;
- equivalence margin или minimum effect и отдельное обоснование этой величины;
- family ID, familywise alpha и multiplicity method;
- target power, variance source, planned sample size и precision stop rule;
- missing/invalid-cluster policy;
- `supported`, `meaningfully_different`, `inconclusive` и `invalid` decision rules;
- все raw estimates, intervals, p-values и adjusted p-values, сохраняемые в artifacts.

Gate без обоснованного margin/MDE не является confirmatory: он остаётся descriptive.
Pilot outputs разрешено использовать для variance/power planning, но confirmatory
estimate строится на новых independent seeds/sessions/holdout.

## 3. Gate classes

### 3.1. Deterministic и operational

К этому class относятся algebraic identities, solver residuals, pathwise state
invariants, deterministic replay, timestamp/no-look-ahead checks, ledger reconciliation,
artifact completeness, license и schema validation. Они используют exact comparison,
floating-point error bound, property test или zero-violation stop. `p_value`, power и
statistical significance для них равны `not_applicable`.

Нулевой observed violation count сам по себе не доказывает probabilistic reliability
за пределами проверенного domain. Если заявляется rate of failures, для него создаётся
отдельный stochastic gate с exposure unit и binomial/cluster interval.

### 3.2. Equality/equivalence

Для target $\theta_0$ и symmetric SESOI margin $\Delta>0$ проверяются две hypotheses:

$$
H_{01}:\theta-\theta_0\le-\Delta,
\qquad
H_{02}:\theta-\theta_0\ge\Delta.
$$

TOST pass требует отклонить обе null hypotheses. Для одного Student-$t$ estimand это
эквивалентно попаданию $(1-2\alpha)$ CI внутрь
$[\theta_0-\Delta,\theta_0+\Delta]$. Metric-level equivalence p-value равен
$\max(p_{lower},p_{upper})$; затем применяется declared family correction.

Одновременно публикуется ordinary two-sided $(1-\alpha)$ compatibility CI и p-value
для point null $\theta=\theta_0$. Decisions:

- `equivalent`: adjusted TOST passes;
- `meaningfully_different`: compatibility CI целиком за equivalence region;
- `inconclusive`: всё остальное, включая target внутри слишком широкого CI.

Asymmetric margins допустимы, если downside/upside имеют различную предметную цену.

### 3.3. Directional superiority/non-inferiority

Для minimum meaningful effect $\Delta_{min}$ superiority claim использует

$$
H_0:\theta\le\Delta_{min},
\qquad
H_1:\theta>\Delta_{min}.
$$

`supported` требует, чтобы multiplicity-adjusted one-sided lower confidence bound был
выше $\Delta_{min}$. Отдельно публикуется test против zero, чтобы различать
statistically detectable sign и practically meaningful magnitude. Opposite direction
и non-inferiority формулируются симметрично; отсутствие superiority даёт
`inconclusive`, а не доказательство отсутствия эффекта.

### 3.4. Numerical refinement

Estimand — difference primary minus fine/reference resolution. При доказанном common
random path используются seed-level paired differences; одинаковый integer seed без
path coupling не считается paired design. В остальных случаях используются Welch,
independent cluster bootstrap или другой preregistered dependence-aware estimator.

Refinement проходит только как equivalence внутри numerical margin. Margin должен быть
строже scientific margin и обоснован budget-ом downstream distortion. Условие
`absolute difference <= max(tolerance, SE)` запрещено: большая SE означает low
precision и приводит к `inconclusive`.

## 4. Multiplicity

Каждая primary claim family объявляется до target run. Default — Holm FWER correction
при familywise $\alpha=0.05`; для correlated seed/session vectors предпочтителен
preregistered max-$t$ или simultaneous cluster bootstrap. Нельзя объявлять отдельными
families метрики, разделённые только после просмотра результатов.

Для TOST сначала вычисляется metric-level $p_{equiv}=\max(p_{lower},p_{upper})$, затем
корректируется vector этих values. Descriptive diagnostics публикуются полностью, но
не влияют на confirmatory pass.

## 5. Power и precision

- Primary stochastic gates планируются на power не ниже `0.90`; secondary — не ниже
  `0.80`, если protocol не обосновывает иной error budget.
- Variance берётся из external study, disjoint pilot или conservative bound. Target
  data не используется для изменения margin/sample size без нового independent test.
- Sample size выбирается по наиболее demanding required gate, а не по удобной primary
  metric. Для rare states отдельно моделируются occupancy, cluster dependence и число
  effective transitions.
- Fixed-sample design является default. Sequential design допустим только с заранее
  заданными looks, alpha spending и maximum sample size.
- Если planned sample недоступен по compute/data, claim заранее понижается до
  exploratory либо stage получает `blocked-precision`; нельзя сохранять binary gate с
  заведомо низкой power.

## 6. Stage-level aggregation

Stage status не является простым `all(point_estimate < threshold)`:

- `supported`: все required deterministic gates valid и все required stochastic gates
  supported/equivalent;
- `meaningfully-different` или `negative`: хотя бы один preregistered scientific gate
  подтверждает practically meaningful opposite result;
- `inconclusive`: evidence недостаточно точна, но meaningful contradiction не доказана;
- `invalid`: нарушен data/code/protocol contract.

Operational completion хранится отдельно: честно выполненный experiment может быть
`completed / inconclusive` или `completed / negative`.

## 7. Machine-readable implementation

Reference implementation находится в `ot_micromr.statistical_gates` и предоставляет
Student-$t$ TOST, one-sided superiority, paired/independent refinement equivalence,
Holm adjustment и normal-approximation power planning. Это primitives, а не hidden
defaults: future RunSpec обязан явно передать target, margin, alpha, family и power.

Bootstrap/max-$t$ gates должны сохранять complete cluster vectors и resampling seed.
Любой optimized/vectorized/parallel implementation проверяется против reference
implementation и не меняет cluster, seed или decision semantics.

## References

- Schuirmann, D. J. (1987), *A comparison of the Two One-Sided Tests Procedure and
  the Power Approach for assessing the equivalence of average bioavailability*,
  [`doi:10.1007/BF01068419`](https://doi.org/10.1007/BF01068419).
- Holm, S. (1979), *A Simple Sequentially Rejective Multiple Test Procedure*,
  [`doi:10.2307/4615733`](https://doi.org/10.2307/4615733).
- Lakens, D. (2017), *Equivalence Tests: A Practical Primer for t Tests,
  Correlations, and Meta-Analyses*,
  [`doi:10.1177/1948550617697177`](https://doi.org/10.1177/1948550617697177).
