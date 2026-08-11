# ADR-0005: statistical decision gates вместо point-estimate heuristics

- **Статус:** Accepted
- **Дата:** 2026-08-11
- **Связанные этапы:** P3--P9
- **Связанный protocol:** [`statistical-gates.md`](../protocols/common/statistical-gates.md)
- **Связанный audit:** [`statistical-gate-audit.md`](../reports/statistical-gate-audit.md)

## Контекст

P1 до target runs зафиксировал численные thresholds для P3, но не потребовал для
каждого equality gate доказательства statistical equivalence и мощности. В
`SIM-MOMENTS-001` это привело к двум разным ошибкам decision rule:

- open-flow point estimate не прошёл границу `0.03`, хотя difference-from-zero test
  дал `p=0.642` и данные не свидетельствуют о нарушении identity;
- open-parity drift point estimate прошёл `5%` gate и theoretical target оказался
  внутри широкого interval, хотя equivalence в пределах `5%` не установлена.

Refinement rule `difference <= max(tolerance, SE)` также может разрешить pass именно
из-за большой uncertainty. Отсутствие statistical significance не доказывает
равенство, а увеличение SE не является evidence numerical convergence.

## Решение

1. Ввести единый gate taxonomy и three-way scientific decision:
   `supported/equivalent`, `meaningfully-different/opposite` или `inconclusive`.
   Protocol/data/invariant failures получают отдельный status `invalid`, а не
   интерпретируются как отрицательный научный эффект.
2. Exact deterministic calculations, pathwise invariants и provenance остаются
   deterministic gates. Statistical significance для них `not_applicable`.
3. Stochastic equality claims проверяются TOST/equivalence test с заранее
   обоснованной SESOI margin. Одновременно публикуются estimate, SE, compatibility CI,
   equivalence CI, point-null p-value и equivalence p-value. `target inside 95% CI`
   означает только compatibility и не даёт pass.
4. Directional/superiority claims проходят только если multiplicity-adjusted
   one-sided lower bound выше заранее обоснованного minimum effect. Проверка только
   знака публикуется отдельно и не заменяет practically meaningful boundary.
5. Resolution/refinement сравнивается equivalence test для seed/session-level
   differences. Paired inference допустима только при доказанном common-random-path
   contract; иначе используется independent/cluster bootstrap. Rule `tolerance OR
   uncertainty` запрещён для новых runs.
6. Каждый confirmatory family фиксирует familywise alpha, correction, effect margins,
   independent unit и power до target run. Default project error budget — FWER `0.05`;
   primary stochastic gate планируется на power не ниже `0.90`, secondary — не ниже
   `0.80`. Это declared decision-risk policy, не универсальная математическая истина.
7. Margin нельзя выбирать по близости уже увиденного point estimate к удобной границе.
   Он выводится из downstream claim sensitivity, economic relevance, measurement
   granularity или externally fixed reporting precision. Если такого основания нет,
   metric остаётся descriptive и не получает binary acceptance.
8. `SIM-MOMENTS-001` и `SIM-UNBALANCED-001`, их configs, artifacts и historical
   `acceptance_failed` status неизменны. Retrospective tests служат audit, а не
   post-hoc rescue. Future P3 validation получает новые experiment IDs и новые
   independent seeds; old P3 seeds допустимы только как pilot variance evidence.
9. `SIM-FIG4-001` не запускается в старом виде. До нового Figure 4 run нужен новый
   config с statistical-gates-v1 contract, significance/equivalence rules и power.

## Последствия

- Старый P3 остаётся `completed / historical gate failed / scientifically
  inconclusive`; он не превращается ни в pass, ни в contradiction.
- Простое увеличение числа seeds под старый heuristic gate отменяется. Сначала
  выполняется downstream sensitivity и power design, затем регистрируется новый run.
- P2 deterministic reproduction не меняет status: p-values к floating-point identities
  неприменимы, а numerical tolerances подтверждаются conditioning и independent
  cross-checks.
- Future empirical profitability требует не только lower bound above zero, но также
  predeclared minimum economically relevant effect, power и multiplicity control;
  stress/control comparisons получают superiority или non-inferiority formulations.
- Бинарный общий `acceptance_passed` может существовать как operational summary, но
  machine-readable artifacts обязаны хранить per-gate class, adjusted p-values,
  intervals, margins, power status и three-way decision.

## Отклонённые варианты

1. **Добавить только `p < 0.05` к старым thresholds.** Отклонено: point-null test не
   доказывает equivalence и поощряет вывод `no effect` при low power.
2. **Считать pass, когда theory лежит внутри 95% CI.** Отклонено: сколь угодно широкий
   interval почти гарантирует такой pass.
3. **Ослабить `3%` до наблюдавшихся `3.34%`.** Отклонено как post-hoc tuning.
4. **Автоматически увеличить P3 до примерно нужного числа seeds.** Отклонено до
   sensitivity/compute analysis; pilot variance показывает, что naive seed scaling
   может потребовать сотни replications и быть хуже увеличения horizon или улучшения
   estimator.
