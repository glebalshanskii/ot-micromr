# ADR-0004: P3 gate failure, stop перед Figure 4 и граница precision extension

- **Статус:** Accepted
- **Дата:** 2026-08-11
- **Связанный этап:** P3
- **Связанные runs:** `SIM-MOMENTS-001`, `SIM-UNBALANCED-001`
- **Связанный отчёт:** [`p3-controlled-simulation.md`](../reports/p3-controlled-simulation.md)

## Контекст

P3 реализовал controlled numerical simulation и выполнил полный preregistered ladder.
Pathwise, generator, balanced moment/ACF и most refinement checks прошли, однако
balanced open-flow error и два unbalanced-control statistical gates не прошли. Open
parity наблюдается около 0.3% времени; raw open sample count и effective information
оказались существенно слабее tight parity.

После target inspection нельзя менять существующие configs, estimator, acceptance или
model regime. Protocol допускает увеличение compute только через amendment с сохранением
старых artifacts, original seed prefix и scientific parameters.

## Решение

1. P3 фиксируется как `completed; acceptance_failed; inconclusive`. Failed runs не
   перезаписываются и не заменяются более удачными seeds.
2. P4 `SIM-FIG4-001` не запускается: обязательный moments/control gate остаётся
   unresolved.
3. Flow identity не объявляется contradicted. Primary signed residual t-interval
   включает zero, fine-resolution point estimate проходит numeric limit, а miss primary
   gate мал относительно observed SE. Корректный статус — insufficient precision under
   the preregistered design.
4. Unbalanced mechanism подтверждён только на generator level. Realized directional
   contrast имеет ожидаемый positive sign at primary, но confidence lower bound и
   primary--fine stability не проходят; claim status — `inconclusive`.
5. Любое продолжение получает новые experiment IDs и dated protocol amendment **до**
   новых outputs. Разрешено менять только horizon и/или добавлять seeds после original
   ordered prefix на основании explicit power analysis. Primitives, resolutions,
   estimators, thresholds и existing seeds остаются неизменными.
6. Precision extension не запускается автоматически только ради превращения near-miss
   в pass. Сначала документируется required information budget и practical compute/data
   cost; исходный failure остаётся primary historical evidence.

## Последствия

- Project сохраняет научно честный negative/inconclusive result и не переходит к P4 с
  невалидированным rare-state diagnostic.
- Simulator implementation и successful invariant/moment evidence остаются полезными,
  но не превращаются в общий `P3 passed` label.
- Следующий synthetic task — отдельное preregistration/power решение, не code tuning.
- Empirical stage не может ссылаться на текущий P3 как на полностью прошедшую
  validation; при независимом старте этот limitation должен быть явным.
