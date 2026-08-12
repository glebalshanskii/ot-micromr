# ADR-0015: Marked multi-spread causal model direction

- **Статус:** Accepted
- **Дата:** 2026-08-12
- **Связанный этап:** P6M
- **Зависимость:** P6 `EMP-FILTER-001` completed / negative empirical feasibility

## Контекст

P6 подтвердил causal particle-filter implementation на synthetic six-event model, но
real-data filter был вынужден reinitialize state на каждом BBO transition вне exact
`slide/open/close` support. На audit day только `7.7319%` transitions соответствовали
шести paper events. Основные excluded classes: `73.9666%` tight-to-tight batch moves
больше одного tick и `15.5492%` transitions со spread шире двух ticks хотя бы на одном
конце. Wider spread занимал только `0.0824%` clock time, поэтому это короткий event-rich
excursion, а не противоречие time-weighted large-tick classification P5.

Текущая модель не может отличить реальное отсутствие информации о latent price от потери
информации из-за слишком узкого observation support. Простое добавление same-venue spot
measurement одновременно с изменением event model не позволило бы атрибутировать эффект.

## Решение

1. Следующий empirical этап — P6M marked multi-spread causal extension. Он относится к
   `practical-local` track и не изменяет six-event paper reproduction или P6 result.
2. Наблюдаемое состояние задаётся integer bid/ask ticks, spread
   $D=a-b$, midpoint $M=\delta(a+b)/2$ и mark
   $r=(\Delta b,\Delta a)$. Derived jump равен
   $J_r=\delta(\Delta b+\Delta a)/2$, а spread change —
   $\Delta D=\Delta a-\Delta b$.
3. Primary likelihood моделирует непосредственно observed exchange-payload batch mark.
   Он не восстанавливает недоступный порядок hypothetical primitive events внутри одного
   timestamp. Six paper events остаются exact nested special cases.
4. Tight translations допускают любое integer magnitude. Все positive observed spread
   states поддерживаются train-frozen finite parameterization и обязательным overflow
   bucket. Multi-tick или wide-spread transition сам по себе не является reset; resets
   остаются только у snapshot/data-health boundaries согласно data contract.
5. Primary marked intensities сохраняют gap-directed mechanism статьи. Directional
   first-moment constraints нормируются так, чтобы conditional corrective drift оставался
   $-\alpha G$ во всех supported states. Unconstrained marked model является diagnostic
   comparator, а не молчаливой заменой paper mechanism.
6. Proper-score comparator получает тот же full mark support: six-event gap-dependent
   P6 channel дополняется train-fitted gap-independent residual channel. Иначе старая
   zero-support likelihood проиграла бы автоматически и comparison ничего бы не проверял.
7. Весь существующий content-addressed P5/P6 dataset, включая `2024-12-15`, разрешён для
   новой модели. P6M использует preregistered rolling-origin/day-blocked evaluation и
   отдельно показывает sensitivity без `2024-12-15`, потому что decomposition этого дня
   мотивировал model class. Новая загрузка данных не является precondition P6M.
8. Same-venue spot остаётся causal diagnostic reference и не входит одновременно в
   primary measurement update. Orders, fills, thresholds и P&L запрещены на P6M.

## Statistical contract boundary

Рост supported-transition fraction является operational property, а не scientific pass.
До target run отдельные P6M protocols/configs обязаны заморозить:

- proper-log-score estimand на общем event support;
- synthetic state-recovery и calibration estimands;
- time-block/day aggregation, dependence-aware inference и multiplicity family;
- downstream-derived SESOI, equivalence/non-inferiority margins и power/precision budget;
- posterior-uncertainty-to-option-margin decision rule;
- one-factor multi-tick-only, multi-spread-only, full-model и unconstrained controls.

P7 разблокируется только при совместном прохождении likelihood, calibration, causal audit
и uncertainty gate. Full support без practically useful latent-state information не
разблокирует trading search.

## Compute

Новый numerical path реализуется векторно на PyTorch. Primary target — CUDA с
`torch.compile`; CPU fallback также PyTorch. Mark tables, particle likelihood, survival
terms и fold/session batches не должны использовать scalar Python loops, кроме неизбежной
stateful raw-archive boundary. Отдельные speed benchmarks запускаются только при реальном
выборе между конкурентными implementations.

## Последствия и отложенные решения

- Parity lock и one-dimensional exact strategy статьи больше не являются точным state
  reduction empirical model; practical policy впоследствии должна видеть $(G,D)$.
- Primary first backtest может запрещать new entry при $D>1$, продолжая causal filter
  update; отдельная optimal switching problem в $(G,D)$ остаётся future extension.
- Mark factorization, tail family/cap, optional depth/trade features, exact folds, seeds и
  numeric statistical margins не выбираются в этом ADR. Их фиксируют executable protocols
  до model outputs.
- Отдельный ещё не просмотренный период остаётся необходимым перед окончательным P9
  profitability claim, но не перед разработкой и multi-day evaluation P6M.

## Outcome — 2026-08-12

Synthetic marked filter прошёл все gates. Empirical model поддержал `100%` healthy
transitions и значительно улучшил held-out event log score, но posterior uncertainty и
time-rescaling calibration дали powered negative result. P7/P8 остаются blocked.
Итоговое решение и provenance зафиксированы в
[`ADR-0016`](0016-p6m-negative-latent-state-usability.md) и
[`P6M report`](../reports/p6m-marked-multi-spread.md).
