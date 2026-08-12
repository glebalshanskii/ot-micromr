# ADR-0021: закрыть текущий empirical track после отрицательного результата идентификации state

- **Status:** accepted
- **Date:** 2026-08-12
- **Stage:** P10 project closeout

## Контекст

Статья предполагает, что efficient price $X_t$, а значит и gap $G_t=M_t-X_t$,
наблюдаемы. Для применимой empirical strategy сначала нужен causal estimator, uncertainty
которого меньше расстояния от торгового порога до границы execution costs.

На frozen OKX development sample 2024 года проверены четыре последовательно обобщённые
empirical models:

1. exact six-event book model статьи;
2. marked multi-spread extension, покрывающая все healthy BBO transitions;
3. continuous integrated-hazard likelihood и event rollout;
4. factorized renewal clock и conditional mark model.

Реализации прошли synthetic или numerical controls. Финальная factorized model также
откалибровала два preregistered time-rescaling moments. Однако conditional gap tilt изменил
held-out mark log score лишь на `-0.0000834 nat/event` при simultaneous interval
`[-0.0001668, 0.00000004]`, а posterior uncertainty была в `8.426` раза больше
optimistic option-value margin. Следовательно, используемая в текущем подходе наблюдаемая
BBO history не идентифицирует latent state, необходимый policy статьи.

## Решение

1. Закрыть текущий проект на P10 с общим outcome `negative` для проверенного empirical
   state-identification approach.
2. Не выполнять P7--P9. После failed precondition P6D нет оснований создавать backtester,
   искать trading parameters или открывать untouched P&L test.
3. Указать profitability claim как `not-confirmed / not tested`, а не как доказательство
   убыточности стратегии. Orders, fills, costs и P&L не рассчитывались.
4. Сохранить analytical reproduction, simulator, configs, reports и отрицательные
   empirical implementations как аудируемый результат проекта.
5. Не открывать frozen validation/test periods 2025 года для остановленной hypothesis.
6. Любая будущая crypto strategy на непосредственно наблюдаемых order-book/trade features
   или независимом causal price observation является новой research hypothesis со своим
   protocol, comparisons и untouched period; это не P7 текущего проекта.

## Последствия

- Figures 2 и 5 остаются illustrative structural reproductions. Figure 4 остаётся
  independent partial reproduction с inconclusive scientific family из трёх строк.
- Положительные synthetic gross rates относятся к модели статьи, а не свидетельствуют о
  real-market profitability.
- Clock factorization сохраняется как поддержанный modelling result, но дальнейший clock
  tuning сам по себе не разблокирует trading, потому что не добавляет информацию о $X_t$.
- P7--P9 фиксируются как `not executed; stopped by precondition`, а не `failed` или
  молча оставленные pending.

## Отклонённые альтернативы

- **Всё равно запустить backtest:** это выбирало бы thresholds по уже признанному
  неинформативным state estimate и превратило бы noise в target поиска стратегии.
- **Ослабить uncertainty gate:** realistic fees и latency уменьшают option margin, поэтому
  ослабление optimistic gate экономически не обосновано.
- **Настраивать новые clock families:** P6D отделил timing от direction и показал, что
  исправление clock не восстанавливает conditional directional information.
- **Добавить новые observations в ту же hypothesis:** это существенно меняет data/model
  contract и относится к отдельному проекту.
