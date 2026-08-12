# ADR-0016: P6M marked event layer accepted, latent-state signal rejected

- **Статус:** Accepted
- **Дата:** 2026-08-12
- **Связанный этап:** P6M
- **Зависимость:** [`ADR-0015`](0015-marked-multi-spread-causal-model.md)
- **Отчёт:** [`p6m-marked-multi-spread.md`](../reports/p6m-marked-multi-spread.md)

## Контекст

P6M проверял, была ли отрицательная empirical feasibility P6 главным образом следствием
слишком узкого six-event observation support. Marked model расширил наблюдения до всех
healthy BBO batch transitions, сохранил направленный mean-reversion mechanism статьи и
не использовал spot как measurement. Orders, thresholds и P&L не вычислялись.

Known-$X$ synthetic dependency прошла state recovery, predictive score, posterior
coverage, constrained drift и replay gates. Empirical rolling-origin run использовал шесть
held-out days, 288 заранее заданных 30-минутных blocks и все `1,122,613` healthy
transitions. В том числе `2024-12-15` участвовал как обычный held-out fold; агрегат без
декабря был отдельной sensitivity-проверкой.

## Наблюдаемые результаты

1. Operational support вырос до `100%`; future timestamp accesses равны нулю, replay
   bitwise exact, все состояния/variance конечны и положительны.
2. Full marked model превзошёл full-support gap-independent comparator на
   `0.30486 nat/event`; односторонняя 97.5% lower bound `0.26239` выше SESOI `0.01`.
3. Multi-tick и multi-spread components дали соответственно `0.25394` и
   `0.07234 nat/event`; оба Holm-adjusted `p < 2e-14`.
4. Однако posterior SD был в среднем по blocks в `1.6087` раза больше optimistic
   option-value margin. Для metric
   $1-\operatorname{median}(SD)/margin$ mean `-0.60873`, а 97.5% interval
   `[-0.69000,-0.52747]` целиком ниже нуля. Это powered negative, не отсутствие precision.
5. Time-rescaling calibration также несовместима с frozen margins: mean `2.2083` при
   допустимом `[0.9,1.1]`, standard deviation `5.3285` при `[0.8,1.2]`; их 95%
   intervals целиком выше equivalence regions.
6. Исключение `2024-12-15` не меняет вывод: score остаётся положительным, uncertainty
   metric равен `-0.61467` с interval `[-0.70923,-0.52010]`.

## Решение

1. P6M завершён как **negative latent-state usability result**. P7/P8 остаются blocked;
   этот filter нельзя передавать в поиск trading P&L.
2. Marked multi-spread event representation принимается как полезный observation/likelihood
   layer: она устранила artificial resets и дала значимое out-of-sample event-prediction
   улучшение. Это не является доказательством качества latent efficient price.
3. Текущий book-only state estimator отклоняется как strategy signal. Relative proper-score
   superiority не заменяет absolute calibration и uncertainty gate.
4. Следующее расширение, если оно будет принято отдельно, должно решать observability/state
   anchoring: например, causal same-venue spot measurement либо явно идентифицируемую
   measurement/state model. Оно требует нового ADR/protocol и не может молча ослаблять
   P6M gates.
5. Существующие даты, включая `2024-12-15`, разрешено повторно использовать для разработки
   другой модели. Такой reuse нужно просто указывать; он не делает данные запрещёнными.
   Отдельный новый период понадобится только для будущего confirmatory profitability claim.

## Интерпретация и риск

Факт — новая mark-модель лучше предсказывает BBO events, но её posterior state слишком
широк и event-time calibration сильно завышена. Наиболее вероятное, но пока не доказанное
объяснение — fit-only 300-second midpoint EWMA смешивает microstructure gap с локальным
price trend, а directional mark flow недостаточно сильно якорит Brownian $X$. Проверять
эту гипотезу можно только новым state-observation experiment, а не post-hoc изменением
порога или запуском P&L.

