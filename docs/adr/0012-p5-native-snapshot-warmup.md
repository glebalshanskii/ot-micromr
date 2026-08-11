# ADR-0012: P5 uses the first native snapshot as the reconstruction boundary

- **Статус:** Accepted
- **Дата:** 2026-08-12
- **Связанный этап:** P5
- **Связанный protocol:**
  [`p5-okx-data-feasibility.md`](../protocols/empirical/p5-okx-data-feasibility.md)

## Контекст

Первый clean target run `EMP-DATA-001/20260811T225819351471Z-56b4c0cb8bc1-det`
остановился до расчёта scientific metrics. Архив
`BTC-USDT-SWAP-L2orderbook-400lv-2024-11-15.tar.gz` начинается с `update`, а не
с `snapshot`. Первый полный snapshot находится в строке 5,878 с timestamp
`2024-11-15T00:00:59.998Z`.

Это не произвольный разрыв: во всех семи проверяемых swap archives полные snapshots идут
ровно с периодом 60,000 ms. В шести днях цикл совпадает с UTC midnight с отклонением не
более 6 ms; 15 ноября архив начинается внутри того же цикла. До первого snapshot нельзя
восстановить полную книгу, но после него state однозначен. Потерянный reconstructible prefix
занимает 59,998 ms, или 0.06944% UTC day.

Исходный gate требовал `snapshot` в первой строке и тем самым проверял совпадение file cut с
snapshot cycle, а не пригодность данных для causal reconstruction. Это операционная ошибка
gate. Исходный failed run сохраняется и не переименовывается.

## Решение

1. Все records до первого snapshot проверяются на schema, instrument, price lattice и
   nondecreasing source order, но не интерпретируются как полная книга и не входят в
   spread/depth estimands.
2. Первый полный snapshot становится единственной usable reconstruction boundary дня;
   последующие deltas применяются в source order как раньше.
3. Допустимый initial warmup равен одному нативному snapshot period плюс endpoint tolerance:
   `60,000 + 100 = 60,100 ms`. Порог выведен из observable source cadence и прежней
   timestamp tolerance, а не из large-tick результата или strategy P&L.
4. Raw archive по-прежнему должен начинаться не дальше 100 ms от UTC midnight, заканчиваться
   не дальше 100 ms от UTC day end и содержать usable snapshot. Окончание дня, ordering,
   empty/locked/crossed, trade, funding и large-tick gates не меняются.
5. В artifacts отдельно сохраняются source rows, usable book rows, число отброшенных
   pre-snapshot updates и timestamp первого usable snapshot.

## Последствия

- Изменяется RunSpec hash и код реконструктора; повторный target run выполняется только из
  нового clean commit.
- Validation/test остаются нетронутыми.
- Если первый snapshot позже 60,100 ms, отсутствует полностью или после него книга
  некорректна, соответствующий day по-прежнему проваливает P5 structural gate.
- Поправка ничего не утверждает о large-tick occupancy: эта метрика впервые вычисляется
  только в повторном полном run.
