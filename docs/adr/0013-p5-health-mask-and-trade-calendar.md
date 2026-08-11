# ADR-0013: P5 quarantines invalid books and normalizes the OKX trade calendar

- **Статус:** Accepted
- **Дата:** 2026-08-12
- **Связанный этап:** P5--P9
- **Связанный protocol:**
  [`p5-okx-data-feasibility.md`](../protocols/empirical/p5-okx-data-feasibility.md)

## Контекст

Полный amended P5 run
`EMP-DATA-001/20260811T230547413166Z-f9d5a47909ca-det` впервые записал все
metrics и выявил три независимых contract issues до просмотра strategy/P&L:

1. В swap L2 archive 2024-03-15 находятся 15 последовательных snapshots, где ask-side
   застыла около 65,662 USDT, а bid-side находится около 70,893 USDT. Невозможные snapshots
   идут примерно с 12:30 до 12:44 UTC; следующий полный корректный snapshot снова даёт
   однозначное состояние.
2. OKX trade archives с датой `D` используют calendar cut UTC+8 и фактически покрывают
   `[D-1 16:00 UTC, D 16:00 UTC)`. L2 archives используют UTC day. Поэтому файл `D` один
   не покрывает последние восемь часов UTC audit day `D`.
3. RunSpec ошибочно ожидал последний train funding в `2025-01-01 00:00 UTC`, уже за
   пределами frozen train. При восьмичасовом расписании правильный последний timestamp
   внутри train — `2024-12-31 16:00 UTC` (`1735632000000`).

Требование нуля ошибок в raw source смешивало две разные вещи: качество каждой исходной
строки и возможность построить однозначный causal input. Невозможное состояние нельзя
передавать стратегии, но полностью отбрасывать сутки после последующего корректного full
snapshot тоже не требуется.

## Решение

1. Любой empty, locked или crossed reconstructed book немедленно включает health
   quarantine. Текущий record и все следующие records исключаются из features, book
   estimands, decisions и fills.
2. Quarantine завершается только на следующем полном snapshot, который сам является
   непустым и uncrossed. Recovery через deltas запрещён: без sequence ID его полноту нельзя
   доказать.
3. Calendar-time внутри quarantine сохраняется как strategy downtime с нулевыми decisions;
   оно не вырезается из будущего P&L denominator. Artifacts хранят episodes, rows, duration,
   invalid snapshots/updates и unrecovered-tail flag.
4. Structural gate проверяет, что в downstream observations нет impossible books и каждый
   quarantine восстановлен полным snapshot до конца archive. Raw anomalies остаются
   измеряемым quality result, но их ненулевое число само по себе больше не является
   эвристическим binary rejection rule.
5. Для каждого frozen UTC audit day и instrument source list содержит trade archives с
   labels `D` и `D+1`. Оба raw файла проверяются отдельно; downstream merge фильтрует rows
   exact UTC interval `D` и сохраняет source order.
6. Trade file boundaries интерпретируются как UTC+8 archive cuts. Timestamp каждого row
   обязан находиться внутри заявленного cut; соседние archives обязаны сохранять timestamp
   и trade-ID order.
7. Последний ожидаемый train funding timestamp исправляется на `1735632000000`. Funding
   maximum-gap rule не меняется.

## Последствия

- Source list расширяется с 32 до 42 assets: L2 и funding не меняются, добавляются десять
  following-day trade archives.
- Dataset content hash, source-list hash и RunSpec hash будут новыми; повторный target run
  выполняется из clean commit после acquisition.
- Validation/test payloads не загружаются и не читаются.
- Large-tick threshold `lower95 > 0.99`, day-cluster bootstrap и frozen audit dates не
  меняются. Метрика считается только по causal reconstructible observations.
- Старые failed runs остаются отрицательными historical records и не переименовываются.
