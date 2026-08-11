# P5: OKX data feasibility and frozen split

- **Статус:** passed
- **Experiment:** `EMP-DATA-001`
- **Run:** `20260811T232210534423Z-45f5a299b7ff-det`
- **Commit:** `6579adface7dc027753bd8778a3f268d9e168641`
- **RunSpec SHA-256:**
  `45f5a299b7ffb5f0a4ccfef82cc901fd433965e24d79131c03adf47d72ffb6ff`
- **Dataset SHA-256:**
  `0e3a6d6e99586b72ccc237bde7f8df4c3651ba4bd4495b391d9a20771c0e3888`
- **Source-list SHA-256:**
  `16036c12ff5813c41f2ebd574b72f06d99ee093a3f78f447e0b1ba9b18426d45`
- **Protocol:**
  [`p5-okx-data-feasibility.md`](../protocols/empirical/p5-okx-data-feasibility.md)

## Scope

P5 проверяет пригодность выбранных OKX event archives, large-tick eligibility
`BTC-USDT-SWAP` и причинно допустимые calendar boundaries. Стратегия, orders, fills,
efficient-price filter и P&L в этом experiment отключены. Validation/test payloads не
загружались и не читались.

Frozen local sample содержит 42 content-addressed assets (`4,428,637,877` compressed bytes):
10 L2 days, 20 trade archives и 12 monthly funding archives. Swap проверен на семи
приблизительно равномерных train dates 2024 года; spot — на трёх из них.

## Method

Daily L2 archives обрабатывались десятью независимыми processes. JSON decoding и
state-dependent snapshot/delta reconstruction оставались последовательными внутри archive;
batch reductions выполнялись vectorized PyTorch CPU tensors. Impossible book state включал
quarantine до следующего полного valid snapshot. Trade archives проверялись относительно их
UTC+8 source cuts; пары с labels `D` и `D+1` вместе покрывают UTC audit day `D`.

Large-tick estimand — equal-weight mean семи swap day fractions со spread в `{1,2}` ticks.
Односторонняя 95% lower bound построена 10,000 cluster-bootstrap resamples полных UTC days,
seed `2026081201`. Acceptance требовал lower bound строго выше `0.99`.

Hardware: Intel Core i9-12900H, 20 logical CPUs, NVIDIA RTX 3080 Ti Laptop GPU (CUDA
доступна, но parser/state-machine workload выполнялся на CPU). Python `3.14.0`, PyTorch
`2.13.0`, orjson `3.11.9`. Wall time — `186.365 s`; сумма CPU processing time по files —
`1,167.471 s`.

## Results

Все десять final acceptance gates прошли:

| Gate | Result |
|---|---:|
| Raw size/SHA and dataset identity | pass |
| L2 structural quality after causal health mask | pass |
| Trade schema/order/value quality | pass |
| UTC+8 trade-archive alignment | pass |
| Funding train coverage | pass |
| Large-tick day-cluster lower bound | pass |
| Spot/swap L2 overlap | pass |
| Expected 42 assets | pass |
| Expected 10 L2 days | pass |
| Expected 20 trade archives | pass |

### Large-tick eligibility

| Swap UTC day | Fraction spread in `{1,2}` ticks |
|---|---:|
| 2024-01-15 | 0.998601 |
| 2024-03-15 | 0.986444 |
| 2024-05-15 | 0.998020 |
| 2024-07-15 | 0.998413 |
| 2024-09-15 | 0.999424 |
| 2024-11-15 | 0.997602 |
| 2024-12-15 | 0.999092 |

Equal-day mean равен `0.9967993`. Односторонняя 95% day-cluster lower bound равна
`0.9934358`, то есть выше minimum useful occupancy `0.99` на `0.0034358` (0.344 percentage
points). На frozen P5 sample `BTC-USDT-SWAP` поэтому получает strict large-tick label.

### Data health and channel alignment

- Прочитано `70,455,943` source L2 rows; downstream-valid observations — `70,370,594`.
- В swap archive 2024-03-15 найдено `3` quarantine episodes: `79,472` rows и `830,430 ms`
  calendar time. В них зафиксированы `15` invalid snapshots и `79,457` invalid updates;
  ни один impossible state не попал в estimands, все episodes восстановились full snapshot.
- Swap archive 2024-11-15 начался с `5,877` pre-snapshot updates; первый usable snapshot
  пришёл через `59,998 ms`, внутри одного нативного 60-second snapshot cycle.
- Проверено `25,208,535` trade rows. Все 20 archives лежат внутри своих UTC+8 cuts;
  timestamp и trade IDs возрастают как внутри files, так и между соседними files.
- Funding содержит `1,097` unique train observations от `2024-01-01 00:00 UTC` до
  `2024-12-31 16:00 UTC`; maximum gap — ровно 8 часов, конфликтующих duplicates нет.
- Frozen splits: train `2024-01-01`--`2024-12-31`, validation
  `2025-01-01`--`2025-06-30`, test `2025-07-01`--`2025-12-31`, все UTC.

## Deviations and preserved negative evidence

Первый clean run остановился на archive, начинавшемся до первого full snapshot. Поправка
native-cycle gate зафиксирована до повторного run в [`ADR-0012`](../adr/0012-p5-native-snapshot-warmup.md).
Второй clean run выявил impossible L2 interval, UTC+8 trade cuts и ошибочный funding endpoint;
health-mask/calendar correction зафиксирован до final run в
[`ADR-0013`](../adr/0013-p5-health-mask-and-trade-calendar.md). Оба прежних failed run
сохранены в `outputs/EMP-DATA-001/` и не переименованы в passed.

## Limitations

- Eligibility подтверждена на семи swap days, а не на всём train year; редкий режим
  2024-03-15 заметно шире остальных и должен сохраняться как regime, а не удаляться.
- Source не содержит sequence IDs. Full-snapshot quarantine однозначен и консервативен, но
  не доказывает отсутствие пропущенных deltas между внешне корректными states.
- Spot availability подтверждена только на трёх days. Выбор causal estimator относится к P6.
- Historical contract metadata и будущие fee/latency assumptions ещё не являются P5 result;
  они должны быть заморожены до execution/P&L experiments.
- Passed P5 ничего не говорит о доходности стратегии.

## Artifacts and conclusion

Canonical local artifacts:
`outputs/EMP-DATA-001/20260811T232210534423Z-45f5a299b7ff-det/`. Primary files:
`metrics/summary.json`, `metrics/bootstrap.json`, `metrics/file_quality.csv`,
`tables/day_quality.csv`, `tables/funding_quality.csv`, `tables/split_freeze.csv` и
`manifest.json`.

P5 завершён как **passed**. OKX `BTC-USDT-SWAP` и same-venue `BTC-USDT` data contract
пригодны для P6 causal efficient-price estimation при обязательном health mask и UTC-normalized
trade merge.
