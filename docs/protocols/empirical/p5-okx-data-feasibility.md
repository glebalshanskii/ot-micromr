# EMP-DATA-001: OKX data feasibility and split freeze

- **Статус:** preregistered before reading the selected 2024 payloads
- **Дата:** 2026-08-12
- **Decision:** [`ADR-0011`](../../adr/0011-okx-single-venue-empirical-data.md)
- **Source list:** `cfg/experiments/emp_data_001_sources.toml`
- **Target RunSpec:** `cfg/experiments/emp_data_001.toml` after acquisition fixes the
  content-addressed raw-manifest hash

## Цель и границы

P5 проверяет, можно ли из официальных OKX archives однозначно построить causal empirical
input для следующих этапов. P5 не рассчитывает signal, orders или P&L, не выбирает filter и
не смотрит validation/test payloads.

Единственный execution candidate — OKX `BTC-USDT-SWAP`. Same-venue `BTC-USDT` spot нужен
только для проверки доступности будущего causal reference. Binance и другие venues запрещены
этим protocol.

## Frozen information set

### Chronological split

Границы фиксируются до strategy P&L:

| Split | UTC interval | Calendar days | Допустимое использование |
|---|---:|---:|---|
| Train | 2024-01-01 00:00:00 -- 2024-12-31 23:59:59.999 | 366 | data quality, eligibility, filter/model fitting |
| Validation | 2025-01-01 00:00:00 -- 2025-06-30 23:59:59.999 | 181 | one frozen model/strategy choice |
| Test | 2025-07-01 00:00:00 -- 2025-12-31 23:59:59.999 | 184 | one untouched P9 evaluation |

P5 читает только семь заранее выбранных train days:
`2024-01-15`, `2024-03-15`, `2024-05-15`, `2024-07-15`, `2024-09-15`,
`2024-11-15`, `2024-12-15`. Они приблизительно равномерно покрывают train year и не
выбирались по market outcome.

Для swap загружаются L2 и trades для всех семи дней. Для проверки same-venue alignment spot
L2/trades загружаются только для `2024-01-15`, `2024-07-15`, `2024-12-15`. Funding
загружается помесячно за весь train year. Index/mark не входят в P5 target: published archive
не обещает сопоставимый event-level channel; их пригодность может быть пересмотрена в P6 без
добавления другой биржи.

### Instrument contract

- execution instrument: `BTC-USDT-SWAP`, linear perpetual;
- same-venue reference candidate: `BTC-USDT` spot;
- canonical timezone and day boundary: UTC;
- audit tick: `0.1 USDT`; каждый price обязан лежать на этой lattice;
- current swap contract metadata snapshot: `ctVal=0.01 BTC`, `ctMult=1`,
  `lotSz=minSz=0.01 contracts`; historical metadata uncertainty сохраняется в report и не
  используется для P&L на P5.

## Acquisition and provenance

Download выполняется только явной командой:

```bash
uv run ot-micromr fetch-data cfg/experiments/emp_data_001_sources.toml
```

Source list фиксирует exact URL, expected byte size, instrument, channel и date. Fetcher
пишет каждый file через `.part`, проверяет byte size, считает SHA-256 в streaming pass и
атомарно переименовывает файл. Canonical dataset hash считается по ordered tuples
`(asset_id, size_bytes, sha256)` и не зависит от retrieval time. Raw files хранятся под
`data/` и не попадают в Git.

## Event normalization

### L2 order book

`tar.gz` содержит NDJSON. Первый record дня обязан быть `snapshot`; последующие
`update` records применяются в file order. Ask/bid element имеет
`[price, size, order_count]`; zero size удаляет level. Price переводится в exact integer
ticks до численных расчётов. На каждом record после полного применения deltas сохраняются
best bid/ask и touch sizes.

Primary ordering key — `(exchange_timestamp_ms, file_row_index)`. Equal timestamps допустимы
и сохраняют source order; убывание timestamp запрещено. Архив не содержит sequence ID,
поэтому P5 не приписывает ему сетевую gap recovery. Daily snapshot является единственной
reinitialization boundary. Malformed row, неверный instrument или невозможность восстановить
непустой uncrossed book делает соответствующий day непригодным.

### Trades and funding

Trades сохраняют source row order; проверяются nondecreasing `created_time`, increasing
`trade_id`, side, positive price/size и tick lattice. Funding объединяется по timestamp,
одинаковые boundary rows дедуплицируются только при одинаковом rate; внутри train interval
ожидается не более восьми часов между соседними observations.

## Compute contract

Archives независимы по дням и обрабатываются process-parallel с максимум десятью workers.
Gzip/CSV/JSON decoding является I/O boundary. Book reconstruction из deltas неизбежно
stateful и изолировано в минимальном loop; все batch reductions, masks, counts и weighted
aggregations выполняются vectorized `torch.Tensor` на CPU. CUDA transfer не используется:
основная работа — независимые branch-heavy parsers/state machines, а итоговый tensor batch
мал относительно transfer/coordination overhead. Отдельный speed benchmark не запускается.

## Estimands and uncertainty

Для каждого UTC day считаются:

- rows, first/last timestamp, duplicate/nonmonotone timestamps и maximum timestamp gap;
- snapshot/update/malformed counts;
- empty, locked и crossed book observations;
- fraction spread равен 1 tick, 2 ticks и вне `{1,2}`;
- best bid/ask touch-size means and minima;
- BBO change rate;
- trades, first/last trade timestamp, trade-ID/tick/side violations;
- raw/compressed bytes and processing time.

Large-tick primary estimand — equal-weight mean семи swap day-level fractions
`Pr(spread_ticks in {1,2})`. Односторонняя 95% lower confidence bound строится cluster
bootstrap по полным UTC days: 10,000 resamples, PyTorch generator seed `2026081201`.
Day, а не 10-ms row, является independent resampling unit. Strict paper-compatible label
требует lower bound выше `0.99`; это superiority gate над заранее заданной minimum useful
availability, а не event-count heuristic. Spot не входит в эту family.

## Acceptance and stop rules

P5 проходит только если одновременно:

1. все source assets скачаны с exact expected byte size, manifest hashes повторно
   подтверждены и canonical dataset hash совпадает с RunSpec;
2. все десять L2 days начинаются snapshot, имеют верный instrument, nondecreasing timestamps,
   full UTC-day endpoint coverage с tolerance 100 ms и ноль malformed/empty/locked/crossed
   observations;
3. все trade files имеют верный instrument, nondecreasing timestamps, strictly increasing
   trade IDs, positive tick-aligned prices/sizes и допустимые sides;
4. funding покрывает весь train year, boundary duplicates согласованы и maximum unique gap
   не превышает восемь часов;
5. lower 95% day-cluster bound swap one/two-tick occupancy больше `0.99`;
6. три spot/swap audit-day pairs имеют overlapping full-day L2 coverage;
7. raw manifest, per-file metrics, day table, bootstrap output и split freeze сохранены.

Gate 2 проверяет структурную пригодность official archive, а не утверждает наличие сетевого
sequence. Любой failed gate остаётся отрицательным P5 result; thresholds после target run не
ослабляются. Strategy/P&L stages не запускаются при `blocked-data`.

## Planned outputs

- local raw manifest with content-addressed dataset hash;
- `metrics/file_quality.csv` and `tables/day_quality.csv`;
- `tables/funding_quality.csv` and `tables/split_freeze.csv`;
- `metrics/bootstrap.json` and `metrics/summary.json`;
- `docs/reports/p5-okx-data-feasibility.md` after the clean target run.

## Amendment 2026-08-12: native snapshot warmup

Первый clean target run остановился до scientific metrics: один из десяти L2 archives
начался внутри нативного 60-second snapshot cycle. Наблюдение, materiality analysis и
изменение gate зафиксированы в [`ADR-0012`](../../adr/0012-p5-native-snapshot-warmup.md).
Исходный failed run и исходные правила выше остаются historical record.

Для всех последующих `EMP-DATA-001` runs пункты 2 и 6 acceptance rules заменяются так:

- source coverage всё ещё обязана начинаться/заканчиваться в пределах 100 ms от UTC day;
- unusable `update` prefix до первого полного snapshot проверяется, но исключается из book
  estimands;
- первый usable snapshot обязан появиться не позже `60,100 ms` после UTC midnight;
- nondecreasing source order и ноль empty/locked/crossed observations проверяются на всей
  reconstructible suffix;
- spot/swap overlap начинается с более позднего из двух первых usable snapshots; обе стороны
  обязаны удовлетворять тому же `60,100 ms` warmup gate.

Large-tick threshold, bootstrap, dates, instruments и все остальные gates не изменены.

## Amendment 2026-08-12: book health mask and UTC trade coverage

Второй clean run сохранил полные metrics и выявил повреждённый L2 interval, UTC+8 cut у
trade archives и неверный expected funding endpoint. Evidence и причинное правило обработки
зафиксированы в [`ADR-0013`](../../adr/0013-p5-health-mask-and-trade-calendar.md).

Для последующих `EMP-DATA-001` runs действуют дополнительные правила:

- impossible book включает quarantine до следующего полного valid snapshot; quarantined
  rows не входят в book estimands, но elapsed calendar time сохраняется как downtime;
- structural pass относится к downstream-valid observations и требует recovery всех
  quarantine episodes; raw faults сохраняются отдельными metrics;
- для каждого UTC audit day `D` загружаются trade archives `D` и `D+1`, потому что source
  archive cut равен UTC+8; каждый raw timestamp проверяется относительно этого cut;
- соседние trade archives сохраняют timestamp/trade-ID order, а последующая empirical
  pipeline обязана merge/filter exact UTC day до построения features;
- expected funding endpoint исправлен на последний scheduled timestamp внутри frozen train:
  `2024-12-31 16:00:00 UTC`.

Порог large-tick, bootstrap family, instruments и даты не изменены. Validation/test payloads
по-прежнему не читаются.
