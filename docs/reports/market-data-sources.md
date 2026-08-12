# Источники market data и локальный inventory

- **Статус:** current source survey and local inventory
- **Дата проверки внешних источников:** 2026-08-12
- **Текущий data contract:** OKX-only, согласно
  [`ADR-0011`](../adr/0011-okx-single-venue-empirical-data.md)
- **Канонический empirical dataset:** `okx-btc-p5-audit-2024-v1`
- **Связанный experiment:** `EMP-DATA-001`

Этот документ отвечает на два разных вопроса:

1. какие официальные и сторонние источники исторических crypto market data были проверены
   как кандидаты для event-level tests;
2. какие данные уже фактически скачаны в текущем workspace, где они лежат и в каком формате.

Наличие источника в обзоре не означает, что он разрешён текущим protocol. В завершённом
empirical track использовался только OKX. Binance, Bybit, BitMEX, Hyperliquid и коммерческие
vendors не загружались, не входили в features и не использовались для model selection или P&L.

## 1. Краткий вывод

| Источник | Проверенные historical channels | Доступ и формат | Пригодность для primary event-level test | Локально скачан |
|---|---|---|---|---|
| **OKX** | high-resolution L2, tick trades, perpetual funding, candles, borrowing rates | официальный download portal; `tar.gz`/NDJSON и `zip`/CSV | **Да**: это единственный принятый baseline, но L2 не содержит sequence ID | **Да**, 42 assets |
| **Binance** | trades, aggTrades, klines, funding, sampled depth aggregates; ограниченный архив `bookTicker` | `data.binance.vision`; daily/monthly ZIP/CSV + SHA-256 sidecars | **Не как drop-in L2 replacement**: `bookDepth` не является raw book delta stream, а подтверждённый `bookTicker` даёт только BBO | Нет |
| **Bybit** | public spot/derivatives trades, premium/spot indices; funding через API | `public.bybit.com`; daily/monthly `csv.gz`; REST/WebSocket для current book | **Не подтверждено**: стабильный public historical L2 archive contract не найден | Нет |
| **BitMEX** | daily BBO quotes и tick trades | public S3; daily `csv.gz`, все symbols в одном файле | **Только для BBO/trade control**: full L2 отсутствует в public daily extracts | Нет |
| **Hyperliquid** | hour-partitioned L2 book snapshots, asset contexts, fills/node data | requester-pays S3; `.lz4`, CSV/JSON-like node records | Потенциально да, после отдельного schema/coverage/license audit; возможны missing data | Нет |
| **Tardis.dev** | exchange-native и normalized trades, incremental L2, BBO, snapshots | коммерческий API/download; NDJSON или daily CSV | Технически наиболее полный multi-venue fallback; требует подписки и отдельной лицензии | Нет |
| **Kaiko** | trades, tick-level BBO, periodic raw L2 snapshots | коммерческий API/cloud delivery; JSON/CSV | Полезен для controls; 30-second raw snapshots сами по себе недостаточны для event-by-event execution | Нет |

Для текущего проекта вывод не меняется: единственный воспроизводимый local dataset —
OKX `BTC-USDT-SWAP` плюс same-venue spot `BTC-USDT`. Добавление другого venue потребует
нового ADR/protocol и нового untouched period, а не изменения уже завершённого результата.

## 2. Требования проекта к данным

Primary test микроструктурной mean-reversion стратегии нельзя корректно проводить только на
candles. Минимальный data contract должен включать:

| Channel | Минимальное требование | Зачем нужен |
|---|---|---|
| Order book | initial snapshot, ordered deltas или event-level BBO, exchange timestamps, recovery semantics | spread state, touch/depth, executable signal, gap detection |
| Trades | individual trades, timestamp, price, size, aggressor side или однозначный maker flag, stable row order | market-order flow, causal alignment, execution controls |
| Funding | historical settled funding rate и settlement timestamp | net P&L perpetual position |
| Instrument metadata | tick size, lot size, contract multiplier/value и их history | unit conversion и executable order sizing |
| Fees/status | maker/taker fees, maintenance/outage records, delist or contract changes | conservative cost and availability model |
| Provenance | exact URL, retrieval time, bytes, SHA-256, source schema/version | immutable dataset identity и replay audit |
| Legal status | разрешённое local use, retention и redistribution policy | допустимое хранение и publication boundary |

Snapshots с редким polling, OHLCV и процентные depth aggregates пригодны для coarse controls,
но не заменяют source-ordered L2/BBO path в primary execution test.

## 3. Проверенные источники

### 3.1. OKX — выбранный и загруженный источник

Официальный [`OKX Historical Market Data`](https://www.okx.com/en-us/historical-data)
заявляет tick-level trades с сентября 2021 года, perpetual funding с марта 2022 года и
high-resolution L2 с марта 2023 года. На portal также доступны candles и borrowing rates.

Преимущества для проекта:

- один venue предоставляет swap L2, swap trades, funding и same-venue spot reference;
- daily L2 archive содержит full snapshots и последующие updates до 400 levels;
- raw archive можно сохранить content-addressed локально и повторно проверить без API key;
- доступный период перекрывает замороженный train interval 2024 года.

Ограничения:

- L2 records не содержат sequence ID; source row order и periodic full snapshot не доказывают
  отсутствие пропущенных deltas;
- trade archive labels используют UTC+8 cuts, тогда как канонический research calendar — UTC;
- historical contract metadata и exact account fee tier не входят в скачанный dataset;
- условия OKX разрешают personal use, включая разработку собственной стратегии, но запрещают
  redistribution/sublicensing, оставляют лицензию revocable и требуют удалить копии при её
  прекращении. Полный текст: [`OKX Historical Data Terms`](https://www.okx.com/en-us/help/historicaldata-terms-and-conditions).

Exact source URLs и expected sizes всех 42 assets находятся в
[`cfg/experiments/emp_data_001_sources.toml`](../../cfg/experiments/emp_data_001_sources.toml).

В frozen local dataset **не входят** mark/index price history, historical instrument metadata,
fee tiers и venue-status events. Current instrument configuration и system status доступны
через [`OKX API guide`](https://www.okx.com/docs-v5/) (`GET /api/v5/public/instruments` и
`GET /api/v5/system/status`), а account fee rate требует account-scoped endpoint. Current API
response нельзя считать historical metadata: для нового backtest эти records нужно начать
сохранять заранее либо восстановить из официальных dated announcements с отдельным provenance.

### 3.2. Binance — хорошие trades/funding, но не эквивалент OKX L2

Официальный [`binance-public-data`](https://github.com/binance/binance-public-data)
описывает public archives для Spot, USD-M и COIN-M Futures. Файлы публикуются daily/monthly,
упакованы в ZIP и сопровождаются `.CHECKSUM`. Базовый browser:
[`data.binance.vision`](https://data.binance.vision/).

Для `BTCUSDT` USD-M Futures подтверждены:

- `trades` и `aggTrades` в CSV;
- monthly `fundingRate`;
- `bookDepth`, но его строки имеют schema
  `timestamp,percentage,depth,notional`: это sampled aggregate liquidity в полосах
  `-5..-1,+1..+5%`, а не price-level snapshot/delta stream;
- `bookTicker` с event-level best bid/ask существовал в просмотренном archive. На дату
  проверки листинг `BTCUSDT` охватывал `2023-05-16`--`2024-03-30`; более поздние probe URLs
  отсутствовали. Это observed archive state, не обещание неизменной coverage.

Критическая schema caveat: официальный README предупреждает, что Spot timestamps начиная с
`2025-01-01` записываются в microseconds, тогда как более ранние Spot и показанные Futures
files используют milliseconds. Unit должен определяться из channel/date contract, а не по
величине числа post hoc.

Практический статус:

- trades/funding подходят для cross-venue control или отдельного Binance experiment;
- `bookDepth` непригоден для реконструкции touch/fills;
- ограниченный `bookTicker` может поддержать BBO-only test на своём покрытии, но не full L2;
- full historical L2 потребует собственного заранее запущенного WebSocket recorder либо
  стороннего vendor;
- MIT license официального GitHub repository не следует автоматически трактовать как
  разрешение перераспространять сами exchange archives; data-use terms требуют отдельной
  проверки перед публикацией.

Ничего из Binance в `data/` или `outputs/` не скачано.

### 3.3. Bybit — public trades есть, historical L2 contract не подтверждён

Public root [`public.bybit.com`](https://public.bybit.com/) на дату проверки содержит
`trading/`, `spot/`, `premium_index/`, `spot_index/` и MetaTrader klines. Проверенные
trade formats:

- derivatives: daily `csv.gz`, например
  `trading/BTCUSDT/BTCUSDT2024-01-15.csv.gz`; header
  `timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional`;
  `timestamp` в проверенном файле — fractional Unix seconds;
- spot: daily/monthly `csv.gz`, например
  `spot/BTCUSDT/BTCUSDT_2024-01-15.csv.gz`; header
  `id,timestamp,price,volume,side`; `timestamp` в проверенном файле — Unix milliseconds.

Официальный V5 endpoint
[`Get Orderbook`](https://bybit-exchange.github.io/docs/v5/market/orderbook) возвращает current
snapshot до 1000 levels для spot/contracts и включает `u` (update ID), `seq`, system timestamp
`ts` и matching-engine timestamp `cts`. WebSocket позволяет записывать future updates.
[`Funding Rate History`](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate)
возвращает historical funding для linear/inverse perpetuals с pagination до 200 records.

Однако public bucket root не показывает отдельный historical order-book archive, а REST
orderbook endpoint документирован как current snapshot. Поэтому Bybit пока не имеет в проекте
проверенного immutable historical L2 source list. До использования нужно отдельно подтвердить
portal entitlement, coverage, snapshot/delta schema, sequence recovery и data terms.

Ничего из Bybit локально не скачано.

### 3.4. BitMEX — простой официальный BBO/trade archive

BitMEX указывает на daily public extracts в
[`API Overview`](https://www.bitmex.com/app/apiOverview); browser расположен на
[`public.bitmex.com`](https://public.bitmex.com/). S3 prefixes:

- `data/quote/YYYYMMDD.csv.gz` — header
  `timestamp,symbol,bidSize,bidPrice,askPrice,askSize`;
- `data/trade/YYYYMMDD.csv.gz` — header
  `timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional,trdType`.

Один daily file содержит все symbols. В проверенном `2024-01-15` timestamps имеют
nanosecond-resolution text form вроде `2024-01-15D00:00:00.451986388`.

Это удобный официальный источник для BBO/trade robustness checks, но `quote` — top of book,
не full L2. Public extracts не дают нужный price-level delta path. BitMEX также предупреждает,
что public data может быть неполным, а hidden orders не видны; см.
[`Exchange Rules, section 6`](https://www.bitmex.com/legal/exchange-rules).

Ничего из BitMEX локально не скачано.

### 3.5. Hyperliquid — L2 в requester-pays S3

Официальная страница
[`Historical data`](https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data)
описывает:

- L2 book snapshots:
  `s3://hyperliquid-archive/market_data/<date>/<hour>/l2Book/<coin>.lz4`;
- daily asset contexts: `s3://hyperliquid-archive/asset_ctxs/<date>.csv.lz4`;
- fills/trades и node data в `s3://hl-mainnet-node-data/...`.

Bucket работает в requester-pays mode; uploader прямо не гарантирует timely updates и
предупреждает о возможных missing data. S3 market archive не предоставляет candles или spot
asset data; дополнительные channels нужно получать через API или писать собственным recorder.
Node path может генерировать raw book diffs и trades, но
[`L1 data schemas`](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/nodes/l1-data-schemas)
оценивают default node logs примерно в 100 GB/day, поэтому storage contract нужно проектировать
заранее.

Источник потенциально пригоден для отдельного on-chain-perp experiment, но не является
same-venue reference для OKX и не может быть подключён к завершённому protocol задним числом.
Ничего из Hyperliquid локально не скачано.

### 3.6. Сторонние multi-venue vendors

#### Tardis.dev

[`Downloadable CSV files`](https://docs.tardis.dev/downloadable-csv-files) включают
`incremental_book_L2`, `book_snapshot_5/25`, trades, quotes, derivative tickers и liquidations.
Для incremental L2 сохраняются exchange и local timestamps в microseconds, `is_snapshot`,
side, price и absolute level amount; zero amount удаляет level. Также доступен
exchange-native historical replay в NDJSON через
[`API/clients`](https://docs.tardis.dev/api/getting-started).

Первый день каждого месяца доступен как sample без API key; полный history коммерческий.
Для multi-venue exact L2 это технически наиболее прямой fallback, но до использования нужны
subscription cost, vendor-version pin, exchange coverage audit и условия публикации derived
artifacts.

#### Kaiko

Kaiko предоставляет daily tick-level
[`best bids/asks`](https://docs.kaiko.com/cloud-delivery/data-feeds/level-1-tick-level/best-bids-and-asks-top-of-book)
и [`all trades`](https://docs.kaiko.com/cloud-delivery/data-feeds/level-1-tick-level/all-trades).
[`Raw order-book snapshots`](https://docs.kaiko.com/cloud-delivery/data-feeds/level-2-aggregations/raw-order-book-snapshot)
снимаются каждые 30 seconds до 10% depth.

Tick BBO подходит для top-of-book controls, но 30-second depth snapshots не восстанавливают
event-by-event queue/depth path. Это коммерческий источник; API key, coverage и лицензия должны
быть заморожены в новом source contract.

## 4. Фактически скачанные данные OKX

### 4.1. Расположение и identity

Все raw assets находятся в:

```text
data/okx/emp_data_001/
├── raw_manifest.json
└── raw/
    ├── funding/
    ├── orderbook_l2/
    └── trades/
```

`data/` исключён из Git через `.gitignore`. Эти файлы существуют только в текущем local
workspace или его backup и не появятся после обычного `git clone`.

Dataset identity:

| Поле | Значение |
|---|---|
| `dataset_id` | `okx-btc-p5-audit-2024-v1` |
| `schema_version` manifest | `okx-raw-manifest-v1` |
| `retrieved_at_utc` final manifest | `2026-08-11T23:20:57.402875Z` |
| asset count | `42` |
| compressed asset bytes | `4,428,637,877` (`4.428638 GB`, `4.124490 GiB`) |
| `dataset_content_sha256` | `0e3a6d6e99586b72ccc237bde7f8df4c3651ba4bd4495b391d9a20771c0e3888` |
| source-list SHA-256 | `16036c12ff5813c41f2ebd574b72f06d99ee093a3f78f447e0b1ba9b18426d45` |
| local manifest size | `26,326` bytes |

`raw_manifest.json` содержит для каждого asset `asset_id`, channel, instrument, source URL,
relative path, bytes и SHA-256. В final fetch `downloaded=true` стоит у 10 файлов, а
`downloaded=false` у 32. Это не означает отсутствие 32 файлов: fetcher нашёл их после
предыдущих attempts, повторно проверил expected size и SHA-256 и не стал скачивать заново.

### 4.2. Полный состав

| Channel / instrument | Assets | Compressed bytes | Labels |
|---|---:|---:|---|
| L2 `BTC-USDT-SWAP` | 7 | `3,576,320,890` | `2024-01-15`, `03-15`, `05-15`, `07-15`, `09-15`, `11-15`, `12-15` |
| L2 `BTC-USDT` spot | 3 | `679,047,395` | `2024-01-15`, `07-15`, `12-15` |
| trades `BTC-USDT-SWAP` | 14 | `148,007,572` | каждая из 7 audit dates плюс следующий calendar label `D+1` |
| trades `BTC-USDT` spot | 6 | `25,245,739` | `2024-01-15/16`, `07-15/16`, `12-15/16` |
| funding `BTC-USDT-SWAP` | 12 | `16,281` | monthly `2024-01`--`2024-12` |
| **Итого** | **42** | **`4,428,637,877`** | frozen P5 train sample |

Пары trade archives `D` и `D+1` нужны потому, что OKX trade filenames используют UTC+8
calendar cuts. Для полного UTC audit day `D` loader берёт релевантные интервалы из обоих
архивов. L2 archive labels соответствуют UTC day, но timestamps всё равно валидируются по
содержимому. Funding month labels также не заменяют фильтрацию по epoch timestamp.

### 4.3. Raw formats

#### L2 order book

Path pattern:

```text
data/okx/emp_data_001/raw/orderbook_l2/
  <instrument>-L2orderbook-400lv-YYYY-MM-DD.tar.gz
```

Каждый `tar.gz` содержит один `.data` file в NDJSON: один JSON object на строку.

```json
{
  "instId": "BTC-USDT-SWAP",
  "action": "snapshot",
  "ts": "1705276800006",
  "asks": [["41738.4", "240.0", "10"]],
  "bids": [["41738.3", "503.0", "11"]]
}
```

- `action`: `snapshot` или `update`;
- `ts`: exchange Unix timestamp в milliseconds, сохранён строкой;
- level: `[price, size, order_count]`, все numeric values в raw JSON — strings;
- `size == "0.0"` удаляет price level;
- ordering key в project parser: `(timestamp_ms, source_row_index)`;
- sequence ID отсутствует; recovery возможен только на новом full snapshot;
- downstream prices переводятся в exact integer ticks (`0.1 USDT` для frozen contract).

#### Trades

Path pattern:

```text
data/okx/emp_data_001/raw/trades/
  <instrument>-trades-YYYY-MM-DD.zip
```

Каждый ZIP содержит один CSV:

```csv
instrument_name,trade_id,side,price,size,created_time
BTC-USDT-SWAP,720073478,buy,42897.9,8.0,1705248000179
```

`created_time` — Unix milliseconds. `side` интерпретируется как trade side из source schema;
`size` остаётся в raw exchange units (`contracts` для swap, base asset для spot) до явного
instrument-aware conversion.

#### Funding

Path pattern:

```text
data/okx/emp_data_001/raw/funding/
  BTC-USDT-SWAP-fundingrates-YYYY-MM.zip
```

Каждый ZIP содержит один CSV:

```csv
instrument_name,funding_rate,funding_time
BTC-USDT-SWAP,0.0002955843024083,1704038400000
```

`funding_time` — Unix milliseconds; `funding_rate` — decimal rate, не percent string.

### 4.4. Derived local artifacts

Raw archives не преобразованы в единый committed Parquet dataset. Производные данные и audit
results находятся в `outputs/`, который также исключён из Git.

Canonical P5 audit:

```text
outputs/EMP-DATA-001/20260811T232210534423Z-45f5a299b7ff-det/
├── metrics/
│   ├── raw_manifest.json
│   ├── file_quality.csv
│   ├── summary.json
│   └── bootstrap.json
├── tables/
│   ├── day_quality.csv
│   ├── funding_quality.csv
│   └── split_freeze.csv
├── source_config.toml
├── resolved_runspec.json
└── manifest.json
```

Это metrics/provenance, а не копия всех market events. `metrics/raw_manifest.json` — frozen
copy local raw manifest на момент run.

Canonical P6 BBO extracts:

```text
outputs/EMP-FILTER-001/20260812T000514761846Z-7075bc32601b-det/state/
├── extraction_manifest.json
├── spot-l2-<date>.pt        # 3 files
├── swap-l2-<date>.pt        # 7 files
└── audit_filter.pt
```

Ten `*-l2-<date>.pt` files имеют schema `bbo-events-v1` и являются PyTorch-serialized
dictionaries с полями:

- metadata: `asset_id`, `date`, `instrument`, `instrument_type`;
- tensors: `timestamps_ms:int64`, `bid_ticks:int64`, `ask_ticks:int64`,
  `snapshot_reset:bool`.

`audit_filter.pt` содержит audit timestamps, filtered efficient price, posterior variance,
causal reference price и reference timestamps. `.pt` нужно загружать только как trusted local
artifact; это Python/PyTorch serialization, не exchange-neutral interchange format.

Последующие canonical empirical filter outputs используют те же verified P6 extracts:

| Experiment | Canonical local directory | Основной state format |
|---|---|---|
| `EMP-MARK-FILTER-001` | `outputs/EMP-MARK-FILTER-001/20260812T063536959101Z-9956cb3f2077-det/` | `state/december_filter.pt` |
| `EMP-MARK-CT-001` | `outputs/EMP-MARK-CT-001/20260812T100151852237Z-c8a620999b93-det/` | `state/december_filter.pt` |
| `EMP-MARK-FACT-001` | `outputs/EMP-MARK-FACT-001/20260812T105127206423Z-44416f08cb43-det/` | `state/december_factorized_filter.pt` |

Failed/diagnostic runs рядом сохранены намеренно и не являются canonical dataset versions.
Canonical run IDs перечислены в [`docs/plan.md`](../plan.md) и
[`final-synthesis.md`](final-synthesis.md).

## 5. Acquisition и проверка local dataset

Единственная команда проекта, выполняющая network download:

```bash
uv run ot-micromr fetch-data cfg/experiments/emp_data_001_sources.toml
```

Fetcher:

1. читает frozen URL/size contract;
2. пишет новый файл через `.part` и поддерживает HTTP Range resume;
3. проверяет exact byte size;
4. считает SHA-256 streaming pass;
5. атомарно публикует asset и записывает `raw_manifest.json`;
6. для уже существующего файла повторно проверяет size/hash и не загружает payload заново.

Если `data/` отсутствует, команда загрузит около `4.43 GB`; automatic download при import,
config validation или experiment run отсутствует.

После acquisition выполняются:

```bash
uv run ot-micromr validate-config cfg/experiments/emp_data_001.toml
uv run ot-micromr run cfg/experiments/emp_data_001.toml
```

P5 audit проверяет size/hash identity, L2 reconstruction and quarantine, trade ordering и
UTC+8 cuts, funding coverage, spot/swap overlap и large-tick eligibility. Итоговый report:
[`p5-okx-data-feasibility.md`](p5-okx-data-feasibility.md).

## 6. Storage, backup и publication boundary

- `data/`, `outputs/`, `*.parquet` и `*.pt` исключены из Git.
- Raw OKX archives и raw-reconstructible BBO extracts нельзя публиковать или передавать
  третьим лицам по принятой интерпретации OKX personal-use terms.
- В backup должны вместе попадать `data/okx/emp_data_001/raw/`, `raw_manifest.json` и tracked
  `cfg/experiments/emp_data_001_sources.toml`; без source list manifest недостаточен для
  повторного acquisition audit.
- Dataset identity определяется ordered tuples `(asset_id, size_bytes, sha256)`, а не file
  mtime или `retrieved_at_utc`.
- Перед удалением local raw data нужно проверить, что downstream canonical runs и будущая
  reproduction больше не требуют его; Git восстановить raw assets не сможет.
- Для public paper/repository допустимы агрегированные metrics и figures только после
  отдельной проверки, что они не позволяют фактически перераспределить raw data.

## 7. Если data track будет возобновлён

Новый source должен пройти отдельный preregistered feasibility stage до просмотра target P&L:

1. заморозить venue, instruments, dates, channel schemas, exact URLs и license boundary;
2. получить небольшой smoke day и проверить snapshots/deltas, timestamps, units, sequences,
   duplicates, gaps и recovery;
3. оценить full-period storage/runtime до массовой загрузки;
4. создать immutable source list и content manifest с SHA-256;
5. определить causal cross-venue alignment и latency/basis controls;
6. заморозить новый chronological split и independent unit;
7. только после data-quality gate разрешить estimator или strategy evaluation.

Предпочтительный порядок кандидатов зависит от цели:

- для расширения same-venue OKX study — дополнительные официальные OKX days/channels;
- для BBO/trade robustness — BitMEX или Binance на отдельно совпадающем interval;
- для full multi-venue L2 — Tardis.dev либо independently audited Hyperliquid archive;
- Bybit — после подтверждения immutable historical L2 access, а не только current API;
- Kaiko periodic snapshots — как lower-frequency control, не замена event execution data.

Любой такой запуск является новым research track. Он не меняет отрицательный итог текущего
book-only OKX experiment и не открывает завершённый holdout задним числом.
