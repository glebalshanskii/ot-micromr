# ADR-0011: OKX-only empirical data and same-venue efficient-price estimates

- **Статус:** Accepted
- **Дата:** 2026-08-12
- **Связанный этап:** P5--P9
- **Связанный план:** [`docs/plan.md`](../plan.md)

## Контекст

Empirical track является личным некоммерческим исследованием и разработкой собственной
торговой стратегии. Он не является открытым исследовательским data project и не требует
права распространять raw market data.

OKX публикует tick-level trades с сентября 2021 года, perpetual funding с марта 2022 года
и high-resolution L2 order-book data с марта 2023 года:
[`OKX Historical Market Data`](https://www.okx.com/historical-data). Дополнительные условия
прямо включают possession, retention и использование данных для разработки собственной
торговой стратегии в разрешённый personal use, но запрещают redistribution/sublicensing и
оставляют лицензию revocable:
[`OKX Historical Data Terms`](https://www.okx.com/en-us/help/historicaldata-terms-and-conditions).

Cross-venue reference может уменьшить venue-specific noise, но одновременно добавляет
clock alignment, transport latency, instrument basis, venue outages и новый data contract.
Для первого causal backtest эта сложность не нужна. Пользователь отдельно решил пока не
использовать Binance или другие биржи.

## Решение

1. **Единственная площадка текущего empirical baseline — OKX.** Binance, Bybit, BitMEX,
   Hyperliquid и любые другие cross-venue feeds не входят в P5--P9 без нового ADR/protocol
   amendment, сделанного до просмотра затрагиваемого holdout P&L.
2. Первый feasibility candidate — `BTC-USDT-SWAP`. Необходимые execution channels:
   high-resolution L2 order book, tick-level trades, historical funding, instrument/tick/
   contract metadata и доступные venue-status records.
3. Разрешённые causal candidates для оценки latent efficient price $X_t$ используют только
   OKX observations, доступные к decision time:
   - фильтрованный midpoint/microprice и event flow самого `BTC-USDT-SWAP`;
   - синхронный OKX spot `BTC-USDT` order book/trades;
   - OKX index/mark price, если P5 подтвердит достаточные timestamp semantics и resolution.
4. Выбор между этими same-venue estimators относится к P6. Он делается по train-only
   filtering likelihood, calibration и causal diagnostics, а не по strategy P&L. Ни один
   estimator пока не объявляется победителем.
5. Raw OKX archives хранятся локально вне Git. Download выполняется только явной командой;
   каждый файл получает source URL, size, retrieval timestamp и SHA-256 в immutable manifest.
   Raw data и воспроизводящие их extracts нельзя публиковать или перераспределять.
6. Текущий stage требует права на personal local use, а не права публиковать derived metrics.
   Если цель проекта изменится на публикацию, commercial use или передачу данных третьим
   лицам, data license и допустимые artifacts пересматриваются до такого использования.
7. Source selection не означает, что P5 gate уже пройден. Отдельно проверяются archive
   schema, ordering/continuity, timestamp resolution, missing intervals, spot--swap alignment,
   tick/contract history, fees, funding и storage/compute feasibility. Неоднозначные intervals
   исключаются по заранее заданному recovery rule.
8. Финальные calendar boundaries, universe и train/validation/test split замораживаются только
   после data-quality pilot, но до любого strategy P&L inspection. Расширение с BTC на ETH/SOL
   допускается только по preregistered train-only large-tick eligibility rule.

## Последствия

- OKX получает статус selected source, а P5 переходит из `pending-data-source` в
  `in-progress`.
- Overlapping empirical history не может начинаться раньше доступного L2 coverage в марте
  2023 года; точные даты будут определены manifest audit.
- Cross-venue price не является обязательным feature, control или robustness gate текущей
  стратегии.
- Same-venue spot/index candidates всё ещё требуют causal time alignment и basis diagnostics:
  принадлежность одной площадке не устраняет look-ahead или spot--perpetual basis.
- Exact historical account fee tier может отсутствовать в archive. В этом случае до backtest
  фиксируются документированный base fee и conservative fee stresses; funding учитывается по
  историческому каналу.
- Отзыв лицензии, недоступность данных в применимой юрисдикции или невозможность однозначно
  восстановить event order переводят P5 в `blocked-data`, а не разрешают тихую замену candles.
