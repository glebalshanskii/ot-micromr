# P6M: marked multi-spread causal filtering

- **Статус:** completed; synthetic supported, empirical latent-state usability negative
- **Дата:** 2026-08-12
- **Protocol:** [`p6m-marked-multi-spread.md`](../protocols/empirical/p6m-marked-multi-spread.md)
- **Decision:** [`ADR-0016`](../adr/0016-p6m-negative-latent-state-usability.md)
- **Synthetic config:** [`filter_mark_syn_001.toml`](../../cfg/experiments/filter_mark_syn_001.toml)
- **Empirical config:** [`emp_mark_filter_001.toml`](../../cfg/experiments/emp_mark_filter_001.toml)

## Result

P6M устранил главную техническую потерю P6: marked filter обработал `100%` healthy
multi-tick и multi-spread BBO transitions без artificial reset. Full model намного лучше
gap-independent proper-score baseline предсказывает следующий event, а оба новых
компонента имеют отдельный статистически значимый вклад.

Это не сделало latent efficient price пригодным для стратегии. Posterior uncertainty
статистически значимо превышает даже optimistic option-value margin, а event time-rescaling
далёк от equivalence region. Поэтому итог P6M — `negative`, P7/P8 не разблокированы.
Orders, fills, thresholds и P&L не рассчитывались.

## Provenance

| Leg | Run | Clean commit | Status | Runtime | Manifest SHA-256 |
|---|---|---|---:|---:|---|
| Synthetic target | `FILTER-MARK-SYN-001/20260812T061258615041Z-6daac30b7613-det` | `e065c0d0` | passed | `129.50 s` | `3042afd41a7b9acc0bd3d343527c8553ecad20dc63cc2762bbafc5af5698bfb9` |
| Empirical target | `EMP-MARK-FILTER-001/20260812T063536959101Z-9956cb3f2077-det` | `ca248141` | acceptance failed / scientific negative | `111.65 s` | `1337e43d879736275dadf64bb21a093869b6947c226cd8f415496c1358463390` |

Оба target runs выполнены на NVIDIA GeForce RTX 3080 Ti Laptop GPU, PyTorch `2.13.0`,
Python `3.14.0`: state/particle kernels — CUDA `float32`, final inference — PyTorch
`float64`, compiled mode — `reduce-overhead`. Empirical RunSpec SHA-256:
`9956cb3f2077c62d323bb1c264f0c93b7c1217c20a62e38cd4e3ccfd5435935c`.

### Сохранённые неуспешные попытки

- Synthetic run `20260812T010035485774Z-737bceddd9b4-det` прошёл научные metrics, но
  обнаружил слишком крупный frozen step probability и CUDA boundary bug в floating
  magnitude buckets. Датированный protocol amendment зафиксировал refinement; failed run
  сохранён.
- Три empirical attempts `20260812T062729702730Z-9956cb3f2077-det`,
  `20260812T063008668431Z-9956cb3f2077-det` и
  `20260812T063256692342Z-9956cb3f2077-det` полностью вычислили folds, но остановились
  при записи sparse diagnostic CSV/state directory. Это artifact-layer failures без
  scientific decision. Исправления покрыты tests; единственный decision-bearing empirical
  run указан в основной provenance table.

## Method and data

Fixed observed model использует previous-spread buckets `1..7,8+` и 729 coarsened marks
на spread: price direction, spread direction и power-of-two magnitude buckets для midpoint
и spread jumps. Unseen cells получают train-only Dirichlet smoothing `0.01`; exact bid/ask
ticks остаются доступны через content-addressed P6 dependency.

Theory-constrained intensity имеет full event support и нормирует corrective directional
first moments до единицы, сохраняя conditional drift $-\alpha G$. Comparator имеет тот же
mark support, но gap-independent intensities. Controls отключают по одному multi-tick и
multi-spread correction; unconstrained model сохраняет raw directional asymmetry.

Empirical evaluation использовала все существующие OKX BTC-USDT-SWAP даты:

`2024-01-15, 2024-03-15, 2024-05-15, 2024-07-15, 2024-09-15,
2024-11-15, 2024-12-15`.

В каждом из шести rolling-origin folds fit использовал только более ранние даты, а
следующая дата была held out. Всего получено ровно 288 preregistered 30-minute blocks,
что выше minimum precision budget 117. `2024-12-15` использован полноценно; aggregate без
него является только sensitivity check. Same-venue spot доступен как causal as-of
diagnostic на July/December held-out folds и никогда не входит в particle update.

## Synthetic identification

Known-$X$ experiment прошёл все 11 gates на 64 independent sessions и 44,011 measured
events:

| Estimand | Mean | Adjusted one-sided lower bound | Gate | Decision |
|---|---:|---:|---:|---:|
| `1 - RMSE(PF)/RMSE(mid)` | `0.49954` | `0.47969` | `> 0.10` | passed |
| PF minus midpoint plug-in log score | `0.04405 nat/event` | `0.04173` | `> 0.01` | passed |
| 90% posterior coverage | `0.88630` | equivalence CI `[0.87391,0.89868]` | inside `[0.85,0.95]` | passed |

PF RMSE равен `2.02589`, midpoint RMSE — `4.06466`. Baseline drift error равен нулю,
directional correction error `1.19e-7`, maximum one-step event probability `0.07193`,
replay digest совпал bitwise. Следовательно, empirical failure нельзя объяснить тем, что
filter вообще не способен восстанавливать известное synthetic state своей model class.

## Empirical operational checks

| Check | Result |
|---|---:|
| Healthy transitions supported | `1,122,613 / 1,122,613` (`100%`) |
| Valid inference blocks | `288 / 288` |
| Future timestamp accesses | `0` |
| Finite filter outputs | passed |
| Positive posterior variance | passed |
| Frozen fold parameters | passed |
| Deterministic December replay | bitwise exact |
| Wall time | `111.65 s < 600 s` |

December filter digest:
`5fd1d41a044e5bb059dc59c5ccf95dfa3d5635767fb7e6ad56c697375361f8e0`.
Это подтверждает, что результат valid и не вызван прежним unsupported-event reset.

## Empirical scientific decisions

### Primary usability family

| Estimand | Mean | Adjusted bound / 95% interval | Frozen boundary | Decision |
|---|---:|---:|---:|---:|
| Full minus residual log score | `0.30486 nat/event` | lower `0.26239` | `> 0.01` | superior |
| `1 - median posterior SD / option margin` | `-0.60873` | `[-0.69000,-0.52747]` | `> 0` | **inferior / negative** |

Log-score improvement положителен во всех 288 blocks. Но joint Bonferroni family требует
обе метрики; сильная event prediction не компенсирует непригодную state uncertainty.
Среднее block ratio `posterior SD / option margin` равно `1.6087`, то есть uncertainty
примерно на 61% больше optimistic margin.

### Calibration and model integrity

| Estimand | Mean | 95% interval | Equivalence/non-inferiority region | Decision |
|---|---:|---:|---:|---:|
| Time-rescaling block mean | `2.20828` | `[1.97583,2.44073]` | `[0.9,1.1]` | above region |
| Time-rescaling block SD | `5.32846` | `[4.75586,5.90105]` | `[0.8,1.2]` | above region |
| Constrained minus unconstrained score | `0.001919` | lower `0.001420` | `> -0.005` | non-inferior |

Calibration intervals не просто не смогли доказать equivalence: они целиком лежат выше
допустимых regions. Marked point process переоценивает integrated event intensity и имеет
слишком тяжёлую rescaling dispersion на held-out blocks.

Target inference CSV использовал старый reporting label `inconclusive` для любого
непрошедшего superiority/equivalence test. После run reporting-only logic уточнена:
`inferior`, если adjusted upper bound ниже minimum, и `above_equivalence_region`, если
adjusted lower bound выше верхней margin. Приведённые labels вычислены из immutable block
vectors; thresholds, metrics и acceptance decision не изменялись, повторный scientific
run для переименования не выполнялся.

### Component attribution

| Ablation contrast | Mean | One-sided lower bound | Holm adjusted p | Decision |
|---|---:|---:|---:|---:|
| Full minus no-multi-tick | `0.25394 nat/event` | `0.22586` | `< 2e-14` | supported |
| Full minus no-multi-spread | `0.07234 nat/event` | `0.06143` | `< 2e-14` | supported |

Multi-tick component улучшил score во всех blocks; multi-spread — в 236 из 288. Значит,
обе группы переходов действительно несут predictive information, а их добавление не было
только механическим ростом support.

## December reuse and sensitivity

`2024-12-15` не исключался из новой модели. Он был шестым held-out fold, а его filter
повторён bitwise. Обязательный descriptive aggregate без декабря оставил mean log-score
improvement `0.30282`, но uncertainty metric `-0.61467`; его 97.5% interval
`[-0.70923,-0.52010]` целиком ниже нуля. Следовательно, отрицательный usability conclusion
не создаётся декабрьским днём и сохраняется при его исключении.

## Interpretation

P6M разделил две разные задачи:

1. **Предсказать, какое BBO-событие произойдёт следующим.** Здесь marked representation
   полезна и явно лучше gap-independent comparator.
2. **Точно восстановить ненаблюдаемый efficient price.** Здесь book flow не даёт достаточно
   узкий и calibrated posterior для пороговой стратегии статьи.

Relative proper score отвечает только на первый вопрос. Даже лучший из двух likelihood
models может быть абсолютно miscalibrated и неидентифицирующим для latent state.

Наблюдаемый diagnostic согласуется с этим: train-only $s_G$ менялся от `40.56` до
`125.24 USDT`, held-out median posterior SD — от `2.60` до `24.53 USDT`. На двух held-out
spot dates filter RMSE к causal spot reference был `123.99` и `146.10 USDT`, тогда как
midpoint RMSE — `5.39` и `9.12 USDT`. Spot не является truth, поэтому это diagnostic, но
он не поддерживает практическое качество filter.

Гипотеза, не установленный факт: 300-second midpoint EWMA смешивает trend с microstructure
gap, а directional mark flow слишком слаб для anchoring Brownian state. Проверка требует
отдельной causal measurement/state model, а не изменения acceptance threshold.

### Descriptive one-step BBO visualization

После завершения target run построен отдельно помеченный descriptive график для первых
10 минут held-out `2024-12-15`. До каждого следующего observation используется фактический
предыдущий BBO и сохранённые causal posterior mean/variance; mark intensities интегрируются
по Gaussian moment approximation, а category переводится в bid/ask delta через train-only
cell mean. Поэтому это условный one-step forecast при известном факте наступления event,
не свободная будущая trajectory и не новый acceptance artifact.

На 317 transitions model/persistence MAE составили `2.146/2.128 USDT` для midpoint и
`0.350/0.086 USDT` для spread. В шести точках raw predicted conditional-mean spread был
неположительным; post-hoc clipping не применялся. Этот diagnostic согласуется с основной
calibration failure и показывает, почему визуальное совпадение absolute price levels нельзя
считать predictive success: обе линии причинно anchored на предыдущем фактическом BBO.

Reproducible command:

```bash
uv run python scripts/plot_p6m_predictions.py \
  outputs/EMP-MARK-FILTER-001/20260812T063536959101Z-9956cb3f2077-det \
  --start-utc 2024-12-15T00:00:00Z --window-minutes 10
```

Отдельный free-running multi-step diagnostic запускает из каждой healthy real origin
`1024` Monte Carlo paths на 10 следующих events. После origin ни actual BBO, ни actual
event time в trajectory не подставляются: модель сама семплирует mark, допустимый exact
train-only bid/ask delta, waiting time и Brownian efficient-price increment. Initial
latent state семплируется из Gaussian moment approximation сохранённых causal posterior
mean/variance. Это
event-index forecast с approximate frozen-hazard timing, соответствующим likelihood
implementation, а не новая fitted model.

Для первых двух минут December holdout получено 60 origins. Midpoint MAE model/persistence
равны `1.795/1.783 USDT` на horizon 1 и `13.033/12.665 USDT` на horizon 10. Модель ожидает
10 events в среднем за `4.24 s`, actual mean — `14.79 s`. Таким образом, она не обгоняет
даже persistence по midpoint и генерирует события примерно в 3.5 раза быстрее наблюдений.

```bash
uv run python scripts/plot_p6m_multistep.py \
  outputs/EMP-MARK-FILTER-001/20260812T063536959101Z-9956cb3f2077-det \
  --start-utc 2024-12-15T00:00:00Z --window-minutes 2 \
  --horizon-events 10 --paths 1024
```

## Decision and next boundary

P6M completed как powered negative latent-state usability result. Marked event layer можно
переиспользовать, но текущий filtered $\widehat X$ запрещено передавать в P7. Поэтому
event-driven backtest и поиск profitability не начинаются.

Если empirical track продолжается, следующий research decision должен выбрать и заранее
зарегистрировать observability extension — causal same-venue spot measurement либо другую
явную state-observation model. Текущие данные, включая декабрь, разрешены для разработки
такой другой модели; отдельный новый период нужен перед окончательным confirmatory P&L
claim, а не потому, что старые данные становятся «запрещёнными».

## Artifacts

Локальные untracked target artifacts:

- `outputs/FILTER-MARK-SYN-001/20260812T061258615041Z-6daac30b7613-det/`;
- `outputs/EMP-MARK-FILTER-001/20260812T063536959101Z-9956cb3f2077-det/`.

Empirical summary SHA-256:
`d55e2ccc76f10b29af82fb5de1c868674d83b7244c4b114ffc33807f0611aa9b`;
block metrics SHA-256:
`39fbe0772d8cc036e71441c549df88a74e4a382d27c9179d3870fe81bac9ba17`;
December state SHA-256:
`bcf5e490c510082be30f83960202fcd4199c6edb8147b47e0a48db34c24ed8d7`.
