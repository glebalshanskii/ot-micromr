# P10: итоговый синтез и закрытие проекта

- **Date:** 2026-08-12
- **Итог проекта:** `negative`
- **Воспроизведение статьи:** analytical results reproduced; exact-jump figures
  independently and partially reproduced
- **Empirical strategy:** profitability `not-confirmed / not tested`
- **Decision:** [`ADR-0021`](../adr/0021-close-current-empirical-track.md)
- **Paper:** Amaral, *Optimal Trading of Microstructure Mean Reversion*,
  `arXiv:2608.00885v1`

## Итог

В проекте воспроизведены closed-form расчёты Gaussian surrogate и проверен независимый
controlled jump simulator. Simulator восстановил balanced-flow mechanism и deliberately
unbalanced control. Figures 2 и 5 воспроизведены структурно. В Figure 4 оптимальный
jump-model band сдвинулся внутрь в двух из трёх parameter rows, но полная
multiplicity-adjusted scientific family и numerical-refinement family остались
inconclusive. Автор не опубликовал primitives, seeds, horizon или code, поэтому это
independent partial reproduction, а не точное повторение author run.

Empirical extension остановлен до backtest. OKX `BTC-USDT-SWAP` прошёл frozen large-tick
и data-quality gate, но стратегии статьи требуется latent gap $G_t=M_t-X_t$. Exact,
marked, continuous-hazard и factorized causal filters не смогли оценить этот state с
достаточной точностью. Финальная factorized model откалибровала event-clock moments, однако
её gap-dependent mark term не дал held-out predictive gain, а posterior uncertainty
составила `8.426` option margins. После такого результата P&L search оптимизировал бы noise,
а не проверял mechanism статьи.

Поэтому итоговый label равен `negative` для проверенного book-only empirical approach.
Утверждения об убыточности реальной стратегии нет: orders, fills и P&L не рассчитывались,
следовательно, profitability имеет статус `not-confirmed / not tested`.

## Матрица claims и evidence

| Claim | Итоговый статус | Evidence |
|---|---|---|
| Dawson optimum, Kramers approximation и Figure 3 curves | **reproduced** | [`P2 report`](p2-analytical-reproduction.md), `ANA-SMOKE-001`, `ANA-FIG3-001`, configs [`ana_smoke_001.toml`](../../cfg/experiments/ana_smoke_001.toml) и [`ana_fig3_001.toml`](../../cfg/experiments/ana_fig3_001.toml) |
| Balanced jump-flow identity и planted unbalanced response | **supported в controlled simulation** | [`P3V report`](p3v-sensitivity-and-power.md), `SIM-MOMENTS-002`, `SIM-UNBALANCED-002` |
| Figure 2 book mechanism | **reproduced illustratively** | [`P4 report`](paper-reproduction.md), [`figure2.png`](../../outputs/SIM-FIG4-002/20260811T211511185484Z-e51b9cf3d54d-det/figures/figure2.png) |
| Figure 4: inward jump-model optimum и rate loss | **partially reproduced; family inconclusive** | [`P4 report`](paper-reproduction.md), `SIM-FIG4-002`, [`figure4.png`](../../outputs/SIM-FIG4-002/20260811T211511185484Z-e51b9cf3d54d-det/figures/figure4.png) и [`figure4-data.csv`](../../outputs/SIM-FIG4-002/20260811T211511185484Z-e51b9cf3d54d-det/figures/figure4-data.csv) |
| Figure 5: one-lot entry и two-lot flip mechanics | **reproduced illustratively** | [`P4 report`](paper-reproduction.md), [`figure5.png`](../../outputs/SIM-FIG4-002/20260811T211511185484Z-e51b9cf3d54d-det/figures/figure5.png) |
| Frozen OKX sample пригоден, а `BTC-USDT-SWAP` является large-tick | **supported на семи train days 2024 года** | [`P5 report`](p5-okx-data-feasibility.md), `EMP-DATA-001` |
| Six-event filter восстанавливает $X$ в собственной generative model | **supported synthetically** | [`P6 report`](p6-causal-efficient-price.md), `FILTER-SYN-001` |
| Paper-faithful six-event filter применим на OKX | **negative** | [`P6 report`](p6-causal-efficient-price.md), `EMP-FILTER-001`; uncertainty/margin `2.759` |
| Marked multi-spread extension идентифицирует usable real-data gap | **negative** | [`P6M report`](p6m-marked-multi-spread.md), `EMP-MARK-FILTER-001`; event layer улучшен, state uncertainty осталась слишком большой |
| Continuous integrated hazard устраняет empirical failure | **negative** | [`P6C report`](p6c-continuous-hazard.md), `EMP-MARK-CT-001`; refinement passed, state/calibration failed |
| Отдельный event clock откалиброван по moments | **supported на registered margins** | [`P6D report`](p6d-factorized-clock-mark.md), `EMP-MARK-FACT-001`; rescaling mean/SD `1.069/1.138` |
| Conditional BBO mark содержит полезную latent-gap information | **not supported; negative center** | [`P6D report`](p6d-factorized-clock-mark.md); gain `-0.0000834 nat/event`, верхняя граница interval практически равна нулю |
| Стратегия доходна на реальных crypto data | **not-confirmed / not tested** | P7--P9 не выполнялись после failed causal-state precondition |

## Provenance канонических runs

Ниже перечислены 12 immutable runs, использованных в release synthesis. Полные commit,
config, seed, software, hardware и artifact hashes сохранены в каждом local manifest.

| Experiment | Run ID | Commit | Runtime | Статус run | RunSpec prefix |
|---|---|---:|---:|---|---:|
| `ANA-SMOKE-001` | `20260811T170052058822Z-290ea5809cb6-det` | `710efa9b` | `0.03 s` | passed | `290ea5809cb6` |
| `ANA-FIG3-001` | `20260811T170104389894Z-4c1014e843c6-det` | `710efa9b` | `1.11 s` | passed | `4c1014e843c6` |
| `SIM-MOMENTS-002` | `20260811T184531842286Z-4cb501542645-det` | `9dedb6b4` | `1189.46 s` | passed | `4cb501542645` |
| `SIM-UNBALANCED-002` | `20260811T190621510298Z-f3c0ff8a3b29-det` | `9dedb6b4` | `583.86 s` | passed | `f3c0ff8a3b29` |
| `SIM-FIG4-002` | `20260811T211511185484Z-e51b9cf3d54d-det` | `5a83b655` | `28.99 s` | operationally passed | `e51b9cf3d54d` |
| `EMP-DATA-001` | `20260811T232210534423Z-45f5a299b7ff-det` | `6579adfa` | `186.37 s` | passed | `45f5a299b7ff` |
| `FILTER-SYN-001` | `20260811T234700354892Z-9e7f2939b506-det` | `4cf3212d` | `64.60 s` | passed | `9e7f2939b506` |
| `EMP-FILTER-001` | `20260812T000514761846Z-7075bc32601b-det` | `cd5aabf0` | `155.51 s` | scientific negative | `7075bc32601b` |
| `FILTER-MARK-SYN-001` | `20260812T061258615041Z-6daac30b7613-det` | `e065c0d0` | `129.50 s` | passed | `6daac30b7613` |
| `EMP-MARK-FILTER-001` | `20260812T063536959101Z-9956cb3f2077-det` | `ca248141` | `111.65 s` | scientific negative | `9956cb3f2077` |
| `EMP-MARK-CT-001` | `20260812T100151852237Z-c8a620999b93-det` | `6b2306e1` | `126.13 s` | scientific negative | `c8a620999b93` |
| `EMP-MARK-FACT-001` | `20260812T105127206423Z-44416f08cb43-det` | `be0f33d6` | `127.62 s` | scientific negative | `44416f08cb43` |

Суммарное время canonical runs на записанном hardware равно `2704.82 s` (`45.08 min`).
Это не весь project compute: pilot, sensitivity, historical failed и technical repair runs
намеренно не входят в canonical release table и остаются описанными в stage reports и
локальных директориях `outputs/`.

## Search budget, deviations и negative evidence

- Trading thresholds, execution parameters и P&L cells не перебирались. P7--P9 не
  начинались, frozen validation/test periods 2025 года для этой hypothesis не открывались.
- Empirical development включал четыре последовательные, мотивированные механизмом
  estimator families на одних development data 2024 года: P6, P6M, P6C и P6D. Это
  development evidence, а не четыре независимых подтверждения.
- Исходные point-estimate gates P3 были проаудированы и prospectively заменены на
  SESOI-based equivalence/superiority tests, multiplicity correction и powered new runs.
  Historical failures не переименовывались в passes.
- Figure 4 использовала 30 strategy seeds, замороженных по pilot variance. После просмотра
  target не добавлялись seeds или horizon extensions. Author-level exact reproduction
  невозможна, потому что в статье нет primitive parameters и simulation details.
- В P5 сохранены два failed precursor runs, обнаружившие snapshot warm-up,
  impossible-book и UTC-calendar issues до canonical run.
- В P6M сохранены artifact-layer failed attempts. В P6D сохранены попытки с отсутствующим
  CUDA kernel и nondeterministic replay; оба дефекта исправлены без изменения frozen model,
  config или scientific gates.
- Local raw data и outputs намеренно игнорируются Git. Reports и manifests фиксируют пути
  и hashes; clone репозитория не загружает и не перераспределяет эти данные.

## Artifact и release audit

2026-08-12 этап P10 заново рассчитал inventory каждого canonical run и сравнил relative
path, byte size и SHA-256 с immutable `manifest.json`. Прошли все `12` manifests: `143`
declared artifacts, `57,578,544` bytes, без missing, extra или changed files. Идентичность
experiment/run directories также совпала с полями manifests.

Дополнительно release validation выполнила:

```bash
uv sync --locked
uv run python main.py --help
uv run ot-micromr --help
uv run python -m unittest discover -s tests -t . -v
```

Оба entrypoints завершились успешно, все `97` tests прошли за `7.811 s`. Затем из того же
lockfile создана отдельная environment под `/tmp` командой
`UV_PROJECT_ENVIRONMENT=<fresh-path> uv sync --locked`; в ней прошли оба entrypoints и
focused suite `tests.test_analytics`, `tests.test_config`, `tests.test_cli` (`30/30`). Fresh
environment использовала локальный package cache `uv`, но имела собственные заново созданные
virtual environment и installation.

## Ограничения и границы результата

- Статья является preliminary theoretical arXiv preprint без official code или data.
- Synthetic simulations используют документированную controlled approximation; они не
  доказывают global optimality threshold policy на exact jump process.
- Empirical sample покрывает выбранные OKX days 2024 года для одного primary swap
  instrument и не устанавливает универсальное свойство crypto markets.
- Same-venue spot — causal diagnostic proxy, а не наблюдаемый ground-truth $X_t$.
- P6D clock прошёл только registered mean и SD margins; duration distribution не
  воспроизводит резкий exchange-timestamp mass около `0.01 s`.
- Отсутствие predictive information в проверенном BBO state не исключает signals из
  trades, depth, order flow, liquidations, funding или independent venues. Их проверка —
  новый проект, а не переинтерпретация этого результата.

## Финальное решение

P0--P6D и P10 завершены. P7--P9 намеренно не выполнялись, потому что их causal-state
precondition не пройдена. Репозиторий выпускается как аудируемое partial paper reproduction
и информативное negative empirical feasibility study. Внутри текущего плана следующего
experiment нет.
