# P6: causal efficient-price estimation

- **Статус:** completed; synthetic supported, empirical feasibility negative
- **Дата:** 2026-08-12
- **Protocol:** [`p6-causal-efficient-price.md`](../protocols/empirical/p6-causal-efficient-price.md)
- **Decision:** [`ADR-0014`](../adr/0014-p6-causal-efficient-price-filter.md)
- **Synthetic config:** [`filter_syn_001.toml`](../../cfg/experiments/filter_syn_001.toml)
- **Empirical config:** [`emp_filter_001.toml`](../../cfg/experiments/emp_filter_001.toml)

## Result

P6 подтвердил, что causal six-event particle filter восстанавливает latent price в точно
заданной synthetic model, но не подтвердил практическую точность book-only estimator на
frozen OKX BTC train sample. Empirical uncertainty превышает optimistic option-value margin
в `2.759` раза. По preregistered stop rule этот estimator нельзя передавать в P7/P8 для
поиска P&L; результат классифицируется как `negative`, не `invalid` и не `inconclusive`.

Orders, fills, trading thresholds и P&L в P6 не рассчитывались.

## Provenance

| Leg | Run | Clean commit | Status | Runtime |
|---|---|---|---:|---:|
| Synthetic | `FILTER-SYN-001/20260811T234700354892Z-9e7f2939b506-det` | `4cf3212` | passed | `64.60 s` |
| Empirical | `EMP-FILTER-001/20260811T235854474663Z-7075bc32601b-det` | `06b5053` | negative gate | `165.99 s` |

Оба run выполнены на NVIDIA GeForce RTX 3080 Ti Laptop GPU, PyTorch `2.13.0+cu130`,
Python `3.14.0`. Tensor state использовал CUDA `float32`, final reductions — PyTorch
`float64`; compiled mode — `reduce-overhead`. Empirical extraction использовал 10 независимых
CPU workers и проверил все `70,455,943` frozen L2 source rows. Manifest SHA-256 empirical
run: `9092a588b836921c90c99817cf3650953b3fe958e894a104cb0ba5d707299edb`.

## Synthetic identification

64 independent synthetic sessions содержали `22,190` measured book events. Primary
Bonferroni-adjusted tests дали:

| Estimand | Mean | One-sided lower bound | Minimum useful effect | Decision |
|---|---:|---:|---:|---:|
| `1 - RMSE(PF)/RMSE(current mid)` | `0.17594` | `0.16383` | `0.10` | supported |
| PF minus current-mid log score | `0.29961 nat/event` | `0.28752` | `0.01` | supported |

PF mean RMSE был `1.13270`, current-mid RMSE `1.37663`, causal Kalman RMSE `1.29574`.
90% Gaussian posterior interval имел coverage `0.89398`; equivalence interval
`[0.88824, 0.89973]` полностью лежит внутри preregistered `[0.85, 0.95]`. Maximum
one-event probability `0.08309 < 0.10`. Повторный CUDA filter pass совпал bitwise.

Это показывает, что отрицательный empirical result не объясняется неспособностью реализации
идентифицировать state в собственном generative model.

## Empirical setup

Из P5 archives извлечено `1,594,219` BBO records. Exact six-event transitions статьи
оставлялись только для tight slide, tight-to-open и open-to-tight с допустимым half-tick
jump; любой snapshot, wider spread или иной jump разрывал segment. Causal spot as-of
alignment использовал только observation с timestamp `<=` текущему swap timestamp;
timestamp audit нашёл `0` future accesses.

Frozen chronology:

- fit: `2024-01-15`;
- balanced/unbalanced selection: `2024-07-15`, 48 independent 30-minute blocks;
- untouched audit: `2024-12-15`.

Fit-day reduced estimates: `alpha=0.16486 s^-1`, `s_G=2.80863 USDT`,
`sigma_X=3.04740 USDT/sqrt(s)`. Balanced fit imposed
`2*alpha_s + alpha_o = alpha_c` and obtained final NLL `1.749980 nat/event`.
Unbalanced final NLL отличался менее чем на `1.3e-7 nat/event` in-sample; его OOS
selection lower bound был только `3.83e-7 nat/event`, далеко ниже minimum
`0.01 nat/event`. Поэтому по parsimony был заморожен balanced model.

## Empirical diagnostics and gate

| Metric | Result | Interpretation |
|---|---:|---|
| Exact compatible transition fraction, fit | `15.69%` | большинство BBO transitions разрывает paper-model segment |
| Exact compatible transition fraction, selection | `7.14%` | frequent causal reinitialization |
| Exact compatible transition fraction, audit | `7.73%` | frequent causal reinitialization |
| Median posterior SD | `2.80863 USDT` | empirical estimator uncertainty |
| Optimistic Dawson option margin | `1.01787 USDT` | `theta_D - delta/2`, no fees/latency |
| Uncertainty / margin | `2.759` | required `< 1`; gate failed |
| PF RMSE to causal spot reference | `9.12175 USDT` | descriptive, spot is not ground truth |
| Current-mid RMSE to same reference | `9.12180 USDT` | PF relative improvement only `5.65e-6` |
| Reference coverage by PF 90% interval | `55.22%` | descriptive severe miscalibration under domain shift |

Все operational checks прошли: finite state, strictly positive variance, frozen model,
exact replay и wall-time limit. Единственный failed gate — uncertainty below option margin.
Частые resets объясняют, почему median posterior SD совпадает с fit-day prior scale `s_G`:
при несовместимом transition filter обязан причинно reinitialize state, а не притворяться,
что six-event likelihood описывает наблюдение.

## Limitations and validity

- Causal OKX spot reference — наблюдаемый same-venue proxy, а не истинный efficient price;
  real-data RMSE и coverage являются diagnostics, не ground-truth claims.
- Fit, selection и audit представлены тремя отдельными train days; stationarity и
  time-rescaling metrics не обобщаются на market distribution.
- Active estimator использует spot только для train-only parameter identification и audit
  reference, но не как measurement update. Добавление causal spot observation model является
  новой моделью и требует отдельного preregistered experiment.
- Option margin оптимистичен: fees, latency и execution costs уменьшили бы его, поэтому
  empirical failure не может быть исправлен реалистичными costs.

## Decision and next work

P7/P8 остаются заблокированы для текущего estimator. Следующий допустимый шаг — отдельный
P6 extension, preregistered до outputs: causal fusion same-venue OKX spot observation with
book-event likelihood и explicit train-only measurement-noise calibration. Он должен
сравниваться с current-mid и causal-spot controls по likelihood/calibration, а не по P&L.
Расширять six-event support post hoc или ослаблять uncertainty gate нельзя.
