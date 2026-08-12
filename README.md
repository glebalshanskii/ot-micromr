# Optimal Trading of Microstructure Mean Reversion

Завершённое независимое исследование Amaral (2026). Closed-form Dawson optimum и Figure 3
воспроизведены; controlled jump simulator поддержал balanced-flow identities. Figures 2 и
5 воспроизведены структурно, а Figure 4 — частично: две из трёх строк показали ожидаемый
inward shift, но полная statistical family осталась `inconclusive`.

Empirical extension на frozen OKX crypto sample завершён с результатом `negative` для
проверенного book-only causal state estimator. Финальная factorized модель исправила
event-clock moments, но conditional gap signal не улучшил held-out mark score, а
posterior uncertainty составила `8.426` optimistic option margins. Поэтому backtest и P&L
search не запускались. Доходность имеет статус `not-confirmed / not tested`, а не
«стратегия доказанно убыточна».

Итоговая claim-to-evidence матрица, canonical runs, deviations и limitations находятся в
[`final-synthesis.md`](docs/reports/final-synthesis.md). Подробная paper reproduction — в
[`paper-reproduction.md`](docs/reports/paper-reproduction.md), empirical endpoint — в
[`p6d-factorized-clock-mark.md`](docs/reports/p6d-factorized-clock-mark.md).

## Environment

Требуются Python 3.14 и [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --locked
```

PyTorch является обязательной вычислительной зависимостью для CPU и CUDA paths;
отдельного `gpu` extra нет. Figure 4 использует CUDA-capable installation из lockfile.

## Experiments

Analytical reproduction:

```bash
uv run ot-micromr validate-config cfg/experiments/ana_smoke_001.toml
uv run ot-micromr run cfg/experiments/ana_smoke_001.toml
uv run ot-micromr validate-config cfg/experiments/ana_fig3_001.toml
uv run ot-micromr run cfg/experiments/ana_fig3_001.toml
```

Controlled simulations:

```bash
uv run ot-micromr validate-config cfg/experiments/sim_moments_002.toml
uv run ot-micromr validate-config cfg/experiments/sim_unbalanced_002.toml
uv run ot-micromr run cfg/experiments/sim_moments_002.toml
uv run ot-micromr run cfg/experiments/sim_unbalanced_002.toml
```

После обоих P3V runs joint Holm decision вычисляется `scripts/evaluate_p3v_family.py`.
Оба simulations используют 10 CPU workers, не пишут тяжёлый event log и требуют clean
worktree. Полные horizons — `40000` и `20000` reversion times.

Figure 4 reproduction:

```bash
uv run ot-micromr validate-config cfg/experiments/sim_fig4_002.toml
uv run ot-micromr run cfg/experiments/sim_fig4_002.toml
```

P4 генерирует adaptive market endpoints на 10 CPU processes в `float64`, а vectorised
crossing/policy evaluation выполняет на CUDA в `float32` через `torch.compile`. Operational
acceptance является частью runner-а; отдельного post-hoc review command нет. В config
заморожен wall-clock budget 150 секунд для проверенного RTX 3080 Ti Laptop GPU.

Empirical data audit:

```bash
uv run ot-micromr fetch-data cfg/experiments/emp_data_001_sources.toml
uv run ot-micromr validate-config cfg/experiments/emp_data_001.toml
uv run ot-micromr run cfg/experiments/emp_data_001.toml
```

Download выполняется только первой явной командой. P5 использует OKX train sample,
нормализует UTC+8 trade archive cuts, применяет full-snapshot health quarantine и не
рассчитывает strategy/P&L. Raw files и outputs остаются локальными и не коммитятся.
Полный обзор альтернативных sources и точный local inventory:
[`docs/reports/market-data-sources.md`](docs/reports/market-data-sources.md).

Causal-filter experiments:

```bash
uv run ot-micromr validate-config cfg/experiments/filter_syn_001.toml
uv run ot-micromr run cfg/experiments/filter_syn_001.toml
uv run ot-micromr validate-config cfg/experiments/emp_filter_001.toml
uv run ot-micromr run cfg/experiments/emp_filter_001.toml
uv run ot-micromr validate-config cfg/experiments/filter_mark_syn_001.toml
uv run ot-micromr run cfg/experiments/filter_mark_syn_001.toml
uv run ot-micromr validate-config cfg/experiments/emp_mark_filter_001.toml
uv run ot-micromr run cfg/experiments/emp_mark_filter_001.toml
uv run ot-micromr validate-config cfg/experiments/emp_mark_ct_001.toml
uv run ot-micromr run cfg/experiments/emp_mark_ct_001.toml
uv run ot-micromr validate-config cfg/experiments/emp_mark_fact_001.toml
uv run ot-micromr run cfg/experiments/emp_mark_fact_001.toml
```

Empirical marked run требует прошедший synthetic dependency и verified P6 processed
tensors. Paths используют vectorized PyTorch CUDA `float32`, `torch.compile` и
PyTorch `float64` для final statistics. P6D дополнительно сохраняет causal
time-series и actual-versus-predictive distribution figures. Ни один empirical filter
не выполняет orders, fills или P&L. Эти команды воспроизводят завершённое исследование;
они не являются очередью будущих этапов.

Каждый run создаёт неперезаписываемую директорию `outputs/<experiment_id>/<run_id>/` с
source config, resolved `RunSpec`, manifest, raw metrics, tables и figures. `outputs/`
намеренно не коммитится.

## Checks

```bash
uv run python main.py --help
uv run ot-micromr --help
uv run python -m unittest discover -s tests -t . -v
```

Канонический финальный статус: [`docs/plan.md`](docs/plan.md). Statistical policy:
[`statistical-gates-v1`](docs/protocols/common/statistical-gates.md).
