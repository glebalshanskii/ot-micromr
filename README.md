# Optimal Trading of Microstructure Mean Reversion

Независимое воспроизведение Amaral (2026) и подготовка к причинной проверке торговой
стратегии на event-level рыночных данных. Реализованы analytical baseline для Dawson
optimum/Figure 3, controlled synthetic jump simulator и гибридный CPU/CUDA эксперимент
Figure 4. P3V поддержан глобальной Holm family. P4 operational validity пройдена, но
scientific family осталась `inconclusive`; подробности — в
[`paper-reproduction.md`](docs/reports/paper-reproduction.md).

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

Current controlled simulations:

```bash
uv run ot-micromr validate-config cfg/experiments/sim_moments_002.toml
uv run ot-micromr validate-config cfg/experiments/sim_unbalanced_002.toml
uv run ot-micromr run cfg/experiments/sim_moments_002.toml
uv run ot-micromr run cfg/experiments/sim_unbalanced_002.toml
```

После обоих P3V runs joint Holm decision вычисляется `scripts/evaluate_p3v_family.py`.
Оба simulations используют 10 CPU workers, не пишут тяжёлый event log и требуют clean
worktree. Полные horizons — `40000` и `20000` reversion times.

Current Figure 4 reproduction:

```bash
uv run ot-micromr validate-config cfg/experiments/sim_fig4_002.toml
uv run ot-micromr run cfg/experiments/sim_fig4_002.toml
```

P4 генерирует adaptive market endpoints на 10 CPU processes в `float64`, а vectorised
crossing/policy evaluation выполняет на CUDA в `float32` через `torch.compile`. Operational
acceptance является частью runner-а; отдельного post-hoc review command нет. В config
заморожен wall-clock budget 150 секунд для проверенного RTX 3080 Ti Laptop GPU.

Каждый run создаёт неперезаписываемую директорию `outputs/<experiment_id>/<run_id>/` с
source config, resolved `RunSpec`, manifest, raw metrics, tables и figures. `outputs/`
намеренно не коммитится.

## Checks

```bash
uv run python main.py --help
uv run ot-micromr --help
uv run python -m unittest discover -s tests -t . -v
```

Канонический статус и следующий этап: [`docs/plan.md`](docs/plan.md). Statistical policy:
[`statistical-gates-v1`](docs/protocols/common/statistical-gates.md).
