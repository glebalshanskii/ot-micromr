# Optimal Trading of Microstructure Mean Reversion

Независимое воспроизведение результатов Amaral (2026), а затем причинная проверка
торговой стратегии на event-level рыночных данных. Сейчас реализован analytical
baseline для Dawson optimum и Figure 3; synthetic jump simulator и empirical
backtest относятся к следующим этапам.

## Environment

Требуются Python 3.14 и [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --locked
```

## Analytical reproduction

Сначала проверьте preregistered config и выполните smoke gate:

```bash
uv run ot-micromr validate-config cfg/experiments/ana_smoke_001.toml
uv run ot-micromr run cfg/experiments/ana_smoke_001.toml
```

Только после успешного smoke gate запускайте Figure 3:

```bash
uv run ot-micromr validate-config cfg/experiments/ana_fig3_001.toml
uv run ot-micromr run cfg/experiments/ana_fig3_001.toml
```

Каждый run создаёт неперезаписываемую директорию в `outputs/<experiment_id>/` с
source config, canonical `RunSpec`, manifest, raw metrics, table и, где применимо,
figure data. `outputs/` намеренно не коммитится.

## Checks

```bash
uv run python main.py --help
uv run ot-micromr --help
uv run python -m unittest discover -s tests -t . -v
```

Канонический статус и порядок работ: [`docs/plan.md`](docs/plan.md). Scientific
protocol: [`docs/protocols/synthetic/paper-reproduction.md`](docs/protocols/synthetic/paper-reproduction.md).
