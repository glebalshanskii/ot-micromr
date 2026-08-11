# Optimal Trading of Microstructure Mean Reversion

Независимое воспроизведение результатов Amaral (2026), а затем причинная проверка
торговой стратегии на event-level рыночных данных. Реализованы analytical baseline
для Dawson optimum/Figure 3 и controlled synthetic jump simulator; empirical
backtest относится к следующим этапам. Historical P3 gate не пройден, но
его historical confirmatory gate не пройден. Последующий statistical audit показал,
что rare-open flow/drift и почти весь refinement имеют status `inconclusive`, а старые
point-estimate rules нельзя использовать для следующих experiments. P3V добавляет
integrated hazards/compensators, sensitivity-informed margins и powered configs.

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

## Controlled jump simulation

Следующие runs воспроизводят сохранённый P3 negative result и занимают несколько минут
каждый, создавая примерно 56 MiB локального event log:

```bash
uv run ot-micromr run cfg/experiments/sim_moments_001.toml
uv run ot-micromr run cfg/experiments/sim_unbalanced_001.toml
```

Оба используют adaptive frozen-intensity approximation, а не exact sampler. Текущие
run IDs и failed gates приведены в
[`docs/reports/p3-controlled-simulation.md`](docs/reports/p3-controlled-simulation.md).
До успешного powered validation переход к Figure 4 запрещён.

Новые stochastic runs следуют
[`statistical-gates-v1`](docs/protocols/common/statistical-gates.md): equality claims
требуют equivalence test и power, directional claims — superiority over a justified
minimum effect, а недостаточная precision даёт `inconclusive`. Downstream sensitivity
и power/compute design описаны в
[`p3v-sensitivity-and-power.md`](docs/reports/p3v-sensitivity-and-power.md). Новые
claim-eligible runs используют 10 CPU workers, не пишут тяжёлые event logs и требуют
чистого worktree:

```bash
uv run ot-micromr validate-config cfg/experiments/sim_moments_002.toml
uv run ot-micromr validate-config cfg/experiments/sim_unbalanced_002.toml
uv run ot-micromr run cfg/experiments/sim_moments_002.toml
uv run ot-micromr run cfg/experiments/sim_unbalanced_002.toml
```

Это длинные runs: `SIM-MOMENTS-002` имеет measured horizon `40000`, control —
`20000`. После обоих запусков joint Holm decision рассчитывается
`scripts/evaluate_p3v_family.py`; дополнительные seeds после просмотра результата не
добавляются.

Канонический статус и порядок работ: [`docs/plan.md`](docs/plan.md). Scientific
protocol: [`docs/protocols/synthetic/paper-reproduction.md`](docs/protocols/synthetic/paper-reproduction.md).
