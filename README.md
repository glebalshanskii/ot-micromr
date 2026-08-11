# Optimal Trading of Microstructure Mean Reversion

Независимое воспроизведение результатов Amaral (2026), а затем причинная проверка
торговой стратегии на event-level рыночных данных. Реализованы analytical baseline
для Dawson optimum/Figure 3 и controlled synthetic jump simulator; empirical
backtest относится к следующим этапам. Historical P3 gate не прошёл, после чего P3V
заменил эвристические пороги на integrated estimators, sensitivity-informed margins и
powered equivalence/superiority tests. Оба P3V run и глобальная Holm family поддержаны;
независимый P4 target выполнен. P4 частично воспроизвёл Figure 4, но formal operational
acceptance failed и общая scientific family осталась inconclusive; подробности в
[`paper-reproduction.md`](docs/reports/paper-reproduction.md).

## Environment

Требуются Python 3.14 и [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --locked
```

Для optional CUDA backend:

```bash
uv sync --locked --extra gpu
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
Historical results сохранены как negative evidence и не определяют текущий P3V status.

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
добавляются. Канонические runs и adjusted p-values приведены в linked report; global
status — `supported`.

Для P4 выбран гибридный execution path: генерация adaptive event paths на 10 CPU
processes, vectorised threshold/policy evaluation на CUDA через `torch.compile`. Решение
основано на локальном transfer-inclusive benchmark; CPU reference остаётся обязательным
regression oracle.

Claim-eligible P4 config требует CUDA и использует 30 frozen seeds. Канонический target уже
выполнен и не расширяется; команды ниже предназначены только для exact rerun:

```bash
uv run ot-micromr validate-config cfg/experiments/sim_fig4_002.toml
uv run ot-micromr run cfg/experiments/sim_fig4_002.toml
```

В config заморожен wall-clock budget 150 секунд на проверенном RTX 3080 Ti Laptop GPU.

Канонический статус и порядок работ: [`docs/plan.md`](docs/plan.md). Scientific
protocol: [`docs/protocols/synthetic/paper-reproduction.md`](docs/protocols/synthetic/paper-reproduction.md).
