# P2 report: analytical reproduction и Figure 3

- **Stage:** P2 — Minimal analytical baseline
- **Status:** completed; gate passed
- **Scientific status:** analytical claims reproduced
- **Дата:** 2026-08-11
- **Ветка:** `feat/p2-analytical-baseline`
- **Target implementation commit:** `710efa9bcba43ba74dfe9cfa152a6682abd4a9c0`
- **Protocol:** [`paper-reproduction.md`, v1.1](../protocols/synthetic/paper-reproduction.md)
- **Dataset/checkpoint:** not applicable
- **RNG:** не использовался; contract seed `20260811` сохранён как `consumed=false`

## Outcome

`ANA-SMOKE-001` прошёл первым, поэтому protocol разрешил запуск `ANA-FIG3-001`.
Все preregistered acceptance gates обоих runs прошли. На уровне closed-form
OU-surrogate claims `PAPER-B.1-DAWSON-OPTIMUM`, `PAPER-3.7-KRAMERS-ROOT`,
`PAPER-FIG3-THRESHOLDS` и `PAPER-FIG3-RATE-CURVES` получают status `reproduced`.

Это не reproduction exact jump-process Monte Carlo, Figure 4 или реальной торговой
доходности. P2 не использует market data и не создаёт orders/fills.

## Method и implementation

Реализованы dimensionless Dawson FOC, Kramers threshold и surrogate rate:

$$
F(u;\gamma)=u-\gamma-\sqrt 2D(u/\sqrt 2),
\qquad
u^*=\frac{\gamma+\sqrt{\gamma^2+4}}{2},
$$

$$
\frac{\widetilde R(u)}{\alpha s_G}
=\frac{2}{\pi}\frac{u-\gamma}{\operatorname{erfi}(u/\sqrt 2)}.
$$

Primary optimum найден `brentq`; bounded `minimize_scalar` является независимым
diagnostic. `erfi` отдельно проверен numerical quadrature, uniqueness — положительной
производной FOC, scaling — преобразованием `normalized_rate * alpha * s_G`.

Package использует Python 3.14.0, NumPy 2.5.2, SciPy 1.18.0 и Matplotlib 3.11.1,
зафиксированные `uv.lock` SHA-256
`e5a470b4ad10bf736118f2b8e18d49ac58381a84fcb398d981e69cc66f1ac802`.
Matplotlib формирует только PNG; scientific rows сохраняются отдельно.

## Results

### `ANA-SMOKE-001`

- Run ID: `20260811T170052058822Z-290ea5809cb6-det`
- RunSpec SHA-256: `290ea5809cb61ec8f5ec1269e55334e6465ee4bae63ba647128cf03e7a9b7c42`
- Runtime: 0.0297 s

| Metric | Result | Gate | Status |
|---|---:|---:|---|
| $u_D$ at $\gamma=0.4$ | 1.1558728538424254 | $u_D>\gamma$ | pass |
| Dawson absolute residual | $2.2204\times10^{-16}$ | $\le10^{-10}$ | pass |
| Root vs direct optimizer error | $1.7976\times10^{-8}$ | $\le10^{-7}$ | pass |
| Optimum-rate identity error | $1.6653\times10^{-16}$ | $\le10^{-10}$ | pass |
| Normalized optimum rate | 0.4090931575640008 | finite | pass |

### `ANA-FIG3-001`

- Run ID: `20260811T170104389894Z-4c1014e843c6-det`
- RunSpec SHA-256: `4c1014e843c6d40968975e2b0e1ab666e7e4c8eac13aa9fd84685de1cdda8d3b`
- Runtime: 1.1114 s
- Grid: 296 threshold rows и 1453 rate-curve rows

| $\gamma$ | $u_D$ | $u^*$ | Rate loss at $u^*$ |
|---:|---:|---:|---:|
| 0.05 | 0.5416928365107848 | 1.0253124511871279 | 0.08712403878872865 |
| 0.28 | 1.0065633047502411 | 1.1497524449091472 | 0.012915266208403997 |
| 0.40 | 1.1558728538424254 | 1.2198039027185570 | 0.003000796650572624 |
| 1.70 | 2.2649755449467333 | 2.1624404748406690 | 0.022995669417009257 |

Maximum absolute checkpoint error равен $4.98\times10^{-11}$ для thresholds и
$4.94\times10^{-11}$ для rate-loss, существенно ниже gates. Maximum Dawson residual
на полном grid равен $1.16\times10^{-13}$. Для $\gamma\in[0.4,3]$ maximum loss
`0.023035703614617375` получен at $\gamma=1.66$; ошибки относительно preregistered
audit — $1.46\times10^{-11}$ и $2.22\times10^{-16}$ соответственно. Все три myopic
rates в $u=\gamma$ точно равны нулю, все figure rows finite, а endpoint каждой
rate curve находится ниже её interior peak.

Diagnostic, не отдельный acceptance gate: maximum discrepancy Dawson root с direct
optimizer на 296 points равен $2.86\times10^{-8}$; maximum distance exact peak от
0.005-spaced plotted grid peak — 0.0012785, меньше half-grid step.

## Provenance и artifacts

Оба runs выполнены из clean tree commit `710efa9` на CPU
`12th Gen Intel(R) Core(TM) i9-12900H`, Linux 6.8 x86-64, CPython 3.14.0,
`float64`, GPU не использовался. Manifest warnings/deviations пусты.
После target runs изменены только no-argument CLI help behavior, соответствующий unit
test и stage documentation; analytical formulas, configs, tolerances и artifacts не
пересчитывались. Поэтому scientific provenance намеренно указывает target commit, а не
последующий closeout commit.

Artifacts локальны и игнорируются Git:

- smoke: `outputs/ANA-SMOKE-001/20260811T170052058822Z-290ea5809cb6-det/`;
- Figure 3: `outputs/ANA-FIG3-001/20260811T170104389894Z-4c1014e843c6-det/`;
- rendered figure: `figures/figure3.png`, SHA-256
  `92782dc48ca0e5b536c19a29437cf36b893fc7cb5110bca0e0f513c283a32916`;
- table: `tables/figure3_checkpoints.csv`;
- machine-readable source: `figures/figure3-data-thresholds.csv` и
  `figures/figure3-data-rate-curves.csv`;
- каждый run содержит `source_config.toml`, `resolved_runspec.json`, `manifest.json`,
  log и metrics summary; manifest inventory подтверждает все required classes и hashes.

## Verification

Пройдены:

```bash
uv lock --check
uv sync --locked
uv run python main.py
uv run ot-micromr
uv run python -m unittest discover -s tests -t . -v
uv run ot-micromr validate-config cfg/experiments/ana_smoke_001.toml
uv run ot-micromr validate-config cfg/experiments/ana_fig3_001.toml
git diff --check
```

Test suite содержит 25 tests: formulas/checkpoints, independent quadrature, direct
optimization, derivative/uniqueness, asymptotic scale, units/scaling, grids,
strict unknown/missing/non-finite rejection, deep immutability, artifact hashing и CLI.
Target runner дополнительно подтвердил clean-tree gate, atomic manifest и required
artifact inventory.

Setup note: первый `uv lock --check` ожидаемо сообщил stale lock после добавления build
backend; выполнены `uv lock` и `uv sync --locked`, после чего check проходит. Direct
system `python` недоступен из-за локального pyenv без 3.14; все project commands
успешно выполнены через locked `uv` environment, как требует repository workflow.

## Limitations и threats to validity

- Figure 3 воспроизведена по exact formulas статьи, но исходные author plotting code и
  raw figure data отсутствуют; это independent numerical reconstruction, не pixel match.
- Direct optimizer использует тот же rate function, поэтому он проверяет root/optimum,
  но не является независимым derivation. `erfi` integral и analytical derivative tests
  дают дополнительные orthogonal checks.
- Plotted grid заканчивается при $u=3$: он подтверждает нисходящую ветвь, не numerical
  zero. Предел rate к нулю следует из asymptotic роста `erfi` и проверен formula tests.
- P2 validator намеренно полный только для `ANA-*`; simulator-specific typed schema
  появляется в P3. `SIM-*` сейчас rejected вместо partial execution.
- Generated outputs не коммитятся и должны быть архивированы отдельно для внешней
  публикации; compact report сохраняет IDs, hashes и основные metrics.
- Synthetic surrogate result не учитывает discrete jumps, spread states, fees,
  latency, estimation error или market impact и не поддерживает profitability claim.

## Следующий gate

P3 должен реализовать controlled jump simulator из ADR-0002, сначала пройти pathwise
invariants и deterministic replay, затем выполнить `SIM-MOMENTS-001` с полным
resolution ladder. До этого Figure 4 и empirical trading conclusions остаются pending.
