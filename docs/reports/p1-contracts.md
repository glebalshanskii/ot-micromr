# P1 report: research contracts и synthetic preregistration

- **Stage:** P1 — Mathematical, simulation and experiment contracts
- **Status:** completed; gate passed
- **Дата:** 2026-08-11
- **Ветка:** `docs/p1-research-contracts`
- **Contract commit:** `36d606e`
- **Target experiment runs:** не выполнялись
- **Generated output artifacts:** отсутствуют

## Scope

P1 фиксирует постановку будущих analytical и simulation experiments до реализации
target algorithms и просмотра outputs. Это отчёт о завершении contract stage, а не
paper-reproduction result: численные claims статьи ещё не получили project status
`reproduced`/`not-reproduced`.

## Принятые решения

- [`ADR-0001`](../adr/0001-research-modes-package-and-run-contract.md) определил
  research tracks/modes, import package `ot_micromr`, strict TOML `RunSpec v1`, units,
  runtime manifest и immutable artifact layout.
- [`ADR-0002`](../adr/0002-controlled-jump-simulation-semantics.md) отделил exact
  continuous-time target от controlled numerical simulation. Зафиксированы algorithm
  ID, event/observation ordering, adaptive hazard cap, Brownian first-hit semantics,
  RNG mapping, refinement и `jump_simulator_invariants_v1`.
- [`paper-reproduction.md`](../protocols/synthetic/paper-reproduction.md) заранее
  задаёт analytical formulas/checkpoints, project-chosen simulator primitives, 20 seeds,
  horizons, estimators, uncertainty, acceptance и stop rules.
- [`cfg/experiments/`](../../cfg/experiments/) содержит пять parseable contracts:
  `ANA-SMOKE-001`, `ANA-FIG3-001`, `SIM-MOMENTS-001`, negative control
  `SIM-UNBALANCED-001` и `SIM-FIG4-001`.
- Figure 4 остаётся `independent_partial_reproduction`: author primitives, simulator
  code, seeds и raw outputs неизвестны, а match с paper ranges не является acceptance
  gate.
- `.gitignore` расширен до первого artifact/data run; `AGENTS.md` использует валидный
  import path `src/ot_micromr/`.

## Validation evidence

Выполнены только проверки, соответствующие docs/config stage:

1. Standard-library `tomllib` parsed все пять TOML без errors; проверены unique IDs,
   common required fields/tables, modes, units, artifact root и non-empty seed lists.
2. Cross-config assertions подтвердили balanced response, one-factor unbalanced change,
   20 unique simulation seeds, epsilon ladder, Figure 4 calibration/burn-in identities,
   realized-spread accounting, solver tolerances и explicit refinement normalization.
3. SHA-256 локального paper PDF повторно совпал с
   `fd1a0dfc0d8fc8d7feb26ee23231232ac4263e95a5bb0ef41d18e4c0a8c611ba`.
4. Local Markdown links разрешаются, display-math delimiters сбалансированы,
   `git diff --check` проходит, а ignore checks подтверждают exclusions для outputs,
   datasets, logs, checkpoints, secrets и tool caches.
5. Два независимых adversarial review проверили scientific protocol и configuration
   contract. После исправлений final gates не содержат High/Medium blockers.

P1 не добавлял dependencies. На дату проверки official PyPI показывал CPython 3.14
wheels для [NumPy 2.5.2](https://pypi.org/project/numpy/) и
[SciPy 1.18.0](https://pypi.org/project/scipy/); это compatibility evidence, не pin.
Package/CLI и tests ещё не созданы, поэтому `uv run ot-micromr` и unit-test discovery к
этому docs-only stage неприменимы. Existing scaffold прошёл `.venv/bin/python main.py`
(`Hello from ot-micromr!`). Target `ANA-SMOKE-001` намеренно не запускался.

## Results и artifacts

Единственный результат P1 — versioned research contract в Git. В `outputs/` ничего не
создавалось; runtime, hardware, Monte Carlo metrics, plots и scientific acceptance
statuses отсутствуют. Исполняемые inputs находятся в `cfg/experiments/`, decisions — в
`docs/adr/`, preregistration — в `docs/protocols/synthetic/`.

## Limitations и threats to validity

- `RunSpec v1` пока проверен structurally, но strict typed validator, canonical
  serialization и atomic manifest writer появятся только в P2.
- `adaptive_left_hazard_single_jump_v1` не реализован и не имеет заявленного formal
  convergence order. Frozen intensity, right-end book events и Brownian localization
  обязаны пройти preregistered epsilon refinement в P3.
- Figure 4 author settings остаются underdetermined; project parameters не являются их
  реконструкцией.
- Compatibility проверена только для ожидаемого analytical NumPy/SciPy stack, не для
  будущих empirical libraries.
- Synthetic contracts не дают evidence о market profitability и не используют real
  exchange data.

## Следующий gate

P2 должен через `uv add` зафиксировать NumPy/SciPy в `uv.lock`, создать
`src/ot_micromr/`, реализовать strict config validation, Dawson/surrogate functions и
tests, затем выполнить только `ANA-SMOKE-001`. Figure 3 не запускается до прохождения
этого smoke gate.
