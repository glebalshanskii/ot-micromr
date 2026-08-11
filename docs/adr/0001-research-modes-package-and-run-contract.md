# ADR-0001: Research modes, Python package и RunSpec v1

- **Статус:** Accepted
- **Дата:** 2026-08-11
- **Связанный этап:** P1
- **Связанный план:** [`docs/plan.md`](../plan.md)

## Контекст

Проект должен независимо воспроизвести теоретические результаты статьи, а затем
проверить causal-стратегию на рыночных данных. Эти задачи используют разные
information sets, execution assumptions и критерии допустимых утверждений. Если
смешать paper assumptions, latent-state oracle и практически доступный signal, можно
получить формально воспроизводимый, но непригодный empirical backtest.

До первого executable run также нужно устранить несколько неоднозначностей:

- distribution и CLI могут называться `ot-micromr`, но hyphen недопустим в Python
  import namespace;
- planned YAML filenames не задают строгий schema contract и требуют отдельной
  зависимости;
- часть полей задаётся исследователем до запуска, а commit, hardware и runtime можно
  узнать только во время запуска;
- paper formulas используют одновременно dimensionless ratios, price, tick, lot и
  inverse-reversion time, поэтому неявные units создают риск тихих scaling errors;
- artifacts разных режимов нельзя складывать в общий неаудируемый набор файлов.

Этот ADR фиксирует границы режимов, package naming, immutable `RunSpec v1`, runtime
manifest и artifact layout. Simulator time/event semantics фиксируются отдельным ADR.

## Решение

### 1. Research tracks и modes

`track` описывает источник evidence и имеет два значения: `synthetic` и `empirical`.
`mode` описывает допустимый information set и имеет три значения:
`paper-faithful`, `practical-local` и `oracle-diagnostic`.

| Mode | Допустимые данные и assumptions | Допустимые результаты | Запрещено |
|---|---|---|---|
| `paper-faithful` | Формулы и synthetic jump model статьи; истинный $X_t$ доступен там, где он является state paper model; paper execution/cost conventions; явно названные numerical approximations | Reproduction status аналитических claims, theorem checks и independent reconstruction figures | Переносить synthetic P&L в claim о реальной доходности; скрывать approximation или подменять неизвестные author settings подобранными значениями |
| `practical-local` | Только observations, доступные к `decision_time`; causal estimate $\widehat X_t$; immutable market data version/splits; реализуемые fills, costs и latency | Out-of-sample empirical claims в границах указанного venue, instrument, period, size и stress scenario | Использовать истинный/будущий $X_t$, retrospective smoother, future quotes/trades либо test-period tuning для orders или parameter selection |
| `oracle-diagnostic` | Истинный latent state на synthetic data и/или явно retrospective estimator; тот же base scenario, что у сравниваемого causal run | Upper bound, recovery error, attribution и leakage diagnostic | Генерировать executable backtest orders, выбирать primary strategy/parameters, проходить profitability gate или поддерживать empirical trading claim |

`oracle-diagnostic` не является третьим scientific track. Он должен указывать base
`track`, но каждый такой `RunSpec` обязан иметь `orders_enabled = false` и
`claim_eligible = false`. Его outputs хранятся в отдельном run directory; их нельзя
агрегировать с feasible results без явной колонки `mode`. Если один анализ сравнивает
oracle и causal estimator, это две независимые спецификации либо две явно разделённые
legs с разными manifests, а не runtime switch внутри одного run.

Любое отклонение от paper assumptions в synthetic исследовании получает новый
`RunSpec` и label `extension`; оно не заменяет paper-faithful baseline. Любая логика с
future access автоматически относится к `oracle-diagnostic`, даже если она запущена
на real data.

### 2. Distribution, import package и CLI

Принимается `src` layout:

```text
src/
└── ot_micromr/
    ├── __init__.py
    ├── __main__.py
    └── cli.py
```

- Python import namespace: `ot_micromr`.
- Distribution name в `pyproject.toml`: `ot-micromr`.
- Console script: `ot-micromr = "ot_micromr.cli:main"`.
- `python -m ot_micromr` должен делегировать тому же `main`.

Директория `src/ot-micromr/` из общего structural description не создаётся: она не
является валидным import package. Domain state, config DTOs, simulator runtime,
analytics, estimation, execution, metrics и artifact storage далее разделяются
модулями внутри единственного namespace; конкретные модули добавляются только при
появлении соответствующего use case.

CLI не делает implicit downloads и import-time work. Минимальный future interface —
`ot-micromr validate-config <path>` и `ot-micromr run <path>`. Точные subcommands могут
расширяться обратно совместимо, но один config всегда соответствует одному immutable
`RunSpec`.

### 3. TOML как executable configuration format

Executable configs хранятся только как `cfg/experiments/*.toml` и читаются standard
library `tomllib`. Config является полным input contract, а не набором overrides:

- отсутствующий required field и любой unknown field являются validation error;
- parser не подставляет научно значимые defaults из кода, environment, current
  directory, wall clock или hardware;
- seed никогда не генерируется неявно;
- arbitrary CLI overrides в `RunSpec v1` запрещены; для другого значения создаётся
  новый TOML и новый hash;
- non-finite floats, неоднозначные local timestamps и несовместимые units отклоняются;
- secrets/credentials не попадают в config или manifest; внешний credential может
  дать доступ к данным, но immutable dataset identity/version/hash остаются в config;
- пути внутри config разрешаются относительно repository root, который manifest
  фиксирует явно, а не относительно случайного process working directory.

Комментарии могут объяснять provenance, но не заменяют typed fields. Исполнитель
сохраняет source TOML побайтно, его SHA-256 и канонически сериализованный `RunSpec`.

### 4. Immutable `RunSpec v1`

После parse/validation config превращается в immutable value object. Он не содержит
mutable counters, open file handles, RNG objects, measured runtime или обнаруженный
hardware. Эти значения принадлежат отдельному `RunState`; итоговая фактическая запись
принадлежит `RunManifest`.

Все `RunSpec v1` обязаны содержать следующие top-level fields/tables:

| Field/table | Contract |
|---|---|
| `schema_version` | Строка `runspec-v1`; несовместимое изменение требует нового major schema и ADR |
| `experiment_id` | Стабильный ID из experiment matrix, например `ANA-SMOKE-001` |
| `track` | `synthetic` или `empirical` |
| `mode` | Один из трёх modes этого ADR |
| `objective` | Краткое заранее заданное назначение run |
| `claim_ids` | Явный список проверяемых hypotheses/claims; для чистого diagnostic допускается пустой список |
| `orders_enabled` | Явный boolean; обязательно `false` для `oracle-diagnostic` |
| `claim_eligible` | Явный boolean; обязательно `false` для smoke/exploratory/oracle runs |
| `seed_policy` | Непустой ordered list `seeds`, RNG algorithm и mapping seed-to-replication |
| `units` | Base units и normalization contract из следующего раздела |
| `numerics` | `float_dtype`, algorithm/solver, tolerances и все result-affecting numerical controls |
| `inputs` | Paper/version либо dataset/model/checkpoint identities и hashes; значение `not_applicable` задаётся явно с причиной |
| `model` | Все используемые analytical/simulator/filter parameters, включая decomposition и constraints; неиспользуемый section явно отключён |
| `simulation` | Horizon, burn-in, discretization/refinement и sampling policy либо явное `enabled = false` |
| `strategy` | Policy, threshold/cost convention и position semantics либо явное `enabled = false` |
| `execution` | Fill, spread, fee, slippage, latency, quantity и session policy либо явное `enabled = false` |
| `evaluation` | Metrics, estimators, confidence/bootstrap/multiplicity procedure и aggregation level |
| `acceptance` | Численные thresholds, comparison direction и stop criteria, зафиксированные до target output |
| `artifacts` | Явный `output_root = "outputs"` и перечень ожидаемых artifact classes |

`seed_policy.seeds` содержит один seed для deterministic smoke и preregistered ordered
list для multi-seed run. Повторения не используют `seed + worker_id` без записи этого
mapping. Parallel scheduling не должен менять random streams.

Mode-specific обязательные данные:

- `paper-faithful`: citation/version, paper equation/claim IDs, все primitive или
  dimensionless model parameters и точная маркировка неизвестных author settings;
- `practical-local`: dataset ID/version/SHA-256, venue, symbol, tick and contract
  specification, timezone/session policy, chronological split boundaries, causal
  timestamp policy, filter version, cost and latency scenarios;
- `oracle-diagnostic`: `oracle_kind`, `base_experiment_id`, `future_access` и
  diagnostic metrics; flags `orders_enabled` и `claim_eligible` принудительно false.

`not_applicable` не является пустым placeholder: рядом требуется `reason`. Значение
`unknown` допустимо только для source facts, отсутствующих в статье; оно не допускается
для исполняемого numerical parameter. Несовместимые mode/track/field combinations
отклоняются до создания run directory.

### 5. Units и normalization

Canonical internal units:

- time — `second`;
- price и gap — instrument `price_unit`;
- intensity и mean-reversion rate — `1/second`;
- diffusion volatility — `price_unit/sqrt(second)`;
- quantity — integer `lot`;
- cash/P&L — `quote_currency`, после применения explicit `contract_multiplier`;
- dimensionless значения — `ratio`, включая $\gamma$, probabilities и normalized
  thresholds.

Каждый config явно задаёт `units.time`, `units.price`, `units.quantity`,
`units.cash` и `units.timezone`. Dimensional field names кодируют размерность:
`alpha_per_second`, `sigma_x_price_per_sqrt_second`, `delta_price`,
`latency_seconds`, `fee_quote_currency_per_lot`. Нельзя оставлять голые `alpha`,
`sigma`, `dt` или `fee`, если их units нельзя вывести из schema однозначно.
Persisted timestamps всегда переводятся в UTC; timezone торговой сессии хранится
отдельно как IANA identifier и не меняет ordering событий.

Synthetic horizon и burn-in задаются dimensionless полями
`horizon_reversion_times` и `burn_in_reversion_times`, то есть в units $1/\alpha$;
runtime записывает также полученные seconds. Analytical normalized runs явно ставят
`normalization = "ou_dimensionless"`; преобразование обратно в canonical units
сохраняется в manifest. Tick-normalized reporting разрешён только как derived metric:
исходные `delta_price`, $s_G$ и half-spread остаются в price units.

Для empirical config `contract_multiplier`, tick history и currency обязательны.
Execution at bid/ask уже включает spread; отдельные fees/slippage/funding имеют каждый
свою basis и conversion rule, чтобы spread не учитывался дважды.

### 6. Runtime manifest и provenance

Config остаётся неизменным после запуска. Runtime-derived facts сохраняются в
`manifest.json` со schema `run-manifest-v1`; manifest не становится источником новых
параметров. Он обязан содержать:

- `run_id`, `experiment_id`, `status`, UTC start/end timestamps и elapsed runtime;
- launch command, repository root, source config relative path, source TOML SHA-256,
  canonical `RunSpec` и его SHA-256;
- Git remote, full commit SHA, branch, `dirty` flag; для dirty smoke — hash и artifact
  path sanitized patch, для confirmatory/full run требуется clean tree;
- Python version/implementation, OS/platform, `uv.lock` SHA-256, installed distribution
  versions and hashes sufficient for replay;
- CPU/GPU model, accelerator/driver, available memory, thread/process counts,
  precision and known nondeterministic kernels;
- actual seed-to-replication mapping и RNG implementations;
- dataset/model/checkpoint version and hash, если применимо;
- фактически полученные derived parameters/conversions, warnings, deviations и
  termination/failure reason;
- primary metrics summary и inventory всех artifacts: relative path, media type, size
  и SHA-256.

Manifest сначала пишется во временный файл в том же filesystem, `fsync`-ится и
атомарно заменяет `manifest.json`. При failure сохраняется manifest со статусом
`failed`, а не теряется отрицательный run. Final manifest записывается последним;
сам manifest не включает собственный hash в inventory.

Actual commit, environment, hardware, time и derived values намеренно не записываются
заранее в source TOML: это не hidden defaults, а observed provenance. В config
фиксируется требование к ним (`claim_eligible`, precision, clean-run policy), manifest
фиксирует факт.

### 7. Artifact layout

Каждый run получает неперезаписываемую директорию
`outputs/<experiment_id>/<run_id>/`. `run_id` строится из UTC start timestamp,
короткого `RunSpec` hash и seed/group marker; timestamp служит только уникальности и не
влияет на расчёт. Существующую директорию runner не перезаписывает.

```text
outputs/<experiment_id>/<run_id>/
├── source_config.toml
├── resolved_runspec.json
├── manifest.json
├── logs/
│   └── run.log
├── metrics/
│   ├── summary.json
│   └── raw.*
├── records/               # when required by RunSpec
│   ├── events.*
│   └── fills.*
├── tables/
├── figures/
│   ├── figure-data.*
│   └── figure.*
└── state/                  # optional checkpoints/filter state
```

Пустые optional directories не создаются. Plot data хранится отдельно от rendered
image. Raw outputs, logs, datasets, checkpoints и generated artifacts остаются
untracked; configs, ADRs, protocols и compact reports коммитятся. Published report
ссылается на `experiment_id`, `run_id`, manifest и конкретные artifact paths.

### 8. Dependencies и Python version

P1 не добавляет runtime или development dependencies: TOML parsing обеспечивает
Python standard library `tomllib`. Constraint `requires-python = ">=3.14"` пока
сохраняется. На дату решения official PyPI публикует CPython 3.14 wheels для
[NumPy 2.5.2](https://pypi.org/project/numpy/) и
[SciPy 1.18.0](https://pypi.org/project/scipy/), поэтому ожидаемый P2 analytical stack
не требует понижения Python только из-за этих двух библиотек.

Это compatibility evidence, а не dependency pin. В P2 зависимости добавляются только
через `uv add`; реально разрешённые versions и hashes фиксирует `uv.lock`, после чего
выполняются install и analytical smoke checks. Если другая необходимая библиотека не
поддерживает Python 3.14, изменение Python constraint требует отдельного решения, а не
implicit downgrade.

## Обоснование

- Явные modes не позволяют synthetic oracle или retrospective smoother незаметно
  превратиться в feasible trading signal.
- Valid underscore namespace совместим с Python tooling, сохраняя узнаваемое
  hyphenated project/CLI name.
- TOML доступен в standard library, хорошо подходит для review и исключает executable
  behavior Python configs; strict schema делает каждый result-affecting choice видимым.
- Immutable `RunSpec` отделяет preregistration от измеренных runtime facts, а manifest
  связывает результат с кодом, environment, data и artifacts.
- Canonical units и размерные имена полей делают scaling testable и предотвращают
  смешение seconds, $1/\alpha$, ticks и quote currency.
- Run-scoped immutable artifact layout сохраняет отрицательные результаты и позволяет
  воспроизвести figure/table независимо от PNG или prose report.

## Отклонённые альтернативы

1. **`src/ot-micromr/` как import package.** Отклонено: hyphen не является валидной
   частью Python identifier.
2. **YAML configs.** Отклонено для v1: нужен parser dependency, а implicit scalar
   coercion и широкая dialect surface усложняют строгий audit. Planned `.yaml` names
   заменяются на `.toml` до запуска.
3. **Executable Python configs или notebook state.** Отклонено: они допускают imports,
   environment branching и hidden global state.
4. **Library defaults плюс короткие override files.** Отклонено: изменение версии кода
   может молча изменить старый experiment. В v1 все result-affecting values явны.
5. **Один mode для synthetic, oracle и empirical runs.** Отклонено: невозможно
   автоматически обнаружить information leakage и ограничить claims.
6. **Runtime fields внутри preregistered config.** Отклонено: hardware, commit и
   timestamps неизвестны до запуска и вынуждали бы мутировать config.
7. **Только manifest без source config.** Отклонено: preregistration и review должны
   существовать до run; manifest является фактической записью, а не protocol.
8. **Неявные units из symbol или paper notation.** Отклонено: одинаковое имя может
   означать seconds, normalized time, ticks или price units.
9. **Одна flat output directory либо перезапись последнего run.** Отклонено: теряются
   provenance, failed runs и соответствие plots исходным metrics.

## Последствия

Положительные:

- каждый executable result можно однозначно классифицировать по допустимому
  information set и claim scope;
- smoke/full runs имеют один serializable contract и единый audit trail;
- future simulator, estimator и backtester разделяют provenance/artifact machinery;
- config diff соответствует реальному изменению experiment, а не смене hidden default.

Издержки и ограничения:

- configs более многословны; даже неприменимые sections требуют явного disable/reason;
- до P2 нужно реализовать strict validator, canonical serialization, atomic manifest и
  tests на invalid mode/unit combinations;
- изменение schema требует мигратора/reader для старой версии либо нового major
  schema; старые manifests не переписываются;
- `oracle-diagnostic` требует отдельных runs и storage, даже если вычислительно его
  удобно было бы смешать с causal evaluation;
- наличие CPython 3.14 wheels проверено только для NumPy/SciPy и не гарантирует
  совместимость всего будущего empirical stack.

Этот ADR не утверждает, что какой-либо experiment уже выполнен, и не задаёт simulator
discretization, market-event ordering или fill semantics. Они фиксируются отдельными
ADRs/protocols до соответствующих runs.
