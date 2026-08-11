# План воспроизведения Optimal Trading of Microstructure Mean Reversion

> **Обновлено:** 2026-08-11
>
> **Ветка:** `feat/p4-figure4-reproduction`
>
> **Текущий статус:** P4 completed; operational acceptance failed; scientific family inconclusive
>
> **Следующий шаг:** P5 licensed event-level data feasibility и frozen universe/split

## 1. Цель, scope и критерий научного утверждения

Проект состоит из двух явно разделённых треков:

1. **Paper-faithful reproduction** — независимо реализовать аналитические формулы,
   exact jump model и Monte Carlo проверки статьи
   [`arXiv:2608.00885v1`](https://arxiv.org/abs/2608.00885v1).
2. **Practical-local empirical extension** — построить causal-версию стратегии для
   реальных event-level quotes/trades, провести backtests без look-ahead и определить,
   при каких наблюдаемых режимах и заранее заданных параметрах стратегия имеет
   положительный net edge после всех доступных издержек.

Второй трек не является буквальным воспроизведением статьи: статья теоретическая,
не использует рыночный dataset и предполагает наблюдаемый latent efficient price.

### 1.1. Что означает «воспроизвести»

- Аналитические результаты получены независимым кодом и совпадают с формулами.
- Stochastic simulator удовлетворяет model invariants, а theoretical moments проходят
  multiplicity-aware equivalence с заранее обоснованной margin и достаточной power;
  простое попадание target в широкий confidence interval не считается reproduction.
- Каждый рисунок классифицирован как `reproduced`, `partially-reproduced`,
  `not-reproduced` или `underdetermined`; отрицательный результат не меняет protocol
  задним числом.
- Результат имеет полный provenance: commit, config, seed, environment, hardware,
  runtime и artifact paths.

### 1.2. Что означает «стратегия доходна»

Основной empirical claim разрешён только для **однократно открытого, заранее
замороженного holdout** и означает одновременно:

- средняя net P&L rate положительна после bid--ask spread, taker fees, известных
  exchange/regulatory costs, borrow/funding и реалистичного slippage/latency scenario;
- нижняя граница одностороннего 95% block-bootstrap confidence interval для средней
  daily/session net P&L выше нуля;
- результат не сводится к одному instrument, короткому regime или одной выбранной
  post hoc parameter cell и выдерживает заранее заданные cost/latency stresses;
- учтены все проверенные variants и multiple testing; secondary findings помечены
  exploratory, если для них нет multiplicity correction или независимой проверки.

Если gate не пройден, корректный итог — «доходность не подтверждена при проверенных
условиях», а не дополнительная настройка на holdout.

### 1.3. Вне текущего scope

- live trading и отправка заявок;
- обещание доходности или инвестиционная рекомендация;
- доказательство глобальной оптимальности band policy на exact jump process — в статье
  это открытая conjecture;
- large-order impact, market making, strategic response liquidity providers и portfolio
  allocation до прохождения one-lot baseline;
- использование платного API, приватных данных или данных с неясной лицензией в
  default path.

## 2. Зафиксированный разбор статьи

Полная bibliographic запись и provenance локального PDF находятся в
[`docs/papers/registry.md`](papers/registry.md). Локальный PDF побайтно совпадает с
official arXiv v1; SHA-256:
`fd1a0dfc0d8fc8d7feb26ee23231232ac4263e95a5bb0ef41d18e4c0a8c611ba`.

### 2.1. Central claim и novelty

Статья строит limit-order-book model для liquid large-tick asset, в котором transient
microstructure gap

$$
G_t = M_t - X_t
$$

между displayed mid $M_t$ и latent efficient price $X_t$ mean-reverts благодаря
state-dependent book flow. Parity mid на half-tick grid полностью определяет, равен ли
spread одному или двум ticks, поэтому непрерывное состояние сводится к $G$ плюс parity
bit. При balanced-response condition условное среднее и stationary autocovariance gap
в точности совпадают с Ornstein--Uhlenbeck process, хотя sample paths остаются jump
paths. На соответствующем Gaussian surrogate symmetric flip band имеет closed-form
оптимум.

Новизна — соединение endogenous queue-reactive jump book с optimal switching для
mean-reversion signal и вывод простой one-dimensional band formula. Это не empirical
демонстрация прибыльной стратегии.

### 2.2. Модель и проверяемые формулы

Для tick size $\delta$, bid $B_t\in\delta\mathbb Z$ и
$S_t\in\{\delta,2\delta\}$:

$$
M_t=B_t+\frac{S_t}{2}\in\frac{\delta}{2}\mathbb Z.
$$

Tight book соответствует $M_t\in\delta\mathbb Z+\delta/2$, open book —
$M_t\in\delta\mathbb Z$. Efficient price exogenous:

$$
dX_t=\sigma_X\,dZ_t.
$$

В tight state действуют slide и open events, в open state — close events. Для каждого
типа $i\in\{s,o,c\}$ directional intensities имеют one-sided linear ramps

$$
\lambda_i^\uparrow=\mu_i+\frac{2\alpha_i}{\delta}G^-,\qquad
\lambda_i^\downarrow=\mu_i+\frac{2\alpha_i}{\delta}G^+,
$$

с соответствующим state indicator. Slides меняют $M$ на $\pm\delta$, opens/closes —
на $\pm\delta/2$. Balanced response

$$
2\alpha_s+\alpha_o=\alpha_c=: \alpha
$$

даёт точный drift и conditional mean

$$
\mathbb E[dG_t\mid\mathcal F_{t-}]=-\alpha G_{t-}\,dt,
\qquad
\mathbb E[G_{t+h}\mid\mathcal F_t]=G_t e^{-\alpha h}.
$$

В stationary regime

$$
\mathbb E_\pi[G]=0,\qquad
s_G^2=\operatorname{Var}_\pi(G)=\frac{\sigma_X^2+\sigma_M^2}{2\alpha},
\qquad
\operatorname{Cov}_\pi(G_t,G_{t+h})=s_G^2e^{-\alpha h},
$$

где
$\sigma_M^2=\mathbb E_\pi\!\left[\sum_m\lambda^m\Delta_m^2\right]$.
Из этих identities также следует permanent-component forecast

$$
\lim_{h\to\infty}\mathbb E[M_{t+h}\mid\mathcal F_t]=X_t.
$$

Статья дополнительно ограничивает неявный $s_G$ через baseline jump variance

$$
\sigma_{M,0}^2=\delta^2\left[(1-p)
\left(2\mu_s+\frac{\mu_o}{2}\right)+p\frac{\mu_c}{2}\right],
\qquad
b=\delta\left(\alpha-\frac{\alpha_o}{2}\right),
$$

$$
\sqrt{\frac{\sigma_X^2+\sigma_{M,0}^2}{2\alpha}}
\le s_G\le
\frac{b+\sqrt{b^2+8\alpha(\sigma_X^2+\sigma_{M,0}^2)}}{4\alpha}.
$$

Стратегия после первого входа всегда держит $q_t\in\{-1,+1\}$ и **переворачивает**
позицию: target $+1$ при $G_t\le-\theta$, target $-1$ при
$G_t\ge\theta$, иначе сохраняет предыдущую позицию. Первый entry равен одному lot,
каждый последующий flip — двум lots. Это не обычная flat-between-trades pairs strategy.

Exact jump-model band rate:

$$
R(\theta)=
\frac{2\left(\mathbb E_{\pi_F}|G_F|-\mathbb E_{\pi_F}\phi_F\right)}{m(\theta)},
\qquad
\theta\le\mathbb E_{\pi_F}|G_F|<\theta+\delta.
$$

Gaussian moment-matched surrogate задаётся как

$$
d\widetilde G_t=-\alpha\widetilde G_t\,dt
+\sqrt{2\alpha}\,s_G\,d\widetilde Z_t.
$$

При frozen tight-book half-spread $\phi=\delta/2$:

$$
\widetilde m(\theta)
=\frac{\pi}{\alpha}\operatorname{erfi}
\left(\frac{\theta}{\sqrt 2s_G}\right),
\qquad
\widetilde R(\theta)=\frac{2(\theta-\phi)}{\widetilde m(\theta)}.
$$

Для $u=\theta/s_G$ и $\gamma=\phi/s_G$ exact surrogate optimum $u_D$ —
единственный root

$$
u-\gamma=\sqrt 2\,D\left(\frac{u}{\sqrt 2}\right),
$$

где $D$ — Dawson function. Его large-threshold approximation:

$$
\theta^*(\theta^*-\phi)=s_G^2,
\qquad
\theta^*=\frac{\phi+\sqrt{\phi^2+4s_G^2}}{2}.
$$

В optimum surrogate rate равен

$$
R_D^*=\alpha s_G\sqrt{\frac{2}{\pi}}e^{-u_D^2/2}.
$$

Myopic threshold $\theta=\phi$ имеет нулевую surrogate reward. Если дополнительная
per-lot cost в price units равна $c$, paper model допускает заменить $\phi$ на
$\phi+c$; для фиксированного $\theta$ surrogate net rate положительна лишь при
$\theta>\phi+c$. Это теоретическое необходимое условие внутри surrogate, а не
доказательство empirical profitability.

### 2.3. Уровни доказанности

| Claim | Статус в статье | Что проверяем |
|---|---|---|
| Parity lock spread/mid | Exact grid identity | Pathwise invariant и event transitions |
| Linear reversion under balance | Exact theorem | Conditional drift regression по parity и $G$ |
| Ergodicity, variance и exponential ACF | Exact theorem | Stationary Monte Carlo moments и confidence bands |
| Full positions suffice | Exact pathwise layer result | Wealth/accounting identity; не численная optimality claim |
| Threshold optimality | Theorem только на Gaussian surrogate | Numeric optimizer против Dawson root |
| Threshold optimality на jump process | Conjecture | Не заявлять как доказанный результат; только band sweep |
| Overshoot reward error | Rigorous bound $<\delta$ | Pathwise overshoot test |
| Timing error $O(\delta/\theta)$ | Heuristic | Timestep/refinement и parameter sweep; не выдавать за theorem |
| Frozen tight-spread cost error | Controlled occupancy argument | Realized fill-state cost против frozen cost |
| Figure 4 optimum внутри $\theta_D$ примерно на 20% | Monte Carlo claim | Independent reconstruction, не forced acceptance |
| Profitability на реальном рынке | В статье не исследована | Отдельный causal out-of-sample protocol |

### 2.4. Что раскрыто и чего не хватает

| Поле | Статья v1 |
|---|---|
| Dataset, exchange, instrument, period | Отсутствуют; работа theoretical |
| Preprocessing/features | Не применимо к paper simulation; empirical contract отсутствует |
| Official code/data | Не заявлены и не найдены на 2026-08-11 |
| Official source archive | Только `main.tex`, `00README.json` и четыре готовых PDF figures; generator отсутствует, figure metadata указывает Matplotlib 3.10.8 |
| Simulator algorithm/discretization | Не раскрыты достаточно для exact numeric replication |
| Primitive parameters Figures 2, 4, 5 | Не раскрыты; figures 2/5 названы illustrative and uncalibrated |
| Figure 4 disclosed settings | Только $\gamma\approx0.28,0.36,0.47$, sweep ramp slopes при fixed unnamed baselines, bands = 1 SE |
| Seeds, paths, horizon, burn-in | Не указаны |
| Hardware, runtime, precision | Не указаны |
| Statistical protocol | Figure 4 сообщает 1 SE; число replications и construction не указаны |
| Optimizer/scheduler/training budget | Не применимо: trainable ML model отсутствует |
| Peer review | Не указан; preliminary arXiv v1 |

Следствие: Figure 3 можно воспроизвести численно из closed forms; Figures 2 и 5 —
структурно, но не pixel/numeric-identically; Figure 4 — только independent
reconstruction с явно выбранным protocol. Exact reproduction Figure 4 остаётся
`underdetermined`, пока не появятся авторские configs/code.

### 2.5. Главные assumptions и failure modes

1. $X$ — exogenous Brownian martingale и в теории наблюдаем, хотя на рынке latent.
2. Актив применим к two-valued spread regime; wider, locked/crossed и illiquid books
   лежат вне модели.
3. Balanced response устраняет parity-dependent drift; на данных он может не выполняться.
4. Trader мал, немедленно исполняется at touch и не меняет intensities; latency,
   adverse selection и market impact отсутствуют.
5. Единственная paper cost — realized half-spread; fees, slippage, borrow/funding и
   session boundaries отсутствуют.
6. Stationarity и constant $(\alpha,s_G)$ могут нарушаться intraday и между regimes.
7. OU surrogate совпадает только по conditional mean и covariance, не по first-passage
   law; jump overshoot существенен при большом $\delta/\theta$.
8. Always-full risk-neutral objective максимизирует expected long-run P&L rate, не
   utility, Sharpe, drawdown или capital efficiency.
9. Формулировка paper «one number governs everything» относится к normalized OU
   surrogate. Exact jump dynamics дополнительно зависят от $\delta/s_G$, baseline
   intensities, decomposition slopes, $\sigma_X^2/\sigma_M^2$ и open occupancy.
10. В Figure 4 $\gamma\approx0.28$--$0.47$ при $\phi=\delta/2$ означает
    $\delta/s_G=2\gamma\approx0.56$--$0.94$; заявленный asymptotic regime
    $\delta\ll\theta-\phi$ там не очевиден и должен быть отдельной convergence check.
11. Fast-parity $\alpha_{eff}$ при нарушенном balance — averaging approximation, не
    замена exact theorem; sweep $\gamma$ через ramp slopes меняет higher moments вместе
    с $\gamma$.
12. Последующие two-lot flips совместимы с inventory cap one lot на сторону, но требуют
    вдвое большей executable size и capacity, чем первый entry.
13. Empirical search легко переобучить через instrument, period, filter и threshold
   selection; нужен nested chronological design и untouched holdout.

## 3. Research questions и preregistered hypotheses

Формальные hypotheses будут перенесены без изменения смысла в соответствующие
protocols до запусков.

- **RQ1 / H-SYN-1:** реализованный jump model сохраняет parity, balanced conditional
  drift, stationary variance identity и exponential ACF в Monte Carlo uncertainty.
- **RQ2 / H-SYN-2:** Dawson root максимизирует независимо вычисленную surrogate rate;
  large-threshold root близок к нему в заявленном regime $\gamma\gtrsim0.4$.
- **RQ3 / H-SYN-3:** внутри symmetric band class exact jump optimum систематически
  сдвигается внутрь относительно $\theta_D$ из-за overshoot. Направление
  preregistered; диапазон paper claim не превращается в acceptance target.
- **RQ4 / H-EST-1:** causal filter восстанавливает usable gap signal на synthetic data
  и превосходит заранее выбранные causal baselines; retrospective smoother служит
  только diagnostic oracle и никогда не генерирует backtest orders.
- **RQ5 / H-EMP-1:** после costs net edge зависит прежде всего от effective
  $\gamma$, reversion speed, filter uncertainty, open-book occupancy, latency и
  balance violation. Знак empirical result заранее не предполагается.
- **RQ6 / H-EMP-2:** theory-derived $\theta_D$ является competitive baseline; tuned
  multiplier допускается только в nested validation и должен подтверждаться на untouched
  test.

## 4. Planned contracts и структура artifacts

### 4.1. Code boundaries

Будущая реализация должна разнести:

- immutable config DTOs и serializable `RunSpec`;
- paper primitives и immutable domain state;
- отдельный mutable simulator/runtime state;
- analytical surrogate/Dawson solver;
- jump simulator и random-number streams;
- causal state estimation;
- raw market adapters и canonical event schema;
- strategy policy, execution model и accounting;
- metrics/statistics/report generation;
- artifact store с atomic run manifest.

[`ADR-0001`](adr/0001-research-modes-package-and-run-contract.md) зафиксировал
import-package `src/ot_micromr/`, distribution/CLI `ot-micromr` и единый module CLI.
Package ещё не создан: это P2 implementation, а не результат P1.

### 4.2. Configuration contract

Каждый executable config в `cfg/experiments/` использует strict TOML `RunSpec v1` и
полностью задаёт:

- `experiment_id`, `track`, `mode` (`paper-faithful`, `practical-local` или
  non-claim `oracle-diagnostic`);
- model/simulator/filter/strategy/execution parameters и units;
- seed или ordered seed list;
- dataset ID, immutable version/hash, venue, symbol, tick/contract specification;
- chronological split boundaries и timezone/session policy;
- cost and latency scenarios;
- metrics, statistical tests, bootstrap scheme и acceptance thresholds;
- output root, precision и требование clean/claim-eligible run.

Config не содержит неизвестные до запуска commit, environment, hardware и timestamps.
Resolved config и эти runtime-derived facts сохраняются в immutable run manifest;
source TOML после запуска не переписывается под результат.

### 4.3. Canonical market-event contract

До загрузки данных ADR должен определить как минимум:

- exchange timestamp, receive timestamp (если доступен), sequence number, venue,
  symbol и session;
- best bid/ask prices and sizes до и после event, tick size и contract multiplier;
- trades, quote updates, locked/crossed states, gaps in sequence и recovery semantics;
- детерминированную классификацию `slide/open/close/other` и simultaneous-event policy;
- causal feature timestamps: ни один signal value не может использовать payload,
  опубликованный после decision time;
- corporate actions/rolls/funding/borrow для выбранного asset class;
- raw data неизменяемы и не коммитятся; processed dataset получает version/hash и
  data-quality report.

### 4.4. Artifact layout

- `docs/protocols/synthetic/` — preregistered analytical/simulation runs;
- `docs/protocols/empirical/` — data, estimation, backtest и statistical protocols;
- `docs/adr/` — architecture, simulator semantics, data contract, costs и acceptance
  decisions;
- `docs/reports/` — фактические results/limitations/provenance;
- `cfg/experiments/` — executable contracts;
- `outputs/<experiment_id>/<run_id>/` — resolved config, manifest, raw metrics, logs,
  plots и optional checkpoints/filter state; heavy artifacts остаются untracked.

## 5. Порядок работ и acceptance gates

Статус scientific claims ведётся отдельно от completion stage: research stage может быть
завершён с результатом `not-reproduced` или `negative`, если protocol исполнен честно.

| Stage | Track | Status | Completion gate |
|---|---|---|---|
| P0. Source audit и план | Common | **completed** | PDF/source проверены, unknowns и scope зафиксированы |
| P1. Contracts и protocols | Common | **completed** | ADRs, schema, configs и acceptance thresholds приняты до runs |
| P2. Analytical executable baseline | Synthetic | **completed** | Closed forms, tests, smoke run и Figure 3 reproduction |
| P3. Jump simulator и theorem checks | Synthetic | **completed; gate failed** | Invariants прошли; flow/control precision gates не прошли |
| P3S. Statistical gate audit | Common | **completed** | Gates classified; TOST/superiority/multiplicity/power policy accepted |
| P3V. Margin sensitivity и powered validation | Synthetic | **completed; supported** | SESOI bounded by downstream distortion; new independent confirmatory runs passed |
| P4. Figures 2/4/5 и paper report | Synthetic | **in progress** | Claim matrix с reproduced/not-reproduced/underdetermined |
| P5. Data feasibility и universe freeze | Empirical | pending | Licensed dataset, quality gate, immutable splits |
| P6. Causal efficient-price estimation | Empirical | pending | Synthetic recovery и real-data diagnostics без P&L tuning |
| P7. Event-driven backtester | Empirical | pending | Accounting, timestamp, fill, cost и latency tests |
| P8. Nested development/validation | Empirical | pending | Заморожена одна primary strategy и limited secondary set |
| P9. Untouched test и robustness | Empirical | pending | Profitability gate либо честный negative result |
| P10. Synthesis/release | Common | pending | Plan, ADRs, reports, README и provenance согласованы |

### P0. Source audit и план — completed

Выполнено:

- подтверждены version, metadata, primary URL, license и hash PDF;
- проверены official source archive и отсутствие заявленного official code/data;
- извлечены claims, equations, assumptions и открытые вопросы;
- literal replication отделена от нового empirical extension;
- создан paper registry и настоящий канонический план.

Проверки P0 — только documentation/source checks; experiment results ещё не получены.

### P1. Mathematical, simulation и experiment contracts — completed

Выполнено до любых target runs:

1. [`ADR-0001`](adr/0001-research-modes-package-and-run-contract.md) разделил tracks и
   information-set modes, зафиксировал units, package/CLI naming, strict TOML
   `RunSpec v1`, runtime manifest и immutable artifact layout.
2. [`ADR-0002`](adr/0002-controlled-jump-simulation-semantics.md) зафиксировал
   `adaptive_left_hazard_single_jump_v1`: frozen-left event approximation, mandatory
   observation boundaries, event ordering, hybrid Brownian first-hit semantics,
   deterministic RNG mapping, invariant suite и epsilon refinement.
3. Зарегистрирован
   [`paper-reproduction.md`](protocols/synthetic/paper-reproduction.md) с primitives,
   20 seeds, horizons, statistical estimands, simultaneous inference, acceptance и stop
   rules. Protocol имеет статус `preregistered`; target outputs не просматривались.
4. Созданы пять executable contracts: `ANA-SMOKE-001`, `ANA-FIG3-001`,
   `SIM-MOMENTS-001`, one-factor negative control `SIM-UNBALANCED-001` и independent
   partial reconstruction `SIM-FIG4-001`. Последний сохраняет author settings как
   `underdetermined` и не требует numeric match с paper ranges.
5. Default precision — CPU `float64`. P1 не добавил dependencies; подтверждено наличие
   CPython 3.14 wheels у актуальных NumPy/SciPy, а exact pins откладываются до `uv add`
   и `uv.lock` в P2.
6. `.gitignore` до artifact runs исключает outputs, raw datasets, logs, checkpoints,
   secrets и caches. Import path в `AGENTS.md` исправлен на `src/ot_micromr/`.

Gate P1 — **passed**: все пять TOML parse standard-library `tomllib`, IDs/required
tables/units/seeds/refinement contracts согласованы, local links и paper hash проверены,
`git diff --check` проходит. Два независимых adversarial review не оставили High/Medium
blockers. Это structural/preregistration evidence, не scientific result: simulator,
strict typed validator и target experiments ещё не реализованы и не запускались.

Factual closeout: [`docs/reports/p1-contracts.md`](reports/p1-contracts.md). Contract
commit: `36d606e`.

### P2. Minimal analytical baseline — completed

Задачи:

1. Реализовать `erfi` passage time, Dawson FOC, robust bracketing/root solve,
   $\theta^*$ и rate curves.
2. Начать с одного dimensionless config по $(\alpha,s_G,\gamma)$ и одной метрики:
   residual Dawson FOC в найденном optimum.
3. Добавить unit tests для scaling, units, boundary $u_D>\gamma$, uniqueness,
   derivative sign и asymptotic root.
4. Независимо построить Figure 3 и табулировать threshold/rate discrepancy по $\gamma$.
5. Сохранить resolved config, numerical library versions и figure data отдельно от PNG.

Reference checkpoints из независимо вычисляемых paper formulas, которые нужно
пересчитать нашим кодом, а не hardcode как output:

| $\gamma$ | $u_D$ | $u^*$ | rate loss at $u^*$ |
|---:|---:|---:|---:|
| 0.05 | 0.541693 | 1.025312 | 8.71% |
| 0.28 | 1.006563 | 1.149752 | 1.29% |
| 0.40 | 1.155873 | 1.219804 | 0.30% |
| 1.70 | 2.264976 | 2.162440 | 2.30% |

Acceptance:

- closed forms согласованы независимыми direct integration/optimization checks;
- double-precision identities проходят с exact tolerances, уже зафиксированными в
  `ana_smoke_001.toml` и `ana_fig3_001.toml`;
- myopic $\widetilde R(\phi)=0$ и rate стремится к нулю на дальнем конце grid;
- reproduce/contradict status для claims Figure 3 дан с numeric table, не только plot;
- проходят `uv run python main.py`, package CLI и relevant unit tests после появления
  package/CLI.

Результат P2 — **passed / reproduced**. Из clean commit `710efa9` сначала прошёл
`ANA-SMOKE-001`, затем `ANA-FIG3-001`. Smoke дал $u_D=1.1558728538424254$ при
$\gamma=0.4$, Dawson residual $2.22\times10^{-16}$ и discrepancy с независимым
bounded optimizer $1.80\times10^{-8}$ при gate $10^{-7}$. На 296-point Figure 3 grid
maximum residual равен $1.16\times10^{-13}$; maximum rate loss при
$\gamma\ge0.4$ равен `0.023035703614617375` at $\gamma=1.66$, в соответствии с
preregistered audit. Все myopic rates в точках $u=\gamma$ равны нулю, grid endpoints
лежат на нисходящей ветви rate curves, а exact tail стремится к нулю по asymptotic
росту `erfi`.

Implementation включает strict immutable validator для двух P2 analytical contracts,
canonical serialization, atomic `run-manifest-v1`, raw CSV/table/PNG separation и
CLI. Зафиксированы NumPy 2.5.2, SciPy 1.18.0 и Matplotlib 3.11.1 в `uv.lock`; 25 unit
tests и оба entrypoints проходят. `SIM-*` typed schema намеренно остаётся P3 scope.

Factual closeout: [`docs/reports/p2-analytical-reproduction.md`](reports/p2-analytical-reproduction.md).
Architecture/dependency decision: [`ADR-0003`](adr/0003-analytical-baseline-stack.md).

### P3. Exact-model dynamics и controlled numerical simulator

Задачи:

1. Реализовать six event types, Brownian $X$, parity-locked spread, state-dependent
   intensities и event log.
2. Начать с adaptive small-step scheme с ограничением total event probability на step,
   либо обосновать более точный sampler в ADR. Для full result сравнить минимум две
   resolutions; все results называть simulation of the exact model, но не exact sampler,
   если есть time discretization. Continuous band crossings от Brownian $X$ между grid
   points учитывать Brownian-bridge/first-passage mechanism либо отдельно измерять
   missed-crossing bias при refinement.
3. Добавить pathwise tests: nonnegative intensities, allowed jump sizes, no illegal parity
   transition, at most one event per discretized step, deterministic replay by seed,
   overshoot $<\delta$ и wealth-marking identity $W-W^X=qG$.
4. На stationary paths проверить conditional drift отдельно для tight/open parity,
   $\mathbb E G$, variance identity, $p$ flow balance и ACF decay.
5. Проверить balanced и deliberately unbalanced control; менять один factor за run.

Acceptance:

- ни одного invariant violation в smoke/full logs;
- halving step меняет primary estimates не больше preregistered numerical tolerance или
  их Monte Carlo uncertainty;
- theoretical finite-h parity slopes и ACF лежат в simultaneous confidence bands на
  preregistered lags; gap bins остаются заранее заданным descriptive diagnostic;
- negative/unbalanced control обнаруживает parity split ожидаемого знака;
- report содержит effective sample size, seeds, burn-in, horizon, runtime и hardware.

Результат P3 — **completed / acceptance failed / inconclusive**. Реализован
`adaptive_left_hazard_single_jump_v1`; оба preregistered experiments выполнены из
clean commit `6aa0e48` на 20 seeds и всей ladder
$\epsilon\in\{0.02,0.01,0.005\}$. Все pathwise invariants, bitwise primary replays,
generator identities и balanced refinement coordinates прошли. Balanced mean,
variance, finite-$h$ slopes и шесть ACF lags согласованы с theory.

Два заранее заданных statistical gates не подтверждены при редкой open parity:

- `SIM-MOMENTS-001`: primary flow-balance error `0.0333788` превысил limit `0.03`,
  хотя fine-resolution error равен `0.0114669`, primary t-interval signed residual
  включает zero и весь refinement gate прошёл;
- `SIM-UNBALANCED-001`: directional contrast mean положителен (`0.257337`), но
  one-sided 95% lower bound `0.0240021` ниже требуемых `0.05`; primary--fine shift
  `0.189537` также превысил conservative difference-SE threshold `0.168574`.

Open occupancy составила лишь около `0.0032--0.0036`, поэтому open-parity slopes и
flow ratios имеют высокую seed-level uncertainty. Это не исправлялось post hoc:
configs, horizons, seeds, estimators и thresholds не менялись; оба failed runs и
примерно 112 MiB event logs сохранены локально. Paper flow identity и negative control
получают status `inconclusive`, а не `contradicted`; P4 запрещён до нового
preregistered precision experiment ID или честного решения остановить synthetic track.

Factual closeout: [`docs/reports/p3-controlled-simulation.md`](reports/p3-controlled-simulation.md).
Stop/next-direction decision: [`ADR-0004`](adr/0004-p3-gate-failure-and-precision-stop.md).

Post-run review [`ADR-0005`](adr/0005-statistical-decision-gates.md) показал, что
проблема шире двух historical failures. Старые point-estimate gates не различали
compatibility, equivalence и low power, а refinement rule мог проходить при большой
SE. Retrospective TOST с original margins подтверждает mean, variance и tight drift,
но flow и open drift остаются inconclusive; family-adjusted refinement equivalence
почти нигде не установлена. Historical configs/results не меняются и не получают
post-hoc pass. Полный audit: [`statistical-gate-audit.md`](reports/statistical-gate-audit.md).

### P3S. Statistical gate audit — completed

Для всех текущих и planned gates введена taxonomy:

- deterministic/operational gates без фиктивных p-values;
- equality через multiplicity-aware TOST/equivalence;
- directional claims через one-sided superiority/non-inferiority over justified SESOI;
- refinement через paired/independent equivalence без `tolerance OR SE`;
- `inconclusive` как обязательный outcome при недостаточной precision;
- power design до target output с independent seed/session/fold unit.

Reference implementation находится в `ot_micromr.statistical_gates`; policy —
[`statistical-gates-v1`](protocols/common/statistical-gates.md). Никакие новые
confirmatory outputs в P3S не создавались.

### P3V. Margin sensitivity и powered validation — completed

До регистрации `SIM-*002` требуется:

1. Для каждого P3 estimand измерить downstream sensitivity Figure 4 peak/rate claims
   и обосновать equivalence margin; old `2/3/5%` thresholds не переносятся автоматически.
2. Объявить one primary family, familywise alpha/correction и three-way decisions.
3. Использовать old P3 seed metrics только как pilot variance; confirmatory seeds должны
   быть новыми и независимыми.
4. Сравнить power per compute для большего horizon, большего seed count и более
   эффективного event/compensator estimator без изменения paper-model semantics.
5. До запуска заморозить required sample size, maximum compute, stopping rule и новые
   config IDs. Underpowered design получает `blocked-precision`, а не binary gate.

Рабочий protocol заморожен в
[`p3v-powered-validation.md`](protocols/synthetic/p3v-powered-validation.md), решение об
estimators/compute — в
[`ADR-0006`](adr/0006-p3v-estimators-power-and-compute.md). Зафиксированы новые IDs
`SIM-MOMENTS-002` и `SIM-UNBALANCED-002`, 20 независимых новых seeds, measured horizon
`20000`, resolutions `{0.01, 0.005}`, отсутствие optional stopping и раздельные Holm
families для primary и refinement claims. План работ был следующим:

1. реализовать integrated hazards/compensators и process-parallel runner;
2. benchmark `1/4/10/20` CPU workers на pilot workload и проверить bitwise equality;
3. выпустить sensitivity/power table до чтения `SIM-*002` target outputs;
4. выполнить оба powered runs и применить global family decision;
5. только при supported decision перейти к `SIM-FIG4-002`.

Pilot sensitivity и compute design выполнены до target runs. Endpoint-only fault injection
на 20 historical seed labels и horizon `10000` отвергла provisional margins `0.10/0.15`:
open-drift fault `+15%` менял normalised peak rate с paired 95% interval примерно
`[-2.14%, -0.34%]`. Датированный amendment уменьшил flow margin до `0.05`, вернул
точный drift claim к deterministic generator gate и оставил stochastic
jump-compensator negative control. Power design теперь задаёт horizon `40000` для
`SIM-MOMENTS-002`, `20000` для `SIM-UNBALANCED-002`, 20 новых seeds и две отдельные
Holm families. Factual pilot report:
[`p3v-sensitivity-and-power.md`](reports/p3v-sensitivity-and-power.md).

Оба target run выполнены 2026-08-11 из clean commit
`9dedb6b44d87933147420a316e445b19cf4c5080` без optional extension:

- `SIM-MOMENTS-002/20260811T184531842286Z-4cb501542645-det`, `1189.46 s`;
- `SIM-UNBALANCED-002/20260811T190621510298Z-f3c0ff8a3b29-det`, `583.86 s`.

Primary Holm family поддержана: flow equivalence estimate `0.01567`, margin `0.05`,
adjusted $p=0.00629$; planted unbalanced contrast estimate `0.23015` против minimum
effect `0.10`, adjusted $p=0.00297$. Refinement family также поддержана: adjusted
$p=0.01508$ для обеих equivalence checks. Все deterministic/operational gates прошли.
Global artifact: `outputs/p3v/global-gate.json`; подробности и limitations находятся в
[`p3v-sensitivity-and-power.md`](reports/p3v-sensitivity-and-power.md). Gate P3V имеет
status **supported**, переход к P4 разрешён.

До реализации P4 выполнен backend benchmark на representative endpoint band kernel
(20 paths × 50000 observations × 21 thresholds). Compiled CUDA с включённым transfer
получил `123.2x` speedup в `float64` относительно vectorized NumPy при максимальной
ошибке rate `2.9e-13`. Дополнительный 84-million-element benchmark показал, что compiled
`float32` ещё в `1.26x` быстрее `float64`, уменьшает runtime на 20.7%, сохраняет counts
точно и имеет maximum rate error `7.19e-6` (`1.82e-7` reference scale). Поэтому policy/threshold post-processing P4
использует optional PyTorch CUDA backend и `torch.compile`, с `float32` как primary
candidate и `float64` как regression oracle. CPU event-path generator сохраняет
измеренный 10-process backend, пока отдельная реализация не докажет end-to-end выигрыш
без изменения scientific semantics.

Backend decision: [`ADR-0007`](adr/0007-p4-hybrid-cpu-cuda-backend.md).

### P4. Independent paper-result reconstruction

Задачи:

1. Figure 2: structural sample path с явно выбранными illustrative primitives; не
   называть pixel-identical reproduction.
2. Figure 4: sweep $\theta/\theta_D$ для preregistered parameter families, включая
   $\gamma\approx0.28,0.36,0.47$; оценить discrete-grid peak mean-rate curve,
   diagnostic fitted peak, 1 SE, overshoot, loss at $\theta_D$ и $\theta^*$.
3. Проверить refinement по $\delta/\theta$, open occupancy $p$, ramp share и horizon.
4. Figure 5: pathwise signal, fills, two-lot flips, spread-aware bands и both wealth
   markings; slope comparison только после stationary burn-in. Figure 5 использует
   parity-dependent $\theta_D(S_t)$, тогда как main closed-form analysis freezing делает
   $\phi=\delta/2$; это отдельная strategy variant/ablation, не тот же config.
5. Выпустить `docs/reports/paper-reproduction.md` и обновить claim matrix/status здесь.

Acceptance означает полноту и воспроизводимость проверки, а не обязательное совпадение.
Для Figure 4 сохраняется label `independent partial reproduction`, пока неизвестны
author primitives/seeds. Опубликованные диапазоны (примерно 20% inward shift, 3--4%
rate loss at $\theta_D$, 5--6% at $\theta^*$) сравниваются с confidence intervals, но не
служат tuning objective.

Перед новым Figure 4 run legacy `SIM-FIG4-001` заменяется новым ID. Inward-shift claim
требует family-adjusted one-sided lower bound выше заранее обоснованного minimum shift;
rate/loss estimates публикуются с seed-cluster intervals. Numerical refinement проходит
только equivalence test. Pilot amendment заменил эвристический floor 100 на pathwise floor
20 и отдельный powered seed-cluster design; эти два gate не подменяют друг друга.

Непосредственный следующий шаг: реализовать continuous-crossing policy monitor,
preregister `SIM-FIG4-002` и до target inspection benchmark-ом выбрать CPU/GPU execution
для полного, а не endpoint-only, kernel.

P4 pilot и target freeze выполнены 2026-08-11. CPU pilot
`20260811T200458288216Z-d9d6dc74bb9e-det` занял `186.996 s`; эквивалентный hybrid
CPU/CUDA pilot `20260811T201826249541Z-d9d6dc74bb9e-det` — `34.424 s` (`5.43x`).
Target `SIM-FIG4-002` preregistered на трёх response rows, 30 новых seeds, horizon 300,
grid step 0.05, 10,000 bootstrap draws и 150-second stop budget. Power calculation,
minimum effect 0.05 и secondary refinement limitation зафиксированы в
[`p4-figure-reconstruction.md`](protocols/synthetic/p4-figure-reconstruction.md) до target.
Figures 2 и 5 генерируются как явно illustrative artifacts того же run.

Target выполнен один раз из clean commit `ca9aa7c1e9841fccb35f47f41a8e0863e795d3c7`:
`SIM-FIG4-002/20260811T202753134457Z-837035232ead-det`, runtime `34.244 s`.
Operational acceptance failed только из-за all-cell interval floor: minimum 12 против 20
в far-right high-gamma cells; остальные deterministic/numerical gates прошли. Inward shift
15% и 20% supported для gamma `0.272/0.342`, high-gamma 5% inconclusive; refinement family
inconclusive. P4 получает status **completed / acceptance failed / scientific inconclusive**.
Полный evidence report: [`paper-reproduction.md`](reports/paper-reproduction.md).

Следующий шаг — P5: выбрать licensed event-level source и до просмотра P&L заморозить
universe, eligibility и chronological split. P4 target не перезапускается.

### P5. Data feasibility, licensing и frozen universe

До выбора data source empirical код не должен делать implicit network downloads.

Задачи:

1. Выбрать licensed event-level L1/L2 quotes и trades с sequence/timestamp semantics,
   достаточными для восстановления touch и marketable fills. Aggregated candles не
   подходят для primary test.
2. До holdout screening зафиксировать market/venue, candidate universe, calendar,
   contract/tick history, timezone, sessions и asset-specific costs.
3. На development interval проверить large-tick eligibility: occupancy one/two-tick
   spread, fraction wider/locked/crossed, update rate, touch depth, gaps and bad records.
   Numeric eligibility cutoffs preregister до universe screening; paper-faithful и
   relaxed practical cohorts хранить отдельно.
4. Сделать immutable chronological train/validation/test splits. Test не используется
   для instrument eligibility, feature/filter choice или parameter search.
5. Провести power/precision analysis по числу independent sessions/trades; если данных
   недостаточно, результаты пометить exploratory.

Gate P5:

- license разрешает локальный research use и потенциальную публикацию derived metrics;
- raw provenance/hash и data-quality report сохранены;
- event ordering/recovery semantics однозначны;
- universe и split freeze датированы до strategy P&L inspection;
- uncertainty по eligibility/data-quality rates опубликована, planned stochastic gates
  имеют justified margin/MDE, multiplicity family и достаточную power по independent sessions;
- при отсутствии подходящего dataset empirical track получает `blocked-data`, но
  synthetic reproduction остаётся валидной и продолжается.

### P6. Causal estimation of latent efficient price

Primary scientific problem practical track — не threshold tuning, а оценка $X_t$.

Задачи:

1. Реализовать synthetic oracle с известным $X$ как upper bound.
2. Реализовать causal point-process filter/particle filter, соответствующий six-event
   likelihood, включая информацию и от events, и от silence/no-event survival.
3. Реализовать простой causal Gaussian/Kalman surrogate и заранее выбранный naive
   causal baseline; retrospective smoother разрешён только как labelled diagnostic.
4. Оценивать $\mu_i,\alpha_i,\sigma_X$ и reduced $(\alpha,s_G)$ только на past training
   windows с constraints/uncertainty. Balanced model сравнить с unbalanced alternative
   по out-of-sample likelihood и calibration, не по trading P&L.
5. На real data проверить stationarity, ACF, event residuals/time rescaling, parity-specific
   drift, parameter stability и posterior uncertainty.

Stop/gate P6:

- на synthetic data paired, multiplicity-aware one-sided lower bound improvement causal
  estimator над preregistered naive baseline должен превышать заранее обоснованный
  minimum useful effect по state error и out-of-sample observation likelihood; oracle
  gap не попадает в feasible backtest;
- calibration/equality claims проходят equivalence test; non-significant difference
  без equivalence получает `inconclusive`;
- все empirical signals вычисляются online и timestamp audit не находит future access;
- если filter uncertainty сравнима с или больше option-value margin
  $\theta-(\phi+c)$ и signal не превосходит controls, P&L optimization останавливается
  и результат фиксируется как negative feasibility finding.

### P7. Event-driven backtest и accounting

Задачи:

1. Реализовать target policy точно по статье: first one-lot entry, затем two-lot flips;
   отдельно labelled practical variant может иметь flat zone/session flattening.
2. Decision возникает только после доступного event; fill price — первый исполнимый
   displayed quote не раньше `decision_time + latency`. Zero-latency paper convention,
   realistic latency и stress latency — разные scenarios.
3. Проверять touch depth и size/participation; при недостаточной depth применять
   preregistered partial/rejected/walk-the-book rule, а не optimistic fill.
4. Считать P&L из cash and inventory ledger. Не double-count spread: execution at bid/ask
   уже включает его; отдельно добавляются fee, slippage, borrow/funding, rolls и forced
   liquidation.
5. Сохранять gross, paper spread-only и full-net decompositions, mid-marked and cash
   reconciliations, every decision/fill/rejection и causal feature snapshot.
6. Добавить tests на timestamp ordering, no-look-ahead, flip quantity, fees, latency,
   session boundary, missing quotes, mark-to-market и deterministic replay.

Gate P7: hand-computed toy paths и property tests полностью сходятся; каждый P&L change
воспроизводится из fill ledger; impossible/optimistic fills отсутствуют.

### P8. Nested development и limited parameter search

Filter/model hyperparameters выбираются по filtering likelihood/calibration, а не по
strategy test P&L. Trading search начинается только после их freeze.

Primary baseline:

- causal filtered $\widehat G_t$;
- realized state-dependent half-spread и known per-lot costs;
- Dawson $\theta_D$ from rolling/past-only $(\widehat\alpha,\widehat s_G)$;
- one-lot/two-lot flip semantics;
- preregistered eligibility/risk gates.

Required controls/ablations, по одному изменению за comparison:

- no-trade accounting baseline;
- myopic break-even threshold;
- asymptotic $\theta^*$ versus Dawson $\theta_D$;
- limited multiplier grid around $\theta_D$ (candidate set
  $k_D\in\{0.7,0.8,0.9,1.0,1.1\}$, окончательно freeze в protocol);
- oracle versus causal filter только на synthetic data;
- balanced versus unbalanced response;
- frozen tight spread versus realized spread;
- zero versus realistic latency; spread-only versus full costs;
- parity-blind versus parity-aware band;
- paper always-full versus separately labelled practical risk controls;
- random/same-turnover and sign-reversed or delayed-signal negative controls.

Использовать nested chronological walk-forward: train filter/model, validation выбирает
единственный primary trading variant, следующий fold тестирует без адаптации. Overlapping
holding periods учитываются purge/embargo или session-block inference. Полный search
budget и все failed cells сохраняются.

Gate P8: до untouched test заморожены один primary config, secondary family,
aggregation/statistical method и stress scenarios; test hash и даты записаны, но P&L не
просмотрена.

### P9. Untouched test, profitability conditions и robustness

Primary analysis выполняется один раз. Повторное открытие test после изменения метода
требует нового dataset/time period и нового protocol.

Метрики:

- net/gross P&L per time, session, round trip и traded notional;
- turnover, fill/reject counts, holding time, exposure и cost attribution;
- daily/session Sharpe с явной annualization, downside deviation, hit rate, maximum
  drawdown и tail loss;
- confidence interval, effective sample size и probability of backtest overfitting;
- capacity proxy: touch depth, participation и P&L under size/slippage scenarios.

Statistical protocol:

- session/day block bootstrap, сохраняющий intraday dependence; block choice фиксируется
  до target test;
- primary one-sided 95% lower confidence bound; secondary family — SPA/Reality Check
  или заранее выбранная FWER/FDR correction;
- primary family фиксирует minimum economically relevant net P&L rate, target power
  не ниже 0.90 и число independent sessions до открытия untouched test;
- uncertainty reported both across time blocks and, где применимо, across instruments;
- не смешивать folds/instruments как independent ticks;
- cost stress не менее 1.25x для uncertain fee/slippage components и минимум один
  conservative latency scenario; конкретные значения следуют из venue evidence.

Карта условий строится **только из out-of-fold/holdout observations** и включает:

| Condition variable | Dimensionless form | Preregistered expectation, не факт |
|---|---|---|
| Effective spread and costs | $\gamma_{eff}=(\phi+c)/s_G$ | Меньше — больше option-value margin и rate |
| Reversion speed | $\alpha$ и half-life $\log 2/\alpha$ | Быстрее полезно лишь при latency намного меньше half-life |
| Latency | $\alpha L$ | Рост должен уменьшать realized edge |
| Jump granularity | $\delta/\theta$ и $\delta/(\theta-\phi-c)$ | Большие значения ухудшают surrogate accuracy |
| Open occupancy | $p=\Pr(S=2\delta)$ и fill-state occupancy | Большое $p$ повышает realized costs/model mismatch |
| Balance violation | $|\alpha_c-(2\alpha_s+\alpha_o)|/\alpha_{eff}$ | Большое значение ломает single-rate theory |
| Filter uncertainty | posterior SD / $s_G$ и / margin | Большая uncertainty размывает threshold crossings |
| Parameter stability | drift/variance across past windows | Нестабильность должна ухудшать transfer |
| Liquidity/capacity | order size / touch depth | Рост должен повышать rejections/slippage/impact risk |

Condition считается найденным не по красивой heatmap cell, а если её sign/direction
стабилен между preregistered folds/instruments, uncertainty исключает practically null
effect и результат повторяется на untouched data. Иначе это exploratory hypothesis.

Profitability gate:

1. multiplicity-adjusted primary full-net lower 95% bound выше нуля и заранее
   обоснованного minimum economically relevant net rate;
2. positive result после multiplicity-aware selection;
3. non-inferiority under preregistered moderate cost and latency stress подтверждена
   относительно заранее заданной downside margin, а не point estimate;
4. paired superiority над no-trade/random controls подтверждена по заранее выбранной
   risk-adjusted metric и minimum effect;
5. достигнут preregistered power/precision budget по independent sessions; иначе status
   `inconclusive`, независимо от знака point estimate;
6. нет критического data leakage, fill optimism или single-regime concentration.

### P10. Synthesis и release

Задачи:

- финально обновить этот plan, ADRs и отдельные synthetic/empirical reports;
- для каждого claim дать links на experiment IDs, configs, tables, plots и logs;
- перечислить deviations, failed runs, tuning budget и unresolved assumptions;
- обновить README с воспроизводимыми `uv run ...` командами;
- выполнить baseline CLI/tests, fresh-environment smoke и artifact manifest audit;
- дать итог один из: `profitable-under-stated-conditions`, `not-confirmed`,
  `negative`, `data-blocked`, без усиления формулировки.

## 6. Начальная experiment matrix

P1 configs зарегистрированы как strict TOML contracts; будущие имена остаются planned
до своих protocol stages. Разные information-set modes не агрегируются без явной
колонки `mode`.

| Experiment ID | Config | Claim/output | Status |
|---|---|---|---|
| `ANA-SMOKE-001` | `cfg/experiments/ana_smoke_001.toml` | One config, Dawson residual, one metric | passed; `20260811T170052058822Z-290ea5809cb6-det` |
| `ANA-FIG3-001` | `cfg/experiments/ana_fig3_001.toml` | Surrogate thresholds/rate curves | reproduced; `20260811T170104389894Z-4c1014e843c6-det` |
| `SIM-MOMENTS-001` | `cfg/experiments/sim_moments_001.toml` | Parity, drift, variance, ACF, occupancy | acceptance failed; `20260811T172240473272Z-060cfab011c3-det` |
| `SIM-UNBALANCED-001` | `cfg/experiments/sim_unbalanced_001.toml` | One-factor parity-drift negative control | acceptance failed; `20260811T172916459381Z-5497c39af2dd-det` |
| `SIM-FIG4-001` | `cfg/experiments/sim_fig4_001.toml` | Jump versus surrogate band sweep | blocked by P3; not run; underdetermined-author-settings |
| `SIM-FIG4-PILOT-002` | `cfg/experiments/sim_fig4_pilot_002.toml` | Crossing/variance/compute pilot | completed; non-claim; operational floor expectedly missed |
| `SIM-FIG4-002` | `cfg/experiments/sim_fig4_002.toml` | Powered jump versus surrogate band sweep | completed; acceptance failed; science inconclusive; `20260811T202753134457Z-837035232ead-det` |
| `SIM-FIG5-001` | integrated into `SIM-FIG4-002` artifacts | Strategy path, fills, wealth identities | completed illustrative artifact |
| `EMP-DATA-001` | `cfg/experiments/emp_data_001.toml` | Eligibility, quality, split freeze | pending-data-source |
| `EMP-FILTER-001` | `cfg/experiments/emp_filter_001.toml` | Oracle/causal filter diagnostics | pending |
| `BT-SMOKE-001` | `cfg/experiments/bt_smoke_001.toml` | Toy ledger and no-look-ahead | pending |
| `BT-WF-001` | `cfg/experiments/bt_wf_001.toml` | Nested development/validation | pending |
| `BT-HOLDOUT-001` | `cfg/experiments/bt_holdout_001.toml` | Locked primary test | pending |

## 7. Stop criteria и порядок решений

- Не переходить к Figure 4 claims, пока simulator не прошёл invariants и resolution
  convergence.
- Не переходить к real-data P&L search, пока causal filter не прошёл P6 на synthetic
  controls и timestamp audit.
- Не пытаться «найти прибыль» подбором costs, venue interval или universe после
  просмотра test.
- Если gross edge не покрывает spread ещё до uncertain costs, остановить parameter
  expansion и записать отрицательный результат.
- Если эффект существует только при zero latency, oracle $X$, ignored fees или
  unrealistic fills, классифицировать его как theoretical upper bound.
- Если подходящие licensed event data недоступны, не заменять их candles для primary
  claim; остановить empirical track как `blocked-data`.
- После отрицательного holdout новый поиск — отдельная hypothesis, protocol, config и
  новый untouched period, а не amendment старого результата.
- Новые filters, flat zones, risk objectives или execution logic вводить только как
  отдельно специфицированные extensions с baseline/ablation.

## 8. Definition of done

Проект завершён, когда:

1. Все paper claims из Section 2.3 имеют evidence-linked status и independent report.
2. Analytical baseline и simulator воспроизводятся из fresh checkout одной CLI-командой
   на tiny config и documented full configs.
3. Empirical data/filter/backtest имеют immutable provenance и causal audit либо явно
   зафиксированный blocker.
4. Profitability statement, если оно есть, относится только к названным venue,
   instruments, periods, size, costs, latency и confidence level.
5. Условия доходности подтверждены out of sample; либо честно документировано, что при
   проверенных условиях они не найдены.
6. Plan, ADRs, protocols, reports, README, tests и experiment manifests согласованы;
   negative results и deviations сохранены.

## 9. Ближайший исполняемый шаг

P4 завершён без post-target extension. Следующий шаг — P5 data feasibility: выбрать
licensed event-level L1/L2 source, зафиксировать venue/instruments/calendar/cost metadata,
preregister large-tick eligibility cutoffs и chronological train/validation/test split до
любого strategy P&L. При отсутствии лицензируемых данных stage получает `blocked-data`.
