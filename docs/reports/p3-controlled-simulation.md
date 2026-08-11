# P3 report: controlled jump simulation и theorem checks

- **Stage:** P3 — Exact-model dynamics and controlled numerical simulator
- **Status:** completed; overall gate failed
- **Scientific status:** mixed evidence; flow/control claims inconclusive
- **Дата:** 2026-08-11
- **Ветка:** `feat/p3-jump-simulator`
- **Target implementation commit:** `6aa0e4806381254f5773cba25fc9281d92173eb1`
- **Protocol:** [`paper-reproduction.md`, frozen v1.1](../protocols/synthetic/paper-reproduction.md)
- **Dataset/checkpoint:** not applicable; synthetic paper model with project parameters
- **RNG:** NumPy `PCG64DXSM`, 20 declared master seeds, four named spawned streams

## Outcome

Оба preregistered P3 runs выполнены полностью: 20 seeds, burn-in 100 reversion times,
measured horizon 2000 reversion times и
$\epsilon\in\{0.02,0.01,0.005\}$. Каждый primary seed replayed bitwise. Ни одного
pathwise invariant failure, NaN/Inf, illegal transition, simultaneous book event или
replay mismatch не обнаружено.

Overall P3 acceptance не пройден:

- balanced flow-balance error при primary resolution немного превысил numeric gate;
- unbalanced directional lower confidence bound и its refinement stability не прошли.

Результат классифицируется как `inconclusive / insufficient rare-state precision`, а
не как опровержение paper model. Figure 4 и P4 заблокированы.

## Method

Реализован `adaptive_left_hazard_single_jump_v1`, то есть controlled approximation,
не exact sampler. Canonical state — `(t, X, mid_half_ticks)`; mid, spread, bid и
$G=M-X$ всегда derived. На каждом step:

1. шесть intensities вычисляются из frozen left state;
2. step ограничивается horizon/observation boundary, $\epsilon/\alpha_{ref}$ и
   frozen hazard cap;
3. выбирается не более одного book event;
4. Brownian endpoint проводится при fixed mid;
5. optional event применяется в right endpoint;
6. observation записывается после полного end state.

Streams `brownian_increment`, `book_occurrence`, `book_channel` и dormant
`brownian_bridge` получены стабильным `SeedSequence.spawn`. Strategy monitoring в P3
configs выключен, поэтому crossing/wealth invariants имеют explicit
`not_applicable_strategy_monitoring_disabled`, а не synthetic zero.

Seed-level stationary estimators, finite-$h$ slopes и ACF вычислены на equal-spaced
samples. Two parity slopes и six ACF coordinates используют centered studentized
10,000-replicate max-$t$ seed-cluster bootstrap. Resolution shifts сравниваются как
independent estimates с conservative difference SE, без paired-coupling claim.

## `SIM-MOMENTS-001`

- Run ID: `20260811T172240473272Z-060cfab011c3-det`
- RunSpec SHA-256: `060cfab011c37d36fcc93bfdbec661b8284acaba8a24c3353148a1e608889ffa`
- Status: `acceptance_failed`
- Runtime: 332.905 s
- Recorded book events: 376,287
- Numerical steps excluding replays: 50,777,759; primary replay adds 14,762,116

### Primary results, $\epsilon=0.01$

| Gate | Result | Limit/target | Status |
|---|---:|---:|---|
| All invariant/replay counts | 0 | 0 | pass |
| Maximum generator residual | $8.88\times10^{-16}$ | $\le10^{-12}$ | pass |
| $|E[G]/s_G|$ | 0.006287 | $\le0.02$ | pass |
| Variance-identity relative error | 0.003724 | $\le0.03$ | pass |
| Open-flow relative error | **0.033379** | $\le0.03$ | **fail** |
| Maximum finite-$h$ slope relative error | 0.043062 | $\le0.05$ | pass |
| Drift targets inside simultaneous bands | yes | required | pass |
| Six ACF targets inside simultaneous bands | yes | required | pass |
| All primary--fine refinement coordinates | yes | required | pass |

Primary stationary variance is `1.6007629`, $s_G=1.2649110$, jump variance rate
`2.9633938`, and open occupancy `0.0035535`. Mean normalized gap is `0.0062869`.
Finite-$h$ target is `-0.9950166`; estimates are `-1.0022972` tight and
`-1.0378642` open. ACF means at lags 0.25, 0.5, 1, 2, 3 and 5 are respectively
`0.777496`, `0.605373`, `0.367826`, `0.139472`, `0.054951`, `0.007425`; every
theoretical $e^{-h}$ lies inside its simultaneous band.

Flow signed residual mean is `-0.0333788`, SE `0.0707411`, two-sided t interval
`[-0.181442, 0.114684]`. Fine-resolution error is `0.0114669`; primary--fine shift
`0.0219118` passes because conservative difference SE is `0.0831040`. Thus the exact
central-error gate fails, but evidence is too imprecise to call the identity contradicted.

## `SIM-UNBALANCED-001`

- Run ID: `20260811T172916459381Z-5497c39af2dd-det`
- RunSpec SHA-256: `5497c39af2dd7595d136500130bfe2445ee549adeff4ea9228035f0878ffeeb5`
- Status: `acceptance_failed`
- Runtime: 328.729 s
- Recorded book events: 376,548
- Numerical steps excluding replays: 50,802,562; primary replay adds 14,795,089

Generator slopes are exactly `-1.0` tight and `-1.25` open within floating error;
maximum pointwise residual is $8.88\times10^{-16}$. The one-factor mechanism is
therefore present in the implemented generator.

At primary resolution realized finite-$h$ slopes are `-0.9867732` tight and
`-1.2441102` open. Their directional contrast is positive (`0.2573371`), but SE is
`0.1349433` and one-sided 95% lower bound `0.0240021` misses required `0.05`.
At fine resolution contrast falls to `0.0678002`; absolute shift `0.1895369` exceeds
both fixed tolerance `0.02` and conservative difference SE `0.1685741`. Поэтому
directional statistical control и refinement gates fail.

## Provenance и artifacts

Оба runs выполнены из clean target commit на CPU
`12th Gen Intel(R) Core(TM) i9-12900H`, Linux 6.8 x86-64, CPython 3.14.0,
NumPy 2.5.2, SciPy 1.18.0, Matplotlib 3.11.1, `float64`; GPU не использовался.
`uv.lock` SHA-256:
`e5a470b4ad10bf736118f2b8e18d49ac58381a84fcb398d981e69cc66f1ac802`.
Manifest warnings/deviations пусты, source config hashes совпадают.

Local immutable artifacts:

- balanced: `outputs/SIM-MOMENTS-001/20260811T172240473272Z-060cfab011c3-det/`;
- control: `outputs/SIM-UNBALANCED-001/20260811T172916459381Z-5497c39af2dd-det/`;
- raw seed metrics: `metrics/seed_metrics.csv`;
- resolution tables: `tables/resolution_summary.csv`;
- balanced binned diagnostic: `tables/binned_drift.csv`;
- machine-readable plot source: `figures/simulation-data.csv`;
- full book-event logs: `records/book_events.csv`, about 56 MiB per run;
- rendered figure hashes:
  - balanced `3588eae4172bf55008893e70884e7fe1b18019e3763e0822e5ffdc974629eefb`;
  - control `b675e1a2e734ada01be86e9bf73ec7e5a5b971c66cea656073a01ef7755a5d7a`.

Generated outputs остаются ignored и не коммитятся. Каждый manifest содержит hashes
всех artifacts; failed evidence сохранено, а не заменено rerun.

## Verification

До target runs прошли 39 tests, включая:

- exact six-channel intensities и balanced/unbalanced generator slopes;
- legal moves, parity lock, inactive channels и canonical derived state;
- adaptive event-probability cap и exact observation schedule;
- fixed-seed bitwise replay и independence named streams;
- shortened 20-seed/three-resolution pipelines обоих experiments;
- strict unknown/missing/non-finite RunSpec validation;
- raw/table/figure/event-log artifact creation.

После closeout повторно выполняются:

```bash
uv sync --locked
uv run python main.py
uv run ot-micromr
uv run python -m unittest discover -s tests -t . -v
uv run ot-micromr validate-config cfg/experiments/sim_moments_001.toml
uv run ot-micromr validate-config cfg/experiments/sim_unbalanced_001.toml
git diff --check
```

## Limitations и threats to validity

- Scheme freezes left intensities, places a selected event at the right endpoint and
  omits second within-step events. Successful refinement is empirical error evidence,
  not a formal convergence proof or exact sampler claim.
- Open occupancy near 0.3% gives only 409--1,139 open observations per seed at primary
  resolution, with strong serial dependence. This is sufficient for the hard minimum
  200 but weak for precise seed-level ratios/slopes.
- Same seed labels across resolutions are not coupled paths; comparison uses independent
  SE exactly as preregistered.
- Strategy monitoring is disabled. Brownian band localization, fills, overshoot and
  wealth identity remain P4 implementation work and receive no artificial P3 pass.
- Project-chosen primitives are not recovered author settings. Results test theorem
  behavior on one registered instance only.
- No market data, efficient-price filter, fees, latency or executable orders exist;
  P3 provides no evidence of real trading profitability.

## Claim status и next gate

| Claim/control | P3 status |
|---|---|
| `PAPER-FACT1-PARITY-LOCK` | numerically reproduced; zero violations |
| `PAPER-2.12-BALANCED-DRIFT` | consistent/reproduced within controlled approximation |
| `PAPER-2.14-STATIONARY-MOMENTS` | mean/variance/ACF gates passed |
| `PAPER-2.9-OPEN-FLOW-BALANCE` | inconclusive; primary central-error gate failed |
| `CONTROL-UNBALANCED-PARITY-SPLIT` | generator confirmed, realized statistical control inconclusive |

P4 не начинается. Продолжение требует нового experiment ID и dated precision
amendment, сохраняющего original artifacts, seeds prefix, primitives, estimators,
resolutions и thresholds.
