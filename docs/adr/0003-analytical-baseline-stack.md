# ADR-0003: Analytical baseline stack и граница P2 validator

- **Статус:** Accepted
- **Дата:** 2026-08-11
- **Связанный этап:** P2
- **Связанные runs:** `ANA-SMOKE-001`, `ANA-FIG3-001`
- **Связанный отчёт:** [`p2-analytical-reproduction.md`](../reports/p2-analytical-reproduction.md)

## Контекст

ADR-0001 зафиксировал namespace, CLI, `RunSpec v1` и artifact contract, но не выбирал
реальные numerical packages или границу первого typed validator. P2 должен получить
минимальный исполняемый analytical baseline до реализации stochastic state machine.
Нельзя делать вид, что preregistered `SIM-*` contracts уже полностью типизированы,
если simulator-specific invariants ещё не представлены domain types.

## Решение

1. Analytical formulas реализуются в `ot_micromr.analytics` поверх NumPy/SciPy:
   `scipy.special.dawsn`, `scipy.special.erfi`, `scipy.optimize.brentq` и независимый
   bounded `minimize_scalar` cross-check. Project code сохраняет formulas, validation,
   scaling и acceptance logic; сторонний paper implementation не используется.
2. Exact resolved environment фиксирует `uv.lock`: NumPy 2.5.2, SciPy 1.18.0 и
   Matplotlib 3.11.1 на Python 3.14. Matplotlib используется только для deterministic
   rendering; численные source data Figure 3 сохраняются отдельными CSV.
3. P2 strict validator принимает только `ANA-SMOKE-001` и `ANA-FIG3-001`, проверяя
   полный exact field set, types, finite values, units, mode, source hash и semantic
   constraints. `SIM-*` rejected as not-yet-executable до расширения schema в P3;
   partial/permissive validation запрещена.
4. Analytical computation остаётся pure и отделено от artifact runner. Runner создаёт
   неперезаписываемый run directory, пишет source TOML, canonical JSON, metrics, log и
   final atomic `run-manifest-v1`. Figure строится из тех же rows, которые сохраняются
   в CSV/table.
5. Claim-eligible run прекращается до создания directory, если Git tree dirty.
   Deterministic analytical seed хранится как contract metadata и помечается
   `consumed=false`.

## Последствия

- Analytical reproduction доступно через один CLI и имеет replayable environment.
- Figure/data discrepancy можно проверять без PNG и plotting dependency.
- Simulator configs остаются preregistered, но ошибочно запустить их через неполный P2
  runtime нельзя.
- Matplotlib увеличивает default install, однако непосредственно требуется P2 artifact
  class `figure`; более тяжёлый research stack не добавляется.
- Любое изменение formula/backend/tolerances после runs требует нового config/run, а
  не перезаписи P2 artifacts.

## Проверка решения

Оба target runs выполнены без deviations из clean implementation commit `710efa9`.
`ANA-SMOKE-001` и `ANA-FIG3-001` прошли acceptance; environment и artifact hashes
зафиксированы manifests. Детали и limitations приведены в связанном factual report.
