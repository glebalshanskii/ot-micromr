# ADR-0014: P6 causal efficient-price filtering contract

- **Статус:** Accepted
- **Дата:** 2026-08-12
- **Связанный этап:** P6
- **Зависимость:** P5 `EMP-DATA-001` passed

## Контекст

Статья предполагает наблюдаемый efficient price $X_t$, тогда как в реальном backtest
наблюдаемы только quotes/trades. Подстановка будущего mid, retrospective smoother или
cross-venue price превратила бы practical signal в oracle. P6 должен отделить проверяемую
идентификацию latent state от последующей настройки торговой стратегии.

## Решение

### Synthetic leg

1. `FILTER-SYN-001` имеет mode `oracle-diagnostic`: истинный $X_t$ используется только для
   оценки ошибки, orders/P&L запрещены.
2. Causal particle filter использует six-event likelihood статьи. Для каждого interval
   likelihood включает и event channel, и survival/no-event term; efficient-price particles
   распространяются Brownian transition.
3. Feasible controls: current displayed mid как frozen naive baseline и causal Gaussian
   two-state Kalman filter. Oracle state является только upper bound.
4. Primary paired independent unit — synthetic session. Required claims: practically useful
   relative RMSE reduction и predictive log-score improvement particle filter против naive.
   Calibration 90% interval является отдельной equivalence family.

### Empirical leg

1. `EMP-FILTER-001` читает только frozen P5 train assets. Validation/test payloads запрещены.
2. Swap BBO transitions классифицируются в шесть paper channels только при допустимом
   spread/parity/jump. Иной transition разрывает model segment и причинно reinitializes
   filter; невозможно корректная книга остаётся под P5 full-snapshot quarantine.
3. OKX spot используется как causal as-of reference: для timestamp $t$ разрешён только
   последний spot observation с exchange timestamp $\le t$. Он не объявляется ground truth.
4. Point-process parameters $(\mu_s,\mu_o,\mu_c,\alpha_s,\alpha_o,\alpha_c)$ оцениваются
   vectorized constrained MLE. Balanced model задаёт $2\alpha_s+\alpha_o=\alpha_c$;
   unbalanced alternative свободен. Reduced $(\alpha,s_G)$ и $\sigma_X$ оцениваются только
   на past fit window.
5. Frozen internal train chronology: 2024-01-15 — fit, 2024-07-15 — model selection,
   2024-12-15 — untouched P6 audit. Остальные четыре swap days используются только для
   descriptive stability diagnostics, не для выбора filter.
6. При отсутствии practically meaningful out-of-sample likelihood advantage выбирается
   balanced model по parsimony. Выбор не использует trading P&L.
7. Retrospective smoothing не реализуется в active path. Все persisted signals являются
   filtered, а timestamp audit обязан иметь ноль future accesses.

## Compute

Весь новый numerical code использует PyTorch. Synthetic sessions, particles, likelihoods,
MLE reductions и as-of alignment векторизуются; primary target использует CUDA `float32`,
final statistics — PyTorch `float64`. Stateful recurrences компилируются через
`torch.compile` блоками фиксированной длины. Архивный JSON/state loop остаётся минимальной
CPU boundary и parallelized по независимым days. Отдельный speed benchmark не запускается.

## Последствия

- Synthetic support является обязательной зависимостью empirical feasibility, но oracle
  outputs никогда не становятся input стратегии.
- P6 может завершиться `negative` или `inconclusive`; это блокирует P7/P8 P&L search, но не
  меняет P5 или synthetic paper reproduction.
- P6 не выбирает trading threshold, fees, latency или execution variant.
