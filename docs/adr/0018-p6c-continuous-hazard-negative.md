# ADR-0018: continuous hazard does not repair empirical state usability

- **Status:** accepted
- **Date:** 2026-08-12
- **Stage:** P6C result

## Evidence

`EMP-MARK-CT-001/20260812T100151852237Z-c8a620999b93-det` ran from clean commit
`6b2306e19eff55bba1d90033301a13b39bc5477a`. All operational gates and the nested
4/8-substep numerical equivalence family passed. Scientific acceptance failed because
posterior SD remained `1.620` times the option margin and time-rescaling mean/SD remained
`2.209/5.330`, far above their equivalence regions. Continuous several-event rollout still
generated ten BBO events in `4.23 s` versus actual `14.79 s` and did not beat midpoint
persistence.

## Decision

1. Mark P6C completed with a valid negative scientific result.
2. Reject insufficient within-interval hazard resolution as an explanation of P6M failure.
3. Keep P7/P8 blocked; do not compute orders, fills or P&L from P6C state.
4. Retain the continuous-hazard implementation as the only event-clock path for future
   marked-model extensions; P6M frozen results remain historical comparison evidence.
5. Do not spend additional runs on quadrature refinement under this model. A subsequent
   extension must change an identified model boundary, principally decoupling total event
   activity from directional mark correction or adding a causal state observation.

## Consequences

The event marks retain strong predictive information, but that information is insufficient
for the article's threshold strategy. The next step, if continued, is model redesign rather
than more seeds, a looser gate or a finer time grid.
