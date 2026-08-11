# ADR-0008: P4 crossing, execution and inference semantics

- Status: accepted; target completed with operational failure and inconclusive science
- Date: 2026-08-11
- Scope: `SIM-FIG4-PILOT-001` and the future `SIM-FIG4-002`

## Context

Figure 4 compares the exact jump model with the Gaussian-surrogate optimum, but the paper
does not publish primitives, seeds, horizon, simulator or raw Monte Carlo outputs. The legacy
`SIM-FIG4-001` contract also predates the project's statistical-gate policy and contains an
invalid `tolerance OR SE` refinement rule. It is superseded without execution.

The market simulator advances the Brownian efficient price between adaptive endpoints and
applies at most one frozen-left book event at the right endpoint. Endpoint-only monitoring
misses Brownian crossings and is not adequate for Figure 4.

## Decision

1. Label every result `independent_partial_reproduction`; author parameters are not inferred
   from plotted points.
2. Use a balanced project family with fixed baselines and sweep only the ramp slopes. Estimate
   one pooled stationary `s_G` per response row from calibration paths that are independent of
   strategy seeds, then freeze `gamma`, `theta_D`, `theta_star` and physical thresholds before
   strategy P&L.
3. Within an adaptive step, process diffusion before the right-end book event. For an active
   one-sided boundary, use the exact Brownian-bridge crossing probability

   $$
   p_{hit}=\exp\left[-\frac{2(b-G_0)(b-G_1)}{\sigma_X^2h}\right]
   $$

   when both endpoints lie on the non-hit side; endpoint straddles are deterministic. A
   separate bridge stream supplies uniforms. A bridge-only hit is recorded at the step
   midpoint and at gap exactly equal to the boundary. The time error is at most `h/2`; the
   rate denominator error vanishes with horizon and is explicitly refined in epsilon.
4. Flat-entry bridge ambiguity is confined to strategy burn-in and excluded from measured
   rewards. Every policy must hold `+1` or `-1` before measurement. After entry, at most one
   diffusive flip and one post-event flip are processed per step. The accumulated upper bound
   for an omitted full-band recrossing must stay below its operational budget.
5. Execution is post-state at the displayed touch. A completed flip contributes
   `2 * (abs(G_fill) - S_fill/2)` to the primary renewal numerator. The first measured fill is
   only the left renewal boundary. Overshoot and frozen-tight-cost rates are diagnostics.
6. The discrete-grid peak is the primary nonparametric peak. A local quadratic fitted peak is
   diagnostic because the paper plots a fitted marker but does not disclose the fit.
7. Complete seed rate vectors are the independent clusters. Peak and loss uncertainty uses
   cluster bootstrap of the full threshold vector. Directional claims test a nonzero minimum
   inward shift; equality/refinement claims use equivalence and multiplicity correction under
   `statistical-gates-v1`.
8. Pilot seeds cannot enter the target run. Pilot output may determine target horizon and seed
   count only through a dated amendment written before any target seed is evaluated. No
   optional extension is allowed after target inspection.
9. Adaptive market endpoints remain CPU `float64`. The accepted 2026-08-11 compute amendment
   moves the full time-by-seed-by-threshold bridge crossing, alternating fill state and reward
   reduction to compiled CUDA `float32`; bootstrap may independently select CPU or CUDA by
   measured wall-clock. CPU `float64` remains the semantic oracle. Exact NumPy/Torch RNG
   identity is not required, but both stream mappings and backend regression are recorded.

## Target amendment

The completed pilot freezes 30 target strategy seeds, horizon 300, a 0.05 multiplier grid,
and a primary minimum inward shift of 0.05. Pilot SD 0.234 implies 29 seeds for 90% power at
the paper-level 0.20 planning alternative under conservative three-test alpha allocation;
30 are used. The former 100-interfill floor is replaced by a 20-interval pathwise denominator
check plus powered seed-cluster inference. This is not a relaxation after target inspection:
the amendment precedes every target seed.

## Consequences

The strategy evaluates the exact paper execution reward on the project's approximate jump
path, including diffusion crossings and jump overshoot. It does not claim an exact continuous
first-hit timestamp or recover the author's hidden simulator. Resolution refinement, omitted
recrossing bounds and CPU/CUDA regression make those approximations visible rather than
silently treating endpoint detection as exact.

## Target outcome

`SIM-FIG4-002/20260811T202753134457Z-837035232ead-det` completed in 34.244 seconds.
All deterministic and numerical-budget gates passed except the preregistered all-cell
minimum-interval floor: 38 far-right cells fell below 20, with minimum 12. Two inward-shift
tests were supported, the high-gamma test was inconclusive, and the refinement family was
inconclusive. The target is not rerun or extended; the canonical interpretation is recorded
in [`paper-reproduction.md`](../reports/paper-reproduction.md).
