# ADR-0008: P4 crossing, execution and inference semantics

- Status: accepted for pilot; target sample size pending preregistered pilot amendment
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
9. Adaptive market simulation and bridge decisions remain CPU `float64`. Vectorised bootstrap
   and policy-curve reductions use compiled CUDA `float32` when an end-to-end benchmark wins;
   CPU `float64` remains the regression oracle. Exact NumPy/Torch RNG identity is not required.

## Consequences

The strategy evaluates the exact paper execution reward on the project's approximate jump
path, including diffusion crossings and jump overshoot. It does not claim an exact continuous
first-hit timestamp or recover the author's hidden simulator. Resolution refinement, omitted
recrossing bounds and CPU/CUDA regression make those approximations visible rather than
silently treating endpoint detection as exact.
