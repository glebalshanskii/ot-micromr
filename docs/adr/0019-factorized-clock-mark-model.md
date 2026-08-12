# ADR-0019: factorize total BBO activity and conditional mark

- **Status:** accepted
- **Date:** 2026-08-12
- **Stage:** P6D

## Context

P6C uses one family of gap ramps both to prefer corrective marks and to increase their
absolute intensities. Consequently a useful directional effect necessarily increases the
sum of intensities. On the OKX audit sample the model predicts ten BBO events about 3.5
times too quickly, while finer continuous-hazard integration has a negligible effect.

A gap-independent Poisson rate by spread was considered and rejected as the P6D primary
clock: a causal design calculation on the existing development sample produced pooled
time-rescaling block mean/SD about `2.36/5.57`. A separate renewal clock is needed rather
than merely setting the P6C correction to zero. A small design scan over causal rolling
log-duration histories selected 200 previous events with a 50-event training prior; this
selection is disclosed and is not counted as target-run evidence.

## Decision

1. Factorize every marked intensity exactly as
   $\lambda_m(t)=\Lambda_\psi(t)p_\theta(m\mid\mathcal F_{t^-})$.
2. Model the event clock as a lognormal renewal law. At every prediction origin its
   location and scale use only the previous 200 valid durations plus 50 pseudo-observations
   from the rolling-origin training history. The current duration is excluded.
3. Model conditional marks with a smoothed gap-independent table and one nonnegative
   normalized directional tilt. The tilt changes $p_m$ but cannot change its unit sum or
   the total clock.
4. The particle filter propagates the Brownian endpoint exactly over each irregular
   interval. Clock likelihood is common to all particles; only conditional mark likelihood
   updates the latent state.
5. Use vectorized PyTorch CUDA `float32` and `torch.compile` for fitted/filtering kernels;
   use PyTorch `float64` reductions for inference. Plot rendering may transfer finalized
   tensors to Matplotlib but may not recompute scientific metrics with NumPy.
6. Keep strategy, orders and P&L disabled. P7 requires simultaneous clock calibration,
   positive conditional-mark ablation and posterior uncertainty below the option margin.

## Consequences and limitations

P6D is an empirical extension, not the paper's exact intensity family. The factorization is
an exact identity for marked point processes, but the chosen renewal and directional-tilt
parameterizations are project choices. Reusing the fixed audit dates is allowed for model
development and rolling-origin comparison; it does not create a new untouched market
period. All actual/predictive distribution data and figures are required artifacts so that
mean metrics cannot hide distributional failure.
