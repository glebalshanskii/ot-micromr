# ADR-0020: factorized clock passes moments but latent-state result is negative

- **Status:** accepted
- **Date:** 2026-08-12
- **Stage:** P6D result

## Evidence

Clean target run `EMP-MARK-FACT-001/20260812T105127206423Z-44416f08cb43-det` used
commit `be0f33d6f014877786005f3437c63c88a8d382c5`, all `1,122,613` held-out healthy
transitions and 288 thirty-minute blocks. Operational gates, required figures/data and
bitwise December replay passed.

Separating the causal renewal clock repaired the preregistered moment calibration:
rescaling mean `1.0692` with interval `[1.0583,1.0801]`, and block SD `1.1380` with
interval `[1.1262,1.1497]`. Both lie inside the simultaneous equivalence regions.

The scientific model as a whole failed. Conditional gap tilt versus the identical zero-tilt
mark table gave `-0.0000834 nat/event`, interval
`[-0.0001668,0.00000004]`, so superiority is not supported. Posterior SD is `8.426` times
the optimistic option-value margin; the usability upper bound remains far below zero.

Required plots also show that passing the two clock moments is not full distributional
equivalence: a continuous lognormal forecast smooths the large actual duration mass near
`0.01 s` (descriptive histogram TV distance `0.483`). The direction aggregate is close
(TV `0.00225`), while magnitude buckets remain less accurate (midpoint TV `0.0792`).

## Decision

1. Mark P6D complete with calibrated clock moments and a negative conditional-mark/state
   result.
2. Keep P7/P8 and every real-data P&L experiment blocked.
3. Accept the structural explanation that P6C conflated event speed with direction, but
   reject the stronger hypothesis that separating them makes the book-only latent gap
   tradable.
4. Do not tune clock history or distribution merely to unlock the strategy. A richer clock
   could improve distributional fit but cannot by itself supply missing information about
   $X$.
5. If empirical research continues, preregister an independent causal state-observation
   mechanism and test it with one-factor ablation against P6D.

## Technical deviations

Two pre-result implementation defects were corrected without changing the model, config or
gates: unavailable CUDA `torch.histogram` was replaced by vectorized
`bucketize+bincount`, and deterministic PyTorch algorithms were enabled after `float64`
CUDA prefix sums differed at about `1e-13` between replays. The failed technical run and the
first operationally valid but replay-failed run remain in ignored local outputs.
