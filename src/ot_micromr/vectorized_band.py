from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BatchedBandProxyResult:
    reward_rate_per_second: np.ndarray
    fill_count: np.ndarray
    completed_flip_count: np.ndarray
    open_fill_share: np.ndarray


def evaluate_discrete_band_proxy_batched_numpy(
    gaps: np.ndarray,
    tight: np.ndarray,
    observation_interval_seconds: float,
    thresholds_price: np.ndarray,
) -> BatchedBandProxyResult:
    """Evaluate all paths and thresholds without Python loops.

    The state machine is exactly the endpoint-only diagnostic used by
    ``evaluate_discrete_band_proxy``: the first boundary hit opens a position and every
    subsequent hit of the opposite boundary completes one flip.  Zeros between boundary
    hits are forward-filled only for detecting the next opposite-side observation.
    """

    gap_values = np.asarray(gaps)
    tight_values = np.asarray(tight, dtype=np.bool_)
    thresholds = np.asarray(thresholds_price, dtype=gap_values.dtype)
    if gap_values.ndim != 2 or tight_values.shape != gap_values.shape:
        raise ValueError("gaps and tight must be equal two-dimensional arrays")
    if gap_values.shape[1] < 2:
        raise ValueError("each path must contain at least two observations")
    if not np.issubdtype(gap_values.dtype, np.floating):
        raise ValueError("gaps must use a floating dtype")
    if thresholds.ndim != 1 or thresholds.size == 0 or np.any(thresholds <= 0.0):
        raise ValueError("thresholds must be a non-empty positive vector")
    if observation_interval_seconds <= 0.0:
        raise ValueError("observation interval must be positive")

    path_count, observation_count = gap_values.shape
    threshold_count = thresholds.size
    expanded_gaps = gap_values[:, :, None]
    side = np.zeros((path_count, observation_count, threshold_count), dtype=np.int8)
    side[expanded_gaps >= thresholds[None, None, :]] = 1
    side[expanded_gaps <= -thresholds[None, None, :]] = -1
    observed = side != 0

    time_index = np.arange(observation_count, dtype=np.int64)[None, :, None]
    last_index = np.maximum.accumulate(np.where(observed, time_index, -1), axis=1)
    last_side = np.take_along_axis(side, np.maximum(last_index, 0), axis=1)
    last_side[last_index < 0] = 0
    previous_side = np.empty_like(last_side)
    previous_side[:, 0, :] = 0
    previous_side[:, 1:, :] = last_side[:, :-1, :]

    first_fill = observed & (previous_side == 0)
    completed_flip = observed & (previous_side != 0) & (side != previous_side)
    fill_count = np.sum(first_fill | completed_flip, axis=1, dtype=np.int64)
    completed_flip_count = np.sum(completed_flip, axis=1, dtype=np.int64)

    half_spread = np.where(tight_values, 0.5, 1.0)
    reward = 2.0 * (np.abs(gap_values) - half_spread)
    reward_sum = np.sum(
        np.where(completed_flip, reward[:, :, None], 0.0),
        axis=1,
        dtype=np.float64,
    )
    horizon = (observation_count - 1) * observation_interval_seconds
    reward_rate = reward_sum / horizon
    open_fill_count = np.sum(
        completed_flip & (~tight_values[:, :, None]), axis=1, dtype=np.int64
    )
    open_fill_share = np.divide(
        open_fill_count,
        completed_flip_count,
        out=np.zeros_like(reward_rate, dtype=np.float64),
        where=completed_flip_count != 0,
    )
    return BatchedBandProxyResult(
        reward_rate_per_second=reward_rate,
        fill_count=fill_count,
        completed_flip_count=completed_flip_count,
        open_fill_share=open_fill_share,
    )
