from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BandProxyResult:
    threshold_price: float
    fill_count: int
    completed_flip_count: int
    reward_rate_per_second: float
    open_fill_share: float


def evaluate_discrete_band_proxy(
    gaps: np.ndarray,
    tight: np.ndarray,
    observation_interval_seconds: float,
    thresholds_price: np.ndarray,
) -> tuple[BandProxyResult, ...]:
    """Evaluate a diagnostic band at observation endpoints.

    This intentionally is not the continuous-crossing Figure 4 monitor. Array comparisons,
    hit extraction and reward calculation are vectorised; only the alternating renewal
    traversal remains sequential.
    """

    gap_values = np.asarray(gaps, dtype=np.float64)
    tight_values = np.asarray(tight, dtype=np.bool_)
    thresholds = np.asarray(thresholds_price, dtype=np.float64)
    if gap_values.ndim != 1 or tight_values.shape != gap_values.shape or gap_values.size < 2:
        raise ValueError("gaps and tight must be equal one-dimensional arrays")
    if thresholds.ndim != 1 or thresholds.size == 0 or np.any(thresholds <= 0.0):
        raise ValueError("thresholds must be a non-empty positive vector")
    if observation_interval_seconds <= 0.0:
        raise ValueError("observation interval must be positive")

    horizon = (gap_values.size - 1) * observation_interval_seconds
    results: list[BandProxyResult] = []
    for threshold in thresholds:
        lower_hits = np.flatnonzero(gap_values <= -threshold)
        upper_hits = np.flatnonzero(gap_values >= threshold)
        if lower_hits.size == 0 and upper_hits.size == 0:
            results.append(BandProxyResult(float(threshold), 0, 0, 0.0, 0.0))
            continue
        first_lower = int(lower_hits[0]) if lower_hits.size else gap_values.size
        first_upper = int(upper_hits[0]) if upper_hits.size else gap_values.size
        position = 1 if first_lower < first_upper else -1
        current_index = min(first_lower, first_upper)
        fill_indices = [current_index]
        while True:
            candidates = upper_hits if position == 1 else lower_hits
            location = int(np.searchsorted(candidates, current_index, side="right"))
            if location >= candidates.size:
                break
            current_index = int(candidates[location])
            fill_indices.append(current_index)
            position = -position
        indices = np.asarray(fill_indices[1:], dtype=np.int64)
        if indices.size == 0:
            reward_rate = 0.0
            open_share = 0.0
        else:
            half_spreads = np.where(tight_values[indices], 0.5, 1.0)
            rewards = 2.0 * (np.abs(gap_values[indices]) - half_spreads)
            reward_rate = float(np.sum(rewards, dtype=np.float64) / horizon)
            open_share = float(np.mean(~tight_values[indices]))
        results.append(
            BandProxyResult(
                threshold_price=float(threshold),
                fill_count=len(fill_indices),
                completed_flip_count=int(indices.size),
                reward_rate_per_second=reward_rate,
                open_fill_share=open_share,
            )
        )
    return tuple(results)
