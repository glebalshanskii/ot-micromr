from __future__ import annotations

import hashlib
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from ot_micromr.figure4 import CalibrationRow, _domain_streams, _row_model, _select_channel
from ot_micromr.jump_model import BookParameters, apply_book_event, initial_state, intensities


@dataclass(frozen=True, slots=True)
class Figure4MarketTrace:
    row_index: int
    alpha_per_second: float
    epsilon: float
    seed: int
    left_time_seconds: np.ndarray
    step_seconds: np.ndarray
    left_gap_price: np.ndarray
    pre_event_gap_price: np.ndarray
    post_event_gap_price: np.ndarray
    left_mid_price: np.ndarray
    post_event_mid_price: np.ndarray
    left_spread_price: np.ndarray
    post_event_spread_price: np.ndarray
    book_event: np.ndarray
    terminal_efficient_price: float
    replay_digest: str
    maximum_left_event_probability: float


def simulate_market_trace(
    values: Mapping[str, Any], calibration: CalibrationRow, epsilon: float, seed: int
) -> Figure4MarketTrace:
    alpha = calibration.alpha_per_second
    model = _row_model(values, alpha)
    parameters = BookParameters.from_model(model)
    state = initial_state(model)
    endpoint_rng, occurrence_rng, channel_rng, _ = _domain_streams(
        seed, calibration.row_index, epsilon, 1
    )
    simulation = values["simulation"]
    market_burn_end = float(simulation["market_burn_in_reversion_times"]) / alpha
    measurement_start = market_burn_end + float(
        simulation["strategy_burn_in_reversion_times"]
    ) / alpha
    total_end = measurement_start + float(simulation["horizon_reversion_times"]) / alpha
    phase_boundaries = (market_burn_end, measurement_start, total_end)
    next_phase_index = 0
    next_phase = phase_boundaries[0]
    max_step = epsilon / alpha
    hazard_numerator = -math.log1p(-epsilon)
    tolerance = 5e-12

    left_times: list[float] = []
    steps: list[float] = []
    left_gaps: list[float] = []
    pre_gaps: list[float] = []
    post_gaps: list[float] = []
    left_mids: list[float] = []
    post_mids: list[float] = []
    left_spreads: list[float] = []
    post_spreads: list[float] = []
    book_events: list[bool] = []
    maximum_event_probability = 0.0

    while state.time_seconds < total_end - tolerance:
        left_time = state.time_seconds
        left_gap = state.gap_price(parameters)
        left_mid = state.mid_price(parameters)
        left_spread = state.spread_price(parameters)
        rates = intensities(state, parameters)
        total_rate = sum(rates)
        step = min(
            total_end - left_time,
            next_phase - left_time,
            max_step,
            hazard_numerator / total_rate,
        )
        if step <= 0.0 or not math.isfinite(step):
            raise RuntimeError("market trace produced a nonpositive step")
        event_probability = -math.expm1(-total_rate * step)
        maximum_event_probability = max(maximum_event_probability, event_probability)
        event_occurs = occurrence_rng.random() < event_probability
        channel = (
            _select_channel(rates, total_rate, float(channel_rng.random()))
            if event_occurs
            else None
        )
        state.efficient_price += (
            parameters.sigma_x_price_per_sqrt_second
            * math.sqrt(step)
            * float(endpoint_rng.normal())
        )
        pre_gap = state.gap_price(parameters)
        new_time = left_time + step
        if abs(new_time - next_phase) <= tolerance:
            new_time = next_phase
        if abs(new_time - total_end) <= tolerance:
            new_time = total_end
        state.time_seconds = new_time
        if channel is not None:
            apply_book_event(state, channel, parameters)

        left_times.append(left_time)
        steps.append(step)
        left_gaps.append(left_gap)
        pre_gaps.append(pre_gap)
        post_gaps.append(state.gap_price(parameters))
        left_mids.append(left_mid)
        post_mids.append(state.mid_price(parameters))
        left_spreads.append(left_spread)
        post_spreads.append(state.spread_price(parameters))
        book_events.append(channel is not None)

        if state.time_seconds == next_phase:
            next_phase_index += 1
            next_phase = (
                phase_boundaries[next_phase_index]
                if next_phase_index < len(phase_boundaries)
                else total_end
            )

    arrays = (
        np.asarray(left_times, dtype=np.float64),
        np.asarray(steps, dtype=np.float64),
        np.asarray(left_gaps, dtype=np.float64),
        np.asarray(pre_gaps, dtype=np.float64),
        np.asarray(post_gaps, dtype=np.float64),
        np.asarray(left_mids, dtype=np.float64),
        np.asarray(post_mids, dtype=np.float64),
        np.asarray(left_spreads, dtype=np.float64),
        np.asarray(post_spreads, dtype=np.float64),
        np.asarray(book_events, dtype=np.bool_),
    )
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(array.tobytes(order="C"))
    digest.update(np.asarray([state.efficient_price], dtype=np.float64).tobytes())
    return Figure4MarketTrace(
        row_index=calibration.row_index,
        alpha_per_second=alpha,
        epsilon=epsilon,
        seed=seed,
        left_time_seconds=arrays[0],
        step_seconds=arrays[1],
        left_gap_price=arrays[2],
        pre_event_gap_price=arrays[3],
        post_event_gap_price=arrays[4],
        left_mid_price=arrays[5],
        post_event_mid_price=arrays[6],
        left_spread_price=arrays[7],
        post_event_spread_price=arrays[8],
        book_event=arrays[9],
        terminal_efficient_price=float(state.efficient_price),
        replay_digest=digest.hexdigest(),
        maximum_left_event_probability=maximum_event_probability,
    )


def _market_worker(
    payload: tuple[Mapping[str, Any], CalibrationRow, float, int]
) -> Figure4MarketTrace:
    values, calibration, epsilon, seed = payload
    with threadpool_limits(limits=1):
        return simulate_market_trace(values, calibration, epsilon, seed)


def simulate_market_traces(
    values: Mapping[str, Any],
    calibrations: Sequence[CalibrationRow],
    coordinates: Sequence[tuple[int, float, int]],
    workers: int,
) -> tuple[Figure4MarketTrace, ...]:
    by_row = {row.row_index: row for row in calibrations}
    payloads = [
        (values, by_row[row_index], epsilon, seed)
        for row_index, epsilon, seed in coordinates
    ]
    worker_count = min(workers, len(payloads))
    if worker_count == 1:
        return tuple(simulate_market_trace(*payload) for payload in payloads)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return tuple(executor.map(_market_worker, payloads))
