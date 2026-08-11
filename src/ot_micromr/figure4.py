from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from ot_micromr.analytics import kramers_threshold_ratio, solve_dawson_optimum
from ot_micromr.jump_model import (
    EVENT_CHANNELS,
    BookParameters,
    apply_book_event,
    initial_state,
    intensities,
)


@dataclass(frozen=True, slots=True)
class CalibrationPath:
    row_index: int
    alpha_per_second: float
    seed: int
    gaps: np.ndarray
    open_occupancy: float
    step_count: int
    book_event_count: int


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    row_index: int
    alpha_per_second: float
    s_g_price: float
    gamma_ratio: float
    u_d_ratio: float
    u_star_ratio: float
    theta_d_price: float
    theta_star_price: float
    surrogate_optimum_rate_per_second: float
    root_abs_residual: float
    open_occupancy: float
    observation_count: int


@dataclass(frozen=True, slots=True)
class Figure4Replication:
    row_index: int
    alpha_per_second: float
    epsilon: float
    seed: int
    policy_rows: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]
    replay_digest: str


def _row_model(values: Mapping[str, Any], alpha: float) -> dict[str, Any]:
    source = values["model"]
    initial = source["initial_state"]
    return {
        "delta_price": float(source["delta_price"]),
        "sigma_x_price_per_sqrt_second": float(source["sigma_x_price_per_sqrt_second"]),
        "mu_s_per_second": float(source["mu_s_per_second"]),
        "mu_o_per_second": float(source["mu_o_per_second"]),
        "mu_c_per_second": float(source["mu_c_per_second"]),
        "alpha_s_per_second": alpha * float(source["alpha_s_fraction_of_alpha"]),
        "alpha_o_per_second": alpha * float(source["alpha_o_fraction_of_alpha"]),
        "alpha_c_per_second": alpha * float(source["alpha_c_fraction_of_alpha"]),
        "alpha_per_second": alpha,
        "initial_state": {
            "time_seconds": float(initial["time_seconds"]),
            "mid_half_ticks": int(initial["mid_half_ticks"]),
            "efficient_price": float(initial["efficient_price"]),
        },
    }


def _domain_streams(
    seed: int, row_index: int, epsilon: float, domain: int
) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator, np.random.Generator]:
    epsilon_code = int(round(epsilon * 1_000_000))
    sequence = np.random.SeedSequence([seed, row_index, epsilon_code, domain])
    children = sequence.spawn(4)
    return tuple(
        np.random.Generator(np.random.PCG64DXSM(child)) for child in children
    )  # type: ignore[return-value]


def _select_channel(rates: Sequence[float], total: float, uniform: float) -> str:
    target = uniform * total
    cumulative = 0.0
    active = EVENT_CHANNELS[0]
    for channel, rate in zip(EVENT_CHANNELS, rates, strict=True):
        if rate > 0.0:
            active = channel
        cumulative += rate
        if target < cumulative:
            return channel
    return active


def _simulate_calibration_coordinate(
    values: Mapping[str, Any], coordinate: tuple[int, float, int]
) -> CalibrationPath:
    row_index, alpha, seed = coordinate
    model = _row_model(values, alpha)
    parameters = BookParameters.from_model(model)
    state = initial_state(model)
    simulation = values["simulation"]
    epsilon = float(values["numerics"]["primary_resolution_epsilon"])
    burn = float(simulation["calibration_burn_in_reversion_times"]) / alpha
    measured = float(simulation["calibration_sampling_reversion_times"]) / alpha
    requested_interval = (
        float(simulation["calibration_observation_interval_reversion_times"]) / alpha
    )
    interval_count = round(measured / requested_interval)
    endpoint_rng, occurrence_rng, channel_rng, _ = _domain_streams(
        seed, row_index, epsilon, 0
    )
    total_end = burn + measured
    next_observation = burn
    observation_index = 0
    gaps = np.empty(interval_count + 1, dtype=np.float64)
    tight = np.empty(interval_count + 1, dtype=np.bool_)
    max_step = epsilon / alpha
    hazard_numerator = -math.log1p(-epsilon)
    tolerance = 5e-12
    step_count = 0
    event_count = 0
    while state.time_seconds < total_end - tolerance:
        left_time = state.time_seconds
        rates = intensities(state, parameters)
        total_rate = sum(rates)
        step = min(
            total_end - left_time,
            next_observation - left_time,
            max_step,
            hazard_numerator / total_rate,
        )
        if step <= 0.0 or not math.isfinite(step):
            raise RuntimeError("calibration produced a nonpositive step")
        event_occurs = occurrence_rng.random() < -math.expm1(-total_rate * step)
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
        state.time_seconds = left_time + step
        if abs(state.time_seconds - next_observation) <= tolerance:
            state.time_seconds = next_observation
        if abs(state.time_seconds - total_end) <= tolerance:
            state.time_seconds = total_end
        if channel is not None:
            apply_book_event(state, channel, parameters)
            event_count += 1
        step_count += 1
        if state.time_seconds == next_observation:
            gaps[observation_index] = state.gap_price(parameters)
            tight[observation_index] = state.is_tight
            observation_index += 1
            next_observation = (
                total_end
                if observation_index >= interval_count
                else burn + measured * observation_index / interval_count
            )
    if observation_index != interval_count + 1:
        raise RuntimeError(
            "calibration observation schedule is incomplete: "
            f"observed={observation_index}, expected={interval_count + 1}, "
            f"time={state.time_seconds:.17g}, next={next_observation:.17g}, "
            f"end={total_end:.17g}"
        )
    return CalibrationPath(
        row_index=row_index,
        alpha_per_second=alpha,
        seed=seed,
        gaps=gaps,
        open_occupancy=float(np.mean(~tight)),
        step_count=step_count,
        book_event_count=event_count,
    )


def _calibration_worker(
    payload: tuple[Mapping[str, Any], tuple[int, float, int]]
) -> CalibrationPath:
    values, coordinate = payload
    with threadpool_limits(limits=1):
        return _simulate_calibration_coordinate(values, coordinate)


def calibrate_rows(values: Mapping[str, Any]) -> tuple[CalibrationRow, ...]:
    alpha_grid = tuple(
        float(value) for value in values["model"]["response_scale_alpha_per_second_grid"]
    )
    seeds = tuple(int(value) for value in values["seed_policy"]["calibration_seeds"])
    coordinates = [
        (row_index, alpha, seed)
        for row_index, alpha in enumerate(alpha_grid)
        for seed in seeds
    ]
    workers = min(int(values["numerics"]["cpu_workers"]), len(coordinates))
    if workers == 1:
        paths = [_simulate_calibration_coordinate(values, item) for item in coordinates]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            paths = list(
                executor.map(_calibration_worker, ((values, item) for item in coordinates))
            )
    rows: list[CalibrationRow] = []
    for row_index, alpha in enumerate(alpha_grid):
        selected = [path for path in paths if path.row_index == row_index]
        pooled = np.concatenate([path.gaps for path in selected])
        s_g = float(np.std(pooled, ddof=0))
        gamma = float(values["execution"]["threshold_reference_phi_price"]) / s_g
        solution = solve_dawson_optimum(gamma, values["numerics"])
        u_star = kramers_threshold_ratio(gamma)
        theta_d = s_g * solution.u_d_ratio
        rows.append(
            CalibrationRow(
                row_index=row_index,
                alpha_per_second=alpha,
                s_g_price=s_g,
                gamma_ratio=gamma,
                u_d_ratio=solution.u_d_ratio,
                u_star_ratio=u_star,
                theta_d_price=theta_d,
                theta_star_price=s_g * u_star,
                surrogate_optimum_rate_per_second=alpha * s_g * solution.normalized_rate,
                root_abs_residual=solution.root_abs_residual,
                open_occupancy=float(np.mean([path.open_occupancy for path in selected])),
                observation_count=int(sum(path.gaps.size for path in selected)),
            )
        )
    return tuple(rows)


def _policy_definition(
    values: Mapping[str, Any], calibration: CalibrationRow
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    multipliers = np.asarray(
        values["strategy"]["threshold_multiplier_theta_over_theta_d_grid"],
        dtype=np.float64,
    )
    thresholds = multipliers * calibration.theta_d_price
    labels = tuple(f"grid:{value:.12g}" for value in multipliers) + ("theta_star",)
    return (
        np.concatenate((thresholds, np.asarray([calibration.theta_star_price]))),
        labels,
        np.concatenate(
            (
                multipliers,
                np.asarray([calibration.theta_star_price / calibration.theta_d_price]),
            )
        ),
    )
