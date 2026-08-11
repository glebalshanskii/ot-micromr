from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ot_micromr.jump_model import (
    EVENT_CHANNELS,
    BookParameters,
    BookState,
    InvariantViolation,
    apply_book_event,
    generator_mid_drift,
    initial_state,
    intensities,
)


STREAM_NAMES = (
    "brownian_increment",
    "book_occurrence",
    "book_channel",
    "brownian_bridge",
)


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    burn_seconds: float
    horizon_seconds: float
    observation_interval_seconds: float
    alpha_ref_per_second: float
    acf_lags_seconds: tuple[float, ...]
    minimum_slope_observations: int


@dataclass(frozen=True, slots=True)
class ReplicationResult:
    epsilon: float
    seed: int
    seed_metrics: Mapping[str, Any]
    gaps: np.ndarray
    tight: np.ndarray
    replay_digest: str
    stream_spawn_keys: Mapping[str, tuple[int, ...]]


def settings_from_spec(values: Mapping[str, Any]) -> SimulationSettings:
    simulation = values["simulation"]
    model = values["model"]
    evaluation = values["evaluation"]
    reference_rate = float(
        simulation.get("reference_reversion_rate_per_second", model.get("alpha_per_second", 1.0))
    )
    burn_seconds = float(simulation["burn_in_reversion_times"]) / reference_rate
    horizon_seconds = float(simulation["horizon_reversion_times"]) / reference_rate
    observation_seconds = float(simulation["observation_interval_reversion_times"]) / reference_rate
    lags = tuple(
        float(lag) / reference_rate
        for lag in evaluation.get("acf_lags_reversion_times", (observation_seconds * reference_rate,))
    )
    return SimulationSettings(
        burn_seconds=burn_seconds,
        horizon_seconds=horizon_seconds,
        observation_interval_seconds=observation_seconds,
        alpha_ref_per_second=float(values["numerics"]["alpha_ref_per_second"]),
        acf_lags_seconds=lags,
        minimum_slope_observations=int(evaluation["minimum_observations_per_seed_and_parity_for_slope"]),
    )


def named_streams(seed: int) -> tuple[dict[str, np.random.Generator], dict[str, tuple[int, ...]]]:
    sequence = np.random.SeedSequence(seed)
    children = sequence.spawn(len(STREAM_NAMES))
    generators = {
        name: np.random.Generator(np.random.PCG64DXSM(child))
        for name, child in zip(STREAM_NAMES, children, strict=True)
    }
    spawn_keys = {
        name: tuple(int(value) for value in child.spawn_key)
        for name, child in zip(STREAM_NAMES, children, strict=True)
    }
    return generators, spawn_keys


def _select_channel(
    values: tuple[float, float, float, float, float, float], total: float, uniform: float
) -> tuple[int, str]:
    target = uniform * total
    cumulative = 0.0
    last_active = -1
    for index, value in enumerate(values):
        if value > 0.0:
            last_active = index
        cumulative += value
        if target < cumulative:
            return index, EVENT_CHANNELS[index]
    if last_active < 0:
        raise InvariantViolation("book channel selected with zero total intensity")
    return last_active, EVENT_CHANNELS[last_active]


def _path_digest(
    state: BookState,
    gaps: np.ndarray,
    tight: np.ndarray,
    step_count: int,
    book_event_count: int,
    channel_hazard_integrals: np.ndarray,
    measured_channel_counts: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(gaps.tobytes(order="C"))
    digest.update(tight.tobytes(order="C"))
    digest.update(
        struct.pack(
            "<ddiQQ",
            state.time_seconds,
            state.efficient_price,
            state.mid_half_ticks,
            step_count,
            book_event_count,
        )
    )
    digest.update(channel_hazard_integrals.tobytes(order="C"))
    digest.update(measured_channel_counts.tobytes(order="C"))
    return digest.hexdigest()


def simulate_replication(
    values: Mapping[str, Any],
    epsilon: float,
    seed: int,
    *,
    settings: SimulationSettings | None = None,
) -> ReplicationResult:
    parameters = BookParameters.from_model(values["model"])
    simulation_settings = settings or settings_from_spec(values)
    if epsilon <= 0.0 or epsilon >= 1.0:
        raise ValueError("epsilon must lie in (0, 1)")
    if simulation_settings.horizon_seconds <= 0.0 or simulation_settings.observation_interval_seconds <= 0.0:
        raise ValueError("horizon and observation interval must be positive")
    interval_count_float = (
        simulation_settings.horizon_seconds / simulation_settings.observation_interval_seconds
    )
    interval_count = round(interval_count_float)
    if not math.isclose(interval_count_float, interval_count, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("horizon must be an integer multiple of observation interval")

    streams, spawn_keys = named_streams(seed)
    brownian_rng = streams["brownian_increment"]
    occurrence_rng = streams["book_occurrence"]
    channel_rng = streams["book_channel"]
    state = initial_state(values["model"])
    total_end = simulation_settings.burn_seconds + simulation_settings.horizon_seconds
    gaps = np.empty(interval_count + 1, dtype=np.float64)
    tight = np.empty(interval_count + 1, dtype=np.bool_)
    observation_index = 0
    if simulation_settings.burn_seconds == 0.0:
        gaps[0] = state.gap_price(parameters)
        tight[0] = state.is_tight
        observation_index = 1
    next_boundary = (
        simulation_settings.burn_seconds
        if observation_index == 0
        else simulation_settings.burn_seconds
        + observation_index * simulation_settings.observation_interval_seconds
    )

    max_step = epsilon / simulation_settings.alpha_ref_per_second
    hazard_numerator = -math.log1p(-epsilon)
    step_count = 0
    book_event_count = 0
    channel_hazard_integrals = np.zeros(len(EVENT_CHANNELS), dtype=np.float64)
    measured_channel_counts = np.zeros(len(EVENT_CHANNELS), dtype=np.int64)
    maximum_left_event_probability = 0.0
    generator_max_residual = {True: 0.0, False: 0.0}
    generator_numerator = {True: 0.0, False: 0.0}
    generator_denominator = {True: 0.0, False: 0.0}
    realised_jump_drift_numerator = {True: 0.0, False: 0.0}
    realised_jump_drift_denominator = {True: 0.0, False: 0.0}
    tolerance = 5e-12

    while state.time_seconds < total_end - tolerance:
        left_time = state.time_seconds
        left_mid_ticks = state.mid_half_ticks
        left_tight = state.is_tight
        left_gap = state.gap_price(parameters)
        left_rates = intensities(state, parameters)
        total_rate = sum(left_rates)
        if not math.isfinite(total_rate) or total_rate <= 0.0:
            raise InvariantViolation("total book intensity must be positive and finite")
        boundary_remaining = next_boundary - left_time
        end_remaining = total_end - left_time
        hazard_step = hazard_numerator / total_rate
        step = min(end_remaining, boundary_remaining, max_step, hazard_step)
        if not math.isfinite(step) or step <= 0.0:
            raise InvariantViolation(
                f"nonpositive numerical step: t={left_time}, boundary={next_boundary}, h={step}"
            )
        event_probability = -math.expm1(-total_rate * step)
        if event_probability > epsilon + 2e-15:
            raise InvariantViolation("frozen-left event probability exceeded epsilon")
        maximum_left_event_probability = max(maximum_left_event_probability, event_probability)
        event_occurs = bool(occurrence_rng.random() < event_probability)
        selected_index = -1
        selected_channel: str | None = None
        if event_occurs:
            selected_index, selected_channel = _select_channel(
                left_rates, total_rate, float(channel_rng.random())
            )

        measured_step = left_time >= simulation_settings.burn_seconds - tolerance
        if measured_step:
            channel_hazard_integrals += np.asarray(left_rates, dtype=np.float64) * step

        state.efficient_price += (
            parameters.sigma_x_price_per_sqrt_second * math.sqrt(step) * float(brownian_rng.normal())
        )
        new_time = left_time + step
        if abs(new_time - next_boundary) <= tolerance:
            new_time = next_boundary
        elif abs(new_time - total_end) <= tolerance:
            new_time = total_end
        if new_time <= left_time:
            raise InvariantViolation("numerical time failed to increase")
        state.time_seconds = new_time

        delta_mid = 0.0
        if selected_channel is not None:
            pre_event_x = state.efficient_price
            delta_mid = apply_book_event(state, selected_channel, parameters)
            if state.efficient_price != pre_event_x:
                raise InvariantViolation("X is discontinuous at book event")
            measured = state.time_seconds > simulation_settings.burn_seconds + tolerance
            book_event_count += 1
            if measured:
                measured_channel_counts[selected_index] += 1
                realised_jump_drift_numerator[left_tight] += left_gap * delta_mid

        step_count += 1
        drift = generator_mid_drift(state, parameters, left_rates)
        coefficient = (
            parameters.tight_drift_coefficient_per_second
            if left_tight
            else parameters.open_drift_coefficient_per_second
        )
        residual = abs(drift + coefficient * left_gap)
        generator_max_residual[left_tight] = max(generator_max_residual[left_tight], residual)
        if measured_step:
            generator_numerator[left_tight] += left_gap * drift
            generator_denominator[left_tight] += left_gap * left_gap
            realised_jump_drift_denominator[left_tight] += left_gap * left_gap * step

        if state.mid_half_ticks % 2 == 0 and state.spread_price(parameters) != 2.0 * parameters.delta_price:
            raise InvariantViolation("open parity/spread mismatch")
        if state.mid_half_ticks % 2 != 0 and state.spread_price(parameters) != parameters.delta_price:
            raise InvariantViolation("tight parity/spread mismatch")
        if not all(
            math.isfinite(value)
            for value in (
                state.time_seconds,
                state.efficient_price,
                state.mid_price(parameters),
                state.gap_price(parameters),
            )
        ):
            raise InvariantViolation("nonfinite state")
        if left_mid_ticks != state.mid_half_ticks and selected_channel is None:
            raise InvariantViolation("mid changed without a book event")
        if left_mid_ticks == state.mid_half_ticks and selected_channel is not None:
            raise InvariantViolation("book event did not move the mid")

        if state.time_seconds == next_boundary:
            if state.time_seconds >= simulation_settings.burn_seconds - tolerance:
                if observation_index >= gaps.size:
                    raise InvariantViolation("observation schedule overflow")
                gaps[observation_index] = state.gap_price(parameters)
                tight[observation_index] = state.is_tight
                observation_index += 1
            if observation_index <= interval_count:
                next_boundary = (
                    simulation_settings.burn_seconds
                    + observation_index * simulation_settings.observation_interval_seconds
                )
            else:
                next_boundary = total_end

    if observation_index != interval_count + 1:
        raise InvariantViolation(
            f"observation schedule incomplete: {observation_index} != {interval_count + 1}"
        )
    if not math.isclose(state.time_seconds, total_end, rel_tol=0.0, abs_tol=tolerance):
        raise InvariantViolation("simulation did not end on the horizon boundary")

    mean_gap = float(np.mean(gaps))
    centered = gaps - mean_gap
    variance_gap = float(np.mean(centered * centered))
    if variance_gap <= 0.0 or not math.isfinite(variance_gap):
        raise InvariantViolation("stationary variance estimate is nonpositive")
    s_g = math.sqrt(variance_gap)
    open_mask = ~tight
    open_occupancy = float(np.mean(open_mask))
    if not 0.0 < open_occupancy < 1.0:
        raise InvariantViolation("both parity states must be observed")

    observation_h = simulation_settings.observation_interval_seconds
    increments_per_second = np.diff(gaps) / observation_h
    left_gaps = gaps[:-1]
    left_tight_mask = tight[:-1]
    finite_slopes: dict[bool, float] = {}
    slope_counts: dict[bool, int] = {}
    for parity in (True, False):
        mask = left_tight_mask == parity
        count = int(np.count_nonzero(mask))
        slope_counts[parity] = count
        if count < simulation_settings.minimum_slope_observations:
            raise InvariantViolation(
                f"insufficient observations for {'tight' if parity else 'open'} slope: {count}"
            )
        denominator = float(np.dot(left_gaps[mask], left_gaps[mask]))
        if denominator <= 0.0:
            raise InvariantViolation("finite-h slope denominator is nonpositive")
        finite_slopes[parity] = float(
            np.dot(left_gaps[mask], increments_per_second[mask]) / denominator
        )

    acf: dict[float, float] = {}
    for lag_seconds in simulation_settings.acf_lags_seconds:
        lag_steps_float = lag_seconds / observation_h
        lag_steps = round(lag_steps_float)
        if not math.isclose(lag_steps_float, lag_steps, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("ACF lag is not on the observation grid")
        if lag_steps <= 0 or lag_steps >= gaps.size:
            raise ValueError("ACF lag lies outside the observed path")
        acf[lag_seconds] = float(np.mean(centered[:-lag_steps] * centered[lag_steps:]) / variance_gap)
    generator_slopes = {
        parity: generator_numerator[parity] / generator_denominator[parity]
        for parity in (True, False)
    }
    realised_jump_drift_slopes = {
        parity: realised_jump_drift_numerator[parity]
        / realised_jump_drift_denominator[parity]
        for parity in (True, False)
    }
    opening_hazard = float(channel_hazard_integrals[2] + channel_hazard_integrals[3])
    closing_hazard = float(channel_hazard_integrals[4] + channel_hazard_integrals[5])
    opening_count = int(measured_channel_counts[2] + measured_channel_counts[3])
    closing_count = int(measured_channel_counts[4] + measured_channel_counts[5])
    hazard_flow_denominator = opening_hazard + closing_hazard
    if hazard_flow_denominator <= 0.0:
        raise InvariantViolation("measured opening and closing hazards must be positive")
    integrated_hazard_flow_residual = (
        2.0 * (closing_hazard - opening_hazard) / hazard_flow_denominator
    )
    count_flow_denominator = opening_count + closing_count
    if count_flow_denominator <= 0:
        raise InvariantViolation("measured opening and closing transition counts must be positive")
    realised_count_flow_residual = (
        2.0 * (closing_count - opening_count) / count_flow_denominator
    )
    transition_count_imbalance = closing_count - opening_count
    if abs(transition_count_imbalance) > 1:
        raise InvariantViolation("opening/closing transition counts differ by more than one")
    channel_compensator_residuals = measured_channel_counts.astype(np.float64) - channel_hazard_integrals
    channel_compensator_z = np.divide(
        channel_compensator_residuals,
        np.sqrt(channel_hazard_integrals),
        out=np.zeros_like(channel_compensator_residuals),
        where=channel_hazard_integrals > 0.0,
    )
    metrics: dict[str, Any] = {
        "epsilon": float(epsilon),
        "seed": int(seed),
        "observation_count": int(gaps.size),
        "tight_observation_count_for_slope": slope_counts[True],
        "open_observation_count_for_slope": slope_counts[False],
        "stationary_s_g": s_g,
        "open_occupancy": open_occupancy,
        "finite_h_drift_slope_tight_per_second": finite_slopes[True],
        "finite_h_drift_slope_open_per_second": finite_slopes[False],
        "generator_drift_slope_tight_per_second": generator_slopes[True],
        "generator_drift_slope_open_per_second": generator_slopes[False],
        "realised_jump_drift_slope_tight_per_second": realised_jump_drift_slopes[True],
        "realised_jump_drift_slope_open_per_second": realised_jump_drift_slopes[False],
        "realised_jump_parity_drift_slope_contrast_per_second": realised_jump_drift_slopes[True]
        - realised_jump_drift_slopes[False],
        "generator_drift_abs_residual_tight": generator_max_residual[True],
        "generator_drift_abs_residual_open": generator_max_residual[False],
        "step_count": step_count,
        "book_event_count": book_event_count,
        "maximum_left_event_probability": maximum_left_event_probability,
        "deterministic_replay_mismatch_count": 0,
        "deterministic_replay_checked": False,
        "transition_count_imbalance": transition_count_imbalance,
        "integrated_hazard_flow_signed_relative_residual": integrated_hazard_flow_residual,
        "realised_count_flow_signed_relative_residual": realised_count_flow_residual,
    }
    for index, channel in enumerate(EVENT_CHANNELS):
        metrics[f"hazard_integral_{channel}"] = float(channel_hazard_integrals[index])
        metrics[f"measured_count_{channel}"] = int(measured_channel_counts[index])
        metrics[f"compensator_residual_{channel}"] = float(channel_compensator_residuals[index])
        metrics[f"compensator_z_{channel}"] = float(channel_compensator_z[index])
    for lag_seconds, value in acf.items():
        metrics[f"acf_lag_{lag_seconds:g}_seconds"] = value

    digest = _path_digest(
        state,
        gaps,
        tight,
        step_count,
        book_event_count,
        channel_hazard_integrals,
        measured_channel_counts,
    )
    return ReplicationResult(
        epsilon=float(epsilon),
        seed=int(seed),
        seed_metrics=metrics,
        gaps=gaps,
        tight=tight,
        replay_digest=digest,
        stream_spawn_keys=spawn_keys,
    )
