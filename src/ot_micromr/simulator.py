from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ot_micromr.jump_model import (
    EVENT_CHANNELS,
    BookEventRecord,
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
    diagnostic_quantiles: tuple[float, ...]
    acf_lags_seconds: tuple[float, ...]
    minimum_slope_observations: int


@dataclass(frozen=True, slots=True)
class ReplicationResult:
    epsilon: float
    seed: int
    seed_metrics: Mapping[str, Any]
    gaps: np.ndarray
    tight: np.ndarray
    events: tuple[BookEventRecord, ...]
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
        diagnostic_quantiles=tuple(
            float(value) for value in values["numerics"]["diagnostic_quantile_probabilities"]
        ),
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


def _quantile_metrics(prefix: str, values: np.ndarray, probabilities: tuple[float, ...]) -> dict[str, float]:
    quantiles = np.quantile(values, probabilities)
    return {
        f"{prefix}_q_{probability:g}": float(value)
        for probability, value in zip(probabilities, quantiles, strict=True)
    }


def _path_digest(
    state: BookState,
    gaps: np.ndarray,
    tight: np.ndarray,
    events: list[BookEventRecord],
    step_count: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(gaps.tobytes(order="C"))
    digest.update(tight.tobytes(order="C"))
    digest.update(struct.pack("<ddiQ", state.time_seconds, state.efficient_price, state.mid_half_ticks, step_count))
    for event in events:
        digest.update(
            struct.pack(
                "<Qdidddid?",
                event.event_index,
                event.time_seconds,
                event.pre_mid_half_ticks,
                event.left_gap_price,
                event.pre_event_gap_price,
                event.post_event_gap_price,
                event.post_mid_half_ticks,
                event.left_channel_intensity_per_second,
                event.measured,
            )
        )
        digest.update(event.channel.encode("ascii"))
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
    step_sizes: list[float] = []
    right_lambda_steps: list[float] = []
    events: list[BookEventRecord] = []
    step_count = 0
    measured_step_count = 0
    measured_event_count = 0
    measured_jump_square_sum = 0.0
    maximum_left_event_probability = 0.0
    generator_max_residual = {True: 0.0, False: 0.0}
    generator_numerator = {True: 0.0, False: 0.0}
    generator_denominator = {True: 0.0, False: 0.0}
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
            pre_event_gap = state.gap_price(parameters)
            pre_event_ticks = state.mid_half_ticks
            pre_event_x = state.efficient_price
            delta_mid = apply_book_event(state, selected_channel, parameters)
            if state.efficient_price != pre_event_x:
                raise InvariantViolation("X is discontinuous at book event")
            measured = state.time_seconds > simulation_settings.burn_seconds + tolerance
            event = BookEventRecord(
                epsilon=float(epsilon),
                seed=int(seed),
                event_index=len(events),
                time_seconds=state.time_seconds,
                channel=selected_channel,
                left_gap_price=left_gap,
                pre_event_gap_price=pre_event_gap,
                post_event_gap_price=state.gap_price(parameters),
                pre_mid_half_ticks=pre_event_ticks,
                post_mid_half_ticks=state.mid_half_ticks,
                efficient_price=state.efficient_price,
                delta_mid_price=delta_mid,
                left_channel_intensity_per_second=left_rates[selected_index],
                measured=measured,
            )
            events.append(event)
            if measured:
                measured_event_count += 1
                measured_jump_square_sum += delta_mid * delta_mid

        right_rates = intensities(state, parameters)
        right_lambda_steps.append(sum(right_rates) * step)
        step_sizes.append(step)
        step_count += 1
        drift = generator_mid_drift(state, parameters, left_rates)
        coefficient = (
            parameters.tight_drift_coefficient_per_second
            if left_tight
            else parameters.open_drift_coefficient_per_second
        )
        residual = abs(drift + coefficient * left_gap)
        generator_max_residual[left_tight] = max(generator_max_residual[left_tight], residual)
        if left_time >= simulation_settings.burn_seconds - tolerance:
            measured_step_count += 1
            generator_numerator[left_tight] += left_gap * drift
            generator_denominator[left_tight] += left_gap * left_gap

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
    normalized_mean = mean_gap / s_g
    open_mask = ~tight
    open_occupancy = float(np.mean(open_mask))
    if not 0.0 < open_occupancy < 1.0:
        raise InvariantViolation("both parity states must be observed")
    mean_abs_tight = float(np.mean(np.abs(gaps[tight])))
    mean_abs_open = float(np.mean(np.abs(gaps[open_mask])))
    occupancy_odds = open_occupancy / (1.0 - open_occupancy)
    intensity_odds = (
        parameters.mu_o_per_second
        + parameters.alpha_o_per_second * mean_abs_tight / parameters.delta_price
    ) / (
        parameters.mu_c_per_second
        + parameters.alpha_c_per_second * mean_abs_open / parameters.delta_price
    )
    flow_denominator = abs(occupancy_odds) + abs(intensity_odds)
    flow_residual = 2.0 * (occupancy_odds - intensity_odds) / flow_denominator

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
    lag_one_acf = float(np.mean(centered[:-1] * centered[1:]) / variance_gap)
    effective_sample_size = float(
        gaps.size * max(1.0 - lag_one_acf, 0.0) / max(1.0 + lag_one_acf, np.finfo(float).eps)
    )

    jump_variance_rate = measured_jump_square_sum / simulation_settings.horizon_seconds
    variance_alpha = parameters.tight_drift_coefficient_per_second
    variance_target = (
        parameters.sigma_x_price_per_sqrt_second**2 + jump_variance_rate
    ) / (2.0 * variance_alpha)
    variance_residual = (variance_gap - variance_target) / variance_target
    generator_slopes = {
        parity: generator_numerator[parity] / generator_denominator[parity]
        for parity in (True, False)
    }
    step_array = np.asarray(step_sizes, dtype=np.float64)
    right_array = np.asarray(right_lambda_steps, dtype=np.float64)
    metrics: dict[str, Any] = {
        "epsilon": float(epsilon),
        "seed": int(seed),
        "observation_count": int(gaps.size),
        "tight_observation_count_for_slope": slope_counts[True],
        "open_observation_count_for_slope": slope_counts[False],
        "stationary_mean_gap": mean_gap,
        "stationary_variance_gap": variance_gap,
        "stationary_s_g": s_g,
        "stationary_mean_gap_over_s_g": normalized_mean,
        "jump_variance_rate": jump_variance_rate,
        "stationary_variance_identity_target": variance_target,
        "stationary_variance_identity_signed_relative_residual": variance_residual,
        "open_occupancy": open_occupancy,
        "open_close_flow_signed_relative_residual": flow_residual,
        "mean_abs_gap_tight": mean_abs_tight,
        "mean_abs_gap_open": mean_abs_open,
        "finite_h_drift_slope_tight_per_second": finite_slopes[True],
        "finite_h_drift_slope_open_per_second": finite_slopes[False],
        "finite_h_parity_drift_slope_contrast_per_second": finite_slopes[True]
        - finite_slopes[False],
        "generator_drift_slope_tight_per_second": generator_slopes[True],
        "generator_drift_slope_open_per_second": generator_slopes[False],
        "generator_drift_abs_residual_tight": generator_max_residual[True],
        "generator_drift_abs_residual_open": generator_max_residual[False],
        "step_count": step_count,
        "measured_step_count": measured_step_count,
        "book_event_count": len(events),
        "measured_book_event_count": measured_event_count,
        "event_step_fraction": len(events) / step_count,
        "maximum_left_event_probability": maximum_left_event_probability,
        "effective_sample_size_ou_approximation": effective_sample_size,
        "invariant_violation_count": 0,
        "parity_violation_count": 0,
        "illegal_transition_count": 0,
        "negative_intensity_count": 0,
        "nonzero_inactive_intensity_count": 0,
        "nonfinite_value_count": 0,
        "multiple_book_event_step_count": 0,
        "deterministic_replay_mismatch_count": 0,
        "deterministic_replay_checked": False,
        "bridge_only_crossing_count": None,
        "bridge_only_crossing_applicability": "not_applicable_strategy_monitoring_disabled",
        "multiple_crossing_refinement_count": None,
        "multiple_crossing_refinement_applicability": "not_applicable_strategy_monitoring_disabled",
    }
    metrics.update(_quantile_metrics("step_size_seconds", step_array, simulation_settings.diagnostic_quantiles))
    metrics.update(
        _quantile_metrics(
            "lambda_total_right_times_step", right_array, simulation_settings.diagnostic_quantiles
        )
    )
    for lag_seconds, value in acf.items():
        metrics[f"acf_lag_{lag_seconds:g}_seconds"] = value

    digest = _path_digest(state, gaps, tight, events, step_count)
    return ReplicationResult(
        epsilon=float(epsilon),
        seed=int(seed),
        seed_metrics=metrics,
        gaps=gaps,
        tight=tight,
        events=tuple(events),
        replay_digest=digest,
        stream_spawn_keys=spawn_keys,
    )
