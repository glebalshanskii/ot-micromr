from __future__ import annotations

import hashlib
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
    interval = float(simulation["calibration_observation_interval_reversion_times"]) / alpha
    interval_count = round(measured / interval)
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
                burn + observation_index * interval
                if observation_index <= interval_count
                else total_end
            )
    if observation_index != interval_count + 1:
        raise RuntimeError("calibration observation schedule is incomplete")
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
    alpha_grid = tuple(float(value) for value in values["model"]["response_scale_alpha_per_second_grid"])
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
        values["strategy"]["threshold_multiplier_theta_over_theta_d_grid"], dtype=np.float64
    )
    thresholds = multipliers * calibration.theta_d_price
    labels = tuple(f"grid:{value:.12g}" for value in multipliers) + ("theta_star",)
    return (
        np.concatenate((thresholds, np.asarray([calibration.theta_star_price]))),
        labels,
        np.concatenate((multipliers, np.asarray([calibration.theta_star_price / calibration.theta_d_price]))),
    )


def _simulate_strategy_coordinate(
    values: Mapping[str, Any], calibration: CalibrationRow, epsilon: float, seed: int
) -> Figure4Replication:
    alpha = calibration.alpha_per_second
    model = _row_model(values, alpha)
    parameters = BookParameters.from_model(model)
    state = initial_state(model)
    thresholds, labels, multipliers = _policy_definition(values, calibration)
    policy_count = thresholds.size
    endpoint_rng, occurrence_rng, channel_rng, bridge_rng = _domain_streams(
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
    probability_cutoff = float(values["numerics"]["bridge_probability_cutoff"])
    tolerance = 5e-12

    positions = np.zeros(policy_count, dtype=np.int8)
    first_time = np.full(policy_count, np.nan, dtype=np.float64)
    last_time = np.full(policy_count, np.nan, dtype=np.float64)
    fill_count = np.zeros(policy_count, dtype=np.int64)
    reward_sum = np.zeros(policy_count, dtype=np.float64)
    frozen_reward_sum = np.zeros(policy_count, dtype=np.float64)
    overshoot_sum = np.zeros(policy_count, dtype=np.float64)
    open_fill_count = np.zeros(policy_count, dtype=np.int64)
    bridge_fill_count = np.zeros(policy_count, dtype=np.int64)
    jump_fill_count = np.zeros(policy_count, dtype=np.int64)
    cash = np.zeros(policy_count, dtype=np.float64)
    initial_positions = np.zeros(policy_count, dtype=np.int8)
    initial_mid = 0.0
    initial_x = 0.0
    initial_gap = 0.0
    measurement_reset = False

    step_count = 0
    event_count = 0
    maximum_event_probability = 0.0
    omitted_probability_sum = 0.0
    recrossing_probability_bound = 0.0
    flat_competing_probability_sum = 0.0
    invariant_count = 0

    def apply_fills(
        indices: np.ndarray,
        new_positions: np.ndarray,
        fill_times: np.ndarray,
        fill_gaps: np.ndarray,
        mid_price: float,
        spread_price: float,
        *,
        bridge: bool,
        measured: bool,
    ) -> None:
        if indices.size == 0:
            return
        old_positions = positions[indices].copy()
        if np.any((new_positions != 1) & (new_positions != -1)):
            raise RuntimeError("policy fill produced an invalid position")
        if np.any((old_positions != 0) & (new_positions == old_positions)):
            raise RuntimeError("policy fill failed to flip inventory")
        positions[indices] = new_positions
        if not measured:
            return
        delta_q = new_positions.astype(np.int64) - old_positions.astype(np.int64)
        touch = np.where(delta_q > 0, mid_price + spread_price / 2.0, mid_price - spread_price / 2.0)
        cash[indices] -= delta_q * touch
        for local, policy_index in enumerate(indices.tolist()):
            fill_time = float(fill_times[local])
            fill_gap = float(fill_gaps[local])
            if fill_count[policy_index] == 0:
                first_time[policy_index] = fill_time
            else:
                reward_sum[policy_index] += 2.0 * (abs(fill_gap) - spread_price / 2.0)
                frozen_reward_sum[policy_index] += 2.0 * (
                    abs(fill_gap) - float(values["execution"]["threshold_reference_phi_price"])
                )
            last_time[policy_index] = fill_time
            fill_count[policy_index] += 1
            overshoot_sum[policy_index] += max(abs(fill_gap) - thresholds[policy_index], 0.0)
            open_fill_count[policy_index] += int(spread_price > parameters.delta_price)
            if bridge:
                bridge_fill_count[policy_index] += 1
            else:
                jump_fill_count[policy_index] += 1

    while state.time_seconds < total_end - tolerance:
        left_time = state.time_seconds
        if left_time >= measurement_start - tolerance and not measurement_reset:
            if np.any(positions == 0):
                invariant_count += int(np.count_nonzero(positions == 0))
            first_time.fill(np.nan)
            last_time.fill(np.nan)
            fill_count.fill(0)
            reward_sum.fill(0.0)
            frozen_reward_sum.fill(0.0)
            overshoot_sum.fill(0.0)
            open_fill_count.fill(0)
            bridge_fill_count.fill(0)
            jump_fill_count.fill(0)
            cash.fill(0.0)
            initial_positions[:] = positions
            initial_mid = state.mid_price(parameters)
            initial_x = state.efficient_price
            initial_gap = state.gap_price(parameters)
            measurement_reset = True

        left_gap = state.gap_price(parameters)
        left_mid = state.mid_price(parameters)
        left_spread = state.spread_price(parameters)
        if left_time >= market_burn_end - tolerance and np.any(positions == 0):
            flat_indices = np.flatnonzero(positions == 0)
            upper = left_gap >= thresholds[flat_indices]
            lower = left_gap <= -thresholds[flat_indices]
            immediate = upper | lower
            selected = flat_indices[immediate]
            if selected.size:
                apply_fills(
                    selected,
                    np.where(upper[immediate], -1, 1).astype(np.int8),
                    np.full(selected.size, left_time),
                    np.where(upper[immediate], thresholds[selected], -thresholds[selected]),
                    left_mid,
                    left_spread,
                    bridge=True,
                    measured=left_time >= measurement_start - tolerance,
                )
        rates = intensities(state, parameters)
        total_rate = sum(rates)
        step = min(
            total_end - left_time,
            next_phase - left_time,
            max_step,
            hazard_numerator / total_rate,
        )
        if step <= 0.0 or not math.isfinite(step):
            raise RuntimeError("strategy simulation produced a nonpositive step")
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
        pre_event_gap = state.gap_price(parameters)
        new_time = left_time + step
        if abs(new_time - next_phase) <= tolerance:
            new_time = next_phase
        if abs(new_time - total_end) <= tolerance:
            new_time = total_end
        measured_step = left_time >= measurement_start - tolerance

        if left_time >= market_burn_end - tolerance:
            flat = positions == 0
            active_at_diffusion_start = ~flat
            if np.any(flat):
                flat_indices = np.flatnonzero(flat)
                local_thresholds = thresholds[flat_indices]
                upper_cross = pre_event_gap >= local_thresholds
                lower_cross = pre_event_gap <= -local_thresholds
                inside = ~(upper_cross | lower_cross)
                if np.any(inside):
                    variance = parameters.sigma_x_price_per_sqrt_second**2 * step
                    inside_thresholds = local_thresholds[inside]
                    p_upper = np.exp(
                        -2.0
                        * (inside_thresholds - left_gap)
                        * (inside_thresholds - pre_event_gap)
                        / variance
                    )
                    p_lower = np.exp(
                        -2.0
                        * (left_gap + inside_thresholds)
                        * (pre_event_gap + inside_thresholds)
                        / variance
                    )
                    p_upper = np.clip(p_upper, 0.0, 1.0)
                    p_lower = np.clip(p_lower, 0.0, 1.0)
                    flat_competing_probability_sum += float(np.sum(np.minimum(p_upper, p_lower)))
                    draw = bridge_rng.random(p_upper.size)
                    total_probability = np.minimum(p_upper + p_lower, 1.0)
                    hit = draw < total_probability
                    choose_upper = hit & (draw < p_upper)
                    choose_lower = hit & ~choose_upper
                    inside_locations = np.flatnonzero(inside)
                    upper_cross[inside_locations[choose_upper]] = True
                    lower_cross[inside_locations[choose_lower]] = True
                hit_mask = upper_cross | lower_cross
                selected = flat_indices[hit_mask]
                if selected.size:
                    new_positions = np.where(upper_cross[hit_mask], -1, 1).astype(np.int8)
                    hit_gaps = np.where(
                        upper_cross[hit_mask], thresholds[selected], -thresholds[selected]
                    )
                    apply_fills(
                        selected,
                        new_positions,
                        np.full(selected.size, left_time + step / 2.0),
                        hit_gaps,
                        left_mid,
                        left_spread,
                        bridge=True,
                        measured=measured_step,
                    )

            active = active_at_diffusion_start
            if np.any(active):
                active_indices = np.flatnonzero(active)
                active_positions = positions[active_indices]
                boundaries = np.where(
                    active_positions == 1,
                    thresholds[active_indices],
                    -thresholds[active_indices],
                )
                deterministic = np.where(
                    active_positions == 1,
                    pre_event_gap >= boundaries,
                    pre_event_gap <= boundaries,
                )
                probabilities = np.zeros(active_indices.size, dtype=np.float64)
                stochastic = ~deterministic
                if np.any(stochastic):
                    variance = parameters.sigma_x_price_per_sqrt_second**2 * step
                    b = boundaries[stochastic]
                    pos = active_positions[stochastic]
                    distance_left = np.where(pos == 1, b - left_gap, left_gap - b)
                    distance_right = np.where(pos == 1, b - pre_event_gap, pre_event_gap - b)
                    valid = (distance_left > 0.0) & (distance_right > 0.0)
                    local = np.zeros(b.size, dtype=np.float64)
                    local[valid] = np.exp(
                        -2.0 * distance_left[valid] * distance_right[valid] / variance
                    )
                    probabilities[stochastic] = np.clip(local, 0.0, 1.0)
                omitted = (probabilities > 0.0) & (probabilities < probability_cutoff)
                omitted_probability_sum += float(np.sum(probabilities[omitted]))
                candidates = probabilities >= probability_cutoff
                stochastic_hits = np.zeros(active_indices.size, dtype=np.bool_)
                if np.any(candidates):
                    stochastic_hits[candidates] = bridge_rng.random(
                        int(np.count_nonzero(candidates))
                    ) < probabilities[candidates]
                hits = deterministic | stochastic_hits
                selected = active_indices[hits]
                if selected.size:
                    selected_positions = positions[selected].copy()
                    new_positions = (-selected_positions).astype(np.int8)
                    hit_gaps = np.where(
                        selected_positions == 1, thresholds[selected], -thresholds[selected]
                    )
                    hit_times = np.full(selected.size, left_time + step / 2.0)
                    selected_deterministic = deterministic[hits]
                    if np.any(selected_deterministic) and pre_event_gap != left_gap:
                        deterministic_boundaries = hit_gaps[selected_deterministic]
                        fractions = np.clip(
                            (deterministic_boundaries - left_gap) / (pre_event_gap - left_gap),
                            0.0,
                            1.0,
                        )
                        hit_times[selected_deterministic] = left_time + fractions * step
                    apply_fills(
                        selected,
                        new_positions,
                        hit_times,
                        hit_gaps,
                        left_mid,
                        left_spread,
                        bridge=True,
                        measured=measured_step,
                    )
                    remaining = np.maximum(new_time - hit_times, 0.0)
                    full_width = 2.0 * thresholds[selected]
                    positive_remaining = remaining > 0.0
                    if np.any(positive_remaining):
                        recrossing_probability_bound += float(
                            np.sum(
                                np.exp(
                                    -0.5
                                    * full_width[positive_remaining] ** 2
                                    / (
                                        parameters.sigma_x_price_per_sqrt_second**2
                                        * remaining[positive_remaining]
                                    )
                                )
                            )
                        )

        state.time_seconds = new_time
        if channel is not None:
            apply_book_event(state, channel, parameters)
            event_count += 1
            if state.time_seconds >= market_burn_end - tolerance:
                post_gap = state.gap_price(parameters)
                active_indices = np.flatnonzero(positions != 0)
                active_positions = positions[active_indices]
                boundaries = np.where(
                    active_positions == 1,
                    thresholds[active_indices],
                    -thresholds[active_indices],
                )
                hits = np.where(
                    active_positions == 1, post_gap >= boundaries, post_gap <= boundaries
                )
                selected = active_indices[hits]
                if selected.size:
                    selected_positions = positions[selected].copy()
                    apply_fills(
                        selected,
                        (-selected_positions).astype(np.int8),
                        np.full(selected.size, state.time_seconds),
                        np.full(selected.size, post_gap),
                        state.mid_price(parameters),
                        state.spread_price(parameters),
                        bridge=False,
                        measured=measured_step,
                    )
        step_count += 1
        if state.time_seconds == next_phase:
            next_phase_index += 1
            next_phase = (
                phase_boundaries[next_phase_index]
                if next_phase_index < len(phase_boundaries)
                else total_end
            )

    if not measurement_reset:
        raise RuntimeError("measurement phase was never activated")
    terminal_mid = state.mid_price(parameters)
    terminal_x = state.efficient_price
    terminal_gap = state.gap_price(parameters)
    policy_rows: list[Mapping[str, Any]] = []
    minimum_intervals = int(
        values["evaluation"]["minimum_complete_interfill_intervals_per_seed_and_policy"]
    )
    for policy_index in range(policy_count):
        completed = max(int(fill_count[policy_index]) - 1, 0)
        duration = (
            float(last_time[policy_index] - first_time[policy_index])
            if completed > 0
            else 0.0
        )
        rate = float(reward_sum[policy_index] / duration) if duration > 0.0 else 0.0
        frozen_rate = (
            float(frozen_reward_sum[policy_index] / duration) if duration > 0.0 else 0.0
        )
        mid_pnl = float(
            cash[policy_index]
            + positions[policy_index] * terminal_mid
            - initial_positions[policy_index] * initial_mid
        )
        efficient_pnl = float(
            cash[policy_index]
            + positions[policy_index] * terminal_x
            - initial_positions[policy_index] * initial_x
        )
        identity_residual = abs(
            (mid_pnl - efficient_pnl)
            - (positions[policy_index] * terminal_gap - initial_positions[policy_index] * initial_gap)
        )
        policy_rows.append(
            {
                "row_index": calibration.row_index,
                "alpha_per_second": alpha,
                "gamma_ratio": calibration.gamma_ratio,
                "epsilon": epsilon,
                "seed": seed,
                "policy_index": policy_index,
                "policy_label": labels[policy_index],
                "threshold_multiplier_theta_d": float(multipliers[policy_index]),
                "threshold_price": float(thresholds[policy_index]),
                "fill_count": int(fill_count[policy_index]),
                "complete_interval_count": completed,
                "minimum_interval_requirement_met": completed >= minimum_intervals,
                "renewal_rate_per_second": rate,
                "renewal_rate_over_alpha_s_g": rate / (alpha * calibration.s_g_price),
                "renewal_rate_over_surrogate_optimum": rate
                / calibration.surrogate_optimum_rate_per_second,
                "frozen_cost_rate_per_second": frozen_rate,
                "mean_interfill_seconds": duration / completed if completed else None,
                "mean_fill_overshoot_price": float(overshoot_sum[policy_index] / fill_count[policy_index])
                if fill_count[policy_index]
                else None,
                "open_fill_share": float(open_fill_count[policy_index] / fill_count[policy_index])
                if fill_count[policy_index]
                else None,
                "bridge_fill_count": int(bridge_fill_count[policy_index]),
                "jump_fill_count": int(jump_fill_count[policy_index]),
                "mid_marked_fixed_horizon_pnl": mid_pnl,
                "efficient_price_marked_fixed_horizon_pnl": efficient_pnl,
                "wealth_marking_identity_abs_residual": identity_residual,
                "terminal_position": int(positions[policy_index]),
            }
        )

    digest = hashlib.sha256()
    for array in (
        positions,
        fill_count,
        reward_sum,
        frozen_reward_sum,
        overshoot_sum,
        open_fill_count,
    ):
        digest.update(np.asarray(array).tobytes(order="C"))
    digest.update(
        np.asarray(
            [state.time_seconds, state.efficient_price, state.mid_half_ticks, step_count, event_count],
            dtype=np.float64,
        ).tobytes()
    )
    diagnostics = {
        "step_count": step_count,
        "book_event_count": event_count,
        "maximum_left_event_probability": maximum_event_probability,
        "omitted_bridge_probability_sum": omitted_probability_sum,
        "full_band_recrossing_probability_bound": recrossing_probability_bound,
        "flat_entry_competing_probability_sum_burn_in_only": flat_competing_probability_sum,
        "nonflat_policy_count_at_end": int(np.count_nonzero(positions)),
        "policy_count": policy_count,
        "invariant_violation_count": invariant_count,
        "nonfinite_value_count": 0,
        "wealth_marking_identity_abs_residual_max": max(
            float(row["wealth_marking_identity_abs_residual"]) for row in policy_rows
        ),
    }
    return Figure4Replication(
        row_index=calibration.row_index,
        alpha_per_second=alpha,
        epsilon=epsilon,
        seed=seed,
        policy_rows=tuple(policy_rows),
        diagnostics=diagnostics,
        replay_digest=digest.hexdigest(),
    )


def _strategy_worker(
    payload: tuple[Mapping[str, Any], CalibrationRow, float, int]
) -> Figure4Replication:
    values, calibration, epsilon, seed = payload
    with threadpool_limits(limits=1):
        return _simulate_strategy_coordinate(values, calibration, epsilon, seed)


def simulate_figure4(
    values: Mapping[str, Any],
    calibrations: Sequence[CalibrationRow],
    coordinates: Sequence[tuple[int, float, int]],
    workers: int,
) -> tuple[Figure4Replication, ...]:
    by_row = {row.row_index: row for row in calibrations}
    payloads = [
        (values, by_row[row_index], epsilon, seed)
        for row_index, epsilon, seed in coordinates
    ]
    worker_count = min(workers, len(payloads))
    if worker_count == 1:
        return tuple(
            _simulate_strategy_coordinate(values, calibration, epsilon, seed)
            for _, calibration, epsilon, seed in payloads
        )
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return tuple(executor.map(_strategy_worker, payloads))


def replay_figure4_coordinate(
    values: Mapping[str, Any], calibration: CalibrationRow, epsilon: float, seed: int
) -> Figure4Replication:
    return _simulate_strategy_coordinate(values, calibration, epsilon, seed)
