from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ot_micromr.artifacts import atomic_write_json, write_csv
from ot_micromr.config import RunSpec
from ot_micromr.efficient_price import systematic_resample
from ot_micromr.errors import ExperimentError


DIRECTION_COUNT = 3
FAMILY_COUNT = 3
MAGNITUDE_BUCKET_COUNT = 9
MARK_COUNT = (
    DIRECTION_COUNT * FAMILY_COUNT * MAGNITUDE_BUCKET_COUNT * MAGNITUDE_BUCKET_COUNT
)


@dataclass(frozen=True, slots=True)
class MarkedEvaluationResult:
    metrics: Mapping[str, Any]
    acceptance: Mapping[str, bool]
    derived_parameters: Mapping[str, Any]
    log_lines: Sequence[str]

    @property
    def passed(self) -> bool:
        return all(self.acceptance.values())


@dataclass(frozen=True, slots=True)
class MarkTables:
    probabilities: torch.Tensor
    correction_weights: torch.Tensor
    jump_abs_price: torch.Tensor
    mark_dy: torch.Tensor
    mark_dd: torch.Tensor
    baseline_rates: torch.Tensor
    direction_sign: torch.Tensor
    family_sign: torch.Tensor
    midpoint_bucket: torch.Tensor
    spread_bucket: torch.Tensor
    correction_sum_down: torch.Tensor
    correction_sum_up: torch.Tensor
    active_ids: torch.Tensor
    active_probabilities: torch.Tensor
    active_correction_weights: torch.Tensor
    active_dy: torch.Tensor
    active_dd: torch.Tensor


def magnitude_power_bucket(values: torch.Tensor, maximum_power: int = 7) -> torch.Tensor:
    absolute = torch.abs(values).to(torch.int64)
    thresholds = torch.pow(
        absolute.new_tensor(2),
        torch.arange(0, maximum_power + 1, device=absolute.device, dtype=torch.int64),
    )
    return torch.sum(absolute.unsqueeze(-1) >= thresholds, dim=-1).to(torch.int64)


def encode_mark(delta_bid: torch.Tensor, delta_ask: torch.Tensor) -> torch.Tensor:
    delta_y = delta_bid + delta_ask
    delta_d = delta_ask - delta_bid
    direction = torch.sign(delta_y).to(torch.int64) + 1
    family = torch.sign(delta_d).to(torch.int64) + 1
    midpoint_bucket = magnitude_power_bucket(delta_y)
    spread_bucket = magnitude_power_bucket(delta_d)
    return (
        ((direction * FAMILY_COUNT + family) * MAGNITUDE_BUCKET_COUNT + midpoint_bucket)
        * MAGNITUDE_BUCKET_COUNT
        + spread_bucket
    )


def mark_metadata(device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
    mark = torch.arange(MARK_COUNT, device=device, dtype=torch.int64)
    spread_bucket = torch.remainder(mark, MAGNITUDE_BUCKET_COUNT)
    remainder = torch.div(mark, MAGNITUDE_BUCKET_COUNT, rounding_mode="floor")
    midpoint_bucket = torch.remainder(remainder, MAGNITUDE_BUCKET_COUNT)
    remainder = torch.div(remainder, MAGNITUDE_BUCKET_COUNT, rounding_mode="floor")
    family = torch.remainder(remainder, FAMILY_COUNT)
    direction = torch.div(remainder, FAMILY_COUNT, rounding_mode="floor")
    return direction - 1, family - 1, midpoint_bucket, spread_bucket


def previous_spread_bucket(spread_ticks: torch.Tensor, exact_maximum: int = 7) -> torch.Tensor:
    return torch.clamp(spread_ticks.to(torch.int64), min=1, max=exact_maximum + 1) - 1


def _tensor_digest(values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = value.detach().to(device="cpu").contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(bytes(contiguous.untyped_storage()))
    return digest.hexdigest()


def _normal_lower_bound(values: torch.Tensor, alpha: float) -> tuple[float, float, float]:
    sample = values.to(torch.float64)
    mean = sample.mean()
    standard_error = sample.std(unbiased=True) / math.sqrt(sample.numel())
    critical = torch.distributions.Normal(0.0, 1.0).icdf(
        sample.new_tensor(1.0 - alpha)
    )
    return float(mean), float(standard_error), float(mean - critical * standard_error)


def _normal_equivalence_interval(
    values: torch.Tensor, alpha: float
) -> tuple[float, float, float, float]:
    sample = values.to(torch.float64)
    mean = sample.mean()
    standard_error = sample.std(unbiased=True) / math.sqrt(sample.numel())
    critical = torch.distributions.Normal(0.0, 1.0).icdf(
        sample.new_tensor(1.0 - alpha)
    )
    return (
        float(mean),
        float(standard_error),
        float(mean - critical * standard_error),
        float(mean + critical * standard_error),
    )


def _synthetic_mark_tables(spec: RunSpec, device: torch.device) -> MarkTables:
    model = spec.values["model"]
    spread_count = int(model["spread_state_count"])
    delta = float(model["delta_price"])
    sizes = torch.tensor(
        model["translation_tick_sizes"], device=device, dtype=torch.int64
    )
    translation_weights = torch.tensor(
        model["translation_weights"], device=device, dtype=torch.float32
    )
    signs = torch.tensor((-1, 1), device=device, dtype=torch.int64)
    states = torch.arange(1, spread_count + 1, device=device, dtype=torch.int64)

    translation_grid = torch.cartesian_prod(states, sizes, signs)
    state_t = translation_grid[:, 0]
    size_t = translation_grid[:, 1]
    sign_t = translation_grid[:, 2]
    dy_translation = 2 * size_t * sign_t
    dd_translation = torch.zeros_like(dy_translation)
    size_weight_index = torch.searchsorted(sizes, size_t.contiguous())
    weight_translation = translation_weights[size_weight_index]

    maximum_change = int(model["maximum_spread_change_ticks"])
    changes = torch.arange(1, maximum_change + 1, device=device, dtype=torch.int64)
    spread_grid = torch.cartesian_prod(states, changes, signs)
    state_s = spread_grid[:, 0]
    change_s = spread_grid[:, 1]
    sign_s = spread_grid[:, 2]

    widening_valid = state_s + change_s <= spread_count
    narrowing_valid = state_s - change_s >= 1
    widening_weight = float(model["widening_weight"]) / change_s.to(torch.float32)
    narrowing_weight = (
        float(model["narrowing_weight_per_tick"])
        * (state_s - 1).to(torch.float32)
        / change_s.to(torch.float32)
    )

    all_states = torch.cat(
        (
            state_t,
            state_s[widening_valid],
            state_s[narrowing_valid],
        )
    )
    all_dy = torch.cat(
        (
            dy_translation,
            (change_s * sign_s)[widening_valid],
            (change_s * sign_s)[narrowing_valid],
        )
    )
    all_dd = torch.cat(
        (
            dd_translation,
            change_s[widening_valid],
            -change_s[narrowing_valid],
        )
    )
    all_weights = torch.cat(
        (
            weight_translation,
            widening_weight[widening_valid],
            narrowing_weight[narrowing_valid],
        )
    )
    delta_bid = torch.div(all_dy - all_dd, 2, rounding_mode="floor")
    delta_ask = torch.div(all_dy + all_dd, 2, rounding_mode="floor")
    marks = encode_mark(delta_bid, delta_ask)
    state_index = all_states - 1
    flat_index = state_index * MARK_COUNT + marks

    flat_weights = torch.zeros(
        spread_count * MARK_COUNT, device=device, dtype=torch.float32
    )
    flat_dy_weighted = torch.zeros_like(flat_weights)
    flat_dd_weighted = torch.zeros_like(flat_weights)
    flat_weights.scatter_add_(0, flat_index, all_weights)
    flat_dy_weighted.scatter_add_(0, flat_index, all_weights * all_dy.to(torch.float32))
    flat_dd_weighted.scatter_add_(0, flat_index, all_weights * all_dd.to(torch.float32))
    weight_matrix = flat_weights.reshape(spread_count, MARK_COUNT)
    probabilities = weight_matrix / weight_matrix.sum(dim=-1, keepdim=True)
    safe_weight = torch.clamp_min(weight_matrix, torch.finfo(torch.float32).tiny)
    mark_dy = torch.round(
        flat_dy_weighted.reshape_as(weight_matrix) / safe_weight
    ).to(torch.int64)
    mark_dd = torch.round(
        flat_dd_weighted.reshape_as(weight_matrix) / safe_weight
    ).to(torch.int64)
    jump_abs = torch.abs(mark_dy).to(torch.float32) * (delta / 2.0)

    direction_sign, family_sign, midpoint_bucket, spread_bucket = mark_metadata(device)
    down = (direction_sign < 0).to(torch.float32).unsqueeze(0)
    up = (direction_sign > 0).to(torch.float32).unsqueeze(0)
    down_moment = torch.sum(probabilities * jump_abs * down, dim=-1, keepdim=True)
    up_moment = torch.sum(probabilities * jump_abs * up, dim=-1, keepdim=True)
    correction_weights = probabilities * (
        down / torch.clamp_min(down_moment, 1e-12)
        + up / torch.clamp_min(up_moment, 1e-12)
    )
    correction_sum_down = torch.sum(correction_weights * down, dim=-1)
    correction_sum_up = torch.sum(correction_weights * up, dim=-1)

    active_count = int(torch.max(torch.count_nonzero(weight_matrix, dim=-1)))
    active_probabilities, active_ids = torch.topk(
        probabilities, active_count, dim=-1, largest=True, sorted=False
    )
    active_correction = torch.gather(correction_weights, 1, active_ids)
    active_dy = torch.gather(mark_dy, 1, active_ids)
    active_dd = torch.gather(mark_dd, 1, active_ids)
    baseline_rates = torch.tensor(
        model["baseline_rates_per_second"], device=device, dtype=torch.float32
    )
    return MarkTables(
        probabilities=probabilities,
        correction_weights=correction_weights,
        jump_abs_price=jump_abs,
        mark_dy=mark_dy,
        mark_dd=mark_dd,
        baseline_rates=baseline_rates,
        direction_sign=direction_sign,
        family_sign=family_sign,
        midpoint_bucket=midpoint_bucket,
        spread_bucket=spread_bucket,
        correction_sum_down=correction_sum_down,
        correction_sum_up=correction_sum_up,
        active_ids=active_ids,
        active_probabilities=active_probabilities,
        active_correction_weights=active_correction,
        active_dy=active_dy,
        active_dd=active_dd,
    )


def _marked_interval_score(
    gap: torch.Tensor,
    spread_index: torch.Tensor,
    events: torch.Tensor,
    dt: float | torch.Tensor,
    tables: MarkTables,
    alpha: float,
) -> torch.Tensor:
    safe_events = torch.clamp_min(events, 0)
    base = tables.baseline_rates[spread_index] * tables.probabilities[
        spread_index, safe_events
    ]
    kappa = tables.correction_weights[spread_index, safe_events]
    direction = tables.direction_sign[safe_events]
    directional_gap = torch.where(
        direction > 0,
        torch.clamp_min(-gap, 0.0),
        torch.where(direction < 0, torch.clamp_min(gap, 0.0), torch.zeros_like(gap)),
    )
    selected = base + alpha * kappa * directional_gap
    total = tables.baseline_rates[spread_index] + alpha * torch.where(
        gap >= 0.0,
        gap * tables.correction_sum_down[spread_index],
        -gap * tables.correction_sum_up[spread_index],
    )
    event_score = torch.log(torch.clamp_min(selected, 1e-30)) - total * dt
    return torch.where(events >= 0, event_score, -total * dt)


def _make_marked_simulator_chunk(
    spec: RunSpec, tables: MarkTables
):
    model = spec.values["model"]
    simulation = spec.values["simulation"]
    numerics = spec.values["numerics"]
    dt = float(simulation["time_step_seconds"])
    chunk_steps = int(numerics["chunk_steps"])
    delta = float(model["delta_price"])
    alpha = float(model["alpha_per_second"])
    sigma_step = float(model["sigma_x_price_per_sqrt_second"]) * math.sqrt(dt)
    active_direction = tables.direction_sign[tables.active_ids]

    def simulator_chunk(
        efficient_price: torch.Tensor,
        mid_half_ticks: torch.Tensor,
        spread_ticks: torch.Tensor,
        brownian: torch.Tensor,
        occurrence_uniform: torch.Tensor,
        mark_uniform: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        session_count = efficient_price.shape[0]
        x_path = torch.empty(
            (chunk_steps, session_count), device=efficient_price.device, dtype=torch.float32
        )
        mid_path = torch.empty(
            (chunk_steps, session_count), device=efficient_price.device, dtype=torch.int64
        )
        spread_path = torch.empty_like(mid_path)
        event_path = torch.empty_like(mid_path)
        maximum_probability = efficient_price.new_zeros(())
        for step in range(chunk_steps):
            spread_index = spread_ticks - 1
            gap = mid_half_ticks.to(torch.float32) * (delta / 2.0) - efficient_price
            direction = active_direction[spread_index]
            directional_gap = torch.where(
                direction > 0,
                torch.clamp_min(-gap, 0.0).unsqueeze(-1),
                torch.where(
                    direction < 0,
                    torch.clamp_min(gap, 0.0).unsqueeze(-1),
                    torch.zeros_like(direction, dtype=torch.float32),
                ),
            )
            rates = (
                tables.baseline_rates[spread_index].unsqueeze(-1)
                * tables.active_probabilities[spread_index]
                + alpha
                * tables.active_correction_weights[spread_index]
                * directional_gap
            )
            total = rates.sum(dim=-1)
            occurrence_probability = -torch.expm1(-total * dt)
            maximum_probability = torch.maximum(maximum_probability, occurrence_probability.max())
            occurs = occurrence_uniform[step] < occurrence_probability
            conditional_cdf = torch.cumsum(rates, dim=-1) / total.unsqueeze(-1)
            selected_position = torch.sum(
                mark_uniform[step].unsqueeze(-1) > conditional_cdf, dim=-1
            ).clamp_max(rates.shape[-1] - 1)
            selected_id = torch.gather(
                tables.active_ids[spread_index], 1, selected_position.unsqueeze(-1)
            ).squeeze(-1)
            selected_dy = torch.gather(
                tables.active_dy[spread_index], 1, selected_position.unsqueeze(-1)
            ).squeeze(-1)
            selected_dd = torch.gather(
                tables.active_dd[spread_index], 1, selected_position.unsqueeze(-1)
            ).squeeze(-1)
            mid_half_ticks = mid_half_ticks + torch.where(
                occurs, selected_dy, torch.zeros_like(selected_dy)
            )
            spread_ticks = spread_ticks + torch.where(
                occurs, selected_dd, torch.zeros_like(selected_dd)
            )
            efficient_price = efficient_price + sigma_step * brownian[step]
            x_path[step] = efficient_price
            mid_path[step] = mid_half_ticks
            spread_path[step] = spread_ticks
            event_path[step] = torch.where(
                occurs, selected_id, torch.full_like(selected_id, -1)
            )
        return (
            efficient_price,
            mid_half_ticks,
            spread_ticks,
            x_path,
            mid_path,
            spread_path,
            event_path,
            maximum_probability,
        )

    return torch.compile(
        simulator_chunk,
        mode=str(numerics["compile_mode"]),
        fullgraph=True,
    )


def _simulate_marked_market(
    spec: RunSpec, tables: MarkTables
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    model = spec.values["model"]
    simulation = spec.values["simulation"]
    numerics = spec.values["numerics"]
    device = tables.probabilities.device
    sessions = int(simulation["session_count"])
    dt = float(simulation["time_step_seconds"])
    total_steps = round(
        (float(simulation["burn_in_seconds"]) + float(simulation["horizon_seconds"])) / dt
    )
    chunk_steps = int(numerics["chunk_steps"])
    generator = torch.Generator(device=device).manual_seed(
        int(spec.values["seed_policy"]["seeds"][0])
    )
    efficient_price = torch.full(
        (sessions,), float(model["initial_efficient_price"]), device=device, dtype=torch.float32
    )
    mid_half_ticks = torch.full(
        (sessions,), int(model["initial_mid_half_ticks"]), device=device, dtype=torch.int64
    )
    spread_ticks = torch.full(
        (sessions,), int(model["initial_spread_ticks"]), device=device, dtype=torch.int64
    )
    x_path = torch.empty((total_steps, sessions), device=device, dtype=torch.float32)
    mid_path = torch.empty((total_steps, sessions), device=device, dtype=torch.int64)
    spread_path = torch.empty_like(mid_path)
    events = torch.empty_like(mid_path)
    simulator = _make_marked_simulator_chunk(spec, tables)
    maximum_probability = 0.0
    for start in range(0, total_steps, chunk_steps):
        brownian = torch.randn(
            (chunk_steps, sessions), device=device, dtype=torch.float32, generator=generator
        )
        occurrence = torch.rand(
            (chunk_steps, sessions), device=device, dtype=torch.float32, generator=generator
        )
        mark_uniform = torch.rand(
            (chunk_steps, sessions), device=device, dtype=torch.float32, generator=generator
        )
        (
            efficient_price,
            mid_half_ticks,
            spread_ticks,
            x_chunk,
            mid_chunk,
            spread_chunk,
            event_chunk,
            chunk_maximum,
        ) = simulator(
            efficient_price,
            mid_half_ticks,
            spread_ticks,
            brownian,
            occurrence,
            mark_uniform,
        )
        stop = start + chunk_steps
        x_path[start:stop] = x_chunk
        mid_path[start:stop] = mid_chunk
        spread_path[start:stop] = spread_chunk
        events[start:stop] = event_chunk
        maximum_probability = max(maximum_probability, float(chunk_maximum))
        efficient_price = efficient_price.clone()
        mid_half_ticks = mid_half_ticks.clone()
        spread_ticks = spread_ticks.clone()
    return x_path, mid_path, spread_path, events, maximum_probability


def _make_marked_particle_chunk(spec: RunSpec, tables: MarkTables):
    model = spec.values["model"]
    simulation = spec.values["simulation"]
    numerics = spec.values["numerics"]
    dt = float(simulation["time_step_seconds"])
    chunk_steps = int(numerics["chunk_steps"])
    sigma_step = float(model["sigma_x_price_per_sqrt_second"]) * math.sqrt(dt)
    delta = float(model["delta_price"])
    alpha = float(model["alpha_per_second"])

    def particle_chunk(
        particles: torch.Tensor,
        log_weights: torch.Tensor,
        prior_mid: torch.Tensor,
        prior_spread: torch.Tensor,
        end_mid: torch.Tensor,
        events: torch.Tensor,
        transition_normals: torch.Tensor,
        resampling_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        particle_ends = particles.unsqueeze(0) + sigma_step * torch.cumsum(
            transition_normals, dim=0
        )
        particle_starts = torch.cat((particles.unsqueeze(0), particle_ends[:-1]), dim=0)
        start_mid = torch.cat((prior_mid.unsqueeze(0), end_mid[:-1]), dim=0)
        gap = start_mid.to(torch.float32).unsqueeze(-1) * (delta / 2.0) - particle_starts
        spread_index = prior_spread - 1
        safe_events = torch.clamp_min(events, 0)
        base = tables.baseline_rates[spread_index] * tables.probabilities[
            spread_index, safe_events
        ]
        kappa = tables.correction_weights[spread_index, safe_events]
        direction = tables.direction_sign[safe_events]
        directional_gap = torch.where(
            direction.unsqueeze(-1) > 0,
            torch.clamp_min(-gap, 0.0),
            torch.where(
                direction.unsqueeze(-1) < 0,
                torch.clamp_min(gap, 0.0),
                torch.zeros_like(gap),
            ),
        )
        selected = base.unsqueeze(-1) + alpha * kappa.unsqueeze(-1) * directional_gap
        total = tables.baseline_rates[spread_index].unsqueeze(-1) + alpha * torch.where(
            gap >= 0.0,
            gap * tables.correction_sum_down[spread_index].unsqueeze(-1),
            -gap * tables.correction_sum_up[spread_index].unsqueeze(-1),
        )
        interval_score = torch.where(
            events.unsqueeze(-1) >= 0,
            torch.log(torch.clamp_min(selected, 1e-30)) - total * dt,
            -total * dt,
        )
        cumulative = log_weights.unsqueeze(0) + torch.cumsum(interval_score, dim=0)
        normalizers = torch.logsumexp(cumulative, dim=-1)
        previous = torch.cat((normalizers.new_zeros((1, normalizers.shape[1])), normalizers[:-1]))
        predictive_scores = normalizers - previous
        normalized = cumulative - normalizers.unsqueeze(-1)
        weights = torch.exp(normalized)
        estimates = torch.sum(weights * particle_ends, dim=-1)
        variances = torch.sum(
            weights * torch.square(particle_ends - estimates.unsqueeze(-1)), dim=-1
        )
        particles, log_weights = systematic_resample(
            particle_ends[-1], normalized[-1], resampling_offsets
        )
        return particles, log_weights, estimates, variances, predictive_scores

    return torch.compile(
        particle_chunk,
        mode=str(numerics["compile_mode"]),
        fullgraph=True,
    )


def _filter_marked_market(
    spec: RunSpec,
    tables: MarkTables,
    mid_path: torch.Tensor,
    spread_path: torch.Tensor,
    events: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    model = spec.values["model"]
    numerics = spec.values["numerics"]
    total_steps, sessions = events.shape
    chunk_steps = int(numerics["chunk_steps"])
    particle_count = int(numerics["particle_count"])
    device = events.device
    generator = torch.Generator(device=device).manual_seed(
        int(spec.values["seed_policy"]["seeds"][0]) + 1
    )
    particles = float(model["initial_efficient_price"]) + float(
        model["particle_initial_standard_deviation_price"]
    ) * torch.randn(
        (sessions, particle_count), device=device, dtype=torch.float32, generator=generator
    )
    log_weights = torch.full_like(particles, -math.log(particle_count))
    estimates = torch.empty_like(mid_path, dtype=torch.float32)
    variances = torch.empty_like(estimates)
    scores = torch.empty_like(estimates)
    naive_scores = torch.empty_like(estimates)
    particle_chunk = _make_marked_particle_chunk(spec, tables)
    prior_mid = torch.full(
        (sessions,), int(model["initial_mid_half_ticks"]), device=device, dtype=torch.int64
    )
    prior_spread = torch.full(
        (sessions,), int(model["initial_spread_ticks"]), device=device, dtype=torch.int64
    )
    dt = float(spec.values["simulation"]["time_step_seconds"])
    delta = float(model["delta_price"])
    alpha = float(model["alpha_per_second"])
    for start in range(0, total_steps, chunk_steps):
        stop = start + chunk_steps
        transition = torch.randn(
            (chunk_steps, sessions, particle_count),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        offsets = torch.rand((sessions,), device=device, dtype=torch.float32, generator=generator)
        start_spread = torch.cat((prior_spread.unsqueeze(0), spread_path[start : stop - 1]), dim=0)
        particles, log_weights, estimate, variance, score = particle_chunk(
            particles,
            log_weights,
            prior_mid,
            start_spread,
            mid_path[start:stop],
            events[start:stop],
            transition,
            offsets,
        )
        estimates[start:stop] = estimate
        variances[start:stop] = variance
        scores[start:stop] = score
        start_mid = torch.cat((prior_mid.unsqueeze(0), mid_path[start : stop - 1]), dim=0)
        zero_gap = torch.zeros_like(start_mid, dtype=torch.float32)
        naive_scores[start:stop] = _marked_interval_score(
            zero_gap,
            start_spread - 1,
            events[start:stop],
            dt,
            tables,
            alpha,
        )
        prior_mid = mid_path[stop - 1]
        prior_spread = spread_path[stop - 1]
        particles = particles.clone()
        log_weights = log_weights.clone()
    return estimates, variances, scores, naive_scores, _tensor_digest((estimates, variances, scores))


def evaluate_marked_synthetic_filter(
    spec: RunSpec, run_directory: Path
) -> MarkedEvaluationResult:
    if not torch.cuda.is_available():
        raise ExperimentError("FILTER-MARK-SYN-001 requires an available CUDA device")
    started = time.perf_counter()
    device = torch.device("cuda")
    tables = _synthetic_mark_tables(spec, device)
    x_path, mid_path, spread_path, events, maximum_probability = _simulate_marked_market(
        spec, tables
    )
    estimates, variances, scores, naive_scores, first_digest = _filter_marked_market(
        spec, tables, mid_path, spread_path, events
    )
    replay_estimates, replay_variances, replay_scores, _, second_digest = _filter_marked_market(
        spec, tables, mid_path, spread_path, events
    )
    deterministic_replay = (
        first_digest == second_digest
        and torch.equal(estimates, replay_estimates)
        and torch.equal(variances, replay_variances)
        and torch.equal(scores, replay_scores)
    )

    model = spec.values["model"]
    simulation = spec.values["simulation"]
    evaluation = spec.values["evaluation"]
    dt = float(simulation["time_step_seconds"])
    burn_steps = round(float(simulation["burn_in_seconds"]) / dt)
    delta = float(model["delta_price"])
    measured_x = x_path[burn_steps:]
    measured_mid = mid_path[burn_steps:].to(torch.float32) * (delta / 2.0)
    measured_estimates = estimates[burn_steps:]
    measured_variances = variances[burn_steps:]
    measured_events = events[burn_steps:]
    event_counts = torch.count_nonzero(measured_events >= 0, dim=0)
    if torch.any(event_counts == 0):
        raise ExperimentError("synthetic session with zero measured marked events")

    pf_rmse = torch.sqrt(torch.mean(torch.square(measured_estimates - measured_x), dim=0))
    naive_rmse = torch.sqrt(torch.mean(torch.square(measured_mid - measured_x), dim=0))
    state_improvement = 1.0 - pf_rmse / naive_rmse
    score_improvement = torch.sum(
        scores[burn_steps:] - naive_scores[burn_steps:], dim=0
    ) / event_counts
    z90 = measured_x.new_tensor(1.6448536269514722)
    posterior_sd = torch.sqrt(torch.clamp_min(measured_variances, 0.0))
    coverage = torch.mean(
        (
            (measured_x >= measured_estimates - z90 * posterior_sd)
            & (measured_x <= measured_estimates + z90 * posterior_sd)
        ).to(torch.float32),
        dim=0,
    )
    state_mean, state_se, state_lower = _normal_lower_bound(
        state_improvement, float(evaluation["per_metric_alpha"])
    )
    score_mean, score_se, score_lower = _normal_lower_bound(
        score_improvement, float(evaluation["per_metric_alpha"])
    )
    coverage_mean, coverage_se, coverage_lower, coverage_upper = _normal_equivalence_interval(
        coverage, float(evaluation["calibration_alpha"])
    )

    signed_jump = tables.mark_dy.to(torch.float32) * (delta / 2.0)
    baseline_drift = torch.sum(tables.probabilities * signed_jump, dim=-1)
    down_moment = torch.sum(
        tables.correction_weights
        * tables.jump_abs_price
        * (tables.direction_sign < 0).to(torch.float32).unsqueeze(0),
        dim=-1,
    )
    up_moment = torch.sum(
        tables.correction_weights
        * tables.jump_abs_price
        * (tables.direction_sign > 0).to(torch.float32).unsqueeze(0),
        dim=-1,
    )
    baseline_drift_error = float(torch.max(torch.abs(baseline_drift)))
    corrective_error = float(
        torch.max(torch.abs(torch.cat((down_moment - 1.0, up_moment - 1.0))))
    )
    nonzero = tables.probabilities > 0.0
    delta_bid_twice = tables.mark_dy - tables.mark_dd
    delta_ask_twice = tables.mark_dy + tables.mark_dd
    arithmetic_valid = bool(
        torch.all(torch.remainder(delta_bid_twice[nonzero], 2) == 0)
        and torch.all(torch.remainder(delta_ask_twice[nonzero], 2) == 0)
    )
    positive_spread = bool(
        torch.all(spread_path >= 1)
        and torch.all(spread_path <= int(model["spread_state_count"]))
    )
    all_finite = all(
        bool(torch.all(torch.isfinite(value)))
        for value in (x_path, estimates, variances, scores, naive_scores, state_improvement, coverage)
    )
    elapsed = time.perf_counter() - started
    coverage_target = float(evaluation["posterior_interval_nominal_coverage"])
    coverage_margin = float(evaluation["posterior_coverage_equivalence_margin"])
    acceptance = {
        "state_superiority": state_lower > float(evaluation["state_minimum_effect"]),
        "log_score_superiority": score_lower
        > float(evaluation["log_score_minimum_effect_nat_per_event"]),
        "calibration_equivalence": coverage_lower > coverage_target - coverage_margin
        and coverage_upper < coverage_target + coverage_margin,
        "mark_arithmetic": arithmetic_valid,
        "positive_spread": positive_spread,
        "drift_constraint": baseline_drift_error
        < float(evaluation["baseline_drift_absolute_error_max"])
        and corrective_error < float(evaluation["corrective_moment_absolute_error_max"]),
        "deterministic_replay": deterministic_replay,
        "maximum_event_probability": maximum_probability
        < float(evaluation["maximum_event_probability"]),
        "all_values_finite": all_finite,
        "posterior_variance_positive": bool(torch.all(measured_variances > 0.0)),
        "wall_time_within_limit": elapsed < float(evaluation["maximum_wall_seconds"]),
    }
    metrics = {
        "session_count": int(events.shape[1]),
        "measured_steps_per_session": int(measured_x.shape[0]),
        "measured_events_total": int(event_counts.sum()),
        "state_improvement_mean": state_mean,
        "state_improvement_standard_error": state_se,
        "state_improvement_bonferroni_lower_bound": state_lower,
        "log_score_improvement_nat_per_event_mean": score_mean,
        "log_score_improvement_standard_error": score_se,
        "log_score_improvement_bonferroni_lower_bound": score_lower,
        "posterior_90_coverage_mean": coverage_mean,
        "posterior_90_coverage_standard_error": coverage_se,
        "posterior_90_coverage_equivalence_lower": coverage_lower,
        "posterior_90_coverage_equivalence_upper": coverage_upper,
        "pf_rmse_mean": float(pf_rmse.mean()),
        "naive_rmse_mean": float(naive_rmse.mean()),
        "baseline_drift_absolute_error": baseline_drift_error,
        "corrective_moment_absolute_error": corrective_error,
        "maximum_event_probability": maximum_probability,
        "evaluation_elapsed_seconds": elapsed,
        "deterministic_replay": deterministic_replay,
        "filter_digest_sha256": first_digest,
    }

    session_rows = [
        {
            "session": index,
            "event_count": int(event_counts[index]),
            "pf_rmse": float(pf_rmse[index]),
            "naive_rmse": float(naive_rmse[index]),
            "state_improvement": float(state_improvement[index]),
            "log_score_improvement_nat_per_event": float(score_improvement[index]),
            "posterior_90_coverage": float(coverage[index]),
        }
        for index in range(events.shape[1])
    ]
    write_csv(
        run_directory / "metrics" / "session_metrics.csv",
        list(session_rows[0]),
        session_rows,
    )
    inference_rows = [
        {
            "metric": "state_improvement",
            "mean": state_mean,
            "standard_error": state_se,
            "lower_bound": state_lower,
            "upper_bound": "",
            "required_lower": float(evaluation["state_minimum_effect"]),
            "required_upper": "",
            "alpha": float(evaluation["per_metric_alpha"]),
            "passed": acceptance["state_superiority"],
        },
        {
            "metric": "log_score_improvement_nat_per_event",
            "mean": score_mean,
            "standard_error": score_se,
            "lower_bound": score_lower,
            "upper_bound": "",
            "required_lower": float(evaluation["log_score_minimum_effect_nat_per_event"]),
            "required_upper": "",
            "alpha": float(evaluation["per_metric_alpha"]),
            "passed": acceptance["log_score_superiority"],
        },
        {
            "metric": "posterior_90_coverage",
            "mean": coverage_mean,
            "standard_error": coverage_se,
            "lower_bound": coverage_lower,
            "upper_bound": coverage_upper,
            "required_lower": coverage_target - coverage_margin,
            "required_upper": coverage_target + coverage_margin,
            "alpha": float(evaluation["calibration_alpha"]),
            "passed": acceptance["calibration_equivalence"],
        },
    ]
    write_csv(run_directory / "tables" / "inference.csv", list(inference_rows[0]), inference_rows)
    active_rows: list[dict[str, Any]] = []
    for spread_index in range(tables.probabilities.shape[0]):
        ids = torch.nonzero(tables.probabilities[spread_index] > 0.0, as_tuple=False).squeeze(-1)
        for mark in ids:
            active_rows.append(
                {
                    "spread_ticks": spread_index + 1,
                    "mark_id": int(mark),
                    "direction": int(tables.direction_sign[mark]),
                    "family": int(tables.family_sign[mark]),
                    "midpoint_bucket": int(tables.midpoint_bucket[mark]),
                    "spread_bucket": int(tables.spread_bucket[mark]),
                    "delta_y_half_ticks": int(tables.mark_dy[spread_index, mark]),
                    "delta_spread_ticks": int(tables.mark_dd[spread_index, mark]),
                    "baseline_probability": float(tables.probabilities[spread_index, mark]),
                    "correction_weight": float(tables.correction_weights[spread_index, mark]),
                }
            )
    write_csv(run_directory / "tables" / "mark_table.csv", list(active_rows[0]), active_rows)
    atomic_write_json(
        run_directory / "metrics" / "replay.json",
        {
            "schema_version": "marked-filter-replay-v1",
            "first_digest_sha256": first_digest,
            "second_digest_sha256": second_digest,
            "exact_tensor_equality": deterministic_replay,
            "seed_streams": {
                "market": 0,
                "particle_initialization_and_transition": 1,
                "systematic_resampling": "same particle generator after transition draws",
            },
        },
    )
    return MarkedEvaluationResult(
        metrics=metrics,
        acceptance=acceptance,
        derived_parameters={
            "mark_count": MARK_COUNT,
            "active_marks_per_spread": int(tables.active_ids.shape[1]),
            "particle_count": int(spec.values["numerics"]["particle_count"]),
            "cuda_device": torch.cuda.get_device_name(device),
        },
        log_lines=(
            f"state improvement mean={state_mean:.6g}; lower={state_lower:.6g}",
            f"log-score improvement={score_mean:.6g}; lower={score_lower:.6g}",
            f"coverage={coverage_mean:.6g}; interval=[{coverage_lower:.6g},{coverage_upper:.6g}]",
            f"drift_error={baseline_drift_error:.6g}; correction_error={corrective_error:.6g}",
            f"replay={deterministic_replay}; digest={first_digest}",
        ),
    )
