from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ot_micromr.artifacts import atomic_write_json, sha256_file, write_csv
from ot_micromr.config import RunSpec
from ot_micromr.efficient_price import systematic_resample
from ot_micromr.empirical_filter import _causal_ewma, _dawson_threshold_margin
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


def _union_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(dict.fromkeys(key for row in rows for key in row))


def _save_torch_artifact(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


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


@dataclass(frozen=True, slots=True)
class EmpiricalMarkedDay:
    date: str
    timestamps_ms: torch.Tensor
    bid_ticks: torch.Tensor
    ask_ticks: torch.Tensor
    reset: torch.Tensor
    dt_seconds: torch.Tensor
    valid_interval: torch.Tensor
    previous_spread_ticks: torch.Tensor
    current_spread_ticks: torch.Tensor
    previous_spread_bucket: torch.Tensor
    mark_id: torch.Tensor
    delta_y: torch.Tensor
    delta_d: torch.Tensor
    prior_mid_price: torch.Tensor
    proxy_price: torch.Tensor
    proxy_gap: torch.Tensor
    spot_reference: torch.Tensor | None
    spot_reference_timestamp_ms: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class EmpiricalModel:
    name: str
    probabilities: torch.Tensor
    correction_weights: torch.Tensor
    correction_sum_down: torch.Tensor
    correction_sum_up: torch.Tensor
    baseline_rates: torch.Tensor
    alpha: torch.Tensor
    sigma_x: float
    s_g: float
    fit_initial_nll: float
    fit_final_nll: float
    baseline_drift: torch.Tensor
    parameter_digest: str


@dataclass(frozen=True, slots=True)
class DayFilterOutput:
    estimate: torch.Tensor
    variance: torch.Tensor
    predictive_score: torch.Tensor
    expected_rescaling: torch.Tensor
    digest: str


def _find_passed_synthetic_dependency(spec: RunSpec) -> tuple[str, str]:
    config_hash = str(spec.values["inputs"]["synthetic_dependency_config_sha256"])
    root = spec.repository_root / "outputs" / "FILTER-MARK-SYN-001"
    matches: list[tuple[str, str]] = []
    if root.is_dir():
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            manifest_path = directory / "manifest.json"
            summary_path = directory / "metrics" / "summary.json"
            if not manifest_path.is_file() or not summary_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                manifest.get("source", {}).get("config_sha256") == config_hash
                and summary.get("status") == "passed"
                and summary.get("acceptance_passed") is True
            ):
                matches.append((directory.name, sha256_file(manifest_path)))
    if not matches:
        raise ExperimentError("no passed FILTER-MARK-SYN-001 run matches the frozen config")
    return matches[0]


def _load_verified_p6_payloads(
    spec: RunSpec,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    inputs = spec.values["inputs"]
    dependency = spec.repository_root / "outputs" / str(inputs["p6_dependency_run"])
    state_root = dependency / "state"
    extraction_manifest_path = state_root / "extraction_manifest.json"
    manifest = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
    rows = list(manifest["assets"])
    swap: dict[str, Mapping[str, Any]] = {}
    spot: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        path = state_root / str(row["relative_path"])
        if sha256_file(path) != row["sha256"]:
            raise ExperimentError(f"processed P6 tensor hash mismatch: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        date = str(row["date"])
        if row["instrument_type"] == "SWAP":
            swap[date] = payload
        elif row["instrument_type"] == "SPOT":
            spot[date] = payload
    expected = set(str(value) for value in spec.values["evaluation"]["ordered_dates"])
    if set(swap) != expected:
        raise ExperimentError("P6 processed swap dates disagree with P6M freeze")
    return swap, spot, rows


def _causal_spot_reference(
    swap_timestamps: torch.Tensor,
    swap_mid: torch.Tensor,
    spot_payload: Mapping[str, Any],
    tau_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    spot_timestamps = spot_payload["timestamps_ms"].to(swap_timestamps.device)
    spot_mid = (
        spot_payload["bid_ticks"].to(swap_timestamps.device)
        + spot_payload["ask_ticks"].to(swap_timestamps.device)
    ).to(torch.float32) * 0.05
    indices = torch.searchsorted(spot_timestamps, swap_timestamps, right=True) - 1
    valid = indices >= 0
    safe = torch.clamp_min(indices, 0)
    selected_timestamps = spot_timestamps[safe]
    raw_basis = swap_mid - spot_mid[safe]
    causal_basis = _causal_ewma(raw_basis, swap_timestamps, tau_seconds)
    reference = torch.where(valid, spot_mid[safe] + causal_basis, swap_mid)
    selected_timestamps = torch.where(valid, selected_timestamps, swap_timestamps)
    future = int(torch.count_nonzero(valid & (selected_timestamps > swap_timestamps)))
    return reference, selected_timestamps, future


def _prepare_empirical_marked_days(
    spec: RunSpec,
    swap_payloads: Mapping[str, Mapping[str, Any]],
    spot_payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, EmpiricalMarkedDay], int]:
    device = torch.device(str(spec.values["numerics"]["compute_device"]))
    tau = float(spec.values["model"]["gap_proxy_time_constant_seconds"])
    spot_dates = set(str(value) for value in spec.values["evaluation"]["spot_dates"])
    days: dict[str, EmpiricalMarkedDay] = {}
    future_accesses = 0
    for date in spec.values["evaluation"]["ordered_dates"]:
        payload = swap_payloads[str(date)]
        timestamps = payload["timestamps_ms"].to(device)
        bids = payload["bid_ticks"].to(device)
        asks = payload["ask_ticks"].to(device)
        reset = payload["snapshot_reset"].to(device)
        dt = (timestamps[1:] - timestamps[:-1]).to(torch.float32) / 1000.0
        previous_spread = asks[:-1] - bids[:-1]
        current_spread = asks[1:] - bids[1:]
        delta_bid = bids[1:] - bids[:-1]
        delta_ask = asks[1:] - asks[:-1]
        delta_y = delta_bid + delta_ask
        delta_d = delta_ask - delta_bid
        marks = encode_mark(delta_bid, delta_ask)
        spread_bucket = previous_spread_bucket(
            previous_spread, int(spec.values["model"]["previous_spread_exact_bucket_max"])
        )
        valid = (~reset[1:]) & (dt > 0.0) & (previous_spread > 0) & (current_spread > 0)
        mid = (bids + asks).to(torch.float32) * 0.05
        proxy = _causal_ewma(mid, timestamps, tau)
        spot_reference: torch.Tensor | None = None
        spot_timestamp: torch.Tensor | None = None
        if str(date) in spot_dates:
            spot_reference, spot_timestamp, future = _causal_spot_reference(
                timestamps, mid, spot_payloads[str(date)], tau
            )
            future_accesses += future
        days[str(date)] = EmpiricalMarkedDay(
            date=str(date),
            timestamps_ms=timestamps,
            bid_ticks=bids,
            ask_ticks=asks,
            reset=reset,
            dt_seconds=dt,
            valid_interval=valid,
            previous_spread_ticks=previous_spread,
            current_spread_ticks=current_spread,
            previous_spread_bucket=spread_bucket,
            mark_id=marks,
            delta_y=delta_y,
            delta_d=delta_d,
            prior_mid_price=mid[:-1],
            proxy_price=proxy,
            proxy_gap=mid[:-1] - proxy[:-1],
            spot_reference=spot_reference,
            spot_reference_timestamp_ms=spot_timestamp,
        )
    return days, future_accesses


def _partner_mark_ids(device: torch.device) -> torch.Tensor:
    direction_sign, family_sign, midpoint_bucket, spread_bucket = mark_metadata(device)
    direction_index = direction_sign + 1
    partner_direction = 2 - direction_index
    family_index = family_sign + 1
    return (
        ((partner_direction * FAMILY_COUNT + family_index) * MAGNITUDE_BUCKET_COUNT + midpoint_bucket)
        * MAGNITUDE_BUCKET_COUNT
        + spread_bucket
    )


def _stack_train_intervals(
    days: Mapping[str, EmpiricalMarkedDay], dates: Sequence[str]
) -> tuple[torch.Tensor, ...]:
    fields = (
        "proxy_gap",
        "previous_spread_bucket",
        "mark_id",
        "dt_seconds",
        "delta_y",
        "delta_d",
        "previous_spread_ticks",
        "current_spread_ticks",
    )
    outputs: list[torch.Tensor] = []
    for field in fields:
        values = []
        for date in dates:
            day = days[date]
            tensor = getattr(day, field)
            values.append(tensor[day.valid_interval])
        outputs.append(torch.cat(values))
    return tuple(outputs)


def _empirical_probability_tables(
    train: tuple[torch.Tensor, ...],
    beta: float,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gap, spread, mark, dt, delta_y, _, previous_spread, current_spread = train
    del gap, dt
    flat = spread * MARK_COUNT + mark
    counts = torch.bincount(flat, minlength=8 * MARK_COUNT).to(torch.float64).reshape(8, MARK_COUNT)
    jump_sum = torch.zeros_like(counts)
    jump_sum.reshape(-1).scatter_add_(0, flat, torch.abs(delta_y).to(torch.float64) * 0.05)
    raw_probabilities = (counts + beta) / (
        counts.sum(dim=-1, keepdim=True) + beta * MARK_COUNT
    )
    direction_sign, family_sign, midpoint_bucket, spread_bucket = mark_metadata(counts.device)
    partner = _partner_mark_ids(counts.device)
    nonzero_direction = (direction_sign != 0).unsqueeze(0)
    symmetric_counts = torch.where(
        nonzero_direction,
        0.5 * (counts + counts[:, partner]),
        counts,
    )
    probabilities = (symmetric_counts + beta) / (
        symmetric_counts.sum(dim=-1, keepdim=True) + beta * MARK_COUNT
    )

    fallback_half_ticks = counts.new_tensor((0.0, 1.0, 2.5, 5.5, 11.5, 23.5, 47.5, 95.5, 192.0))
    fallback_jump = fallback_half_ticks[midpoint_bucket].unsqueeze(0) * 0.05
    observed_jump = torch.where(counts > 0.0, jump_sum / torch.clamp_min(counts, 1.0), fallback_jump)
    symmetric_jump = torch.where(
        nonzero_direction,
        0.5 * (observed_jump + observed_jump[:, partner]),
        observed_jump,
    )
    if variant == "unconstrained":
        model_probabilities = raw_probabilities
        jump_abs = observed_jump
    else:
        model_probabilities = probabilities
        jump_abs = symmetric_jump

    eligible = direction_sign.ne(0).unsqueeze(0).expand(8, -1).clone()
    if variant == "no_multi_tick":
        eligible &= ~(
            family_sign.eq(0).unsqueeze(0) & midpoint_bucket.gt(2).unsqueeze(0)
        )
    elif variant == "no_multi_spread":
        spread_states = torch.arange(1, 9, device=counts.device).unsqueeze(1)
        eligible &= spread_states <= 2
        widening = family_sign.gt(0).unsqueeze(0)
        widening_safe = (spread_states == 1) & spread_bucket.eq(1).unsqueeze(0)
        eligible &= ~widening | widening_safe
    elif variant not in {"full", "unconstrained"}:
        raise ValueError(f"unsupported correction variant: {variant}")

    down = (direction_sign < 0).to(torch.float64).unsqueeze(0) * eligible
    up = (direction_sign > 0).to(torch.float64).unsqueeze(0) * eligible
    down_moment = torch.sum(model_probabilities * jump_abs * down, dim=-1, keepdim=True)
    up_moment = torch.sum(model_probabilities * jump_abs * up, dim=-1, keepdim=True)
    correction = model_probabilities * (
        down / torch.clamp_min(down_moment, 1e-12)
        + up / torch.clamp_min(up_moment, 1e-12)
    )
    correction_down = torch.sum(correction * (direction_sign < 0).unsqueeze(0), dim=-1)
    correction_up = torch.sum(correction * (direction_sign > 0).unsqueeze(0), dim=-1)
    signed_jump = symmetric_jump * direction_sign.to(torch.float64).unsqueeze(0)
    baseline_drift = torch.sum(model_probabilities * signed_jump, dim=-1)
    return (
        model_probabilities.to(torch.float32),
        correction.to(torch.float32),
        correction_down.to(torch.float32),
        correction_up.to(torch.float32),
        raw_probabilities.to(torch.float32),
        baseline_drift.to(torch.float32),
    )


def _initial_baseline_rates(train: tuple[torch.Tensor, ...]) -> torch.Tensor:
    _, spread, _, dt, *_ = train
    counts = torch.bincount(spread, minlength=8).to(torch.float64)
    exposure = torch.zeros(8, device=spread.device, dtype=torch.float64)
    exposure.scatter_add_(0, spread, dt.to(torch.float64))
    global_rate = counts.sum() / torch.clamp_min(exposure.sum(), 1e-6)
    return torch.where(exposure > 0.1, counts / exposure, global_rate).clamp_min(1e-4)


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    return torch.where(values > 20.0, values, torch.log(torch.expm1(values)))


def _fit_empirical_model(
    spec: RunSpec,
    train: tuple[torch.Tensor, ...],
    variant: str,
    sigma_x: float,
    s_g: float,
) -> EmpiricalModel:
    gap, spread, mark, dt, *_ = train
    beta = float(spec.values["model"]["dirichlet_smoothing_beta"])
    probabilities, correction, correction_down, correction_up, _, baseline_drift = (
        _empirical_probability_tables(train, beta, variant)
    )
    initial_rates = _initial_baseline_rates(train).to(torch.float32)
    initial = torch.cat((initial_rates, initial_rates.new_tensor((0.1,))))
    raw = torch.nn.Parameter(_inverse_softplus(initial))
    batch_size = int(spec.values["numerics"]["fit_batch_events"])
    learning_rate = float(spec.values["numerics"]["optimizer_learning_rate"])
    optimizer = torch.optim.Adam((raw,), lr=learning_rate)

    def loss_function(
        raw_parameters: torch.Tensor,
        batch_gap: torch.Tensor,
        batch_spread: torch.Tensor,
        batch_mark: torch.Tensor,
        batch_dt: torch.Tensor,
        probabilities_arg: torch.Tensor,
        correction_arg: torch.Tensor,
        correction_down_arg: torch.Tensor,
        correction_up_arg: torch.Tensor,
    ) -> torch.Tensor:
        parameters = torch.nn.functional.softplus(raw_parameters) + 1e-7
        rates = parameters[:8]
        alpha = parameters[8]
        directions = mark_metadata(batch_gap.device)[0][batch_mark]
        directional_gap = torch.where(
            directions > 0,
            torch.clamp_min(-batch_gap, 0.0),
            torch.where(directions < 0, torch.clamp_min(batch_gap, 0.0), torch.zeros_like(batch_gap)),
        )
        selected = rates[batch_spread] * probabilities_arg[batch_spread, batch_mark]
        selected = selected + alpha * correction_arg[batch_spread, batch_mark] * directional_gap
        total = rates[batch_spread] + alpha * torch.where(
            batch_gap >= 0.0,
            batch_gap * correction_down_arg[batch_spread],
            -batch_gap * correction_up_arg[batch_spread],
        )
        return torch.mean(total * batch_dt - torch.log(torch.clamp_min(selected, 1e-30)))

    loss = torch.compile(
        loss_function,
        mode=str(spec.values["numerics"]["compile_mode"]),
        fullgraph=True,
        dynamic=True,
    )
    n = gap.numel()

    def batch_at(step: int) -> tuple[torch.Tensor, ...]:
        if n <= batch_size:
            return gap, spread, mark, dt
        start = (step * batch_size) % n
        index = torch.remainder(
            torch.arange(batch_size, device=gap.device, dtype=torch.int64) + start, n
        )
        return gap[index], spread[index], mark[index], dt[index]

    first_batch = batch_at(0)
    initial_loss = float(
        loss(raw, *first_batch, probabilities, correction, correction_down, correction_up).detach()
    )
    for step in range(int(spec.values["numerics"]["optimizer_steps"])):
        optimizer.zero_grad(set_to_none=True)
        batch = batch_at(step)
        objective = loss(
            raw, *batch, probabilities, correction, correction_down, correction_up
        )
        objective.backward()
        optimizer.step()
    parameters = torch.nn.functional.softplus(raw.detach()) + 1e-7
    final_loss = float(
        loss(raw, gap, spread, mark, dt, probabilities, correction, correction_down, correction_up).detach()
    )
    parameter_digest = _tensor_digest(
        (probabilities, correction, parameters, baseline_drift)
    )
    return EmpiricalModel(
        name=variant,
        probabilities=probabilities,
        correction_weights=correction,
        correction_sum_down=correction_down,
        correction_sum_up=correction_up,
        baseline_rates=parameters[:8],
        alpha=parameters[8],
        sigma_x=sigma_x,
        s_g=s_g,
        fit_initial_nll=initial_loss,
        fit_final_nll=final_loss,
        baseline_drift=baseline_drift,
        parameter_digest=parameter_digest,
    )


def _fit_residual_model(
    train: tuple[torch.Tensor, ...], beta: float, sigma_x: float, s_g: float
) -> EmpiricalModel:
    probabilities, _, _, _, raw_probabilities, _ = _empirical_probability_tables(
        train, beta, "full"
    )
    del probabilities
    rates = _initial_baseline_rates(train).to(torch.float32)
    zeros = torch.zeros_like(rates)
    direction = mark_metadata(rates.device)[0]
    _, spread, mark, dt, *_ = train
    score = torch.log(torch.clamp_min(rates[spread] * raw_probabilities[spread, mark], 1e-30)) - rates[spread] * dt
    return EmpiricalModel(
        name="residual",
        probabilities=raw_probabilities,
        correction_weights=torch.zeros_like(raw_probabilities),
        correction_sum_down=zeros,
        correction_sum_up=zeros,
        baseline_rates=rates,
        alpha=rates.new_zeros(()),
        sigma_x=sigma_x,
        s_g=s_g,
        fit_initial_nll=float(-score.mean()),
        fit_final_nll=float(-score.mean()),
        baseline_drift=torch.sum(
            raw_probabilities
            * direction.to(torch.float32).unsqueeze(0), dim=-1
        ),
        parameter_digest=_tensor_digest((raw_probabilities, rates)),
    )


def _train_reduced_parameters(
    days: Mapping[str, EmpiricalMarkedDay], train_dates: Sequence[str]
) -> tuple[float, float]:
    gaps: list[torch.Tensor] = []
    increments: list[torch.Tensor] = []
    for date in train_dates:
        day = days[date]
        start = int(day.timestamps_ms[0])
        stop = int(day.timestamps_ms[-1])
        grid = torch.arange(start, stop + 1, 1000, device=day.timestamps_ms.device)
        indices = torch.searchsorted(day.timestamps_ms, grid, right=True) - 1
        indices = indices[indices >= 0]
        mid = (day.bid_ticks[indices] + day.ask_ticks[indices]).to(torch.float32) * 0.05
        proxy = day.proxy_price[indices]
        gaps.append(mid - proxy)
        increments.append(proxy[1:] - proxy[:-1])
    gap = torch.cat(gaps).to(torch.float64)
    increment = torch.cat(increments).to(torch.float64)
    s_g = float(torch.std(gap, unbiased=False).clamp_min(0.05))
    sigma_x = float(torch.sqrt(torch.mean(torch.square(increment))).clamp_min(1e-4))
    return sigma_x, s_g


def _valid_segments(valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padded = torch.cat(
        (
            torch.zeros(1, device=valid.device, dtype=torch.bool),
            valid,
            torch.zeros(1, device=valid.device, dtype=torch.bool),
        )
    )
    changes = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)
    starts = torch.nonzero(changes == 1, as_tuple=False).squeeze(-1)
    ends = torch.nonzero(changes == -1, as_tuple=False).squeeze(-1)
    return starts, ends - starts


def _make_empirical_particle_chunk(spec: RunSpec):
    chunk_events = int(spec.values["numerics"]["particle_chunk_events"])

    def particle_chunk(
        particles: torch.Tensor,
        log_weights: torch.Tensor,
        prior_mid: torch.Tensor,
        prior_spread: torch.Tensor,
        events: torch.Tensor,
        dt: torch.Tensor,
        normals: torch.Tensor,
        offsets: torch.Tensor,
        probabilities: torch.Tensor,
        correction: torch.Tensor,
        correction_down: torch.Tensor,
        correction_up: torch.Tensor,
        rates: torch.Tensor,
        alpha: torch.Tensor,
        sigma_x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        increments = sigma_x * torch.sqrt(dt).unsqueeze(-1) * normals
        particle_ends = particles.unsqueeze(0) + torch.cumsum(increments, dim=0)
        particle_starts = torch.cat((particles.unsqueeze(0), particle_ends[:-1]), dim=0)
        gap = prior_mid.unsqueeze(-1) - particle_starts
        safe_events = torch.clamp_min(events, 0)
        direction = mark_metadata(events.device)[0][safe_events]
        directional_gap = torch.where(
            direction.unsqueeze(-1) > 0,
            torch.clamp_min(-gap, 0.0),
            torch.where(direction.unsqueeze(-1) < 0, torch.clamp_min(gap, 0.0), torch.zeros_like(gap)),
        )
        selected = rates[prior_spread] * probabilities[prior_spread, safe_events]
        selected = selected.unsqueeze(-1) + alpha * correction[
            prior_spread, safe_events
        ].unsqueeze(-1) * directional_gap
        total = rates[prior_spread].unsqueeze(-1) + alpha * torch.where(
            gap >= 0.0,
            gap * correction_down[prior_spread].unsqueeze(-1),
            -gap * correction_up[prior_spread].unsqueeze(-1),
        )
        interval_score = torch.where(
            events.unsqueeze(-1) >= 0,
            torch.log(torch.clamp_min(selected, 1e-30)) - total * dt.unsqueeze(-1),
            -total * dt.unsqueeze(-1),
        )
        cumulative = log_weights.unsqueeze(0) + torch.cumsum(interval_score, dim=0)
        normalizers = torch.logsumexp(cumulative, dim=-1)
        previous_normalizers = torch.cat(
            (normalizers.new_zeros((1, normalizers.shape[1])), normalizers[:-1]), dim=0
        )
        predictive = normalizers - previous_normalizers
        normalized = cumulative - normalizers.unsqueeze(-1)
        weights = torch.exp(normalized)
        estimates = torch.sum(weights * particle_ends, dim=-1)
        variances = torch.sum(
            weights * torch.square(particle_ends - estimates.unsqueeze(-1)), dim=-1
        )
        prior_cumulative = torch.cat((log_weights.unsqueeze(0), cumulative[:-1]), dim=0)
        prior_normalized = prior_cumulative - torch.logsumexp(prior_cumulative, dim=-1, keepdim=True)
        expected_total = torch.sum(torch.exp(prior_normalized) * total, dim=-1)
        particles, log_weights = systematic_resample(
            particle_ends[-1], normalized[-1], offsets
        )
        return particles, log_weights, estimates, variances, predictive, expected_total * dt

    return torch.compile(
        particle_chunk,
        mode=str(spec.values["numerics"]["compile_mode"]),
        fullgraph=True,
    )


def _filter_empirical_day(
    spec: RunSpec,
    day: EmpiricalMarkedDay,
    model: EmpiricalModel,
    seed: int,
    particle_chunk,
) -> DayFilterOutput:
    device = day.timestamps_ms.device
    particle_count = int(spec.values["numerics"]["particle_count"])
    chunk_events = int(spec.values["numerics"]["particle_chunk_events"])
    group_size = 64
    starts, lengths = _valid_segments(day.valid_interval)
    mid = (day.bid_ticks + day.ask_ticks).to(torch.float32) * 0.05
    estimate = mid.clone()
    variance = torch.full_like(mid, model.s_g * model.s_g)
    score = torch.zeros_like(day.dt_seconds)
    rescaling = torch.zeros_like(day.dt_seconds)
    generator = torch.Generator(device=device).manual_seed(seed)
    arange_chunk = torch.arange(chunk_events, device=device).unsqueeze(1)
    for group_start in range(0, starts.numel(), group_size):
        actual = min(group_size, starts.numel() - group_start)
        group_starts = torch.zeros(group_size, device=device, dtype=torch.int64)
        group_lengths = torch.zeros_like(group_starts)
        group_starts[:actual] = starts[group_start : group_start + actual]
        group_lengths[:actual] = lengths[group_start : group_start + actual]
        particles = day.prior_mid_price[group_starts].unsqueeze(-1) + model.s_g * torch.randn(
            (group_size, particle_count), device=device, dtype=torch.float32, generator=generator
        )
        log_weights = torch.full_like(particles, -math.log(particle_count))
        maximum_length = int(group_lengths.max())
        for offset in range(0, maximum_length, chunk_events):
            interval_index = group_starts.unsqueeze(0) + offset + arange_chunk
            valid = arange_chunk < torch.clamp_min(group_lengths - offset, 0).unsqueeze(0)
            safe = torch.clamp(interval_index, 0, day.dt_seconds.numel() - 1)
            dt = torch.where(valid, day.dt_seconds[safe], torch.zeros_like(day.dt_seconds[safe]))
            events = torch.where(valid, day.mark_id[safe], torch.full_like(safe, -1))
            prior_mid = day.prior_mid_price[safe]
            prior_spread = day.previous_spread_bucket[safe]
            normals = torch.randn(
                (chunk_events, group_size, particle_count),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            offsets = torch.rand(
                (group_size,), device=device, dtype=torch.float32, generator=generator
            )
            (
                particles,
                log_weights,
                chunk_estimate,
                chunk_variance,
                chunk_score,
                chunk_rescaling,
            ) = particle_chunk(
                particles,
                log_weights,
                prior_mid,
                prior_spread,
                events,
                dt,
                normals,
                offsets,
                model.probabilities,
                model.correction_weights,
                model.correction_sum_down,
                model.correction_sum_up,
                model.baseline_rates,
                model.alpha,
                prior_mid.new_tensor(model.sigma_x),
            )
            target = safe + 1
            estimate[target[valid]] = chunk_estimate[valid]
            variance[target[valid]] = chunk_variance[valid]
            score[safe[valid]] = chunk_score[valid]
            rescaling[safe[valid]] = chunk_rescaling[valid]
            particles = particles.clone()
            log_weights = log_weights.clone()
    return DayFilterOutput(
        estimate=estimate,
        variance=variance,
        predictive_score=score,
        expected_rescaling=rescaling,
        digest=_tensor_digest((estimate, variance, score, rescaling)),
    )


def _residual_scores(day: EmpiricalMarkedDay, model: EmpiricalModel) -> torch.Tensor:
    spread = day.previous_spread_bucket
    mark = day.mark_id
    rates = model.baseline_rates[spread]
    return torch.log(torch.clamp_min(rates * model.probabilities[spread, mark], 1e-30)) - rates * day.dt_seconds


def _block_reduce(
    day: EmpiricalMarkedDay,
    values: torch.Tensor,
    *,
    reduction: str,
    block_minutes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    timestamps = day.timestamps_ms[1:]
    valid = day.valid_interval & torch.isfinite(values)
    block_ms = block_minutes * 60 * 1000
    day_start = torch.div(timestamps.min(), 86_400_000, rounding_mode="floor") * 86_400_000
    block = torch.div(timestamps - day_start, block_ms, rounding_mode="floor")
    count = torch.bincount(block[valid], minlength=48).to(torch.float64)
    if reduction == "mean":
        total = torch.zeros(48, device=values.device, dtype=torch.float64)
        total.scatter_add_(0, block[valid], values[valid].to(torch.float64))
        reduced = total / torch.clamp_min(count, 1.0)
    elif reduction == "std":
        total = torch.zeros(48, device=values.device, dtype=torch.float64)
        square = torch.zeros_like(total)
        total.scatter_add_(0, block[valid], values[valid].to(torch.float64))
        square.scatter_add_(0, block[valid], torch.square(values[valid].to(torch.float64)))
        mean = total / torch.clamp_min(count, 1.0)
        reduced = torch.sqrt(
            torch.clamp_min((square - count * torch.square(mean)) / torch.clamp_min(count - 1.0, 1.0), 0.0)
        )
    elif reduction == "median":
        selected_values = values[valid]
        selected_blocks = block[valid]
        value_order = torch.argsort(selected_values)
        block_order = torch.argsort(selected_blocks[value_order], stable=True)
        ordered_values = selected_values[value_order[block_order]].to(torch.float64)
        offsets = torch.cumsum(count.to(torch.int64), dim=0) - count.to(torch.int64)
        lower_index = offsets + torch.clamp_min(count.to(torch.int64) - 1, 0) // 2
        upper_index = offsets + count.to(torch.int64) // 2
        safe_last = max(ordered_values.numel() - 1, 0)
        reduced = 0.5 * (
            ordered_values[torch.clamp(lower_index, 0, safe_last)]
            + ordered_values[torch.clamp(upper_index, 0, safe_last)]
        )
    else:
        raise ValueError(f"unsupported reduction: {reduction}")
    eligible = count > (1.0 if reduction == "std" else 0.0)
    return reduced[eligible], count[eligible], torch.nonzero(eligible, as_tuple=False).squeeze(-1)


def _student_t_cdf(values: torch.Tensor, degrees_of_freedom: int) -> torch.Tensor:
    values = values.to(torch.float64)
    magnitude = torch.clamp(torch.abs(values), max=12.0)
    nodes = torch.linspace(0.0, 1.0, 8193, device=values.device, dtype=torch.float64)
    x = magnitude.unsqueeze(-1) * nodes
    df = values.new_tensor(float(degrees_of_freedom))
    log_constant = torch.lgamma((df + 1.0) / 2.0) - torch.lgamma(df / 2.0) - 0.5 * (
        torch.log(df) + math.log(math.pi)
    )
    density = torch.exp(log_constant - 0.5 * (df + 1.0) * torch.log1p(torch.square(x) / df))
    integral = magnitude * torch.trapezoid(density, nodes, dim=-1)
    return torch.where(values >= 0.0, 0.5 + integral, 0.5 - integral).clamp(0.0, 1.0)


def _student_t_critical(probability: float, degrees_of_freedom: int, device: torch.device) -> float:
    target = torch.tensor(probability, device=device, dtype=torch.float64)
    lower = torch.zeros((), device=device, dtype=torch.float64)
    upper = torch.full((), 12.0, device=device, dtype=torch.float64)
    for _ in range(48):
        midpoint = 0.5 * (lower + upper)
        cdf = _student_t_cdf(midpoint.unsqueeze(0), degrees_of_freedom)[0]
        lower = torch.where(cdf < target, midpoint, lower)
        upper = torch.where(cdf >= target, midpoint, upper)
    return float(0.5 * (lower + upper))


def _superiority_row(
    values: torch.Tensor, minimum: float, alpha: float, metric: str
) -> dict[str, Any]:
    sample = values.to(torch.float64)
    mean = sample.mean()
    se = sample.std(unbiased=True) / math.sqrt(sample.numel())
    df = sample.numel() - 1
    critical = _student_t_critical(1.0 - alpha, df, sample.device)
    lower = mean - critical * se
    statistic = (mean - minimum) / torch.clamp_min(se, 1e-30)
    p = 1.0 - _student_t_cdf(statistic.unsqueeze(0), df)[0]
    return {
        "metric": metric,
        "n_blocks": int(sample.numel()),
        "mean": float(mean),
        "standard_error": float(se),
        "lower_bound": float(lower),
        "upper_bound": "",
        "minimum_effect": minimum,
        "target": "",
        "margin": "",
        "alpha": alpha,
        "p_value": float(p),
        "status": "superior" if float(lower) > minimum else "inconclusive",
    }


def _equivalence_row(
    values: torch.Tensor, target: float, margin: float, alpha: float, metric: str
) -> dict[str, Any]:
    sample = values.to(torch.float64)
    mean = sample.mean()
    se = sample.std(unbiased=True) / math.sqrt(sample.numel())
    df = sample.numel() - 1
    critical = _student_t_critical(1.0 - alpha, df, sample.device)
    lower = mean - critical * se
    upper = mean + critical * se
    lower_stat = (mean - (target - margin)) / torch.clamp_min(se, 1e-30)
    upper_stat = (mean - (target + margin)) / torch.clamp_min(se, 1e-30)
    p_lower = 1.0 - _student_t_cdf(lower_stat.unsqueeze(0), df)[0]
    p_upper = _student_t_cdf(upper_stat.unsqueeze(0), df)[0]
    p = torch.maximum(p_lower, p_upper)
    return {
        "metric": metric,
        "n_blocks": int(sample.numel()),
        "mean": float(mean),
        "standard_error": float(se),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "minimum_effect": "",
        "target": target,
        "margin": margin,
        "alpha": alpha,
        "p_value": float(p),
        "status": "equivalent"
        if float(lower) > target - margin and float(upper) < target + margin
        else "inconclusive",
    }


def _holm_adjust(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    sorted_values = values[order]
    factors = torch.arange(values.numel(), 0, -1, device=values.device, dtype=values.dtype)
    adjusted_sorted = torch.cummax(torch.clamp(sorted_values * factors, max=1.0), dim=0).values
    adjusted = torch.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def evaluate_empirical_marked_filter(
    spec: RunSpec, run_directory: Path
) -> MarkedEvaluationResult:
    if not torch.cuda.is_available():
        raise ExperimentError("EMP-MARK-FILTER-001 requires an available CUDA device")
    started = time.perf_counter()
    values = spec.values
    evaluation = values["evaluation"]
    synthetic_run, synthetic_manifest_hash = _find_passed_synthetic_dependency(spec)
    swap, spot, dependency_rows = _load_verified_p6_payloads(spec)
    days, future_accesses = _prepare_empirical_marked_days(spec, swap, spot)
    ordered_dates = [str(value) for value in evaluation["ordered_dates"]]
    particle_chunk = _make_empirical_particle_chunk(spec)
    block_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    mark_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    full_outputs: dict[str, DayFilterOutput] = {}
    replay_seed: int | None = None
    replay_model: EmpiricalModel | None = None
    all_parameter_digests_before: list[str] = []
    total_supported = 0
    total_valid = 0
    master_seed = int(values["seed_policy"]["seeds"][0])
    block_minutes = int(evaluation["block_minutes"])

    for fold_index in range(1, len(ordered_dates)):
        train_dates = ordered_dates[:fold_index]
        heldout_date = ordered_dates[fold_index]
        train = _stack_train_intervals(days, train_dates)
        sigma_x, s_g = _train_reduced_parameters(days, train_dates)
        residual = _fit_residual_model(
            train, float(values["model"]["dirichlet_smoothing_beta"]), sigma_x, s_g
        )
        models = {"residual": residual}
        for variant in ("no_multi_tick", "no_multi_spread", "full", "unconstrained"):
            models[variant] = _fit_empirical_model(spec, train, variant, sigma_x, s_g)
        all_parameter_digests_before.extend(model.parameter_digest for model in models.values())
        _, _, option_margin = _dawson_threshold_margin(s_g, 0.05)
        day = days[heldout_date]
        valid_count = int(torch.count_nonzero(day.valid_interval))
        total_valid += valid_count
        total_supported += int(torch.count_nonzero(day.valid_interval & (day.mark_id >= 0)))
        outputs: dict[str, DayFilterOutput] = {}
        fold_seed = master_seed + fold_index * 10_000
        for variant_index, variant in enumerate(
            ("no_multi_tick", "no_multi_spread", "full", "unconstrained")
        ):
            # Identical seed across variants gives paired market-state proposals.
            outputs[variant] = _filter_empirical_day(
                spec, day, models[variant], fold_seed, particle_chunk
            )
        full_outputs[heldout_date] = outputs["full"]
        if heldout_date == "2024-12-15":
            replay_seed = fold_seed
            replay_model = models["full"]

        residual_score = _residual_scores(day, residual)
        full_score = outputs["full"].predictive_score
        score_improvement = full_score - residual_score
        posterior_sd_interval = torch.sqrt(
            torch.clamp_min(outputs["full"].variance[1:], 0.0)
        )
        uncertainty_ratio = posterior_sd_interval / option_margin
        uncertainty_metric = 1.0 - uncertainty_ratio
        rescaling = outputs["full"].expected_rescaling

        reductions: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
            "log_score_improvement": _block_reduce(
                day, score_improvement, reduction="mean", block_minutes=block_minutes
            ),
            "uncertainty_metric": _block_reduce(
                day, uncertainty_metric, reduction="median", block_minutes=block_minutes
            ),
            "rescaling_mean": _block_reduce(
                day, rescaling, reduction="mean", block_minutes=block_minutes
            ),
            "rescaling_std": _block_reduce(
                day, rescaling, reduction="std", block_minutes=block_minutes
            ),
            "full_minus_no_multi_tick": _block_reduce(
                day,
                full_score - outputs["no_multi_tick"].predictive_score,
                reduction="mean",
                block_minutes=block_minutes,
            ),
            "full_minus_no_multi_spread": _block_reduce(
                day,
                full_score - outputs["no_multi_spread"].predictive_score,
                reduction="mean",
                block_minutes=block_minutes,
            ),
            "full_minus_unconstrained": _block_reduce(
                day,
                full_score - outputs["unconstrained"].predictive_score,
                reduction="mean",
                block_minutes=block_minutes,
            ),
        }
        canonical_blocks = reductions["log_score_improvement"][2]
        for position, block in enumerate(canonical_blocks):
            row: dict[str, Any] = {
                "fold": fold_index,
                "train_dates": ";".join(train_dates),
                "heldout_date": heldout_date,
                "block": int(block),
                "valid_events": int(reductions["log_score_improvement"][1][position]),
                "option_margin_usdt": option_margin,
                "train_s_g_usdt": s_g,
                "train_sigma_x_usdt_per_sqrt_second": sigma_x,
            }
            for metric, (metric_values, _, metric_blocks) in reductions.items():
                location = torch.nonzero(metric_blocks == block, as_tuple=False).squeeze(-1)
                row[metric] = float(metric_values[location[0]]) if location.numel() else ""
            block_rows.append(row)

        for name, model in models.items():
            parameter_rows.append(
                {
                    "fold": fold_index,
                    "train_dates": ";".join(train_dates),
                    "heldout_date": heldout_date,
                    "model": name,
                    "alpha_per_second": float(model.alpha),
                    "s_g_usdt": model.s_g,
                    "sigma_x_usdt_per_sqrt_second": model.sigma_x,
                    "fit_initial_nll_nat_per_event": model.fit_initial_nll,
                    "fit_final_nll_nat_per_event": model.fit_final_nll,
                    "baseline_drift_abs_max": float(torch.max(torch.abs(model.baseline_drift))),
                    "parameter_digest_sha256": model.parameter_digest,
                    **{
                        f"baseline_rate_spread_{index + 1 if index < 7 else '8plus'}": float(rate)
                        for index, rate in enumerate(model.baseline_rates)
                    },
                }
            )
        direction, family, midpoint_bucket, spread_bucket = mark_metadata(day.timestamps_ms.device)
        heldout_flat = day.previous_spread_bucket[day.valid_interval] * MARK_COUNT + day.mark_id[
            day.valid_interval
        ]
        heldout_counts = torch.bincount(heldout_flat, minlength=8 * MARK_COUNT).reshape(8, MARK_COUNT)
        for spread_index, mark_id in torch.nonzero(heldout_counts > 0, as_tuple=False):
            mark_rows.append(
                {
                    "heldout_date": heldout_date,
                    "previous_spread_bucket": int(spread_index) + 1,
                    "mark_id": int(mark_id),
                    "direction": int(direction[mark_id]),
                    "family": int(family[mark_id]),
                    "midpoint_bucket": int(midpoint_bucket[mark_id]),
                    "spread_bucket": int(spread_bucket[mark_id]),
                    "heldout_count": int(heldout_counts[spread_index, mark_id]),
                    "train_full_probability": float(models["full"].probabilities[spread_index, mark_id]),
                    "train_residual_probability": float(residual.probabilities[spread_index, mark_id]),
                }
            )
        posterior_sd = torch.sqrt(torch.clamp_min(outputs["full"].variance[1:][day.valid_interval], 0.0))
        state_row: dict[str, Any] = {
            "fold": fold_index,
            "heldout_date": heldout_date,
            "bbo_rows": int(day.timestamps_ms.numel()),
            "valid_marked_events": valid_count,
            "supported_fraction": 1.0,
            "posterior_sd_median_usdt": float(torch.median(posterior_sd)),
            "option_margin_usdt": option_margin,
            "posterior_sd_to_margin_ratio": float(torch.median(posterior_sd) / option_margin),
            "full_filter_digest_sha256": outputs["full"].digest,
        }
        if day.spot_reference is not None:
            estimate = outputs["full"].estimate
            reference = day.spot_reference
            midpoint = (day.bid_ticks + day.ask_ticks).to(torch.float32) * 0.05
            state_row.update(
                filter_rmse_to_spot_reference_usdt=float(
                    torch.sqrt(torch.mean(torch.square(estimate - reference)))
                ),
                midpoint_rmse_to_spot_reference_usdt=float(
                    torch.sqrt(torch.mean(torch.square(midpoint - reference)))
                ),
            )
        state_rows.append(state_row)

    if replay_seed is None or replay_model is None:
        raise ExperimentError("December replay fold was not evaluated")
    december_day = days["2024-12-15"]
    replay_output = _filter_empirical_day(
        spec, december_day, replay_model, replay_seed, particle_chunk
    )
    deterministic_replay = (
        replay_output.digest == full_outputs["2024-12-15"].digest
        and torch.equal(replay_output.estimate, full_outputs["2024-12-15"].estimate)
        and torch.equal(replay_output.variance, full_outputs["2024-12-15"].variance)
        and torch.equal(replay_output.predictive_score, full_outputs["2024-12-15"].predictive_score)
    )

    block_device = days[ordered_dates[0]].timestamps_ms.device
    metric_names = (
        "log_score_improvement", "uncertainty_metric", "rescaling_mean", "rescaling_std",
        "full_minus_no_multi_tick", "full_minus_no_multi_spread", "full_minus_unconstrained",
    )
    vectors = {
        name: torch.tensor(
            [float(row[name]) for row in block_rows if row[name] != ""],
            device=block_device,
            dtype=torch.float64,
        )
        for name in metric_names
    }
    minimum_valid = int(evaluation["minimum_valid_blocks"])
    precision_sufficient = min(vector.numel() for vector in vectors.values()) >= minimum_valid
    primary_rows = [
        _superiority_row(
            vectors["log_score_improvement"],
            float(evaluation["log_score_minimum_effect_nat_per_event"]),
            float(evaluation["per_metric_alpha"]),
            "log_score_improvement_nat_per_event",
        ),
        _superiority_row(
            vectors["uncertainty_metric"],
            float(evaluation["uncertainty_margin_metric_minimum"]),
            float(evaluation["per_metric_alpha"]),
            "one_minus_posterior_sd_over_option_margin",
        ),
    ]
    calibration_rows = [
        _equivalence_row(
            vectors["rescaling_mean"],
            float(evaluation["rescaling_mean_target"]),
            float(evaluation["rescaling_mean_equivalence_margin"]),
            float(evaluation["calibration_per_metric_alpha"]),
            "time_rescaling_block_mean",
        ),
        _equivalence_row(
            vectors["rescaling_std"],
            float(evaluation["rescaling_standard_deviation_target"]),
            float(evaluation["rescaling_standard_deviation_equivalence_margin"]),
            float(evaluation["calibration_per_metric_alpha"]),
            "time_rescaling_block_standard_deviation",
        ),
    ]
    noninferiority_row = _superiority_row(
        vectors["full_minus_unconstrained"],
        -float(evaluation["constrained_noninferiority_margin_nat_per_event"]),
        float(evaluation["constrained_noninferiority_alpha"]),
        "constrained_minus_unconstrained_log_score",
    )
    component_rows = [
        _superiority_row(
            vectors["full_minus_no_multi_tick"], 0.0,
            float(evaluation["component_family_alpha"]), "multi_tick_component"
        ),
        _superiority_row(
            vectors["full_minus_no_multi_spread"], 0.0,
            float(evaluation["component_family_alpha"]), "multi_spread_component"
        ),
    ]
    component_p = torch.tensor(
        [float(row["p_value"]) for row in component_rows], device=block_device, dtype=torch.float64
    )
    component_adjusted = _holm_adjust(component_p)
    for row, adjusted in zip(component_rows, component_adjusted, strict=True):
        row["adjusted_p_value"] = float(adjusted)
        row["status"] = "superior" if float(adjusted) < float(evaluation["component_family_alpha"]) and float(row["mean"]) > 0.0 else "inconclusive"

    without_december = [row for row in block_rows if row["heldout_date"] != "2024-12-15"]
    sensitivity_score = torch.tensor(
        [float(row["log_score_improvement"]) for row in without_december], device=block_device
    ).mean()
    sensitivity_uncertainty = torch.tensor(
        [float(row["uncertainty_metric"]) for row in without_december], device=block_device
    ).mean()
    sensitivity_passed = float(sensitivity_score) > 0.0 and float(sensitivity_uncertainty) > 0.0
    elapsed = time.perf_counter() - started
    all_outputs = [output for date_outputs in full_outputs.values() for output in (date_outputs,)]
    all_finite = all(
        bool(torch.all(torch.isfinite(tensor)))
        for output in all_outputs
        for tensor in (output.estimate, output.variance, output.predictive_score)
    )
    positive_variance = all(
        bool(torch.all(output.variance > 0.0)) for output in all_outputs
    )
    parameter_digests_after = [row["parameter_digest_sha256"] for row in parameter_rows]
    frozen_parameters = parameter_digests_after == all_parameter_digests_before
    primary_passed = precision_sufficient and all(row["status"] == "superior" for row in primary_rows)
    calibration_passed = precision_sufficient and all(row["status"] == "equivalent" for row in calibration_rows)
    noninferiority_passed = precision_sufficient and noninferiority_row["status"] == "superior"
    acceptance = {
        "synthetic_dependency_passed": True,
        "dependency_hashes": True,
        "full_healthy_transition_support": total_supported == total_valid,
        "zero_future_timestamp_accesses": future_accesses == 0,
        "deterministic_replay": deterministic_replay,
        "all_filter_values_finite": all_finite,
        "positive_posterior_variance": positive_variance,
        "frozen_fold_parameters": frozen_parameters,
        "precision_budget": precision_sufficient,
        "primary_usability_family": primary_passed,
        "calibration_equivalence": calibration_passed,
        "constrained_noninferiority": noninferiority_passed,
        "december_exclusion_sensitivity": sensitivity_passed,
        "wall_time_within_limit": elapsed < float(evaluation["maximum_wall_seconds"]),
    }
    metrics = {
        "synthetic_dependency_run": synthetic_run,
        "synthetic_dependency_manifest_sha256": synthetic_manifest_hash,
        "fold_count": len(ordered_dates) - 1,
        "valid_blocks": int(vectors["log_score_improvement"].numel()),
        "minimum_valid_blocks": minimum_valid,
        "valid_healthy_transitions": total_valid,
        "supported_healthy_transitions": total_supported,
        "supported_fraction": total_supported / max(total_valid, 1),
        "future_timestamp_accesses": future_accesses,
        "primary_log_score_improvement_mean_nat_per_event": float(primary_rows[0]["mean"]),
        "primary_log_score_improvement_lower_bound": float(primary_rows[0]["lower_bound"]),
        "primary_uncertainty_metric_mean": float(primary_rows[1]["mean"]),
        "primary_uncertainty_metric_lower_bound": float(primary_rows[1]["lower_bound"]),
        "posterior_sd_to_margin_ratio_mean_block": 1.0 - float(primary_rows[1]["mean"]),
        "rescaling_block_mean": float(calibration_rows[0]["mean"]),
        "rescaling_block_standard_deviation_mean": float(calibration_rows[1]["mean"]),
        "constrained_minus_unconstrained_mean_nat_per_event": float(noninferiority_row["mean"]),
        "multi_tick_component_mean_nat_per_event": float(component_rows[0]["mean"]),
        "multi_tick_component_holm_p": float(component_rows[0]["adjusted_p_value"]),
        "multi_spread_component_mean_nat_per_event": float(component_rows[1]["mean"]),
        "multi_spread_component_holm_p": float(component_rows[1]["adjusted_p_value"]),
        "without_december_log_score_mean": float(sensitivity_score),
        "without_december_uncertainty_metric_mean": float(sensitivity_uncertainty),
        "deterministic_replay": deterministic_replay,
        "december_filter_digest_sha256": full_outputs["2024-12-15"].digest,
        "elapsed_seconds": elapsed,
    }

    inference_rows = primary_rows + calibration_rows + [noninferiority_row] + component_rows
    write_csv(
        run_directory / "metrics" / "block_metrics.csv",
        _union_fieldnames(block_rows),
        block_rows,
    )
    write_csv(
        run_directory / "metrics" / "day_state.csv",
        _union_fieldnames(state_rows),
        state_rows,
    )
    write_csv(
        run_directory / "tables" / "fold_parameters.csv",
        _union_fieldnames(parameter_rows),
        parameter_rows,
    )
    write_csv(
        run_directory / "tables" / "mark_diagnostics.csv",
        _union_fieldnames(mark_rows),
        mark_rows,
    )
    write_csv(
        run_directory / "tables" / "inference.csv",
        _union_fieldnames(inference_rows),
        inference_rows,
    )
    atomic_write_json(
        run_directory / "metrics" / "dependency_audit.json",
        {
            "schema_version": "p6m-dependency-audit-v1",
            "p6_dependency_run": str(values["inputs"]["p6_dependency_run"]),
            "p6_manifest_sha256": str(values["inputs"]["p6_dependency_manifest_sha256"]),
            "p6_state_manifest_sha256": str(values["inputs"]["p6_dependency_state_manifest_sha256"]),
            "verified_processed_assets": len(dependency_rows),
            "synthetic_dependency_run": synthetic_run,
            "synthetic_dependency_manifest_sha256": synthetic_manifest_hash,
        },
    )
    atomic_write_json(
        run_directory / "metrics" / "replay.json",
        {
            "schema_version": "p6m-empirical-replay-v1",
            "date": "2024-12-15",
            "first_digest_sha256": full_outputs["2024-12-15"].digest,
            "second_digest_sha256": replay_output.digest,
            "exact_tensor_equality": deterministic_replay,
            "seed": replay_seed,
        },
    )
    atomic_write_json(
        run_directory / "metrics" / "sensitivity.json",
        {
            "schema_version": "p6m-sensitivity-v1",
            "excluded_date": "2024-12-15",
            "included_blocks": len(without_december),
            "log_score_improvement_mean": float(sensitivity_score),
            "uncertainty_metric_mean": float(sensitivity_uncertainty),
            "positive_both": sensitivity_passed,
        },
    )
    atomic_write_json(
        run_directory / "metrics" / "timestamp_audit.json",
        {
            "schema_version": "timestamp-audit-v1",
            "future_timestamp_accesses": future_accesses,
            "spot_role": str(values["model"]["spot_role"]),
            "equality_allowed": True,
        },
    )
    _save_torch_artifact(
        {
            "schema_version": "p6m-december-filter-state-v1",
            "date": "2024-12-15",
            "timestamps_ms": december_day.timestamps_ms.to("cpu"),
            "filtered_efficient_price": full_outputs["2024-12-15"].estimate.to("cpu"),
            "posterior_variance": full_outputs["2024-12-15"].variance.to("cpu"),
            "mark_id": december_day.mark_id.to("cpu"),
            "valid_interval": december_day.valid_interval.to("cpu"),
            "spot_reference": december_day.spot_reference.to("cpu")
            if december_day.spot_reference is not None
            else None,
        },
        run_directory / "state" / "december_filter.pt",
    )
    return MarkedEvaluationResult(
        metrics=metrics,
        acceptance=acceptance,
        derived_parameters={
            "mark_count": MARK_COUNT,
            "spread_bucket_count": 8,
            "fold_count": len(ordered_dates) - 1,
            "cuda_device": torch.cuda.get_device_name(),
            "synthetic_dependency_run": synthetic_run,
        },
        log_lines=(
            f"folds={len(ordered_dates)-1}; blocks={vectors['log_score_improvement'].numel()}",
            f"log_score_mean={primary_rows[0]['mean']:.6g}; lower={primary_rows[0]['lower_bound']:.6g}",
            f"uncertainty_metric={primary_rows[1]['mean']:.6g}; lower={primary_rows[1]['lower_bound']:.6g}",
            f"calibration_mean={calibration_rows[0]['mean']:.6g}; sd={calibration_rows[1]['mean']:.6g}",
            f"replay={deterministic_replay}; digest={full_outputs['2024-12-15'].digest}",
            f"acceptance_passed={all(acceptance.values())}",
        ),
    )
