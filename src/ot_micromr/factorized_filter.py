from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import torch

from ot_micromr.artifacts import atomic_write_json, write_csv
from ot_micromr.config import RunSpec
from ot_micromr.efficient_price import systematic_resample
from ot_micromr.empirical_filter import _dawson_threshold_margin
from ot_micromr.errors import ExperimentError
from ot_micromr.marked_filter import (
    FAMILY_COUNT,
    MAGNITUDE_BUCKET_COUNT,
    MARK_COUNT,
    EmpiricalMarkedDay,
    MarkedEvaluationResult,
    _block_reduce,
    _empirical_probability_tables,
    _equivalence_row,
    _find_passed_synthetic_dependency,
    _load_verified_p6_payloads,
    _prepare_empirical_marked_days,
    _save_torch_artifact,
    _stack_train_intervals,
    _superiority_row,
    _tensor_digest,
    _train_reduced_parameters,
    _union_fieldnames,
    _valid_segments,
    mark_metadata,
)


@dataclass(frozen=True, slots=True)
class FactorizedModel:
    probabilities: torch.Tensor
    direction_probabilities: torch.Tensor
    directional_beta: torch.Tensor
    sigma_x: float
    s_g: float
    clock_log_mean: torch.Tensor
    clock_log_variance: torch.Tensor
    fit_initial_nll: float
    fit_final_nll: float
    parameter_digest: str


@dataclass(frozen=True, slots=True)
class FactorizedDayOutput:
    estimate: torch.Tensor
    variance: torch.Tensor
    conditional_mark_score: torch.Tensor
    clock_log_density: torch.Tensor
    time_rescaling: torch.Tensor
    clock_log_mean: torch.Tensor
    clock_log_scale: torch.Tensor
    predicted_direction_probabilities: torch.Tensor
    digest: str


def causal_rolling_lognormal_parameters(
    durations: torch.Tensor,
    valid: torch.Tensor,
    training_log_mean: torch.Tensor,
    training_log_variance: torch.Tensor,
    history_events: int,
    prior_events: float,
    variance_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pre-event lognormal parameters using only earlier valid durations."""
    selected = torch.log(torch.clamp_min(durations[valid].to(torch.float64), 1e-6))
    count = selected.numel()
    index = torch.arange(count, device=selected.device, dtype=torch.int64)
    start = torch.clamp_min(index - history_events, 0)
    prefix = torch.cat((selected.new_zeros(1), torch.cumsum(selected, dim=0)))
    square_prefix = torch.cat(
        (selected.new_zeros(1), torch.cumsum(torch.square(selected), dim=0))
    )
    history_count = (index - start).to(torch.float64)
    history_sum = prefix[index] - prefix[start]
    history_square_sum = square_prefix[index] - square_prefix[start]
    denominator = history_count + prior_events
    mean = (history_sum + prior_events * training_log_mean) / denominator
    second_moment = (
        history_square_sum
        + prior_events * (training_log_variance + torch.square(training_log_mean))
    ) / denominator
    scale = torch.sqrt(
        torch.clamp_min(second_moment - torch.square(mean), variance_floor)
    )
    all_mean = torch.full_like(durations, float(training_log_mean), dtype=torch.float64)
    all_scale = torch.full_like(
        durations,
        float(torch.sqrt(torch.clamp_min(training_log_variance, variance_floor))),
        dtype=torch.float64,
    )
    all_mean[valid] = mean
    all_scale[valid] = scale
    return all_mean, all_scale


def lognormal_clock_terms(
    durations: torch.Tensor, log_mean: torch.Tensor, log_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    duration = torch.clamp_min(durations.to(torch.float64), 1e-6)
    log_duration = torch.log(duration)
    standardized = (log_duration - log_mean) / log_scale
    log_density = (
        -log_duration
        - torch.log(log_scale)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * torch.square(standardized)
    )
    survival = 0.5 * torch.erfc(standardized / math.sqrt(2.0))
    survival = torch.clamp(survival, min=1e-30, max=1.0)
    rescaling = -torch.log(survival)
    pit = 1.0 - survival
    return log_density, rescaling, pit


def conditional_direction_probabilities(
    gap: torch.Tensor,
    spread: torch.Tensor,
    base_direction_probabilities: torch.Tensor,
    beta: torch.Tensor,
    s_g: float | torch.Tensor,
) -> torch.Tensor:
    direction = gap.new_tensor((-1.0, 0.0, 1.0))
    logits = (
        torch.log(torch.clamp_min(base_direction_probabilities[spread], 1e-30))
        - beta * (gap / s_g).unsqueeze(-1) * direction
    )
    return torch.softmax(logits, dim=-1)


def _fit_factorized_model(
    spec: RunSpec,
    train: tuple[torch.Tensor, ...],
    sigma_x: float,
    s_g: float,
) -> FactorizedModel:
    gap, spread, mark, _, *rest = train
    endpoint_gap = rest[4]
    beta_smoothing = float(spec.values["model"]["dirichlet_smoothing_beta"])
    _, _, _, _, probabilities, _ = _empirical_probability_tables(
        train, beta_smoothing, "full"
    )
    direction_sign = mark_metadata(gap.device)[0]
    direction_one_hot = torch.nn.functional.one_hot(
        direction_sign + 1, 3
    ).to(torch.float32)
    direction_probabilities = probabilities @ direction_one_hot
    raw_beta = torch.nn.Parameter(gap.new_tensor(-4.6))
    batch_size = int(spec.values["numerics"]["fit_batch_events"])
    learning_rate = float(spec.values["numerics"]["optimizer_learning_rate"])
    optimizer = torch.optim.Adam((raw_beta,), lr=learning_rate)

    def loss_function(
        raw_parameter: torch.Tensor,
        batch_gap: torch.Tensor,
        batch_spread: torch.Tensor,
        batch_mark: torch.Tensor,
        probabilities_arg: torch.Tensor,
        direction_probabilities_arg: torch.Tensor,
        direction_sign_arg: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        beta = torch.nn.functional.softplus(raw_parameter)
        normalized_gap = batch_gap / scale
        direction = direction_sign_arg[batch_mark].to(batch_gap.dtype)
        log_tilt = -beta * direction * normalized_gap
        direction_values = batch_gap.new_tensor((-1.0, 0.0, 1.0))
        log_normalizer = torch.logsumexp(
            torch.log(torch.clamp_min(direction_probabilities_arg[batch_spread], 1e-30))
            - beta * normalized_gap.unsqueeze(-1) * direction_values,
            dim=-1,
        )
        log_probability = (
            torch.log(
                torch.clamp_min(probabilities_arg[batch_spread, batch_mark], 1e-30)
            )
            + log_tilt
            - log_normalizer
        )
        return -torch.mean(log_probability)

    loss = torch.compile(
        loss_function,
        mode=str(spec.values["numerics"]["compile_mode"]),
        fullgraph=True,
        dynamic=True,
    )
    n = gap.numel()

    def batch_at(step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if n <= batch_size:
            return endpoint_gap, spread, mark
        start = (step * batch_size) % n
        index = torch.remainder(
            torch.arange(batch_size, device=gap.device, dtype=torch.int64) + start, n
        )
        return endpoint_gap[index], spread[index], mark[index]

    scale = gap.new_tensor(s_g)
    first = batch_at(0)
    initial_loss = float(
        loss(
            raw_beta,
            *first,
            probabilities,
            direction_probabilities,
            direction_sign,
            scale,
        ).detach()
    )
    for step in range(int(spec.values["numerics"]["optimizer_steps"])):
        optimizer.zero_grad(set_to_none=True)
        batch = batch_at(step)
        objective = loss(
            raw_beta,
            *batch,
            probabilities,
            direction_probabilities,
            direction_sign,
            scale,
        )
        objective.backward()
        optimizer.step()
    beta = torch.nn.functional.softplus(raw_beta.detach())
    final_loss = float(
        loss(
            raw_beta,
            endpoint_gap,
            spread,
            mark,
            probabilities,
            direction_probabilities,
            direction_sign,
            scale,
        ).detach()
    )
    log_duration = torch.log(torch.clamp_min(train[3].to(torch.float64), 1e-6))
    clock_mean = torch.mean(log_duration)
    clock_variance = torch.var(log_duration, unbiased=False)
    digest = _tensor_digest(
        (
            probabilities,
            direction_probabilities,
            beta,
            clock_mean,
            clock_variance,
        )
    )
    return FactorizedModel(
        probabilities=probabilities,
        direction_probabilities=direction_probabilities,
        directional_beta=beta,
        sigma_x=sigma_x,
        s_g=s_g,
        clock_log_mean=clock_mean,
        clock_log_variance=clock_variance,
        fit_initial_nll=initial_loss,
        fit_final_nll=final_loss,
        parameter_digest=digest,
    )


def _make_factorized_particle_chunk(spec: RunSpec):
    direction_values = torch.tensor((-1.0, 0.0, 1.0), device="cuda")
    direction_sign = mark_metadata("cuda")[0]

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
        base_direction: torch.Tensor,
        beta: torch.Tensor,
        s_g: torch.Tensor,
        sigma_x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        increments = sigma_x * torch.sqrt(dt).unsqueeze(-1) * normals
        particle_ends = particles.unsqueeze(0) + torch.cumsum(increments, dim=0)
        particle_starts = torch.cat((particles.unsqueeze(0), particle_ends[:-1]), dim=0)
        del particle_starts
        gap = prior_mid.unsqueeze(-1) - particle_ends
        safe_events = torch.clamp_min(events, 0)
        normalized_gap = gap / s_g
        log_direction_numerator = (
            torch.log(torch.clamp_min(base_direction[prior_spread], 1e-30)).unsqueeze(-2)
            - beta * normalized_gap.unsqueeze(-1) * direction_values
        )
        log_normalizer = torch.logsumexp(log_direction_numerator, dim=-1)
        event_direction = direction_sign[safe_events] + 1
        selected_tilt = torch.gather(
            -beta * normalized_gap.unsqueeze(-1) * direction_values,
            -1,
            event_direction.unsqueeze(-1).unsqueeze(-1).expand(
                *event_direction.shape, normalized_gap.shape[-1], 1
            ),
        ).squeeze(-1)
        log_mark = (
            torch.log(
                torch.clamp_min(probabilities[prior_spread, safe_events], 1e-30)
            ).unsqueeze(-1)
            + selected_tilt
            - log_normalizer
        )
        interval_score = torch.where(
            events.unsqueeze(-1) >= 0, log_mark, torch.zeros_like(log_mark)
        )
        cumulative = log_weights.unsqueeze(0) + torch.cumsum(interval_score, dim=0)
        normalizers = torch.logsumexp(cumulative, dim=-1)
        previous_normalizers = torch.cat(
            (normalizers.new_zeros((1, normalizers.shape[1])), normalizers[:-1]), dim=0
        )
        predictive_mark_score = normalizers - previous_normalizers
        normalized = cumulative - normalizers.unsqueeze(-1)
        weights = torch.exp(normalized)
        estimates = torch.sum(weights * particle_ends, dim=-1)
        variances = torch.sum(
            weights * torch.square(particle_ends - estimates.unsqueeze(-1)), dim=-1
        )
        prior_cumulative = torch.cat((log_weights.unsqueeze(0), cumulative[:-1]), dim=0)
        prior_normalized = prior_cumulative - torch.logsumexp(
            prior_cumulative, dim=-1, keepdim=True
        )
        particle_direction = torch.exp(
            log_direction_numerator - log_normalizer.unsqueeze(-1)
        )
        predicted_direction = torch.sum(
            torch.exp(prior_normalized).unsqueeze(-1) * particle_direction, dim=-2
        )
        particles, log_weights = systematic_resample(
            particle_ends[-1], normalized[-1], offsets
        )
        return (
            particles,
            log_weights,
            estimates,
            variances,
            predictive_mark_score,
            predicted_direction,
        )

    return torch.compile(
        particle_chunk,
        mode=str(spec.values["numerics"]["compile_mode"]),
        fullgraph=True,
    )


def _filter_factorized_day(
    spec: RunSpec,
    day: EmpiricalMarkedDay,
    model: FactorizedModel,
    seed: int,
    particle_chunk,
) -> FactorizedDayOutput:
    device = day.timestamps_ms.device
    particle_count = int(spec.values["numerics"]["particle_count"])
    chunk_events = int(spec.values["numerics"]["particle_chunk_events"])
    history_events = int(spec.values["model"]["clock_history_events"])
    prior_events = float(spec.values["model"]["clock_training_prior_events"])
    variance_floor = float(spec.values["model"]["clock_log_variance_floor"])
    clock_mean, clock_scale = causal_rolling_lognormal_parameters(
        day.dt_seconds,
        day.valid_interval,
        model.clock_log_mean,
        model.clock_log_variance,
        history_events,
        prior_events,
        variance_floor,
    )
    clock_log_density, rescaling, _ = lognormal_clock_terms(
        day.dt_seconds, clock_mean, clock_scale
    )
    clock_log_density = torch.where(
        day.valid_interval, clock_log_density, torch.zeros_like(clock_log_density)
    )
    rescaling = torch.where(
        day.valid_interval, rescaling, torch.zeros_like(rescaling)
    )
    starts, lengths = _valid_segments(day.valid_interval)
    mid = (day.bid_ticks + day.ask_ticks).to(torch.float32) * 0.05
    estimate = mid.clone()
    variance = torch.full_like(mid, model.s_g * model.s_g)
    conditional_score = torch.zeros_like(day.dt_seconds)
    predicted_direction = model.direction_probabilities[
        day.previous_spread_bucket
    ].clone()
    generator = torch.Generator(device=device).manual_seed(seed)
    group_size = 64
    arange_chunk = torch.arange(chunk_events, device=device).unsqueeze(1)
    for group_start in range(0, starts.numel(), group_size):
        actual = min(group_size, starts.numel() - group_start)
        group_starts = torch.zeros(group_size, device=device, dtype=torch.int64)
        group_lengths = torch.zeros_like(group_starts)
        group_starts[:actual] = starts[group_start : group_start + actual]
        group_lengths[:actual] = lengths[group_start : group_start + actual]
        particles = day.prior_mid_price[group_starts].unsqueeze(-1) + model.s_g * torch.randn(
            (group_size, particle_count),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        log_weights = torch.full_like(particles, -math.log(particle_count))
        maximum_length = int(group_lengths.max())
        for offset in range(0, maximum_length, chunk_events):
            interval_index = group_starts.unsqueeze(0) + offset + arange_chunk
            valid = arange_chunk < torch.clamp_min(
                group_lengths - offset, 0
            ).unsqueeze(0)
            safe = torch.clamp(interval_index, 0, day.dt_seconds.numel() - 1)
            dt = torch.where(
                valid, day.dt_seconds[safe], torch.zeros_like(day.dt_seconds[safe])
            )
            events = torch.where(valid, day.mark_id[safe], torch.full_like(safe, -1))
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
                chunk_direction,
            ) = particle_chunk(
                particles,
                log_weights,
                day.prior_mid_price[safe],
                day.previous_spread_bucket[safe],
                events,
                dt,
                normals,
                offsets,
                model.probabilities,
                model.direction_probabilities,
                model.directional_beta,
                day.prior_mid_price.new_tensor(model.s_g),
                day.prior_mid_price.new_tensor(model.sigma_x),
            )
            target = safe + 1
            estimate[target[valid]] = chunk_estimate[valid]
            variance[target[valid]] = chunk_variance[valid]
            conditional_score[safe[valid]] = chunk_score[valid]
            predicted_direction[safe[valid]] = chunk_direction[valid]
            particles = particles.clone()
            log_weights = log_weights.clone()
    digest = _tensor_digest(
        (
            estimate,
            variance,
            conditional_score,
            clock_log_density,
            rescaling,
            clock_mean,
            clock_scale,
            predicted_direction,
        )
    )
    return FactorizedDayOutput(
        estimate=estimate,
        variance=variance,
        conditional_mark_score=conditional_score,
        clock_log_density=clock_log_density,
        time_rescaling=rescaling,
        clock_log_mean=clock_mean,
        clock_log_scale=clock_scale,
        predicted_direction_probabilities=predicted_direction,
        digest=digest,
    )


def _category_conditionals(probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
    direction, family, midpoint_bucket, spread_bucket = mark_metadata(probabilities.device)
    direction_one_hot = torch.nn.functional.one_hot(direction + 1, 3).to(torch.float32)
    base_direction = probabilities @ direction_one_hot
    categories = {
        "direction": direction + 1,
        "family": family + 1,
        "midpoint_magnitude": midpoint_bucket,
        "spread_magnitude": spread_bucket,
    }
    counts = {
        "direction": 3,
        "family": FAMILY_COUNT,
        "midpoint_magnitude": MAGNITUDE_BUCKET_COUNT,
        "spread_magnitude": MAGNITUDE_BUCKET_COUNT,
    }
    result: dict[str, torch.Tensor] = {}
    for name, category in categories.items():
        category_one_hot = torch.nn.functional.one_hot(
            category, counts[name]
        ).to(torch.float32)
        joint = torch.einsum(
            "sm,md,mk->sdk", probabilities, direction_one_hot, category_one_hot
        )
        result[name] = joint / torch.clamp_min(base_direction.unsqueeze(-1), 1e-30)
    return result


def _predicted_categories(
    day: EmpiricalMarkedDay,
    output: FactorizedDayOutput,
    model: FactorizedModel,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    valid = day.valid_interval
    spread = day.previous_spread_bucket[valid]
    predicted_direction = output.predicted_direction_probabilities[valid]
    direction, family, midpoint_bucket, spread_bucket = mark_metadata(
        day.timestamps_ms.device
    )
    actual_categories = {
        "direction": direction[day.mark_id[valid]] + 1,
        "family": family[day.mark_id[valid]] + 1,
        "midpoint_magnitude": midpoint_bucket[day.mark_id[valid]],
        "spread_magnitude": spread_bucket[day.mark_id[valid]],
    }
    counts = {
        "direction": 3,
        "family": 3,
        "midpoint_magnitude": 9,
        "spread_magnitude": 9,
    }
    conditionals = _category_conditionals(model.probabilities)
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, actual_index in actual_categories.items():
        actual = torch.nn.functional.one_hot(
            actual_index, counts[name]
        ).to(torch.float64).sum(dim=0)
        if name == "direction":
            predicted = predicted_direction.to(torch.float64).sum(dim=0)
        else:
            predicted = torch.einsum(
                "nd,ndk->nk",
                predicted_direction,
                conditionals[name][spread],
            ).to(torch.float64).sum(dim=0)
        result[name] = actual, predicted
    return result


def _histogram_rows(
    actual_duration: torch.Tensor,
    predictive_duration: torch.Tensor,
    rescaling: torch.Tensor,
    category_sums: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> list[dict[str, Any]]:
    actual_log = torch.log10(torch.clamp_min(actual_duration, 1e-6))
    predictive_log = torch.log10(torch.clamp_min(predictive_duration, 1e-6))
    combined = torch.cat((actual_log, predictive_log))
    lower = torch.quantile(combined, 0.001)
    upper = torch.quantile(combined, 0.999)
    duration_edges = torch.linspace(
        float(lower), float(upper), 51, device=combined.device, dtype=torch.float64
    )
    actual_hist = torch.histogram(actual_log, duration_edges, density=True).hist
    predictive_hist = torch.histogram(
        predictive_log, duration_edges, density=True
    ).hist
    duration_centers = 0.5 * (duration_edges[:-1] + duration_edges[1:])

    rescaling_upper = max(8.0, float(torch.quantile(rescaling, 0.995)))
    rescaling_edges = torch.linspace(
        0.0, rescaling_upper, 51, device=rescaling.device, dtype=torch.float64
    )
    rescaling_hist = torch.histogram(
        torch.clamp(rescaling, max=rescaling_upper), rescaling_edges, density=True
    ).hist
    rescaling_centers = 0.5 * (rescaling_edges[:-1] + rescaling_edges[1:])

    rows: list[dict[str, Any]] = []
    for index in range(duration_centers.numel()):
        rows.append(
            {
                "distribution": "log10_duration_seconds",
                "category": index,
                "x": float(duration_centers[index]),
                "actual": float(actual_hist[index]),
                "predicted": float(predictive_hist[index]),
                "reference": "",
            }
        )
    for index in range(rescaling_centers.numel()):
        rows.append(
            {
                "distribution": "time_rescaling",
                "category": index,
                "x": float(rescaling_centers[index]),
                "actual": float(rescaling_hist[index]),
                "predicted": "",
                "reference": float(torch.exp(-rescaling_centers[index])),
            }
        )
    for name, (actual_sum, predicted_sum) in category_sums.items():
        total = torch.clamp_min(actual_sum.sum(), 1.0)
        for index in range(actual_sum.numel()):
            rows.append(
                {
                    "distribution": name,
                    "category": index,
                    "x": index,
                    "actual": float(actual_sum[index] / total),
                    "predicted": float(predicted_sum[index] / total),
                    "reference": "",
                }
            )
    return rows


def _timeseries_rows(
    day: EmpiricalMarkedDay,
    output: FactorizedDayOutput,
    minutes: int,
) -> list[dict[str, Any]]:
    endpoint_time = day.timestamps_ms[1:]
    valid = day.valid_interval & (
        endpoint_time <= day.timestamps_ms[0] + minutes * 60 * 1000
    )
    indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    maximum_points = 4_000
    if indices.numel() > maximum_points:
        selection = torch.linspace(
            0,
            indices.numel() - 1,
            maximum_points,
            device=indices.device,
            dtype=torch.float64,
        ).round().to(torch.int64)
        indices = indices[selection]
    direction = mark_metadata(day.timestamps_ms.device)[0][day.mark_id[indices]]
    posterior_sd = torch.sqrt(torch.clamp_min(output.variance[indices + 1], 0.0))
    p10 = torch.exp(output.clock_log_mean[indices] - 1.281551565545 * output.clock_log_scale[indices])
    median = torch.exp(output.clock_log_mean[indices])
    p90 = torch.exp(output.clock_log_mean[indices] + 1.281551565545 * output.clock_log_scale[indices])
    rows: list[dict[str, Any]] = []
    for position, index in enumerate(indices):
        rows.append(
            {
                "time_minutes": float(
                    (endpoint_time[index] - day.timestamps_ms[0]).to(torch.float64) / 60_000.0
                ),
                "bid_usdt": float(day.bid_ticks[index + 1]) * 0.1,
                "ask_usdt": float(day.ask_ticks[index + 1]) * 0.1,
                "filtered_efficient_price_usdt": float(output.estimate[index + 1]),
                "posterior_sd_usdt": float(posterior_sd[position]),
                "actual_duration_seconds": float(day.dt_seconds[index]),
                "predicted_duration_p10_seconds": float(p10[position]),
                "predicted_duration_median_seconds": float(median[position]),
                "predicted_duration_p90_seconds": float(p90[position]),
                "actual_direction": int(direction[position]),
                "predicted_down_probability": float(
                    output.predicted_direction_probabilities[index, 0]
                ),
                "predicted_flat_probability": float(
                    output.predicted_direction_probabilities[index, 1]
                ),
                "predicted_up_probability": float(
                    output.predicted_direction_probabilities[index, 2]
                ),
            }
        )
    return rows


def _render_figures(
    run_directory: Path,
    timeseries_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    day_rows: list[dict[str, Any]],
) -> None:
    plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25})

    time = [row["time_minutes"] for row in timeseries_rows]
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].plot(time, [row["bid_usdt"] for row in timeseries_rows], label="bid", lw=0.8)
    axes[0].plot(time, [row["ask_usdt"] for row in timeseries_rows], label="ask", lw=0.8)
    axes[0].plot(
        time,
        [row["filtered_efficient_price_usdt"] for row in timeseries_rows],
        label="filtered X",
        lw=1.0,
    )
    axes[0].set_ylabel("price (USDT)")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].scatter(
        time,
        [row["actual_duration_seconds"] for row in timeseries_rows],
        s=5,
        alpha=0.45,
        label="actual duration",
    )
    median = [row["predicted_duration_median_seconds"] for row in timeseries_rows]
    axes[1].plot(time, median, color="tab:orange", lw=1.0, label="predicted median")
    axes[1].fill_between(
        time,
        [row["predicted_duration_p10_seconds"] for row in timeseries_rows],
        [row["predicted_duration_p90_seconds"] for row in timeseries_rows],
        color="tab:orange",
        alpha=0.2,
        label="predicted 10-90%",
    )
    axes[1].set(yscale="log", ylabel="next-event duration (s)")
    axes[1].legend(frameon=False, ncol=3)
    axes[2].scatter(
        time,
        [row["actual_direction"] for row in timeseries_rows],
        s=4,
        alpha=0.25,
        label="actual direction",
    )
    axes[2].plot(
        time,
        [
            row["predicted_up_probability"] - row["predicted_down_probability"]
            for row in timeseries_rows
        ],
        color="tab:red",
        lw=1.0,
        label="predicted E[direction]",
    )
    axes[2].set(xlabel="minutes from day start", ylabel="direction", ylim=(-1.1, 1.1))
    axes[2].legend(frameon=False)
    figure.suptitle("P6D observed BBO and causal factorized predictions")
    figure.savefig(
        run_directory / "figures" / "timeseries.png",
        metadata={"Software": "ot-micromr 0.1.0"},
    )
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    names = (
        "log10_duration_seconds",
        "time_rescaling",
        "direction",
        "family",
        "midpoint_magnitude",
        "spread_magnitude",
    )
    titles = (
        "Duration distribution",
        "Time rescaling vs Exp(1)",
        "Direction distribution",
        "Spread-family distribution",
        "Midpoint magnitude bucket",
        "Spread magnitude bucket",
    )
    for axis, name, title in zip(axes.flat, names, titles, strict=True):
        rows = [row for row in distribution_rows if row["distribution"] == name]
        x = [row["x"] for row in rows]
        if name in {"log10_duration_seconds", "time_rescaling"}:
            axis.plot(x, [row["actual"] for row in rows], label="actual", lw=1.3)
            if name == "log10_duration_seconds":
                axis.plot(x, [row["predicted"] for row in rows], label="predicted", lw=1.3)
                axis.set_xlabel("log10 seconds")
            else:
                axis.plot(x, [row["reference"] for row in rows], label="Exp(1)", lw=1.3)
                axis.set_xlabel("integrated hazard")
        else:
            width = 0.38
            axis.bar([value - width / 2 for value in x], [row["actual"] for row in rows], width, label="actual")
            axis.bar([value + width / 2 for value in x], [row["predicted"] for row in rows], width, label="predicted")
            axis.set_xlabel("bucket")
        axis.set_title(title)
        axis.legend(frameon=False)
    figure.savefig(
        run_directory / "figures" / "predictive-distributions.png",
        metadata={"Software": "ot-micromr 0.1.0"},
    )
    plt.close(figure)

    dates = [row["heldout_date"] for row in day_rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(dates, [row["rescaling_mean"] for row in day_rows], marker="o")
    axes[0, 0].axhspan(0.9, 1.1, color="tab:green", alpha=0.15)
    axes[0, 0].set(title="Clock rescaling mean", ylabel="mean")
    axes[0, 1].plot(dates, [row["rescaling_sd"] for row in day_rows], marker="o")
    axes[0, 1].axhspan(0.8, 1.2, color="tab:green", alpha=0.15)
    axes[0, 1].set(title="Clock rescaling SD", ylabel="SD")
    axes[1, 0].bar(dates, [row["mark_log_score_gain"] for row in day_rows])
    axes[1, 0].axhline(0.0, color="black", lw=0.8)
    axes[1, 0].set(title="Conditional mark gain", ylabel="nat/event")
    axes[1, 1].bar(dates, [row["posterior_sd_to_margin_ratio"] for row in day_rows])
    axes[1, 1].axhline(1.0, color="tab:red", lw=1.0)
    axes[1, 1].set(title="State uncertainty", ylabel="posterior SD / option margin")
    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=35)
    figure.savefig(
        run_directory / "figures" / "fold-calibration.png",
        metadata={"Software": "ot-micromr 0.1.0"},
    )
    plt.close(figure)


def evaluate_factorized_empirical_filter(
    spec: RunSpec, run_directory: Path
) -> MarkedEvaluationResult:
    if not torch.cuda.is_available():
        raise ExperimentError("EMP-MARK-FACT-001 requires an available CUDA device")
    started = time.perf_counter()
    values = spec.values
    evaluation = values["evaluation"]
    synthetic_run, synthetic_manifest_hash = _find_passed_synthetic_dependency(spec)
    swap, spot, dependency_rows = _load_verified_p6_payloads(spec)
    days, future_accesses = _prepare_empirical_marked_days(spec, swap, spot)
    ordered_dates = [str(value) for value in evaluation["ordered_dates"]]
    particle_chunk = _make_factorized_particle_chunk(spec)
    block_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []
    distribution_fold_rows: list[dict[str, Any]] = []
    outputs: dict[str, FactorizedDayOutput] = {}
    models: dict[str, FactorizedModel] = {}
    parameter_digests: list[str] = []
    total_valid = 0
    duration_actual: list[torch.Tensor] = []
    duration_mean: list[torch.Tensor] = []
    duration_scale: list[torch.Tensor] = []
    rescaling_all: list[torch.Tensor] = []
    category_sums: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    master_seed = int(values["seed_policy"]["seeds"][0])
    block_minutes = int(evaluation["block_minutes"])
    replay_seed: int | None = None

    for fold_index in range(1, len(ordered_dates)):
        train_dates = ordered_dates[:fold_index]
        heldout_date = ordered_dates[fold_index]
        train = _stack_train_intervals(days, train_dates)
        sigma_x, s_g = _train_reduced_parameters(days, train_dates)
        model = _fit_factorized_model(spec, train, sigma_x, s_g)
        models[heldout_date] = model
        parameter_digests.append(model.parameter_digest)
        day = days[heldout_date]
        fold_seed = master_seed + fold_index * 10_000
        output = _filter_factorized_day(spec, day, model, fold_seed, particle_chunk)
        outputs[heldout_date] = output
        if heldout_date == str(evaluation["visualization_date"]):
            replay_seed = fold_seed
        valid = day.valid_interval
        valid_count = int(torch.count_nonzero(valid))
        total_valid += valid_count
        baseline_mark_score = torch.log(
            torch.clamp_min(
                model.probabilities[
                    day.previous_spread_bucket, day.mark_id
                ],
                1e-30,
            )
        )
        mark_gain = output.conditional_mark_score - baseline_mark_score
        _, _, option_margin = _dawson_threshold_margin(s_g, 0.05)
        posterior_sd = torch.sqrt(torch.clamp_min(output.variance[1:], 0.0))
        uncertainty_metric = 1.0 - posterior_sd / option_margin
        reductions = {
            "rescaling_mean": _block_reduce(
                day, output.time_rescaling, reduction="mean", block_minutes=block_minutes
            ),
            "rescaling_std": _block_reduce(
                day, output.time_rescaling, reduction="std", block_minutes=block_minutes
            ),
            "conditional_mark_gain": _block_reduce(
                day, mark_gain, reduction="mean", block_minutes=block_minutes
            ),
            "uncertainty_metric": _block_reduce(
                day, uncertainty_metric, reduction="median", block_minutes=block_minutes
            ),
        }
        canonical_blocks = reductions["rescaling_mean"][2]
        for position, block in enumerate(canonical_blocks):
            row: dict[str, Any] = {
                "fold": fold_index,
                "train_dates": ";".join(train_dates),
                "heldout_date": heldout_date,
                "block": int(block),
                "valid_events": int(reductions["rescaling_mean"][1][position]),
                "option_margin_usdt": option_margin,
                "train_s_g_usdt": s_g,
                "train_sigma_x_usdt_per_sqrt_second": sigma_x,
            }
            for metric, (metric_values, _, metric_blocks) in reductions.items():
                location = torch.nonzero(
                    metric_blocks == block, as_tuple=False
                ).squeeze(-1)
                row[metric] = float(metric_values[location[0]]) if location.numel() else ""
            block_rows.append(row)

        predicted_categories = _predicted_categories(day, output, model)
        tv_by_name: dict[str, float] = {}
        for name, (actual_sum, predicted_sum) in predicted_categories.items():
            if name not in category_sums:
                category_sums[name] = (
                    torch.zeros_like(actual_sum),
                    torch.zeros_like(predicted_sum),
                )
            category_sums[name][0].add_(actual_sum)
            category_sums[name][1].add_(predicted_sum)
            total = torch.clamp_min(actual_sum.sum(), 1.0)
            actual_probability = actual_sum / total
            predicted_probability = predicted_sum / total
            tv = 0.5 * torch.sum(torch.abs(actual_probability - predicted_probability))
            tv_by_name[name] = float(tv)
            for category in range(actual_sum.numel()):
                distribution_fold_rows.append(
                    {
                        "heldout_date": heldout_date,
                        "distribution": name,
                        "category": category,
                        "actual_probability": float(actual_probability[category]),
                        "predicted_probability": float(predicted_probability[category]),
                    }
                )
        valid_rescaling = output.time_rescaling[valid]
        duration_actual.append(day.dt_seconds[valid].to(torch.float64))
        duration_mean.append(output.clock_log_mean[valid])
        duration_scale.append(output.clock_log_scale[valid])
        rescaling_all.append(valid_rescaling)
        posterior_ratio = torch.median(posterior_sd[valid] / option_margin)
        day_rows.append(
            {
                "fold": fold_index,
                "heldout_date": heldout_date,
                "valid_events": valid_count,
                "rescaling_mean": float(torch.mean(valid_rescaling)),
                "rescaling_sd": float(torch.std(valid_rescaling, unbiased=True)),
                "mark_log_score_gain": float(torch.mean(mark_gain[valid])),
                "posterior_sd_to_margin_ratio": float(posterior_ratio),
                "direction_total_variation": tv_by_name["direction"],
                "family_total_variation": tv_by_name["family"],
                "midpoint_magnitude_total_variation": tv_by_name["midpoint_magnitude"],
                "spread_magnitude_total_variation": tv_by_name["spread_magnitude"],
                "filter_digest_sha256": output.digest,
            }
        )
        parameter_rows.append(
            {
                "fold": fold_index,
                "train_dates": ";".join(train_dates),
                "heldout_date": heldout_date,
                "directional_beta": float(model.directional_beta),
                "clock_train_log_mean": float(model.clock_log_mean),
                "clock_train_log_sd": float(torch.sqrt(model.clock_log_variance)),
                "s_g_usdt": model.s_g,
                "sigma_x_usdt_per_sqrt_second": model.sigma_x,
                "fit_initial_mark_nll_nat_per_event": model.fit_initial_nll,
                "fit_final_mark_nll_nat_per_event": model.fit_final_nll,
                "parameter_digest_sha256": model.parameter_digest,
            }
        )

    visualization_date = str(evaluation["visualization_date"])
    if replay_seed is None:
        raise ExperimentError("visualization/replay fold was not evaluated")
    replay = _filter_factorized_day(
        spec,
        days[visualization_date],
        models[visualization_date],
        replay_seed,
        particle_chunk,
    )
    primary = outputs[visualization_date]
    deterministic_replay = replay.digest == primary.digest and all(
        torch.equal(left, right)
        for left, right in (
            (replay.estimate, primary.estimate),
            (replay.variance, primary.variance),
            (replay.conditional_mark_score, primary.conditional_mark_score),
            (replay.time_rescaling, primary.time_rescaling),
            (
                replay.predicted_direction_probabilities,
                primary.predicted_direction_probabilities,
            ),
        )
    )

    block_device = days[ordered_dates[0]].timestamps_ms.device
    vectors = {
        name: torch.tensor(
            [float(row[name]) for row in block_rows if row[name] != ""],
            device=block_device,
            dtype=torch.float64,
        )
        for name in (
            "rescaling_mean",
            "rescaling_std",
            "conditional_mark_gain",
            "uncertainty_metric",
        )
    }
    minimum_valid = int(evaluation["minimum_valid_blocks"])
    precision_sufficient = min(vector.numel() for vector in vectors.values()) >= minimum_valid
    clock_rows = [
        _equivalence_row(
            vectors["rescaling_mean"],
            float(evaluation["rescaling_mean_target"]),
            float(evaluation["rescaling_mean_equivalence_margin"]),
            float(evaluation["clock_per_metric_alpha"]),
            "time_rescaling_block_mean",
        ),
        _equivalence_row(
            vectors["rescaling_std"],
            float(evaluation["rescaling_standard_deviation_target"]),
            float(evaluation["rescaling_standard_deviation_equivalence_margin"]),
            float(evaluation["clock_per_metric_alpha"]),
            "time_rescaling_block_standard_deviation",
        ),
    ]
    mark_row = _superiority_row(
        vectors["conditional_mark_gain"],
        float(evaluation["mark_signal_minimum_effect_nat_per_event"]),
        float(evaluation["mark_signal_alpha"]),
        "conditional_mark_log_score_gain_nat_per_event",
    )
    state_row = _superiority_row(
        vectors["uncertainty_metric"],
        float(evaluation["uncertainty_margin_metric_minimum"]),
        float(evaluation["state_usability_alpha"]),
        "one_minus_posterior_sd_over_option_margin",
    )
    clock_passed = precision_sufficient and all(
        row["status"] == "equivalent" for row in clock_rows
    )
    mark_passed = precision_sufficient and mark_row["status"] == "superior"
    state_passed = precision_sufficient and state_row["status"] == "superior"

    all_outputs = list(outputs.values()) + [replay]
    all_finite = all(
        bool(torch.all(torch.isfinite(tensor)))
        for output in all_outputs
        for tensor in (
            output.estimate,
            output.variance,
            output.conditional_mark_score,
            output.clock_log_density,
            output.time_rescaling,
            output.clock_log_mean,
            output.clock_log_scale,
            output.predicted_direction_probabilities,
        )
    )
    positive_variance = all(
        bool(torch.all(output.variance > 0.0)) for output in all_outputs
    )
    frozen_parameters = parameter_digests == [
        models[date].parameter_digest for date in ordered_dates[1:]
    ]

    actual_duration = torch.cat(duration_actual)
    predictive_mean = torch.cat(duration_mean)
    predictive_scale = torch.cat(duration_scale)
    all_rescaling = torch.cat(rescaling_all)
    generator = torch.Generator(device=actual_duration.device).manual_seed(
        master_seed + int(evaluation["predictive_duration_seed_offset"])
    )
    predictive_duration = torch.exp(
        predictive_mean
        + predictive_scale
        * torch.randn(
            predictive_mean.shape,
            device=predictive_mean.device,
            dtype=torch.float64,
            generator=generator,
        )
    )
    distribution_rows = _histogram_rows(
        actual_duration, predictive_duration, all_rescaling, category_sums
    )
    timeseries_rows = _timeseries_rows(
        days[visualization_date],
        primary,
        int(evaluation["visualization_minutes"]),
    )
    write_csv(
        run_directory / "figures" / "timeseries-data.csv",
        _union_fieldnames(timeseries_rows),
        timeseries_rows,
    )
    write_csv(
        run_directory / "figures" / "predictive-distributions-data.csv",
        _union_fieldnames(distribution_rows),
        distribution_rows,
    )
    write_csv(
        run_directory / "figures" / "fold-calibration-data.csv",
        _union_fieldnames(day_rows),
        day_rows,
    )
    _render_figures(run_directory, timeseries_rows, distribution_rows, day_rows)
    visual_complete = all(
        path.is_file()
        for path in (
            run_directory / "figures" / "timeseries.png",
            run_directory / "figures" / "predictive-distributions.png",
            run_directory / "figures" / "fold-calibration.png",
            run_directory / "figures" / "timeseries-data.csv",
            run_directory / "figures" / "predictive-distributions-data.csv",
            run_directory / "figures" / "fold-calibration-data.csv",
        )
    )

    inference_rows = clock_rows + [mark_row, state_row]
    write_csv(
        run_directory / "metrics" / "block_metrics.csv",
        _union_fieldnames(block_rows),
        block_rows,
    )
    write_csv(
        run_directory / "metrics" / "day_diagnostics.csv",
        _union_fieldnames(day_rows),
        day_rows,
    )
    write_csv(
        run_directory / "tables" / "fold_parameters.csv",
        _union_fieldnames(parameter_rows),
        parameter_rows,
    )
    write_csv(
        run_directory / "tables" / "distribution_calibration.csv",
        _union_fieldnames(distribution_fold_rows),
        distribution_fold_rows,
    )
    write_csv(
        run_directory / "tables" / "inference.csv",
        _union_fieldnames(inference_rows),
        inference_rows,
    )
    inputs = values["inputs"]
    atomic_write_json(
        run_directory / "metrics" / "dependency_audit.json",
        {
            "schema_version": "p6d-dependency-audit-v1",
            "p6_dependency_run": str(inputs["p6_dependency_run"]),
            "verified_processed_assets": len(dependency_rows),
            "synthetic_dependency_run": synthetic_run,
            "synthetic_dependency_manifest_sha256": synthetic_manifest_hash,
            "p6c_reference_run": str(inputs["p6c_reference_run"]),
            "p6c_reference_manifest_sha256": str(
                inputs["p6c_reference_manifest_sha256"]
            ),
            "p6c_reference_summary_sha256": str(
                inputs["p6c_reference_summary_sha256"]
            ),
        },
    )
    atomic_write_json(
        run_directory / "metrics" / "replay.json",
        {
            "schema_version": "p6d-replay-v1",
            "date": visualization_date,
            "first_digest_sha256": primary.digest,
            "second_digest_sha256": replay.digest,
            "exact_tensor_equality": deterministic_replay,
            "seed": replay_seed,
        },
    )
    atomic_write_json(
        run_directory / "metrics" / "timestamp_audit.json",
        {
            "schema_version": "timestamp-audit-v1",
            "future_timestamp_accesses": future_accesses,
            "clock_current_duration_accesses": 0,
            "spot_role": str(values["model"]["spot_role"]),
            "equality_allowed": True,
        },
    )
    visualization_day = days[visualization_date]
    _save_torch_artifact(
        {
            "schema_version": "p6d-december-factorized-state-v1",
            "date": visualization_date,
            "timestamps_ms": visualization_day.timestamps_ms.to("cpu"),
            "bid_ticks": visualization_day.bid_ticks.to("cpu"),
            "ask_ticks": visualization_day.ask_ticks.to("cpu"),
            "valid_interval": visualization_day.valid_interval.to("cpu"),
            "mark_id": visualization_day.mark_id.to("cpu"),
            "filtered_efficient_price": primary.estimate.to("cpu"),
            "posterior_variance": primary.variance.to("cpu"),
            "clock_log_mean": primary.clock_log_mean.to("cpu"),
            "clock_log_scale": primary.clock_log_scale.to("cpu"),
            "time_rescaling": primary.time_rescaling.to("cpu"),
            "predicted_direction_probabilities": primary.predicted_direction_probabilities.to(
                "cpu"
            ),
        },
        run_directory / "state" / "december_factorized_filter.pt",
    )

    elapsed = time.perf_counter() - started
    acceptance = {
        "synthetic_dependency_passed": True,
        "dependency_hashes": True,
        "full_healthy_transition_support": total_valid
        == sum(int(torch.count_nonzero(days[date].valid_interval)) for date in ordered_dates[1:]),
        "zero_future_timestamp_accesses": future_accesses == 0,
        "deterministic_replay": deterministic_replay,
        "all_filter_values_finite": all_finite,
        "positive_posterior_variance": positive_variance,
        "frozen_fold_parameters": frozen_parameters,
        "precision_budget": precision_sufficient,
        "clock_calibration_equivalence": clock_passed,
        "conditional_mark_superiority": mark_passed,
        "state_usability": state_passed,
        "complete_visual_diagnostics": visual_complete,
        "wall_time_within_limit": elapsed < float(evaluation["maximum_wall_seconds"]),
    }
    pooled_category_tv = {
        name: float(
            0.5
            * torch.sum(
                torch.abs(
                    actual / torch.clamp_min(actual.sum(), 1.0)
                    - predicted / torch.clamp_min(actual.sum(), 1.0)
                )
            )
        )
        for name, (actual, predicted) in category_sums.items()
    }
    metrics = {
        "synthetic_dependency_run": synthetic_run,
        "fold_count": len(ordered_dates) - 1,
        "valid_blocks": int(vectors["rescaling_mean"].numel()),
        "valid_healthy_transitions": total_valid,
        "future_timestamp_accesses": future_accesses,
        "clock_rescaling_block_mean": float(clock_rows[0]["mean"]),
        "clock_rescaling_block_mean_interval": [
            float(clock_rows[0]["lower_bound"]),
            float(clock_rows[0]["upper_bound"]),
        ],
        "clock_rescaling_block_sd": float(clock_rows[1]["mean"]),
        "clock_rescaling_block_sd_interval": [
            float(clock_rows[1]["lower_bound"]),
            float(clock_rows[1]["upper_bound"]),
        ],
        "conditional_mark_gain_mean_nat_per_event": float(mark_row["mean"]),
        "conditional_mark_gain_lower_bound": float(mark_row["lower_bound"]),
        "uncertainty_metric_mean": float(state_row["mean"]),
        "uncertainty_metric_lower_bound": float(state_row["lower_bound"]),
        "posterior_sd_to_margin_ratio_mean_block": 1.0 - float(state_row["mean"]),
        "directional_beta_by_fold": [
            float(models[date].directional_beta) for date in ordered_dates[1:]
        ],
        "distribution_total_variation": pooled_category_tv,
        "deterministic_replay": deterministic_replay,
        "december_filter_digest_sha256": primary.digest,
        "elapsed_seconds": elapsed,
    }
    return MarkedEvaluationResult(
        metrics=metrics,
        acceptance=acceptance,
        derived_parameters={
            "mark_count": MARK_COUNT,
            "fold_count": len(ordered_dates) - 1,
            "clock_history_events": int(values["model"]["clock_history_events"]),
            "clock_training_prior_events": float(
                values["model"]["clock_training_prior_events"]
            ),
            "cuda_device": torch.cuda.get_device_name(),
            "factorization": str(values["model"]["factorization"]),
        },
        log_lines=(
            f"folds={len(ordered_dates)-1}; blocks={vectors['rescaling_mean'].numel()}",
            f"clock_mean={clock_rows[0]['mean']:.6g}; clock_sd={clock_rows[1]['mean']:.6g}",
            f"mark_gain={mark_row['mean']:.6g}; lower={mark_row['lower_bound']:.6g}",
            f"uncertainty_metric={state_row['mean']:.6g}; lower={state_row['lower_bound']:.6g}",
            f"replay={deterministic_replay}; digest={primary.digest}",
            f"acceptance_passed={all(acceptance.values())}",
        ),
    )
