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
from ot_micromr.errors import ExperimentError


EVENT_HALF_TICK_DELTAS = torch.tensor([0, 2, -2, 1, -1, 1, -1], dtype=torch.int64)


@dataclass(frozen=True, slots=True)
class FilterEvaluationResult:
    metrics: Mapping[str, Any]
    acceptance: Mapping[str, bool]
    derived_parameters: Mapping[str, Any]
    log_lines: Sequence[str]

    @property
    def passed(self) -> bool:
        return all(self.acceptance.values())


def six_event_intensities(
    mid_price: torch.Tensor,
    efficient_price: torch.Tensor,
    is_tight: torch.Tensor,
    *,
    delta: float,
    mu_s: float,
    mu_o: float,
    mu_c: float,
    alpha_s: float,
    alpha_o: float,
    alpha_c: float,
) -> torch.Tensor:
    """Return paper-order intensities for arbitrary broadcast-compatible tensors."""
    gap = mid_price - efficient_price
    positive = torch.clamp_min(gap, 0.0)
    negative = torch.clamp_min(-gap, 0.0)
    scale = 2.0 / delta
    tight = is_tight.to(dtype=torch.bool)
    zero = torch.zeros_like(gap)
    return torch.stack(
        (
            torch.where(tight, mu_s + scale * alpha_s * negative, zero),
            torch.where(tight, mu_s + scale * alpha_s * positive, zero),
            torch.where(tight, mu_o + scale * alpha_o * negative, zero),
            torch.where(tight, mu_o + scale * alpha_o * positive, zero),
            torch.where(tight, zero, mu_c + scale * alpha_c * negative),
            torch.where(tight, zero, mu_c + scale * alpha_c * positive),
        ),
        dim=-1,
    )


def interval_log_likelihood(
    mid_price: torch.Tensor,
    efficient_price: torch.Tensor,
    is_tight: torch.Tensor,
    event_code: torch.Tensor,
    dt: float,
    model: Mapping[str, float],
) -> torch.Tensor:
    """Frozen-left event-or-silence log likelihood; event code 0 means silence."""
    rates = six_event_intensities(
        mid_price,
        efficient_price,
        is_tight,
        delta=float(model["delta_price"]),
        mu_s=float(model["mu_s_per_second"]),
        mu_o=float(model["mu_o_per_second"]),
        mu_c=float(model["mu_c_per_second"]),
        alpha_s=float(model["alpha_s_per_second"]),
        alpha_o=float(model["alpha_o_per_second"]),
        alpha_c=float(model["alpha_c_per_second"]),
    )
    total = rates.sum(dim=-1)
    code = event_code.to(torch.int64)
    selected = torch.gather(rates, -1, torch.clamp_min(code - 1, 0).unsqueeze(-1)).squeeze(-1)
    event_log = torch.log(-torch.expm1(-total * dt)) + torch.log(selected) - torch.log(total)
    return torch.where(code == 0, -total * dt, event_log)


def systematic_resample(
    particles: torch.Tensor, log_weights: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched systematic resampling over the last particle dimension."""
    particle_count = particles.shape[-1]
    weights = torch.softmax(log_weights, dim=-1)
    positions = (
        offsets.unsqueeze(-1)
        + torch.arange(particle_count, device=particles.device, dtype=particles.dtype)
    ) / particle_count
    indices = torch.searchsorted(torch.cumsum(weights, dim=-1), positions, right=False)
    indices = torch.clamp_max(indices, particle_count - 1)
    resampled = torch.gather(particles, -1, indices)
    reset_weights = torch.full_like(log_weights, -math.log(particle_count))
    return resampled, reset_weights


def _normal_lower_bound(values: torch.Tensor, alpha: float) -> tuple[float, float, float]:
    data = values.to(dtype=torch.float64, device="cpu")
    mean = data.mean()
    standard_error = data.std(unbiased=True) / math.sqrt(data.numel())
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - alpha, dtype=torch.float64))
    return float(mean), float(standard_error), float(mean - z * standard_error)


def _normal_equivalence_interval(
    values: torch.Tensor, alpha: float
) -> tuple[float, float, float, float]:
    data = values.to(dtype=torch.float64, device="cpu")
    mean = data.mean()
    standard_error = data.std(unbiased=True) / math.sqrt(data.numel())
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - alpha, dtype=torch.float64))
    return float(mean), float(standard_error), float(mean - z * standard_error), float(
        mean + z * standard_error
    )


def _tensor_digest(values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        contiguous = value.detach().to(device="cpu").contiguous()
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(bytes(contiguous.untyped_storage()))
    return digest.hexdigest()


def _make_simulator_chunk(
    model: Mapping[str, Any], dt: float, chunk_steps: int, *, compile_enabled: bool, compile_mode: str
):
    delta = float(model["delta_price"])
    sigma_step = float(model["sigma_x_price_per_sqrt_second"]) * math.sqrt(dt)
    mu_s = float(model["mu_s_per_second"])
    mu_o = float(model["mu_o_per_second"])
    mu_c = float(model["mu_c_per_second"])
    alpha_s = float(model["alpha_s_per_second"])
    alpha_o = float(model["alpha_o_per_second"])
    alpha_c = float(model["alpha_c_per_second"])

    def simulator_chunk(
        efficient_price: torch.Tensor,
        mid_half_ticks: torch.Tensor,
        brownian: torch.Tensor,
        occurrence_uniform: torch.Tensor,
        channel_uniform: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        session_count = efficient_price.shape[0]
        x_path = torch.empty(
            (chunk_steps, session_count), device=efficient_price.device, dtype=efficient_price.dtype
        )
        mid_path = torch.empty(
            (chunk_steps, session_count), device=mid_half_ticks.device, dtype=mid_half_ticks.dtype
        )
        event_path = torch.empty_like(mid_path)
        maximum_probability = efficient_price.new_zeros(())
        event_deltas = EVENT_HALF_TICK_DELTAS.to(device=mid_half_ticks.device)
        for step in range(chunk_steps):
            mid_price = mid_half_ticks.to(efficient_price.dtype) * (delta / 2.0)
            tight = torch.remainder(mid_half_ticks, 2) != 0
            rates = six_event_intensities(
                mid_price,
                efficient_price,
                tight,
                delta=delta,
                mu_s=mu_s,
                mu_o=mu_o,
                mu_c=mu_c,
                alpha_s=alpha_s,
                alpha_o=alpha_o,
                alpha_c=alpha_c,
            )
            total = rates.sum(dim=-1)
            occurrence_probability = -torch.expm1(-total * dt)
            maximum_probability = torch.maximum(maximum_probability, occurrence_probability.max())
            occurs = occurrence_uniform[step] < occurrence_probability
            conditional_cdf = torch.cumsum(rates, dim=-1) / total.unsqueeze(-1)
            selected = torch.sum(
                channel_uniform[step].unsqueeze(-1) > conditional_cdf, dim=-1
            ).to(torch.int64) + 1
            event_code = torch.where(occurs, selected, torch.zeros_like(selected))
            mid_half_ticks = mid_half_ticks + event_deltas[event_code]
            efficient_price = efficient_price + sigma_step * brownian[step]
            x_path[step] = efficient_price
            mid_path[step] = mid_half_ticks
            event_path[step] = event_code
        return efficient_price, mid_half_ticks, x_path, mid_path, event_path, maximum_probability

    if not compile_enabled:
        return simulator_chunk
    return torch.compile(simulator_chunk, mode=compile_mode, fullgraph=True)


def _make_particle_chunk(
    model: Mapping[str, Any], dt: float, chunk_steps: int, *, compile_enabled: bool, compile_mode: str
):
    delta = float(model["delta_price"])
    sigma_step = float(model["sigma_x_price_per_sqrt_second"]) * math.sqrt(dt)
    likelihood_model = {
        key: float(model[key])
        for key in (
            "delta_price",
            "mu_s_per_second",
            "mu_o_per_second",
            "mu_c_per_second",
            "alpha_s_per_second",
            "alpha_o_per_second",
            "alpha_c_per_second",
        )
    }

    def particle_chunk(
        particles: torch.Tensor,
        log_weights: torch.Tensor,
        start_mid_half_ticks: torch.Tensor,
        end_mid_half_ticks: torch.Tensor,
        events: torch.Tensor,
        transition_normals: torch.Tensor,
        resampling_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        particle_ends = particles.unsqueeze(0) + sigma_step * torch.cumsum(
            transition_normals, dim=0
        )
        particle_starts = torch.cat((particles.unsqueeze(0), particle_ends[:-1]), dim=0)
        start_mids = torch.cat(
            (start_mid_half_ticks.unsqueeze(0), end_mid_half_ticks[:-1]), dim=0
        )
        interval_scores = interval_log_likelihood(
            start_mids.to(particles.dtype).unsqueeze(-1) * (delta / 2.0),
            particle_starts,
            (torch.remainder(start_mids, 2) != 0).unsqueeze(-1),
            events.unsqueeze(-1).expand_as(particle_starts),
            dt,
            likelihood_model,
        )
        cumulative_log_weights = log_weights.unsqueeze(0) + torch.cumsum(
            interval_scores, dim=0
        )
        log_normalizers = torch.logsumexp(cumulative_log_weights, dim=-1)
        previous_normalizers = torch.cat(
            (log_normalizers.new_zeros((1, log_normalizers.shape[1])), log_normalizers[:-1]),
            dim=0,
        )
        predictive_scores = log_normalizers - previous_normalizers
        normalized_log_weights = cumulative_log_weights - log_normalizers.unsqueeze(-1)
        weights = torch.exp(normalized_log_weights)
        estimates = torch.sum(weights * particle_ends, dim=-1)
        variances = torch.sum(
            weights * torch.square(particle_ends - estimates.unsqueeze(-1)), dim=-1
        )
        particles, log_weights = systematic_resample(
            particle_ends[-1], normalized_log_weights[-1], resampling_offsets
        )
        return particles, log_weights, estimates, variances, predictive_scores

    if not compile_enabled:
        return particle_chunk
    return torch.compile(particle_chunk, mode=compile_mode, fullgraph=True)


def _make_kalman_chunk(
    *,
    dt: float,
    alpha: float,
    sigma_x: float,
    measurement_variance: float,
    chunk_steps: int,
    compile_enabled: bool,
    compile_mode: str,
):
    decay = math.exp(-alpha * dt)
    stationary_g_variance = sigma_x * sigma_x / (2.0 * alpha)
    q_x = sigma_x * sigma_x * dt
    q_g = stationary_g_variance * (1.0 - decay * decay)

    def kalman_chunk(
        state: torch.Tensor, covariance: torch.Tensor, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        session_count = state.shape[0]
        estimates = torch.empty(
            (chunk_steps, session_count), device=state.device, dtype=state.dtype
        )
        for step in range(chunk_steps):
            predicted_state = torch.stack((state[:, 0], decay * state[:, 1]), dim=-1)
            p00 = covariance[:, 0, 0] + q_x
            p01 = decay * covariance[:, 0, 1]
            p10 = decay * covariance[:, 1, 0]
            p11 = decay * decay * covariance[:, 1, 1] + q_g
            innovation_variance = p00 + p01 + p10 + p11 + measurement_variance
            gain0 = (p00 + p01) / innovation_variance
            gain1 = (p10 + p11) / innovation_variance
            innovation = observations[step] - predicted_state.sum(dim=-1)
            state = predicted_state + torch.stack((gain0 * innovation, gain1 * innovation), dim=-1)
            hp0 = p00 + p10
            hp1 = p01 + p11
            covariance = torch.stack(
                (
                    torch.stack((p00 - gain0 * hp0, p01 - gain0 * hp1), dim=-1),
                    torch.stack((p10 - gain1 * hp0, p11 - gain1 * hp1), dim=-1),
                ),
                dim=-2,
            )
            estimates[step] = state[:, 0]
        return state, covariance, estimates

    if not compile_enabled:
        return kalman_chunk
    return torch.compile(kalman_chunk, mode=compile_mode, fullgraph=True)


def _simulate_market(spec: RunSpec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    values = spec.values
    model = values["model"]
    simulation = values["simulation"]
    numerics = values["numerics"]
    device = torch.device(str(numerics["compute_device"]))
    session_count = int(simulation["session_count"])
    dt = float(simulation["time_step_seconds"])
    total_steps = round(
        (float(simulation["burn_in_seconds"]) + float(simulation["horizon_seconds"])) / dt
    )
    chunk_steps = int(numerics["chunk_steps"])
    master_seed = int(values["seed_policy"]["seeds"][0])
    generators = [torch.Generator(device=device).manual_seed(master_seed + offset) for offset in range(3)]
    x = torch.full(
        (session_count,), float(model["initial_efficient_price"]), device=device, dtype=torch.float32
    )
    mid = torch.full(
        (session_count,), int(model["initial_mid_half_ticks"]), device=device, dtype=torch.int64
    )
    x_path = torch.empty((total_steps, session_count), device=device, dtype=torch.float32)
    mid_path = torch.empty((total_steps, session_count), device=device, dtype=torch.int64)
    events = torch.empty_like(mid_path)
    simulator = _make_simulator_chunk(
        model,
        dt,
        chunk_steps,
        compile_enabled=bool(numerics["compile_enabled"]),
        compile_mode=str(numerics["compile_mode"]),
    )
    maximum_probability = 0.0
    for start in range(0, total_steps, chunk_steps):
        normals = torch.randn(
            (chunk_steps, session_count), device=device, dtype=torch.float32, generator=generators[0]
        )
        occurrence = torch.rand(
            (chunk_steps, session_count), device=device, dtype=torch.float32, generator=generators[1]
        )
        channel = torch.rand(
            (chunk_steps, session_count), device=device, dtype=torch.float32, generator=generators[2]
        )
        x, mid, x_chunk, mid_chunk, event_chunk, chunk_max = simulator(
            x, mid, normals, occurrence, channel
        )
        stop = start + chunk_steps
        x_path[start:stop] = x_chunk
        mid_path[start:stop] = mid_chunk
        events[start:stop] = event_chunk
        maximum_probability = max(maximum_probability, float(chunk_max))
        # Compiled reduce-overhead graphs reuse output storage on their next call.
        x = x.clone()
        mid = mid.clone()
    return x_path, mid_path, events, maximum_probability


def _filter_market(
    spec: RunSpec,
    mid_path: torch.Tensor,
    events: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    values = spec.values
    model = values["model"]
    simulation = values["simulation"]
    numerics = values["numerics"]
    device = mid_path.device
    total_steps, session_count = events.shape
    particle_count = int(numerics["particle_count"])
    chunk_steps = int(numerics["chunk_steps"])
    dt = float(simulation["time_step_seconds"])
    delta = float(model["delta_price"])
    master_seed = int(values["seed_policy"]["seeds"][0])
    initial_generator = torch.Generator(device=device).manual_seed(master_seed + 3)
    transition_generator = torch.Generator(device=device).manual_seed(master_seed + 4)
    resampling_generator = torch.Generator(device=device).manual_seed(master_seed + 5)
    initial_x = float(model["initial_efficient_price"])
    particles = initial_x + float(model["particle_initial_standard_deviation_price"]) * torch.randn(
        (session_count, particle_count), device=device, dtype=torch.float32, generator=initial_generator
    )
    log_weights = torch.full_like(particles, -math.log(particle_count))
    estimates = torch.empty_like(mid_path, dtype=torch.float32)
    variances = torch.empty_like(estimates)
    scores = torch.empty_like(estimates)
    naive_scores = torch.empty_like(estimates)
    oracle_placeholder = torch.empty_like(estimates)
    particle_chunk = _make_particle_chunk(
        model,
        dt,
        chunk_steps,
        compile_enabled=bool(numerics["compile_enabled"]),
        compile_mode=str(numerics["compile_mode"]),
    )
    initial_mid = torch.full(
        (session_count,), int(model["initial_mid_half_ticks"]), device=device, dtype=torch.int64
    )
    prior_mid = initial_mid
    for start in range(0, total_steps, chunk_steps):
        stop = start + chunk_steps
        transition = torch.randn(
            (chunk_steps, session_count, particle_count),
            device=device,
            dtype=torch.float32,
            generator=transition_generator,
        )
        offsets = torch.rand(
            (session_count,), device=device, dtype=torch.float32, generator=resampling_generator
        )
        particles, log_weights, estimate_chunk, variance_chunk, score_chunk = particle_chunk(
            particles,
            log_weights,
            prior_mid,
            mid_path[start:stop],
            events[start:stop],
            transition,
            offsets,
        )
        estimates[start:stop] = estimate_chunk
        variances[start:stop] = variance_chunk
        scores[start:stop] = score_chunk
        particles = particles.clone()
        log_weights = log_weights.clone()

        start_mids = torch.cat((prior_mid.unsqueeze(0), mid_path[start : stop - 1]), dim=0)
        naive_x = start_mids.to(torch.float32) * (delta / 2.0)
        naive_scores[start:stop] = interval_log_likelihood(
            naive_x,
            naive_x,
            torch.remainder(start_mids, 2) != 0,
            events[start:stop],
            dt,
            model,
        )
        prior_mid = mid_path[stop - 1]
    oracle_placeholder.zero_()
    digest = _tensor_digest((estimates, variances, scores))
    return estimates, variances, scores, naive_scores, oracle_placeholder, digest


def _kalman_filter(spec: RunSpec, mid_path: torch.Tensor) -> torch.Tensor:
    values = spec.values
    model = values["model"]
    numerics = values["numerics"]
    simulation = values["simulation"]
    total_steps, session_count = mid_path.shape
    chunk_steps = int(numerics["chunk_steps"])
    delta = float(model["delta_price"])
    initial_mid = int(model["initial_mid_half_ticks"]) * delta / 2.0
    state = torch.zeros((session_count, 2), device=mid_path.device, dtype=torch.float32)
    state[:, 0] = initial_mid
    covariance = torch.eye(2, device=mid_path.device, dtype=torch.float32).expand(
        session_count, -1, -1
    ).clone()
    estimates = torch.empty_like(mid_path, dtype=torch.float32)
    kalman_chunk = _make_kalman_chunk(
        dt=float(simulation["time_step_seconds"]),
        alpha=float(model["alpha_c_per_second"]),
        sigma_x=float(model["sigma_x_price_per_sqrt_second"]),
        measurement_variance=float(model["kalman_measurement_variance_price_squared"]),
        chunk_steps=chunk_steps,
        compile_enabled=bool(numerics["compile_enabled"]),
        compile_mode=str(numerics["compile_mode"]),
    )
    for start in range(0, total_steps, chunk_steps):
        stop = start + chunk_steps
        observations = mid_path[start:stop].to(torch.float32) * (delta / 2.0)
        state, covariance, chunk_estimates = kalman_chunk(state, covariance, observations)
        estimates[start:stop] = chunk_estimates
        state = state.clone()
        covariance = covariance.clone()
    return estimates


def evaluate_synthetic_filter(spec: RunSpec, run_directory: Path) -> FilterEvaluationResult:
    evaluation_started = time.perf_counter()
    if not torch.cuda.is_available():
        raise ExperimentError("FILTER-SYN-001 requires an available CUDA device")
    values = spec.values
    model = values["model"]
    simulation = values["simulation"]
    evaluation = values["evaluation"]
    dt = float(simulation["time_step_seconds"])
    burn_steps = round(float(simulation["burn_in_seconds"]) / dt)
    x_path, mid_path, events, maximum_event_probability = _simulate_market(spec)
    estimates, variances, scores, naive_scores, _, first_digest = _filter_market(
        spec, mid_path, events
    )
    replay_estimates, replay_variances, replay_scores, _, _, second_digest = _filter_market(
        spec, mid_path, events
    )
    deterministic_replay = (
        first_digest == second_digest
        and torch.equal(estimates, replay_estimates)
        and torch.equal(variances, replay_variances)
        and torch.equal(scores, replay_scores)
    )
    kalman_estimates = _kalman_filter(spec, mid_path)

    measured_x = x_path[burn_steps:]
    measured_mid = mid_path[burn_steps:].to(torch.float32) * (float(model["delta_price"]) / 2.0)
    measured_estimates = estimates[burn_steps:]
    measured_variances = variances[burn_steps:]
    measured_events = events[burn_steps:]
    event_counts = torch.count_nonzero(measured_events, dim=0)
    if torch.any(event_counts == 0):
        raise ExperimentError("synthetic session with zero measured book events")

    pf_rmse = torch.sqrt(torch.mean(torch.square(measured_estimates - measured_x), dim=0))
    naive_rmse = torch.sqrt(torch.mean(torch.square(measured_mid - measured_x), dim=0))
    kalman_rmse = torch.sqrt(
        torch.mean(torch.square(kalman_estimates[burn_steps:] - measured_x), dim=0)
    )
    oracle_rmse = torch.sqrt(torch.mean(torch.square(measured_x - measured_x), dim=0))
    state_improvement = 1.0 - pf_rmse / naive_rmse
    log_score_improvement = (
        torch.sum(scores[burn_steps:] - naive_scores[burn_steps:], dim=0) / event_counts
    )
    initial_x = torch.full_like(
        x_path[:1], float(model["initial_efficient_price"])
    )
    initial_mid = torch.full_like(
        mid_path[:1], int(model["initial_mid_half_ticks"])
    )
    x_starts = torch.cat((initial_x, x_path[:-1]), dim=0)
    mid_starts = torch.cat((initial_mid, mid_path[:-1]), dim=0)
    oracle_scores = interval_log_likelihood(
        mid_starts.to(torch.float32) * (float(model["delta_price"]) / 2.0),
        x_starts,
        torch.remainder(mid_starts, 2) != 0,
        events,
        dt,
        model,
    )
    oracle_log_score_advantage = (
        torch.sum(oracle_scores[burn_steps:] - naive_scores[burn_steps:], dim=0)
        / event_counts
    )
    z90 = torch.tensor(1.6448536269514722, device=measured_x.device, dtype=torch.float32)
    lower = measured_estimates - z90 * torch.sqrt(torch.clamp_min(measured_variances, 0.0))
    upper = measured_estimates + z90 * torch.sqrt(torch.clamp_min(measured_variances, 0.0))
    coverage = torch.mean(((measured_x >= lower) & (measured_x <= upper)).to(torch.float32), dim=0)

    state_mean, state_se, state_lower = _normal_lower_bound(
        state_improvement, float(evaluation["per_metric_alpha"])
    )
    log_mean, log_se, log_lower = _normal_lower_bound(
        log_score_improvement, float(evaluation["per_metric_alpha"])
    )
    coverage_mean, coverage_se, coverage_lower, coverage_upper = _normal_equivalence_interval(
        coverage, float(evaluation["calibration_alpha"])
    )
    coverage_target = float(evaluation["posterior_interval_nominal_coverage"])
    coverage_margin = float(evaluation["posterior_coverage_equivalence_margin"])
    all_finite = all(
        bool(torch.all(torch.isfinite(value)))
        for value in (
            x_path,
            estimates,
            variances,
            scores,
            naive_scores,
            kalman_estimates,
            state_improvement,
            log_score_improvement,
            coverage,
            oracle_scores,
        )
    )
    positive_variance = bool(torch.all(measured_variances > 0.0))
    evaluation_elapsed_seconds = time.perf_counter() - evaluation_started

    acceptance = {
        "state_superiority": state_lower > float(evaluation["state_minimum_effect"]),
        "log_score_superiority": log_lower
        > float(evaluation["log_score_minimum_effect_nat_per_event"]),
        "calibration_equivalence": coverage_lower > coverage_target - coverage_margin
        and coverage_upper < coverage_target + coverage_margin,
        "oracle_rmse_zero": float(oracle_rmse.max()) == 0.0,
        "deterministic_replay": deterministic_replay,
        "maximum_event_probability": maximum_event_probability
        < float(evaluation["maximum_event_probability"]),
        "all_values_finite": all_finite,
        "posterior_variance_positive": positive_variance,
        "wall_time_within_limit": evaluation_elapsed_seconds
        < float(evaluation["maximum_wall_seconds"]),
    }
    metrics = {
        "session_count": int(events.shape[1]),
        "measured_steps_per_session": int(measured_x.shape[0]),
        "measured_events_total": int(event_counts.sum()),
        "state_improvement_mean": state_mean,
        "state_improvement_standard_error": state_se,
        "state_improvement_bonferroni_lower_bound": state_lower,
        "log_score_improvement_nat_per_event_mean": log_mean,
        "log_score_improvement_standard_error": log_se,
        "log_score_improvement_bonferroni_lower_bound": log_lower,
        "posterior_90_coverage_mean": coverage_mean,
        "posterior_90_coverage_standard_error": coverage_se,
        "posterior_90_coverage_equivalence_lower": coverage_lower,
        "posterior_90_coverage_equivalence_upper": coverage_upper,
        "pf_rmse_mean": float(pf_rmse.mean()),
        "naive_rmse_mean": float(naive_rmse.mean()),
        "kalman_rmse_mean": float(kalman_rmse.mean()),
        "oracle_rmse_max": float(oracle_rmse.max()),
        "oracle_log_score_advantage_nat_per_event_mean": float(
            oracle_log_score_advantage.mean()
        ),
        "maximum_event_probability": maximum_event_probability,
        "evaluation_elapsed_seconds": evaluation_elapsed_seconds,
        "deterministic_replay": deterministic_replay,
        "filter_digest_sha256": first_digest,
    }

    session_rows = []
    for index in range(events.shape[1]):
        session_rows.append(
            {
                "session": index,
                "event_count": int(event_counts[index]),
                "pf_rmse": float(pf_rmse[index]),
                "naive_rmse": float(naive_rmse[index]),
                "kalman_rmse": float(kalman_rmse[index]),
                "oracle_rmse": float(oracle_rmse[index]),
                "state_improvement": float(state_improvement[index]),
                "log_score_improvement_nat_per_event": float(log_score_improvement[index]),
                "posterior_90_coverage": float(coverage[index]),
            }
        )
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
            "mean": log_mean,
            "standard_error": log_se,
            "lower_bound": log_lower,
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
    write_csv(
        run_directory / "tables" / "inference.csv", list(inference_rows[0]), inference_rows
    )
    atomic_write_json(
        run_directory / "metrics" / "replay.json",
        {
            "schema_version": "filter-replay-v1",
            "first_digest_sha256": first_digest,
            "second_digest_sha256": second_digest,
            "exact_tensor_equality": deterministic_replay,
            "seed_offsets": {
                "market_brownian": 0,
                "market_occurrence": 1,
                "market_channel": 2,
                "particle_initialization": 3,
                "particle_transition": 4,
                "systematic_resampling": 5,
            },
        },
    )
    return FilterEvaluationResult(
        metrics=metrics,
        acceptance=acceptance,
        derived_parameters={
            "time_step_seconds": dt,
            "burn_steps": burn_steps,
            "particle_count": int(values["numerics"]["particle_count"]),
            "cuda_device": torch.cuda.get_device_name(mid_path.device),
            "seed_offsets": "master_seed + integers 0..5; see metrics/replay.json",
        },
        log_lines=(
            f"state improvement mean={state_mean:.6g}; lower={state_lower:.6g}",
            f"log-score improvement={log_mean:.6g} nat/event; lower={log_lower:.6g}",
            f"posterior coverage={coverage_mean:.6g}; equivalence CI=[{coverage_lower:.6g}, {coverage_upper:.6g}]",
            f"deterministic_replay={deterministic_replay}; digest={first_digest}",
            f"acceptance_passed={all(acceptance.values())}",
        ),
    )
