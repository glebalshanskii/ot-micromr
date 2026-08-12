from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch

from ot_micromr.artifacts import atomic_write_json, sha256_file, write_csv
from ot_micromr.config import load_runspec
from ot_micromr.marked_filter import (
    MARK_COUNT,
    _empirical_probability_tables,
    _load_verified_p6_payloads,
    _prepare_empirical_marked_days,
    _stack_train_intervals,
    mark_metadata,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "cfg" / "experiments" / "emp_mark_filter_001.toml"
REJECTION_ATTEMPTS = 64


@dataclass(frozen=True, slots=True)
class ExactTransitionCsr:
    order: torch.Tensor
    offsets: torch.Tensor
    counts: torch.Tensor
    delta_bid: torch.Tensor
    delta_ask: torch.Tensor
    observed_cells: torch.Tensor


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot free-running several-event P6M forecasts from every held-out BBO point."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--start-utc", default="2024-12-15T00:00:00Z")
    parser.add_argument("--window-minutes", type=int, default=2)
    parser.add_argument("--horizon-events", type=int, default=10)
    parser.add_argument("--paths", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=202608121200)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args(argv)


def _load_full_fold(
    run_directory: Path, date: str, device: torch.device
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, str]:
    with (run_directory / "tables" / "fold_parameters.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    row = next(item for item in rows if item["heldout_date"] == date and item["model"] == "full")
    rates = torch.tensor(
        [
            float(row[f"baseline_rate_spread_{index + 1 if index < 7 else '8plus'}"])
            for index in range(8)
        ],
        device=device,
        dtype=torch.float32,
    )
    return (
        row["train_dates"].split(";"),
        rates,
        torch.tensor(float(row["alpha_per_second"]), device=device, dtype=torch.float32),
        torch.tensor(
            float(row["sigma_x_usdt_per_sqrt_second"]), device=device, dtype=torch.float32
        ),
        row["parameter_digest_sha256"],
    )


def _exact_transition_csr(train: tuple[torch.Tensor, ...]) -> ExactTransitionCsr:
    _, spread, mark, _, delta_y, delta_d, *_ = train
    delta_bid = torch.div(delta_y - delta_d, 2, rounding_mode="floor").to(torch.int64)
    delta_ask = torch.div(delta_y + delta_d, 2, rounding_mode="floor").to(torch.int64)
    direction = mark_metadata(spread.device)[0][mark]
    mirrored_bid = -delta_ask
    mirrored_ask = -delta_bid
    mirrored_mark = torch.where(
        direction != 0,
        mark + (-2 * direction.to(torch.int64)) * 3 * 9 * 9,
        mark,
    )
    augmented_spread = torch.cat((spread, spread))
    augmented_mark = torch.cat((mark, mirrored_mark))
    augmented_bid = torch.cat((delta_bid, mirrored_bid))
    augmented_ask = torch.cat((delta_ask, mirrored_ask))
    cell = augmented_spread * MARK_COUNT + augmented_mark
    counts = torch.bincount(cell, minlength=8 * MARK_COUNT).to(torch.int64)
    offsets = torch.cumsum(counts, dim=0) - counts
    order = torch.argsort(cell, stable=True)
    return ExactTransitionCsr(
        order=order,
        offsets=offsets,
        counts=counts,
        delta_bid=augmented_bid,
        delta_ask=augmented_ask,
        observed_cells=counts.reshape(8, MARK_COUNT) > 0,
    )


def _eligible_origins(
    timestamps_ms: torch.Tensor,
    valid_interval: torch.Tensor,
    start_ms: int,
    stop_ms: int,
    horizon: int,
) -> torch.Tensor:
    candidates = torch.arange(
        0, valid_interval.numel() - horizon + 1,
        device=valid_interval.device,
        dtype=torch.int64,
    )
    invalid = (~valid_interval).to(torch.int64)
    cumulative = torch.cat(
        (torch.zeros(1, device=invalid.device, dtype=torch.int64), torch.cumsum(invalid, dim=0))
    )
    consecutive = cumulative[candidates + horizon] == cumulative[candidates]
    in_window = (timestamps_ms[candidates] >= start_ms) & (timestamps_ms[candidates] < stop_ms)
    return candidates[consecutive & in_window]


def _make_rollout_step(
    compile_mode: str, rejection_attempts: int = REJECTION_ATTEMPTS
):
    def step(
        bid_ticks: torch.Tensor,
        ask_ticks: torch.Tensor,
        efficient_price: torch.Tensor,
        mark_uniform: torch.Tensor,
        raw_uniform: torch.Tensor,
        waiting_uniform: torch.Tensor,
        brownian_normal: torch.Tensor,
        probabilities: torch.Tensor,
        correction: torch.Tensor,
        rates: torch.Tensor,
        alpha: torch.Tensor,
        sigma_x: torch.Tensor,
        direction: torch.Tensor,
        observed_cells: torch.Tensor,
        raw_order: torch.Tensor,
        raw_offsets: torch.Tensor,
        raw_counts: torch.Tensor,
        raw_delta_bid: torch.Tensor,
        raw_delta_ask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        shape = bid_ticks.shape
        flat_bid = bid_ticks.reshape(-1)
        flat_ask = ask_ticks.reshape(-1)
        flat_x = efficient_price.reshape(-1)
        spread_ticks = flat_ask - flat_bid
        spread_bucket = torch.clamp(spread_ticks, min=1, max=8) - 1
        midpoint = (flat_bid + flat_ask).to(torch.float32) * 0.05
        gap = midpoint - flat_x
        directional_gap = torch.where(
            direction.unsqueeze(0) > 0,
            torch.clamp_min(-gap.unsqueeze(1), 0.0),
            torch.where(
                direction.unsqueeze(0) < 0,
                torch.clamp_min(gap.unsqueeze(1), 0.0),
                torch.zeros_like(gap).unsqueeze(1),
            ),
        )
        intensity = rates[spread_bucket].unsqueeze(1) * probabilities[spread_bucket]
        intensity = intensity + alpha * correction[spread_bucket] * directional_gap
        intensity = intensity * observed_cells[spread_bucket]
        cumulative = torch.cumsum(intensity, dim=1).contiguous()
        total = cumulative[:, -1]
        thresholds = (mark_uniform.reshape(rejection_attempts, -1).T * total.unsqueeze(1)).contiguous()
        candidate_mark = torch.searchsorted(cumulative, thresholds, right=False).T
        candidate_mark = torch.clamp_max(candidate_mark, MARK_COUNT - 1)
        candidate_cell = spread_bucket.unsqueeze(0) * MARK_COUNT + candidate_mark
        count = raw_counts[candidate_cell]
        rank = torch.floor(raw_uniform.reshape(rejection_attempts, -1) * count).to(torch.int64)
        rank = torch.minimum(torch.clamp_min(rank, 0), torch.clamp_min(count - 1, 0))
        transition_index = raw_order[raw_offsets[candidate_cell] + rank]
        candidate_bid_delta = raw_delta_bid[transition_index]
        candidate_ask_delta = raw_delta_ask[transition_index]
        candidate_spread = (
            spread_ticks.unsqueeze(0) + candidate_ask_delta - candidate_bid_delta
        )
        valid = candidate_spread > 0
        any_valid = torch.any(valid, dim=0)
        first_valid = torch.argmax(valid.to(torch.int64), dim=0)
        gather = first_valid.unsqueeze(0)
        selected_bid_delta = torch.gather(candidate_bid_delta, 0, gather).squeeze(0)
        selected_ask_delta = torch.gather(candidate_ask_delta, 0, gather).squeeze(0)
        selected_bid_delta = torch.where(any_valid, selected_bid_delta, torch.zeros_like(selected_bid_delta))
        selected_ask_delta = torch.where(any_valid, selected_ask_delta, torch.zeros_like(selected_ask_delta))
        waiting_time = -torch.log(torch.clamp_min(waiting_uniform.reshape(-1), 1e-7)) / total
        next_x = flat_x + sigma_x * torch.sqrt(waiting_time) * brownian_normal.reshape(-1)
        return (
            (flat_bid + selected_bid_delta).reshape(shape),
            (flat_ask + selected_ask_delta).reshape(shape),
            next_x.reshape(shape),
            waiting_time.reshape(shape),
            torch.count_nonzero(~any_valid),
        )

    return torch.compile(step, mode=compile_mode, fullgraph=True)


def _rollout(
    *,
    origins: torch.Tensor,
    bid_ticks: torch.Tensor,
    ask_ticks: torch.Tensor,
    state_mean: torch.Tensor,
    state_variance: torch.Tensor,
    probabilities: torch.Tensor,
    correction: torch.Tensor,
    rates: torch.Tensor,
    alpha: torch.Tensor,
    sigma_x: torch.Tensor,
    transitions: ExactTransitionCsr,
    horizon: int,
    paths: int,
    seed: int,
    compile_mode: str,
) -> tuple[dict[str, torch.Tensor], int]:
    device = bid_ticks.device
    origin_count = origins.numel()
    generator = torch.Generator(device=device).manual_seed(seed)
    current_bid = bid_ticks[origins].unsqueeze(1).expand(-1, paths).clone()
    current_ask = ask_ticks[origins].unsqueeze(1).expand(-1, paths).clone()
    current_x = state_mean[origins].unsqueeze(1) + torch.sqrt(
        torch.clamp_min(state_variance[origins], 1e-12)
    ).unsqueeze(1) * torch.randn(
        (origin_count, paths), device=device, dtype=torch.float32, generator=generator
    )
    bids = [current_bid]
    asks = [current_ask]
    event_times = [torch.zeros((origin_count, paths), device=device, dtype=torch.float32)]
    direction = mark_metadata(device)[0]
    step = _make_rollout_step(compile_mode)
    fallback_count = 0
    for _ in range(horizon):
        torch.compiler.cudagraph_mark_step_begin()
        random_shape = (REJECTION_ATTEMPTS, origin_count, paths)
        mark_uniform = torch.rand(random_shape, device=device, generator=generator)
        raw_uniform = torch.rand(random_shape, device=device, generator=generator)
        waiting_uniform = torch.rand((origin_count, paths), device=device, generator=generator)
        brownian_normal = torch.randn(
            (origin_count, paths), device=device, dtype=torch.float32, generator=generator
        )
        current_bid, current_ask, current_x, waiting_time, fallbacks = step(
            current_bid,
            current_ask,
            current_x,
            mark_uniform,
            raw_uniform,
            waiting_uniform,
            brownian_normal,
            probabilities,
            correction,
            rates,
            alpha,
            sigma_x,
            direction,
            transitions.observed_cells,
            transitions.order,
            transitions.offsets,
            transitions.counts,
            transitions.delta_bid,
            transitions.delta_ask,
        )
        current_bid = current_bid.clone()
        current_ask = current_ask.clone()
        current_x = current_x.clone()
        waiting_time = waiting_time.clone()
        bids.append(current_bid)
        asks.append(current_ask)
        event_times.append(event_times[-1] + waiting_time)
        fallback_count += int(fallbacks)
    bid_tick_path = torch.stack(bids, dim=1)
    ask_tick_path = torch.stack(asks, dim=1)
    minimum_spread_ticks = torch.min(ask_tick_path - bid_tick_path)
    bid_path = bid_tick_path.to(torch.float32) * 0.1
    ask_path = ask_tick_path.to(torch.float32) * 0.1
    midpoint_path = 0.5 * (bid_path + ask_path)
    time_path = torch.stack(event_times, dim=1)
    return {
        "bid_mean": bid_path.mean(dim=2),
        "ask_mean": ask_path.mean(dim=2),
        "midpoint_mean": midpoint_path.mean(dim=2),
        "bid_q10": torch.quantile(bid_path, 0.1, dim=2),
        "bid_q90": torch.quantile(bid_path, 0.9, dim=2),
        "ask_q10": torch.quantile(ask_path, 0.1, dim=2),
        "ask_q90": torch.quantile(ask_path, 0.9, dim=2),
        "midpoint_q10": torch.quantile(midpoint_path, 0.1, dim=2),
        "midpoint_q90": torch.quantile(midpoint_path, 0.9, dim=2),
        "model_event_time_mean_seconds": time_path.mean(dim=2),
        "minimum_simulated_spread_ticks": minimum_spread_ticks,
    }, fallback_count


def _forecast(
    config_path: Path,
    run_directory: Path,
    start_ms: int,
    stop_ms: int,
    horizon: int,
    paths: int,
    seed: int,
) -> dict[str, Any]:
    spec = load_runspec(config_path.resolve())
    swap, spot, _ = _load_verified_p6_payloads(spec)
    days, future_accesses = _prepare_empirical_marked_days(spec, swap, spot)
    state_path = run_directory / "state" / "december_filter.pt"
    state = torch.load(
        state_path,
        map_location=spec.values["numerics"]["compute_device"],
        weights_only=True,
    )
    date = str(state["date"])
    day = days[date]
    device = day.timestamps_ms.device
    train_dates, rates, alpha, sigma_x, parameter_digest = _load_full_fold(
        run_directory, date, device
    )
    train = _stack_train_intervals(days, train_dates)
    probabilities, correction, _, _, _, _ = _empirical_probability_tables(
        train, float(spec.values["model"]["dirichlet_smoothing_beta"]), "full"
    )
    transitions = _exact_transition_csr(train)
    origins = _eligible_origins(
        day.timestamps_ms, day.valid_interval, start_ms, stop_ms, horizon
    )
    if origins.numel() == 0:
        raise RuntimeError("plot window has no origin with a complete healthy forecast horizon")
    paths_out, fallback_count = _rollout(
        origins=origins,
        bid_ticks=day.bid_ticks,
        ask_ticks=day.ask_ticks,
        state_mean=state["filtered_efficient_price"].to(device),
        state_variance=state["posterior_variance"].to(device),
        probabilities=probabilities,
        correction=correction,
        rates=rates,
        alpha=alpha,
        sigma_x=sigma_x,
        transitions=transitions,
        horizon=horizon,
        paths=paths,
        seed=seed,
        compile_mode=str(spec.values["numerics"]["compile_mode"]),
    )
    horizon_index = torch.arange(horizon + 1, device=device).unsqueeze(0)
    actual_index = origins.unsqueeze(1) + horizon_index
    actual_bid = day.bid_ticks[actual_index].to(torch.float32) * 0.1
    actual_ask = day.ask_ticks[actual_index].to(torch.float32) * 0.1
    actual_midpoint = 0.5 * (actual_bid + actual_ask)
    actual_timestamp = day.timestamps_ms[actual_index]
    actual_elapsed_seconds = (actual_timestamp - start_ms).to(torch.float64) / 1000.0
    actual = {"bid": actual_bid, "ask": actual_ask, "midpoint": actual_midpoint}
    metrics: dict[str, Any] = {
        "origin_count": int(origins.numel()),
        "horizon_events": horizon,
        "paths_per_origin": paths,
        "future_timestamp_accesses": future_accesses,
        "rejection_fallback_count": fallback_count,
        "minimum_simulated_spread_ticks": int(paths_out["minimum_simulated_spread_ticks"]),
    }
    horizon_rows = []
    for step_index in range(1, horizon + 1):
        row: dict[str, Any] = {"horizon_events": step_index}
        for name in ("bid", "ask", "midpoint"):
            model_error = torch.abs(paths_out[f"{name}_mean"][:, step_index] - actual[name][:, step_index])
            persistence_error = torch.abs(actual[name][:, 0] - actual[name][:, step_index])
            row[f"{name}_model_mae_usdt"] = float(model_error.mean())
            row[f"{name}_persistence_mae_usdt"] = float(persistence_error.mean())
        row["model_mean_elapsed_seconds"] = float(
            paths_out["model_event_time_mean_seconds"][:, step_index].mean()
        )
        row["actual_mean_elapsed_seconds"] = float(
            (actual_timestamp[:, step_index] - actual_timestamp[:, 0]).to(torch.float64).mean()
            / 1000.0
        )
        horizon_rows.append(row)
    metrics.update(
        midpoint_h1_model_mae_usdt=horizon_rows[0]["midpoint_model_mae_usdt"],
        midpoint_h1_persistence_mae_usdt=horizon_rows[0]["midpoint_persistence_mae_usdt"],
        midpoint_final_horizon_model_mae_usdt=horizon_rows[-1]["midpoint_model_mae_usdt"],
        midpoint_final_horizon_persistence_mae_usdt=horizon_rows[-1]["midpoint_persistence_mae_usdt"],
        final_horizon_model_mean_elapsed_seconds=horizon_rows[-1]["model_mean_elapsed_seconds"],
        final_horizon_actual_mean_elapsed_seconds=horizon_rows[-1]["actual_mean_elapsed_seconds"],
    )
    return {
        "date": date,
        "origins": origins,
        "paths": paths_out,
        "actual": actual,
        "actual_elapsed_seconds": actual_elapsed_seconds,
        "horizon_rows": horizon_rows,
        "metrics": metrics,
        "train_dates": train_dates,
        "parameter_digest_sha256": parameter_digest,
        "state_path": state_path,
    }


def _trajectory_segments(x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    return [torch.stack((x[index], y[index]), dim=1) for index in range(x.shape[0])]


def _render_trajectories(result: dict[str, Any], path: Path, start_text: str) -> None:
    actual_x = result["actual_elapsed_seconds"].to("cpu")
    forecast_x = actual_x[:, :1] + result["paths"]["model_event_time_mean_seconds"].to("cpu")
    figure, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, constrained_layout=True)
    for axis, name, title in zip(
        axes, ("bid", "ask", "midpoint"), ("Bid", "Ask", "Midpoint"), strict=True
    ):
        forecast = result["paths"][f"{name}_mean"].to("cpu")
        segments = _trajectory_segments(forecast_x, forecast)
        axis.add_collection(
            LineCollection(segments, colors="#ea580c", linewidths=0.8, alpha=0.22)
        )
        day_x = actual_x[:, 0]
        day_y = result["actual"][name][:, 0].to("cpu")
        tail_x = actual_x[-1, 1:]
        tail_y = result["actual"][name][-1, 1:].to("cpu")
        axis.plot(
            torch.cat((day_x, tail_x)),
            torch.cat((day_y, tail_y)),
            color="#172554",
            linewidth=1.5,
            label="Actual BBO",
        )
        axis.scatter(day_x, day_y, s=6, color="#172554", alpha=0.7)
        axis.autoscale()
        axis.set_title(title)
        axis.set_ylabel("USDT")
        axis.grid(alpha=0.25)
        axis.legend(
            handles=[
                plt.Line2D([], [], color="#172554", label="Actual BBO"),
                plt.Line2D(
                    [], [], color="#ea580c", alpha=0.7,
                    label=f"Conditional mean rollout ({result['metrics']['horizon_events']} events)",
                ),
            ],
            loc="upper left",
        )
    axes[-1].set_xlabel("Seconds after window start; forecast values do not use future BBO")
    metrics = result["metrics"]
    figure.suptitle(
        "P6M free-running forecasts from every held-out BBO point\n"
        f"{start_text}; {metrics['origin_count']} origins x {metrics['paths_per_origin']} paths; "
        f"midpoint MAE model/persistence: h=1 "
        f"{metrics['midpoint_h1_model_mae_usdt']:.3f}/"
        f"{metrics['midpoint_h1_persistence_mae_usdt']:.3f}, h={metrics['horizon_events']} "
        f"{metrics['midpoint_final_horizon_model_mae_usdt']:.3f}/"
        f"{metrics['midpoint_final_horizon_persistence_mae_usdt']:.3f} USDT; "
        f"h={metrics['horizon_events']} time model/actual "
        f"{metrics['final_horizon_model_mean_elapsed_seconds']:.2f}/"
        f"{metrics['final_horizon_actual_mean_elapsed_seconds']:.2f} s",
        fontsize=12,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _render_horizon_errors(rows: list[dict[str, Any]], path: Path) -> None:
    horizon = torch.tensor([row["horizon_events"] for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, constrained_layout=True)
    for axis, name, title in zip(
        axes, ("bid", "ask", "midpoint"), ("Bid", "Ask", "Midpoint"), strict=True
    ):
        model = torch.tensor([row[f"{name}_model_mae_usdt"] for row in rows])
        persistence = torch.tensor([row[f"{name}_persistence_mae_usdt"] for row in rows])
        axis.plot(horizon, model, marker="o", color="#ea580c", label="P6M rollout")
        axis.plot(horizon, persistence, marker="o", color="#64748b", label="Persistence")
        axis.set_title(title)
        axis.set_xlabel("Forecast horizon, BBO events")
        axis.set_ylabel("MAE, USDT")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Several-event forecast error by horizon", fontsize=13)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    start = datetime.fromisoformat(arguments.start_utc.replace("Z", "+00:00")).astimezone(UTC)
    start_ms = int(start.timestamp() * 1000)
    stop_ms = start_ms + arguments.window_minutes * 60_000
    run_directory = arguments.run_directory.resolve()
    output_directory = arguments.output_directory
    if output_directory is None:
        output_directory = (
            REPOSITORY_ROOT
            / "outputs"
            / "P6M-MULTISTEP-VIZ"
            / (
                f"{start.strftime('%Y%m%dT%H%M%SZ')}-{arguments.window_minutes}min-"
                f"h{arguments.horizon_events}-p{arguments.paths}"
            )
        )
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    result = _forecast(
        arguments.config,
        run_directory,
        start_ms,
        stop_ms,
        arguments.horizon_events,
        arguments.paths,
        arguments.seed,
    )
    trajectory_path = output_directory / "multistep-trajectories.png"
    error_path = output_directory / "horizon-mae.png"
    _render_trajectories(result, trajectory_path, arguments.start_utc)
    _render_horizon_errors(result["horizon_rows"], error_path)
    write_csv(
        output_directory / "horizon-metrics.csv",
        list(result["horizon_rows"][0]),
        result["horizon_rows"],
    )
    atomic_write_json(
        output_directory / "provenance.json",
        {
            "schema_version": "p6m-multistep-visualization-v1",
            "role": "descriptive_only_not_an_acceptance_artifact",
            "forecast": "free-running Monte Carlo event-index rollout from each causal posterior",
            "initial_latent_state": "Gaussian moment approximation from persisted causal posterior mean and variance",
            "horizontal_alignment": "each rollout starts at the real origin timestamp and advances by model-expected event times; realized future timestamps are used only for error evaluation",
            "raw_transition_decoder": "train-only exact deltas conditional on spread bucket and symmetrized mark; positive-spread rejection",
            "date": result["date"],
            "start_utc": arguments.start_utc,
            "window_minutes": arguments.window_minutes,
            "horizon_events": arguments.horizon_events,
            "paths_per_origin": arguments.paths,
            "seed": arguments.seed,
            "train_dates": result["train_dates"],
            "metrics": result["metrics"],
            "source_run_directory": str(run_directory.relative_to(REPOSITORY_ROOT)),
            "source_manifest_sha256": sha256_file(run_directory / "manifest.json"),
            "source_state_sha256": sha256_file(result["state_path"]),
            "parameter_digest_sha256": result["parameter_digest_sha256"],
            "config_path": str(arguments.config.resolve().relative_to(REPOSITORY_ROOT)),
            "config_sha256": sha256_file(arguments.config.resolve()),
            "visualization_script_sha256": sha256_file(Path(__file__).resolve()),
            "trajectory_plot": trajectory_path.name,
            "error_plot": error_path.name,
        },
    )
    print(json.dumps({"output_directory": str(output_directory), **result["metrics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
