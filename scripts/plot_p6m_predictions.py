from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
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


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a descriptive one-step P6M BBO forecast from a completed target run."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--start-utc", default="2024-12-15T00:00:00Z")
    parser.add_argument("--window-minutes", type=int, default=10)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args(argv)


def _decode_mark_means(train: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    _, spread, mark, _, delta_y, delta_d, *_ = train
    flat = spread * MARK_COUNT + mark
    cells = 8 * MARK_COUNT
    counts = torch.bincount(flat, minlength=cells).to(torch.float64).reshape(8, MARK_COUNT)
    sum_y = torch.zeros(cells, device=spread.device, dtype=torch.float64)
    sum_d = torch.zeros_like(sum_y)
    sum_y.scatter_add_(0, flat, delta_y.to(torch.float64))
    sum_d.scatter_add_(0, flat, delta_d.to(torch.float64))
    sum_y = sum_y.reshape(8, MARK_COUNT)
    sum_d = sum_d.reshape(8, MARK_COUNT)

    direction, family, midpoint_bucket, spread_bucket = mark_metadata(spread.device)
    representatives = counts.new_tensor((0.0, 1.0, 2.5, 5.5, 11.5, 23.5, 47.5, 95.5, 128.0))
    fallback_y = direction.to(torch.float64) * representatives[midpoint_bucket]
    fallback_d = family.to(torch.float64) * representatives[spread_bucket]
    mean_y = torch.where(counts > 0.0, sum_y / torch.clamp_min(counts, 1.0), fallback_y)
    mean_d = torch.where(counts > 0.0, sum_d / torch.clamp_min(counts, 1.0), fallback_d)
    mean_bid = 0.5 * (mean_y - mean_d)
    mean_ask = 0.5 * (mean_y + mean_d)
    return mean_bid.to(torch.float32), mean_ask.to(torch.float32), counts > 0.0


def _posterior_directional_gaps(
    gap_mean: torch.Tensor, posterior_variance: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    standard_deviation = torch.sqrt(torch.clamp_min(posterior_variance, 1e-12))
    z = gap_mean / standard_deviation
    density = torch.exp(-0.5 * torch.square(z)) / math.sqrt(2.0 * math.pi)
    distribution = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    positive = standard_deviation * density + gap_mean * distribution
    negative = standard_deviation * density - gap_mean * (1.0 - distribution)
    return torch.where(
        direction.unsqueeze(0) < 0,
        positive.unsqueeze(1),
        torch.where(
            direction.unsqueeze(0) > 0,
            negative.unsqueeze(1),
            torch.zeros_like(gap_mean).unsqueeze(1),
        ),
    )


def _load_fold_parameters(run_directory: Path, date: str, device: torch.device) -> tuple[Any, ...]:
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
        row["parameter_digest_sha256"],
    )


def _predict_window(
    config_path: Path,
    run_directory: Path,
    start_ms: int,
    stop_ms: int,
) -> dict[str, Any]:
    spec = load_runspec(config_path.resolve())
    swap, spot, _ = _load_verified_p6_payloads(spec)
    days, future_accesses = _prepare_empirical_marked_days(spec, swap, spot)
    state_path = run_directory / "state" / "december_filter.pt"
    state = torch.load(state_path, map_location=spec.values["numerics"]["compute_device"], weights_only=True)
    date = str(state["date"])
    day = days[date]
    device = day.timestamps_ms.device
    train_dates, rates, alpha, parameter_digest = _load_fold_parameters(
        run_directory, date, device
    )
    train = _stack_train_intervals(days, train_dates)
    probabilities, correction, _, _, _, _ = _empirical_probability_tables(
        train, float(spec.values["model"]["dirichlet_smoothing_beta"]), "full"
    )
    decoder_bid, decoder_ask, decoder_seen = _decode_mark_means(train)

    selected = (
        day.valid_interval
        & (day.timestamps_ms[1:] >= start_ms)
        & (day.timestamps_ms[1:] < stop_ms)
    )
    interval = torch.nonzero(selected, as_tuple=False).squeeze(-1)
    if interval.numel() < 2:
        raise RuntimeError("selected plotting window has fewer than two valid BBO transitions")

    state_mean = state["filtered_efficient_price"].to(device)[interval]
    state_variance = state["posterior_variance"].to(device)[interval]
    midpoint = day.prior_mid_price[interval]
    spread_bucket = day.previous_spread_bucket[interval]
    gap_mean = midpoint - state_mean
    direction = mark_metadata(device)[0]
    directional_gap = _posterior_directional_gaps(gap_mean, state_variance, direction)
    intensity = rates[spread_bucket].unsqueeze(1) * probabilities[spread_bucket]
    intensity = intensity + alpha * correction[spread_bucket] * directional_gap
    event_probability = intensity / torch.clamp_min(intensity.sum(dim=1, keepdim=True), 1e-30)

    predicted_bid_delta = torch.sum(
        event_probability * decoder_bid[spread_bucket], dim=1
    )
    predicted_ask_delta = torch.sum(
        event_probability * decoder_ask[spread_bucket], dim=1
    )
    price_tick = float(spec.values["model"]["price_tick"])
    actual_bid = day.bid_ticks[interval + 1].to(torch.float64) * price_tick
    actual_ask = day.ask_ticks[interval + 1].to(torch.float64) * price_tick
    persistence_bid = day.bid_ticks[interval].to(torch.float64) * price_tick
    persistence_ask = day.ask_ticks[interval].to(torch.float64) * price_tick
    predicted_bid = (
        day.bid_ticks[interval].to(torch.float64) + predicted_bid_delta.to(torch.float64)
    ) * price_tick
    predicted_ask = (
        day.ask_ticks[interval].to(torch.float64) + predicted_ask_delta.to(torch.float64)
    ) * price_tick
    actual_midpoint = 0.5 * (actual_bid + actual_ask)
    predicted_midpoint = 0.5 * (predicted_bid + predicted_ask)
    persistence_midpoint = 0.5 * (persistence_bid + persistence_ask)
    actual_spread = actual_ask - actual_bid
    predicted_spread = predicted_ask - predicted_bid
    persistence_spread = persistence_ask - persistence_bid
    known_decoder_mass = torch.sum(event_probability * decoder_seen[spread_bucket], dim=1)
    elapsed_minutes = (day.timestamps_ms[interval + 1] - start_ms).to(torch.float64) / 60_000.0

    tensors = {
        "elapsed_minutes": elapsed_minutes,
        "timestamps_ms": day.timestamps_ms[interval + 1],
        "actual_bid": actual_bid,
        "predicted_bid": predicted_bid,
        "persistence_bid": persistence_bid,
        "actual_ask": actual_ask,
        "predicted_ask": predicted_ask,
        "persistence_ask": persistence_ask,
        "actual_midpoint": actual_midpoint,
        "predicted_midpoint": predicted_midpoint,
        "persistence_midpoint": persistence_midpoint,
        "actual_spread": actual_spread,
        "predicted_spread": predicted_spread,
        "persistence_spread": persistence_spread,
        "known_decoder_mass": known_decoder_mass,
    }
    metrics = {
        "event_count": int(interval.numel()),
        "bid_mae_usdt": float(torch.mean(torch.abs(predicted_bid - actual_bid))),
        "ask_mae_usdt": float(torch.mean(torch.abs(predicted_ask - actual_ask))),
        "midpoint_mae_usdt": float(
            torch.mean(torch.abs(predicted_midpoint - actual_midpoint))
        ),
        "spread_mae_usdt": float(torch.mean(torch.abs(predicted_spread - actual_spread))),
        "persistence_bid_mae_usdt": float(torch.mean(torch.abs(persistence_bid - actual_bid))),
        "persistence_ask_mae_usdt": float(torch.mean(torch.abs(persistence_ask - actual_ask))),
        "persistence_midpoint_mae_usdt": float(
            torch.mean(torch.abs(persistence_midpoint - actual_midpoint))
        ),
        "persistence_spread_mae_usdt": float(
            torch.mean(torch.abs(persistence_spread - actual_spread))
        ),
        "nonpositive_predicted_spread_count": int(torch.count_nonzero(predicted_spread <= 0.0)),
        "mean_known_decoder_probability_mass": float(known_decoder_mass.mean()),
        "future_timestamp_accesses": future_accesses,
    }
    return {
        "date": date,
        "tensors": tensors,
        "metrics": metrics,
        "train_dates": train_dates,
        "parameter_digest_sha256": parameter_digest,
        "state_path": state_path,
    }


def _render(result: dict[str, Any], output_path: Path, start_text: str, minutes: int) -> None:
    tensors = {key: value.detach().to("cpu") for key, value in result["tensors"].items()}
    x = tensors["elapsed_minutes"]
    actual_color = "#172554"
    predicted_color = "#ea580c"
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, constrained_layout=True)
    panels = (
        ("Bid", "actual_bid", "predicted_bid", "persistence_bid", "USDT"),
        ("Ask", "actual_ask", "predicted_ask", "persistence_ask", "USDT"),
        (
            "Midpoint",
            "actual_midpoint",
            "predicted_midpoint",
            "persistence_midpoint",
            "USDT",
        ),
        ("Spread", "actual_spread", "predicted_spread", "persistence_spread", "USDT"),
    )
    for axis, (title, actual, predicted, persistence, unit) in zip(
        axes.flat, panels, strict=True
    ):
        axis.plot(x, tensors[actual], color=actual_color, linewidth=1.4, label="Actual next BBO")
        axis.plot(
            x,
            tensors[predicted],
            color=predicted_color,
            linewidth=1.1,
            linestyle="--",
            label="Predicted conditional mean",
        )
        axis.plot(
            x,
            tensors[persistence],
            color="#64748b",
            linewidth=0.8,
            linestyle=":",
            label="Previous BBO (persistence)",
        )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
    axes[1, 1].axhline(0.0, color="#991b1b", linewidth=0.8, alpha=0.7)
    axes[1, 0].set_xlabel("Minutes after window start")
    axes[1, 1].set_xlabel("Minutes after window start")
    metrics = result["metrics"]
    figure.suptitle(
        "P6M one-step BBO forecast on held-out 2024-12-15\n"
        f"{start_text}, {minutes} min; conditional on an event occurring\n"
        f"model/persistence midpoint MAE={metrics['midpoint_mae_usdt']:.3f}/"
        f"{metrics['persistence_midpoint_mae_usdt']:.3f} USDT; "
        f"spread MAE={metrics['spread_mae_usdt']:.3f}/"
        f"{metrics['persistence_spread_mae_usdt']:.3f} USDT; "
        f"nonpositive predicted spread={metrics['nonpositive_predicted_spread_count']}",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
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
            / "P6M-PREDICTION-VIZ"
            / f"{start.strftime('%Y%m%dT%H%M%SZ')}-{arguments.window_minutes}min"
        )
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)

    result = _predict_window(arguments.config, run_directory, start_ms, stop_ms)
    plot_path = output_directory / "bbo-midpoint-spread.png"
    _render(result, plot_path, arguments.start_utc, arguments.window_minutes)
    tensors = result["tensors"]
    fields = list(tensors)
    rows = []
    for index in range(tensors[fields[0]].numel()):
        row = {field: float(tensors[field][index]) for field in fields}
        row["timestamps_ms"] = int(tensors["timestamps_ms"][index])
        rows.append(row)
    write_csv(output_directory / "prediction-data.csv", fields, rows)
    atomic_write_json(
        output_directory / "provenance.json",
        {
            "schema_version": "p6m-prediction-visualization-v1",
            "role": "descriptive_only_not_an_acceptance_artifact",
            "forecast": "posterior-Gaussian-integrated conditional mark mean given an event",
            "decoder": "train-only exact-delta cell mean with fixed bucket representative for unseen cells",
            "date": result["date"],
            "start_utc": arguments.start_utc,
            "window_minutes": arguments.window_minutes,
            "train_dates": result["train_dates"],
            "metrics": result["metrics"],
            "source_run_directory": str(run_directory.relative_to(REPOSITORY_ROOT)),
            "source_manifest_sha256": sha256_file(run_directory / "manifest.json"),
            "source_state_sha256": sha256_file(result["state_path"]),
            "parameter_digest_sha256": result["parameter_digest_sha256"],
            "config_path": str(arguments.config.resolve().relative_to(REPOSITORY_ROOT)),
            "config_sha256": sha256_file(arguments.config.resolve()),
            "visualization_script_sha256": sha256_file(Path(__file__).resolve()),
            "plot_path": plot_path.name,
        },
    )
    print(
        json.dumps(
            {"output_directory": str(output_directory), **result["metrics"]},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
