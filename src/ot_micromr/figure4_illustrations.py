from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ot_micromr.analytics import solve_dawson_optimum
from ot_micromr.artifacts import write_csv
from ot_micromr.figure4 import CalibrationRow
from ot_micromr.figure4_market import Figure4MarketTrace, simulate_market_trace


def _sample_indices(length: int, maximum: int, required: set[int] | None = None) -> np.ndarray:
    base = np.linspace(0, max(length - 1, 0), min(length, maximum), dtype=np.int64)
    if required:
        base = np.concatenate((base, np.fromiter(required, dtype=np.int64)))
    return np.unique(base[(base >= 0) & (base < length)])


def _render_figure2(
    values: Mapping[str, Any], calibration: CalibrationRow, run_directory: Path
) -> None:
    local = copy.deepcopy(dict(values))
    local["model"] = copy.deepcopy(dict(values["model"]))
    local["simulation"] = copy.deepcopy(dict(values["simulation"]))
    local["model"]["mu_o_per_second"] = 0.2
    local["simulation"]["market_burn_in_reversion_times"] = 1.0
    local["simulation"]["strategy_burn_in_reversion_times"] = 1.0
    local["simulation"]["horizon_reversion_times"] = 30.0
    seed = int(values["seed_policy"]["strategy_seeds"][0]) + 100_000
    trace = simulate_market_trace(local, calibration, 0.01, seed)
    indices = _sample_indices(trace.left_time_seconds.size, 4_000)
    time_seconds = trace.left_time_seconds[indices] + trace.step_seconds[indices]
    mid = trace.post_event_mid_price[indices]
    spread = trace.post_event_spread_price[indices]
    gap = trace.post_event_gap_price[indices]
    efficient = mid - gap
    rows = [
        {
            "time_seconds": float(time),
            "efficient_price": float(x),
            "bid_price": float(local_mid - local_spread / 2.0),
            "ask_price": float(local_mid + local_spread / 2.0),
            "mid_price": float(local_mid),
            "gap_price": float(local_gap),
            "spread_price": float(local_spread),
        }
        for time, x, local_mid, local_gap, local_spread in zip(
            time_seconds, efficient, mid, gap, spread, strict=True
        )
    ]
    write_csv(
        run_directory / "figures" / "figure2-data.csv",
        list(rows[0].keys()),
        rows,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True, constrained_layout=True)
    axes[0].step(time_seconds, mid - spread / 2.0, where="post", label="bid", linewidth=0.8)
    axes[0].step(time_seconds, mid + spread / 2.0, where="post", label="ask", linewidth=0.8)
    axes[0].plot(time_seconds, efficient, label=r"efficient price $X$", linewidth=1.0)
    axes[0].set(ylabel="price", title="Figure 2 structural reconstruction (illustrative)")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(time_seconds, gap, linewidth=0.9, label=r"gap $G=M-X$")
    axes[1].step(time_seconds, spread, where="post", linewidth=0.8, label="spread")
    axes[1].set(xlabel="time (s)", ylabel="price", title="Gap and one/two-tick spread")
    axes[1].legend(frameon=False)
    figure.savefig(
        run_directory / "figures" / "figure2.png",
        dpi=180,
        metadata={"Software": "ot-micromr 0.1.0"},
    )
    plt.close(figure)


def _render_figure5(
    values: Mapping[str, Any],
    calibration: CalibrationRow,
    trace: Figure4MarketTrace,
    run_directory: Path,
) -> None:
    alpha = calibration.alpha_per_second
    start = (
        float(values["simulation"]["market_burn_in_reversion_times"])
        + float(values["simulation"]["strategy_burn_in_reversion_times"])
    ) / alpha
    end = start + min(80.0, float(values["simulation"]["horizon_reversion_times"])) / alpha
    first = int(np.searchsorted(trace.left_time_seconds, start, side="left"))
    last = int(np.searchsorted(trace.left_time_seconds, end, side="left"))
    time = trace.left_time_seconds[first:last] + trace.step_seconds[first:last]
    mid = trace.post_event_mid_price[first:last]
    spread = trace.post_event_spread_price[first:last]
    gap = trace.post_event_gap_price[first:last]
    efficient = mid - gap
    tight_threshold = calibration.theta_d_price
    open_gamma = float(values["model"]["delta_price"]) / calibration.s_g_price
    open_solution = solve_dawson_optimum(open_gamma, values["numerics"])
    open_threshold = calibration.s_g_price * open_solution.u_d_ratio
    threshold = np.where(
        spread <= float(values["model"]["delta_price"]),
        tight_threshold,
        open_threshold,
    )
    signals = np.where(gap >= threshold, -1, np.where(gap <= -threshold, 1, 0)).astype(
        np.int8
    )
    signal_indices = np.where(signals != 0, np.arange(time.size), -1)
    latest_signal_indices = np.maximum.accumulate(signal_indices)
    inventory = np.where(
        latest_signal_indices >= 0,
        signals[np.maximum(latest_signal_indices, 0)],
        0,
    ).astype(np.int8)
    inventory_changes = np.diff(np.concatenate((np.zeros(1, dtype=np.int8), inventory)))
    fill_indices = np.flatnonzero(inventory_changes)
    fills = set(int(index) for index in fill_indices)
    touches = np.where(
        inventory_changes > 0,
        mid + spread / 2.0,
        mid - spread / 2.0,
    )
    cash_path = np.cumsum(-inventory_changes * touches)
    mid_wealth = cash_path + inventory * mid
    efficient_wealth = cash_path + inventory * efficient
    baseline = mid_wealth[0] + calibration.surrogate_optimum_rate_per_second * (time - time[0])
    indices = _sample_indices(time.size, 5_000, fills)
    rows = [
        {
            "time_seconds": float(time[index]),
            "gap_price": float(gap[index]),
            "threshold_price": float(threshold[index]),
            "spread_price": float(spread[index]),
            "inventory_lots": int(inventory[index]),
            "mid_marked_wealth": float(mid_wealth[index]),
            "efficient_price_marked_wealth": float(efficient_wealth[index]),
            "surrogate_slope_baseline": float(baseline[index]),
            "is_fill": index in fills,
        }
        for index in indices
    ]
    write_csv(
        run_directory / "figures" / "figure5-data.csv",
        list(rows[0].keys()),
        rows,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(time[indices], gap[indices], linewidth=0.8, label=r"gap $G$")
    axes[0].plot(time[indices], threshold[indices], linestyle="--", linewidth=0.8, label=r"$+\theta_D(S)$")
    axes[0].plot(time[indices], -threshold[indices], linestyle="--", linewidth=0.8, label=r"$-\theta_D(S)$")
    if fill_indices.size:
        axes[0].scatter(time[fill_indices], gap[fill_indices], s=10, color="black", label="fills")
    axes[0].set(ylabel="gap", title="Figure 5 parity-dependent band illustration")
    axes[0].legend(frameon=False, ncol=4)
    axes[1].step(time[indices], inventory[indices], where="post", linewidth=0.9)
    axes[1].set(ylabel="inventory", yticks=(-1, 0, 1))
    axes[2].plot(time[indices], mid_wealth[indices], label="mid-marked", linewidth=0.9)
    axes[2].plot(time[indices], efficient_wealth[indices], label="efficient-price-marked", linewidth=0.9)
    axes[2].plot(time[indices], baseline[indices], linestyle=":", label="surrogate slope", linewidth=1.0)
    axes[2].set(xlabel="time (s)", ylabel="wealth")
    axes[2].legend(frameon=False)
    figure.savefig(
        run_directory / "figures" / "figure5.png",
        dpi=180,
        metadata={"Software": "ot-micromr 0.1.0"},
    )
    plt.close(figure)


def render_paper_illustrations(
    values: Mapping[str, Any],
    calibration: CalibrationRow,
    strategy_trace: Figure4MarketTrace,
    run_directory: Path,
) -> None:
    _render_figure2(values, calibration, run_directory)
    _render_figure5(values, calibration, strategy_trace, run_directory)
