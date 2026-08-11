from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from ot_micromr.artifacts import write_csv
from ot_micromr.config import RunSpec
from ot_micromr.jump_model import BookEventRecord, BookParameters
from ot_micromr.simulator import ReplicationResult, settings_from_spec, simulate_replication


@dataclass(frozen=True, slots=True)
class SimulationEvaluation:
    metrics: Mapping[str, Any]
    acceptance: Mapping[str, bool]
    derived_parameters: Mapping[str, Any]
    log_lines: Sequence[str]

    @property
    def passed(self) -> bool:
        return all(self.acceptance.values())


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("mean/SE requires at least two finite values")
    return float(np.mean(array)), float(np.std(array, ddof=1) / math.sqrt(array.size))


def _student_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float, float]:
    mean, se = _mean_se(values)
    critical = float(student_t.ppf(0.5 + confidence / 2.0, len(values) - 1))
    return mean, se, mean - critical * se, mean + critical * se


def _one_sided_lower(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float]:
    mean, se = _mean_se(values)
    critical = float(student_t.ppf(confidence, len(values) - 1))
    return mean, se, mean - critical * se


def _simultaneous_intervals(
    matrix: np.ndarray, bootstrap_seed: int, replications: int
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("simultaneous intervals require a finite seed-by-coordinate matrix")
    seed_count = matrix.shape[0]
    means = np.mean(matrix, axis=0)
    standard_errors = np.std(matrix, axis=0, ddof=1) / math.sqrt(seed_count)
    if np.any(standard_errors <= 0.0):
        raise ValueError("simultaneous interval coordinate has zero standard error")
    centered = matrix - means
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(bootstrap_seed)))
    indices = rng.integers(0, seed_count, size=(replications, seed_count))
    samples = centered[indices]
    bootstrap_means = np.mean(samples, axis=1)
    bootstrap_se = np.std(samples, axis=1, ddof=1) / math.sqrt(seed_count)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistics = np.abs(bootstrap_means / bootstrap_se)
    statistics[~np.isfinite(statistics)] = 0.0
    maximum_statistics = np.max(statistics, axis=1)
    critical = float(np.quantile(maximum_statistics, 0.95))
    lower = means - critical * standard_errors
    upper = means + critical * standard_errors
    return critical, means, standard_errors, lower, upper


def _append_binned_drift(
    accumulator: dict[tuple[float, bool, int], dict[str, float]],
    result: ReplicationResult,
    observation_h: float,
    edges: np.ndarray,
) -> None:
    gaps = result.gaps[:-1]
    changes = np.diff(result.gaps) / observation_h
    normalized = gaps / float(result.seed_metrics["stationary_s_g"])
    bin_indices = np.searchsorted(edges, normalized, side="right") - 1
    valid = (bin_indices >= 0) & (bin_indices < edges.size - 1)
    for parity in (True, False):
        parity_mask = result.tight[:-1] == parity
        for bin_index in range(edges.size - 1):
            mask = valid & parity_mask & (bin_indices == bin_index)
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            x = gaps[mask]
            y = changes[mask]
            key = (result.epsilon, parity, bin_index)
            record = accumulator.setdefault(
                key,
                {"count": 0.0, "sum_x": 0.0, "sum_y": 0.0, "sum_xy": 0.0, "sum_x2": 0.0},
            )
            record["count"] += count
            record["sum_x"] += float(np.sum(x))
            record["sum_y"] += float(np.sum(y))
            record["sum_xy"] += float(np.dot(x, y))
            record["sum_x2"] += float(np.dot(x, x))


def _binned_rows(
    accumulator: Mapping[tuple[float, bool, int], Mapping[str, float]],
    edges: np.ndarray,
    minimum_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    epsilons = sorted({key[0] for key in accumulator})
    for epsilon in epsilons:
        for parity in (True, False):
            for bin_index in range(edges.size - 1):
                record = accumulator.get((epsilon, parity, bin_index))
                count = int(record["count"]) if record else 0
                sufficient = count >= minimum_count
                rows.append(
                    {
                        "epsilon": epsilon,
                        "parity": "tight" if parity else "open",
                        "bin_left_s_g": float(edges[bin_index]),
                        "bin_right_s_g": float(edges[bin_index + 1]),
                        "observation_count": count,
                        "status": "sufficient" if sufficient else "insufficient",
                        "mean_gap_price": float(record["sum_x"] / count) if count else None,
                        "mean_gap_change_per_second": float(record["sum_y"] / count)
                        if count
                        else None,
                        "ols_slope_per_second": float(record["sum_xy"] / record["sum_x2"])
                        if record and record["sum_x2"] > 0.0
                        else None,
                    }
                )
    return rows


def _event_rows(events: Sequence[BookEventRecord]) -> list[dict[str, Any]]:
    return [
        {
            "epsilon": event.epsilon,
            "seed": event.seed,
            "event_index": event.event_index,
            "time_seconds": event.time_seconds,
            "channel": event.channel,
            "left_gap_price": event.left_gap_price,
            "pre_event_gap_price": event.pre_event_gap_price,
            "post_event_gap_price": event.post_event_gap_price,
            "pre_mid_half_ticks": event.pre_mid_half_ticks,
            "post_mid_half_ticks": event.post_mid_half_ticks,
            "efficient_price": event.efficient_price,
            "delta_mid_price": event.delta_mid_price,
            "left_channel_intensity_per_second": event.left_channel_intensity_per_second,
            "measured": event.measured,
        }
        for event in events
    ]


def _write_common_artifacts(
    run_directory: Path,
    seed_rows: list[dict[str, Any]],
    resolution_rows: list[dict[str, Any]],
    figure_rows: list[dict[str, Any]],
    events: list[BookEventRecord],
    binned_rows: list[dict[str, Any]] | None = None,
) -> None:
    seed_fields = sorted({key for row in seed_rows for key in row})
    resolution_fields = sorted({key for row in resolution_rows for key in row})
    figure_fields = ["panel", "epsilon", "series", "x", "value", "standard_error", "lower", "upper", "theory"]
    write_csv(run_directory / "metrics" / "seed_metrics.csv", seed_fields, seed_rows)
    write_csv(run_directory / "tables" / "resolution_summary.csv", resolution_fields, resolution_rows)
    write_csv(run_directory / "figures" / "simulation-data.csv", figure_fields, figure_rows)
    event_rows = _event_rows(events)
    event_fields = list(event_rows[0].keys()) if event_rows else [
        "epsilon",
        "seed",
        "event_index",
        "time_seconds",
        "channel",
        "left_gap_price",
        "pre_event_gap_price",
        "post_event_gap_price",
        "pre_mid_half_ticks",
        "post_mid_half_ticks",
        "efficient_price",
        "delta_mid_price",
        "left_channel_intensity_per_second",
        "measured",
    ]
    write_csv(run_directory / "records" / "book_events.csv", event_fields, event_rows)
    if binned_rows is not None:
        write_csv(
            run_directory / "tables" / "binned_drift.csv",
            list(binned_rows[0].keys()),
            binned_rows,
        )


def _render_balanced_figure(path: Path, figure_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    acf_rows = [row for row in figure_rows if row["panel"] == "acf_primary"]
    x = np.asarray([row["x"] for row in acf_rows], dtype=np.float64)
    y = np.asarray([row["value"] for row in acf_rows], dtype=np.float64)
    lower = np.asarray([row["lower"] for row in acf_rows], dtype=np.float64)
    upper = np.asarray([row["upper"] for row in acf_rows], dtype=np.float64)
    theory = np.asarray([row["theory"] for row in acf_rows], dtype=np.float64)
    axes[0].errorbar(x, y, yerr=np.vstack((y - lower, upper - y)), marker="o", capsize=3, label="simulation")
    axes[0].plot(x, theory, linestyle="--", label=r"$e^{-\alpha h}$")
    axes[0].set(xlabel="Lag (reversion times)", ylabel="ACF", title="Primary resolution ACF")
    axes[0].legend(frameon=False)
    slope_rows = [row for row in figure_rows if row["panel"] == "drift_by_resolution"]
    for series in ("tight", "open"):
        selected = [row for row in slope_rows if row["series"] == series]
        x_values = np.asarray([row["epsilon"] for row in selected], dtype=np.float64)
        y_values = np.asarray([row["value"] for row in selected], dtype=np.float64)
        low = np.asarray([row["lower"] for row in selected], dtype=np.float64)
        high = np.asarray([row["upper"] for row in selected], dtype=np.float64)
        axes[1].errorbar(
            x_values,
            y_values,
            yerr=np.vstack((y_values - low, high - y_values)),
            marker="o",
            capsize=3,
            label=series,
        )
    axes[1].axhline(float(slope_rows[0]["theory"]), linestyle="--", color="black", label="finite-h target")
    axes[1].set(xlabel=r"Resolution $\epsilon$", ylabel="Slope per second", title="Conditional drift")
    axes[1].invert_xaxis()
    axes[1].legend(frameon=False)
    figure.suptitle("SIM-MOMENTS-001 controlled numerical simulation")
    figure.savefig(path, dpi=180, metadata={"Software": "ot-micromr 0.1.0"})
    plt.close(figure)


def _render_unbalanced_figure(path: Path, figure_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for series, theory, color in (("tight", -1.0, "tab:blue"), ("open", -1.25, "tab:orange")):
        selected = [row for row in figure_rows if row["series"] == series]
        x = np.asarray([row["epsilon"] for row in selected], dtype=np.float64)
        y = np.asarray([row["value"] for row in selected], dtype=np.float64)
        lower = np.asarray([row["lower"] for row in selected], dtype=np.float64)
        upper = np.asarray([row["upper"] for row in selected], dtype=np.float64)
        axis.errorbar(x, y, yerr=np.vstack((y - lower, upper - y)), marker="o", capsize=3, color=color, label=series)
        axis.axhline(theory, linestyle="--", color=color, alpha=0.75)
    axis.set(
        xlabel=r"Resolution $\epsilon$",
        ylabel="Finite-h slope per second",
        title="SIM-UNBALANCED-001 parity drift split",
    )
    axis.invert_xaxis()
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180, metadata={"Software": "ot-micromr 0.1.0"})
    plt.close(figure)


def _run_replications(
    spec: RunSpec,
) -> tuple[list[dict[str, Any]], list[BookEventRecord], dict[tuple[float, bool, int], dict[str, float]], dict[str, tuple[int, ...]]]:
    values = spec.values
    settings = settings_from_spec(values)
    epsilons = tuple(float(value) for value in values["numerics"]["refinement_epsilons"])
    seeds = tuple(int(value) for value in values["seed_policy"]["seeds"])
    binned: dict[tuple[float, bool, int], dict[str, float]] = {}
    edges = np.asarray(values["evaluation"].get("drift_gap_bin_edges_s_g", (-10.0, 10.0)), dtype=np.float64)
    seed_rows: list[dict[str, Any]] = []
    all_events: list[BookEventRecord] = []
    stream_spawn_keys: dict[str, tuple[int, ...]] = {}
    primary = float(values["numerics"]["primary_resolution_epsilon"])

    for epsilon in epsilons:
        for seed in seeds:
            result = simulate_replication(values, epsilon, seed, settings=settings)
            row = dict(result.seed_metrics)
            if epsilon == primary:
                replay = simulate_replication(values, epsilon, seed, settings=settings)
                mismatch = int(replay.replay_digest != result.replay_digest)
                row["deterministic_replay_mismatch_count"] = mismatch
                row["deterministic_replay_checked"] = True
            seed_rows.append(row)
            all_events.extend(result.events)
            stream_spawn_keys = dict(result.stream_spawn_keys)
            if spec.experiment_id == "SIM-MOMENTS-001":
                _append_binned_drift(
                    binned,
                    result,
                    settings.observation_interval_seconds,
                    edges,
                )
    return seed_rows, all_events, binned, stream_spawn_keys


def _balanced_summary(
    spec: RunSpec,
    seed_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, bool]]:
    values = spec.values
    evaluation = values["evaluation"]
    acceptance_spec = values["acceptance"]
    parameters = BookParameters.from_model(values["model"])
    settings = settings_from_spec(values)
    epsilons = tuple(float(value) for value in values["numerics"]["refinement_epsilons"])
    primary = float(values["numerics"]["primary_resolution_epsilon"])
    target_slope = math.expm1(-float(values["model"]["alpha_per_second"]) * settings.observation_interval_seconds) / settings.observation_interval_seconds
    lag_keys = [f"acf_lag_{lag:g}_seconds" for lag in settings.acf_lags_seconds]
    resolution_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    by_epsilon: dict[float, dict[str, Any]] = {}

    for epsilon in epsilons:
        rows = [row for row in seed_rows if row["epsilon"] == epsilon]
        slope_matrix = np.asarray(
            [
                [
                    row["finite_h_drift_slope_tight_per_second"],
                    row["finite_h_drift_slope_open_per_second"],
                    *(row[key] for key in lag_keys),
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
        critical, coordinate_means, coordinate_se, coordinate_lower, coordinate_upper = _simultaneous_intervals(
            slope_matrix,
            int(values["seed_policy"]["bootstrap_seed"]),
            int(evaluation["bootstrap_replications"]),
        )
        normalized_mean, normalized_mean_se, normalized_mean_lower, normalized_mean_upper = _student_interval(
            [row["stationary_mean_gap_over_s_g"] for row in rows]
        )
        variance_residual, variance_residual_se, variance_residual_lower, variance_residual_upper = _student_interval(
            [row["stationary_variance_identity_signed_relative_residual"] for row in rows]
        )
        flow_residual, flow_residual_se, flow_residual_lower, flow_residual_upper = _student_interval(
            [row["open_close_flow_signed_relative_residual"] for row in rows]
        )
        open_occupancy, open_occupancy_se, open_occupancy_lower, open_occupancy_upper = _student_interval(
            [row["open_occupancy"] for row in rows]
        )
        generator_tight = max(float(row["generator_drift_abs_residual_tight"]) for row in rows)
        generator_open = max(float(row["generator_drift_abs_residual_open"]) for row in rows)
        summary: dict[str, Any] = {
            "epsilon": epsilon,
            "replication_count": len(rows),
            "stationary_mean_gap_over_s_g_mean": normalized_mean,
            "stationary_mean_gap_over_s_g_se": normalized_mean_se,
            "stationary_mean_gap_over_s_g_t95_lower": normalized_mean_lower,
            "stationary_mean_gap_over_s_g_t95_upper": normalized_mean_upper,
            "stationary_variance_identity_signed_relative_residual_mean": variance_residual,
            "stationary_variance_identity_signed_relative_residual_se": variance_residual_se,
            "stationary_variance_identity_signed_relative_residual_t95_lower": variance_residual_lower,
            "stationary_variance_identity_signed_relative_residual_t95_upper": variance_residual_upper,
            "stationary_variance_identity_relative_error": abs(variance_residual),
            "open_close_flow_signed_relative_residual_mean": flow_residual,
            "open_close_flow_signed_relative_residual_se": flow_residual_se,
            "open_close_flow_signed_relative_residual_t95_lower": flow_residual_lower,
            "open_close_flow_signed_relative_residual_t95_upper": flow_residual_upper,
            "open_close_flow_relative_error": abs(flow_residual),
            "open_occupancy_mean": open_occupancy,
            "open_occupancy_se": open_occupancy_se,
            "open_occupancy_t95_lower": open_occupancy_lower,
            "open_occupancy_t95_upper": open_occupancy_upper,
            "finite_h_drift_target_per_second": target_slope,
            "finite_h_drift_slope_tight_mean_per_second": float(coordinate_means[0]),
            "finite_h_drift_slope_tight_se": float(coordinate_se[0]),
            "finite_h_drift_slope_tight_simultaneous_lower": float(coordinate_lower[0]),
            "finite_h_drift_slope_tight_simultaneous_upper": float(coordinate_upper[0]),
            "finite_h_drift_slope_open_mean_per_second": float(coordinate_means[1]),
            "finite_h_drift_slope_open_se": float(coordinate_se[1]),
            "finite_h_drift_slope_open_simultaneous_lower": float(coordinate_lower[1]),
            "finite_h_drift_slope_open_simultaneous_upper": float(coordinate_upper[1]),
            "simultaneous_max_t_critical": critical,
            "generator_drift_abs_residual_tight_max": generator_tight,
            "generator_drift_abs_residual_open_max": generator_open,
            "maximum_left_event_probability": max(float(row["maximum_left_event_probability"]) for row in rows),
            "step_count_total": sum(int(row["step_count"]) for row in rows),
            "book_event_count_total": sum(int(row["book_event_count"]) for row in rows),
            "invariant_violation_count": sum(int(row["invariant_violation_count"]) for row in rows),
            "deterministic_replay_mismatch_count": sum(
                int(row["deterministic_replay_mismatch_count"]) for row in rows
            ),
        }
        for output_name, seed_key in (
            ("stationary_mean_gap", "stationary_mean_gap"),
            ("stationary_variance_gap", "stationary_variance_gap"),
            ("stationary_s_g", "stationary_s_g"),
            ("jump_variance_rate", "jump_variance_rate"),
            ("effective_sample_size_ou_approximation", "effective_sample_size_ou_approximation"),
        ):
            mean, se, lower, upper = _student_interval([float(row[seed_key]) for row in rows])
            summary[f"{output_name}_mean"] = mean
            summary[f"{output_name}_se"] = se
            summary[f"{output_name}_t95_lower"] = lower
            summary[f"{output_name}_t95_upper"] = upper
        for index, (lag, key) in enumerate(zip(settings.acf_lags_seconds, lag_keys, strict=True), start=2):
            summary[f"acf_lag_{lag:g}_mean"] = float(coordinate_means[index])
            summary[f"acf_lag_{lag:g}_se"] = float(coordinate_se[index])
            summary[f"acf_lag_{lag:g}_simultaneous_lower"] = float(coordinate_lower[index])
            summary[f"acf_lag_{lag:g}_simultaneous_upper"] = float(coordinate_upper[index])
            summary[f"acf_lag_{lag:g}_theory"] = math.exp(
                -float(values["model"]["alpha_per_second"]) * lag
            )
        resolution_rows.append(summary)
        by_epsilon[epsilon] = summary
        t_critical = float(student_t.ppf(0.975, len(rows) - 1))
        for series, index in (("tight", 0), ("open", 1)):
            figure_rows.append(
                {
                    "panel": "drift_by_resolution",
                    "epsilon": epsilon,
                    "series": series,
                    "x": epsilon,
                    "value": float(coordinate_means[index]),
                    "standard_error": float(coordinate_se[index]),
                    "lower": float(coordinate_means[index] - t_critical * coordinate_se[index]),
                    "upper": float(coordinate_means[index] + t_critical * coordinate_se[index]),
                    "theory": target_slope,
                }
            )
        if epsilon == primary:
            for index, lag in enumerate(settings.acf_lags_seconds, start=2):
                figure_rows.append(
                    {
                        "panel": "acf_primary",
                        "epsilon": epsilon,
                        "series": "acf",
                        "x": lag * float(values["model"]["alpha_per_second"]),
                        "value": float(coordinate_means[index]),
                        "standard_error": float(coordinate_se[index]),
                        "lower": float(coordinate_lower[index]),
                        "upper": float(coordinate_upper[index]),
                        "theory": math.exp(-float(values["model"]["alpha_per_second"]) * lag),
                    }
                )

    primary_summary = by_epsilon[primary]
    fine_summary = by_epsilon[0.005]
    refinement_coordinates: list[dict[str, Any]] = []
    refinement_pairs = [
        (
            "stationary_mean_gap_over_s_g",
            "stationary_mean_gap_over_s_g_mean",
            "stationary_mean_gap_over_s_g_se",
        ),
        (
            "stationary_variance_identity_signed_relative_residual",
            "stationary_variance_identity_signed_relative_residual_mean",
            "stationary_variance_identity_signed_relative_residual_se",
        ),
        (
            "open_close_flow_signed_relative_residual",
            "open_close_flow_signed_relative_residual_mean",
            "open_close_flow_signed_relative_residual_se",
        ),
    ]
    for name, mean_key, se_key in refinement_pairs:
        difference = abs(float(primary_summary[mean_key]) - float(fine_summary[mean_key]))
        difference_se = math.hypot(float(primary_summary[se_key]), float(fine_summary[se_key]))
        refinement_coordinates.append(
            {
                "metric": name,
                "absolute_difference": difference,
                "difference_se": difference_se,
                "passed": difference
                <= max(
                    float(acceptance_spec["refinement_each_metric_abs_difference_max"]),
                    difference_se,
                ),
            }
        )
    for parity in ("tight", "open"):
        mean_key = f"finite_h_drift_slope_{parity}_mean_per_second"
        se_key = f"finite_h_drift_slope_{parity}_se"
        primary_ratio = float(primary_summary[mean_key]) / target_slope
        fine_ratio = float(fine_summary[mean_key]) / target_slope
        difference = abs(primary_ratio - fine_ratio)
        difference_se = math.hypot(
            float(primary_summary[se_key]) / abs(target_slope),
            float(fine_summary[se_key]) / abs(target_slope),
        )
        refinement_coordinates.append(
            {
                "metric": f"finite_h_drift_slope_over_target_{parity}",
                "absolute_difference": difference,
                "difference_se": difference_se,
                "passed": difference
                <= max(float(acceptance_spec["refinement_each_metric_abs_difference_max"]), difference_se),
            }
        )
    for lag in settings.acf_lags_seconds:
        mean_key = f"acf_lag_{lag:g}_mean"
        se_key = f"acf_lag_{lag:g}_se"
        difference = abs(float(primary_summary[mean_key]) - float(fine_summary[mean_key]))
        difference_se = math.hypot(float(primary_summary[se_key]), float(fine_summary[se_key]))
        refinement_coordinates.append(
            {
                "metric": f"acf_lag_{lag:g}",
                "absolute_difference": difference,
                "difference_se": difference_se,
                "passed": difference
                <= max(float(acceptance_spec["refinement_each_metric_abs_difference_max"]), difference_se),
            }
        )

    invariant_keys = (
        "parity_violation_count",
        "illegal_transition_count",
        "negative_intensity_count",
        "nonzero_inactive_intensity_count",
        "invariant_violation_count",
        "nonfinite_value_count",
        "multiple_book_event_step_count",
        "deterministic_replay_mismatch_count",
    )
    invariant_totals = {
        key: sum(int(row[key]) for row in seed_rows)
        for key in invariant_keys
    }
    target_inside_slopes = all(
        float(primary_summary[f"finite_h_drift_slope_{parity}_simultaneous_lower"])
        <= target_slope
        <= float(primary_summary[f"finite_h_drift_slope_{parity}_simultaneous_upper"])
        for parity in ("tight", "open")
    )
    target_inside_acf = all(
        float(primary_summary[f"acf_lag_{lag:g}_simultaneous_lower"])
        <= math.exp(-float(values["model"]["alpha_per_second"]) * lag)
        <= float(primary_summary[f"acf_lag_{lag:g}_simultaneous_upper"])
        for lag in settings.acf_lags_seconds
    )
    slope_relative_errors = {
        parity: abs(
            (float(primary_summary[f"finite_h_drift_slope_{parity}_mean_per_second"]) - target_slope)
            / target_slope
        )
        for parity in ("tight", "open")
    }
    acceptance = {
        **{key: value <= int(acceptance_spec[f"{key}_max"]) for key, value in invariant_totals.items()},
        "generator_drift_abs_residual": max(
            float(primary_summary["generator_drift_abs_residual_tight_max"]),
            float(primary_summary["generator_drift_abs_residual_open_max"]),
        )
        <= float(acceptance_spec["generator_drift_abs_residual_max"]),
        "stationary_mean_gap_abs_over_s_g": abs(
            float(primary_summary["stationary_mean_gap_over_s_g_mean"])
        )
        <= float(acceptance_spec["stationary_mean_gap_abs_over_s_g_max"]),
        "stationary_variance_identity_relative_error": float(
            primary_summary["stationary_variance_identity_relative_error"]
        )
        <= float(acceptance_spec["stationary_variance_identity_relative_error_max"]),
        "open_close_flow_relative_error": float(primary_summary["open_close_flow_relative_error"])
        <= float(acceptance_spec["open_close_flow_relative_error_max"]),
        "conditional_drift_slope_relative_error": max(slope_relative_errors.values())
        <= float(acceptance_spec["conditional_drift_slope_relative_error_max"]),
        "theoretical_drift_slopes_inside_simultaneous_interval": target_inside_slopes,
        "theoretical_acf_inside_simultaneous_interval": target_inside_acf,
        "refinement_all_coordinates": all(row["passed"] for row in refinement_coordinates),
        "all_20_replications_each_resolution": all(
            sum(row["epsilon"] == epsilon for row in seed_rows) == 20 for epsilon in epsilons
        ),
    }
    metrics = {
        "primary_resolution_epsilon": primary,
        "primary": primary_summary,
        "resolution_summaries": {f"{epsilon:g}": by_epsilon[epsilon] for epsilon in epsilons},
        "refinement": refinement_coordinates,
        "invariant_totals": invariant_totals,
        "finite_h_drift_slope_relative_errors": slope_relative_errors,
        "target_inside_simultaneous_drift_bands": target_inside_slopes,
        "target_inside_simultaneous_acf_bands": target_inside_acf,
    }
    return resolution_rows, figure_rows, metrics, acceptance


def _unbalanced_summary(
    spec: RunSpec,
    seed_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, bool]]:
    values = spec.values
    acceptance_spec = values["acceptance"]
    epsilons = tuple(float(value) for value in values["numerics"]["refinement_epsilons"])
    primary = float(values["numerics"]["primary_resolution_epsilon"])
    resolution_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    by_epsilon: dict[float, dict[str, Any]] = {}
    for epsilon in epsilons:
        rows = [row for row in seed_rows if row["epsilon"] == epsilon]
        tight_values = [float(row["finite_h_drift_slope_tight_per_second"]) for row in rows]
        open_values = [float(row["finite_h_drift_slope_open_per_second"]) for row in rows]
        contrast_values = [float(row["finite_h_parity_drift_slope_contrast_per_second"]) for row in rows]
        tight_mean, tight_se, tight_lower, tight_upper = _student_interval(tight_values)
        open_mean, open_se, open_lower, open_upper = _student_interval(open_values)
        contrast_mean, contrast_se, contrast_lower = _one_sided_lower(contrast_values)
        summary = {
            "epsilon": epsilon,
            "replication_count": len(rows),
            "finite_h_drift_slope_tight_mean_per_second": tight_mean,
            "finite_h_drift_slope_tight_se": tight_se,
            "finite_h_drift_slope_tight_lower": tight_lower,
            "finite_h_drift_slope_tight_upper": tight_upper,
            "finite_h_drift_slope_open_mean_per_second": open_mean,
            "finite_h_drift_slope_open_se": open_se,
            "finite_h_drift_slope_open_lower": open_lower,
            "finite_h_drift_slope_open_upper": open_upper,
            "finite_h_parity_contrast_mean_per_second": contrast_mean,
            "finite_h_parity_contrast_se": contrast_se,
            "finite_h_parity_contrast_one_sided_95_lower": contrast_lower,
            "generator_drift_slope_tight_mean_per_second": float(
                np.mean([row["generator_drift_slope_tight_per_second"] for row in rows])
            ),
            "generator_drift_slope_open_mean_per_second": float(
                np.mean([row["generator_drift_slope_open_per_second"] for row in rows])
            ),
            "generator_drift_abs_residual_max": max(
                max(float(row["generator_drift_abs_residual_tight"]) for row in rows),
                max(float(row["generator_drift_abs_residual_open"]) for row in rows),
            ),
            "open_occupancy_mean": float(np.mean([row["open_occupancy"] for row in rows])),
            "maximum_left_event_probability": max(float(row["maximum_left_event_probability"]) for row in rows),
            "step_count_total": sum(int(row["step_count"]) for row in rows),
            "book_event_count_total": sum(int(row["book_event_count"]) for row in rows),
            "invariant_violation_count": sum(int(row["invariant_violation_count"]) for row in rows),
            "deterministic_replay_mismatch_count": sum(
                int(row["deterministic_replay_mismatch_count"]) for row in rows
            ),
        }
        by_epsilon[epsilon] = summary
        resolution_rows.append(summary)
        for series, mean, se, lower, upper, theory in (
            ("tight", tight_mean, tight_se, tight_lower, tight_upper, -1.0),
            ("open", open_mean, open_se, open_lower, open_upper, -1.25),
        ):
            figure_rows.append(
                {
                    "panel": "unbalanced_drift",
                    "epsilon": epsilon,
                    "series": series,
                    "x": epsilon,
                    "value": mean,
                    "standard_error": se,
                    "lower": lower,
                    "upper": upper,
                    "theory": theory,
                }
            )

    primary_summary = by_epsilon[primary]
    fine_summary = by_epsilon[0.005]
    contrast_difference = abs(
        float(primary_summary["finite_h_parity_contrast_mean_per_second"])
        - float(fine_summary["finite_h_parity_contrast_mean_per_second"])
    )
    contrast_difference_se = math.hypot(
        float(primary_summary["finite_h_parity_contrast_se"]),
        float(fine_summary["finite_h_parity_contrast_se"]),
    )
    invariant_total = sum(int(row["invariant_violation_count"]) for row in seed_rows)
    replay_total = sum(int(row["deterministic_replay_mismatch_count"]) for row in seed_rows)
    acceptance = {
        "invariant_violation_count": invariant_total
        <= int(acceptance_spec["invariant_violation_count_max"]),
        "deterministic_replay_mismatch_count": replay_total == 0,
        "generator_tight_drift_slope": abs(
            float(primary_summary["generator_drift_slope_tight_mean_per_second"]) + 1.0
        )
        <= float(acceptance_spec["generator_tight_drift_slope_abs_error_max"]),
        "generator_open_drift_slope": abs(
            float(primary_summary["generator_drift_slope_open_mean_per_second"]) + 1.25
        )
        <= float(acceptance_spec["generator_open_drift_slope_abs_error_max"]),
        "generator_drift_abs_residual": float(primary_summary["generator_drift_abs_residual_max"])
        <= float(acceptance_spec["generator_drift_abs_residual_max"]),
        "finite_h_parity_contrast_one_sided_lower": float(
            primary_summary["finite_h_parity_contrast_one_sided_95_lower"]
        )
        >= float(
            acceptance_spec[
                "finite_h_parity_contrast_one_sided_95_percent_lower_bound_min_per_second"
            ]
        ),
        "refinement_primary_metric": contrast_difference
        <= max(
            float(acceptance_spec["refinement_primary_metric_abs_difference_max_per_second"]),
            contrast_difference_se,
        ),
        "all_20_replications_each_resolution": all(
            sum(row["epsilon"] == epsilon for row in seed_rows) == 20 for epsilon in epsilons
        ),
    }
    metrics = {
        "primary_resolution_epsilon": primary,
        "primary": primary_summary,
        "resolution_summaries": {f"{epsilon:g}": by_epsilon[epsilon] for epsilon in epsilons},
        "refinement": {
            "metric": "finite_h_parity_drift_slope_contrast_per_second",
            "absolute_difference": contrast_difference,
            "difference_se": contrast_difference_se,
        },
        "invariant_violation_count": invariant_total,
        "deterministic_replay_mismatch_count": replay_total,
    }
    return resolution_rows, figure_rows, metrics, acceptance


def evaluate_simulation(spec: RunSpec, run_directory: Path) -> SimulationEvaluation:
    seed_rows, events, binned, stream_spawn_keys = _run_replications(spec)
    settings = settings_from_spec(spec.values)
    if spec.experiment_id == "SIM-MOMENTS-001":
        resolution_rows, figure_rows, metrics, acceptance = _balanced_summary(spec, seed_rows)
        edges = np.asarray(spec.values["evaluation"]["drift_gap_bin_edges_s_g"], dtype=np.float64)
        binned_rows = _binned_rows(
            binned,
            edges,
            int(spec.values["evaluation"]["minimum_pooled_observations_per_drift_bin_and_parity"]),
        )
        _render_balanced_figure(run_directory / "figures" / "sim-moments.png", figure_rows)
    else:
        resolution_rows, figure_rows, metrics, acceptance = _unbalanced_summary(spec, seed_rows)
        binned_rows = None
        _render_unbalanced_figure(run_directory / "figures" / "sim-unbalanced.png", figure_rows)
    _write_common_artifacts(
        run_directory,
        seed_rows,
        resolution_rows,
        figure_rows,
        events,
        binned_rows=binned_rows,
    )
    primary = metrics["primary"]
    return SimulationEvaluation(
        metrics=metrics,
        acceptance=acceptance,
        derived_parameters={
            "simulation_algorithm": spec.values["numerics"]["simulation_algorithm"],
            "invariant_suite_id": spec.values["numerics"]["invariant_suite_id"],
            "burn_seconds": settings.burn_seconds,
            "horizon_seconds": settings.horizon_seconds,
            "observation_interval_seconds": settings.observation_interval_seconds,
            "resolution_epsilons": list(spec.values["numerics"]["refinement_epsilons"]),
            "stream_mapping_version": spec.values["seed_policy"]["stream_mapping_version"],
            "stream_spawn_keys": {key: list(value) for key, value in stream_spawn_keys.items()},
            "strategy_monitoring": "not_applicable_disabled",
        },
        log_lines=(
            f"Completed {len(seed_rows)} resolution-seed replications and 20 primary deterministic replays",
            f"Recorded {len(events)} book events",
            f"Primary epsilon={metrics['primary_resolution_epsilon']}",
            f"Primary invariant violations={primary.get('invariant_violation_count', metrics.get('invariant_violation_count', 0))}",
            f"acceptance_passed={all(acceptance.values())}",
        ),
    )
