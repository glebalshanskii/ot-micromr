from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from ot_micromr.analytics import normalized_surrogate_rate
from ot_micromr.artifacts import write_csv
from ot_micromr.config import RunSpec
from ot_micromr.figure4 import (
    CalibrationRow,
    Figure4Replication,
    calibrate_rows,
    replay_figure4_coordinate,
    simulate_figure4,
)


def _mean_se_interval(values: Sequence[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(array.size))
    critical = float(student_t.ppf(0.975, array.size - 1))
    return mean, se, mean - critical * se, mean + critical * se


def _selected_rows(
    calibrations: Sequence[CalibrationRow], targets: Sequence[float]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for target in targets:
        row = min(calibrations, key=lambda item: (abs(item.gamma_ratio - target), item.row_index))
        selected.append(
            {
                "target_gamma_ratio": float(target),
                "row_index": row.row_index,
                "realized_gamma_ratio": row.gamma_ratio,
                "absolute_gamma_error": abs(row.gamma_ratio - target),
            }
        )
    return selected


def _curve_rows(
    results: Sequence[Figure4Replication], calibrations: Sequence[CalibrationRow]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_row = {item.row_index: item for item in calibrations}
    keys = sorted(
        {
            (result.epsilon, result.row_index, int(policy["policy_index"]))
            for result in results
            for policy in result.policy_rows
        }
    )
    for epsilon, row_index, policy_index in keys:
        policies = [
            policy
            for result in results
            if result.epsilon == epsilon and result.row_index == row_index
            for policy in result.policy_rows
            if int(policy["policy_index"]) == policy_index
        ]
        calibration = by_row[row_index]
        mean, se, lower, upper = _mean_se_interval(
            [float(policy["renewal_rate_over_surrogate_optimum"]) for policy in policies]
        )
        multiplier = float(policies[0]["threshold_multiplier_theta_d"])
        u_ratio = multiplier * calibration.u_d_ratio
        surrogate = float(normalized_surrogate_rate(u_ratio, calibration.gamma_ratio)) / (
            calibration.surrogate_optimum_rate_per_second
            / (calibration.alpha_per_second * calibration.s_g_price)
        )
        overshoots = [
            float(policy["mean_fill_overshoot_price"])
            for policy in policies
            if policy["mean_fill_overshoot_price"] is not None
        ]
        open_shares = [
            float(policy["open_fill_share"])
            for policy in policies
            if policy["open_fill_share"] is not None
        ]
        rows.append(
            {
                "epsilon": epsilon,
                "row_index": row_index,
                "alpha_per_second": calibration.alpha_per_second,
                "gamma_ratio": calibration.gamma_ratio,
                "policy_index": policy_index,
                "policy_label": policies[0]["policy_label"],
                "threshold_multiplier_theta_d": multiplier,
                "threshold_price": float(policies[0]["threshold_price"]),
                "mean_rate_over_surrogate_optimum": mean,
                "rate_standard_error": se,
                "rate_t95_lower": lower,
                "rate_t95_upper": upper,
                "surrogate_rate_over_surrogate_optimum": surrogate,
                "mean_overshoot_price": float(np.mean(overshoots)) if overshoots else None,
                "mean_open_fill_share": float(np.mean(open_shares)) if open_shares else None,
                "minimum_complete_interval_count": min(
                    int(policy["complete_interval_count"]) for policy in policies
                ),
            }
        )
    return rows


def _rate_tensor(
    results: Sequence[Figure4Replication], epsilon: float
) -> tuple[np.ndarray, list[int], list[int], np.ndarray]:
    selected = [result for result in results if result.epsilon == epsilon]
    seeds = sorted({result.seed for result in selected})
    rows = sorted({result.row_index for result in selected})
    policy_count = len(selected[0].policy_rows)
    tensor = np.empty((len(seeds), len(rows), policy_count), dtype=np.float64)
    multipliers = np.empty(policy_count, dtype=np.float64)
    for seed_index, seed in enumerate(seeds):
        for row_offset, row_index in enumerate(rows):
            result = next(
                item for item in selected if item.seed == seed and item.row_index == row_index
            )
            tensor[seed_index, row_offset] = [
                float(policy["renewal_rate_over_surrogate_optimum"])
                for policy in result.policy_rows
            ]
            multipliers[:] = [
                float(policy["threshold_multiplier_theta_d"])
                for policy in result.policy_rows
            ]
    return tensor, seeds, rows, multipliers


def _functional_reduction_numpy(
    rates: np.ndarray,
    bootstrap_indices: np.ndarray,
    grid_multipliers: np.ndarray,
    theta_d_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(rates[bootstrap_indices], axis=1)
    grid_means = means[:, :, : grid_multipliers.size]
    peak_indices = np.argmax(grid_means, axis=2)
    peak_rates = np.take_along_axis(grid_means, peak_indices[:, :, None], axis=2)[:, :, 0]
    inward_shifts = 1.0 - grid_multipliers[peak_indices]
    loss_d = 1.0 - grid_means[:, :, theta_d_index] / peak_rates
    loss_star = 1.0 - means[:, :, -1] / peak_rates
    return inward_shifts, loss_d, loss_star


def _bootstrap_functionals(
    values: Mapping[str, Any], rates: np.ndarray, multipliers: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    grid_multipliers = multipliers[:-1]
    theta_d_locations = np.flatnonzero(np.isclose(grid_multipliers, 1.0, atol=1e-12))
    if theta_d_locations.size != 1:
        raise RuntimeError("threshold grid must contain theta_D exactly once")
    theta_d_index = int(theta_d_locations[0])
    replications = int(values["evaluation"]["bootstrap_replications"])
    seed_count = rates.shape[0]
    rng = np.random.Generator(
        np.random.PCG64DXSM(np.random.SeedSequence(int(values["seed_policy"]["bootstrap_seed"])))
    )
    indices = rng.integers(0, seed_count, size=(replications, seed_count), dtype=np.int64)
    started = time.perf_counter()
    cpu = _functional_reduction_numpy(rates, indices, grid_multipliers, theta_d_index)
    cpu_seconds = time.perf_counter() - started
    backend: dict[str, Any] = {
        "selected": "numpy_float64",
        "cpu_float64_seconds": cpu_seconds,
        "gpu_available": False,
    }
    output = cpu
    if bool(values["numerics"]["gpu_reduction_enabled"]):
        try:
            import torch
        except ImportError:
            backend["fallback_reason"] = "torch_not_installed"
        else:
            backend["gpu_available"] = torch.cuda.is_available()
            backend["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                device_rates = torch.as_tensor(rates, dtype=torch.float32, device="cuda")
                device_indices = torch.as_tensor(indices, dtype=torch.int64, device="cuda")
                device_grid = torch.as_tensor(grid_multipliers, dtype=torch.float32, device="cuda")

                def reduction(local_rates: Any, local_indices: Any, local_grid: Any) -> tuple[Any, ...]:
                    means = local_rates[local_indices].mean(dim=1)
                    grid_means = means[:, :, : local_grid.numel()]
                    peak_indices = torch.argmax(grid_means, dim=2)
                    peak_rates = torch.gather(grid_means, 2, peak_indices.unsqueeze(-1)).squeeze(-1)
                    shifts = 1.0 - local_grid[peak_indices]
                    losses_d = 1.0 - grid_means[:, :, theta_d_index] / peak_rates
                    losses_star = 1.0 - means[:, :, -1] / peak_rates
                    return shifts, losses_d, losses_star

                compiled = (
                    torch.compile(reduction, fullgraph=True)
                    if bool(values["numerics"]["gpu_compile_enabled"])
                    else reduction
                )
                torch.cuda.synchronize()
                cold_started = time.perf_counter()
                cold = compiled(device_rates, device_indices, device_grid)
                torch.cuda.synchronize()
                cold_seconds = time.perf_counter() - cold_started
                steady_started = time.perf_counter()
                candidate = compiled(device_rates, device_indices, device_grid)
                torch.cuda.synchronize()
                steady_seconds = time.perf_counter() - steady_started
                gpu = tuple(item.detach().cpu().numpy().astype(np.float64) for item in candidate)
                shift_exact = bool(np.array_equal(cpu[0], gpu[0]))
                maximum_error = max(
                    float(np.max(np.abs(reference - candidate_value)))
                    for reference, candidate_value in zip(cpu, gpu, strict=True)
                )
                backend.update(
                    {
                        "gpu_device": torch.cuda.get_device_name(0),
                        "gpu_float32_cold_seconds": cold_seconds,
                        "gpu_float32_steady_seconds": steady_seconds,
                        "gpu_speedup_vs_cpu": cpu_seconds / steady_seconds,
                        "peak_shift_exact": shift_exact,
                        "functional_max_abs_error": maximum_error,
                    }
                )
                if steady_seconds < cpu_seconds and shift_exact and maximum_error <= 1e-5:
                    output = gpu
                    backend["selected"] = "torch_compile_cuda_float32"
                else:
                    backend["fallback_reason"] = "no_validated_end_to_end_advantage"
    return {
        "inward_shift": output[0],
        "loss_at_theta_d": output[1],
        "loss_at_theta_star": output[2],
    }, backend


def _functionals(
    values: Mapping[str, Any], results: Sequence[Figure4Replication]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    backend_records: dict[str, Any] = {}
    for epsilon in sorted({result.epsilon for result in results}, reverse=True):
        tensor, _, row_indices, multipliers = _rate_tensor(results, epsilon)
        bootstrap, backend = _bootstrap_functionals(values, tensor, multipliers)
        backend_records[f"epsilon_{epsilon:g}"] = backend
        point_mean = np.mean(tensor, axis=0)
        grid = multipliers[:-1]
        peak_indices = np.argmax(point_mean[:, :-1], axis=1)
        theta_d_index = int(np.flatnonzero(np.isclose(grid, 1.0, atol=1e-12))[0])
        for offset, row_index in enumerate(row_indices):
            peak_index = int(peak_indices[offset])
            peak_rate = float(point_mean[offset, peak_index])
            fitted_peak: float | None = None
            if 2 <= peak_index < grid.size - 2:
                fit_slice = slice(peak_index - 2, peak_index + 3)
                coefficients = np.polyfit(grid[fit_slice], point_mean[offset, fit_slice], 2)
                if coefficients[0] < 0.0:
                    vertex = float(-coefficients[1] / (2.0 * coefficients[0]))
                    if grid[peak_index - 2] <= vertex <= grid[peak_index + 2]:
                        fitted_peak = vertex
            shifts = bootstrap["inward_shift"][:, offset]
            losses_d = bootstrap["loss_at_theta_d"][:, offset]
            losses_star = bootstrap["loss_at_theta_star"][:, offset]
            rows.append(
                {
                    "epsilon": epsilon,
                    "row_index": row_index,
                    "discrete_peak_multiplier_theta_d": float(grid[peak_index]),
                    "discrete_inward_shift_fraction": 1.0 - float(grid[peak_index]),
                    "fitted_peak_multiplier_theta_d": fitted_peak,
                    "peak_rate_over_surrogate_optimum": peak_rate,
                    "rate_loss_at_theta_d_fraction": 1.0
                    - float(point_mean[offset, theta_d_index]) / peak_rate,
                    "rate_loss_at_theta_star_fraction": 1.0
                    - float(point_mean[offset, -1]) / peak_rate,
                    "bootstrap_inward_shift_95_lower": float(np.quantile(shifts, 0.025)),
                    "bootstrap_inward_shift_95_upper": float(np.quantile(shifts, 0.975)),
                    "bootstrap_inward_shift_standard_deviation": float(np.std(shifts, ddof=1)),
                    "bootstrap_loss_theta_d_95_lower": float(np.quantile(losses_d, 0.025)),
                    "bootstrap_loss_theta_d_95_upper": float(np.quantile(losses_d, 0.975)),
                    "bootstrap_loss_theta_star_95_lower": float(
                        np.quantile(losses_star, 0.025)
                    ),
                    "bootstrap_loss_theta_star_95_upper": float(
                        np.quantile(losses_star, 0.975)
                    ),
                }
            )
    return rows, backend_records


def _render_figure4(
    path: Path,
    calibrations: Sequence[CalibrationRow],
    curve_rows: Sequence[Mapping[str, Any]],
    functional_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    primary_epsilon: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    distinct = []
    for selection in selected_rows:
        if selection["row_index"] not in distinct:
            distinct.append(selection["row_index"])
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(distinct)))
    by_row = {row.row_index: row for row in calibrations}
    for color, row_index in zip(colors, distinct, strict=True):
        selected = [
            row
            for row in curve_rows
            if row["epsilon"] == primary_epsilon
            and row["row_index"] == row_index
            and str(row["policy_label"]).startswith("grid:")
        ]
        x = np.asarray([row["threshold_multiplier_theta_d"] for row in selected])
        y = np.asarray([row["mean_rate_over_surrogate_optimum"] for row in selected])
        se = np.asarray([row["rate_standard_error"] for row in selected])
        theory = np.asarray([row["surrogate_rate_over_surrogate_optimum"] for row in selected])
        gamma = by_row[int(row_index)].gamma_ratio
        axes[0].plot(x, theory, color=color, linewidth=1.5)
        axes[0].errorbar(
            x,
            y,
            yerr=se,
            fmt="o",
            markersize=3.5,
            capsize=2,
            color=color,
            label=fr"$\gamma={gamma:.3f}$",
        )
    axes[0].axvline(1.0, linestyle=":", color="black", linewidth=1)
    axes[0].set(
        xlabel=r"band half-width $\theta/\theta_D$",
        ylabel="realised rate / surrogate optimum",
        title="Independent exact-model band sweep",
    )
    axes[0].legend(frameon=False)

    primary_functionals = sorted(
        [row for row in functional_rows if row["epsilon"] == primary_epsilon],
        key=lambda item: by_row[int(item["row_index"])].gamma_ratio,
    )
    x = np.asarray([by_row[int(row["row_index"])].gamma_ratio for row in primary_functionals])
    y = 100.0 * np.asarray([row["discrete_inward_shift_fraction"] for row in primary_functionals])
    lower = 100.0 * np.asarray(
        [row["bootstrap_inward_shift_95_lower"] for row in primary_functionals]
    )
    upper = 100.0 * np.asarray(
        [row["bootstrap_inward_shift_95_upper"] for row in primary_functionals]
    )
    axes[1].errorbar(x, y, yerr=np.vstack((y - lower, upper - y)), marker="o", capsize=3)
    axes[1].axhspan(15.0, 20.0, color="0.9", label="paper descriptive range")
    axes[1].set(
        xlabel=r"spread-to-dispersion ratio $\gamma$",
        ylabel=r"inward shift $1-\theta_{peak}/\theta_D$ (%)",
        title="Discrete exact-model optimum",
    )
    axes[1].legend(frameon=False)
    figure.savefig(path, dpi=180, metadata={"Software": "ot-micromr 0.1.0"})
    plt.close(figure)


def evaluate_figure4(spec: RunSpec, run_directory: Path) -> Any:
    from ot_micromr.experiments import EvaluationResult

    values = spec.to_dict()
    calibrations = calibrate_rows(values)
    calibration_rows = [asdict(row) for row in calibrations]
    write_csv(
        run_directory / "tables" / "calibration.csv",
        list(calibration_rows[0].keys()),
        calibration_rows,
    )
    epsilons = tuple(float(value) for value in values["numerics"]["refinement_epsilons"])
    seeds = tuple(int(value) for value in values["seed_policy"]["strategy_seeds"])
    coordinates = [
        (row.row_index, epsilon, seed)
        for epsilon in epsilons
        for row in calibrations
        for seed in seeds
    ]
    results = simulate_figure4(
        values, calibrations, coordinates, int(values["numerics"]["cpu_workers"])
    )
    policy_rows = [dict(policy) for result in results for policy in result.policy_rows]
    diagnostic_rows = [
        {
            "row_index": result.row_index,
            "alpha_per_second": result.alpha_per_second,
            "epsilon": result.epsilon,
            "seed": result.seed,
            "replay_digest": result.replay_digest,
            **result.diagnostics,
        }
        for result in results
    ]
    curve_rows = _curve_rows(results, calibrations)
    functional_rows, backend = _functionals(values, results)
    selections = _selected_rows(
        calibrations, values["evaluation"]["target_realized_gamma_ratios"]
    )
    replay_mismatches = 0
    for replay_seed in values["seed_policy"]["deterministic_replay_seeds"]:
        original = next(
            item
            for item in results
            if item.row_index == 0 and item.epsilon == epsilons[0] and item.seed == replay_seed
        )
        replay = replay_figure4_coordinate(values, calibrations[0], epsilons[0], replay_seed)
        replay_mismatches += int(original.replay_digest != replay.replay_digest)

    write_csv(
        run_directory / "metrics" / "seed_threshold_metrics.csv",
        sorted({key for row in policy_rows for key in row}),
        policy_rows,
    )
    write_csv(
        run_directory / "metrics" / "path_diagnostics.csv",
        sorted({key for row in diagnostic_rows for key in row}),
        diagnostic_rows,
    )
    write_csv(
        run_directory / "tables" / "curve_summary.csv",
        list(curve_rows[0].keys()),
        curve_rows,
    )
    write_csv(
        run_directory / "tables" / "functionals.csv",
        list(functional_rows[0].keys()),
        functional_rows,
    )
    write_csv(
        run_directory / "figures" / "figure4-data.csv",
        list(curve_rows[0].keys()),
        curve_rows,
    )
    write_csv(
        run_directory / "records" / "fills.csv",
        ["row_index", "epsilon", "seed", "policy_index", "time_seconds", "side", "gap", "spread"],
        [],
    )
    primary_epsilon = float(values["numerics"]["primary_resolution_epsilon"])
    _render_figure4(
        run_directory / "figures" / "figure4.png",
        calibrations,
        curve_rows,
        functional_rows,
        selections,
        primary_epsilon,
    )

    acceptance_spec = values["acceptance"]
    expected_result_count = len(epsilons) * len(calibrations) * len(seeds)
    expected_policy_rows = expected_result_count * len(results[0].policy_rows)
    minimum_intervals = int(
        acceptance_spec["minimum_complete_interfill_intervals_per_seed_and_policy"]
    )
    max_omitted = max(
        float(row["omitted_bridge_probability_sum"])
        + float(row["full_band_recrossing_probability_bound"])
        for row in diagnostic_rows
    )
    max_identity = max(float(row["wealth_marking_identity_abs_residual"]) for row in policy_rows)
    acceptance = {
        "all_replications": len(results) == expected_result_count,
        "all_response_rows": len({item.row_index for item in results}) == len(calibrations),
        "all_thresholds": len(policy_rows) == expected_policy_rows,
        "nonflat_before_measurement": all(
            int(row["invariant_violation_count"]) == 0 for row in diagnostic_rows
        ),
        "minimum_complete_intervals": all(
            int(row["complete_interval_count"]) >= minimum_intervals for row in policy_rows
        ),
        "invariant_violation_count": all(
            int(row["invariant_violation_count"])
            <= int(acceptance_spec["invariant_violation_count_max"])
            for row in diagnostic_rows
        ),
        "nonfinite_value_count": all(
            int(row["nonfinite_value_count"])
            <= int(acceptance_spec["nonfinite_value_count_max"])
            for row in diagnostic_rows
        ),
        "omitted_probability_budget": max_omitted
        <= float(acceptance_spec["omitted_probability_sum_max"]),
        "dawson_root_residual": max(row.root_abs_residual for row in calibrations)
        <= float(acceptance_spec["dawson_root_abs_residual_max"]),
        "wealth_marking_identity": max_identity <= 1e-10,
        "deterministic_replay": replay_mismatches == 0,
    }
    primary_functionals = [
        row for row in functional_rows if row["epsilon"] == primary_epsilon
    ]
    metrics = {
        "result_label": acceptance_spec["result_label"],
        "claim_eligible": bool(values["claim_eligible"]),
        "calibration": calibration_rows,
        "selected_gamma_rows": selections,
        "primary_functionals": primary_functionals,
        "backend": backend,
        "maximum_omitted_probability_bound": max_omitted,
        "maximum_wealth_marking_identity_abs_residual": max_identity,
        "minimum_complete_interval_count": min(
            int(row["complete_interval_count"]) for row in policy_rows
        ),
        "fraction_cells_with_at_least_100_intervals": float(
            np.mean([int(row["complete_interval_count"]) >= 100 for row in policy_rows])
        ),
        "deterministic_replay_mismatch_count": replay_mismatches,
        "result_count": len(results),
        "seed_threshold_row_count": len(policy_rows),
    }
    return EvaluationResult(
        metrics=metrics,
        acceptance=acceptance,
        derived_parameters={
            "calibration_rows": calibration_rows,
            "selected_gamma_rows": selections,
            "cpu_workers": int(values["numerics"]["cpu_workers"]),
            "resolution_epsilons": list(epsilons),
            "strategy_seed_count": len(seeds),
            "policy_count": len(results[0].policy_rows),
            "crossing_monitor": values["numerics"]["bridge_crossing_probability"],
        },
        log_lines=(
            f"calibrated_rows={len(calibrations)}",
            f"strategy_results={len(results)}",
            f"seed_threshold_rows={len(policy_rows)}",
            f"minimum_complete_intervals={metrics['minimum_complete_interval_count']}",
            f"maximum_omitted_probability_bound={max_omitted:.3e}",
            f"deterministic_replay_mismatches={replay_mismatches}",
            f"acceptance_passed={all(acceptance.values())}",
        ),
    )
