from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

from ot_micromr.artifacts import write_csv
from ot_micromr.config import RunSpec
from ot_micromr.simulator import ReplicationResult, settings_from_spec, simulate_replication
from ot_micromr.statistical_gates import (
    one_sample_equivalence,
    one_sample_superiority,
    paired_equivalence,
)


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


def _write_common_artifacts(
    run_directory: Path,
    seed_rows: list[dict[str, Any]],
    resolution_rows: list[dict[str, Any]],
    figure_rows: list[dict[str, Any]],
) -> None:
    seed_fields = sorted({key for row in seed_rows for key in row})
    resolution_fields = sorted({key for row in resolution_rows for key in row})
    figure_fields = ["panel", "epsilon", "series", "x", "value", "standard_error", "lower", "upper", "theory"]
    write_csv(run_directory / "metrics" / "seed_metrics.csv", seed_fields, seed_rows)
    write_csv(run_directory / "tables" / "resolution_summary.csv", resolution_fields, resolution_rows)
    write_csv(run_directory / "figures" / "simulation-data.csv", figure_fields, figure_rows)


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
    figure.suptitle("SIM-MOMENTS-002 controlled numerical simulation")
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
        title="SIM-UNBALANCED-002 parity drift split",
    )
    axis.invert_xaxis()
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180, metadata={"Software": "ot-micromr 0.1.0"})
    plt.close(figure)


def _mutable_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_tree(item) for item in value]
    if isinstance(value, list):
        return [_mutable_tree(item) for item in value]
    return value


def _simulate_task(task: tuple[Mapping[str, Any], float, int]) -> ReplicationResult:
    values, epsilon, seed = task
    with threadpool_limits(limits=1):
        return simulate_replication(values, epsilon, seed)


def _ordered_simulations(
    values: Mapping[str, Any],
    coordinates: Sequence[tuple[float, int]],
    workers: int,
) -> list[ReplicationResult]:
    tasks = [(values, epsilon, seed) for epsilon, seed in coordinates]
    if workers == 1:
        return [_simulate_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_simulate_task, tasks, chunksize=1))


def _run_replications(
    spec: RunSpec,
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[int, ...]],
    dict[str, Any],
]:
    values = _mutable_tree(spec.values)
    epsilons = tuple(float(value) for value in values["numerics"]["refinement_epsilons"])
    seeds = tuple(int(value) for value in values["seed_policy"]["seeds"])
    seed_rows: list[dict[str, Any]] = []
    stream_spawn_keys: dict[str, tuple[int, ...]] = {}
    primary = float(values["numerics"]["primary_resolution_epsilon"])
    workers = int(values["simulation"].get("cpu_workers", 1))
    if not 1 <= workers <= min(20, os.cpu_count() or 1):
        raise ValueError("simulation.cpu_workers must lie between 1 and min(20, os.cpu_count())")
    coordinates = [(epsilon, seed) for epsilon in epsilons for seed in seeds]
    results = _ordered_simulations(values, coordinates, workers)
    primary_rows: dict[int, dict[str, Any]] = {}
    primary_digests: dict[int, str] = {}
    for result in results:
        row = dict(result.seed_metrics)
        seed_rows.append(row)
        stream_spawn_keys = dict(result.stream_spawn_keys)
        if result.epsilon == primary:
            primary_rows[result.seed] = row
            primary_digests[result.seed] = result.replay_digest

    replay_seeds = tuple(
        int(seed) for seed in values["simulation"].get("deterministic_replay_seeds", seeds)
    )
    if not replay_seeds or any(seed not in primary_rows for seed in replay_seeds):
        raise ValueError("deterministic replay seeds must be a non-empty subset of experiment seeds")
    replays = _ordered_simulations(values, [(primary, seed) for seed in replay_seeds], workers)
    for replay in replays:
        row = primary_rows[replay.seed]
        row["deterministic_replay_mismatch_count"] = int(
            replay.replay_digest != primary_digests[replay.seed]
        )
        row["deterministic_replay_checked"] = True

    execution = {
        "cpu_workers": workers,
        "logical_cpu_count": os.cpu_count(),
        "native_threads_per_worker": 1,
        "replication_task_count": len(coordinates),
        "deterministic_replay_count": len(replay_seeds),
    }
    return seed_rows, stream_spawn_keys, execution


def _prefix_result(prefix: str, result: Any) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in asdict(result).items()}


def _balanced_summary(
    spec: RunSpec,
    seed_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, bool]]:
    values = spec.values
    settings = settings_from_spec(values)
    acceptance_spec = values["acceptance"]
    epsilons = tuple(float(value) for value in values["numerics"]["refinement_epsilons"])
    primary = float(values["numerics"]["primary_resolution_epsilon"])
    target_slope = math.expm1(
        -float(values["model"]["alpha_per_second"])
        * settings.observation_interval_seconds
    ) / settings.observation_interval_seconds
    resolution_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    by_epsilon: dict[float, dict[str, Any]] = {}
    rows_by_epsilon: dict[float, list[dict[str, Any]]] = {}

    for epsilon in epsilons:
        rows = sorted(
            (row for row in seed_rows if float(row["epsilon"]) == epsilon),
            key=lambda row: int(row["seed"]),
        )
        rows_by_epsilon[epsilon] = rows
        flow_values = [float(row["integrated_hazard_flow_signed_relative_residual"]) for row in rows]
        open_ratios = [float(row["finite_h_drift_slope_open_per_second"]) / target_slope for row in rows]
        jump_open_ratios = [
            -float(row["realised_jump_drift_slope_open_per_second"])
            / float(values["model"]["alpha_per_second"])
            for row in rows
        ]
        flow_result = one_sample_equivalence(
            flow_values,
            target=0.0,
            margin=float(acceptance_spec["flow_equivalence_margin"]),
        )
        flow_mean, flow_se, flow_lower, flow_upper = _student_interval(flow_values)
        open_mean, open_se, open_lower, open_upper = _student_interval(open_ratios)
        jump_open_mean, jump_open_se, jump_open_lower, jump_open_upper = _student_interval(
            jump_open_ratios
        )
        summary: dict[str, Any] = {
            "epsilon": epsilon,
            "replication_count": len(rows),
            "integrated_hazard_flow_residual_mean": flow_mean,
            "integrated_hazard_flow_residual_se": flow_se,
            "integrated_hazard_flow_residual_t95_lower": flow_lower,
            "integrated_hazard_flow_residual_t95_upper": flow_upper,
            "open_drift_target_ratio_mean": open_mean,
            "open_drift_target_ratio_se": open_se,
            "open_drift_target_ratio_t95_lower": open_lower,
            "open_drift_target_ratio_t95_upper": open_upper,
            "jump_open_drift_target_ratio_mean": jump_open_mean,
            "jump_open_drift_target_ratio_se": jump_open_se,
            "jump_open_drift_target_ratio_t95_lower": jump_open_lower,
            "jump_open_drift_target_ratio_t95_upper": jump_open_upper,
            "finite_h_drift_target_per_second": target_slope,
            "transition_count_imbalance_abs_max": max(
                abs(int(row["transition_count_imbalance"])) for row in rows
            ),
            "open_occupancy_mean": float(np.mean([row["open_occupancy"] for row in rows])),
            "stationary_s_g_mean": float(np.mean([row["stationary_s_g"] for row in rows])),
            "step_count_total": sum(int(row["step_count"]) for row in rows),
            "book_event_count_total": sum(int(row["book_event_count"]) for row in rows),
        }
        summary.update(_prefix_result("flow_equivalence", flow_result))
        for channel in (
            "slide_up",
            "slide_down",
            "open_up",
            "open_down",
            "close_up",
            "close_down",
        ):
            mean, se, lower, upper = _student_interval(
                [float(row[f"compensator_z_{channel}"]) for row in rows]
            )
            summary[f"compensator_z_{channel}_mean"] = mean
            summary[f"compensator_z_{channel}_se"] = se
            summary[f"compensator_z_{channel}_t95_lower"] = lower
            summary[f"compensator_z_{channel}_t95_upper"] = upper
        by_epsilon[epsilon] = summary
        resolution_rows.append(summary)

        for parity in ("tight", "open"):
            observations = [float(row[f"finite_h_drift_slope_{parity}_per_second"]) for row in rows]
            mean, se, lower, upper = _student_interval(observations)
            figure_rows.append(
                {
                    "panel": "drift_by_resolution",
                    "epsilon": epsilon,
                    "series": parity,
                    "x": epsilon,
                    "value": mean,
                    "standard_error": se,
                    "lower": lower,
                    "upper": upper,
                    "theory": target_slope,
                }
            )
        if epsilon == primary:
            for lag in settings.acf_lags_seconds:
                observations = [float(row[f"acf_lag_{lag:g}_seconds"]) for row in rows]
                mean, se, lower, upper = _student_interval(observations)
                figure_rows.append(
                    {
                        "panel": "acf_primary",
                        "epsilon": epsilon,
                        "series": "acf",
                        "x": lag * float(values["model"]["alpha_per_second"]),
                        "value": mean,
                        "standard_error": se,
                        "lower": lower,
                        "upper": upper,
                        "theory": math.exp(-float(values["model"]["alpha_per_second"]) * lag),
                    }
                )

    primary_rows = rows_by_epsilon[primary]
    fine_rows = rows_by_epsilon[0.005]
    primary_by_seed = {int(row["seed"]): row for row in primary_rows}
    fine_by_seed = {int(row["seed"]): row for row in fine_rows}
    ordered_seeds = sorted(primary_by_seed)
    if ordered_seeds != sorted(fine_by_seed):
        raise ValueError("P3V refinement requires identical seed labels at both resolutions")
    flow_refinement = paired_equivalence(
        [primary_by_seed[seed]["integrated_hazard_flow_signed_relative_residual"] for seed in ordered_seeds],
        [fine_by_seed[seed]["integrated_hazard_flow_signed_relative_residual"] for seed in ordered_seeds],
        margin=float(acceptance_spec["refinement_flow_equivalence_margin"]),
    )
    replay_mismatches = sum(
        int(row["deterministic_replay_mismatch_count"]) for row in seed_rows
    )
    generator_max = max(
        max(float(row["generator_drift_abs_residual_tight"]) for row in seed_rows),
        max(float(row["generator_drift_abs_residual_open"]) for row in seed_rows),
    )
    checked_replays = sum(bool(row["deterministic_replay_checked"]) for row in seed_rows)
    acceptance = {
        "deterministic_replay_mismatch_count": replay_mismatches
        <= int(acceptance_spec["deterministic_replay_mismatch_count_max"]),
        "generator_drift_abs_residual": generator_max
        <= float(acceptance_spec["generator_drift_abs_residual_max"]),
        "transition_count_conservation": max(
            abs(int(row["transition_count_imbalance"])) for row in seed_rows
        )
        <= 1,
        "all_replications_each_resolution": all(
            len(rows_by_epsilon[epsilon]) == 20 for epsilon in epsilons
        ),
        "deterministic_replays_complete": checked_replays == 3,
    }
    metrics = {
        "primary_resolution_epsilon": primary,
        "primary": by_epsilon[primary],
        "resolution_summaries": {f"{epsilon:g}": by_epsilon[epsilon] for epsilon in epsilons},
        "scientific_components": {
            "flow_equivalence": asdict(
                one_sample_equivalence(
                    [row["integrated_hazard_flow_signed_relative_residual"] for row in primary_rows],
                    target=0.0,
                    margin=float(acceptance_spec["flow_equivalence_margin"]),
                )
            ),
        },
        "refinement_components": {
            "flow_equivalence": asdict(flow_refinement),
        },
        "open_drift_checks": {
            "finite_h_target_ratio_mean": by_epsilon[primary]["open_drift_target_ratio_mean"],
            "finite_h_target_ratio_t95_lower": by_epsilon[primary]["open_drift_target_ratio_t95_lower"],
            "finite_h_target_ratio_t95_upper": by_epsilon[primary]["open_drift_target_ratio_t95_upper"],
            "jump_target_ratio_mean": by_epsilon[primary]["jump_open_drift_target_ratio_mean"],
            "jump_target_ratio_t95_lower": by_epsilon[primary]["jump_open_drift_target_ratio_t95_lower"],
            "jump_target_ratio_t95_upper": by_epsilon[primary]["jump_open_drift_target_ratio_t95_upper"],
            "role": "reported_model_checks_exact_generator_is_the_gate",
        },
        "generator_drift_abs_residual_max": generator_max,
        "deterministic_replay_mismatch_count": replay_mismatches,
        "deterministic_replay_checked_count": checked_replays,
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
    rows_by_epsilon: dict[float, list[dict[str, Any]]] = {}
    by_epsilon: dict[float, dict[str, Any]] = {}
    superiority_results: dict[float, Any] = {}
    for epsilon in epsilons:
        rows = sorted(
            (row for row in seed_rows if float(row["epsilon"]) == epsilon),
            key=lambda row: int(row["seed"]),
        )
        rows_by_epsilon[epsilon] = rows
        contrasts = [
            float(row["realised_jump_parity_drift_slope_contrast_per_second"])
            for row in rows
        ]
        result = one_sample_superiority(
            contrasts,
            minimum_effect=float(acceptance_spec["contrast_minimum_effect_per_second"]),
        )
        superiority_results[epsilon] = result
        summary: dict[str, Any] = {
            "epsilon": epsilon,
            "replication_count": len(rows),
            "open_occupancy_mean": float(np.mean([row["open_occupancy"] for row in rows])),
            "step_count_total": sum(int(row["step_count"]) for row in rows),
            "book_event_count_total": sum(int(row["book_event_count"]) for row in rows),
        }
        summary.update(_prefix_result("contrast_superiority", result))
        for parity, theory in (("tight", -1.0), ("open", -1.25)):
            observations = [float(row[f"finite_h_drift_slope_{parity}_per_second"]) for row in rows]
            mean, se, lower, upper = _student_interval(observations)
            summary[f"finite_h_drift_slope_{parity}_mean_per_second"] = mean
            summary[f"finite_h_drift_slope_{parity}_se"] = se
            figure_rows.append(
                {
                    "panel": "unbalanced_drift",
                    "epsilon": epsilon,
                    "series": parity,
                    "x": epsilon,
                    "value": mean,
                    "standard_error": se,
                    "lower": lower,
                    "upper": upper,
                    "theory": theory,
                }
            )
        by_epsilon[epsilon] = summary
        resolution_rows.append(summary)

    primary_by_seed = {int(row["seed"]): row for row in rows_by_epsilon[primary]}
    fine_by_seed = {int(row["seed"]): row for row in rows_by_epsilon[0.005]}
    ordered_seeds = sorted(primary_by_seed)
    refinement = paired_equivalence(
        [
            primary_by_seed[seed]["realised_jump_parity_drift_slope_contrast_per_second"]
            for seed in ordered_seeds
        ],
        [
            fine_by_seed[seed]["realised_jump_parity_drift_slope_contrast_per_second"]
            for seed in ordered_seeds
        ],
        margin=float(acceptance_spec["refinement_contrast_equivalence_margin_per_second"]),
    )
    replay_mismatches = sum(
        int(row["deterministic_replay_mismatch_count"]) for row in seed_rows
    )
    generator_max = max(
        max(float(row["generator_drift_abs_residual_tight"]) for row in seed_rows),
        max(float(row["generator_drift_abs_residual_open"]) for row in seed_rows),
    )
    checked_replays = sum(bool(row["deterministic_replay_checked"]) for row in seed_rows)
    acceptance = {
        "deterministic_replay_mismatch_count": replay_mismatches
        <= int(acceptance_spec["deterministic_replay_mismatch_count_max"]),
        "generator_drift_abs_residual": generator_max
        <= float(acceptance_spec["generator_drift_abs_residual_max"]),
        "transition_count_conservation": max(
            abs(int(row["transition_count_imbalance"])) for row in seed_rows
        )
        <= 1,
        "all_replications_each_resolution": all(
            len(rows_by_epsilon[epsilon]) == 20 for epsilon in epsilons
        ),
        "deterministic_replays_complete": checked_replays == 3,
    }
    metrics = {
        "primary_resolution_epsilon": primary,
        "primary": by_epsilon[primary],
        "resolution_summaries": {f"{epsilon:g}": by_epsilon[epsilon] for epsilon in epsilons},
        "scientific_components": {
            "unbalanced_contrast_superiority": asdict(superiority_results[primary]),
        },
        "refinement_components": {"contrast_equivalence": asdict(refinement)},
        "generator_drift_abs_residual_max": generator_max,
        "deterministic_replay_mismatch_count": replay_mismatches,
        "deterministic_replay_checked_count": checked_replays,
    }
    return resolution_rows, figure_rows, metrics, acceptance


def evaluate_simulation(spec: RunSpec, run_directory: Path) -> SimulationEvaluation:
    seed_rows, stream_spawn_keys, execution = _run_replications(spec)
    settings = settings_from_spec(spec.values)
    if spec.experiment_id == "SIM-MOMENTS-002":
        resolution_rows, figure_rows, metrics, acceptance = _balanced_summary(spec, seed_rows)
        _render_balanced_figure(run_directory / "figures" / "sim-moments.png", figure_rows)
    elif spec.experiment_id == "SIM-UNBALANCED-002":
        resolution_rows, figure_rows, metrics, acceptance = _unbalanced_summary(spec, seed_rows)
        _render_unbalanced_figure(run_directory / "figures" / "sim-unbalanced.png", figure_rows)
    else:
        raise ValueError(f"unsupported simulation experiment: {spec.experiment_id}")
    _write_common_artifacts(
        run_directory,
        seed_rows,
        resolution_rows,
        figure_rows,
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
            "compute_execution": execution,
        },
        log_lines=(
            f"Completed {len(seed_rows)} resolution-seed replications and {execution['deterministic_replay_count']} primary deterministic replays",
            f"CPU workers={execution['cpu_workers']}; native threads per worker=1",
            f"Primary epsilon={metrics['primary_resolution_epsilon']}",
            f"acceptance_passed={all(acceptance.values())}",
        ),
    )
