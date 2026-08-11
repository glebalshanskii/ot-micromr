from __future__ import annotations

import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ot_micromr.analytics import (
    dimensional_surrogate_rate,
    direct_optimum_ratio,
    inclusive_grid,
    kramers_threshold_ratio,
    normalized_surrogate_rate,
    rate_loss_fraction,
    solve_dawson_optimum,
)
from ot_micromr.artifacts import (
    artifact_inventory,
    atomic_write_bytes,
    atomic_write_json,
    copy_source,
    environment_provenance,
    git_provenance,
    sha256_bytes,
    sha256_file,
    utc_now,
    utc_text,
    write_csv,
)
from ot_micromr.config import RunSpec
from ot_micromr.errors import ExperimentError
from ot_micromr.figure4_experiments import evaluate_figure4
from ot_micromr.simulation_experiments import evaluate_simulation


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: Mapping[str, Any]
    acceptance: Mapping[str, bool]
    derived_parameters: Mapping[str, Any]
    log_lines: Sequence[str]

    @property
    def passed(self) -> bool:
        return all(self.acceptance.values())


@dataclass(frozen=True, slots=True)
class RunResult:
    experiment_id: str
    run_id: str
    run_directory: Path
    status: str
    acceptance_passed: bool
    metrics: Mapping[str, Any]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_smoke(spec: RunSpec) -> EvaluationResult:
    values = spec.values
    model = values["model"]
    numerics = values["numerics"]
    acceptance_spec = values["acceptance"]
    gamma = float(model["gamma_ratio"])
    solution = solve_dawson_optimum(gamma, numerics)
    optimizer_u = direct_optimum_ratio(
        gamma,
        numerics,
        lower_margin_ratio=float(numerics["optimizer_lower_margin_ratio"]),
        upper_u_ratio=float(numerics["optimizer_upper_u_ratio"]),
    )
    optimizer_error = abs(solution.u_d_ratio - optimizer_u)
    rate_identity_error = abs(solution.normalized_rate - solution.identity_normalized_rate)
    dimensional_rate = float(
        dimensional_surrogate_rate(
            solution.u_d_ratio,
            gamma,
            float(model["alpha_per_second"]),
            float(model["s_g_price"]),
        )
    )
    metrics = {
        "gamma_ratio": gamma,
        "u_d_ratio": solution.u_d_ratio,
        "direct_optimizer_u_ratio": optimizer_u,
        "dawson_root_abs_residual": solution.root_abs_residual,
        "root_vs_direct_optimizer_abs_error": optimizer_error,
        "normalized_optimum_rate": solution.normalized_rate,
        "optimum_rate_identity_normalized": solution.identity_normalized_rate,
        "optimum_rate_identity_abs_error": rate_identity_error,
        "dimensional_optimum_rate_price_per_second": dimensional_rate,
    }
    acceptance = {
        "dawson_root_abs_residual": solution.root_abs_residual
        <= float(acceptance_spec["dawson_root_abs_residual_max"]),
        "root_vs_direct_optimizer_abs_error": optimizer_error
        <= float(acceptance_spec["root_vs_direct_optimizer_abs_error_max"]),
        "optimum_rate_identity_abs_error": rate_identity_error
        <= float(acceptance_spec["optimum_rate_identity_abs_error_max"]),
        "u_d_strictly_greater_than_gamma": solution.u_d_ratio > gamma,
        "all_values_finite": all(math.isfinite(float(value)) for value in metrics.values()),
    }
    return EvaluationResult(
        metrics=metrics,
        acceptance=acceptance,
        derived_parameters={
            "alpha_per_second": float(model["alpha_per_second"]),
            "s_g_price": float(model["s_g_price"]),
            "phi_price": float(model["phi_price"]),
            "theta_d_price": solution.u_d_ratio * float(model["s_g_price"]),
            "normalization": "rate = alpha_per_second * s_g_price * normalized_rate",
        },
        log_lines=(
            f"Solved Dawson FOC at gamma={gamma:.17g}",
            f"u_D={solution.u_d_ratio:.17g}; residual={solution.root_abs_residual:.3e}",
            f"direct optimizer u={optimizer_u:.17g}; abs error={optimizer_error:.3e}",
            f"acceptance_passed={all(acceptance.values())}",
        ),
    )


def _threshold_rows(spec: RunSpec) -> tuple[list[dict[str, float]], float]:
    evaluation = spec.values["evaluation"]
    numerics = spec.values["numerics"]
    grid = evaluation["gamma_grid"]
    gamma_values = inclusive_grid(
        float(grid["start_ratio"]), float(grid["stop_ratio"]), float(grid["step_ratio"])
    )
    rows: list[dict[str, float]] = []
    max_optimizer_error = 0.0
    for gamma in gamma_values:
        gamma_value = float(gamma)
        solution = solve_dawson_optimum(gamma_value, numerics)
        u_star = kramers_threshold_ratio(gamma_value)
        loss = rate_loss_fraction(u_star, solution.u_d_ratio, gamma_value)
        direct_u = direct_optimum_ratio(gamma_value, numerics)
        optimizer_error = abs(solution.u_d_ratio - direct_u)
        max_optimizer_error = max(max_optimizer_error, optimizer_error)
        rows.append(
            {
                "gamma_ratio": gamma_value,
                "u_d_ratio": solution.u_d_ratio,
                "u_star_ratio": u_star,
                "threshold_relative_difference": (u_star - solution.u_d_ratio)
                / solution.u_d_ratio,
                "normalized_rate_at_u_d": solution.normalized_rate,
                "normalized_rate_at_u_star": float(normalized_surrogate_rate(u_star, gamma_value)),
                "rate_loss_at_u_star_fraction": loss,
                "dawson_root_abs_residual": solution.root_abs_residual,
                "direct_optimizer_u_ratio": direct_u,
                "root_vs_direct_optimizer_abs_error": optimizer_error,
            }
        )
    return rows, max_optimizer_error


def _rate_curve_rows(spec: RunSpec) -> list[dict[str, float]]:
    grid = spec.values["evaluation"]["rate_curve_grid"]
    rows: list[dict[str, float]] = []
    for gamma in grid["gamma_ratios"]:
        gamma_value = float(gamma)
        u_values = inclusive_grid(
            gamma_value, float(grid["u_stop_ratio"]), float(grid["u_step_ratio"])
        )
        rates = normalized_surrogate_rate(u_values, gamma_value)
        for u_value, rate in zip(u_values, rates, strict=True):
            rows.append(
                {
                    "gamma_ratio": gamma_value,
                    "u_ratio": float(u_value),
                    "normalized_rate": float(rate),
                }
            )
    return rows


def _render_figure3(
    path: Path,
    threshold_rows: Sequence[Mapping[str, float]],
    rate_rows: Sequence[Mapping[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    gammas = np.asarray([row["gamma_ratio"] for row in threshold_rows], dtype=np.float64)
    u_d = np.asarray([row["u_d_ratio"] for row in threshold_rows], dtype=np.float64)
    u_star = np.asarray([row["u_star_ratio"] for row in threshold_rows], dtype=np.float64)
    axes[0].plot(gammas, u_d, label=r"Dawson optimum $u_D$", linewidth=2.0)
    axes[0].plot(gammas, u_star, label=r"Kramers $u^*$", linestyle="--", linewidth=1.8)
    axes[0].set(xlabel=r"Cost ratio $\gamma$", ylabel=r"Threshold ratio $u$")
    axes[0].legend(frameon=False)

    rate_gammas = sorted({float(row["gamma_ratio"]) for row in rate_rows})
    for gamma in rate_gammas:
        selected = [row for row in rate_rows if float(row["gamma_ratio"]) == gamma]
        axes[1].plot(
            [row["u_ratio"] for row in selected],
            [row["normalized_rate"] for row in selected],
            label=rf"$\gamma={gamma:g}$",
            linewidth=1.8,
        )
    axes[1].set(
        xlabel=r"Threshold ratio $u$",
        ylabel=r"Normalized rate $\widetilde R/(\alpha s_G)$",
    )
    axes[1].legend(frameon=False)
    figure.suptitle("Independent reconstruction of paper Figure 3")
    figure.savefig(path, metadata={"Software": "ot-micromr 0.1.0"})
    plt.close(figure)


def evaluate_fig3(spec: RunSpec, run_directory: Path) -> EvaluationResult:
    evaluation = spec.values["evaluation"]
    acceptance_spec = spec.values["acceptance"]
    threshold_rows, max_optimizer_error = _threshold_rows(spec)
    rate_rows = _rate_curve_rows(spec)

    rows_by_gamma = {round(row["gamma_ratio"], 12): row for row in threshold_rows}
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_u_errors: list[float] = []
    checkpoint_loss_errors: list[float] = []
    for reference in evaluation["reference_checkpoints"]:
        gamma = float(reference["gamma_ratio"])
        actual = rows_by_gamma[round(gamma, 12)]
        u_d_error = abs(actual["u_d_ratio"] - float(reference["u_d_ratio"]))
        u_star_error = abs(actual["u_star_ratio"] - float(reference["u_star_ratio"]))
        loss_error = abs(
            actual["rate_loss_at_u_star_fraction"]
            - float(reference["rate_loss_at_u_star_fraction"])
        )
        checkpoint_u_errors.extend((u_d_error, u_star_error))
        checkpoint_loss_errors.append(loss_error)
        checkpoint_rows.append(
            {
                "gamma_ratio": gamma,
                "u_d_ratio": actual["u_d_ratio"],
                "reference_u_d_ratio": float(reference["u_d_ratio"]),
                "u_d_abs_error": u_d_error,
                "u_star_ratio": actual["u_star_ratio"],
                "reference_u_star_ratio": float(reference["u_star_ratio"]),
                "u_star_abs_error": u_star_error,
                "rate_loss_at_u_star_fraction": actual["rate_loss_at_u_star_fraction"],
                "reference_rate_loss_fraction": float(reference["rate_loss_at_u_star_fraction"]),
                "rate_loss_abs_error": loss_error,
            }
        )

    reference_summary = evaluation["reference_grid_summary"]
    domain_rows = [
        row
        for row in threshold_rows
        if float(reference_summary["domain_gamma_min_ratio"])
        <= row["gamma_ratio"]
        <= float(reference_summary["domain_gamma_max_ratio"])
    ]
    max_loss_row = max(domain_rows, key=lambda row: row["rate_loss_at_u_star_fraction"])
    max_loss_error = abs(
        max_loss_row["rate_loss_at_u_star_fraction"]
        - float(reference_summary["maximum_rate_loss_at_u_star_fraction"])
    )
    argmax_error = abs(
        max_loss_row["gamma_ratio"] - float(reference_summary["argmax_gamma_ratio"])
    )
    myopic_rates = [
        abs(row["normalized_rate"])
        for row in rate_rows
        if row["u_ratio"] == row["gamma_ratio"]
    ]
    max_myopic_rate = max(myopic_rates)
    all_numeric_values = [
        float(value)
        for row in (*threshold_rows, *rate_rows)
        for value in row.values()
        if isinstance(value, (int, float, np.generic))
    ]
    all_finite = bool(np.all(np.isfinite(np.asarray(all_numeric_values, dtype=np.float64))))
    max_root_residual = max(row["dawson_root_abs_residual"] for row in threshold_rows)
    max_checkpoint_u_error = max(checkpoint_u_errors)
    max_checkpoint_loss_error = max(checkpoint_loss_errors)

    rate_curve_peak_errors: list[float] = []
    far_endpoint_below_peak: list[bool] = []
    for gamma in evaluation["rate_curve_grid"]["gamma_ratios"]:
        gamma_value = float(gamma)
        selected = [row for row in rate_rows if row["gamma_ratio"] == gamma_value]
        grid_peak = max(selected, key=lambda row: row["normalized_rate"])
        exact_peak = solve_dawson_optimum(gamma_value, spec.values["numerics"])
        rate_curve_peak_errors.append(abs(grid_peak["u_ratio"] - exact_peak.u_d_ratio))
        far_endpoint_below_peak.append(selected[-1]["normalized_rate"] < grid_peak["normalized_rate"])

    metrics = {
        "threshold_grid_row_count": len(threshold_rows),
        "rate_curve_row_count": len(rate_rows),
        "dawson_root_abs_residual_max": max_root_residual,
        "checkpoint_u_abs_error_max": max_checkpoint_u_error,
        "checkpoint_rate_loss_abs_error_max": max_checkpoint_loss_error,
        "maximum_rate_loss_fraction_for_gamma_ge_0_4": max_loss_row[
            "rate_loss_at_u_star_fraction"
        ],
        "maximum_rate_loss_argmax_gamma_ratio": max_loss_row["gamma_ratio"],
        "reference_grid_max_rate_loss_abs_error": max_loss_error,
        "reference_grid_argmax_gamma_abs_error": argmax_error,
        "myopic_rate_abs_max": max_myopic_rate,
        "root_vs_direct_optimizer_abs_error_max_diagnostic": max_optimizer_error,
        "rate_curve_grid_peak_u_abs_error_max_diagnostic": max(rate_curve_peak_errors),
        "rate_curve_far_endpoint_below_peak_all": all(far_endpoint_below_peak),
        "all_grid_rows_finite": all_finite,
    }
    acceptance = {
        "dawson_root_abs_residual": max_root_residual
        <= float(acceptance_spec["dawson_root_abs_residual_max"]),
        "checkpoint_u_abs_error": max_checkpoint_u_error
        <= float(acceptance_spec["checkpoint_u_abs_error_max"]),
        "checkpoint_rate_loss_abs_error": max_checkpoint_loss_error
        <= float(acceptance_spec["checkpoint_rate_loss_abs_error_max"]),
        "reference_grid_max_rate_loss_abs_error": max_loss_error
        <= float(acceptance_spec["reference_grid_max_rate_loss_abs_error_max"]),
        "reference_grid_argmax_gamma_abs_error": argmax_error
        <= float(acceptance_spec["reference_grid_argmax_gamma_abs_error_max"]),
        "myopic_rate_abs_max": max_myopic_rate
        <= float(acceptance_spec["myopic_rate_abs_max"]),
        "all_grid_rows_finite": all_finite,
    }

    threshold_fields = list(threshold_rows[0].keys())
    rate_fields = list(rate_rows[0].keys())
    checkpoint_fields = list(checkpoint_rows[0].keys())
    write_csv(run_directory / "metrics" / "raw_threshold_grid.csv", threshold_fields, threshold_rows)
    write_csv(run_directory / "tables" / "figure3_checkpoints.csv", checkpoint_fields, checkpoint_rows)
    write_csv(run_directory / "figures" / "figure3-data-thresholds.csv", threshold_fields, threshold_rows)
    write_csv(run_directory / "figures" / "figure3-data-rate-curves.csv", rate_fields, rate_rows)
    _render_figure3(run_directory / "figures" / "figure3.png", threshold_rows, rate_rows)

    return EvaluationResult(
        metrics=metrics,
        acceptance=acceptance,
        derived_parameters={
            "gamma_grid_start_ratio": float(evaluation["gamma_grid"]["start_ratio"]),
            "gamma_grid_stop_ratio": float(evaluation["gamma_grid"]["stop_ratio"]),
            "gamma_grid_step_ratio": float(evaluation["gamma_grid"]["step_ratio"]),
            "threshold_grid_rows": len(threshold_rows),
            "rate_curve_rows": len(rate_rows),
            "rate_normalization": "rate = alpha_per_second * s_g_price * normalized_rate",
        },
        log_lines=(
            f"Evaluated {len(threshold_rows)} threshold rows and {len(rate_rows)} rate-curve rows",
            f"maximum loss={max_loss_row['rate_loss_at_u_star_fraction']:.12g} at gamma={max_loss_row['gamma_ratio']:.12g}",
            f"maximum Dawson residual={max_root_residual:.3e}",
            f"maximum direct-optimizer discrepancy={max_optimizer_error:.3e} (diagnostic)",
            f"acceptance_passed={all(acceptance.values())}",
        ),
    )


def _required_artifacts_present(spec: RunSpec, run_directory: Path) -> dict[str, bool]:
    paths_by_class: dict[str, list[Path]] = {
        "source_config": [run_directory / "source_config.toml"],
        "resolved_runspec": [run_directory / "resolved_runspec.json"],
        "manifest": [run_directory / "manifest.json"],
        "log": [run_directory / "logs" / "run.log"],
        "metrics_summary": [run_directory / "metrics" / "summary.json"],
        "metrics_raw": [run_directory / "metrics" / "raw_threshold_grid.csv"],
        "table": [run_directory / "tables" / "figure3_checkpoints.csv"],
        "figure_data": [
            run_directory / "figures" / "figure3-data-thresholds.csv",
            run_directory / "figures" / "figure3-data-rate-curves.csv",
        ],
        "figure": [run_directory / "figures" / "figure3.png"],
    }
    if spec.experiment_id.startswith(("SIM-MOMENTS-", "SIM-UNBALANCED-")):
        figure_name = (
            "sim-moments.png" if spec.experiment_id.startswith("SIM-MOMENTS-") else "sim-unbalanced.png"
        )
        paths_by_class.update(
            {
                "metrics_raw": [run_directory / "metrics" / "seed_metrics.csv"],
                "table": [run_directory / "tables" / "resolution_summary.csv"],
                "figure_data": [run_directory / "figures" / "simulation-data.csv"],
                "figure": [run_directory / "figures" / figure_name],
                "event_log": [run_directory / "records" / "book_events.csv"],
            }
        )
    if spec.experiment_id.startswith("SIM-FIG4-"):
        paths_by_class.update(
            {
                "metrics_raw": [
                    run_directory / "metrics" / "seed_threshold_metrics.csv",
                    run_directory / "metrics" / "path_diagnostics.csv",
                ],
                "table": [
                    run_directory / "tables" / "curve_summary.csv",
                    run_directory / "tables" / "functionals.csv",
                ],
                "figure_data": [run_directory / "figures" / "figure4-data.csv"],
                "figure": [run_directory / "figures" / "figure4.png"],
                "calibration_table": [run_directory / "tables" / "calibration.csv"],
                "fill_log": [run_directory / "records" / "fills.csv"],
            }
        )
    result: dict[str, bool] = {}
    for artifact_class in spec.values["artifacts"]["required_classes"]:
        if artifact_class == "manifest":
            result[str(artifact_class)] = True
            continue
        paths = paths_by_class.get(str(artifact_class), [])
        result[str(artifact_class)] = bool(paths) and all(path.is_file() for path in paths)
    return result


def _dirty_summary(repository_root: Path) -> bytes:
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    content = "Tracked diff summary:\n" + result.stdout
    content += "\nUntracked paths (contents omitted):\n" + untracked.stdout
    return content.encode("utf-8")


def run_experiment(spec: RunSpec, command: Sequence[str] | None = None) -> RunResult:
    repository_root = spec.repository_root
    git = git_provenance(repository_root)
    require_clean = bool(spec.values["acceptance"]["require_clean_tree_for_claim"])
    if require_clean and git["dirty"]:
        raise ExperimentError("claim-eligible run requires a clean Git worktree")

    started = utc_now()
    seed_marker = "det"
    run_id = f"{started.strftime('%Y%m%dT%H%M%S%fZ')}-{spec.sha256[:12]}-{seed_marker}"
    output_root = repository_root / str(spec.values["artifacts"]["output_root"])
    run_directory = output_root / spec.experiment_id / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    start_monotonic = time.perf_counter()
    copy_source(spec.source_path, run_directory / "source_config.toml")
    atomic_write_bytes(
        run_directory / "resolved_runspec.json", (spec.canonical_json + "\n").encode("utf-8")
    )

    warnings: list[str] = []
    dirty_artifact: dict[str, Any] | None = None
    if git["dirty"]:
        warnings.append("Run started from dirty worktree; claim eligibility is false.")
        content = _dirty_summary(repository_root)
        dirty_path = run_directory / "state" / "dirty-summary.txt"
        atomic_write_bytes(dirty_path, content)
        dirty_artifact = {
            "path": dirty_path.relative_to(run_directory).as_posix(),
            "sha256": sha256_bytes(content),
            "sanitization": "file contents omitted; tracked diff stat and untracked path names only",
        }

    status = "failed"
    acceptance_passed = False
    metrics: Mapping[str, Any] = {}
    acceptance: Mapping[str, bool] = {}
    derived: Mapping[str, Any] = {}
    log_lines = [
        f"start_utc={utc_text(started)}",
        f"experiment_id={spec.experiment_id}",
        f"runspec_sha256={spec.sha256}",
    ]
    failure_reason: str | None = None
    try:
        if spec.experiment_id == "ANA-SMOKE-001":
            evaluation = evaluate_smoke(spec)
        elif spec.experiment_id == "ANA-FIG3-001":
            evaluation = evaluate_fig3(spec, run_directory)
        elif spec.experiment_id.startswith(("SIM-MOMENTS-", "SIM-UNBALANCED-")):
            evaluation = evaluate_simulation(spec, run_directory)
        elif spec.experiment_id.startswith("SIM-FIG4-"):
            evaluation = evaluate_figure4(spec, run_directory)
        else:
            raise ExperimentError(f"experiment runner is not implemented: {spec.experiment_id}")
        metrics = evaluation.metrics
        acceptance = evaluation.acceptance
        derived = evaluation.derived_parameters
        log_lines.extend(evaluation.log_lines)
        acceptance_passed = evaluation.passed
        status = "passed" if acceptance_passed else "acceptance_failed"
    except Exception as error:
        failure_reason = f"{type(error).__name__}: {error}"
        log_lines.append(f"failure={failure_reason}")
        status = "failed"

    ended = utc_now()
    elapsed = time.perf_counter() - start_monotonic
    log_lines.extend((f"status={status}", f"elapsed_seconds={elapsed:.9f}"))
    atomic_write_bytes(
        run_directory / "logs" / "run.log", ("\n".join(log_lines) + "\n").encode("utf-8")
    )
    summary = {
        "schema_version": "metrics-summary-v1",
        "experiment_id": spec.experiment_id,
        "status": status,
        "acceptance_passed": acceptance_passed,
        "acceptance": _plain(acceptance),
        "metrics": _plain(metrics),
        "failure_reason": failure_reason,
    }
    atomic_write_json(run_directory / "metrics" / "summary.json", summary)

    required_artifacts = _required_artifacts_present(spec, run_directory)
    if not all(required_artifacts.values()) and status != "failed":
        status = "failed"
        acceptance_passed = False
        failure_reason = "missing required artifacts: " + ", ".join(
            key for key, present in required_artifacts.items() if not present
        )
        summary.update(
            status=status,
            acceptance_passed=False,
            failure_reason=failure_reason,
        )
        atomic_write_json(run_directory / "metrics" / "summary.json", summary)

    is_figure4 = spec.experiment_id.startswith("SIM-FIG4-")
    numerics_manifest = (
        {
            "market_float_dtype": spec.values["numerics"]["market_float_dtype"],
            "reduction_float_dtype": spec.values["numerics"]["reduction_float_dtype"],
            "device": "cpu_adaptive_market_plus_torch_compile_cuda_crossings_and_reduction",
            "known_nondeterministic_kernels": [],
        }
        if is_figure4
        else {
            "float_dtype": spec.values["numerics"]["float_dtype"],
            "device": "cpu",
            "known_nondeterministic_kernels": [],
        }
    )
    seed_manifest = (
        {
            "rng_used": spec.values["seed_policy"]["rng_used"],
            "rng_algorithm": spec.values["seed_policy"]["rng_algorithm"],
            "calibration_seeds": list(spec.values["seed_policy"]["calibration_seeds"]),
            "strategy_seeds": list(spec.values["seed_policy"]["strategy_seeds"]),
            "deterministic_replay_seeds": list(
                spec.values["seed_policy"]["deterministic_replay_seeds"]
            ),
        }
        if is_figure4
        else {
            "rng_used": spec.values["seed_policy"]["rng_used"],
            "rng_algorithm": spec.values["seed_policy"]["rng_algorithm"],
            "seed_to_replication": [
                {
                    "seed": int(seed),
                    "replication": index,
                    "consumed": bool(spec.values["seed_policy"]["rng_used"]),
                }
                for index, seed in enumerate(spec.values["seed_policy"]["seeds"])
            ],
        }
    )
    manifest = {
        "schema_version": "run-manifest-v1",
        "run_id": run_id,
        "experiment_id": spec.experiment_id,
        "status": status,
        "acceptance_passed": acceptance_passed,
        "started_at_utc": utc_text(started),
        "ended_at_utc": utc_text(ended),
        "elapsed_seconds": elapsed,
        "launch_command": list(command if command is not None else sys.argv),
        "repository": {
            "root": str(repository_root),
            **git,
            "dirty_artifact": dirty_artifact,
        },
        "source": {
            "config_path": spec.source_path.relative_to(repository_root).as_posix(),
            "config_sha256": spec.source_sha256,
            "resolved_runspec_path": "resolved_runspec.json",
            "resolved_runspec_file_sha256": sha256_file(
                run_directory / "resolved_runspec.json"
            ),
            "runspec_sha256": spec.sha256,
            "paper_version": spec.values["inputs"]["paper_version"],
            "paper_pdf_sha256": spec.values["inputs"]["paper_pdf_sha256"],
            "dataset": spec.values["inputs"]["dataset"],
        },
        "environment": environment_provenance(repository_root),
        "numerics": numerics_manifest,
        "seeds": seed_manifest,
        "derived_parameters": _plain(derived),
        "warnings": warnings,
        "deviations": [],
        "termination_reason": failure_reason or ("all acceptance gates passed" if acceptance_passed else "one or more acceptance gates failed"),
        "primary_metrics": _plain(metrics),
        "required_artifact_classes": required_artifacts,
        "artifacts": artifact_inventory(run_directory),
    }
    atomic_write_json(run_directory / "manifest.json", manifest)

    return RunResult(
        experiment_id=spec.experiment_id,
        run_id=run_id,
        run_directory=run_directory,
        status=status,
        acceptance_passed=acceptance_passed,
        metrics=metrics,
    )
