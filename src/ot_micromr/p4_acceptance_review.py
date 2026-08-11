from __future__ import annotations

import csv
import json
import math
import subprocess
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ot_micromr.artifacts import atomic_write_json, sha256_file
from ot_micromr.errors import ExperimentError


POLICY_ID = "p4-operational-validity-v2"
REVIEW_EXPERIMENT_ID = "SIM-FIG4-REVIEW-001"
SUPERSEDED_GATE = "minimum_complete_intervals"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentError(f"cannot read JSON evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExperimentError(f"JSON evidence must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError as error:
        raise ExperimentError(f"cannot read CSV evidence {path}: {error}") from error


def _finite(row: Mapping[str, str], fields: Sequence[str]) -> bool:
    try:
        return all(math.isfinite(float(row[field])) for field in fields)
    except (KeyError, TypeError, ValueError):
        return False


def _git_implementation_provenance(repository_root: Path) -> tuple[str | None, bool | None]:
    implementation_path = Path("src/ot_micromr/p4_acceptance_review.py")
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", implementation_path.as_posix()],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            implementation_path.as_posix(),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        commit.stdout.strip() if commit.returncode == 0 else None,
        bool(status.stdout.strip()) if status.returncode == 0 else None,
    )


def evaluate_p4_acceptance_evidence(
    summary: Mapping[str, Any],
    source_config: Mapping[str, Any],
    diagnostic_rows: Sequence[Mapping[str, str]],
    policy_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Apply the corrected P4 validity policy without rerunning the simulation.

    Complete seed paths are the independent statistical units. Inter-fill counts are
    therefore reported as exposure diagnostics; only an undefined estimator (no complete
    interval or non-finite rate) is an operational validity failure.
    """

    if summary.get("experiment_id") != "SIM-FIG4-002":
        raise ExperimentError("P4 review accepts only SIM-FIG4-002 evidence")

    try:
        epsilons = tuple(float(value) for value in source_config["numerics"]["refinement_epsilons"])
        response_rows = tuple(
            range(len(source_config["model"]["response_scale_alpha_per_second_grid"]))
        )
        seeds = tuple(int(value) for value in source_config["seed_policy"]["strategy_seeds"])
        policy_count = len(source_config["strategy"]["threshold_multiplier_theta_over_theta_d_grid"]) + len(
            source_config["strategy"]["additional_thresholds"]
        )
        diagnostic_floor = int(
            source_config["evaluation"][
                "minimum_complete_interfill_intervals_per_seed_and_policy"
            ]
        )
        omitted_budget = float(source_config["acceptance"]["omitted_probability_sum_max"])
        dawson_budget = float(source_config["acceptance"]["dawson_root_abs_residual_max"])
    except (KeyError, TypeError, ValueError) as error:
        raise ExperimentError(f"invalid frozen P4 source config: {error}") from error

    expected_coordinates = {
        (epsilon, row_index, seed)
        for epsilon in epsilons
        for row_index in response_rows
        for seed in seeds
    }
    diagnostic_coordinates = [
        (float(row["epsilon"]), int(row["row_index"]), int(row["seed"]))
        for row in diagnostic_rows
    ]
    grouped_policies: dict[tuple[float, int, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in policy_rows:
        grouped_policies[
            (float(row["epsilon"]), int(row["row_index"]), int(row["seed"]))
        ].append(row)

    expected_policy_indices = set(range(policy_count))
    complete_policy_grid = set(grouped_policies) == expected_coordinates and all(
        len(rows) == policy_count
        and {int(row["policy_index"]) for row in rows} == expected_policy_indices
        for rows in grouped_policies.values()
    )
    finite_policy_values = all(
        _finite(
            row,
            (
                "renewal_rate_per_second",
                "renewal_rate_over_alpha_s_g",
                "renewal_rate_over_surrogate_optimum",
                "threshold_price",
                "wealth_marking_identity_abs_residual",
            ),
        )
        for row in policy_rows
    )
    defined_estimators = all(
        int(row["complete_interval_count"]) >= 1
        and row.get("mean_interfill_seconds", "") not in ("", None)
        and math.isfinite(float(row["mean_interfill_seconds"]))
        for row in policy_rows
    )
    maximum_omitted_probability = max(
        (
            float(row["omitted_bridge_probability_sum"])
            + float(row["full_band_recrossing_probability_bound"])
            for row in diagnostic_rows
        ),
        default=math.inf,
    )
    maximum_wealth_residual = max(
        (float(row["wealth_marking_identity_abs_residual"]) for row in policy_rows),
        default=math.inf,
    )
    calibration = summary.get("metrics", {}).get("calibration", [])
    maximum_dawson_residual = max(
        (float(row["root_abs_residual"]) for row in calibration), default=math.inf
    )

    operational_gates = {
        "complete_coordinate_grid": set(diagnostic_coordinates) == expected_coordinates
        and len(diagnostic_coordinates) == len(expected_coordinates),
        "complete_policy_grid": complete_policy_grid,
        "defined_rate_estimators": defined_estimators,
        "finite_policy_values": finite_policy_values,
        "nonflat_before_measurement": all(
            int(row["nonflat_policy_count_at_end"]) == int(row["policy_count"])
            for row in diagnostic_rows
        ),
        "invariant_violation_count": all(
            int(row["invariant_violation_count"]) == 0 for row in diagnostic_rows
        ),
        "nonfinite_value_count": all(
            int(row["nonfinite_value_count"]) == 0 for row in diagnostic_rows
        ),
        "omitted_probability_budget": maximum_omitted_probability <= omitted_budget,
        "dawson_root_residual": maximum_dawson_residual <= dawson_budget,
        "wealth_marking_identity": maximum_wealth_residual <= 1e-10,
        "deterministic_replay": int(
            summary.get("metrics", {}).get("deterministic_replay_mismatch_count", -1)
        )
        == 0,
    }
    operational_passed = all(operational_gates.values())

    below_floor = [
        row for row in policy_rows if int(row["complete_interval_count"]) < diagnostic_floor
    ]
    below_by_row_resolution: dict[str, int] = defaultdict(int)
    for row in below_floor:
        key = f"row_{int(row['row_index'])}/epsilon_{float(row['epsilon']):g}"
        below_by_row_resolution[key] += 1

    peak_minimums: list[dict[str, Any]] = []
    primary_epsilon = float(source_config["numerics"]["primary_resolution_epsilon"])
    for functional in summary.get("metrics", {}).get("primary_functionals", []):
        row_index = int(functional["row_index"])
        peak = float(functional["discrete_peak_multiplier_theta_d"])
        matching = [
            int(row["complete_interval_count"])
            for row in policy_rows
            if float(row["epsilon"]) == primary_epsilon
            and int(row["row_index"]) == row_index
            and math.isclose(
                float(row["threshold_multiplier_theta_d"]), peak, rel_tol=0.0, abs_tol=1e-12
            )
        ]
        peak_minimums.append(
            {
                "row_index": row_index,
                "peak_multiplier_theta_d": peak,
                "minimum_complete_interval_count": min(matching) if matching else None,
            }
        )

    legacy_acceptance = dict(summary.get("acceptance", {}))
    scientific_gates = summary.get("metrics", {}).get("scientific_gates", {})
    scientific_status = scientific_gates.get("status", "unknown")
    return {
        "schema_version": "p4-acceptance-review-v2",
        "policy_id": POLICY_ID,
        "source_experiment_id": summary["experiment_id"],
        "source_run_id": summary.get("run_id"),
        "legacy_decision": {
            "status": summary.get("status"),
            "acceptance_passed": summary.get("acceptance_passed"),
            "acceptance": legacy_acceptance,
        },
        "correction": {
            "superseded_gate": SUPERSEDED_GATE,
            "reason": (
                "The all-cell inter-fill count floor was an unpowered heuristic applied "
                "to dependent within-path cycles; complete seed paths are the independent units."
            ),
            "raw_simulation_changed": False,
            "scientific_estimands_or_tests_changed": False,
        },
        "corrected_operational_decision": {
            "acceptance_passed": operational_passed,
            "gates": operational_gates,
        },
        "coverage_diagnostic": {
            "does_not_affect_acceptance": True,
            "historical_floor": diagnostic_floor,
            "minimum_complete_interval_count": min(
                (int(row["complete_interval_count"]) for row in policy_rows), default=None
            ),
            "cells_below_historical_floor": len(below_floor),
            "total_cells": len(policy_rows),
            "fraction_below_historical_floor": (
                len(below_floor) / len(policy_rows) if policy_rows else None
            ),
            "below_floor_by_row_resolution": dict(sorted(below_by_row_resolution.items())),
            "primary_peak_cells": peak_minimums,
        },
        "numerical_diagnostics": {
            "maximum_omitted_probability_bound": maximum_omitted_probability,
            "maximum_wealth_marking_identity_abs_residual": maximum_wealth_residual,
            "maximum_dawson_root_abs_residual": maximum_dawson_residual,
        },
        "scientific_decision": {
            "status": scientific_status,
            "unchanged_from_source": True,
            "gates": scientific_gates,
        },
        "stage_status": (
            f"completed / operational validity "
            f"{'passed' if operational_passed else 'failed'} / scientific {scientific_status}"
        ),
    }


def review_p4_run(run_directory: Path, output_root: Path) -> tuple[Path, dict[str, Any]]:
    run_directory = run_directory.resolve()
    evidence_paths = {
        "summary": run_directory / "metrics" / "summary.json",
        "path_diagnostics": run_directory / "metrics" / "path_diagnostics.csv",
        "seed_threshold_metrics": run_directory / "metrics" / "seed_threshold_metrics.csv",
        "source_config": run_directory / "source_config.toml",
        "manifest": run_directory / "manifest.json",
    }
    missing = [str(path) for path in evidence_paths.values() if not path.is_file()]
    if missing:
        raise ExperimentError("missing P4 review evidence: " + ", ".join(missing))

    summary = _read_json(evidence_paths["summary"])
    manifest = _read_json(evidence_paths["manifest"])
    if manifest.get("experiment_id") != summary.get("experiment_id"):
        raise ExperimentError("P4 summary and manifest experiment IDs disagree")
    manifest_run_id = manifest.get("run_id")
    if not isinstance(manifest_run_id, str) or not manifest_run_id:
        raise ExperimentError("P4 manifest does not contain a valid run_id")
    if summary.get("run_id") not in (None, manifest_run_id):
        raise ExperimentError("P4 summary and manifest run IDs disagree")
    try:
        source_config = tomllib.loads(evidence_paths["source_config"].read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ExperimentError(f"cannot read frozen P4 source config: {error}") from error
    result = evaluate_p4_acceptance_evidence(
        summary,
        source_config,
        _read_csv(evidence_paths["path_diagnostics"]),
        _read_csv(evidence_paths["seed_threshold_metrics"]),
    )
    result["source_run_id"] = manifest_run_id

    repository_root = Path(__file__).resolve().parents[2]
    commit, dirty = _git_implementation_provenance(repository_root)
    result["review_provenance"] = {
        "implementation_commit": commit,
        "implementation_file_dirty": dirty,
        "implementation_file_sha256": sha256_file(Path(__file__)),
        "source_files": {
            name: {
                "path": path.relative_to(repository_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in evidence_paths.items()
        },
    }
    source_run_id = manifest_run_id
    destination = (
        output_root.resolve()
        / REVIEW_EXPERIMENT_ID
        / f"{source_run_id}-acceptance-v2"
        / "metrics"
        / "review.json"
    )
    if destination.exists():
        existing = _read_json(destination)
        if existing != result:
            raise ExperimentError(f"immutable review artifact already differs: {destination}")
    else:
        atomic_write_json(destination, result)
    return destination, result
