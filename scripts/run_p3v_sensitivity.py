from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from ot_micromr.analytics import dimensional_surrogate_rate, solve_dawson_optimum
from ot_micromr.config import load_runspec
from ot_micromr.sensitivity import evaluate_discrete_band_proxy
from ot_micromr.simulation_experiments import _ordered_simulations
from ot_micromr.simulator import settings_from_spec
from ot_micromr.statistical_gates import normal_approximation_sample_size


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the preregistered P3V pilot sensitivity")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/p3v"))
    parser.add_argument("--horizon", type=float, default=2000.0)
    return parser


def _scenario_values() -> list[tuple[str, float, float]]:
    return [
        ("flow_minus_0.10", (2.0 - 0.10) / (2.0 + 0.10), 1.0),
        ("baseline", 1.0, 1.0),
        ("flow_plus_0.10", (2.0 + 0.10) / (2.0 - 0.10), 1.0),
        ("open_drift_ratio_0.85", 1.0, 0.85),
        ("open_drift_ratio_1.15", 1.0, 1.15),
        ("open_drift_ratio_1.25_control", 1.0, 1.25),
    ]


def main() -> int:
    args = _parser().parse_args()
    source = load_runspec("cfg/experiments/sim_moments_001.toml")
    analytical = load_runspec("cfg/experiments/ana_smoke_001.toml")
    base = source.to_dict()
    base["simulation"]["event_log"] = False
    base["simulation"]["burn_in_reversion_times"] = 100.0
    base["simulation"]["horizon_reversion_times"] = args.horizon
    base["evaluation"]["minimum_observations_per_seed_and_parity_for_slope"] = 100
    seeds = tuple(int(seed) for seed in base["seed_policy"]["seeds"])
    multipliers = np.arange(0.60, 1.6001, 0.05, dtype=np.float64)
    scenario_payloads: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    rate_matrices: dict[str, np.ndarray] = {}
    surrogate_optima: dict[str, float] = {}
    estimator_rows: dict[str, dict[str, float]] = {}
    for name, opening_multiplier, open_drift_ratio in _scenario_values():
        values = json.loads(json.dumps(base))
        values["simulation"]["pilot_sampling_rate_multipliers"] = [
            1.0,
            1.0,
            opening_multiplier,
            opening_multiplier,
            1.0,
            1.0,
        ]
        values["model"]["alpha_c_per_second"] = open_drift_ratio
        results = _ordered_simulations(values, [(0.01, seed) for seed in seeds], args.workers)
        s_g = float(np.mean([result.seed_metrics["stationary_s_g"] for result in results]))
        gamma = 0.5 / s_g
        solution = solve_dawson_optimum(gamma, analytical.values["numerics"])
        theta_d = s_g * solution.u_d_ratio
        thresholds = theta_d * multipliers
        settings = settings_from_spec(values)
        rate_matrix = np.empty((len(results), multipliers.size), dtype=np.float64)
        open_fill_matrix = np.empty_like(rate_matrix)
        for row_index, result in enumerate(results):
            policies = evaluate_discrete_band_proxy(
                result.gaps,
                result.tight,
                settings.observation_interval_seconds,
                thresholds,
            )
            rate_matrix[row_index] = [policy.reward_rate_per_second for policy in policies]
            open_fill_matrix[row_index] = [policy.open_fill_share for policy in policies]
        mean_rates = np.mean(rate_matrix, axis=0)
        rate_se = np.std(rate_matrix, axis=0, ddof=1) / math.sqrt(rate_matrix.shape[0])
        surrogate_optimum = float(
            dimensional_surrogate_rate(solution.u_d_ratio, gamma, 1.0, s_g)
        )
        peak_index = int(np.argmax(mean_rates))
        theta_d_index = int(np.flatnonzero(np.isclose(multipliers, 1.0, atol=1e-12))[0])
        peak_rate = float(mean_rates[peak_index])
        loss_at_theta_d = 1.0 - float(mean_rates[theta_d_index]) / peak_rate
        flow_mean = float(
            np.mean(
                [
                    result.seed_metrics["integrated_hazard_flow_signed_relative_residual"]
                    for result in results
                ]
            )
        )
        target = math.expm1(-0.01) / 0.01
        open_drift_mean = float(
            np.mean(
                [result.seed_metrics["finite_h_drift_slope_open_per_second"] / target for result in results]
            )
        )
        jump_open_ratios = np.asarray(
            [-result.seed_metrics["realised_jump_drift_slope_open_per_second"] for result in results],
            dtype=np.float64,
        )
        jump_contrasts = np.asarray(
            [
                result.seed_metrics["realised_jump_parity_drift_slope_contrast_per_second"]
                for result in results
            ],
            dtype=np.float64,
        )
        flow_values = np.asarray(
            [
                result.seed_metrics["integrated_hazard_flow_signed_relative_residual"]
                for result in results
            ],
            dtype=np.float64,
        )
        fit_left = max(0, peak_index - 2)
        fit_right = min(multipliers.size, peak_index + 3)
        coefficients = np.polyfit(
            multipliers[fit_left:fit_right],
            mean_rates[fit_left:fit_right],
            deg=2,
        )
        fitted_peak = (
            float(-coefficients[1] / (2.0 * coefficients[0]))
            if coefficients[0] < 0.0
            else float(multipliers[peak_index])
        )
        fitted_peak = float(np.clip(fitted_peak, multipliers[fit_left], multipliers[fit_right - 1]))
        rate_matrices[name] = rate_matrix
        surrogate_optima[name] = surrogate_optimum
        estimator_rows[name] = {
            "flow_sd": float(np.std(flow_values, ddof=1)),
            "jump_open_drift_ratio_mean": float(np.mean(jump_open_ratios)),
            "jump_open_drift_ratio_sd": float(np.std(jump_open_ratios, ddof=1)),
            "jump_parity_contrast_mean": float(np.mean(jump_contrasts)),
            "jump_parity_contrast_sd": float(np.std(jump_contrasts, ddof=1)),
        }
        scenario_payloads.append(
            {
                "scenario": name,
                "opening_sampling_multiplier": opening_multiplier,
                "declared_open_drift_ratio": open_drift_ratio,
                "measured_flow_residual": flow_mean,
                "measured_open_drift_target_ratio": open_drift_mean,
                "stationary_s_g": s_g,
                "gamma": gamma,
                "theta_d_price": theta_d,
                "selected_peak_multiplier": float(multipliers[peak_index]),
                "fitted_peak_multiplier": fitted_peak,
                "peak_normalized_rate": peak_rate / surrogate_optimum,
                "loss_at_theta_d_fraction": loss_at_theta_d,
                "open_fill_share_at_peak": float(np.mean(open_fill_matrix[:, peak_index])),
            }
        )
        for index, multiplier in enumerate(multipliers):
            curve_rows.append(
                {
                    "scenario": name,
                    "threshold_multiplier": float(multiplier),
                    "mean_rate_per_second": float(mean_rates[index]),
                    "rate_standard_error": float(rate_se[index]),
                    "normalized_rate": float(mean_rates[index] / surrogate_optimum),
                    "mean_open_fill_share": float(np.mean(open_fill_matrix[:, index])),
                }
            )

    pilot_sds = {
        "flow": estimator_rows["baseline"]["flow_sd"],
        "open_drift_ratio": estimator_rows["baseline"]["jump_open_drift_ratio_sd"],
        "unbalanced_contrast": estimator_rows["open_drift_ratio_1.25_control"][
            "jump_parity_contrast_sd"
        ],
    }
    local_alpha = 0.05 / 2.0
    power_rows = [
        {
            "component": "flow_equivalence",
            "planned_horizon": 40000.0,
            "projected_sd": pilot_sds["flow"] * math.sqrt(args.horizon / 40000.0),
            "distance_to_null_boundary": 0.05,
        },
        {
            "component": "unbalanced_contrast_superiority",
            "planned_horizon": 20000.0,
            "projected_sd": pilot_sds["unbalanced_contrast"]
            * math.sqrt(args.horizon / 20000.0),
            "distance_to_null_boundary": 0.25 - 0.10,
        },
    ]
    for row in power_rows:
        row["normal_approx_required_seeds"] = normal_approximation_sample_size(
            standard_deviation=float(row["projected_sd"]),
            distance_to_null=float(row["distance_to_null_boundary"]),
            alpha=local_alpha,
            power=0.90,
        )
    baseline = next(item for item in scenario_payloads if item["scenario"] == "baseline")
    bootstrap_rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(2026081198)))
    bootstrap_indices = bootstrap_rng.integers(0, len(seeds), size=(5000, len(seeds)))
    baseline_bootstrap_rates = np.mean(rate_matrices["baseline"][bootstrap_indices], axis=1)
    baseline_bootstrap_peak = multipliers[np.argmax(baseline_bootstrap_rates, axis=1)]
    baseline_peak_rates = np.max(baseline_bootstrap_rates, axis=1) / surrogate_optima["baseline"]
    theta_d_index = int(np.flatnonzero(np.isclose(multipliers, 1.0, atol=1e-12))[0])
    baseline_losses = 1.0 - baseline_bootstrap_rates[:, theta_d_index] / np.max(
        baseline_bootstrap_rates, axis=1
    )
    for item in scenario_payloads:
        item["peak_grid_change_vs_baseline"] = (
            float(item["selected_peak_multiplier"])
            - float(baseline["selected_peak_multiplier"])
        )
        item["peak_normalized_rate_change_vs_baseline"] = (
            float(item["peak_normalized_rate"])
            - float(baseline["peak_normalized_rate"])
        )
        item["loss_at_theta_d_change_vs_baseline"] = (
            float(item["loss_at_theta_d_fraction"])
            - float(baseline["loss_at_theta_d_fraction"])
        )
        name = str(item["scenario"])
        sampled_rates = np.mean(rate_matrices[name][bootstrap_indices], axis=1)
        peak_changes = multipliers[np.argmax(sampled_rates, axis=1)] - baseline_bootstrap_peak
        rate_changes = np.max(sampled_rates, axis=1) / surrogate_optima[name] - baseline_peak_rates
        loss_changes = (
            1.0 - sampled_rates[:, theta_d_index] / np.max(sampled_rates, axis=1)
        ) - baseline_losses
        item["paired_bootstrap_peak_grid_change_95_interval"] = [
            float(value) for value in np.quantile(peak_changes, (0.025, 0.975))
        ]
        item["paired_bootstrap_peak_normalized_rate_change_95_interval"] = [
            float(value) for value in np.quantile(rate_changes, (0.025, 0.975))
        ]
        item["paired_bootstrap_loss_at_theta_d_change_95_interval"] = [
            float(value) for value in np.quantile(loss_changes, (0.025, 0.975))
        ]
        item["estimator_pilot"] = estimator_rows[name]
    payload = {
        "schema_version": "p3v-sensitivity-pilot-v1",
        "claim_eligible": False,
        "reason": "Old P3 seed labels and endpoint-only band monitor are pilot-only.",
        "seed_count": len(seeds),
        "horizon_reversion_times": args.horizon,
        "epsilon": 0.01,
        "threshold_grid_step_theta_d": 0.05,
        "material_rate_resolution_fraction": 0.01,
        "scenarios": scenario_payloads,
        "bootstrap_replications": 5000,
        "power_design": {
            "familywise_alpha": 0.05,
            "worst_case_local_alpha": local_alpha,
            "target_power": 0.90,
            "planned_seed_count": 20,
            "planned_balanced_horizon_reversion_times": 40000.0,
            "planned_control_horizon_reversion_times": 20000.0,
            "components": power_rows,
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "pilot-sensitivity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_root / "pilot-sensitivity-curves.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
