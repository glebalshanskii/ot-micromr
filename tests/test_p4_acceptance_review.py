from __future__ import annotations

import copy
import unittest

from ot_micromr.p4_acceptance_review import evaluate_p4_acceptance_evidence


class P4AcceptanceReviewTests(unittest.TestCase):
    def _evidence(self) -> tuple[dict, dict, list[dict[str, str]], list[dict[str, str]]]:
        summary = {
            "experiment_id": "SIM-FIG4-002",
            "run_id": "target-run",
            "status": "acceptance_failed",
            "acceptance_passed": False,
            "acceptance": {"minimum_complete_intervals": False},
            "metrics": {
                "calibration": [{"root_abs_residual": 1e-14}],
                "deterministic_replay_mismatch_count": 0,
                "primary_functionals": [
                    {
                        "row_index": 0,
                        "discrete_peak_multiplier_theta_d": 1.0,
                    }
                ],
                "scientific_gates": {"status": "inconclusive"},
            },
        }
        config = {
            "numerics": {
                "primary_resolution_epsilon": 0.01,
                "refinement_epsilons": [0.01],
            },
            "model": {"response_scale_alpha_per_second_grid": [1.0]},
            "seed_policy": {"strategy_seeds": [101]},
            "strategy": {
                "threshold_multiplier_theta_over_theta_d_grid": [1.0],
                "additional_thresholds": [],
            },
            "evaluation": {
                "minimum_complete_interfill_intervals_per_seed_and_policy": 20,
            },
            "acceptance": {
                "omitted_probability_sum_max": 1e-5,
                "dawson_root_abs_residual_max": 1e-10,
            },
        }
        diagnostics = [
            {
                "epsilon": "0.01",
                "row_index": "0",
                "seed": "101",
                "nonflat_policy_count_at_end": "1",
                "policy_count": "1",
                "invariant_violation_count": "0",
                "nonfinite_value_count": "0",
                "omitted_bridge_probability_sum": "1e-12",
                "full_band_recrossing_probability_bound": "0",
            }
        ]
        policies = [
            {
                "epsilon": "0.01",
                "row_index": "0",
                "seed": "101",
                "policy_index": "0",
                "complete_interval_count": "12",
                "mean_interfill_seconds": "1.5",
                "renewal_rate_per_second": "0.2",
                "renewal_rate_over_alpha_s_g": "0.2",
                "renewal_rate_over_surrogate_optimum": "0.8",
                "threshold_multiplier_theta_d": "1.0",
                "threshold_price": "1.2",
                "wealth_marking_identity_abs_residual": "1e-14",
            }
        ]
        return summary, config, diagnostics, policies

    def test_sparse_but_defined_cell_is_diagnostic_not_failure(self) -> None:
        review = evaluate_p4_acceptance_evidence(*self._evidence())
        self.assertTrue(review["corrected_operational_decision"]["acceptance_passed"])
        self.assertEqual(review["coverage_diagnostic"]["cells_below_historical_floor"], 1)
        self.assertTrue(review["coverage_diagnostic"]["does_not_affect_acceptance"])

    def test_undefined_rate_estimator_remains_hard_failure(self) -> None:
        summary, config, diagnostics, policies = self._evidence()
        invalid = copy.deepcopy(policies)
        invalid[0]["complete_interval_count"] = "0"
        invalid[0]["mean_interfill_seconds"] = ""
        review = evaluate_p4_acceptance_evidence(summary, config, diagnostics, invalid)
        self.assertFalse(review["corrected_operational_decision"]["acceptance_passed"])
        self.assertFalse(
            review["corrected_operational_decision"]["gates"]["defined_rate_estimators"]
        )


if __name__ == "__main__":
    unittest.main()
