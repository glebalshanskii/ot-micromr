from __future__ import annotations

import unittest

from ot_micromr.config import load_runspec
from ot_micromr.figure4 import calibrate_rows, replay_figure4_coordinate, simulate_figure4
from ot_micromr.figure4_market import simulate_market_trace


class Figure4SimulationTests(unittest.TestCase):
    def _tiny_values(self) -> dict:
        values = load_runspec("cfg/experiments/sim_fig4_pilot_001.toml").to_dict()
        values["model"]["response_scale_alpha_per_second_grid"] = [0.4]
        values["seed_policy"]["calibration_seeds"] = [101, 102]
        values["simulation"]["calibration_burn_in_reversion_times"] = 2.0
        values["simulation"]["calibration_sampling_reversion_times"] = 5.0
        values["simulation"]["calibration_observation_interval_reversion_times"] = 0.1
        values["simulation"]["market_burn_in_reversion_times"] = 2.0
        values["simulation"]["strategy_burn_in_reversion_times"] = 10.0
        values["simulation"]["horizon_reversion_times"] = 20.0
        values["strategy"]["threshold_multiplier_theta_over_theta_d_grid"] = [0.8, 1.0]
        values["evaluation"]["minimum_complete_interfill_intervals_per_seed_and_policy"] = 1
        values["numerics"]["cpu_workers"] = 1
        return values

    def test_tiny_strategy_is_deterministic_and_preserves_accounting(self) -> None:
        values = self._tiny_values()
        calibrations = calibrate_rows(values)
        self.assertEqual(len(calibrations), 1)
        self.assertGreater(calibrations[0].s_g_price, 0.0)
        result = simulate_figure4(values, calibrations, [(0, 0.01, 201)], 1)[0]
        replay = replay_figure4_coordinate(values, calibrations[0], 0.01, 201)
        self.assertEqual(result.replay_digest, replay.replay_digest)
        self.assertEqual(len(result.policy_rows), 3)
        self.assertEqual(result.diagnostics["invariant_violation_count"], 0)
        self.assertLessEqual(
            result.diagnostics["omitted_bridge_probability_sum"],
            values["numerics"]["omitted_probability_budget"],
        )
        for policy in result.policy_rows:
            self.assertIn(policy["terminal_position"], (-1, 1))
            self.assertLessEqual(policy["wealth_marking_identity_abs_residual"], 1e-10)
            self.assertGreaterEqual(policy["fill_count"], 1)

    def test_calibration_retains_final_observation_under_roundoff(self) -> None:
        values = self._tiny_values()
        values["model"]["response_scale_alpha_per_second_grid"] = [0.3]
        values["simulation"]["calibration_burn_in_reversion_times"] = 1.0
        values["simulation"]["calibration_sampling_reversion_times"] = 2.0
        values["simulation"]["calibration_observation_interval_reversion_times"] = 0.02
        rows = calibrate_rows(values)
        expected_per_seed = 101
        self.assertEqual(
            rows[0].observation_count,
            expected_per_seed * len(values["seed_policy"]["calibration_seeds"]),
        )

    def test_market_trace_replays_bitwise_without_policy_work(self) -> None:
        values = self._tiny_values()
        calibration = calibrate_rows(values)[0]
        first = simulate_market_trace(values, calibration, 0.01, 301)
        second = simulate_market_trace(values, calibration, 0.01, 301)
        self.assertEqual(first.replay_digest, second.replay_digest)
        self.assertEqual(first.left_time_seconds.size, first.post_event_gap_price.size)
        self.assertLessEqual(first.maximum_left_event_probability, 0.01 + 2e-15)


if __name__ == "__main__":
    unittest.main()
