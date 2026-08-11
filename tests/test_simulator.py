import unittest

import numpy as np

from ot_micromr.config import load_runspec
from ot_micromr.simulator import SimulationSettings, named_streams, simulate_replication


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_runspec("cfg/experiments/sim_moments_001.toml")
        cls.values = cls.spec.to_dict()
        cls.values["model"]["mu_o_per_second"] = 2.0
        cls.settings = SimulationSettings(
            burn_seconds=0.1,
            horizon_seconds=1.0,
            observation_interval_seconds=0.01,
            alpha_ref_per_second=1.0,
            diagnostic_quantiles=(0.0, 0.5, 1.0),
            acf_lags_seconds=(0.05, 0.1),
            minimum_slope_observations=1,
        )

    def test_tiny_path_invariants_and_observation_schedule(self) -> None:
        result = simulate_replication(
            self.values,
            epsilon=0.01,
            seed=2026081101,
            settings=self.settings,
        )
        self.assertEqual(result.gaps.shape, (101,))
        self.assertEqual(result.tight.shape, (101,))
        self.assertTrue(np.all(np.isfinite(result.gaps)))
        self.assertLessEqual(result.seed_metrics["maximum_left_event_probability"], 0.01 + 2e-15)
        self.assertEqual(result.seed_metrics["invariant_violation_count"], 0)
        self.assertLessEqual(result.seed_metrics["generator_drift_abs_residual_tight"], 1e-12)
        self.assertLessEqual(result.seed_metrics["generator_drift_abs_residual_open"], 1e-12)
        self.assertIsNone(result.seed_metrics["bridge_only_crossing_count"])
        self.assertFalse(result.seed_metrics["deterministic_replay_checked"])
        self.assertLessEqual(abs(result.seed_metrics["transition_count_imbalance"]), 1)
        self.assertIn("integrated_hazard_flow_signed_relative_residual", result.seed_metrics)
        self.assertIn("compensator_z_open_up", result.seed_metrics)

    def test_event_log_can_be_disabled_without_changing_path(self) -> None:
        recorded = simulate_replication(
            self.values,
            epsilon=0.01,
            seed=2026081105,
            settings=self.settings,
        )
        unrecorded_settings = SimulationSettings(
            burn_seconds=self.settings.burn_seconds,
            horizon_seconds=self.settings.horizon_seconds,
            observation_interval_seconds=self.settings.observation_interval_seconds,
            alpha_ref_per_second=self.settings.alpha_ref_per_second,
            diagnostic_quantiles=self.settings.diagnostic_quantiles,
            acf_lags_seconds=self.settings.acf_lags_seconds,
            minimum_slope_observations=self.settings.minimum_slope_observations,
            record_events=False,
        )
        unrecorded = simulate_replication(
            self.values,
            epsilon=0.01,
            seed=2026081105,
            settings=unrecorded_settings,
        )
        self.assertEqual(recorded.replay_digest, unrecorded.replay_digest)
        self.assertGreater(len(recorded.events), 0)
        self.assertEqual(unrecorded.events, ())
        np.testing.assert_array_equal(recorded.gaps, unrecorded.gaps)

    def test_bitwise_replay_is_deterministic(self) -> None:
        first = simulate_replication(self.values, 0.01, 2026081102, settings=self.settings)
        second = simulate_replication(self.values, 0.01, 2026081102, settings=self.settings)
        self.assertEqual(first.replay_digest, second.replay_digest)
        np.testing.assert_array_equal(first.gaps, second.gaps)
        np.testing.assert_array_equal(first.tight, second.tight)

    def test_distinct_seeds_change_path(self) -> None:
        first = simulate_replication(self.values, 0.01, 2026081103, settings=self.settings)
        second = simulate_replication(self.values, 0.01, 2026081104, settings=self.settings)
        self.assertNotEqual(first.replay_digest, second.replay_digest)

    def test_bridge_draws_do_not_shift_endpoint_or_book_streams(self) -> None:
        first, keys = named_streams(12345)
        second, second_keys = named_streams(12345)
        self.assertEqual(keys, second_keys)
        _ = second["brownian_bridge"].normal(size=100)
        for name in ("brownian_increment", "book_occurrence", "book_channel"):
            np.testing.assert_array_equal(first[name].random(20), second[name].random(20))


if __name__ == "__main__":
    unittest.main()
