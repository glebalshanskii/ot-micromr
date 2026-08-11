import unittest

import numpy as np

from ot_micromr.sensitivity import evaluate_discrete_band_proxy


class SensitivityTests(unittest.TestCase):
    def test_discrete_band_proxy_alternates_and_vectorises_rewards(self) -> None:
        results = evaluate_discrete_band_proxy(
            gaps=np.asarray([-2.0, 0.0, 2.0, 0.0, -2.0]),
            tight=np.asarray([True, True, True, True, False]),
            observation_interval_seconds=1.0,
            thresholds_price=np.asarray([1.0, 3.0]),
        )
        self.assertEqual(results[0].fill_count, 3)
        self.assertEqual(results[0].completed_flip_count, 2)
        self.assertAlmostEqual(results[0].reward_rate_per_second, 1.25)
        self.assertAlmostEqual(results[0].open_fill_share, 0.5)
        self.assertEqual(results[1].fill_count, 0)
        self.assertEqual(results[1].reward_rate_per_second, 0.0)


if __name__ == "__main__":
    unittest.main()
