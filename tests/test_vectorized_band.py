from __future__ import annotations

import unittest

import numpy as np

from ot_micromr.sensitivity import evaluate_discrete_band_proxy
from ot_micromr.vectorized_band import evaluate_discrete_band_proxy_batched_numpy


class VectorizedBandTests(unittest.TestCase):
    def test_matches_scalar_reference(self) -> None:
        rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(917)))
        gaps = rng.normal(size=(4, 1000))
        tight = rng.random((4, 1000)) > 0.1
        thresholds = np.asarray([0.6, 0.9, 1.2, 1.5])
        result = evaluate_discrete_band_proxy_batched_numpy(gaps, tight, 0.01, thresholds)
        for path_index in range(gaps.shape[0]):
            expected = evaluate_discrete_band_proxy(
                gaps[path_index], tight[path_index], 0.01, thresholds
            )
            np.testing.assert_array_equal(
                result.fill_count[path_index], [item.fill_count for item in expected]
            )
            np.testing.assert_array_equal(
                result.completed_flip_count[path_index],
                [item.completed_flip_count for item in expected],
            )
            np.testing.assert_allclose(
                result.reward_rate_per_second[path_index],
                [item.reward_rate_per_second for item in expected],
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                result.open_fill_share[path_index],
                [item.open_fill_share for item in expected],
                rtol=0.0,
                atol=1e-12,
            )

    def test_zero_fills(self) -> None:
        result = evaluate_discrete_band_proxy_batched_numpy(
            np.zeros((2, 10), dtype=np.float32),
            np.ones((2, 10), dtype=np.bool_),
            0.1,
            np.asarray([0.5, 1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(result.fill_count, 0)
        np.testing.assert_array_equal(result.completed_flip_count, 0)
        np.testing.assert_array_equal(result.reward_rate_per_second, 0.0)
        np.testing.assert_array_equal(result.open_fill_share, 0.0)

    def test_rejects_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            evaluate_discrete_band_proxy_batched_numpy(
                np.zeros(10), np.ones(10, dtype=np.bool_), 0.1, np.asarray([1.0])
            )


if __name__ == "__main__":
    unittest.main()
