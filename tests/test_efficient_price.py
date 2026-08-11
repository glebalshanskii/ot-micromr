import math
import unittest

import torch

from ot_micromr.efficient_price import (
    interval_log_likelihood,
    six_event_intensities,
    systematic_resample,
)


MODEL = {
    "delta_price": 1.0,
    "mu_s_per_second": 1.0,
    "mu_o_per_second": 0.02,
    "mu_c_per_second": 2.0,
    "alpha_s_per_second": 0.4,
    "alpha_o_per_second": 0.2,
    "alpha_c_per_second": 1.0,
}


class EfficientPriceTests(unittest.TestCase):
    def test_six_event_intensities_match_tight_and_open_formulas(self) -> None:
        mid = torch.tensor([1.0, 1.0])
        efficient = torch.tensor([2.0, 0.0])
        tight = torch.tensor([True, False])
        rates = six_event_intensities(
            mid,
            efficient,
            tight,
            delta=1.0,
            mu_s=1.0,
            mu_o=0.02,
            mu_c=2.0,
            alpha_s=0.4,
            alpha_o=0.2,
            alpha_c=1.0,
        )
        torch.testing.assert_close(
            rates,
            torch.tensor(
                [
                    [1.8, 1.0, 0.42, 0.02, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 2.0, 4.0],
                ]
            ),
        )

    def test_interval_likelihood_includes_silence_and_channel(self) -> None:
        mid = torch.tensor([0.5, 0.5])
        efficient = torch.tensor([0.5, 0.5])
        tight = torch.tensor([True, True])
        event = torch.tensor([0, 1])
        dt = 0.005
        actual = interval_log_likelihood(mid, efficient, tight, event, dt, MODEL)
        total = 2.0 * (1.0 + 0.02)
        expected = torch.tensor(
            [
                -total * dt,
                math.log(-math.expm1(-total * dt)) + math.log(1.0) - math.log(total),
            ]
        )
        torch.testing.assert_close(actual, expected)

    def test_systematic_resampling_is_batched_and_resets_weights(self) -> None:
        particles = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        weights = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        log_weights = torch.log(weights)
        resampled, reset = systematic_resample(
            particles, log_weights, torch.tensor([0.1])
        )
        torch.testing.assert_close(resampled, torch.tensor([[40.0, 40.0, 40.0, 40.0]]))
        torch.testing.assert_close(reset, torch.full_like(reset, -math.log(4.0)))


if __name__ == "__main__":
    unittest.main()
