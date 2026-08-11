import unittest

import torch

from ot_micromr.empirical_filter import (
    _causal_ewma,
    _empirical_interval_nll,
    classify_paper_channels,
)


class EmpiricalFilterTests(unittest.TestCase):
    def test_exact_paper_channels_and_reset_are_classified(self) -> None:
        # tight [10,11] -> open [10,12] -> tight [11,12] -> slide [12,13]
        bids = torch.tensor([10, 10, 11, 12, 12], dtype=torch.int64)
        asks = torch.tensor([11, 12, 12, 13, 14], dtype=torch.int64)
        reset = torch.tensor([True, False, False, False, True])
        events, compatible = classify_paper_channels(bids, asks, reset)
        torch.testing.assert_close(events, torch.tensor([0, 3, 5, 1, 0]))
        torch.testing.assert_close(
            compatible, torch.tensor([False, True, True, True, False])
        )

    def test_causal_ewma_never_uses_current_value_in_returned_prior(self) -> None:
        values = torch.tensor([10.0, 20.0, 30.0])
        timestamps = torch.tensor([0, 1000, 2000], dtype=torch.int64)
        prior = _causal_ewma(values, timestamps, 1.0)
        coefficient = torch.exp(torch.tensor(-1.0))
        expected_second_posterior = coefficient * 10.0 + (1.0 - coefficient) * 20.0
        torch.testing.assert_close(
            prior, torch.stack((torch.tensor(10.0), torch.tensor(10.0), expected_second_posterior))
        )

    def test_empirical_nll_includes_survival_exposure(self) -> None:
        parameters = torch.tensor([1.0, 0.02, 2.0, 0.4, 0.2, 1.0])
        gap = torch.tensor([0.0])
        tight = torch.tensor([True])
        event = torch.tensor([1])
        first = _empirical_interval_nll(
            parameters, gap, tight, event, torch.tensor([0.1])
        )
        second = _empirical_interval_nll(
            parameters, gap, tight, event, torch.tensor([0.2])
        )
        torch.testing.assert_close(second - first, torch.tensor([0.204]))


if __name__ == "__main__":
    unittest.main()
