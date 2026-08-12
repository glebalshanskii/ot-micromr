import math
import unittest

import torch

from scripts.plot_p6m_predictions import _decode_mark_means, _posterior_directional_gaps
from ot_micromr.marked_filter import MARK_COUNT, encode_mark


class PredictionPlotTests(unittest.TestCase):
    def test_decoder_recovers_observed_exact_bid_and_ask_changes(self) -> None:
        delta_bid = torch.tensor((3, -2), dtype=torch.int64)
        delta_ask = torch.tensor((3, 0), dtype=torch.int64)
        mark = encode_mark(delta_bid, delta_ask)
        train = (
            torch.zeros(2),
            torch.tensor((0, 1), dtype=torch.int64),
            mark,
            torch.ones(2),
            delta_bid + delta_ask,
            delta_ask - delta_bid,
            torch.tensor((1, 2)),
            torch.tensor((1, 4)),
        )
        decoded_bid, decoded_ask, seen = _decode_mark_means(train)
        self.assertEqual(decoded_bid.shape, (8, MARK_COUNT))
        self.assertEqual(float(decoded_bid[0, mark[0]]), 3.0)
        self.assertEqual(float(decoded_ask[0, mark[0]]), 3.0)
        self.assertEqual(float(decoded_bid[1, mark[1]]), -2.0)
        self.assertEqual(float(decoded_ask[1, mark[1]]), 0.0)
        self.assertTrue(bool(seen[0, mark[0]]))

    def test_posterior_directional_gap_integrates_gaussian_uncertainty(self) -> None:
        direction = torch.tensor((-1, 0, 1), dtype=torch.int64)
        values = _posterior_directional_gaps(
            torch.zeros(1), torch.ones(1), direction
        )[0]
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        self.assertAlmostEqual(float(values[0]), expected, places=6)
        self.assertEqual(float(values[1]), 0.0)
        self.assertAlmostEqual(float(values[2]), expected, places=6)


if __name__ == "__main__":
    unittest.main()
