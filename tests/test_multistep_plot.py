import unittest

import torch

from ot_micromr.marked_filter import encode_mark
from scripts.plot_p6m_multistep import _eligible_origins, _exact_transition_csr


class MultistepPlotTests(unittest.TestCase):
    def test_exact_transition_csr_adds_directional_mirror(self) -> None:
        delta_bid = torch.tensor((3,), dtype=torch.int64)
        delta_ask = torch.tensor((3,), dtype=torch.int64)
        mark = encode_mark(delta_bid, delta_ask)
        train = (
            torch.zeros(1),
            torch.zeros(1, dtype=torch.int64),
            mark,
            torch.ones(1),
            delta_bid + delta_ask,
            delta_ask - delta_bid,
            torch.ones(1, dtype=torch.int64),
            torch.ones(1, dtype=torch.int64),
        )
        table = _exact_transition_csr(train)
        mirrored_mark = encode_mark(-delta_ask, -delta_bid)
        self.assertTrue(bool(table.observed_cells[0, mark[0]]))
        self.assertTrue(bool(table.observed_cells[0, mirrored_mark[0]]))
        self.assertEqual(int(table.counts.sum()), 2)

    def test_origins_require_full_healthy_horizon(self) -> None:
        timestamps = torch.arange(7, dtype=torch.int64) * 1000
        valid = torch.tensor((True, True, False, True, True, True))
        origins = _eligible_origins(timestamps, valid, 0, 7000, 2)
        self.assertTrue(torch.equal(origins, torch.tensor((0, 3, 4))))


if __name__ == "__main__":
    unittest.main()
