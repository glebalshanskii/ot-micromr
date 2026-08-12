import unittest

import torch

from ot_micromr.marked_filter import encode_mark
from scripts.plot_p6m_multistep import (
    _brownian_bridge_nodes,
    _eligible_origins,
    _exact_transition_csr,
    _linear_gap_hazard,
)


class MultistepPlotTests(unittest.TestCase):
    def test_linear_gap_hazard_integrates_a_zero_crossing(self) -> None:
        hazard = _linear_gap_hazard(
            torch.tensor((-1.0,)),
            torch.tensor((1.0,)),
            2.0,
            torch.tensor((0.0,)),
            torch.tensor(1.0),
            torch.tensor((1.0,)),
            torch.tensor((1.0,)),
        )
        torch.testing.assert_close(hazard, torch.tensor((1.0,)))

    def test_brownian_bridge_preserves_start_and_sampled_endpoint(self) -> None:
        normals = torch.zeros((2, 8))
        normals[:, 0] = torch.tensor((1.0, -1.0))
        nodes = _brownian_bridge_nodes(
            torch.tensor((3.0, 3.0)), torch.tensor(2.0), 4.0, 3, normals
        )
        self.assertEqual(tuple(nodes.shape), (2, 9))
        torch.testing.assert_close(nodes[:, 0], torch.tensor((3.0, 3.0)))
        torch.testing.assert_close(nodes[:, -1], torch.tensor((7.0, -1.0)))

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
