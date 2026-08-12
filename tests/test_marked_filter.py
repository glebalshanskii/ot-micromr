import unittest
from pathlib import Path

import torch

from ot_micromr.config import load_runspec
from ot_micromr.marked_filter import (
    MARK_COUNT,
    _marked_interval_score,
    _synthetic_mark_tables,
    encode_mark,
    magnitude_power_bucket,
    mark_metadata,
    previous_spread_bucket,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class MarkedFilterTests(unittest.TestCase):
    def test_fixed_mark_alphabet_and_paper_marks(self) -> None:
        delta_bid = torch.tensor((1, -1, 0, -1, 1, 0), dtype=torch.int64)
        delta_ask = torch.tensor((1, -1, 1, 0, 0, -1), dtype=torch.int64)
        marks = encode_mark(delta_bid, delta_ask)
        direction, family, midpoint_bucket, spread_bucket = mark_metadata()
        self.assertEqual(MARK_COUNT, 729)
        self.assertEqual(len(set(int(value) for value in marks)), 6)
        self.assertTrue(torch.equal(direction[marks], torch.tensor((1, -1, 1, -1, 1, -1))))
        self.assertTrue(torch.equal(family[marks], torch.tensor((0, 0, 1, 1, -1, -1))))
        self.assertTrue(torch.equal(midpoint_bucket[marks], torch.tensor((2, 2, 1, 1, 1, 1))))
        self.assertTrue(torch.equal(spread_bucket[marks], torch.tensor((0, 0, 1, 1, 1, 1))))

    def test_power_buckets_and_spread_overflow(self) -> None:
        values = torch.tensor((0, 1, 2, 3, 4, 7, 8, 127, 128, 10_000))
        self.assertTrue(
            torch.equal(
                magnitude_power_bucket(values),
                torch.tensor((0, 1, 2, 2, 3, 3, 4, 7, 8, 8)),
            )
        )
        self.assertTrue(
            torch.equal(
                previous_spread_bucket(torch.tensor((1, 2, 7, 8, 100))),
                torch.tensor((0, 1, 6, 7, 7)),
            )
        )

    def test_synthetic_tables_satisfy_drift_and_state_constraints(self) -> None:
        spec = load_runspec(
            REPOSITORY_ROOT / "cfg" / "experiments" / "filter_mark_syn_001.toml"
        )
        tables = _synthetic_mark_tables(spec, torch.device("cpu"))
        self.assertTrue(torch.allclose(tables.probabilities.sum(dim=-1), torch.ones(4)))
        signed_jump = tables.mark_dy.to(torch.float32) * 0.5
        self.assertLess(float(torch.max(torch.abs((tables.probabilities * signed_jump).sum(-1)))), 1e-6)
        down = (tables.direction_sign < 0).to(torch.float32).unsqueeze(0)
        up = (tables.direction_sign > 0).to(torch.float32).unsqueeze(0)
        self.assertTrue(
            torch.allclose(
                (tables.correction_weights * tables.jump_abs_price * down).sum(-1),
                torch.ones(4),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                (tables.correction_weights * tables.jump_abs_price * up).sum(-1),
                torch.ones(4),
                atol=1e-6,
            )
        )
        active = tables.probabilities > 0
        next_spread = torch.arange(1, 5).unsqueeze(1) + tables.mark_dd
        self.assertTrue(torch.all(next_spread[active] >= 1))
        self.assertTrue(torch.all(next_spread[active] <= 4))

    def test_interval_score_is_finite_for_event_and_silence(self) -> None:
        spec = load_runspec(
            REPOSITORY_ROOT / "cfg" / "experiments" / "filter_mark_syn_001.toml"
        )
        tables = _synthetic_mark_tables(spec, torch.device("cpu"))
        event = tables.active_ids[0, 0]
        scores = _marked_interval_score(
            torch.tensor((1.0, -1.0)),
            torch.tensor((0, 0)),
            torch.stack((event, event.new_tensor(-1))),
            0.005,
            tables,
            1.0,
        )
        self.assertTrue(torch.all(torch.isfinite(scores)))


if __name__ == "__main__":
    unittest.main()
