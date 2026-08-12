import tempfile
import unittest
from pathlib import Path

import torch

from ot_micromr.config import load_runspec
from ot_micromr.marked_filter import (
    MARK_COUNT,
    _block_reduce,
    _empirical_probability_tables,
    _inverse_softplus,
    _save_torch_artifact,
    _superiority_row,
    _equivalence_row,
    _marked_interval_score,
    _synthetic_mark_tables,
    _union_fieldnames,
    EmpiricalMarkedDay,
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

    def test_empirical_tables_have_full_support_and_constrained_drift(self) -> None:
        delta_bid = torch.tensor((2, -2, 0, 0), dtype=torch.int64)
        delta_ask = torch.tensor((2, -2, -1, 1), dtype=torch.int64)
        mark = encode_mark(delta_bid, delta_ask)
        spread = torch.tensor((0, 0, 2, 1), dtype=torch.int64)
        train = (
            torch.zeros(4),
            spread,
            mark,
            torch.ones(4),
            delta_bid + delta_ask,
            delta_ask - delta_bid,
            torch.tensor((1, 1, 3, 2)),
            torch.tensor((1, 1, 2, 3)),
        )
        full = _empirical_probability_tables(train, 0.01, "full")
        no_tick = _empirical_probability_tables(train, 0.01, "no_multi_tick")
        no_spread = _empirical_probability_tables(train, 0.01, "no_multi_spread")
        self.assertTrue(torch.all(full[0] > 0.0))
        self.assertTrue(torch.allclose(full[0].sum(dim=-1), torch.ones(8)))
        self.assertLess(float(torch.max(torch.abs(full[5]))), 1e-7)
        self.assertGreater(float(full[1][0, mark[0]]), 0.0)
        self.assertEqual(float(no_tick[1][0, mark[0]]), 0.0)
        self.assertGreater(float(full[1][2, mark[2]]), 0.0)
        self.assertEqual(float(no_spread[1][2, mark[2]]), 0.0)

    def test_inverse_softplus_is_finite_at_empirical_event_rates(self) -> None:
        rates = torch.tensor((0.1, 1.0, 20.0, 100.0), dtype=torch.float32)
        raw = _inverse_softplus(rates)
        self.assertTrue(torch.all(torch.isfinite(raw)))
        self.assertTrue(
            torch.allclose(torch.nn.functional.softplus(raw), rates, rtol=1e-6, atol=1e-6)
        )

    def test_union_fieldnames_accepts_sparse_diagnostic_columns(self) -> None:
        rows = ({"date": "a", "metric": 1.0}, {"date": "b", "spot_metric": 2.0})
        self.assertEqual(_union_fieldnames(rows), ["date", "metric", "spot_metric"])

    def test_torch_artifact_writer_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "state" / "tensor.pt"
            _save_torch_artifact({"value": torch.tensor((1.0, 2.0))}, path)
            payload = torch.load(path, weights_only=True)
        self.assertTrue(torch.equal(payload["value"], torch.tensor((1.0, 2.0))))

    def test_statistical_rows_distinguish_negative_from_imprecise(self) -> None:
        inferior = _superiority_row(torch.full((32,), -0.5), 0.0, 0.025, "metric")
        above = _equivalence_row(torch.full((32,), 2.0), 1.0, 0.1, 0.025, "calibration")
        self.assertEqual(inferior["status"], "inferior")
        self.assertLess(inferior["upper_bound"], 0.0)
        self.assertEqual(above["status"], "above_equivalence_region")
        self.assertGreater(above["lower_bound"], 1.1)

    def test_block_median_is_vectorized_and_even_count_uses_midpoint(self) -> None:
        day = EmpiricalMarkedDay(
            date="2024-01-15",
            timestamps_ms=torch.tensor((0, 1, 2, 1_800_001, 1_800_002)),
            bid_ticks=torch.zeros(5, dtype=torch.int64),
            ask_ticks=torch.ones(5, dtype=torch.int64),
            reset=torch.zeros(5, dtype=torch.bool),
            dt_seconds=torch.ones(4),
            valid_interval=torch.ones(4, dtype=torch.bool),
            previous_spread_ticks=torch.ones(4, dtype=torch.int64),
            current_spread_ticks=torch.ones(4, dtype=torch.int64),
            previous_spread_bucket=torch.zeros(4, dtype=torch.int64),
            mark_id=torch.zeros(4, dtype=torch.int64),
            delta_y=torch.zeros(4, dtype=torch.int64),
            delta_d=torch.zeros(4, dtype=torch.int64),
            prior_mid_price=torch.zeros(4),
            proxy_price=torch.zeros(5),
            proxy_gap=torch.zeros(4),
            spot_reference=None,
            spot_reference_timestamp_ms=None,
        )
        median, count, block = _block_reduce(
            day, torch.tensor((4.0, 2.0, 10.0, 6.0)), reduction="median", block_minutes=30
        )
        self.assertTrue(torch.equal(block, torch.tensor((0, 1))))
        self.assertTrue(torch.equal(count, torch.tensor((2.0, 2.0), dtype=torch.float64)))
        self.assertTrue(torch.equal(median, torch.tensor((3.0, 8.0), dtype=torch.float64)))


if __name__ == "__main__":
    unittest.main()
