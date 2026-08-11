from __future__ import annotations

import csv
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from ot_micromr.empirical_data import (
    _audit_orderbook_worker,
    _audit_trades_worker,
    _bootstrap_large_tick,
    _price_to_ticks,
)


class EmpiricalDataAuditTests(unittest.TestCase):
    def test_price_conversion_uses_exact_tick_lattice(self) -> None:
        self.assertEqual(_price_to_ticks("93579.1", 10), 935791)
        self.assertEqual(_price_to_ticks("93579.10", 10), 935791)
        with self.assertRaises(ValueError):
            _price_to_ticks("93579.11", 10)

    def test_tiny_orderbook_reconstructs_touch_after_deltas(self) -> None:
        records = [
            {
                "instId": "BTC-USDT-SWAP",
                "action": "snapshot",
                "ts": "1705276800001",
                "asks": [["100.1", "2.0", "1"], ["100.2", "3.0", "1"]],
                "bids": [["100.0", "4.0", "1"], ["99.9", "5.0", "1"]],
            },
            {
                "instId": "BTC-USDT-SWAP",
                "action": "update",
                "ts": "1705276800011",
                "asks": [["100.1", "0.0", "0"]],
                "bids": [["100.0", "6.0", "2"]],
            },
            {
                "instId": "BTC-USDT-SWAP",
                "action": "update",
                "ts": "1705276800021",
                "asks": [["100.1", "1.0", "1"]],
                "bids": [],
            },
        ]
        payload = b"".join(
            json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"
            for record in records
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "book.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                info = tarfile.TarInfo("book.data")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            result = _audit_orderbook_worker(
                {
                    "path": str(path),
                    "asset_id": "book",
                    "instrument_type": "SWAP",
                    "instrument": "BTC-USDT-SWAP",
                    "date": "2024-01-15",
                    "tick_scale": 10,
                    "batch_rows": 2,
                }
            )
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["one_tick_rows"], 2)
        self.assertEqual(result["two_tick_rows"], 1)
        self.assertEqual(result["empty_book_rows"], 0)
        self.assertEqual(result["locked_rows"], 0)
        self.assertEqual(result["crossed_rows"], 0)

    def test_orderbook_ignores_unusable_prefix_until_first_snapshot(self) -> None:
        records = [
            {
                "instId": "BTC-USDT-SWAP",
                "action": "update",
                "ts": "1705276800001",
                "asks": [["101.0", "1.0", "1"]],
                "bids": [["99.0", "1.0", "1"]],
            },
            {
                "instId": "BTC-USDT-SWAP",
                "action": "snapshot",
                "ts": "1705276860000",
                "asks": [["100.1", "2.0", "1"]],
                "bids": [["100.0", "3.0", "1"]],
            },
        ]
        payload = b"".join(
            json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"
            for record in records
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "book.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                info = tarfile.TarInfo("book.data")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            result = _audit_orderbook_worker(
                {
                    "path": str(path),
                    "asset_id": "book",
                    "instrument_type": "SWAP",
                    "instrument": "BTC-USDT-SWAP",
                    "date": "2024-01-15",
                    "tick_scale": 10,
                    "batch_rows": 2,
                }
            )
        self.assertEqual(result["first_action"], "update")
        self.assertEqual(result["pre_snapshot_update_rows"], 1)
        self.assertEqual(result["source_rows"], 2)
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["one_tick_rows"], 1)
        self.assertEqual(result["usable_start_coverage_lag_ms"], 60_000)

    def test_tiny_trades_validate_ordering_and_values(self) -> None:
        rows = [
            ["BTC-USDT-SWAP", "10", "buy", "100.0", "2.0", "1705276800001"],
            ["BTC-USDT-SWAP", "11", "sell", "100.1", "1.0", "1705276800001"],
            ["BTC-USDT-SWAP", "12", "buy", "100.2", "3.0", "1705276800010"],
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trades.zip"
            stream = io.StringIO()
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(
                ["instrument_name", "trade_id", "side", "price", "size", "created_time"]
            )
            writer.writerows(rows)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("trades.csv", stream.getvalue())
            result = _audit_trades_worker(
                {
                    "path": str(path),
                    "asset_id": "trades",
                    "instrument_type": "SWAP",
                    "instrument": "BTC-USDT-SWAP",
                    "date": "2024-01-15",
                    "tick_scale": 10,
                    "batch_rows": 2,
                }
            )
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["nonmonotone_timestamp_rows"], 0)
        self.assertEqual(result["nonincreasing_trade_id_rows"], 0)
        self.assertEqual(result["invalid_side_rows"], 0)

    def test_day_cluster_bootstrap_is_deterministic(self) -> None:
        first = _bootstrap_large_tick([0.995, 0.997, 0.999], 1000, 123)
        second = _bootstrap_large_tick([0.995, 0.997, 0.999], 1000, 123)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
