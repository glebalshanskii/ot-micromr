from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ot_micromr.errors import ConfigError, ExperimentError
from ot_micromr.okx_data import (
    canonical_dataset_sha256,
    load_okx_source_list,
    load_raw_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class OkxSourceListTests(unittest.TestCase):
    def test_current_source_list_is_strict_and_complete(self) -> None:
        source = load_okx_source_list(
            REPOSITORY_ROOT / "cfg" / "experiments" / "emp_data_001_sources.toml"
        )
        self.assertEqual(source.dataset_id, "okx-btc-p5-audit-2024-v1")
        self.assertEqual(source.venue, "OKX")
        self.assertEqual(len(source.assets), 32)
        self.assertEqual(sum(asset.kind == "orderbook_l2" for asset in source.assets), 10)
        self.assertEqual(sum(asset.kind == "trades" for asset in source.assets), 10)
        self.assertEqual(sum(asset.kind == "funding" for asset in source.assets), 12)
        self.assertEqual(len({asset.asset_id for asset in source.assets}), len(source.assets))

    def test_unknown_source_field_is_rejected(self) -> None:
        source_path = REPOSITORY_ROOT / "cfg" / "experiments" / "emp_data_001_sources.toml"
        content = source_path.read_text(encoding="utf-8").replace(
            'venue = "OKX"', 'venue = "OKX"\nunknown = true', 1
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.toml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_okx_source_list(path, repository_root=REPOSITORY_ROOT)

    def test_dataset_hash_ignores_retrieval_metadata_and_order(self) -> None:
        records = [
            {"asset_id": "b", "size_bytes": 2, "sha256": "b" * 64, "etag": "one"},
            {"asset_id": "a", "size_bytes": 1, "sha256": "a" * 64, "etag": "two"},
        ]
        expected = canonical_dataset_sha256("dataset", records)
        records.reverse()
        records[0]["etag"] = "changed"
        self.assertEqual(canonical_dataset_sha256("dataset", records), expected)

    def test_manifest_rejects_inconsistent_content_hash(self) -> None:
        manifest = {
            "schema_version": "okx-raw-manifest-v1",
            "dataset_id": "dataset",
            "dataset_content_sha256": "0" * 64,
            "assets": [{"asset_id": "a", "size_bytes": 1, "sha256": "a" * 64}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ExperimentError):
                load_raw_manifest(path)


if __name__ == "__main__":
    unittest.main()
