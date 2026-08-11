import json
import tempfile
import unittest
from pathlib import Path

from ot_micromr.artifacts import artifact_inventory, atomic_write_json, sha256_file
from ot_micromr.config import load_runspec
from ot_micromr.experiments import evaluate_smoke


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def test_smoke_evaluation_passes_preregistered_gates(self) -> None:
        spec = load_runspec(REPOSITORY_ROOT / "cfg" / "experiments" / "ana_smoke_001.toml")
        result = evaluate_smoke(spec)
        self.assertTrue(result.passed)
        self.assertTrue(all(result.acceptance.values()))

    def test_atomic_json_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "metrics" / "summary.json"
            atomic_write_json(path, {"finite": 1.0, "status": "passed"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "passed")
            records = artifact_inventory(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["sha256"], sha256_file(path))


if __name__ == "__main__":
    unittest.main()
