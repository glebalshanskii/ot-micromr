import json
import tempfile
import unittest
from pathlib import Path

from ot_micromr.artifacts import artifact_inventory, atomic_write_json, sha256_file
from ot_micromr.config import load_runspec
from ot_micromr.experiments import _required_artifacts_present, evaluate_smoke


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

    def test_figure4_artifact_inventory_is_independent_of_p3_figure_name(self) -> None:
        spec = load_runspec(
            REPOSITORY_ROOT / "cfg" / "experiments" / "sim_fig4_002.toml"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            present = _required_artifacts_present(spec, Path(temporary_directory))
        self.assertEqual(set(present), set(spec.values["artifacts"]["required_classes"]))
        self.assertTrue(present["manifest"])
        self.assertFalse(present["figure"])

    def test_p6m_artifact_inventory_requires_empirical_outputs(self) -> None:
        spec = load_runspec(
            REPOSITORY_ROOT / "cfg" / "experiments" / "emp_mark_filter_001.toml"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            present = _required_artifacts_present(spec, Path(temporary_directory))
        self.assertEqual(set(present), set(spec.values["artifacts"]["required_classes"]))
        self.assertTrue(present["manifest"])
        self.assertFalse(present["metrics_raw"])
        self.assertFalse(present["table"])
        self.assertFalse(present["state"])

    def test_p6d_artifact_inventory_requires_figures_and_state(self) -> None:
        spec = load_runspec(
            REPOSITORY_ROOT / "cfg" / "experiments" / "emp_mark_fact_001.toml"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            present = _required_artifacts_present(spec, Path(temporary_directory))
        self.assertEqual(set(present), set(spec.values["artifacts"]["required_classes"]))
        self.assertTrue(present["manifest"])
        self.assertFalse(present["figure"])
        self.assertFalse(present["figure_data"])
        self.assertFalse(present["state"])


if __name__ == "__main__":
    unittest.main()
