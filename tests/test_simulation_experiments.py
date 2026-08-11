import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ot_micromr.config import load_runspec
from ot_micromr.simulation_experiments import evaluate_simulation


class SimulationExperimentTests(unittest.TestCase):
    def test_tiny_balanced_pipeline_reports_equivalence(self) -> None:
        source = load_runspec("cfg/experiments/sim_moments_002.toml")
        values = source.to_dict()
        values["model"]["mu_o_per_second"] = 5.0
        values["simulation"]["burn_in_reversion_times"] = 0.1
        values["simulation"]["horizon_reversion_times"] = 3.0
        values["simulation"]["cpu_workers"] = 2
        values["evaluation"]["acf_lags_reversion_times"] = [0.05, 0.1]
        values["evaluation"]["minimum_observations_per_seed_and_parity_for_slope"] = 1
        spec = SimpleNamespace(values=values, experiment_id="SIM-MOMENTS-002")
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = evaluate_simulation(spec, Path(temporary_directory))
            self.assertIn("flow_equivalence", result.metrics["scientific_components"])
            self.assertEqual(result.metrics["deterministic_replay_checked_count"], 3)
            self.assertTrue(result.acceptance["transition_count_conservation"])
            self.assertTrue((Path(temporary_directory) / "figures" / "sim-moments.png").is_file())

    def test_tiny_unbalanced_pipeline_reports_superiority(self) -> None:
        source = load_runspec("cfg/experiments/sim_unbalanced_002.toml")
        values = source.to_dict()
        values["model"]["mu_o_per_second"] = 5.0
        values["simulation"]["burn_in_reversion_times"] = 0.1
        values["simulation"]["horizon_reversion_times"] = 3.0
        values["simulation"]["cpu_workers"] = 2
        values["evaluation"]["minimum_observations_per_seed_and_parity_for_slope"] = 1
        spec = SimpleNamespace(values=values, experiment_id="SIM-UNBALANCED-002")
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = evaluate_simulation(spec, Path(temporary_directory))
            self.assertIn(
                "unbalanced_contrast_superiority",
                result.metrics["scientific_components"],
            )
            self.assertIn("contrast_equivalence", result.metrics["refinement_components"])
            self.assertEqual(result.metrics["deterministic_replay_checked_count"], 3)


if __name__ == "__main__":
    unittest.main()
